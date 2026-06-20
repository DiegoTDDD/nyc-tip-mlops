"""FastAPI serving layer for the NYC Taxi high-tip classifier.

Loads the champion model straight from the MLflow Model Registry (no
hard-coded file path) plus the zone target-encodings logged alongside it,
then exposes a /predict endpoint that accepts the RAW fields of a taxi
trip and performs the exact same feature engineering used in training
before scoring. This guarantees train/serving parity.

Endpoints
---------
GET  /              -> service metadata + which model version is loaded
GET  /health        -> liveness probe
POST /predict       -> probability + label for one trip
POST /predict/batch -> same for a list of trips

Run locally
-----------
    uvicorn serving.app_api:app --reload --port 8000
    # then open http://localhost:8000/docs  for the interactive UI
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --- Config -----------------------------------------------------------------
# The tracking URI must match the trainer's. Allow override via env var so
# the same code works in Docker (where the path may differ).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"

MODEL_FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "avg_speed_mph",
    "pickup_dayofweek",
    "is_weekend",
    "passenger_count",
    "hour_sin",
    "hour_cos",
    "pu_zone_tip_rate",
    "do_zone_tip_rate",
]

# Exact dtypes the model's MLflow signature requires. Must match what the
# training DataFrame produced, or strict schema enforcement rejects the
# request. pickup_dayofweek was int32 in training (MLflow "integer");
# is_weekend / passenger_count were int64 ("long"); the rest are float64.
FEATURE_DTYPES = {
    "trip_distance": "float64",
    "trip_duration_min": "float64",
    "avg_speed_mph": "float64",
    "pickup_dayofweek": "int32",
    "is_weekend": "int64",
    "passenger_count": "int64",
    "hour_sin": "float64",
    "hour_cos": "float64",
    "pu_zone_tip_rate": "float64",
    "do_zone_tip_rate": "float64",
}


# --- Request / response schemas --------------------------------------------
class Trip(BaseModel):
    """Raw fields of a single taxi trip, known at pickup time."""
    trip_distance: float = Field(..., gt=0, description="Miles")
    pickup_datetime: datetime = Field(..., description="ISO timestamp")
    dropoff_datetime: datetime = Field(..., description="ISO timestamp")
    pu_location_id: int = Field(..., ge=1, le=265)
    do_location_id: int = Field(..., ge=1, le=265)
    passenger_count: int = Field(1, ge=1, le=6)


class Prediction(BaseModel):
    high_tip_probability: float
    high_tip: bool
    model_version: str


# --- App + model loading ----------------------------------------------------
app = FastAPI(
    title="NYC Taxi High-Tip Classifier",
    description="Predicts whether a taxi trip will yield a tip above 20% "
                "of the fare. Model served from the MLflow Model Registry.",
    version="1.0.0",
)

# Filled in at startup.
_state = {"model": None, "version": None, "pu_map": {}, "do_map": {},
          "pu_prior": 0.5, "do_prior": 0.5}


def _load_model() -> None:
    """Load the champion model + its zone encodings from MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    _state["model"] = mlflow.pyfunc.load_model(model_uri)
    _state["version"] = mv.version

    # The trainer logged zone_encodings.json as a run artifact.
    local_dir = client.download_artifacts(mv.run_id, "zone_encodings.json")
    with open(local_dir) as f:
        enc = json.load(f)
    # JSON keys are strings; convert back to int zone ids.
    _state["pu_map"] = {int(k): v for k, v in enc["pu_zone_tip_rate"].items()}
    _state["do_map"] = {int(k): v for k, v in enc["do_zone_tip_rate"].items()}
    _state["pu_prior"] = enc["pu_prior"]
    _state["do_prior"] = enc["do_prior"]


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
        print(f"Loaded champion model version {_state['version']}")
    except Exception as e:  # don't crash; /health will report unready
        print(f"WARNING: could not load model at startup: {e}")


def _engineer(trip: Trip) -> pd.DataFrame:
    """Turn a raw trip into the exact model feature row used in training."""
    duration_min = (trip.dropoff_datetime - trip.pickup_datetime).total_seconds() / 60.0
    if duration_min <= 0:
        raise HTTPException(status_code=422,
                            detail="dropoff must be after pickup")

    hours = max(duration_min / 60.0, 1e-3)
    avg_speed = min(trip.trip_distance / hours, 80.0)
    hour = trip.pickup_datetime.hour
    dow = trip.pickup_datetime.weekday()

    row = {
        "trip_distance": trip.trip_distance,
        "trip_duration_min": duration_min,
        "avg_speed_mph": avg_speed,
        "pickup_dayofweek": dow,
        "is_weekend": 1 if dow >= 5 else 0,
        "passenger_count": trip.passenger_count,
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "pu_zone_tip_rate": _state["pu_map"].get(trip.pu_location_id, _state["pu_prior"]),
        "do_zone_tip_rate": _state["do_map"].get(trip.do_location_id, _state["do_prior"]),
    }
    df = pd.DataFrame([row])[MODEL_FEATURES]

    # Cast each column to the exact dtype the model's MLflow signature
    # expects. The training data produced pickup_dayofweek as int32, while
    # building a DataFrame from a Python dict defaults to int64; MLflow's
    # strict schema enforcement refuses the silent int64->int32 conversion.
    df = df.astype(FEATURE_DTYPES)
    return df


@app.get("/")
def root() -> dict:
    return {
        "service": "NYC Taxi High-Tip Classifier",
        "model": REGISTERED_MODEL_NAME,
        "model_version": _state["version"],
        "status": "ready" if _state["model"] is not None else "model_not_loaded",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "healthy", "model_version": _state["version"]}


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip) -> Prediction:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    X = _engineer(trip)
    proba = float(np.asarray(_state["model"].predict(X)).ravel()[0])
    return Prediction(
        high_tip_probability=round(proba, 4),
        high_tip=proba >= 0.5,
        model_version=str(_state["version"]),
    )


@app.post("/predict/batch")
def predict_batch(trips: list[Trip]) -> list[Prediction]:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    rows = pd.concat([_engineer(t) for t in trips], ignore_index=True)
    probas = np.asarray(_state["model"].predict(rows)).ravel()
    return [
        Prediction(high_tip_probability=round(float(p), 4),
                   high_tip=bool(p >= 0.5),
                   model_version=str(_state["version"]))
        for p in probas
    ]
