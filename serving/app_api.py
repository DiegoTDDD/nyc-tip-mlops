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

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyc_tip.features import ZoneEncodings, trip_features  # noqa: E402

log = logging.getLogger(__name__)

# --- Config -----------------------------------------------------------------
# The tracking URI must match the trainer's. Allow override via env var so
# the same code works in Docker (where the path may differ).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"
THRESHOLD = 0.5


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


# --- Model state ------------------------------------------------------------
# Filled at startup by load_model(); tests can set it directly.
_state: dict[str, Any] = {"model": None, "version": None, "encodings": ZoneEncodings()}


MLRUNS_DIR = os.environ.get("MLRUNS_DIR", "mlruns")


def _to_local(uri: str) -> str | None:
    """Map an artifact location stored in mlflow.db to this machine.

    MLflow stores ABSOLUTE paths (for example /home/runner/work/x/mlruns/1/...
    or C:/Users/me/x/mlruns/1/...). Inside a container the same files live
    under MLRUNS_DIR, so when the stored path does not exist we re-root it
    on the part after "mlruns/". Returns None if nothing matches.
    """
    path = uri.replace("file://", "", 1) if uri.startswith("file://") else uri
    path = path.replace("\\", "/")
    if len(path) > 2 and path[0] == "/" and path[2] == ":":  # /C:/... from file:///C:/...
        path = path[1:]
    if os.path.exists(path):
        return path
    marker = "/mlruns/"
    i = path.rfind(marker)
    if i == -1:
        return None
    candidate = os.path.join(MLRUNS_DIR, path[i + len(marker):])
    return candidate if os.path.exists(candidate) else None


def _load_from_local_mlruns(client: MlflowClient, mv: ModelVersion) -> tuple[Any, str]:
    """Fallback used when the paths stored in mlflow.db do not exist here."""
    source = mv.source
    if source.startswith("models:/m-"):
        source = client.get_logged_model(source.split("/")[-1]).artifact_location
    model_dir = _to_local(source)
    run_dir = _to_local(client.get_run(mv.run_id).info.artifact_uri)
    if model_dir is None or run_dir is None:
        raise RuntimeError(
            f"model artifacts not found under {MLRUNS_DIR!r} "
            f"(registry points at {mv.source!r})")
    return (mlflow.xgboost.load_model(model_dir),
            os.path.join(run_dir, "zone_encodings.json"))


def _load_model() -> None:
    """Load the champion model + its zone encodings from MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    try:
        # The native XGBoost model, not the pyfunc wrapper: pyfunc.predict() returns
        # the class (0/1), and the API reports a probability (ADR 0003).
        model = mlflow.xgboost.load_model(model_uri)
        # The trainer logged zone_encodings.json as a run artifact.
        enc_path = client.download_artifacts(mv.run_id, "zone_encodings.json")
    except Exception as first_error:
        log.warning("Registry paths not usable here (%s); falling back to %s/",
                    first_error, MLRUNS_DIR)
        model, enc_path = _load_from_local_mlruns(client, mv)
    _state["model"] = model
    _state["version"] = str(mv.version)  # MLflow 3 returns an int; the API always reports a string
    _state["encodings"] = ZoneEncodings.load(enc_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion once at startup. A failure leaves the API up but /health returns 503."""
    try:
        _load_model()
        log.info("Loaded champion model version %s", _state["version"])
    except Exception as exc:  # noqa: BLE001 - keep serving /health so the failure is visible
        log.warning("could not load model at startup: %s", exc)
    yield


app = FastAPI(
    title="NYC Taxi High-Tip Classifier",
    description="Predicts whether a taxi trip will yield a tip above 20% "
    "of the fare. Model served from the MLflow Model Registry.",
    version="1.1.0",
    lifespan=lifespan,
)


def _engineer(trip: Trip) -> pd.DataFrame:
    """Raw trip -> the exact feature row used in training (nyc_tip/features.py)."""
    try:
        return trip_features(
            trip.trip_distance, trip.pickup_datetime, trip.dropoff_datetime,
            trip.pu_location_id, trip.do_location_id, trip.passenger_count,
            _state["encodings"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_model() -> None:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")


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
    _require_model()
    return {"status": "healthy", "model_version": _state["version"]}


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip) -> Prediction:
    _require_model()
    proba = float(_state["model"].predict_proba(_engineer(trip))[0, 1])
    return Prediction(
        high_tip_probability=round(proba, 4),
        high_tip=proba >= THRESHOLD,
        model_version=str(_state["version"]),
    )


@app.post("/predict/batch")
def predict_batch(trips: list[Trip]) -> list[Prediction]:
    _require_model()
    rows = pd.concat([_engineer(t) for t in trips], ignore_index=True)
    probas = _state["model"].predict_proba(rows)[:, 1]
    return [
        Prediction(high_tip_probability=round(float(p), 4),
                   high_tip=bool(p >= THRESHOLD),
                   model_version=str(_state["version"]))
        for p in probas
    ]
