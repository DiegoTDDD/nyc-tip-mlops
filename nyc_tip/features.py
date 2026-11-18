"""Feature engineering shared by build_features.py, training and the API.

Training and serving import the same functions, so a trip sent to the API is
turned into exactly the feature row the model was trained on.
tests/test_features.py checks this by comparing both paths on the same trips.

Pipeline:
  clean_trips          raw TLC records -> credit-card trips with sane values + high_tip target
  add_engineered_features   speed, cyclical hour
  ZoneEncodings.fit    smoothed zone tip rates, fit on the training split only
  ZoneEncodings.apply  map them onto any frame (unknown zones get the prior)
  trip_features        one raw API trip -> the model's feature row
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "high_tip"
HIGH_TIP_SHARE = 0.20  # tip above 20% of the fare
MAX_SPEED_MPH = 80.0  # caps GPS and timestamp errors
ZONE_SMOOTHING = 50.0  # pseudo-trips pulling rare zones toward the overall rate

# Columns written by build_features.py (one row per trip).
BASE_FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "passenger_count",
    "pu_location_id",
    "do_location_id",
]
# What the model is trained on, in this order.
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
# Exact dtypes in the training frame, which the MLflow model signature enforces.
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


def high_tip(tip_amount: pd.Series, fare_amount: pd.Series) -> pd.Series:
    """1 when the tip is more than 20% of the fare."""
    return (tip_amount / fare_amount > HIGH_TIP_SHARE).astype(int)


def clean_trips(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw TLC Yellow Taxi records -> one clean row per trip with BASE_FEATURES + target.

    Only credit-card trips (payment_type 1) with a positive fare are kept: cash
    trips almost never record a tip, so including them would let the payment
    channel leak the label. Bounds on distance, duration and passengers only
    remove clear data-entry errors.
    """
    df = raw[(raw["payment_type"] == 1) & (raw["fare_amount"] > 0)].copy()
    df[TARGET] = high_tip(df["tip_amount"], df["fare_amount"])
    pickup, dropoff = df["tpep_pickup_datetime"], df["tpep_dropoff_datetime"]
    df["trip_duration_min"] = (dropoff - pickup).dt.total_seconds() / 60.0
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    df = df.rename(columns={"PULocationID": "pu_location_id", "DOLocationID": "do_location_id"})

    passengers = df["passenger_count"].fillna(1)
    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 100)
        & (df["trip_duration_min"] > 0)
        & (df["trip_duration_min"] < 240)
        & (passengers > 0)
        & (passengers <= 6)
    ].copy()
    out = df[[*BASE_FEATURES, TARGET, "tpep_pickup_datetime"]].copy()
    out["passenger_count"] = out["passenger_count"].fillna(1).astype(int)
    out = out.reset_index(drop=True)
    out["trip_id"] = out.index.astype(int)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Average speed (capped) and the pickup hour on a circle, so 23h is next to 0h."""
    out = df.copy()
    hours = (out["trip_duration_min"] / 60.0).clip(lower=1e-3)
    out["avg_speed_mph"] = (out["trip_distance"] / hours).clip(upper=MAX_SPEED_MPH)
    angle = 2 * np.pi * out["pickup_hour"] / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out


@dataclass
class ZoneEncodings:
    """Smoothed high-tip rate per pickup and dropoff zone, with the overall rate as fallback."""

    pu_rate: dict[int, float] = field(default_factory=dict)
    do_rate: dict[int, float] = field(default_factory=dict)
    pu_prior: float = 0.5
    do_prior: float = 0.5

    @staticmethod
    def _fit_one(train: pd.DataFrame, zone_col: str) -> tuple[dict[int, float], float]:
        prior = float(train[TARGET].mean())
        grouped = train.groupby(zone_col)[TARGET]
        counts, means = grouped.count(), grouped.mean()
        smoothed = (means * counts + prior * ZONE_SMOOTHING) / (counts + ZONE_SMOOTHING)
        return {int(z): float(r) for z, r in smoothed.items()}, prior

    @classmethod
    def fit(cls, train: pd.DataFrame) -> ZoneEncodings:
        """Fit on the training split only, so validation labels never leak in."""
        pu_rate, pu_prior = cls._fit_one(train, "pu_location_id")
        do_rate, do_prior = cls._fit_one(train, "do_location_id")
        return cls(pu_rate, do_rate, pu_prior, do_prior)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["pu_zone_tip_rate"] = out["pu_location_id"].map(self.pu_rate).fillna(self.pu_prior)
        out["do_zone_tip_rate"] = out["do_location_id"].map(self.do_rate).fillna(self.do_prior)
        return out

    def to_dict(self) -> dict:
        """The zone_encodings.json format logged next to each model in MLflow."""
        return {
            "pu_zone_tip_rate": self.pu_rate,
            "do_zone_tip_rate": self.do_rate,
            "pu_prior": self.pu_prior,
            "do_prior": self.do_prior,
            "model_features": MODEL_FEATURES,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ZoneEncodings:
        """Read zone_encodings.json (JSON object keys are strings; zone ids are ints)."""
        return cls(
            pu_rate={int(k): float(v) for k, v in data["pu_zone_tip_rate"].items()},
            do_rate={int(k): float(v) for k, v in data["do_zone_tip_rate"].items()},
            pu_prior=float(data["pu_prior"]),
            do_prior=float(data["do_prior"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> ZoneEncodings:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def model_frame(df: pd.DataFrame, encodings: ZoneEncodings) -> pd.DataFrame:
    """BASE_FEATURES rows -> the model's input: MODEL_FEATURES with the training dtypes."""
    out = encodings.apply(add_engineered_features(df))
    return out[MODEL_FEATURES].astype(FEATURE_DTYPES)


def trip_features(
    trip_distance: float,
    pickup: datetime,
    dropoff: datetime,
    pu_location_id: int,
    do_location_id: int,
    passenger_count: int,
    encodings: ZoneEncodings,
) -> pd.DataFrame:
    """One raw trip (as the API receives it) -> a one-row model input frame.

    Raises ValueError if the dropoff is not after the pickup.
    """
    duration_min = (dropoff - pickup).total_seconds() / 60.0
    if duration_min <= 0:
        raise ValueError("dropoff must be after pickup")
    dayofweek = pickup.weekday()
    base = pd.DataFrame(
        [
            {
                "trip_distance": float(trip_distance),
                "trip_duration_min": duration_min,
                "pickup_hour": pickup.hour,
                "pickup_dayofweek": dayofweek,
                "is_weekend": int(dayofweek >= 5),
                "passenger_count": int(passenger_count),
                "pu_location_id": int(pu_location_id),
                "do_location_id": int(do_location_id),
            }
        ]
    )
    return model_frame(base, encodings)
