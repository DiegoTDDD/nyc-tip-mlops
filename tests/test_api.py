"""The FastAPI service: endpoints with a stand-in model, then with the real trained champion."""

import os
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nyc_tip.features import ZoneEncodings, trip_features
from serving import app_api

TRIP = {
    "trip_distance": 3.2,
    "pickup_datetime": "2026-03-04T18:10:00",
    "dropoff_datetime": "2026-03-04T18:32:00",
    "pu_location_id": 161,
    "do_location_id": 237,
    "passenger_count": 1,
}


class FakeModel:
    def __init__(self, proba):
        self.proba = proba
        self.seen = []

    def predict_proba(self, frame):
        self.seen.append(frame)
        proba = np.full(len(frame), self.proba)
        return np.column_stack([1 - proba, proba])


@pytest.fixture
def client(monkeypatch):
    enc = ZoneEncodings({161: 0.8}, {237: 0.7}, 0.5, 0.5)
    model = FakeModel(0.73)
    monkeypatch.setitem(app_api._state, "model", model)
    monkeypatch.setitem(app_api._state, "version", "7")
    monkeypatch.setitem(app_api._state, "encodings", enc)
    return TestClient(app_api.app), model


def test_root_and_health_report_the_loaded_version(client):
    http, _ = client
    assert http.get("/").json()["status"] == "ready"
    assert http.get("/health").json() == {"status": "healthy", "model_version": "7"}


def test_health_is_503_without_a_model(monkeypatch):
    monkeypatch.setitem(app_api._state, "model", None)
    http = TestClient(app_api.app)
    assert http.get("/health").status_code == 503
    assert http.post("/predict", json=TRIP).status_code == 503
    assert http.get("/").json()["status"] == "model_not_loaded"


def test_predict_returns_probability_label_and_version(client):
    http, _ = client
    body = http.post("/predict", json=TRIP).json()
    assert body == {"high_tip_probability": 0.73, "high_tip": True, "model_version": "7"}


def test_predict_sends_the_model_the_training_feature_row(client):
    http, model = client
    http.post("/predict", json=TRIP)
    pickup = datetime(2026, 3, 4, 18, 10)
    expected = trip_features(3.2, pickup, pickup + timedelta(minutes=22), 161, 237, 1,
                             app_api._state["encodings"])
    assert model.seen[0].equals(expected)


@pytest.mark.parametrize(
    "change",
    [
        {"dropoff_datetime": "2026-03-04T18:00:00"},  # before the pickup
        {"pu_location_id": 0},
        {"do_location_id": 266},
        {"trip_distance": 0},
        {"passenger_count": 7},
    ],
)
def test_invalid_trips_are_rejected_with_422(client, change):
    http, _ = client
    assert http.post("/predict", json={**TRIP, **change}).status_code == 422


def test_batch_scores_every_trip(client):
    http, _ = client
    body = http.post("/predict/batch", json=[TRIP, {**TRIP, "passenger_count": 3}]).json()
    assert len(body) == 2 and all(p["high_tip_probability"] == 0.73 for p in body)


def test_the_real_champion_loads_from_the_registry_and_predicts(registry, monkeypatch):
    monkeypatch.setattr(app_api, "TRACKING_URI", registry["uri"])
    monkeypatch.setattr(app_api, "MLRUNS_DIR", str(registry["folder"] / "mlruns"))
    previous = os.getcwd()
    os.chdir(registry["folder"])
    try:
        with TestClient(app_api.app) as http:  # runs the startup that loads the champion
            assert http.get("/health").json()["model_version"] == "1"
            proba = http.post("/predict", json=TRIP).json()["high_tip_probability"]
    finally:
        os.chdir(previous)
    # A probability, not the class: the pyfunc wrapper used to return exactly 0 or 1.
    assert 0.0 < proba < 1.0
