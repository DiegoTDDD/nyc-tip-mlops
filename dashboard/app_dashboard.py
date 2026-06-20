"""Streamlit dashboard for the NYC Taxi high-tip MLOps project.

A single-pane view that ties the whole system together:

1. Champion model card  -- metrics of the model currently in production,
   read straight from the MLflow registry + tracking store.
2. Model registry history -- every registered version and its AUC, so the
   iteration story (and the automatic promotion) is visible.
3. Data drift status     -- read from monitoring/drift_summary.json, with
   a per-column breakdown and a retrain recommendation.
4. Live prediction        -- a form that scores a trip. It calls the running
   FastAPI service if reachable, and otherwise falls back to loading the
   champion model locally, so the dashboard works in every environment
   (including Streamlit Cloud, where FastAPI is not running).

Run
---
    streamlit run dashboard/app_dashboard.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps used by the live-prediction fallback. Imported lazily so
# the rest of the dashboard renders even if something is missing.
try:
    import requests
except Exception:
    requests = None


st.set_page_config(
    page_title="NYC Taxi High-Tip · MLOps",
    page_icon="🚕",
    layout="wide",
)

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"
DRIFT_SUMMARY_PATH = "monitoring/drift_summary.json"
# Lightweight, version-controlled bundle used when the MLflow store
# (mlflow.db / mlruns/) is not available, e.g. on Streamlit Cloud.
SERVING_MODEL_DIR = "serving_model"
API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8000")

MODEL_FEATURES = [
    "trip_distance", "trip_duration_min", "avg_speed_mph", "pickup_dayofweek",
    "is_weekend", "passenger_count", "hour_sin", "hour_cos",
    "pu_zone_tip_rate", "do_zone_tip_rate",
]
FEATURE_DTYPES = {
    "trip_distance": "float64", "trip_duration_min": "float64",
    "avg_speed_mph": "float64", "pickup_dayofweek": "int32",
    "is_weekend": "int64", "passenger_count": "int64",
    "hour_sin": "float64", "hour_cos": "float64",
    "pu_zone_tip_rate": "float64", "do_zone_tip_rate": "float64",
}


# --------------------------------------------------------------------------
# Data loading helpers (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_mlflow_client():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


@st.cache_data(show_spinner=False)
def load_registry_history() -> pd.DataFrame:
    """All registered versions + their validation AUC.

    Prefers the live MLflow registry; falls back to the version-controlled
    serving_model/registry_history.json when MLflow is unavailable
    (e.g. on Streamlit Cloud).
    """
    try:
        client = get_mlflow_client()
        rows = []
        champ_version = None
        try:
            champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
            champ_version = champ.version
        except Exception:
            pass
        for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
            run = client.get_run(mv.run_id)
            rows.append({
                "version": int(mv.version),
                "val_auc": run.data.metrics.get("val_auc"),
                "val_f1": run.data.metrics.get("val_f1"),
                "val_precision": run.data.metrics.get("val_precision"),
                "val_recall": run.data.metrics.get("val_recall"),
                "is_champion": mv.version == champ_version,
                "created": datetime.fromtimestamp(mv.creation_timestamp / 1000),
            })
        if rows:
            return pd.DataFrame(rows).sort_values("version")
    except Exception:
        pass

    # Fallback: version-controlled bundle.
    hist_path = os.path.join(SERVING_MODEL_DIR, "registry_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            rows = json.load(f)
        return pd.DataFrame(rows).sort_values("version") if rows else pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_drift_summary() -> dict | None:
    if not os.path.exists(DRIFT_SUMMARY_PATH):
        return None
    with open(DRIFT_SUMMARY_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_local_model():
    """Champion model + zone encodings for live prediction.

    Prefers the live MLflow registry; falls back to the version-controlled
    serving_model/ bundle when MLflow is unavailable (e.g. Streamlit Cloud).
    """
    import mlflow.pyfunc

    # Try the live MLflow registry first.
    try:
        client = get_mlflow_client()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
        enc_path = client.download_artifacts(mv.run_id, "zone_encodings.json")
        with open(enc_path) as f:
            enc = json.load(f)
        pu_map = {int(k): v for k, v in enc["pu_zone_tip_rate"].items()}
        do_map = {int(k): v for k, v in enc["do_zone_tip_rate"].items()}
        return model, pu_map, do_map, enc["pu_prior"], enc["do_prior"], mv.version
    except Exception:
        pass

    # Fallback: version-controlled bundle.
    model = mlflow.pyfunc.load_model(os.path.join(SERVING_MODEL_DIR, "model"))
    with open(os.path.join(SERVING_MODEL_DIR, "zone_encodings.json")) as f:
        enc = json.load(f)
    pu_map = {int(k): v for k, v in enc["pu_zone_tip_rate"].items()}
    do_map = {int(k): v for k, v in enc["do_zone_tip_rate"].items()}
    version = "?"
    champ_path = os.path.join(SERVING_MODEL_DIR, "champion.json")
    if os.path.exists(champ_path):
        with open(champ_path) as f:
            version = json.load(f).get("version", "?")
    return model, pu_map, do_map, enc["pu_prior"], enc["do_prior"], version


def engineer_row(dist, pickup_dt, dropoff_dt, pu_id, do_id, pax,
                 pu_map, do_map, pu_prior, do_prior) -> pd.DataFrame:
    dur = (dropoff_dt - pickup_dt).total_seconds() / 60.0
    hours = max(dur / 60.0, 1e-3)
    row = {
        "trip_distance": dist,
        "trip_duration_min": dur,
        "avg_speed_mph": min(dist / hours, 80.0),
        "pickup_dayofweek": pickup_dt.weekday(),
        "is_weekend": 1 if pickup_dt.weekday() >= 5 else 0,
        "passenger_count": pax,
        "hour_sin": np.sin(2 * np.pi * pickup_dt.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * pickup_dt.hour / 24.0),
        "pu_zone_tip_rate": pu_map.get(pu_id, pu_prior),
        "do_zone_tip_rate": do_map.get(do_id, do_prior),
    }
    return pd.DataFrame([row])[MODEL_FEATURES].astype(FEATURE_DTYPES)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚕 NYC Taxi High-Tip Classifier")
st.caption(
    "An end-to-end MLOps system: versioned training (MLflow), model registry "
    "with automatic promotion, a FastAPI serving layer, and data-drift "
    "monitoring (Evidently). This dashboard is the operations console."
)

history = load_registry_history()
drift = load_drift_summary()

# --------------------------------------------------------------------------
# Section 1 + 2: champion card and registry history
# --------------------------------------------------------------------------
st.header("Model")

if history.empty:
    st.warning("No registered model versions found. Run the training script first.")
else:
    champ = history[history["is_champion"]]
    champ = champ.iloc[0] if not champ.empty else history.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion version", f"v{int(champ['version'])}")
    c2.metric("Validation AUC", f"{champ['val_auc']:.4f}" if pd.notna(champ['val_auc']) else "n/a")
    c3.metric("Precision", f"{champ['val_precision']:.3f}" if pd.notna(champ['val_precision']) else "n/a")
    c4.metric("Recall", f"{champ['val_recall']:.3f}" if pd.notna(champ['val_recall']) else "n/a")

    st.subheader("Registry history")
    st.caption("Every registered version and its validation AUC. The champion "
               "is the best performer, promoted automatically at train time.")
    show = history.copy()
    show["champion"] = show["is_champion"].map({True: "★", False: ""})
    show = show[["version", "val_auc", "val_f1", "val_precision",
                 "val_recall", "champion"]]
    st.dataframe(
        show.style.format({
            "val_auc": "{:.4f}", "val_f1": "{:.4f}",
            "val_precision": "{:.4f}", "val_recall": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )
    if len(history) > 1:
        st.bar_chart(history.set_index("version")["val_auc"], height=200)

# --------------------------------------------------------------------------
# Section 3: drift
# --------------------------------------------------------------------------
st.header("Data drift monitoring")
st.caption("2024 training data (reference) vs 2025 production data (current). "
           "January 2025 followed NYC's congestion-pricing rollout.")

if drift is None:
    st.info("No drift report found. Run monitoring/check_drift.py to generate one.")
else:
    d1, d2, d3 = st.columns(3)
    d1.metric("Columns monitored", drift.get("n_columns_monitored", "n/a"))
    d2.metric("Columns drifted", drift.get("n_columns_drifted", "n/a"))
    share = drift.get("drift_share")
    d3.metric("Drift share", f"{share:.1%}" if share is not None else "n/a")

    if drift.get("retrain_recommended"):
        st.error("⚠️ Retrain recommended — drift share crossed the threshold.")
    else:
        st.success("✓ No retrain needed — drift is below the action threshold.")

    per = drift.get("per_column", {})
    if per:
        rows = [{"column": c, "drift_score": v.get("drift_score"),
                 "threshold": v.get("threshold"),
                 "status": "DRIFT" if v.get("drifted") else "ok"}
                for c, v in per.items()]
        dfp = pd.DataFrame(rows).sort_values("drift_score", ascending=False)
        st.dataframe(
            dfp.style.format({"drift_score": "{:.4f}", "threshold": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------
# Section 4: live prediction
# --------------------------------------------------------------------------
st.header("Live prediction")
st.caption("Score a single trip. Uses the running FastAPI service if "
           "available, otherwise loads the champion model directly.")

with st.form("predict"):
    f1, f2, f3 = st.columns(3)
    dist = f1.number_input("Trip distance (miles)", 0.1, 99.0, 3.5, 0.1)
    pax = f2.number_input("Passengers", 1, 6, 1)
    pickup_date = f3.date_input("Pickup date", datetime(2025, 1, 15))

    g1, g2 = st.columns(2)
    pickup_t = g1.time_input("Pickup time", time(19, 30))
    dropoff_t = g2.time_input("Dropoff time", time(19, 55))

    h1, h2 = st.columns(2)
    pu_id = h1.number_input("Pickup zone ID", 1, 265, 132)
    do_id = h2.number_input("Dropoff zone ID", 1, 265, 79)

    submitted = st.form_submit_button("Predict")

if submitted:
    pickup_dt = datetime.combine(pickup_date, pickup_t)
    dropoff_dt = datetime.combine(pickup_date, dropoff_t)
    if dropoff_dt <= pickup_dt:
        st.error("Dropoff must be after pickup.")
    else:
        proba = None
        source = None
        # Try the FastAPI service first.
        if requests is not None:
            try:
                payload = {
                    "trip_distance": dist,
                    "pickup_datetime": pickup_dt.isoformat(),
                    "dropoff_datetime": dropoff_dt.isoformat(),
                    "pu_location_id": int(pu_id),
                    "do_location_id": int(do_id),
                    "passenger_count": int(pax),
                }
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
                if r.status_code == 200:
                    proba = r.json()["high_tip_probability"]
                    source = "FastAPI service"
            except Exception:
                pass
        # Fall back to the local model.
        if proba is None:
            try:
                model, pu_map, do_map, pu_prior, do_prior, _ = load_local_model()
                X = engineer_row(dist, pickup_dt, dropoff_dt, int(pu_id),
                                 int(do_id), int(pax), pu_map, do_map,
                                 pu_prior, do_prior)
                proba = float(np.asarray(model.predict(X)).ravel()[0])
                source = "local champion model"
            except Exception as e:
                st.error(f"Could not score the trip: {e}")

        if proba is not None:
            st.metric("Probability of a high tip (>20%)", f"{proba:.1%}")
            if proba >= 0.5:
                st.success("Prediction: HIGH tip likely")
            else:
                st.info("Prediction: high tip unlikely")
            st.caption(f"Scored via {source}.")
