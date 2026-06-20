"""Improved, leakage-free training for the NYC Taxi high-tip classifier.

This is the "v2" trainer. Versus the baseline it adds:

1. Richer features
   - Target-encoded pickup/dropoff zones: the historical high-tip rate of
     each zone, computed ONLY on the training split and mapped onto the
     validation split. This turns a meaningless location ID into a strong
     predictor without leaking validation labels.
   - avg_speed_mph = trip_distance / trip_duration_hours (an interaction
     between distance and duration).
   - Cyclical hour encoding (sin/cos) so 23h and 0h are "close".

2. Hyperparameter search
   - A small grid over depth / n_estimators / learning_rate. Each
     combination is logged to MLflow as its own nested run, so the UI
     shows the full iteration history. The best by validation AUC is
     logged + registered in the parent run and promoted to "champion" if
     it beats the incumbent.

3. Stratified sampling (default 1,000,000 rows) for fast, representative
   iteration -- a deliberate cost/performance trade-off.

Anti-leakage note
-----------------
Zone target-encoding is fit on the training rows only. A global smoothing
prior is blended in so rare zones (few trips) fall back toward the overall
base rate rather than overfitting to a handful of trips.

Usage
-----
    python training/train_v2.py
    python training/train_v2.py --sample 0          # use all rows
"""

import argparse
import os
import tempfile
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier


EXPERIMENT_NAME = "nyc_tip_classification"
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"

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
TARGET = "high_tip"

# Engineered features the model actually trains on.
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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the non-target-encoded engineered features (no leakage risk)."""
    out = df.copy()
    hours = out["trip_duration_min"] / 60.0
    out["avg_speed_mph"] = out["trip_distance"] / hours.clip(lower=1e-3)
    out["avg_speed_mph"] = out["avg_speed_mph"].clip(upper=80)  # cap GPS errors
    out["hour_sin"] = np.sin(2 * np.pi * out["pickup_hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["pickup_hour"] / 24.0)
    return out


def fit_zone_tip_rate(train_df: pd.DataFrame, zone_col: str,
                      smoothing: float = 50.0) -> tuple[pd.Series, float]:
    """Smoothed mean-target encoding for a zone column, fit on TRAIN only.

    Returns a (zone -> rate) mapping and the global prior used as fallback.
    """
    global_rate = train_df[TARGET].mean()
    grp = train_df.groupby(zone_col)[TARGET]
    counts = grp.count()
    means = grp.mean()
    # Bayesian smoothing toward the global rate for low-count zones.
    smoothed = (means * counts + global_rate * smoothing) / (counts + smoothing)
    return smoothed, float(global_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="features/train.parquet")
    parser.add_argument("--sample", type=int, default=1_000_000,
                        help="Stratified rows to use; 0 = all rows.")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data, columns=BASE_FEATURES + [TARGET])

    if args.sample and len(df) > args.sample:
        # Stratified sample on the target to keep the 76/24 balance.
        # Sample each class separately, then concatenate -- robust across
        # pandas versions (avoids groupby.apply index quirks).
        frac = args.sample / len(df)
        parts = []
        for cls, grp in df.groupby(TARGET):
            parts.append(grp.sample(frac=frac, random_state=42))
        df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  stratified sample -> {len(df):,} rows")

    df = add_engineered_features(df)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    print(f"  train: {len(train_df):,}   val: {len(val_df):,}")

    # --- Target-encode zones on TRAIN only, map onto both splits ---------
    pu_map, pu_prior = fit_zone_tip_rate(train_df, "pu_location_id")
    do_map, do_prior = fit_zone_tip_rate(train_df, "do_location_id")

    for d in (train_df, val_df):
        d["pu_zone_tip_rate"] = d["pu_location_id"].map(pu_map).fillna(pu_prior)
        d["do_zone_tip_rate"] = d["do_location_id"].map(do_map).fillna(do_prior)

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[MODEL_FEATURES]
    y_val = val_df[TARGET]

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = neg / pos if pos > 0 else 1.0

    # --- Hyperparameter grid ---------------------------------------------
    grid = {
        "max_depth": [6, 8, 10],
        "n_estimators": [300, 600],
        "learning_rate": [0.1, 0.05],
    }
    combos = list(itertools.product(
        grid["max_depth"], grid["n_estimators"], grid["learning_rate"]
    ))
    print(f"\nSearching {len(combos)} hyperparameter combinations ...")

    best = {"auc": -1.0, "model": None, "params": None, "metrics": None}

    with mlflow.start_run(run_name="hpo_v2") as parent:
        mlflow.log_param("search_space", str(grid))
        mlflow.log_param("n_combinations", len(combos))
        mlflow.log_param("sample_rows", len(df))
        mlflow.log_param("n_features", len(MODEL_FEATURES))

        for i, (depth, n_est, lr) in enumerate(combos, 1):
            params = dict(
                max_depth=depth,
                n_estimators=n_est,
                learning_rate=lr,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=spw,
                eval_metric="auc",
                n_jobs=-1,
                random_state=42,
            )
            with mlflow.start_run(run_name=f"combo_{i}", nested=True):
                mlflow.log_params(params)
                model = XGBClassifier(**params)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                mlflow.log_metric("val_auc", auc)
                print(f"  [{i:2d}/{len(combos)}] depth={depth} "
                      f"n_est={n_est} lr={lr}  ->  AUC={auc:.4f}")
                if auc > best["auc"]:
                    pred = (proba >= 0.5).astype(int)
                    best = {
                        "auc": auc,
                        "model": model,
                        "params": params,
                        "proba": proba,
                        "pred": pred,
                    }

        # --- Final metrics for the best model ----------------------------
        proba = best["proba"]
        pred = best["pred"]
        metrics = {
            "val_auc": best["auc"],
            "val_accuracy": accuracy_score(y_val, pred),
            "val_precision": precision_score(y_val, pred),
            "val_recall": recall_score(y_val, pred),
            "val_f1": f1_score(y_val, pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
        print("\nBest model validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Persist the zone encodings so serving can reuse them.
        tmpdir = tempfile.mkdtemp()
        enc_path = os.path.join(tmpdir, "zone_encodings.json")
        import json
        with open(enc_path, "w") as f:
            json.dump(
                {
                    "pu_zone_tip_rate": {int(k): float(v) for k, v in pu_map.items()},
                    "do_zone_tip_rate": {int(k): float(v) for k, v in do_map.items()},
                    "pu_prior": pu_prior,
                    "do_prior": do_prior,
                    "model_features": MODEL_FEATURES,
                },
                f,
            )
        mlflow.log_artifact(enc_path)

        # --- Plots -------------------------------------------------------
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['val_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("ROC curve (validation)"); plt.legend(loc="lower right")
        p = os.path.join(tmpdir, "roc_curve.png")
        plt.savefig(p, bbox_inches="tight", dpi=120); plt.close()
        mlflow.log_artifact(p)

        cm = confusion_matrix(y_val, pred)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, cmap="Blues")
        for (a, b), v in np.ndenumerate(cm):
            plt.text(b, a, f"{v:,}", ha="center", va="center")
        plt.xticks([0, 1], ["low", "high"]); plt.yticks([0, 1], ["low", "high"])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Confusion matrix (validation)")
        p = os.path.join(tmpdir, "confusion_matrix.png")
        plt.savefig(p, bbox_inches="tight", dpi=120); plt.close()
        mlflow.log_artifact(p)

        importances = best["model"].feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 4))
        plt.barh([MODEL_FEATURES[i] for i in order][::-1],
                 importances[order][::-1])
        plt.xlabel("Importance (gain)"); plt.title("Feature importance")
        p = os.path.join(tmpdir, "feature_importance.png")
        plt.savefig(p, bbox_inches="tight", dpi=120); plt.close()
        mlflow.log_artifact(p)

        # --- Register + maybe promote ------------------------------------
        signature = mlflow.models.infer_signature(X_val, proba)
        mlflow.xgboost.log_model(
            best["model"], name="model", signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        new_version = max(versions, key=lambda v: int(v.version))

        try:
            champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
            champ_auc = float(client.get_run(champ.run_id).data.metrics.get("val_auc", -1))
        except Exception:
            champ_auc = -1.0

        if metrics["val_auc"] > champ_auc:
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version.version
            )
            print(f"\nPromoted version {new_version.version} to '{CHAMPION_ALIAS}' "
                  f"(AUC {metrics['val_auc']:.4f} > previous {champ_auc:.4f}).")
        else:
            print(f"\nKept champion (AUC {champ_auc:.4f} >= {metrics['val_auc']:.4f}). "
                  f"Version {new_version.version} registered, not promoted.")

        print(f"\nParent run ID: {parent.info.run_id}")
        print("UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
