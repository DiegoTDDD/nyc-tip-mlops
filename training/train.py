"""Baseline trainer for the NYC Taxi high-tip classifier.

Trains a single XGBoost model on the raw feature table (zone ids as plain
numbers, no engineered features) and logs it to MLflow with its metrics and
diagnostic plots, as the reference point for training/train_v2.py.

It does NOT register the model: its inputs differ from the features the API
builds (nyc_tip/features.py), so it must never become the champion the API
serves. Registration and promotion happen only in train_v2.py (ADR 0002).

Backend
-------
Tracking + registry are stored locally in a SQLite file (mlflow.db) with
artifacts under ./mlruns. This keeps the project self-contained (no
server, no cloud) while keeping the same store as train_v2.py.

Usage
-----
    python training/train.py
    python training/train.py --data features/train.parquet --n-estimators 400
"""

import argparse
import logging
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


EXPERIMENT_NAME = "nyc_tip_classification"  # same experiment as train_v2.py, for comparison

FEATURES = [
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="features/train.parquet")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--sample", type=int, default=400_000,
                        help="Rows to sample for speed; 0 = use all.")
    args = parser.parse_args()

    # --- Tracking backend: local SQLite + ./mlruns artifacts -------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Load data --------------------------------------------------------
    log.info(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data, columns=FEATURES + [TARGET])
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        log.info(f"  sampled down to {len(df):,} rows for training speed")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  train: {len(X_train):,}   val: {len(X_val):,}")

    # --- Class imbalance handling ----------------------------------------
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("train_rows", len(X_train))

        log.info("Training XGBoost ...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # --- Evaluate -----------------------------------------------------
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "val_auc": roc_auc_score(y_val, proba),
            "val_accuracy": accuracy_score(y_val, pred),
            "val_precision": precision_score(y_val, pred),
            "val_recall": recall_score(y_val, pred),
            "val_f1": f1_score(y_val, pred),
        }
        mlflow.log_metrics(metrics)
        log.info("Validation metrics:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

        # --- Diagnostic plots --------------------------------------------
        tmpdir = tempfile.mkdtemp()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['val_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve (validation)")
        plt.legend(loc="lower right")
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight", dpi=120)
        plt.close()
        mlflow.log_artifact(roc_path)

        # Confusion matrix
        cm = confusion_matrix(y_val, pred)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, f"{v:,}", ha="center", va="center")
        plt.xticks([0, 1], ["low", "high"])
        plt.yticks([0, 1], ["low", "high"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix (validation)")
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=120)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Feature importance
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(6, 4))
        plt.barh(
            [FEATURES[i] for i in order][::-1],
            importances[order][::-1],
        )
        plt.xlabel("Importance (gain)")
        plt.title("Feature importance")
        fi_path = os.path.join(tmpdir, "feature_importance.png")
        plt.savefig(fi_path, bbox_inches="tight", dpi=120)
        plt.close()
        mlflow.log_artifact(fi_path)

        # --- Log the model, without registering it ----------------------
        # The baseline uses raw zone ids, not the API's features (nyc_tip/features.py),
        # so it must never become the champion the API loads. It is logged only to
        # compare against train_v2.py in the MLflow UI (ADR 0002).
        mlflow.xgboost.log_model(
            model, name="model", signature=mlflow.models.infer_signature(X_val, proba)
        )
        mlflow.set_tag("role", "baseline (not registered)")

        log.info(f"Run complete. Run ID: {run.info.run_id}")
        log.info("View the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
