"""Versioned training for the NYC Taxi high-tip classifier.

Trains an XGBoost classifier on the leakage-free feature table, logs
everything to MLflow (params, metrics, the model artifact, and a few
diagnostic plots), and registers the model in the MLflow Model Registry
under a fixed name. If the new run beats the model currently in the
"champion" alias on validation AUC, it is promoted automatically.

Backend
-------
Tracking + registry are stored locally in a SQLite file (mlflow.db) with
artifacts under ./mlruns. This keeps the project self-contained (no
server, no cloud) while still enabling the Model Registry, which requires
a database backend.

Usage
-----
    python training/train.py
    python training/train.py --data features/train.parquet --n-estimators 400
"""

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
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


# Fixed names so every part of the project (API, dashboard, CI) agrees.
EXPERIMENT_NAME = "nyc_tip_classification"
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"

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


def get_champion_auc(client: MlflowClient) -> float:
    """Return the validation AUC of the current champion, or -1 if none."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return -1.0
    run = client.get_run(mv.run_id)
    return float(run.data.metrics.get("val_auc", -1.0))


def main() -> None:
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
    client = MlflowClient()

    # --- Load data --------------------------------------------------------
    print(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data, columns=FEATURES + [TARGET])
    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"  sampled down to {len(df):,} rows for training speed")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  train: {len(X_train):,}   val: {len(X_val):,}")

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

        print("Training XGBoost ...")
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
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

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

        # --- Log + register the model ------------------------------------
        signature = mlflow.models.infer_signature(X_val, proba)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # The freshly registered version is the latest one.
        new_version = client.get_latest_versions(REGISTERED_MODEL_NAME)[0]

        # --- Promote if it beats the current champion --------------------
        champ_auc = get_champion_auc(client)
        if metrics["val_auc"] > champ_auc:
            client.set_registered_model_alias(
                REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version.version
            )
            print(
                f"\nPromoted version {new_version.version} to '{CHAMPION_ALIAS}' "
                f"(val_auc {metrics['val_auc']:.4f} > previous {champ_auc:.4f})."
            )
        else:
            print(
                f"\nKept existing champion (val_auc {champ_auc:.4f} >= "
                f"new {metrics['val_auc']:.4f}). New version {new_version.version} "
                "registered but not promoted."
            )

        print(f"\nRun complete. Run ID: {run.info.run_id}")
        print("View the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
