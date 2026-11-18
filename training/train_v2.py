"""Leakage-free training with a hyperparameter search, MLflow tracking and champion promotion.

Versus the baseline (training/train.py) it adds:

1. Richer features (nyc_tip/features.py, shared with the API)
   - Pickup/dropoff zones encoded as their smoothed high-tip rate, fit on the
     training split only and mapped onto validation.
   - avg_speed_mph and a cyclical encoding of the pickup hour.
2. A 12-combination grid over depth, trees and learning rate. Each combination
   is a nested MLflow run; the best by validation AUC is registered.
3. Automatic promotion: the new version takes the `champion` alias only if its
   validation AUC is strictly higher than the current champion's (ADR 0002).
4. Stratified sampling (default 1,000,000 rows) for fast iteration.

Usage
-----
    python training/train_v2.py
    python training/train_v2.py --sample 0          # use all rows
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nyc_tip.features import (  # noqa: E402
    BASE_FEATURES,
    MODEL_FEATURES,
    TARGET,
    ZoneEncodings,
    model_frame,
)

log = logging.getLogger(__name__)

EXPERIMENT_NAME = "nyc_tip_classification"
REGISTERED_MODEL_NAME = "nyc_tip_classifier"
CHAMPION_ALIAS = "champion"
SEED = 42
GRID: dict[str, list[Any]] = {
    "max_depth": [6, 8, 10],
    "n_estimators": [300, 600],
    "learning_rate": [0.1, 0.05],
}


@dataclass
class TrainingResult:
    version: str
    metrics: dict[str, float]
    promoted: bool
    previous_champion_auc: float | None


def stratified_sample(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Sample each class separately to keep the class balance; rows=0 (or too many) = all."""
    if not rows or len(df) <= rows:
        return df
    frac = rows / len(df)
    parts = [grp.sample(frac=frac, random_state=SEED) for _, grp in df.groupby(TARGET)]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def should_promote(new_auc: float, champion_auc: float | None) -> bool:
    """A new version becomes champion only by beating the current one; ties keep the incumbent."""
    return champion_auc is None or new_auc > champion_auc


def champion_auc(client: MlflowClient) -> float | None:
    """Validation AUC of the current champion, or None if there is no champion yet."""
    try:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return client.get_run(champ.run_id).data.metrics.get("val_auc")


def _save_plots(y_val: pd.Series, proba: np.ndarray, pred: np.ndarray, model: XGBClassifier,
                auc: float, folder: Path) -> list[Path]:
    paths = []
    fpr, tpr, _ = roc_curve(y_val, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (validation)")
    plt.legend(loc="lower right")
    paths.append(folder / "roc_curve.png")
    plt.savefig(paths[-1], bbox_inches="tight", dpi=120)
    plt.close()

    cm = confusion_matrix(y_val, pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        plt.text(b, a, f"{v:,}", ha="center", va="center")
    plt.xticks([0, 1], ["low", "high"])
    plt.yticks([0, 1], ["low", "high"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (validation)")
    paths.append(folder / "confusion_matrix.png")
    plt.savefig(paths[-1], bbox_inches="tight", dpi=120)
    plt.close()

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh([MODEL_FEATURES[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importance (gain)")
    plt.title("Feature importance")
    paths.append(folder / "feature_importance.png")
    plt.savefig(paths[-1], bbox_inches="tight", dpi=120)
    plt.close()
    return paths


def run(data: str | Path, sample: int = 1_000_000, tracking_uri: str = "sqlite:///mlflow.db",
        grid: dict[str, list[Any]] | None = None) -> TrainingResult:
    """Train, log to MLflow, register the best model and promote it if it beats the champion."""
    grid = grid or GRID
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    log.info("Loading %s ...", data)
    df = stratified_sample(pd.read_parquet(data, columns=[*BASE_FEATURES, TARGET]), sample)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])
    log.info("train: %s   val: %s", f"{len(train_df):,}", f"{len(val_df):,}")

    encodings = ZoneEncodings.fit(train_df)  # training rows only
    X_train, y_train = model_frame(train_df, encodings), train_df[TARGET]
    X_val, y_val = model_frame(val_df, encodings), val_df[TARGET]
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    combos = list(itertools.product(grid["max_depth"], grid["n_estimators"], grid["learning_rate"]))
    log.info("Searching %d hyperparameter combinations ...", len(combos))
    best: dict[str, Any] = {"auc": -1.0}

    with mlflow.start_run(run_name="hpo_v2") as parent:
        mlflow.log_params({"search_space": str(grid), "n_combinations": len(combos),
                           "sample_rows": len(df), "n_features": len(MODEL_FEATURES)})
        for i, (depth, n_est, lr) in enumerate(combos, 1):
            params = {
                "max_depth": depth, "n_estimators": n_est, "learning_rate": lr,
                "subsample": 0.9, "colsample_bytree": 0.9, "scale_pos_weight": spw,
                "eval_metric": "auc", "n_jobs": -1, "random_state": SEED,
            }
            with mlflow.start_run(run_name=f"combo_{i}", nested=True):
                mlflow.log_params(params)
                model = XGBClassifier(**params).fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                auc = float(roc_auc_score(y_val, proba))
                mlflow.log_metric("val_auc", auc)
                log.info("  [%2d/%d] depth=%d n_est=%d lr=%s -> AUC=%.4f",
                         i, len(combos), depth, n_est, lr, auc)
                if auc > best["auc"]:
                    best = {"auc": auc, "model": model, "params": params, "proba": proba}

        proba = best["proba"]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "val_auc": best["auc"],
            "val_accuracy": float(accuracy_score(y_val, pred)),
            "val_precision": float(precision_score(y_val, pred)),
            "val_recall": float(recall_score(y_val, pred)),
            "val_f1": float(f1_score(y_val, pred)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
        for k, v in metrics.items():
            log.info("  %s: %.4f", k, v)

        folder = Path(tempfile.mkdtemp())
        enc_path = folder / "zone_encodings.json"
        enc_path.write_text(json.dumps(encodings.to_dict()), encoding="utf-8")
        mlflow.log_artifact(str(enc_path))  # the API loads it next to the model
        for path in _save_plots(y_val, proba, pred, best["model"], best["auc"], folder):
            mlflow.log_artifact(str(path))

        previous = champion_auc(client)
        mlflow.xgboost.log_model(
            best["model"], name="model",
            signature=mlflow.models.infer_signature(X_val, proba),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        new_version = max(versions, key=lambda v: int(v.version)).version
        promoted = should_promote(metrics["val_auc"], previous)
        if promoted:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version)
            log.info("Promoted version %s to '%s' (AUC %.4f, previous %s).", new_version,
                     CHAMPION_ALIAS, metrics["val_auc"],
                     "none" if previous is None else f"{previous:.4f}")
        else:
            log.info("Kept the champion (AUC %.4f >= %.4f). Version %s registered, not promoted.",
                     previous, metrics["val_auc"], new_version)
        log.info("Parent run ID: %s", parent.info.run_id)

    return TrainingResult(str(new_version), metrics, promoted, previous)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="features/train.parquet")
    parser.add_argument("--sample", type=int, default=1_000_000,
                        help="Stratified rows to use; 0 = all rows.")
    args = parser.parse_args()
    run(args.data, args.sample)
    log.info("UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
