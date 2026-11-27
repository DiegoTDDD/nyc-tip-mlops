"""Leakage-free training with a hyperparameter search, MLflow tracking and champion promotion.

Versus the baseline (training/train.py) it adds:

1. Richer features (nyc_tip/features.py, shared with the API)
   - Pickup/dropoff zones encoded as their smoothed high-tip rate, fit on the
     training split only and mapped onto validation.
   - avg_speed_mph and a cyclical encoding of the pickup hour.
2. A 12-combination grid over depth, trees and learning rate. Each combination
   is a nested MLflow run; the best by validation AUC is registered.
3. Fair promotion: the new version and the current champion are both scored on the
   same holdout file (default features/holdout.parquet, rows neither trained on). The
   new version takes the `champion` alias only if a paired bootstrap of the AUC
   difference puts the lower end of its 95% interval above zero (ADR 0006).
4. Stratified sampling (default 1,000,000 rows) for fast iteration.

Usage
-----
    python training/train_v2.py
    python training/train_v2.py --sample 0          # use all rows
    python training/train_v2.py --holdout features/holdout.parquet
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
from nyc_tip.promotion import Decision, decide  # noqa: E402

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
    previous_champion_auc: float | None  # the champion's AUC on the same holdout
    decision: Decision | None = None


@dataclass
class SearchResult:
    model: XGBClassifier
    encodings: ZoneEncodings
    params: dict[str, Any]
    metrics: dict[str, float]
    y_val: pd.Series
    proba: np.ndarray
    rows: int


def stratified_sample(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    """Sample each class separately to keep the class balance; rows=0 (or too many) = all."""
    if not rows or len(df) <= rows:
        return df
    frac = rows / len(df)
    parts = [grp.sample(frac=frac, random_state=SEED) for _, grp in df.groupby(TARGET)]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def champion_model(client: MlflowClient) -> tuple[XGBClassifier, ZoneEncodings] | None:
    """The current champion and its zone encodings, or None if there is no champion yet."""
    try:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    encodings = ZoneEncodings.load(client.download_artifacts(champ.run_id, "zone_encodings.json"))
    return model, encodings


def score(model: XGBClassifier, encodings: ZoneEncodings, frame: pd.DataFrame) -> np.ndarray:
    """Probability of a high tip for every row of a BASE_FEATURES frame."""
    return model.predict_proba(model_frame(frame, encodings))[:, 1]


def _save_plots(
    y_val: pd.Series,
    proba: np.ndarray,
    pred: np.ndarray,
    model: XGBClassifier,
    auc: float,
    folder: Path,
) -> list[Path]:
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


def search(
    df: pd.DataFrame, grid: dict[str, list[Any]] | None = None, log_runs: bool = False
) -> SearchResult:
    """Fit the grid on a training frame; return the best model by validation AUC.

    The validation split (20%) only chooses hyperparameters. Promotion uses a separate holdout.
    """
    grid = grid or GRID
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])
    log.info("train: %s   val: %s", f"{len(train_df):,}", f"{len(val_df):,}")
    encodings = ZoneEncodings.fit(train_df)  # training rows only
    X_train, y_train = model_frame(train_df, encodings), train_df[TARGET]
    X_val, y_val = model_frame(val_df, encodings), val_df[TARGET]
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    combos = list(itertools.product(grid["max_depth"], grid["n_estimators"], grid["learning_rate"]))
    log.info("Searching %d hyperparameter combinations ...", len(combos))
    best: dict[str, Any] = {"auc": -1.0}
    for i, (depth, n_est, lr) in enumerate(combos, 1):
        params = {
            "max_depth": depth,
            "n_estimators": n_est,
            "learning_rate": lr,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": spw,
            "eval_metric": "auc",
            "n_jobs": -1,
            "random_state": SEED,
        }
        model = XGBClassifier(**params).fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        if log_runs:
            with mlflow.start_run(run_name=f"combo_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("val_auc", auc)
        log.info(
            "  [%2d/%d] depth=%d n_est=%d lr=%s -> AUC=%.4f", i, len(combos), depth, n_est, lr, auc
        )
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
    return SearchResult(best["model"], encodings, best["params"], metrics, y_val, proba, len(df))


def run(
    data: str | Path,
    sample: int = 1_000_000,
    tracking_uri: str = "sqlite:///mlflow.db",
    grid: dict[str, list[Any]] | None = None,
    holdout: str | Path = "features/holdout.parquet",
) -> TrainingResult:
    """Train, log to MLflow, register the best model, and promote it if it beats the champion
    on the holdout by more than the bootstrap noise."""
    grid = grid or GRID
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    log.info("Loading %s ...", data)
    df = stratified_sample(pd.read_parquet(data, columns=[*BASE_FEATURES, TARGET]), sample)
    holdout_df = pd.read_parquet(holdout, columns=[*BASE_FEATURES, TARGET])

    with mlflow.start_run(run_name="hpo_v2") as parent:
        mlflow.log_params(
            {
                "search_space": str(grid),
                "sample_rows": len(df),
                "n_features": len(MODEL_FEATURES),
                "holdout": str(holdout),
            }
        )
        best = search(df, grid, log_runs=True)
        metrics = dict(best.metrics)

        champion = champion_model(client)
        p_new = score(best.model, best.encodings, holdout_df)
        p_champ = score(*champion, holdout_df) if champion else None
        decision = decide(holdout_df[TARGET], p_new, p_champ, holdout_df)
        metrics["holdout_auc"] = decision.challenger_auc
        if decision.champion_auc is not None:
            metrics["champion_holdout_auc"] = decision.champion_auc
            metrics["holdout_gain_ci_low"] = decision.gain_ci_low
            metrics["holdout_gain_ci_high"] = decision.gain_ci_high
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {"holdout_fingerprint": decision.fingerprint, "promotion_reason": decision.reason}
        )
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        for k, v in metrics.items():
            log.info("  %s: %.4f", k, v)

        folder = Path(tempfile.mkdtemp())
        enc_path = folder / "zone_encodings.json"
        enc_path.write_text(json.dumps(best.encodings.to_dict()), encoding="utf-8")
        mlflow.log_artifact(str(enc_path))  # the API loads it next to the model
        pred = (best.proba >= 0.5).astype(int)
        for path in _save_plots(
            best.y_val, best.proba, pred, best.model, metrics["val_auc"], folder
        ):
            mlflow.log_artifact(str(path))

        X_sig = model_frame(holdout_df.head(100), best.encodings)
        mlflow.xgboost.log_model(
            best.model,
            name="model",
            signature=mlflow.models.infer_signature(X_sig, best.model.predict_proba(X_sig)[:, 1]),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        new_version = max(versions, key=lambda v: int(v.version)).version
        if decision.promote:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version)
            log.info(
                "Promoted version %s to '%s': %s", new_version, CHAMPION_ALIAS, decision.reason
            )
        else:
            log.info("Version %s registered, champion kept: %s", new_version, decision.reason)
        log.info("Parent run ID: %s", parent.info.run_id)

    return TrainingResult(
        str(new_version), metrics, decision.promote, decision.champion_auc, decision
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="features/train.parquet")
    parser.add_argument(
        "--sample", type=int, default=1_000_000, help="Stratified rows to use; 0 = all rows."
    )
    parser.add_argument(
        "--holdout",
        default="features/holdout.parquet",
        help="Rows neither model trained on; both are scored on them.",
    )
    args = parser.parse_args()
    run(args.data, args.sample, holdout=args.holdout)
    log.info("UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
