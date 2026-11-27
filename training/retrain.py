"""The monthly production loop on the committed champion bundle (ADR 0007).

    python training/retrain.py                       # newest published month, real TLC data
    python training/retrain.py --eval-month 2026-09  # a given month (trains on the one before)

Tips are recorded with the card payment, so every month TLC publishes labels the trips of that
month. Each run:

1. Finds the newest published month L (looking back from the current month).
2. Scores the champion in serving_model/ on L and appends its AUC to
   monitoring/performance_history.json. Below the floor (0.60) the run reports an alert.
3. Trains a challenger on month L-1 (a stratified sample) with the same search as train_v2.
4. Compares the two on L, the same rows for both, with a paired bootstrap (nyc_tip/promotion.py).
5. On promotion, rewrites serving_model/ (model, encodings, champion pointer). Either way it
   appends the decision to serving_model/registry_history.json.

The bundle in Git is the registry of record for the deployed API and dashboard; local MLflow
stays the registry for experiments (ADR 0004). Exit code 0 in both outcomes; the alert and the
promotion are written to $GITHUB_OUTPUT for the workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from train_v2 import GRID, score, search, stratified_sample  # noqa: E402

from nyc_tip.features import BASE_FEATURES, TARGET, ZoneEncodings, clean_trips, model_frame  # noqa: E402
from nyc_tip.promotion import decide  # noqa: E402

log = logging.getLogger(__name__)

BUNDLE = ROOT / "serving_model"
PERFORMANCE = ROOT / "monitoring" / "performance_history.json"
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
AUC_FLOOR = 0.60  # below this the champion is not worth serving; a person should look
TRAIN_SAMPLE = 300_000
EVAL_SAMPLE = 500_000


def previous_month(month: str) -> str:
    year, mm = int(month[:4]), int(month[5:7])
    return f"{year - 1}-12" if mm == 1 else f"{year}-{mm - 1:02d}"


def newest_published(today: date | None = None, lookback: int = 6) -> str:
    """The newest month whose TLC file exists (TLC publishes about two months late)."""
    import requests

    month = (today or datetime.now(UTC).date()).strftime("%Y-%m")
    for _ in range(lookback):
        month = previous_month(month)
        if requests.head(TLC_URL.format(month=month), timeout=30).status_code == 200:
            return month
    raise SystemExit(f"no TLC file found in the {lookback} months before today")


def load_month(month: str, data_dir: Path) -> pd.DataFrame:
    """One month's clean trips (download once). Trips dated in another month are dropped."""
    import requests

    path = data_dir / f"yellow_tripdata_{month}.parquet"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(TLC_URL.format(month=month), stream=True, timeout=300) as r:
            r.raise_for_status()
            part = path.with_suffix(".part")
            with part.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            part.replace(path)
    trips = clean_trips(pd.read_parquet(path))
    trips = trips[trips["tpep_pickup_datetime"].dt.strftime("%Y-%m") == month]
    return trips[[*BASE_FEATURES, TARGET]].reset_index(drop=True)


def load_bundle(bundle: Path) -> tuple[XGBClassifier, ZoneEncodings, int]:
    model = XGBClassifier()
    model.load_model(bundle / "model" / "model.ubj")
    encodings = ZoneEncodings.load(bundle / "zone_encodings.json")
    version = json.loads((bundle / "champion.json").read_text())["version"]
    return model, encodings, int(version)


def write_bundle(bundle: Path, model: XGBClassifier, encodings: ZoneEncodings, version: int,
                 sample: pd.DataFrame) -> None:  # fmt: skip
    """Replace the bundle's model and encodings in the MLflow format the API and dashboard read."""
    import mlflow.xgboost
    from mlflow.models import infer_signature

    staging = Path(tempfile.mkdtemp()) / "model"
    X = model_frame(sample.head(100), encodings)
    signature = infer_signature(X, model.predict_proba(X)[:, 1])
    mlflow.xgboost.save_model(model, str(staging), model_format="ubj", signature=signature)
    shutil.rmtree(bundle / "model", ignore_errors=True)
    shutil.copytree(staging, bundle / "model")
    (bundle / "zone_encodings.json").write_text(json.dumps(encodings.to_dict()), encoding="utf-8")
    (bundle / "champion.json").write_text(json.dumps({"version": version}))


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


def run_loop(train: pd.DataFrame, evaluation: pd.DataFrame, train_month: str, eval_month: str,
             bundle: Path = BUNDLE, performance: Path = PERFORMANCE,
             grid: dict[str, list[Any]] | None = None, sample: int = TRAIN_SAMPLE,
             eval_sample: int = EVAL_SAMPLE, rounds: int = 200) -> dict:  # fmt: skip
    """One month of the loop on already-cleaned frames. Returns a summary."""
    champion, champ_enc, champ_version = load_bundle(bundle)
    if eval_sample and len(evaluation) > eval_sample:
        evaluation = evaluation.sample(eval_sample, random_state=0).reset_index(drop=True)

    # 1. How is the champion doing on the newest labelled month?
    p_champ = score(champion, champ_enc, evaluation)
    from sklearn.metrics import roc_auc_score

    champ_auc = float(roc_auc_score(evaluation[TARGET], p_champ))
    alert = champ_auc < AUC_FLOOR
    perf = [r for r in _read(performance) if r["month"] != eval_month]
    perf.append({"month": eval_month, "champion_version": champ_version, "auc": round(champ_auc, 4),
                 "rows": len(evaluation),
                 "high_tip_rate": round(float(evaluation[TARGET].mean()), 4),
                 "below_floor": alert})  # fmt: skip
    _write(performance, sorted(perf, key=lambda r: r["month"]))
    log.info("champion v%d on %s: AUC %.4f%s", champ_version, eval_month, champ_auc,
             "  BELOW THE FLOOR" if alert else "")  # fmt: skip

    # 2. Challenger trained on the month before, compared on the same rows.
    result = search(stratified_sample(train, sample), grid or GRID)
    p_new = score(result.model, result.encodings, evaluation)
    decision = decide(evaluation[TARGET], p_new, p_champ, evaluation, rounds=rounds)
    log.info("challenger trained on %s: %s", train_month, decision.reason)

    history = _read(bundle / "registry_history.json")
    new_version = max((int(r["version"]) for r in history), default=champ_version) + 1
    for r in history:
        r["is_champion"] = r["is_champion"] and not decision.promote
    history.append({
        "version": new_version,
        "trained_on": train_month,
        "evaluated_on": eval_month,
        "eval_auc": round(decision.challenger_auc, 4),
        "champion_version": champ_version,
        "champion_eval_auc": round(decision.champion_auc, 4),
        "gain_ci": [round(decision.gain_ci_low, 4), round(decision.gain_ci_high, 4)],
        "eval_rows": decision.rows,
        "eval_fingerprint": decision.fingerprint,
        "val_auc": round(result.metrics["val_auc"], 4),
        "val_f1": round(result.metrics["val_f1"], 4),
        "val_precision": round(result.metrics["val_precision"], 4),
        "val_recall": round(result.metrics["val_recall"], 4),
        "promoted": decision.promote,
        "reason": decision.reason,
        "decided_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "is_champion": decision.promote,
    })  # fmt: skip
    if decision.promote:
        write_bundle(bundle, result.model, result.encodings, new_version, evaluation)
    _write(bundle / "registry_history.json", history)
    return {"eval_month": eval_month, "train_month": train_month, "champion_auc": champ_auc,
            "alert": alert, "promoted": decision.promote, "version": new_version,
            "reason": decision.reason}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-month", help="YYYY-MM; default: the newest month TLC has published")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--sample", type=int, default=TRAIN_SAMPLE)
    args = parser.parse_args(argv)

    eval_month = args.eval_month or newest_published()
    train_month = previous_month(eval_month)
    log.info("evaluate on %s, train the challenger on %s", eval_month, train_month)
    train = load_month(train_month, args.data_dir)
    evaluation = load_month(eval_month, args.data_dir)
    summary = run_loop(train, evaluation,
                       train_month, eval_month, sample=args.sample)  # fmt: skip
    log.info(json.dumps(summary))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for key in ("eval_month", "train_month", "promoted", "alert", "version"):
                value = summary[key]
                f.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
            f.write(f"champion_auc={summary['champion_auc']:.4f}\n")


if __name__ == "__main__":
    main()
