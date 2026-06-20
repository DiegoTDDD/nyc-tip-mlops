"""Data drift monitoring for the NYC Taxi high-tip classifier.

Compares the 2024 training data (reference) against the 2025 production
data (current) using Evidently. The 2025 month follows NYC's January
2025 congestion-pricing rollout, so real distribution shift is expected.

Outputs (written to monitoring/)
--------------------------------
drift_report.html : full interactive Evidently report (visual)
drift_summary.json: machine-readable summary the dashboard and the CI/CD
                    job consume to decide whether a retrain is needed

A retrain is flagged when the share of drifted columns crosses
DRIFT_SHARE_THRESHOLD (default 0.5, i.e. at least half the monitored
features drifted).

Usage
-----
    python monitoring/check_drift.py
    python monitoring/check_drift.py --reference features/train.parquet \
        --current features/prod.parquet --sample 200000
"""

import argparse
import json
import os

import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


# Columns we monitor for drift: the model's raw inputs + payment_type,
# the context column where the 2024->2025 population shift is visible.
# (The target high_tip lives in the feature table, not the monitoring
# snapshot, since the snapshot keeps the full pre-filter population.)
MONITORED_COLUMNS = [
    "trip_distance",
    "trip_duration_min",
    "pickup_hour",
    "pickup_dayofweek",
    "passenger_count",
    "pu_location_id",
    "do_location_id",
    "payment_type",
]

DRIFT_SHARE_THRESHOLD = 0.5
OUTPUT_DIR = "monitoring"


def load(path: str, sample: int) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=MONITORED_COLUMNS)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    return df


def summarize(result_dict: dict) -> dict:
    """Pull a compact summary out of Evidently's full result dict."""
    drifted_count = None
    drifted_share = None
    per_column = {}

    for m in result_dict.get("metrics", []):
        name = m.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            val = m.get("value", {})
            drifted_count = val.get("count")
            drifted_share = val.get("share")
        elif name.startswith("ValueDrift"):
            col = m.get("config", {}).get("column")
            score = m.get("value")
            threshold = m.get("config", {}).get("threshold", 0.1)
            if col is not None:
                per_column[col] = {
                    "drift_score": score,
                    "threshold": threshold,
                    "drifted": bool(score is not None and score > threshold),
                }

    return {
        "n_columns_monitored": len(per_column),
        "n_columns_drifted": int(drifted_count) if drifted_count is not None else None,
        "drift_share": drifted_share,
        "retrain_recommended": bool(
            drifted_share is not None and drifted_share >= DRIFT_SHARE_THRESHOLD
        ),
        "per_column": per_column,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="monitoring/ref_snapshot.parquet")
    parser.add_argument("--current", default="monitoring/cur_snapshot.parquet")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="Rows sampled from each side; 0 = all.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reference: {args.reference}")
    print(f"Current:   {args.current}")
    ref = load(args.reference, args.sample)
    cur = load(args.current, args.sample)
    print(f"Loaded reference={len(ref):,} current={len(cur):,} rows")

    report = Report(metrics=[DataDriftPreset()])
    ref_ds = Dataset.from_pandas(ref, data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(cur, data_definition=DataDefinition())

    print("Running Evidently data-drift report ...")
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    html_path = os.path.join(OUTPUT_DIR, "drift_report.html")
    result.save_html(html_path)
    print(f"Saved {html_path}")

    summary = summarize(result.dict())
    json_path = os.path.join(OUTPUT_DIR, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    print("\n--- Drift summary ---")
    print(f"Columns monitored : {summary['n_columns_monitored']}")
    print(f"Columns drifted   : {summary['n_columns_drifted']}")
    share = summary["drift_share"]
    print(f"Drift share       : {share:.1%}" if share is not None else "Drift share: n/a")
    print(f"Retrain recommended: {summary['retrain_recommended']}")
    print("\nPer-column drift:")
    for col, info in summary["per_column"].items():
        flag = "DRIFT" if info["drifted"] else "ok"
        score = info["drift_score"]
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"  {col:20s} score={score_s:>10s}  [{flag}]")


if __name__ == "__main__":
    main()
