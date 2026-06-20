"""Build monitoring snapshots that preserve the FULL trip population.

The model is trained only on credit-card trips (to avoid tip-channel
leakage), but drift monitoring should watch the whole incoming
population -- including the payment-type mix, which is exactly where the
2024->2025 shift shows up (the share of "unknown" payment_type jumped
after the Jan 2025 congestion-pricing rollout).

So this script produces a separate, lightweight snapshot per month that
keeps payment_type and the model's raw feature columns BEFORE the
credit-card filter. check_drift.py compares these two snapshots.

Usage
-----
    python build_monitoring_snapshot.py data/yellow_tripdata_2024-01.parquet monitoring/ref_snapshot.parquet
    python build_monitoring_snapshot.py data/yellow_tripdata_2025-01.parquet monitoring/cur_snapshot.parquet
"""

import sys
import pandas as pd
import numpy as np

MONITOR_COLS = [
    "trip_distance",
    "trip_duration_min",
    "pickup_hour",
    "pickup_dayofweek",
    "passenger_count",
    "pu_location_id",
    "do_location_id",
    "payment_type",
]


def build(input_path: str, output_path: str) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"  raw rows: {len(df):,}")

    # Keep positive fares only; do NOT filter by payment type (we want the
    # full population so payment_type drift is visible).
    df = df[df["fare_amount"] > 0].copy()

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df = df.rename(columns={"PULocationID": "pu_location_id",
                            "DOLocationID": "do_location_id"})

    df = df[
        (df["trip_distance"] > 0) & (df["trip_distance"] < 100)
        & (df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 240)
        & (df["passenger_count"].fillna(1) > 0)
        & (df["passenger_count"].fillna(1) <= 6)
    ].copy()
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(int)

    out = df[MONITOR_COLS].reset_index(drop=True)
    # Sample to keep the snapshot light for versioning / CI.
    if len(out) > 300_000:
        out = out.sample(n=300_000, random_state=42).reset_index(drop=True)

    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out):,} rows -> {output_path}")
    print("payment_type distribution:")
    print(out["payment_type"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_monitoring_snapshot.py <input> <output>")
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
