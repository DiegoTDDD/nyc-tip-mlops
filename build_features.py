"""Feature engineering for the NYC Taxi high-tip classification project.

Reads a raw Yellow Taxi parquet file and produces a clean, leakage-free
feature table saved as parquet. The same logic is applied to both the
2024 training month and the 2025 production/drift month so the two
datasets are directly comparable.

Target
------
high_tip = 1 if tip_amount / fare_amount > 0.20 else 0

To avoid label leakage from the payment channel (cash trips almost never
record a tip), we keep only credit-card trips (payment_type == 1) with a
positive fare. The model must then predict *who* tips well from trip
characteristics known at pickup time -- never from the payment type or
any post-trip money field.

Features (all known at the start of the trip)
---------------------------------------------
trip_distance      : miles
trip_duration_min  : minutes between pickup and dropoff
pickup_hour        : 0-23
pickup_dayofweek   : 0=Mon ... 6=Sun
is_weekend         : 1 if Sat/Sun
passenger_count    : number of passengers
pu_location_id     : pickup zone
do_location_id     : dropoff zone

Usage
-----
    python build_features.py <input_parquet> <output_parquet>

Example
-------
    python build_features.py data/yellow_tripdata_2024-01.parquet features/train.parquet
"""

import sys
import pandas as pd
import numpy as np


def build(input_path: str, output_path: str) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_parquet(input_path)
    n_raw = len(df)
    print(f"  raw rows: {n_raw:,}")

    # --- Keep only credit-card trips with a positive fare -----------------
    # This removes the payment-channel leakage and the rows where a tip
    # could not be recorded in the first place.
    df = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0)].copy()
    print(f"  after credit-card + positive-fare filter: {len(df):,}")

    # --- Target -----------------------------------------------------------
    tip_pct = df["tip_amount"] / df["fare_amount"]
    df["high_tip"] = (tip_pct > 0.20).astype(int)

    # --- Trip duration in minutes ----------------------------------------
    duration = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["trip_duration_min"] = duration

    # --- Time features ----------------------------------------------------
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)

    # --- Rename a few columns to clean snake_case ------------------------
    df = df.rename(
        columns={
            "PULocationID": "pu_location_id",
            "DOLocationID": "do_location_id",
        }
    )

    # --- Sanity filters on the engineered features -----------------------
    # Drop physically impossible / corrupt records. Bounds are generous;
    # they only cut clear data-entry errors, not legitimate variation.
    before = len(df)
    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 100)
        & (df["trip_duration_min"] > 0)
        & (df["trip_duration_min"] < 240)
        & (df["passenger_count"].fillna(1) > 0)
        & (df["passenger_count"].fillna(1) <= 6)
    ].copy()
    print(f"  after sanity filters: {len(df):,} (dropped {before - len(df):,})")

    # --- Final feature selection -----------------------------------------
    feature_cols = [
        "trip_distance",
        "trip_duration_min",
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
        "passenger_count",
        "pu_location_id",
        "do_location_id",
    ]
    keep = feature_cols + ["high_tip", "tpep_pickup_datetime"]
    out = df[keep].copy()

    # Fill the few missing passenger_count with the median (1).
    out["passenger_count"] = out["passenger_count"].fillna(1).astype(int)

    # A stable row id + the pickup timestamp will be needed by Feast later.
    out = out.reset_index(drop=True)
    out["trip_id"] = out.index.astype(int)

    out.to_parquet(output_path, index=False)

    print(f"\nSaved {len(out):,} rows -> {output_path}")
    print(f"high_tip balance: {out['high_tip'].mean():.1%} positive")
    print("Columns:", list(out.columns))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_features.py <input_parquet> <output_parquet>")
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
