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
characteristics known when the ride ends, before payment -- never from the
payment type or any money field.

Features (all known when the ride ends, before payment)
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

import logging
import sys

import pandas as pd

from nyc_tip.features import clean_trips

log = logging.getLogger(__name__)


def build(input_path: str, output_path: str) -> None:
    raw = pd.read_parquet(input_path)
    log.info("Read %s raw rows from %s", f"{len(raw):,}", input_path)
    out = clean_trips(raw)
    out.to_parquet(output_path, index=False)
    log.info("Saved %s rows -> %s (%.1f%% high tip)", f"{len(out):,}", output_path,
             100 * out["high_tip"].mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 3:
        sys.exit("Usage: python build_features.py <input_parquet> <output_parquet>")
    build(sys.argv[1], sys.argv[2])
