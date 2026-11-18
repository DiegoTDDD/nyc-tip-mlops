"""Generate a tiny synthetic NYC-Taxi-like sample for CI.

GitHub Actions cannot download the multi-hundred-MB monthly parquet files
on every run, so the CI pipeline trains on this small, deterministic
sample instead. It has the same columns build_features.py expects and a
weak but real signal, so a quick training run produces a sane model and
exercises the full MLflow logging + registry path.

Run:
    python tests/make_sample_data.py
Produces:
    features/train.parquet  (a few thousand rows)
"""

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def make_sample(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Deterministic feature table with the columns build_features.py writes."""
    rng = np.random.default_rng(seed)

    distance = rng.exponential(3.0, n).clip(0.1, 60)
    duration = (distance * rng.uniform(2.0, 5.0, n)).clip(1, 180)
    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    pax = rng.integers(1, 6, n)
    pu = rng.integers(1, 265, n)
    do = rng.integers(1, 265, n)

    # Weak but real signal: shorter trips and evening hours tip a bit more.
    p = (0.55
         + 0.20 * (distance < 3).astype(float)
         + 0.10 * (hour >= 18).astype(float)
         + 0.10 * rng.random(n)).clip(0.05, 0.95)
    high_tip = rng.binomial(1, p)

    df = pd.DataFrame({
        "trip_distance": distance,
        "trip_duration_min": duration,
        "pickup_hour": hour.astype("int32"),
        "pickup_dayofweek": dow.astype("int32"),
        "is_weekend": (dow >= 5).astype("int64"),
        "passenger_count": pax.astype("int64"),
        "pu_location_id": pu.astype("int32"),
        "do_location_id": do.astype("int32"),
        "high_tip": high_tip.astype("int64"),
    })

    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = make_sample()
    os.makedirs("features", exist_ok=True)
    df.to_parquet("features/train.parquet", index=False)
    log.info("Wrote features/train.parquet with %s rows (%.1f%% positive)", f"{len(df):,}",
             100 * df["high_tip"].mean())


if __name__ == "__main__":
    main()
