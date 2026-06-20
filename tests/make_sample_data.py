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

import os
import numpy as np
import pandas as pd


def main(n: int = 8000, seed: int = 42) -> None:
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

    os.makedirs("features", exist_ok=True)
    df.to_parquet("features/train.parquet", index=False)
    print(f"Wrote features/train.parquet with {len(df):,} rows "
          f"({df['high_tip'].mean():.1%} positive)")


if __name__ == "__main__":
    main()
