"""Quick inspection of the NYC Taxi parquet files.

Prints schema, row counts, and a few summary stats so we can decide
on features and the binary target (high_tip) before writing any
training code.
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

FILES = {
    "train_2024_01": "data/yellow_tripdata_2024-01.parquet",
    "drift_2025_01": "data/yellow_tripdata_2025-01.parquet",
}

for label, path in FILES.items():
    print("=" * 70)
    print(f"FILE: {label}  ({path})")
    print("=" * 70)

    df = pd.read_parquet(path)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Columns and dtypes:")
    print(df.dtypes)

    print("\nFirst 3 rows:")
    print(df.head(3))

    # Tip percentage = tip_amount / fare_amount (guard against zero/neg fares)
    if "tip_amount" in df.columns and "fare_amount" in df.columns:
        valid = df["fare_amount"] > 0
        tip_pct = (df.loc[valid, "tip_amount"] / df.loc[valid, "fare_amount"])
        high_tip = (tip_pct > 0.20).mean()
        print(f"\nValid-fare rows: {valid.mean():.1%} of total")
        print(f"Mean tip_pct (valid fares): {tip_pct.mean():.3f}")
        print(f"Share with tip_pct > 20% (target = high_tip=1): {high_tip:.1%}")

    print("\npayment_type value counts (1=credit card usually tips):")
    if "payment_type" in df.columns:
        print(df["payment_type"].value_counts(dropna=False).head(10))

    print("\n")
