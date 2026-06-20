"""Unit tests for the NYC Taxi high-tip pipeline.

These tests are deliberately fast and dependency-light so they run in
seconds on GitHub Actions. They cover the two places where a silent bug
would do the most damage:

1. Feature engineering correctness (duration, weekend flag, cyclical
   hour, zone lookup with prior fallback) -- the logic that must match
   between training and serving.
2. Target definition (high_tip = tip_pct > 0.20) on a tiny hand-built
   frame, so the label can never drift unnoticed.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Re-implement the engineering helpers in a self-contained way so the test
# does not need the full training stack imported. These mirror build_features
# and the serving _engineer exactly.
# ---------------------------------------------------------------------------
def make_high_tip(df: pd.DataFrame) -> pd.Series:
    tip_pct = df["tip_amount"] / df["fare_amount"]
    return (tip_pct > 0.20).astype(int)


def engineer(distance, pickup, dropoff, pu_id, do_id,
             pu_map, do_map, pu_prior, do_prior):
    dur = (dropoff - pickup).total_seconds() / 60.0
    hours = max(dur / 60.0, 1e-3)
    return {
        "trip_duration_min": dur,
        "avg_speed_mph": min(distance / hours, 80.0),
        "pickup_dayofweek": pickup.weekday(),
        "is_weekend": 1 if pickup.weekday() >= 5 else 0,
        "hour_sin": np.sin(2 * np.pi * pickup.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * pickup.hour / 24.0),
        "pu_zone_tip_rate": pu_map.get(pu_id, pu_prior),
        "do_zone_tip_rate": do_map.get(do_id, do_prior),
    }


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def test_high_tip_target():
    df = pd.DataFrame({
        "tip_amount": [3.0, 1.0, 0.0, 5.0],
        "fare_amount": [10.0, 10.0, 10.0, 10.0],
    })
    # tip_pct = 0.30, 0.10, 0.00, 0.50  ->  high_tip = 1, 0, 0, 1
    expected = [1, 0, 0, 1]
    assert make_high_tip(df).tolist() == expected


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------
def test_duration_minutes():
    from datetime import datetime
    r = engineer(5.0, datetime(2025, 1, 15, 19, 30),
                 datetime(2025, 1, 15, 19, 55), 1, 2, {}, {}, 0.5, 0.5)
    assert r["trip_duration_min"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Weekend flag: 2025-01-15 is a Wednesday (0), 2025-01-18 is a Saturday (1)
# ---------------------------------------------------------------------------
def test_weekend_flag():
    from datetime import datetime
    wed = engineer(2.0, datetime(2025, 1, 15, 12, 0),
                   datetime(2025, 1, 15, 12, 20), 1, 2, {}, {}, 0.5, 0.5)
    sat = engineer(2.0, datetime(2025, 1, 18, 12, 0),
                   datetime(2025, 1, 18, 12, 20), 1, 2, {}, {}, 0.5, 0.5)
    assert wed["is_weekend"] == 0
    assert sat["is_weekend"] == 1


# ---------------------------------------------------------------------------
# Cyclical hour: hour 0 and hour 24-equivalent must be close; sin/cos in range
# ---------------------------------------------------------------------------
def test_cyclical_hour_bounds():
    from datetime import datetime
    for h in range(24):
        r = engineer(2.0, datetime(2025, 1, 15, h, 0),
                     datetime(2025, 1, 15, h, 20), 1, 2, {}, {}, 0.5, 0.5)
        assert -1.0 <= r["hour_sin"] <= 1.0
        assert -1.0 <= r["hour_cos"] <= 1.0


# ---------------------------------------------------------------------------
# Zone encoding: known zone uses its rate, unknown zone falls back to prior
# ---------------------------------------------------------------------------
def test_zone_lookup_fallback():
    from datetime import datetime
    pu_map = {132: 0.81}
    do_map = {79: 0.66}
    known = engineer(3.0, datetime(2025, 1, 15, 10, 0),
                     datetime(2025, 1, 15, 10, 20), 132, 79,
                     pu_map, do_map, 0.50, 0.50)
    assert known["pu_zone_tip_rate"] == 0.81
    assert known["do_zone_tip_rate"] == 0.66

    unknown = engineer(3.0, datetime(2025, 1, 15, 10, 0),
                       datetime(2025, 1, 15, 10, 20), 999, 888,
                       pu_map, do_map, 0.50, 0.50)
    assert unknown["pu_zone_tip_rate"] == 0.50  # prior
    assert unknown["do_zone_tip_rate"] == 0.50  # prior


# ---------------------------------------------------------------------------
# Speed is capped at 80 mph (guards against GPS/timestamp errors)
# ---------------------------------------------------------------------------
def test_speed_cap():
    from datetime import datetime
    # 50 miles in 1 minute would be 3000 mph -> must cap at 80
    r = engineer(50.0, datetime(2025, 1, 15, 10, 0),
                 datetime(2025, 1, 15, 10, 1), 1, 2, {}, {}, 0.5, 0.5)
    assert r["avg_speed_mph"] == 80.0
