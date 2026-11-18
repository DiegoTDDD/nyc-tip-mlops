"""Feature engineering (nyc_tip/features.py) and parity between training and the API."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nyc_tip.features import (
    FEATURE_DTYPES,
    MODEL_FEATURES,
    ZONE_SMOOTHING,
    ZoneEncodings,
    add_engineered_features,
    clean_trips,
    high_tip,
    model_frame,
    trip_features,
)


def raw_trips(rows):
    """Raw TLC records from (payment, fare, tip, distance, minutes, passengers, pu, do, pickup)."""
    records = []
    for pay, fare, tip, dist, minutes, pax, pu, do, pickup in rows:
        records.append({
            "payment_type": pay, "fare_amount": fare, "tip_amount": tip,
            "trip_distance": dist, "passenger_count": pax,
            "PULocationID": pu, "DOLocationID": do,
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + timedelta(minutes=minutes),
        })
    return pd.DataFrame(records)


WED = datetime(2025, 1, 15, 19, 30)  # a Wednesday
SAT = datetime(2025, 1, 18, 9, 0)  # a Saturday


def test_high_tip_is_more_than_twenty_percent_of_the_fare():
    tips = pd.Series([3.0, 1.0, 0.0, 2.0])
    assert high_tip(tips, pd.Series([10.0] * 4)).tolist() == [1, 0, 0, 0]  # exactly 20% is not high


def test_clean_trips_keeps_only_credit_card_trips_with_sane_values():
    raw = raw_trips([
        (1, 10.0, 3.0, 2.0, 25, 1, 132, 79, WED),   # kept, high tip
        (2, 10.0, 0.0, 2.0, 25, 1, 132, 79, WED),   # cash: dropped (label leak)
        (1, 0.0, 0.0, 2.0, 25, 1, 132, 79, WED),    # zero fare: dropped
        (1, 10.0, 1.0, 0.0, 25, 1, 132, 79, WED),   # zero distance: dropped
        (1, 10.0, 1.0, 2.0, 300, 1, 132, 79, WED),  # 5-hour trip: dropped
        (1, 10.0, 1.0, 2.0, 25, 9, 132, 79, WED),   # 9 passengers: dropped
        (1, 10.0, 1.0, 2.0, 20, None, 4, 5, SAT),   # kept, passengers filled with 1
    ])
    out = clean_trips(raw)
    assert len(out) == 2
    assert out["high_tip"].tolist() == [1, 0]
    assert out["trip_duration_min"].tolist() == [25.0, 20.0]
    assert out["is_weekend"].tolist() == [0, 1]
    assert out["pickup_hour"].tolist() == [19, 9]
    assert out["passenger_count"].tolist() == [1, 1]
    assert out[["pu_location_id", "do_location_id"]].values.tolist() == [[132, 79], [4, 5]]
    assert out["trip_id"].tolist() == [0, 1]


def test_speed_is_capped_and_hours_lie_on_a_circle():
    df = pd.DataFrame({"trip_distance": [50.0, 3.0], "trip_duration_min": [1.0, 30.0],
                       "pickup_hour": [23, 0]})
    out = add_engineered_features(df)
    assert out["avg_speed_mph"].tolist() == [80.0, 6.0]
    late, midnight = out.iloc[0], out.iloc[1]
    assert np.hypot(late.hour_sin - midnight.hour_sin, late.hour_cos - midnight.hour_cos) < 0.3


def test_zone_rates_are_smoothed_toward_the_overall_rate():
    train = pd.DataFrame({"pu_location_id": [1] * 10 + [2] * 10,
                          "do_location_id": [7] * 20,
                          "high_tip": [1] * 10 + [0] * 10})
    enc = ZoneEncodings.fit(train)
    assert enc.pu_prior == 0.5
    expected_zone_1 = (1.0 * 10 + 0.5 * ZONE_SMOOTHING) / (10 + ZONE_SMOOTHING)
    assert enc.pu_rate[1] == pytest.approx(expected_zone_1)
    assert 0.5 < enc.pu_rate[1] < 1.0 and 0.0 < enc.pu_rate[2] < 0.5


def test_unknown_zones_fall_back_to_the_prior():
    enc = ZoneEncodings({132: 0.81}, {79: 0.66}, 0.5, 0.4)
    out = enc.apply(pd.DataFrame({"pu_location_id": [132, 999], "do_location_id": [79, 888]}))
    assert out["pu_zone_tip_rate"].tolist() == [0.81, 0.5]
    assert out["do_zone_tip_rate"].tolist() == [0.66, 0.4]


def test_encodings_survive_the_json_round_trip():
    import json

    enc = ZoneEncodings({132: 0.81}, {79: 0.66}, 0.5, 0.4)
    again = ZoneEncodings.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert again == enc


def test_api_trip_features_match_training_features():
    """The API's one-trip path must give the same row as the batch path used in training."""
    rng = np.random.default_rng(0)
    rows = [
        (1, 20.0, 5.0, float(rng.uniform(0.3, 30)), int(rng.integers(2, 180)),
         int(rng.integers(1, 7)), int(rng.integers(1, 266)), int(rng.integers(1, 266)),
         datetime(2025, 1, 1) + timedelta(minutes=int(rng.integers(0, 60 * 24 * 30))))
        for _ in range(60)
    ]
    trips = clean_trips(raw_trips(rows))
    enc = ZoneEncodings.fit(trips.iloc[:30])  # some zones in the rest are unknown
    batch = model_frame(trips, enc)
    for i, (_, _, _, dist, minutes, pax, pu, do, pickup) in enumerate(rows):
        one = trip_features(dist, pickup, pickup + timedelta(minutes=minutes), pu, do, pax, enc)
        pd.testing.assert_frame_equal(one.reset_index(drop=True),
                                      batch.iloc[[i]].reset_index(drop=True))


def test_trip_features_have_the_model_columns_and_dtypes():
    out = trip_features(3.0, WED, WED + timedelta(minutes=20), 132, 79, 2, ZoneEncodings())
    assert list(out.columns) == MODEL_FEATURES
    assert out.dtypes.astype(str).to_dict() == FEATURE_DTYPES


def test_trip_features_reject_a_dropoff_before_the_pickup():
    with pytest.raises(ValueError):
        trip_features(3.0, WED, WED, 1, 2, 1, ZoneEncodings())
