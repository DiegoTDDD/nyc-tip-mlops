"""The promotion rule: same rows, and a gain larger than the bootstrap noise."""

import numpy as np
import pandas as pd
import pytest

from nyc_tip.promotion import bootstrap_gain, decide, fingerprint

RNG = np.random.default_rng(0)
N = 20_000
Y = RNG.integers(0, 2, N)
SIGNAL = Y + RNG.normal(0, 1.5, N)  # a weak but real signal, AUC about 0.68


def noisier(scale: float) -> np.ndarray:
    return Y + RNG.normal(0, scale, N)


def test_no_champion_promotes():
    d = decide(Y, SIGNAL, None)
    assert d.promote and d.champion_auc is None and d.reason == "no champion yet"


def test_a_clearly_better_challenger_is_promoted():
    d = decide(Y, SIGNAL, noisier(3.0), rounds=100)
    assert d.promote and d.gain > 0 and d.gain_ci_low > 0
    assert d.gain_ci_low <= d.gain <= d.gain_ci_high


def test_the_same_model_is_not_promoted():
    d = decide(Y, SIGNAL, SIGNAL.copy(), rounds=50)
    assert not d.promote and d.gain == 0


def test_a_gain_inside_the_noise_is_not_promoted():
    # a challenger that differs only by tiny random noise
    challenger = SIGNAL + RNG.normal(0, 0.01, N)
    d = decide(Y, challenger, SIGNAL, rounds=100)
    assert not d.promote
    assert d.gain_ci_low <= 0 <= d.gain_ci_high


def test_a_worse_challenger_is_not_promoted():
    d = decide(Y, noisier(3.0), SIGNAL, rounds=100)
    assert not d.promote and d.gain < 0


def test_the_bootstrap_is_paired_and_reproducible():
    # Both models are scored on the same resample, so a model and a rescaled copy of it (same
    # ranking, same AUC) differ by exactly zero in every round.
    assert bootstrap_gain(Y, SIGNAL, SIGNAL * 0.5, rounds=20, seed=3) == pytest.approx((0.0, 0.0))
    other = noisier(2.0)
    assert bootstrap_gain(Y, SIGNAL, other, rounds=30, seed=1) == bootstrap_gain(
        Y, SIGNAL, other, rounds=30, seed=1
    )


def test_the_fingerprint_identifies_the_rows():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    assert fingerprint(frame) == fingerprint(frame.copy())
    assert fingerprint(frame) != fingerprint(frame.iloc[::-1])
    assert len(fingerprint(frame)) == 16
