"""The monthly loop on the committed bundle, with synthetic months."""

import json
import shutil

import numpy as np
import pytest
import retrain
from conftest import ROOT, TINY_GRID
from make_sample_data import make_sample


@pytest.fixture
def bundle(tmp_path):
    """A copy of the committed champion bundle, so the test can rewrite it."""
    target = tmp_path / "serving_model"
    shutil.copytree(ROOT / "serving_model", target)
    return target


def loop(bundle, tmp_path, train, evaluation, eval_month="2026-09"):
    return retrain.run_loop(
        train, evaluation, retrain.previous_month(eval_month), eval_month,
        bundle=bundle, performance=tmp_path / "perf.json", grid=TINY_GRID, sample=0,
        eval_sample=0, rounds=50,
    )  # fmt: skip


def test_months_step_back_across_the_year():
    assert retrain.previous_month("2026-01") == "2025-12"
    assert retrain.previous_month("2026-10") == "2026-09"


def new_pattern(n: int, seed: int):
    """Synthetic months where tipping follows the pickup zone (even zones tip), a pattern the
    committed champion, trained on real 2024 trips, has never seen."""
    df = make_sample(n, seed=seed)
    rng = np.random.default_rng(seed)
    even = (df["pu_location_id"] % 2 == 0).to_numpy()
    df["high_tip"] = rng.binomial(1, np.where(even, 0.9, 0.3))
    return df


def test_a_challenger_that_learns_the_new_month_replaces_the_champion(bundle, tmp_path):
    before = json.loads((bundle / "champion.json").read_text())["version"]
    summary = loop(bundle, tmp_path, new_pattern(6000, 11), new_pattern(6000, 12))
    assert summary["promoted"]
    history = json.loads((bundle / "registry_history.json").read_text())
    last = history[-1]
    assert last["version"] == summary["version"] == before + 1
    assert last["trained_on"] == "2026-08" and last["evaluated_on"] == "2026-09"
    assert last["eval_auc"] > last["champion_eval_auc"] and last["gain_ci"][0] > 0
    assert [r["version"] for r in history if r["is_champion"]] == [before + 1]
    assert json.loads((bundle / "champion.json").read_text())["version"] == before + 1
    model, encodings, version = retrain.load_bundle(bundle)  # the new bundle loads and scores
    assert version == before + 1
    assert 0 < retrain.score(model, encodings, make_sample(50, seed=3)).mean() < 1


def test_a_challenger_that_does_not_win_leaves_the_bundle_alone(bundle, tmp_path):
    # Train on noise: a model that knows nothing cannot beat the champion beyond the noise.
    noise = make_sample(4000, seed=21)
    noise["high_tip"] = noise["high_tip"].sample(frac=1, random_state=0).to_numpy()
    model_before = (bundle / "model" / "model.ubj").read_bytes()
    summary = loop(bundle, tmp_path, noise, new_pattern(6000, 12))
    assert not summary["promoted"]
    assert (bundle / "model" / "model.ubj").read_bytes() == model_before
    history = json.loads((bundle / "registry_history.json").read_text())
    assert history[-1]["promoted"] is False and history[-1]["is_champion"] is False
    assert sum(r["is_champion"] for r in history) == 1


def test_performance_history_and_the_floor_alert(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "AUC_FLOOR", 0.99)  # any real model is below this
    summary = loop(bundle, tmp_path, make_sample(3000, seed=11), make_sample(3000, seed=12))
    assert summary["alert"]
    perf = json.loads((tmp_path / "perf.json").read_text())
    assert perf[-1]["month"] == "2026-09" and perf[-1]["below_floor"] is True
    # running the same month again replaces its row instead of adding one
    loop(bundle, tmp_path, make_sample(3000, seed=11), make_sample(3000, seed=12))
    assert len(json.loads((tmp_path / "perf.json").read_text())) == 1
