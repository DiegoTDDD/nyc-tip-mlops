"""Training: sampling, promotion rule, reproducibility and the champion in the MLflow registry."""

import mlflow
import pytest
import train_v2
from mlflow.tracking import MlflowClient


def test_stratified_sample_keeps_the_class_balance(sample_table):
    sampled = train_v2.stratified_sample(sample_table, 1000)
    assert len(sampled) == pytest.approx(1000, abs=2)
    assert sampled["high_tip"].mean() == pytest.approx(sample_table["high_tip"].mean(), abs=0.01)
    assert train_v2.stratified_sample(sample_table, 0) is sample_table


def test_first_version_becomes_champion(registry):
    first = registry["first"]
    assert first.version == "1" and first.promoted and first.previous_champion_auc is None
    assert 0.5 < first.metrics["val_auc"] < 1.0


def test_training_is_reproducible(registry):
    for key in ("val_auc", "val_f1", "holdout_auc"):
        assert registry["second"].metrics[key] == registry["first"].metrics[key]


def test_the_second_run_is_compared_with_the_champion_on_the_same_holdout(registry):
    first, second = registry["first"].decision, registry["second"].decision
    assert first.fingerprint == second.fingerprint  # the same rows
    assert second.champion_auc == pytest.approx(first.challenger_auc)  # the champion, re-scored
    assert second.gain == pytest.approx(0) and not second.promote  # identical model: no gain


def test_an_equal_model_is_registered_but_not_promoted(registry):
    mlflow.set_tracking_uri(registry["uri"])
    client = MlflowClient()
    champion = client.get_model_version_by_alias("nyc_tip_classifier", "champion")
    versions = client.search_model_versions("name='nyc_tip_classifier'")
    assert str(champion.version) == "1"
    assert sorted(str(v.version) for v in versions) == ["1", "2"]
    assert registry["second"].promoted is False


def test_the_champion_run_carries_the_zone_encodings(registry):
    mlflow.set_tracking_uri(registry["uri"])
    client = MlflowClient()
    champion = client.get_model_version_by_alias("nyc_tip_classifier", "champion")
    names = [a.path for a in client.list_artifacts(champion.run_id)]
    assert "zone_encodings.json" in names
