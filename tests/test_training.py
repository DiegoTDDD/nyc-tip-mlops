"""Training: sampling, promotion rule, reproducibility and the champion in the MLflow registry."""

import mlflow
import pytest
import train_v2
from mlflow.tracking import MlflowClient


@pytest.mark.parametrize(
    ("new", "champion", "expected"),
    [(0.62, None, True), (0.63, 0.62, True), (0.62, 0.62, False), (0.61, 0.62, False)],
)
def test_a_new_version_must_beat_the_champion_to_replace_it(new, champion, expected):
    assert train_v2.should_promote(new, champion) is expected


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
    assert registry["second"].metrics == registry["first"].metrics


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
