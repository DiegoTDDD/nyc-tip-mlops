"""Shared fixtures: project on sys.path, a sample feature table and a trained MLflow registry."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests", ROOT / "training"):
    sys.path.insert(0, str(path))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TINY_GRID = {"max_depth": [3], "n_estimators": [30], "learning_rate": [0.1]}


@pytest.fixture(scope="session")
def sample_table():
    from make_sample_data import make_sample

    return make_sample(n=3000)


@pytest.fixture(scope="session")
def registry(tmp_path_factory, sample_table):
    """Train twice into a throwaway MLflow store; return the store and both results."""
    import train_v2
    from make_sample_data import make_sample

    folder = tmp_path_factory.mktemp("mlflow")
    data = folder / "train.parquet"
    sample_table.to_parquet(data, index=False)
    holdout = folder / "holdout.parquet"
    make_sample(n=3000, seed=7).to_parquet(holdout, index=False)
    uri = f"sqlite:///{folder / 'mlflow.db'}"
    previous = Path.cwd()
    os.chdir(folder)  # MLflow writes artifacts under ./mlruns
    try:
        first = train_v2.run(data, sample=0, tracking_uri=uri, grid=TINY_GRID, holdout=holdout)
        second = train_v2.run(data, sample=0, tracking_uri=uri, grid=TINY_GRID, holdout=holdout)
    finally:
        os.chdir(previous)
    return {"uri": uri, "folder": folder, "first": first, "second": second}
