# 0004. Local MLflow (SQLite + files) and a committed champion bundle for the public dashboard

**Status:** Accepted. Since [ADR 0007](0007-monthly-retrain-loop.md) the monthly loop writes the bundle on promotion, which makes it the registry of record for the deployed model; the API serves it with `MODEL_SOURCE=bundle`, and the dashboard's caches are keyed on its files so a new champion shows up after a push.

## Context
The model registry needs a database backend, and the project must run on one machine at no cost. The public dashboard runs on Streamlit Community Cloud, which has neither the MLflow store nor Docker.

## Decision
- MLflow tracking and the registry use `sqlite:///mlflow.db`, with artifacts under `./mlruns`. Both are gitignored.
- `export_for_deploy.py` copies the champion model, its zone encodings and the registry history into `serving_model/`, which is committed. The dashboard reads the live registry when it exists and falls back to this bundle otherwise.
- In Docker, `mlflow.db` stores absolute host paths, so the API re-roots them onto the mounted `./mlruns` (`_to_local`). The Docker CI job checks this.

## Alternatives considered
- **A hosted MLflow server (Databricks, a VM).** Costs money and needs credentials.
- **Commit `mlflow.db` and `mlruns/`.** Large, machine-specific paths and binary churn in Git.

## Consequences
- Everything runs locally and in CI for free.
- The public dashboard shows the champion as of the last export, not live training.
