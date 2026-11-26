# NYC Taxi High-Tip Classifier — End-to-End MLOps

[![CI](https://github.com/DiegoTDDD/nyc-tip-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/DiegoTDDD/nyc-tip-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Problem.** A model is only useful in production if every step around it can be trusted: training that can be repeated, a clear rule for which version is live, a service that computes features exactly as training did, and a signal when the data changes.

**Solution.** An end-to-end MLOps system around a classifier that predicts, when a NYC taxi ride ends and before the rider pays, whether the tip will be more than 20% of the fare. It has versioned training with MLflow, a model registry with automatic promotion, a FastAPI service, Evidently drift monitoring, a Streamlit operations dashboard and CI that runs the whole loop, including Docker.

- **Live dashboard:** [nyc-tip-mlops.streamlit.app](https://nyc-tip-mlops.streamlit.app/)
- **Design decisions:** [`docs/adr/`](docs/adr/README.md)
- **Author:** Diego Leonardo Alves de Andrade

![Streamlit operations dashboard](docs/screenshots/dashboard_model.png)

---

## Results

| | |
|---|---|
| Champion | `nyc_tip_classifier` v2 (XGBoost), promoted automatically |
| Validation AUC | **0.624** (honest ceiling without label leakage; see [ADR 0001](docs/adr/0001-no-label-leakage.md)) |
| Precision / recall / F1 at 0.5 | 0.813 / 0.620 / 0.704 |
| Hyperparameter search | 12 combinations, each a nested MLflow run. Deeper models overfit |
| Drift, Jan 2024 to Jan 2025 | Model features stable; `payment_type` drifted ("unknown" share 4% to 13%) |
| Tests | 29: the feature code, the training-vs-API feature check, the promotion rule, reproducible training, a real MLflow registry and every API endpoint |
| CI | Lint, tests, an exact-versions install and API call, the Docker Compose stack, a weekly retrain, a Windows run, a security audit and a live-app check |

The project's value is the engineering around the model, not the score: tipping is hard to predict from trip data, and the leakage-free framing keeps AUC modest on purpose.

---

## Architecture

```
NYC Taxi parquet (2024 train / 2025 production)
        │
        ▼
  Feature engineering  ──►  features/train.parquet      (model features, no leakage)
        │                   monitoring/*_snapshot.parquet (full population for drift)
        ▼
  Training (XGBoost)  ──►  MLflow tracking + registry  ──►  champion alias
        │                        │
        │                        ▼
        │                  FastAPI /predict  (loads champion from registry)
        ▼
  Evidently drift report  ──►  monitoring/drift_summary.json
        │
        ▼
  Streamlit dashboard  (metrics + registry + drift + live prediction)

  Docker Compose: API (8000) + MLflow UI (5000)
  GitHub Actions: tests + Docker test on push, scheduled retrain, security audit, app health
```

---

## The problem and the data

The model predicts, when a trip ends and before payment, whether the rider will
tip more than 20% of the fare. That is the moment the payment screen in the cab
offers tip choices, and every feature the model uses is known by then. The data is the public NYC TLC Yellow Taxi
trip records: **January 2024** as the training reference and **January
2025** as the production/drift period (the month following NYC's
congestion-pricing rollout).

### Modeling honesty (no leakage)

Cash trips almost never record a tip, so `payment_type` trivially predicts
"no tip". Leaving it in would produce a deceptively high score that any
reviewer would recognize as leakage. Instead the model trains **only on
credit-card trips** and never sees the payment channel. It must predict
*who* tips well from trip characteristics known when the ride ends: distance,
duration, hour, day of week, passenger count, and pickup/dropoff zone.

**When the prediction is made.** Duration, average speed and the dropoff zone
are only known at the end of the ride, so this is an end-of-ride model, not a
pickup-time one. An earlier version of this README said "at pickup"; that claim
was wrong for these features and was corrected. A pickup-time model would have
to drop them or replace them with estimates (a routed distance and duration).

This is a genuinely hard signal — tipping is idiosyncratic — so the honest
ceiling is low (validation AUC ≈ 0.62). The hyperparameter search confirmed
that deeper, larger models *overfit* and generalized worse, which is itself
evidence that the simple model is the right one. In a portfolio context an
honest 0.62 with a clean pipeline is worth more than a leaky 0.99.

### Features

| Feature | Notes |
|---|---|
| `trip_distance` | miles |
| `trip_duration_min` | from pickup/dropoff timestamps |
| `avg_speed_mph` | distance / duration, capped at 80 |
| `passenger_count` | riders in the trip |
| `pickup_dayofweek`, `is_weekend` | calendar features |
| `hour_sin`, `hour_cos` | cyclical encoding of pickup hour |
| `pu_zone_tip_rate`, `do_zone_tip_rate` | smoothed target-encoding of zones, **fit on the training split only** to avoid leakage |

---

## Versioned training & model registry (MLflow)

Every training run is logged to MLflow with its parameters, metrics, and
diagnostic plots. `train_v2.py` registers each new version of
`nyc_tip_classifier` and moves the `champion` alias to it only if its
validation AUC beats the current champion's. That alias is what the serving
layer loads, so promotion is the single source of truth for "what's in
production". The baseline `train.py` logs to the same experiment for
comparison but never registers, because its inputs differ from what the API
sends ([ADR 0002](docs/adr/0002-one-registered-model-strict-promotion.md)).

**Training runs — versions, sources, and registered models**
![MLflow runs](docs/screenshots/mlflow_runs.png)

The richer trainer runs a 12-combination hyperparameter search, each
combination logged as a nested child run so they can be compared side by side.

**Hyperparameter-search run — metrics and nested child runs**
![MLflow HPO](docs/screenshots/mlflow_hpo.png)

**Model registry — the `champion` alias points at the best version**
![Model registry](docs/screenshots/mlflow_registry.png)

---

## Serving the champion (FastAPI)

The API loads the champion straight from the MLflow registry and builds the
features with the same module training uses (`nyc_tip/features.py`), so a raw
trip in equals the training feature row; a test checks this on 60 random trips.
It loads the native XGBoost model and returns `predict_proba`
([ADR 0003](docs/adr/0003-shared-features-native-model-serving.md)). It exposes `/`, `/health`, `/predict`,
and `/predict/batch`.

**Interactive API docs — the four endpoints**
![API docs](docs/screenshots/api_docs.png)

**`/predict` — request body filled in**
![API predict](docs/screenshots/api_predict.png)

**`/predict` — successful response (HTTP 200)**
![API response](docs/screenshots/api_response.png)

---

## Data-drift monitoring (Evidently)

The monitoring layer compares the 2024 reference population against the 2025
current population and writes both an HTML report and a machine-readable
`drift_summary.json` the dashboard reads. It correctly flags the payment-type
shift while reporting the model's own features as stable.

**Drift summary — 1 of 8 columns drifted**
![Drift summary](docs/screenshots/evidently_summary.png)

**Per-column drift — payment-type drift detected (Jensen-Shannon distance)**
![Drift report](docs/screenshots/evidently_drift.png)

---

## Operations dashboard (Streamlit)

A single console that ties the system together: the champion's model card and
registry history, the live drift status, and an interactive prediction form
that uses the running API when available and falls back to the champion model
otherwise.

**Drift monitoring panel**
![Dashboard drift](docs/screenshots/dashboard_drift.png)

**Live prediction form**
![Dashboard prediction](docs/screenshots/dashboard_predict.png)

---

## Design decisions

| Decision | Why |
|---|---|
| [Credit-card trips only; zone encodings fit on the training split](docs/adr/0001-no-label-leakage.md) | Cash trips would leak the label; encodings fit on all rows would leak validation |
| [Only train_v2 registers; strict promotion](docs/adr/0002-one-registered-model-strict-promotion.md) | The champion must always accept the features the API sends |
| [Shared feature module; native XGBoost serving](docs/adr/0003-shared-features-native-model-serving.md) | Removes three copies of the feature code, and makes the API return a real probability |
| [Local MLflow and a committed champion bundle](docs/adr/0004-local-mlflow-and-deploy-bundle.md) | Free, reproducible, and the public dashboard still works without the store |
| [Drift is reported, not auto-retrained](docs/adr/0005-drift-report-not-auto-retrain.md) | The column that drifted is not a feature, and there are no live labels |

### Fixed in the latest review

- **The API returned a class, not a probability.** MLflow's `pyfunc` wrapper calls `predict()`, so `high_tip_probability` was always 0 or 1. The API and the dashboard now load the native XGBoost model and use `predict_proba`. CI requires a value strictly between 0 and 1.
- **The baseline trainer could take over the champion alias** with a model whose inputs the API does not send. It now logs for comparison only.
- **The tests checked a copy of the feature code** written inside the test file. They now import the real module, which training, the API and the dashboard share.

---

## Limitations

- **Modest signal.** AUC 0.62. Tipping depends on people, not trips, and the model sees only trip data.
- **Promotion compares each run's own validation split.** If the training data changes between runs, the AUCs are not strictly comparable. A fixed, versioned holdout set would make promotion airtight.
- **The weekly CI retrain uses synthetic sample data.** It exercises the code path, but it does not retrain on new TLC months.
- **Drift is reported, not acted on.** Closing the loop needs labelled production data and a performance threshold.
- **Single machine.** SQLite and a local artifact store, with no authentication on the API or the MLflow UI (both bound to localhost).

---

## Tech stack

Python 3.11 · pandas · scikit-learn · XGBoost · **MLflow** (tracking +
registry) · **Evidently** · **FastAPI** + Uvicorn · **Docker
Compose** · **GitHub Actions** · Streamlit · Plotly.

---

## Repository layout

```
nyc-tip-mlops/
├── nyc_tip/
│   └── features.py              # the only feature code: cleaning, target, encodings, API row
├── build_features.py            # raw parquet -> leakage-free feature table
├── build_monitoring_snapshot.py # full-population snapshots for drift
├── inspect_data.py              # quick look at the raw TLC files (schema, row counts)
├── export_for_deploy.py         # exports the champion into serving_model/
├── serving_model/               # versioned champion bundle used by Streamlit Cloud
├── training/
│   ├── train.py                 # baseline (logged for comparison, never registered)
│   └── train_v2.py              # features + hyperparameter search + registry promotion
├── serving/
│   └── app_api.py               # FastAPI serving the champion model
├── monitoring/
│   └── check_drift.py           # Evidently drift report + summary JSON
├── dashboard/
│   ├── app_dashboard.py         # Streamlit operations console
│   └── requirements.txt         # slim deps for Streamlit Cloud
├── tests/                       # pytest: features, training + registry, API
├── docs/
│   ├── adr/                     # architecture decision records
│   └── screenshots/             # images used in this README
├── .github/                     # CI, Windows check, security audit, app health
├── Dockerfile.api               # API image
├── docker-compose.yml           # API + MLflow UI
├── pyproject.toml               # project metadata, pytest and ruff settings
├── requirements.txt             # full project deps
└── requirements-api.txt         # slim deps for the API container
```

---

## How to run it locally

> Data files and the MLflow store are gitignored (they are large and
> machine-specific). The steps below regenerate everything from scratch.
> Several components are long-running servers (MLflow UI, the API, the
> dashboard) — run each of those in its **own terminal**, with the conda
> environment activated.

### 1. Environment and folders

```bash
conda create -n mlops python=3.11 -y
conda activate mlops
pip install -r requirements.txt

# These folders are gitignored (only their contents), so create them on a
# fresh clone before downloading data or writing features.
mkdir -p data features monitoring
```

### 2. Get the data

```bash
curl -o data/yellow_tripdata_2024-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
curl -o data/yellow_tripdata_2025-01.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
```

### 3. Build features

```bash
python build_features.py data/yellow_tripdata_2024-01.parquet features/train.parquet
python build_features.py data/yellow_tripdata_2025-01.parquet features/prod.parquet
```

### 4. Train (logs to MLflow, registers + promotes the champion)

```bash
python training/train_v2.py
```

This trains the model, logs everything to MLflow (`mlflow.db` + `mlruns/`),
and promotes the best run to the `champion` alias. To browse the runs and
the registry, start the MLflow UI **in a separate terminal** (it stays
running):

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# open http://localhost:5000
```

### 5. Serve the model (separate terminal)

```bash
uvicorn serving.app_api:app --port 8000
# open http://localhost:8000/docs
```

### 6. Monitor drift

```bash
python build_monitoring_snapshot.py data/yellow_tripdata_2024-01.parquet monitoring/ref_snapshot.parquet
python build_monitoring_snapshot.py data/yellow_tripdata_2025-01.parquet monitoring/cur_snapshot.parquet
python monitoring/check_drift.py
# open monitoring/drift_report.html
```

### 7. Dashboard (separate terminal)

```bash
streamlit run dashboard/app_dashboard.py
```

### 8. Everything in Docker

Requires steps 1–4 first, so that `mlflow.db` and `mlruns/` exist (they are
mounted into the containers). Then:

```bash
docker compose up --build
# API docs : http://localhost:8000/docs
# MLflow UI: http://localhost:5000
```

Both ports are published on `127.0.0.1` only, so the API and the MLflow UI are not
reachable from other machines on your network.

`mlflow.db` records the absolute path of the folder where the model was trained. That
folder does not exist inside the container, so when the registered path is missing the
API loads the same files from `./mlruns` (mounted at `/app/mlruns`). The *Docker compose
stack* job in CI checks this on every push.

> **Upgrading from MLflow 3.13 – 3.15?** MLflow 3.16 and later fix the AI Gateway SSRF advisory (GHSA-h7x2-h6g9-p789) that affected 3.13.0 – 3.15.2. The newer versions upgrade the SQLite registry (`mlflow.db`) automatically the first time they open it, and older MLflow versions refuse to open an upgraded database (`Detected out-of-date database schema`). Back up `mlflow.db`, then update every environment together: `pip install -r requirements.txt` and `docker compose up --build`.

### Run the tests

```bash
pytest
```

The suite trains two small models into a temporary MLflow store, so it needs the full `requirements.txt` but no data download.

---

## CI/CD

`.github/workflows/ci.yml` defines five jobs:

- **test** — `ruff` (type hints required in `nyc_tip/`, `training/`, `serving/` and `monitoring/`, and no `print` there) and the 29 tests, on every push and pull request.
- **pinned** — installs exactly `requirements.txt` (the documented versions), runs
  `pip check`, trains on the sample data, starts the API and calls `/health` and
  `/predict` (the answer must be a probability strictly between 0 and 1). This is what proves a dependency bump (for example MLflow)
  still works.
- **docker** — trains a sample model on the runner, runs `docker compose up --build`,
  waits for the API and the MLflow UI, calls `/predict` and checks that the published
  ports are bound to `127.0.0.1`.
- **retrain** — on a weekly schedule (and on demand via *Run workflow*),
  regenerates a small synthetic sample, runs the full training + MLflow
  logging + registry-promotion path, and uploads the model artifacts.
  This demonstrates the automated-retraining loop that closes the MLOps
  cycle without depending on the large external dataset.
- **keepalive** — on the weekly schedule, re-enables the workflow, because GitHub
  disables scheduled workflows after 60 days without repository activity (and a
  disabled workflow also stops running on push).

Two more workflows run next to it:

- `security-audit.yml` audits every `requirements*.txt` with `pip-audit` whenever
  dependencies change and every Monday. It fails only when a vulnerable package has a
  fixed version available.
- `app-health.yml` opens the [live dashboard](https://nyc-tip-mlops.streamlit.app) in a
  real browser after each push and once a day (waking it if it was asleep) and fails if
  the app shows an error.
