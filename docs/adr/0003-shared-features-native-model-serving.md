# 0003. One feature module for training, API and dashboard; serve the native XGBoost model

**Status:** Accepted

## Context
Three copies of the feature code existed: in the trainer, in the API and in the dashboard. The unit tests checked a fourth copy written inside the test file, so they did not test the shipped code at all.

The API also loaded the model through MLflow's generic `pyfunc` wrapper. For an XGBoost classifier, `pyfunc.predict()` returns the class (0 or 1), not a probability. `high_tip_probability` was therefore always exactly 0 or 1. CI checked only that the value was between 0 and 1, so the bug went unnoticed.

## Decision
- `nyc_tip/features.py` is the only feature code. `build_features.py`, `train_v2.py`, the API and the dashboard all import it. A test sends 60 random trips through both the API's one-trip path and the training batch path and requires identical rows, including dtypes.
- The API and the dashboard load the model with `mlflow.xgboost.load_model` and call `predict_proba`. The test that loads the real champion, and the CI smoke tests, require 0 < probability < 1.

## Alternatives considered
- **Keep pyfunc with a custom PythonModel that returns probabilities.** It also works, but adds a wrapper class to maintain. The native flavour is simpler, and this API only ever serves XGBoost.
- **A feature store.** Too much infrastructure for one model with ten features.

## Consequences
- Training-serving skew and output-type bugs are now caught by tests on every push.
- Without pyfunc, the MLflow signature is no longer enforced at request time. `model_frame` casts every column to the training dtypes instead, and a test checks those dtypes.
