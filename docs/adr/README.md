# Architecture decision records

| # | Decision | Status |
|---|---|---|
| [0001](0001-no-label-leakage.md) | Credit-card trips only; zone encodings fit on the training split | Accepted |
| [0002](0002-one-registered-model-strict-promotion.md) | Only train_v2.py registers; a new version must beat the champion's AUC | Accepted, supersedes "both trainers register" |
| [0003](0003-shared-features-native-model-serving.md) | One feature module for training, API and dashboard; serve the native XGBoost model | Accepted |
| [0004](0004-local-mlflow-and-deploy-bundle.md) | Local MLflow (SQLite + files) and a committed champion bundle for the public dashboard | Accepted |
| [0005](0005-drift-report-not-auto-retrain.md) | Drift is reported, not wired to automatic retraining | Accepted |
