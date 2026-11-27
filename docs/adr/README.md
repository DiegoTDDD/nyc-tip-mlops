# Architecture decision records

| # | Decision | Status |
|---|---|---|
| [0001](0001-no-label-leakage.md) | Credit-card trips only; zone encodings fit on the training split | Accepted |
| [0002](0002-one-registered-model-strict-promotion.md) | Only train_v2.py registers; a new version must beat the champion's AUC | Accepted; promotion rule superseded by 0006 |
| [0003](0003-shared-features-native-model-serving.md) | One feature module for training, API and dashboard; serve the native XGBoost model | Accepted |
| [0004](0004-local-mlflow-and-deploy-bundle.md) | Local MLflow (SQLite + files) and a committed champion bundle for the public dashboard | Accepted |
| [0005](0005-drift-report-not-auto-retrain.md) | Drift is reported, not wired to automatic retraining | Superseded by 0007 |
| [0006](0006-promotion-on-a-shared-out-of-time-month.md) | Promotion compares both models on the same newest month, beyond the bootstrap noise | Accepted |
| [0007](0007-monthly-retrain-loop.md) | A monthly retrain on real data, with the bundle in Git as the record | Accepted |
