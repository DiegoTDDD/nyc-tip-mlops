# Implementation Plan: Monthly retraining on real data, fair promotion and performance monitoring

**Branch**: `001-monthly-retrain-loop` | **Date**: 2026-11-27 | **Spec**: [spec.md](spec.md)

## Summary

A promotion module compares two models on the same rows with a paired bootstrap of the AUC
difference. `train_v2.py` uses it against the MLflow champion on a holdout file. A new
`training/retrain.py` runs the production loop on the committed bundle: find the newest TLC
month, record the champion's AUC on it, train a challenger on the month before, compare,
and rewrite the bundle on promotion. A monthly workflow runs it on real data, commits the
result and opens an issue when performance drops. The API checks an API key.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: XGBoost 3.2, scikit-learn 1.9, MLflow 3.17, FastAPI 0.136

**Storage**: `serving_model/` (champion bundle, registry history), `monitoring/performance_history.json`

**Testing**: pytest on synthetic months (tests/make_sample_data.py)

**Project Type**: ML service with training, serving, monitoring and a dashboard

**Performance Goals**: monthly job under 20 minutes on a GitHub runner (300,000-row training sample)

**Constraints**: free; no persistent MLflow server in CI, so the bundle is the record

## Constitution Check

| Principle | How |
|---|---|
| I. No leakage | Train on L-1, evaluate on L; encodings fit on the training sample only. |
| II. Comparable | Same rows, fingerprinted; bootstrap interval must exclude zero. |
| III. One feature implementation | The loop uses `clean_trips` and `model_frame`. |
| IV. Loop closes in CI | Monthly workflow on real TLC data commits the decision. |
| V. Honest limits | README rewritten. |

## Project Structure

```text
nyc_tip/promotion.py           # paired bootstrap, fingerprint, decision
training/train_v2.py           # compares with the MLflow champion on a holdout
training/retrain.py            # the monthly loop on the committed bundle
serving/app_api.py             # API key
.github/workflows/retrain.yml  # monthly, real data, commits, alerts
tests/test_promotion.py, tests/test_retrain.py, tests/test_api.py (key)
docs/adr/0006-promotion-on-a-shared-out-of-time-month.md, 0007-monthly-retrain-loop.md
```
