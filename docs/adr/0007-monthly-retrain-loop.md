# 0007. A monthly retrain on real data, with the bundle in Git as the record

**Status:** Accepted. Supersedes [0005](0005-drift-report-not-auto-retrain.md) on what happens after monitoring; the Evidently drift report stays.

## Context
The scheduled retrain trained on synthetic data: it proved the code ran, not that the model
kept up with traffic. Drift was reported but nothing acted on it, because labels were thought
missing. They are not: TLC records the tip with the card payment, so every published month
labels that month's trips.

## Decision
- `training/retrain.py` runs each month (`retrain.yml`, on the 20th, when TLC has usually
  published): find the newest month L; score the champion on it and append the AUC to
  `monitoring/performance_history.json`; train a challenger on L-1 (300,000-trip stratified
  sample, the train_v2 search); compare on L by the rule in ADR 0006; on promotion rewrite
  `serving_model/`; append the decision to `serving_model/registry_history.json`; commit.
- A champion AUC under 0.60 on the newest month opens a GitHub issue labelled `model-alert`.
  0.60 is about two points under where the model has been on every month measured.
- A challenger is trained every month instead of only on drift: it costs a few minutes, and
  the promotion rule already refuses changes that are not real improvements.
- The committed bundle is the registry of record for the deployed API and dashboard. Local
  MLflow stays the registry for experiments (ADR 0004).

## Consequences
- The dashboard shows each version's training and evaluation months, both AUCs, the interval,
  the reason, and the champion's AUC month by month.
- Old months are never reused as evaluation for a later model, so no model is judged on data it
  trained on.
- The newest month stands in for next month's traffic; a sudden change within a month is seen
  only at the next run.
