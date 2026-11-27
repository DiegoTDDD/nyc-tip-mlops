# 0002. Only train_v2.py registers; a new version must beat the champion's AUC

**Status:** Accepted for who registers. The promotion rule below is superseded by [ADR 0006](0006-promotion-on-a-shared-out-of-time-month.md): both models are now scored on the same out-of-time month and the gain must clear a paired bootstrap; `should_promote` was removed, and the weekly CI retrain gave way to the monthly loop of [ADR 0007](0007-monthly-retrain-loop.md). Before this ADR, `train.py` (the baseline) registered versions too.

## Context
The API loads whatever version holds the `champion` alias and sends it the features that `nyc_tip/features.py` builds. The baseline trainer used different inputs: raw zone ids and no engineered features. It nonetheless registered under the same model name and could take the alias whenever its AUC was higher. The API would then have sent it columns it was not trained on.

## Decision
- Only `training/train_v2.py` registers models. `train.py` logs its run to the same MLflow experiment, tagged `role = baseline (not registered)`, so the two can still be compared in the UI.
- Promotion rule (`should_promote`): the new version takes the alias only if its validation AUC is strictly higher than the champion's. A tie keeps the incumbent. The first version is always promoted.
- Tests train twice into a temporary registry: the first version becomes champion, and an identical second run is registered but not promoted.

## Alternatives considered
- **Separate registered models for the baseline and v2.** Workable, but nothing serves the baseline, so registering it adds nothing.
- **Promote on any improvement in a different metric (F1, precision).** AUC does not depend on the threshold, and the API exposes the probability.
- **Manual promotion.** Safer for a real deployment, but the project demonstrates the automated loop. The weekly CI retrain runs it end to end.

## Consequences
- The champion always has the input schema the API sends.
- Runs are compared on their own validation splits. If the data changes between runs, the AUCs are not strictly comparable. A fixed, versioned holdout set is the next step.
