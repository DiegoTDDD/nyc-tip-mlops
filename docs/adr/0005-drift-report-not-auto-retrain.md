# 0005. Drift is reported, not wired to automatic retraining

**Status:** Accepted

## Context
Evidently compares January 2024 (reference) with January 2025 (current). The only column that drifts is `payment_type`: the "unknown" share rose from 4% to 13%. That column is not a model feature, because the model only sees card trips.

## Decision
`monitoring/check_drift.py` writes an HTML report and `drift_summary.json`, which includes a `retrain_recommended` flag. The dashboard shows them. Nothing retrains automatically. The weekly CI retrain runs on synthetic sample data to exercise the code path, not to respond to drift.

## Alternatives considered
- **Retrain whenever drift is detected.** The drifting column is not a feature here, so this would retrain for nothing. Feature drift is not proof of a performance drop either.
- **Monitor live performance.** Needs ground-truth tips for live predictions, which this setup does not have.

## Consequences
- A person decides about retraining, using the report.
- The flag is informative, not an automated control. Closing the loop would need labelled production data and a performance threshold.
