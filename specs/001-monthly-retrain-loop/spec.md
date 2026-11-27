# Feature Specification: Monthly retraining on real data, fair promotion and performance monitoring

**Feature Branch**: `001-monthly-retrain-loop`

**Created**: 2026-11-27

**Status**: Implemented

**Input**: User description: "Bring the MLOps project to senior level. The README's own limits:
promotion compares each run's own validation split, the weekly retrain uses synthetic data,
drift is reported but not acted on, the API has no authentication."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A model is promoted only when it is better on the same, newest data (Priority: P1)

The ML engineer trains a challenger. It replaces the champion only if both models, scored on
the same out-of-time month that neither trained on, show the challenger ahead by more than the
uncertainty of the comparison.

**Why this priority**: Comparing AUCs from different validation splits can promote a worse model.

**Independent Test**: With a challenger that is better, equal, or worse on a shared evaluation
set, the rule promotes, keeps, keeps. A tiny gain inside the bootstrap interval keeps.

**Acceptance Scenarios**:

1. **Given** a champion and a challenger, **When** they are compared, **Then** both are scored on
   the same evaluation rows, and the rows' fingerprint is recorded with the decision.
2. **Given** a paired bootstrap of the AUC difference, **When** the lower end of its 95%
   interval is not above zero, **Then** the champion stays.
3. **Given** no champion yet, **When** a model is trained, **Then** it becomes champion.

---

### User Story 2 - Retraining on the newest real data, every month (Priority: P2)

Every month CI finds the newest month TLC has published, trains a challenger on the month
before it, compares it with the champion on the newest month, and commits the new champion
bundle when it wins, with a row in the registry history saying why.

**Why this priority**: A retrain on synthetic data proves the code runs, not that the model
keeps up.

**Independent Test**: Run the loop on two real months; the history gains a row with both AUCs,
the interval and the decision; the bundle changes only on promotion.

**Acceptance Scenarios**:

1. **Given** TLC has published month L, **When** the monthly job runs, **Then** it trains on L-1
   and evaluates on L, never the other way round.
2. **Given** the challenger wins, **When** the job ends, **Then** `serving_model/` holds the new
   model and encodings and the history marks it champion.
3. **Given** the challenger loses, **When** the job ends, **Then** only the histories change.

---

### User Story 3 - Performance is measured each month and a drop raises an alert (Priority: P3)

Tips arrive with the payment, so each published month labels the previous predictions. The job
records the champion's AUC on the newest month; if it falls below a floor, it opens an issue.

**Independent Test**: A champion whose AUC on the month is under the floor makes the job report
an alert; above the floor it does not.

**Acceptance Scenarios**:

1. **Given** a new month, **When** the job runs, **Then** `monitoring/performance_history.json`
   gains the champion's AUC, rows and high-tip rate for that month.
2. **Given** that AUC is below 0.60, **When** the job runs, **Then** it opens a GitHub issue.

---

### User Story 4 - The API requires a key (Priority: P4)

**Acceptance Scenarios**:

1. **Given** `API_KEY` is set, **When** `/predict` is called without `X-API-Key` or with a wrong
   one, **Then** it answers 401; with the right key, 200.
2. **Given** `API_KEY` is set, **When** `/health` is called, **Then** it answers without a key
   (probes stay simple).

### Edge Cases

- TLC has not published the expected month yet: the job looks back month by month.
- TLC files contain a few trips from other months: each month keeps only its own trips.
- A first run with no MLflow champion: the trained model is promoted.

## Requirements *(mandatory)*

- **FR-001**: Promotion MUST use a paired bootstrap of the AUC difference on one shared
  evaluation set and promote only when the 95% interval's lower bound is above zero.
- **FR-002**: Each decision MUST record both AUCs, the interval, the evaluation rows' count and
  fingerprint, and the months used.
- **FR-003**: `training/train_v2.py` MUST compare with the registry champion on a holdout file
  (default `features/holdout.parquet`), not on its own split.
- **FR-004**: `training/retrain.py` MUST implement the monthly loop on the committed bundle.
- **FR-005**: A monthly workflow MUST run the loop on real data and commit the results; the
  synthetic weekly retrain MUST be removed.
- **FR-006**: The job MUST append monthly champion performance and open an issue below the floor.
- **FR-007**: The API MUST require `X-API-Key` on prediction endpoints when `API_KEY` is set.

## Success Criteria *(mandatory)*

- **SC-001**: The first real run's decision is in the registry history with both AUCs and the
  interval.
- **SC-002**: Tests cover the promotion rule, the bootstrap, the loop on sample months, the
  alert and the API key.
- **SC-003**: The README's limits no longer list the four problems; it lists what remains.

## Assumptions

- The newest month is a fair stand-in for next month's traffic.
- The champion bundle in Git is the registry of record for the deployed system; local MLflow
  stays the registry for experiments.

## Clarifications

### Session 2026-11-27

- Q: Fixed 2024 holdout or newest month? → A: Newest month. Measured: a model trained on
  Aug 2026 scores 0.621 on Sep 2026 against the 2024 champion's 0.617, but 0.600 against
  0.617 on Jun 2024. A fixed 2024 set would block models for adapting to 2026 traffic. The
  set is fixed per decision and fingerprinted, which is what makes the comparison fair.
- Q: How big a gain? → A: Any gain whose 95% paired-bootstrap interval excludes zero.
- Q: Retrain only on drift, or every month? → A: Train a challenger every month (cheap) and let
  the rule decide; the performance floor raises an alert for a person.
