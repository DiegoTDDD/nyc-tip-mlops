# nyc-tip-mlops Constitution

## Core Principles

### I. No leakage, ever
Features are known when the ride ends, before payment. Encodings are fit on training rows only.
An evaluation set is never seen by the model being evaluated (ADR 0001).

### II. Comparable or not compared
Two models are compared only on the same rows, identified by a fingerprint, and a difference
counts only when it is larger than its uncertainty.

### III. One feature implementation
Training, serving, monitoring and the dashboard use `nyc_tip/features.py` (ADR 0003).

### IV. The loop closes in CI
Retraining, evaluation, promotion and monitoring run on real data on a schedule, and their
decisions are committed as data the dashboard reads.

### V. Honest limits
The README states what the model cannot do and what the automation does not cover.

## Constraints
Free: GitHub Actions, local MLflow, Streamlit Community Cloud. Python 3.11.

## Development Workflow
Spec Kit (constitution, specify, clarify, plan, tasks, implement) in `specs/`. Ruff and pytest
gate every push.

## Governance
A principle changes only with an ADR.

**Version**: 1.0.0 | **Ratified**: 2026-11-27 | **Last Amended**: 2026-11-27
