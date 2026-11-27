# 0006. Promotion compares both models on the same newest month, beyond the bootstrap noise

**Status:** Accepted. Supersedes the promotion rule in [0002](0002-one-registered-model-strict-promotion.md) ("a new version must beat the champion's AUC"); only train_v2.py still registers.

## Context
A new version was promoted when its validation AUC beat the champion's recorded validation AUC.
The two numbers came from different validation splits of different data, so a lucky split could
promote a worse model, and any gain, however small, counted.

## Decision
- Both models are scored on the same rows that neither trained on. In the monthly loop that is
  the newest month TLC has published; in `train_v2.py` it is a holdout file
  (`features/holdout.parquet`).
- `nyc_tip/promotion.py` resamples those rows 200 times (the same resample for both models) and
  takes the 95% interval of challenger AUC minus champion AUC. The challenger is promoted only
  if the interval's lower end is above zero.
- The decision records both AUCs, the interval, the row count and a fingerprint of the rows, so
  two numbers in the history can be checked to come from the same data.

## Why the newest month and not a fixed 2024 set
Measured before deciding: a model trained on August 2026 scores 0.621 on September 2026 against
0.617 for the 2024 champion, but 0.600 against 0.617 on June 2024. A fixed 2024 set would block
models for adapting to current traffic. Fixing the set per decision, and fingerprinting it, is
what makes the comparison fair.

## Consequences
- The first monthly run on real data (evaluated on September 2026) promoted v3: AUC 0.6251
  against 0.6153 for v2, gain interval [+0.0082, +0.0112] on 500,000 trips.
- A gain inside the noise keeps the champion; tests cover better, equal, noisy and worse
  challengers.
