# 0001. Credit-card trips only; zone encodings fit on the training split

**Status:** Accepted

## Context
The TLC data records tips reliably only for card payments. Cash trips almost always show a tip of zero. If the model saw the payment type, or trained on cash trips, it would learn "cash means no tip" and score very well without learning anything about tipping behaviour. Zone ids are also strong predictors, and a naive target encoding computed on all rows would leak the validation labels.

## Decision
- `clean_trips` (`nyc_tip/features.py`) keeps only credit-card trips with a positive fare, and the payment type is not a feature.
- Each zone's high-tip rate is fit on the training split only, smoothed toward the overall rate with 50 pseudo-trips, and then applied to the validation split. Zones never seen in training get the overall rate.

## Alternatives considered
- **Keep all payment types.** A much higher AUC that measures nothing useful.
- **One-hot zones.** 265 sparse columns per end of the trip. Target encoding is compact and handles rare zones through smoothing.

## Consequences
- Validation AUC is about 0.62. Tipping is hard to predict from trip data alone, and the project reports that number rather than a leaky one.
- The model applies only to card payments, which is where tips are recorded.
