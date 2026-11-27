"""Decide whether a challenger replaces the champion (ADR 0006).

Both models are scored on the same evaluation rows. The decision rests on the AUC difference
and its uncertainty: a paired bootstrap resamples the rows (the same resample for both models)
and the challenger is promoted only if the lower end of the 95% interval of
challenger AUC - champion AUC is above zero. A gain that could be noise keeps the champion.

The evaluation rows are identified by a fingerprint, recorded with the decision, so anyone can
check that two numbers in the history were measured on the same data.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_ROWS = 200_000  # rows per resample; AUC on more rows barely narrows the interval
SEED = 42


@dataclass(frozen=True)
class Decision:
    promote: bool
    challenger_auc: float
    champion_auc: float | None
    gain: float | None
    gain_ci_low: float | None
    gain_ci_high: float | None
    rows: int
    fingerprint: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def fingerprint(frame: pd.DataFrame) -> str:
    """First 16 hex digits of a SHA-256 over the rows' content (order-sensitive)."""
    hashed = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def bootstrap_gain(
    y: np.ndarray, p_challenger: np.ndarray, p_champion: np.ndarray,
    rounds: int = BOOTSTRAP_ROUNDS, rows: int = BOOTSTRAP_ROWS, seed: int = SEED,
) -> tuple[float, float]:  # fmt: skip
    """95% interval of AUC(challenger) - AUC(champion) over paired resamples of the rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    size = min(n, rows)
    gains = np.empty(rounds)
    for i in range(rounds):
        idx = rng.integers(0, n, size)
        yy = y[idx]
        if yy.min() == yy.max():  # a resample with one class has no AUC; redraw is not worth it
            gains[i] = np.nan
            continue
        gains[i] = roc_auc_score(yy, p_challenger[idx]) - roc_auc_score(yy, p_champion[idx])
    low, high = np.nanpercentile(gains, [2.5, 97.5])
    return float(low), float(high)


def decide(
    y: pd.Series | np.ndarray,
    p_challenger: np.ndarray,
    p_champion: np.ndarray | None,
    evaluation: pd.DataFrame | None = None,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> Decision:
    """Promote the challenger only if it beats the champion beyond the bootstrap noise."""
    y = np.asarray(y)
    fp = fingerprint(evaluation) if evaluation is not None else fingerprint(pd.DataFrame({"y": y}))
    challenger_auc = float(roc_auc_score(y, p_challenger))
    if p_champion is None:
        return Decision(True, challenger_auc, None, None, None, None, len(y), fp, "no champion yet")
    champion_auc = float(roc_auc_score(y, p_champion))
    low, high = bootstrap_gain(y, np.asarray(p_challenger), np.asarray(p_champion), rounds)
    gain = challenger_auc - champion_auc
    promote = low > 0
    reason = f"gain {gain:+.4f}, 95% interval [{low:+.4f}, {high:+.4f}] " + (
        "excludes zero: promoted" if promote else "includes zero or is negative: champion kept"
    )
    return Decision(promote, challenger_auc, champion_auc, gain, low, high, len(y), fp, reason)
