"""Shared statistics helpers for the p-bet counterfactual analyses.

Extracted verbatim from the former ``analysis.decode.counterfactual_policy_lr``
module so that the current actor-direct analysis
(:mod:`analysis.counterfactoral_lr.counterfactual_actor_pbet_lr`) no longer depends on the
archived decoder-based lineage. Values and behaviour are unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


WINDOWS = {
    "trial_1_5": (1, 5),
    "trial_1_20": (1, 20),
    "trial_21_50": (21, 50),
    "trial_51_89": (51, 89),
    "trial_90_119": (90, 119),
    "trial_51_119": (51, 119),
    "full_1_119": (1, 119),
}


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return float("nan")
    return float(stats.sem(values))


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return (float("nan"), float("nan"))
    se = stats.sem(values)
    q = stats.t.ppf(0.975, values.size - 1)
    mean = float(values.mean())
    return (float(mean - q * se), float(mean + q * se))


def one_sample(values: np.ndarray, mu: float = 0.0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {
            "n": int(values.size),
            "mean": float(np.nanmean(values)) if values.size else np.nan,
            "sem": np.nan,
            "t": np.nan,
            "p": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }
    t, p = stats.ttest_1samp(values, popmean=mu, nan_policy="omit")
    lo, hi = ci95(values)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sem": sem(values),
        "t": float(t),
        "p": float(p),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def fdr_bh(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q.tolist()
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.minimum(adjusted, 1.0)
    return q.tolist()


def r2_through_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 3 or float(x @ x) <= 0.0:
        return np.nan, np.nan
    slope = float((x @ y) / (x @ x))
    resid = y - slope * x
    sse = float(resid @ resid)
    sst0 = float(y @ y)
    return slope, float(1.0 - sse / sst0) if sst0 > 0 else np.nan
