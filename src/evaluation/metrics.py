"""Point and interval metrics, plus a day-block bootstrap for uncertainty.

All functions ignore NaN targets: only OBSERVED hours are scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mask(y: pd.Series, *others: pd.Series) -> pd.Series:
    m = y.notna()
    for o in others:
        m &= o.notna()
    return m


def mae(y: pd.Series, p: pd.Series) -> float:
    m = _mask(y, p)
    return float((y[m] - p[m]).abs().mean())


def mase_scale(y_train: pd.Series, m: int = 24) -> float:
    """In-sample MAE of the seasonal naive forecast y(t-m) on TRAIN, observed pairs only.

    Hyndman & Koehler (2006). A fixed denominator computed on the past only, so it never
    depends on the evaluated period and is identical for every model.
    """
    d = (y_train - y_train.shift(m)).abs()
    return float(d.mean())


def mase(y: pd.Series, p: pd.Series, scale: float) -> float:
    return mae(y, p) / scale


def bias(y: pd.Series, p: pd.Series) -> float:
    """Mean of (y - p): > 0 means the model UNDER-predicts on average."""
    m = _mask(y, p)
    return float((y[m] - p[m]).mean())


def under_over(y: pd.Series, p: pd.Series) -> dict[str, float]:
    """Split the absolute error into under- and over-prediction (business asymmetry)."""
    m = _mask(y, p)
    e = y[m] - p[m]
    n = len(e)
    return {"under_share": float((e > 0).mean()), "over_share": float((e < 0).mean()),
            "under_mae_contrib": float(e.clip(lower=0).sum() / n), "over_mae_contrib": float((-e).clip(lower=0).sum() / n)}


def coverage(y: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    m = _mask(y, lo, hi)
    return float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean())


def mean_width(lo: pd.Series, hi: pd.Series) -> float:
    return float((hi - lo).mean())


def interval_score(y: pd.Series, lo: pd.Series, hi: pd.Series, alpha: float = 0.2) -> float:
    """Winkler / interval score (Gneiting & Raftery 2007): width + penalties for misses.

    Lower is better; rewards intervals that are both narrow and well calibrated.
    """
    m = _mask(y, lo, hi)
    y, lo, hi = y[m], lo[m], hi[m]
    s = (hi - lo) + (2 / alpha) * (lo - y).clip(lower=0) + (2 / alpha) * (y - hi).clip(lower=0)
    return float(s.mean())


def block_bootstrap_ci(err: pd.Series, n_boot: int = 2000, level: float = 0.95, seed: int = 0,
                       block: str = "7D") -> tuple[float, float]:
    """CI of the mean of `err` by resampling whole BLOCKS (weeks by default).

    Errors are autocorrelated within a day and across days (ρ(24), ρ(168)), so resampling
    individual hours would give falsely narrow intervals.
    """
    err = err.dropna()
    groups = [g.to_numpy() for _, g in err.groupby(err.index.floor("D") if block == "1D" else
                                                   (err.index - err.index.min()) // pd.Timedelta(block))]
    rng = np.random.default_rng(seed)
    k = len(groups)
    sums = np.array([g.sum() for g in groups]); counts = np.array([len(g) for g in groups])
    idx = rng.integers(0, k, size=(n_boot, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summary(y: pd.Series, p: pd.Series, scale: float, lo: pd.Series | None = None,
            hi: pd.Series | None = None, alpha: float = 0.2) -> dict[str, float]:
    m = _mask(y, p)
    abs_err = (y[m] - p[m]).abs()
    ci = block_bootstrap_ci(abs_err)
    out = {"MAE": mae(y, p), "MAE_ci_low": ci[0], "MAE_ci_high": ci[1], "MASE": mase(y, p, scale),
           "bias": bias(y, p), "n_hours": int(m.sum())}
    if lo is not None and hi is not None:
        out |= {"coverage_80": coverage(y, lo, hi), "width_80": mean_width(lo[m], hi[m]),
                "interval_score": interval_score(y, lo, hi, alpha)}
    return out
