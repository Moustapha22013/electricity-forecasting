"""Walk-forward (rolling-origin) evaluation — the leakage firewall of the project.

Protocol
--------
* Forecast ORIGIN o = 00:00 of day D; targets = the 24 hours of day D (horizon h = 1..24).
* At origin o the model only receives `history = data[: o - 1h]` (sliced HERE, not in the
  model), so no model can read the future even by mistake.
* Parameters are re-estimated at the start of each FOLD (block of days) on data ending
  `gap` hours before the fold start; within a fold, models update their STATE with new
  observations (as a deployed system would) but not their parameters.
* Only OBSERVED target hours are scored, on the same points for every model.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> "Forecaster": ...

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Return a DataFrame indexed by the `horizon` target timestamps with columns pred, lo, hi."""
        ...


def daily_origins(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """All 00:00 origins whose 24 target hours fall inside [start, end]."""
    s = pd.Timestamp(start).ceil("D")
    e = pd.Timestamp(end).floor("D")
    if pd.Timestamp(end) < e + pd.Timedelta("23h"):
        e -= pd.Timedelta("1D")
    return pd.date_range(s, e, freq="D")


def split_origins(train_end: str | pd.Timestamp, dev_end: str | pd.Timestamp, inner_days: int
                  ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Origins of the inner validation (last `inner_days` days of train) and of the dev period.

    Split ends are given as 'YYYY-MM-DD 23:59:59'; the next period starts 1 SECOND later
    (00:00:00), never one hour later (which would skip the first day).
    """
    train_end, dev_end = pd.Timestamp(train_end), pd.Timestamp(dev_end)
    one_s = pd.Timedelta("1s")
    inner = daily_origins(train_end - pd.Timedelta(days=inner_days) + one_s, train_end)
    dev = daily_origins(train_end + one_s, dev_end)
    return inner, dev


def make_folds(origins: pd.DatetimeIndex, fold_days: int) -> list[pd.DatetimeIndex]:
    return [origins[i:i + fold_days] for i in range(0, len(origins), fold_days)]


def walk_forward(model: Forecaster, data: pd.DataFrame, origins: pd.DatetimeIndex, fold_days: int,
                 gap_hours: int = 24, horizon: int = 24) -> pd.DataFrame:
    """Run the protocol; returns one row per (origin, target hour)."""
    rows = []
    one_h = pd.Timedelta("1h")
    for k, fold in enumerate(make_folds(origins, fold_days)):
        fit_end = fold[0] - gap_hours * one_h - one_h       # last timestamp usable for parameters
        model.fit(data.loc[:fit_end])
        for o in fold:
            hist = data.loc[:o - one_h]                      # strictly before the origin
            fc = model.predict(hist, o, horizon)
            expected = pd.date_range(o, periods=horizon, freq="h")
            if not fc.index.equals(expected):
                raise ValueError(f"{model.name}: forecast index mismatch at origin {o}")
            fc = fc.assign(origin=o, horizon=np.arange(1, horizon + 1), fold=k)
            rows.append(fc)
    out = pd.concat(rows)
    out.index.name = "target"
    return out


def attach_truth(fc: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Add the observed target (NaN where the hour is not observed)."""
    return fc.assign(y=y.reindex(fc.index).to_numpy())
