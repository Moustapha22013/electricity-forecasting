"""Reference forecasters: seasonal naive (m = 24, 168) and the 4-week seasonal profile.

Point forecasts use `energy_filled` (causal fill, phase 2) so that a lag falling in a gap
still has a value. Prediction intervals are EMPIRICAL: quantiles of the method's own past
errors, per hour of the day (= per horizon, since the origin is 00:00), computed at fit time
on the history only (a split-conformal style interval).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ONE_H = pd.Timedelta("1h")


class _EmpiricalIntervalMixin:
    """Learn residual quantiles by hour of day from the last `calib_days` of history."""

    alpha: float
    calib_days: int
    q_lo: pd.Series
    q_hi: pd.Series

    def _point_from_series(self, s: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def _fit_intervals(self, history: pd.DataFrame) -> None:
        y = history["energy_kwh"]
        s = history["energy_filled"]
        start = history.index[-1] - pd.Timedelta(days=self.calib_days)
        t = y.loc[start:].index
        # in-sample errors of the method over the calibration window (point uses only past values)
        p = pd.Series(self._point_from_series(s, t), index=t)
        e = (y.loc[t] - p).dropna()
        g = e.groupby(e.index.hour)
        self.q_lo = g.quantile(self.alpha / 2)
        self.q_hi = g.quantile(1 - self.alpha / 2)

    def _with_interval(self, targets: pd.DatetimeIndex, pred: np.ndarray) -> pd.DataFrame:
        hrs = targets.hour
        lo = pred + self.q_lo.reindex(hrs).to_numpy()
        hi = pred + self.q_hi.reindex(hrs).to_numpy()
        return pd.DataFrame({"pred": pred, "lo": np.maximum(lo, 0.0), "hi": hi}, index=targets)


class SeasonalNaive(_EmpiricalIntervalMixin):
    """y_hat(t) = y(t - m). With origin at 00:00 and h <= 24, t - m <= origin - 1 for m >= 24."""

    def __init__(self, m: int = 24, alpha: float = 0.2, calib_days: int = 365):
        if m < 24:
            raise ValueError("m < 24 would use values not yet known at a 24 h horizon (leakage).")
        self.m, self.alpha, self.calib_days = m, alpha, calib_days
        self.name = f"naive_{m}"

    def _point_from_series(self, s: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
        return s.reindex(targets - self.m * ONE_H).to_numpy()

    def fit(self, history: pd.DataFrame) -> "SeasonalNaive":
        self._fit_intervals(history)
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
        targets = pd.date_range(origin, periods=horizon, freq="h")
        return self._with_interval(targets, self._point_from_series(history["energy_filled"], targets))


class SeasonalProfile(_EmpiricalIntervalMixin):
    """y_hat(t) = mean_k y(t - 168 k), k = 1..weeks: same hour, same weekday, last `weeks` weeks.

    Averaging k noisy copies divides the noise variance by ~k while keeping the weekly shape,
    at the cost of reacting more slowly to regime changes (holidays).
    """

    def __init__(self, weeks: int = 4, alpha: float = 0.2, calib_days: int = 365, stat: str = "mean"):
        self.weeks, self.alpha, self.calib_days, self.stat = weeks, alpha, calib_days, stat
        self.name = f"profile_{weeks}w_{stat}"

    def _point_from_series(self, s: pd.Series, targets: pd.DatetimeIndex) -> np.ndarray:
        cols = np.column_stack([s.reindex(targets - 168 * k * ONE_H).to_numpy() for k in range(1, self.weeks + 1)])
        return np.nanmedian(cols, axis=1) if self.stat == "median" else np.nanmean(cols, axis=1)

    def fit(self, history: pd.DataFrame) -> "SeasonalProfile":
        self._fit_intervals(history)
        return self

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
        targets = pd.date_range(origin, periods=horizon, freq="h")
        return self._with_interval(targets, self._point_from_series(history["energy_filled"], targets))
