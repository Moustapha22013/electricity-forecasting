"""SARIMA(p,d,q)(P,D,Q)_24 baseline via statsmodels' state-space SARIMAX.

* The Kalman filter handles missing observations natively: SARIMA is fed the OBSERVED
  target (with NaN) and needs no imputation at all.
* `fit` estimates parameters by maximum likelihood on the last `fit_window_h` hours.
* `predict` keeps those parameters and re-filters the last `filter_window_h` hours up to the
  origin (`results.apply`), i.e. the state is updated with new data but parameters are not.
* transform="log": the model is fitted on log(y). exp(forecast mean of log y) is the
  conditional MEDIAN of y under log-normal errors — the MAE-optimal point forecast — and the
  back-transformed interval is exact and always positive.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ONE_H = pd.Timedelta("1h")


class Sarima:
    def __init__(self, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int],
                 transform: str = "none", fit_window_h: int = 24 * 7 * 8, filter_window_h: int = 24 * 7 * 4,
                 alpha: float = 0.2, maxiter: int = 100):
        if transform not in ("none", "log"):
            raise ValueError("transform must be 'none' or 'log'")
        self.order, self.seasonal_order, self.transform = tuple(order), tuple(seasonal_order), transform
        self.fit_window_h, self.filter_window_h, self.alpha, self.maxiter = fit_window_h, filter_window_h, alpha, maxiter
        self.name = f"sarima{self.order}{self.seasonal_order[:3]}_{self.seasonal_order[3]}_{transform}"
        self.res = None

    # ------------------------------------------------------------------ #
    def _endog(self, history: pd.DataFrame, n: int) -> pd.Series:
        y = history["energy_kwh"].iloc[-n:].asfreq("h")
        return np.log(y) if self.transform == "log" else y

    def _trend(self) -> str:
        d, D = self.order[1], self.seasonal_order[1]
        return "c" if d == 0 and D == 0 else "n"

    def fit(self, history: pd.DataFrame) -> "Sarima":
        y = self._endog(history, self.fit_window_h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order, trend=self._trend()
                               ).fit(disp=False, maxiter=self.maxiter)
        return self

    @property
    def aic(self) -> float:
        return float(self.res.aic)

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
        if self.res is None:
            raise RuntimeError("fit() must be called before predict()")
        y = self._endog(history, self.filter_window_h)
        if y.index[-1] != origin - ONE_H:
            raise ValueError("history must end exactly one hour before the origin")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = self.res.apply(y)
            f = r.get_forecast(horizon).summary_frame(alpha=self.alpha)
        mean, lo, hi = f["mean"].to_numpy(), f["mean_ci_lower"].to_numpy(), f["mean_ci_upper"].to_numpy()
        if self.transform == "log":
            mean, lo, hi = np.exp(mean), np.exp(lo), np.exp(hi)
        else:
            lo = np.maximum(lo, 0.0)
        targets = pd.date_range(origin, periods=horizon, freq="h")
        return pd.DataFrame({"pred": mean, "lo": lo, "hi": hi}, index=targets)

    def residual_diagnostics(self, lags: tuple[int, ...] = (24, 48, 168)) -> dict[str, float]:
        """Ljung-Box p-values on standardized in-sample residuals (H0: no autocorrelation left)."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        r = pd.Series(self.res.standardized_forecasts_error[0]).dropna()
        lb = acorr_ljungbox(r, lags=list(lags))
        return {f"ljungbox_p_lag{l}": float(p) for l, p in zip(lags, lb["lb_pvalue"])}
