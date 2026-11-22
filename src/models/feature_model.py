"""Generic feature-based forecaster (used by Ridge and gradient boosting).

Leakage design
--------------
* `predict` rebuilds the features from `history` (data strictly before the origin) plus 24
  EMPTY rows for the target day. It cannot read a pre-computed feature matrix, so it goes
  through the same firewall as the baselines.
* `fit` builds features from `history` only and trains on OBSERVED targets with complete
  features.

Prediction interval (split-conformal by hour)
---------------------------------------------
The last `calib_days` of the history are held out: a copy of the model trained on the
earlier part predicts them, and the alpha/2 and 1 - alpha/2 quantiles of its errors are
stored PER HOUR OF THE DAY (the error spread depends strongly on the hour, phase 3). The
final model is then refitted on the whole history and the stored quantiles are added to its
point forecasts.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from ..features.engineering import build_features

ONE_H = pd.Timedelta("1h")


class FeatureForecaster:
    def __init__(self, name: str, estimator, features: list[str], alpha: float = 0.2,
                 calib_days: int = 91, lookback_h: int = 24 * 30, train_days: int | None = None):
        self.name, self.estimator, self.features = name, estimator, list(features)
        self.alpha, self.calib_days, self.lookback_h, self.train_days = alpha, calib_days, lookback_h, train_days
        self.model = None
        self.q_lo = self.q_hi = None

    # ------------------------------------------------------------------ #
    def _xy(self, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = build_features(history)
        ok = X["observed"].astype(bool) & X[self.features].notna().all(axis=1)
        if self.train_days is not None:
            ok &= X.index > history.index[-1] - pd.Timedelta(days=self.train_days)
        return X.loc[ok, self.features], X.loc[ok, "y"]

    def fit(self, history: pd.DataFrame) -> "FeatureForecaster":
        X, y = self._xy(history)
        if self.calib_days:
            cut = history.index[-1] - pd.Timedelta(days=self.calib_days)
            m_cal = clone(self.estimator).fit(X[X.index <= cut], y[y.index <= cut])
            Xc, yc = X[X.index > cut], y[y.index > cut]
            e = yc - np.maximum(m_cal.predict(Xc), 0.0)
            g = e.groupby(e.index.hour)
            self.q_lo, self.q_hi = g.quantile(self.alpha / 2), g.quantile(1 - self.alpha / 2)
        self.model = clone(self.estimator).fit(X, y)
        return self

    def features_for_day(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
        """Features of the target day computed from the history ONLY (+ empty target rows)."""
        if history.index[-1] != origin - ONE_H:
            raise ValueError("history must end exactly one hour before the origin")
        targets = pd.date_range(origin, periods=horizon, freq="h")
        empty = pd.DataFrame({"energy_kwh": np.nan, "energy_filled": np.nan, "observed": False}, index=targets)
        frame = pd.concat([history[["energy_kwh", "energy_filled", "observed"]].iloc[-self.lookback_h:], empty])
        return build_features(frame).loc[targets, self.features]

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int = 24) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        X = self.features_for_day(history, origin, horizon)
        pred = np.maximum(self.model.predict(X), 0.0)          # energy cannot be negative
        out = pd.DataFrame({"pred": pred}, index=X.index)
        if self.q_lo is not None:
            h = X.index.hour
            out["lo"] = np.maximum(pred + self.q_lo.reindex(h).to_numpy(), 0.0)
            out["hi"] = pred + self.q_hi.reindex(h).to_numpy()
        else:
            out["lo"] = out["hi"] = np.nan
        return out


Factory = Callable[[list[str]], FeatureForecaster]
