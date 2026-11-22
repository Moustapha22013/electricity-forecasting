"""Phase 4 — strictly causal features for day-ahead hourly forecasting.

Setting: origin o = 00:00 of day D, target t = o + (h - 1) hours, h = 1..24.
A feature for target t may only use observations at times <= o - 1h.

Two families of lags:
* ORIGIN-relative ("what was the house doing when the forecast was issued"):
  lag_1 = y(o - 1)            (last known hour, 23:00 of D-1)
  roll_mean_24 / roll_std_24 = mean / std of y over [o - 24, o - 1]
  roll_mean_168              = mean of y over [o - 168, o - 1]
* TARGET-relative but always >= 24 h old (safe because h <= 24  =>  t - 24 <= o - 1):
  lag_24, lag_48, lag_168, lag_336, same_hour_mean_7d, profile_4w_median

Lags use `energy_filled` (phase 2 causal fill). `frac_filled_168` tells the model how much
of the recent history was imputed. Calendar features are known in advance by definition.

Note on `lag_1`: the course text asks for lag_1; read as y(t - 1) it is unknown for h >= 2
(leakage, demonstrated in TP2). It is therefore defined relative to the ORIGIN. Its
usefulness decays with h, which the model learns through `hour` (= h - 1 here).
"""
from __future__ import annotations

import warnings

import holidays
import numpy as np
import pandas as pd

ONE_H = pd.Timedelta("1h")

FEATURE_GROUPS: dict[str, list[str]] = {
    "lags": ["lag_1", "lag_24", "lag_48", "lag_168", "lag_336"],
    "rolling": ["roll_mean_24", "roll_std_24", "roll_mean_168", "same_hour_mean_7d", "profile_4w_median",
                "frac_filled_168"],
    "calendar": ["hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "doy_sin", "doy_cos"],
    "holidays": ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "lag_24_was_holiday",
                 "lag_168_was_holiday", "is_dec24_or_dec31"],
}
ALL_FEATURES: list[str] = [f for g in FEATURE_GROUPS.values() for f in g]


def _origin(idx: pd.DatetimeIndex, issue_hour: int = 0) -> pd.DatetimeIndex:
    """Origin of the forecast that covers each target (00:00 of the target's day by default)."""
    if issue_hour != 0:
        raise NotImplementedError("only origin = 00:00 is implemented (config forecast.issue_hour = 0)")
    return idx.floor("D")


def _french_holidays(idx: pd.DatetimeIndex) -> set[pd.Timestamp]:
    years = range(idx.min().year - 1, idx.max().year + 2)
    return {pd.Timestamp(d) for d in holidays.France(years=years)}


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic calendar: known for any future date, hence leakage-free."""
    X = pd.DataFrame(index=idx)
    hour, dow, doy = idx.hour.to_numpy(), idx.day_of_week.to_numpy(), idx.day_of_year.to_numpy()
    X["hour"] = hour
    X["dayofweek"] = dow
    X["is_weekend"] = (dow >= 5).astype(int)
    # cyclic encodings: 23 h and 0 h are neighbours, Sunday and Monday too, 31 Dec and 1 Jan too
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    hol = _french_holidays(idx)
    day = idx.normalize()
    is_hol = day.isin(list(hol))
    X["is_holiday"] = is_hol.astype(int)
    X["is_day_before_holiday"] = (day + pd.Timedelta("1D")).isin(list(hol)).astype(int)
    X["is_day_after_holiday"] = (day - pd.Timedelta("1D")).isin(list(hol)).astype(int)
    X["lag_24_was_holiday"] = X["is_day_after_holiday"]          # y(t-24) comes from a holiday
    X["lag_168_was_holiday"] = (day - pd.Timedelta("7D")).isin(list(hol)).astype(int)
    X["is_dec24_or_dec31"] = (((idx.month == 12) & np.isin(idx.day, [24, 31]))).astype(int)
    return X


def lag_features(data: pd.DataFrame, issue_hour: int = 0) -> pd.DataFrame:
    """Lags and windows computed from `energy_filled` (and `observed`), all known at the origin."""
    s = data["energy_filled"]
    idx = data.index
    o = _origin(idx, issue_hour)
    at_origin = o - ONE_H                                         # last known timestamp

    X = pd.DataFrame(index=idx)
    X["lag_1"] = s.reindex(at_origin).to_numpy()
    for k in (24, 48, 168, 336):
        X[f"lag_{k}"] = s.shift(k).to_numpy()                    # t - k <= o - 1 for k >= 24

    r24m, r24s, r168m = s.rolling(24).mean(), s.rolling(24).std(), s.rolling(168).mean()
    X["roll_mean_24"] = r24m.reindex(at_origin).to_numpy()       # window [o-24, o-1]
    X["roll_std_24"] = r24s.reindex(at_origin).to_numpy()
    X["roll_mean_168"] = r168m.reindex(at_origin).to_numpy()
    with warnings.catch_warnings():                              # all-NaN rows only at the very start
        warnings.simplefilter("ignore", RuntimeWarning)
        X["same_hour_mean_7d"] = np.nanmean(np.column_stack([s.shift(24 * k).to_numpy() for k in range(1, 8)]), axis=1)
        X["profile_4w_median"] = np.nanmedian(np.column_stack([s.shift(168 * k).to_numpy() for k in range(1, 5)]), axis=1)
    filled = (~data["observed"]).astype(float)
    X["frac_filled_168"] = filled.rolling(168).mean().reindex(at_origin).to_numpy()
    return X


def build_features(data: pd.DataFrame, issue_hour: int = 0) -> pd.DataFrame:
    """Full design matrix (one row per hour) + target columns `y` and `observed`, `origin`, `horizon`."""
    X = lag_features(data, issue_hour).join(calendar_features(data.index))
    X = X[ALL_FEATURES]
    o = _origin(data.index, issue_hour)
    X["origin"] = o
    X["horizon"] = ((data.index - o) / ONE_H).astype(int) + 1
    X["y"] = data["energy_kwh"]
    X["observed"] = data["observed"]
    return X


def leaky_lag_1(data: pd.DataFrame) -> pd.Series:
    """FOR DEMONSTRATION ONLY: y(t-1) relative to the TARGET — not known at the origin for h >= 2."""
    return data["energy_filled"].shift(1)
