"""Feature causality: features of day D must not change when anything from D 00:00 onwards changes."""
import numpy as np
import pandas as pd
import pytest

from src.features import engineering as F


def _data(days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2009-11-01", periods=24 * days, freq="h")
    y = pd.Series(1 + np.sin(2 * np.pi * np.arange(len(idx)) / 24) + rng.normal(0, 0.2, len(idx)), index=idx).clip(0.05)
    obs = pd.Series(True, index=idx)
    obs.iloc[500:520] = False
    return pd.DataFrame({"energy_kwh": y.where(obs), "energy_filled": y, "observed": obs})


def _perturb_from(d: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    d2 = d.copy()
    m = d2.index >= t0
    d2.loc[m, "energy_filled"] = d2.loc[m, "energy_filled"] * 7 + 3
    d2.loc[m, "energy_kwh"] = d2.loc[m, "energy_filled"]
    d2.loc[m, "observed"] = ~d2.loc[m, "observed"]
    return d2


@pytest.mark.parametrize("day", ["2009-11-20", "2009-12-01", "2009-12-24"])
def test_features_of_day_D_only_use_data_before_D(day):
    d = _data(); D = pd.Timestamp(day)
    X1 = F.build_features(d)[F.ALL_FEATURES]
    X2 = F.build_features(_perturb_from(d, D))[F.ALL_FEATURES]
    rows = (X1.index >= D) & (X1.index < D + pd.Timedelta("1D"))
    pd.testing.assert_frame_equal(X1[rows], X2[rows])


def test_leaky_lag_1_fails_the_same_test():
    """Documents the trap: y(t-1) relative to the target changes when the target day changes."""
    d = _data(); D = pd.Timestamp("2009-12-01")
    a, b = F.leaky_lag_1(d), F.leaky_lag_1(_perturb_from(d, D))
    rows = (a.index >= D) & (a.index < D + pd.Timedelta("1D"))
    assert not np.allclose(a[rows], b[rows])


def test_lag_1_is_constant_within_a_day_and_equals_last_hour():
    d = _data(); X = F.build_features(d)
    day = X.loc["2009-12-02"]
    assert day["lag_1"].nunique() == 1
    assert day["lag_1"].iloc[0] == d.loc["2009-12-01 23:00", "energy_filled"]
    assert day["horizon"].tolist() == list(range(1, 25))


def test_rolling_window_is_the_24h_before_origin():
    d = _data(); X = F.build_features(d)
    expected = d.loc["2009-12-01 00:00":"2009-12-01 23:00", "energy_filled"].mean()
    assert X.loc["2009-12-02 13:00", "roll_mean_24"] == pytest.approx(expected)


def test_french_holiday_flags():
    idx = pd.date_range("2009-12-23", "2009-12-27 23:00", freq="h")
    C = F.calendar_features(idx)
    assert C.loc["2009-12-25 10:00", "is_holiday"] == 1
    assert C.loc["2009-12-24 10:00", "is_day_before_holiday"] == 1 and C.loc["2009-12-24 10:00", "is_dec24_or_dec31"] == 1
    assert C.loc["2009-12-26 10:00", "lag_24_was_holiday"] == 1
    assert C.loc["2009-12-23 10:00", "is_holiday"] == 0
