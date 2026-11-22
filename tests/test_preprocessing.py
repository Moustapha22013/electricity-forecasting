"""Preprocessing: aggregation, validity rule, and CAUSALITY of every fill method."""
import numpy as np
import pandas as pd
import pytest

from src.data import preprocessing as P
from src.data.loader import NUMERIC_COLUMNS


def _minutes(n_hours: int, power: float = 1.0) -> pd.DataFrame:
    ts = pd.date_range("2007-01-01", periods=60 * n_hours, freq="1min")
    df = pd.DataFrame({c: power for c in NUMERIC_COLUMNS}, index=range(len(ts)))
    df.insert(0, "timestamp", ts)
    return df


def _hourly_series(n: int = 24 * 7 * 8, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-01-01", periods=n, freq="1h")
    base = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.Series(base + rng.normal(0, 0.1, n), index=idx)


def test_minute_grid_reindexes_absent_and_duplicate_rows():
    df = _minutes(1)
    df = pd.concat([df.drop(index=[5]), df.iloc[[10]]])          # one absent, one duplicate, unsorted
    m = P.to_minute_grid(df)
    assert len(m) == 60 and m.index.is_monotonic_increasing
    assert m[P.TARGET].isna().sum() == 1


def test_hourly_energy_and_submeter_rescaling():
    df = _minutes(2, power=2.0)                                  # 2 kW and 2 Wh/min
    df.loc[:29, NUMERIC_COLUMNS] = np.nan                        # first hour half missing
    h = P.aggregate_hourly(P.to_minute_grid(df))
    assert np.allclose(h["energy_kwh"], 2.0)                     # 2 kW over 1 h = 2 kWh, both hours
    assert np.allclose(h["Sub_metering_1"], 120.0)               # 2 Wh/min x 60 min, rescaled
    assert h["n_valid"].tolist() == [30, 60]


def test_validity_rule_sets_nan_but_keeps_row():
    df = _minutes(2); df.loc[:29, NUMERIC_COLUMNS] = np.nan
    h = P.apply_validity_rule(P.aggregate_hourly(P.to_minute_grid(df)), min_valid=50)
    assert len(h) == 2 and not h["observed"].iloc[0] and np.isnan(h["energy_kwh"].iloc[0])


@pytest.mark.parametrize("name", P.CAUSAL_METHODS)
def test_fill_methods_are_causal(name):
    """Changing values AFTER time t must not change any filled value up to t."""
    y = _hourly_series()
    y.iloc[600:650] = np.nan
    t = 620                                                      # inside the gap
    f1 = P.FILL_METHODS[name](y)
    y2 = y.copy(); y2.iloc[t + 1:] = y2.iloc[t + 1:] * 10 + 5    # alter the future (incl. after the gap)
    f2 = P.FILL_METHODS[name](y2)
    pd.testing.assert_series_equal(f1.iloc[:t + 1], f2.iloc[:t + 1])


def test_linear_interpolation_is_not_causal():
    """Documents WHY linear interpolation is excluded: it reads the value after the gap."""
    y = _hourly_series(); y.iloc[600:650] = np.nan
    y2 = y.copy(); y2.iloc[650:] += 10
    f1, f2 = P.fill_linear_offline(y), P.fill_linear_offline(y2)
    assert not np.allclose(f1.iloc[600:650], f2.iloc[600:650])


def test_causal_fill_switches_on_elapsed_time_and_never_touches_observed():
    y = _hourly_series(); y.iloc[700:710] = np.nan
    filled, method = P.causal_fill(y, short_max_h=2, short_method="ffill", long_method="profile_4w")
    assert method.iloc[700:702].eq("ffill").all() and method.iloc[702:710].eq("profile_4w").all()
    assert filled.iloc[700] == y.iloc[699]                       # ffill = last observed value
    obs = y.notna()
    pd.testing.assert_series_equal(filled[obs], y[obs])          # observed values untouched
    # causal: altering the future does not change the fill
    y2 = y.copy(); y2.iloc[710:] += 100
    f2, _ = P.causal_fill(y2, 2, "ffill", "profile_4w")
    pd.testing.assert_series_equal(filled.iloc[:710], f2.iloc[:710])
