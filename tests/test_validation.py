"""Leakage firewall: the walk-forward runner must never give a model data at/after the origin."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as M
from src.evaluation.validation import daily_origins, walk_forward
from src.models.naive import SeasonalNaive, SeasonalProfile


def _data(days: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2008-01-01", periods=24 * days, freq="h")
    y = pd.Series(1 + np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)
    return pd.DataFrame({"energy_kwh": y, "energy_filled": y})


class Spy:
    """Records the last timestamp it is allowed to see."""
    name = "spy"

    def __init__(self):
        self.fit_ends, self.pred_ends = [], []

    def fit(self, history):
        self.fit_ends.append(history.index[-1]); return self

    def predict(self, history, origin, horizon):
        self.pred_ends.append((history.index[-1], origin))
        idx = pd.date_range(origin, periods=horizon, freq="h")
        return pd.DataFrame({"pred": 0.0, "lo": 0.0, "hi": 0.0}, index=idx)


def test_runner_never_leaks_and_respects_gap():
    d = _data(); spy = Spy()
    origins = daily_origins("2008-03-01", "2008-03-31 23:00")
    walk_forward(spy, d, origins, fold_days=10, gap_hours=24)
    assert all(end == o - pd.Timedelta("1h") for end, o in spy.pred_ends)
    fold_starts = origins[::10]
    assert all(fe == fs - pd.Timedelta("25h") for fe, fs in zip(spy.fit_ends, fold_starts))


def test_daily_origins_only_full_days():
    o = daily_origins("2008-01-01 05:00", "2008-01-05 12:00")
    assert o[0] == pd.Timestamp("2008-01-02") and o[-1] == pd.Timestamp("2008-01-04")


def test_naive_rejects_short_period():
    with pytest.raises(ValueError):
        SeasonalNaive(m=1)


def test_naive24_perfect_on_periodic_series_and_profile_too():
    d = _data(); origins = daily_origins("2008-03-01", "2008-03-10 23:00")
    for m in (SeasonalNaive(24, calib_days=30), SeasonalProfile(4, calib_days=30)):
        fc = walk_forward(m, d, origins, fold_days=5)
        assert np.allclose(fc["pred"], d["energy_kwh"].reindex(fc.index))


def test_naive_ignores_future_changes():
    """Predictions must be identical if everything from the origin onwards is changed."""
    d = _data(); o = pd.Timestamp("2008-03-15")
    m = SeasonalNaive(168, calib_days=30).fit(d.loc[:o - pd.Timedelta("1h")])
    p1 = m.predict(d.loc[:o - pd.Timedelta("1h")], o, 24)
    d2 = d.copy(); d2.loc[o:] += 50
    p2 = m.predict(d2.loc[:o - pd.Timedelta("1h")], o, 24)
    pd.testing.assert_frame_equal(p1, p2)


def test_metrics_basic():
    idx = pd.date_range("2008-01-01", periods=4, freq="h")
    y = pd.Series([1.0, 2.0, np.nan, 4.0], index=idx); p = pd.Series([1.0, 1.0, 9.0, 5.0], index=idx)
    assert M.mae(y, p) == pytest.approx(2 / 3)                      # NaN target ignored
    assert M.bias(y, p) == pytest.approx(0.0)
    assert M.coverage(y, p - 0.5, p + 0.5) == pytest.approx(1 / 3)


def test_split_origins_do_not_skip_first_day():
    """Regression: train_end '23:59:59' + 1 h used to skip the first dev day."""
    from src.evaluation.validation import split_origins
    inner, dev = split_origins("2009-11-30 23:59:59", "2010-05-31 23:59:59", inner_days=91)
    assert inner[0] == pd.Timestamp("2009-09-01") and len(inner) == 91
    assert dev[0] == pd.Timestamp("2009-12-01") and dev[-1] == pd.Timestamp("2010-05-31") and len(dev) == 182


def test_config_refuses_unimplemented_values():
    from src.config import validate
    good = {"target": "Global_active_power", "forecast": {"horizon": 24, "issue_hour": 0}}
    validate(good)
    for bad in ({**good, "target": "Voltage"}, {**good, "forecast": {"horizon": 48, "issue_hour": 0}},
                {**good, "forecast": {"horizon": 24, "issue_hour": 12}}):
        with pytest.raises(NotImplementedError):
            validate(bad)
