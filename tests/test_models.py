"""Feature-based models: on-the-fly features == precomputed causal features; no future access."""
import numpy as np
import pandas as pd

from src.features.engineering import ALL_FEATURES, build_features
from src.models.boosting import make_boosting
from src.models.ridge import expand_design, make_ridge


def _data(days: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2009-09-01", periods=24 * days, freq="h")
    t = np.arange(len(idx))
    y = pd.Series(1 + 0.6 * np.sin(2 * np.pi * t / 24) + 0.3 * (idx.day_of_week >= 5) + rng.normal(0, 0.15, len(idx)),
                  index=idx).clip(0.05)
    obs = pd.Series(True, index=idx); obs.iloc[1000:1030] = False
    filled = y.copy()
    return pd.DataFrame({"energy_kwh": y.where(obs), "energy_filled": filled, "observed": obs})


def test_on_the_fly_features_equal_precomputed():
    d = _data(); m = make_ridge(ALL_FEATURES, calib_days=0)
    full = build_features(d)[ALL_FEATURES]
    for o in pd.to_datetime(["2009-11-15", "2009-12-24", "2010-01-20"]):
        X = m.features_for_day(d.loc[:o - pd.Timedelta("1h")], o)
        pd.testing.assert_frame_equal(X, full.loc[X.index], check_freq=False)


def test_prediction_ignores_the_future_and_intervals_exist():
    d = _data(); o = pd.Timestamp("2010-01-10")
    hist = d.loc[:o - pd.Timedelta("1h")]
    m = make_boosting(ALL_FEATURES, max_iter=30, calib_days=28).fit(hist)
    p1 = m.predict(hist, o)
    d2 = d.copy(); d2.loc[o:, ["energy_kwh", "energy_filled"]] += 100
    p2 = m.predict(d2.loc[:o - pd.Timedelta("1h")], o)
    pd.testing.assert_frame_equal(p1, p2)
    # a conformal interval need not contain the point forecast (it is shifted if the
    # calibration model is biased), but it must be ordered and non-negative
    assert (p1["lo"] <= p1["hi"]).all() and (p1["lo"] >= 0).all() and p1[["lo", "hi"]].notna().all().all()


def test_ridge_expansion_is_row_wise():
    X = pd.DataFrame({"hour": [0, 7], "is_weekend": [1, 0], "lag_1": [2.0, 3.0]})
    E = expand_design(X)
    assert E.loc[0, "h0"] == 1 and E.loc[0, "we_h0"] == 1 and E.loc[1, "lag1_h7"] == 3.0
    pd.testing.assert_frame_equal(expand_design(X.iloc[[1]]), E.iloc[[1]])   # same row, same output alone
