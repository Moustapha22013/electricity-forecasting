"""Diagnosis functions on small synthetic inputs with known answers."""
import numpy as np
import pandas as pd

from src.data import diagnosis as D
from src.data.loader import NUMERIC_COLUMNS


def _frame(n: int, nan_at: list[int] | None = None, start: str = "2007-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1min")
    df = pd.DataFrame({c: 1.0 for c in NUMERIC_COLUMNS}, index=range(n))
    df.insert(0, "timestamp", ts)
    for i in nan_at or []:
        df.loc[i, NUMERIC_COLUMNS] = np.nan
    return df


def test_timestamp_checks_detects_duplicates_and_gaps():
    df = _frame(10)
    df = pd.concat([df, df.iloc[[3]]]).sort_values("timestamp").reset_index(drop=True)   # one duplicate
    df = df[df["timestamp"] != df["timestamp"].iloc[6]].reset_index(drop=True)            # one absent minute
    r = D.timestamp_checks(df)
    assert r["duplicate_timestamps"] == 1
    assert r["absent_timestamps"] == 1


def test_missing_runs_lengths():
    df = _frame(20, nan_at=[2, 3, 4, 10, 19])
    runs = D.missing_runs(df)
    assert runs["length_min"].tolist() == [3, 1, 1]
    assert runs["start"].iloc[0] == df["timestamp"].iloc[2]


def test_hourly_energy_mean_based_is_unbiased_for_partial_hour():
    df = _frame(120, nan_at=list(range(30)))          # first hour: 30 valid minutes at 1 kW
    h = D.hourly_energy(df)
    assert h["n_valid"].tolist() == [30, 60]
    assert np.isclose(h["energy_kwh_mean_based"].iloc[0], 1.0)   # 1 kW for 1 h = 1 kWh
    assert np.isclose(h["energy_kwh_sum_based"].iloc[0], 0.5)    # sum-based is biased low


def test_seasonal_naive_perfect_on_periodic_series():
    idx = pd.date_range("2007-01-01", periods=24 * 7 * 6, freq="1h")
    y = pd.Series(np.tile(np.arange(24, dtype=float), len(idx) // 24), index=idx)
    r = D.seasonal_naive_scores(y + 0.0, train_end=str(idx[24 * 7 * 4]), dev_end=str(idx[-1]))
    assert r["naive_24"]["MAE_kwh"] == 0.0
