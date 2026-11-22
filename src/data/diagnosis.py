"""Phase 1 — data diagnosis of the raw minute-level file.

Every function takes the raw DataFrame returned by `loader.load_raw` and returns
plain Python / pandas objects, so that results can be unit-tested, written to
JSON, and quoted in the report without copy-paste errors.

Rule followed here: data-QUALITY checks (format, gaps, duplicates, physics) are
run on the whole file, because they do not inform the model. Checks that DRIVE
modelling choices (seasonality, autocorrelation, stationarity, baseline) use
train + dev only: the test period stays sealed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loader import NUMERIC_COLUMNS

TARGET = "Global_active_power"


# --------------------------------------------------------------------------- #
# 1. Timestamp consistency
# --------------------------------------------------------------------------- #
def timestamp_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Order, duplicates, effective sampling step, absent timestamps."""
    ts = df["timestamp"]
    diffs = ts.diff().dropna()
    start, end = ts.min(), ts.max()
    expected = int((end - start) / pd.Timedelta("1min")) + 1
    step_counts = diffs.value_counts()
    return {
        "n_rows": int(len(df)),
        "start": str(start),
        "end": str(end),
        "span_days": round((end - start) / pd.Timedelta("1D"), 2),
        "expected_minutes": expected,
        "absent_timestamps": expected - int(ts.nunique()),
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "step_counts": {str(k): int(v) for k, v in step_counts.items()},
        "seconds_not_zero": int((ts.dt.second != 0).sum()),
    }


def dst_check(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Look at the European DST switch nights (last Sunday of March / October).

    If timestamps were local wall-clock time with DST, the 02:00-02:59 hour would be
    absent in March and duplicated in October.
    """
    ts = pd.DatetimeIndex(df["timestamp"])
    out = []
    for year in sorted(set(ts.year)):
        for month in (3, 10):
            last_day = pd.Timestamp(year=year, month=month, day=31)
            sunday = last_day - pd.Timedelta(days=(last_day.day_of_week + 1) % 7)
            win = (ts >= sunday + pd.Timedelta("2h")) & (ts < sunday + pd.Timedelta("3h"))
            if ts.min() <= sunday <= ts.max():
                out.append({"date": str(sunday.date()), "month": month,
                            "rows_02h_to_03h": int(win.sum()),
                            "unique_02h_to_03h": int(ts[win].nunique())})
    return out


# --------------------------------------------------------------------------- #
# 2. Missing values
# --------------------------------------------------------------------------- #
def missing_pattern(df: pd.DataFrame) -> dict[str, Any]:
    """Is missingness row-wise (acquisition outage) or column-wise (one sensor)?"""
    na = df[NUMERIC_COLUMNS].isna()
    n_any = int(na.any(axis=1).sum())
    n_all = int(na.all(axis=1).sum())
    return {"rows_with_any_nan": n_any, "rows_with_all_nan": n_all,
            "rows_partially_nan": n_any - n_all,
            "pct_rows_missing": round(100 * n_any / len(df), 3),
            "nan_per_column": {c: int(v) for c, v in na.sum().items()}}


def missing_runs(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    """One row per contiguous run of missing minutes: start, end, length (min)."""
    s = df.set_index("timestamp")[col]
    is_na = s.isna().to_numpy()
    if not is_na.any():
        return pd.DataFrame(columns=["start", "end", "length_min"])
    edges = np.diff(np.r_[0, is_na.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    idx = s.index
    return pd.DataFrame({"start": idx[starts], "end": idx[ends - 1], "length_min": ends - starts})


def run_length_summary(runs: pd.DataFrame) -> dict[str, Any]:
    bins = [0, 1, 10, 60, 24 * 60, np.inf]
    labels = ["1 min", "2-10 min", "11-60 min", "1 h - 1 day", "> 1 day"]
    cat = pd.cut(runs["length_min"], bins=bins, labels=labels)
    g = runs.groupby(cat, observed=False)["length_min"]
    return {
        "n_runs": int(len(runs)),
        "by_length": {str(k): {"runs": int(g.size()[k]), "minutes": int(g.sum()[k])} for k in labels},
        "longest": [{"start": str(r.start), "end": str(r.end), "hours": round(r.length_min / 60, 1)}
                    for r in runs.nlargest(6, "length_min").itertuples()],
    }


def missing_by_period(df: pd.DataFrame, col: str = TARGET) -> dict[str, dict[str, float]]:
    """Share of missing minutes per year, month-of-year and hour-of-day (mechanism hints)."""
    s = df.set_index("timestamp")[col].isna()
    return {
        "by_year_pct": (100 * s.groupby(s.index.year).mean()).round(3).to_dict(),
        "by_month_pct": (100 * s.groupby(s.index.month).mean()).round(3).to_dict(),
        "by_hour_pct": (100 * s.groupby(s.index.hour).mean()).round(3).to_dict(),
        "by_weekday_pct": (100 * s.groupby(s.index.day_of_week).mean()).round(3).to_dict(),
    }


def values_around_gaps(df: pd.DataFrame, runs: pd.DataFrame, col: str = TARGET,
                       min_len: int = 10, window: int = 30) -> dict[str, float]:
    """Compare consumption just before / after long gaps with the overall median.

    A systematic difference would suggest the gap is related to the household state
    (e.g. absence, power cut) -> missingness not completely at random.
    """
    s = df.set_index("timestamp")[col]
    pos = pd.Series(np.arange(len(s)), index=s.index)
    before, after = [], []
    for r in runs[runs.length_min >= min_len].itertuples():
        i0, i1 = pos[r.start], pos[r.end]
        before.append(s.iloc[max(0, i0 - window):i0].median())
        after.append(s.iloc[i1 + 1:i1 + 1 + window].median())
    return {"n_gaps_considered": int((runs.length_min >= min_len).sum()),
            "median_overall_kW": round(float(s.median()), 3),
            "median_before_gap_kW": round(float(np.nanmedian(before)), 3),
            "median_after_gap_kW": round(float(np.nanmedian(after)), 3)}


# --------------------------------------------------------------------------- #
# 3. Distributions, outliers, physical consistency
# --------------------------------------------------------------------------- #
def describe(df: pd.DataFrame) -> pd.DataFrame:
    q = df[NUMERIC_COLUMNS].quantile([0.0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1.0]).T
    q.columns = ["min", "q0.1%", "q1%", "q25%", "median", "q75%", "q99%", "q99.9%", "max"]
    q.insert(0, "mean", df[NUMERIC_COLUMNS].mean())
    q["zeros_pct"] = 100 * (df[NUMERIC_COLUMNS] == 0).mean()
    return q


def physical_consistency(df: pd.DataFrame) -> dict[str, Any]:
    """Cross-check variables using electrical relations.

    * Apparent power: S = sqrt(P^2 + Q^2) [kVA] and I ~= 1000 * S / V [A].
    * Energy: P [kW] averaged over one minute -> P*1000/60 Wh. The three sub-meters
      (Wh/min) must not exceed it; the remainder is "other" consumption.
    """
    d = df.dropna(subset=NUMERIC_COLUMNS)
    S = np.sqrt(d["Global_active_power"] ** 2 + d["Global_reactive_power"] ** 2)
    I_est = 1000 * S / d["Voltage"]
    rel_err = (I_est - d["Global_intensity"]).abs() / d["Global_intensity"].clip(lower=0.2)
    e_total = d["Global_active_power"] * 1000 / 60
    e_sub = d[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)
    other = e_total - e_sub
    return {
        "intensity_vs_estimate_corr": round(float(np.corrcoef(I_est, d["Global_intensity"])[0, 1]), 5),
        "intensity_median_rel_err_pct": round(float(100 * rel_err.median()), 2),
        "intensity_rel_err_gt_10pct_rows": int((rel_err > 0.10).sum()),
        "submeter_share_of_energy_pct": round(float(100 * e_sub.sum() / e_total.sum()), 2),
        "other_energy_negative_rows": int((other < -1).sum()),
        "other_energy_negative_pct": round(float(100 * (other < -1).mean()), 4),
        "voltage_outside_207_253V_pct": round(float(100 * ((d["Voltage"] < 207) | (d["Voltage"] > 253)).mean()), 4),
        "power_factor_median": round(float((d["Global_active_power"] / S).median()), 3),
    }


def robust_outliers(df: pd.DataFrame, col: str = TARGET, k: float = 5.0) -> dict[str, Any]:
    """Global robust z-score (median / MAD). Reported, NOT removed: high values are real appliances."""
    s = df[col].dropna()
    med = s.median()
    mad = 1.4826 * (s - med).abs().median()
    z = (s - med) / mad
    return {"median": round(float(med), 3), "mad_scaled": round(float(mad), 3),
            f"pct_abs_z_gt_{k:g}": round(float(100 * (z.abs() > k).mean()), 3),
            "max_value": round(float(s.max()), 3), "max_z": round(float(z.max()), 1)}


# --------------------------------------------------------------------------- #
# 4. Hourly aggregation
# --------------------------------------------------------------------------- #
def hourly_energy(df: pd.DataFrame, col: str = TARGET) -> pd.DataFrame:
    """Hourly energy (kWh) from minute-averaged power (kW).

    E_h = sum_{valid minutes} P_m / 60 is biased low when minutes are missing, so we
    report the minute count and the completeness-corrected estimate
    E_h = mean(P_m over valid minutes) * 1 h. Imputation policy is decided in phase 2.
    """
    s = df.set_index("timestamp")[col]
    g = s.resample("1h")
    out = pd.DataFrame({"n_valid": g.count(), "energy_kwh_mean_based": g.mean(),
                        "energy_kwh_sum_based": g.sum(min_count=1) / 60})
    # resample also creates hours that are totally absent in the raw index -> n_valid = 0
    return out


def hourly_completeness(h: pd.DataFrame) -> dict[str, Any]:
    n = h["n_valid"]
    return {"n_hours": int(len(n)),
            "complete_60": int((n == 60).sum()),
            "partial_1_59": int(((n > 0) & (n < 60)).sum()),
            "empty_0": int((n == 0).sum()),
            "partial_hours_valid_minutes_quantiles": n[(n > 0) & (n < 60)].quantile([0.1, 0.5, 0.9]).to_dict()}


# --------------------------------------------------------------------------- #
# 5. Seasonality, autocorrelation, stationarity (train + dev only)
# --------------------------------------------------------------------------- #
def seasonal_profiles(y: pd.Series) -> dict[str, pd.DataFrame | pd.Series]:
    """Mean hourly energy by (hour x weekday), by month and by year."""
    idx = y.index
    return {
        "hour_x_weekday": y.groupby([idx.hour, idx.day_of_week]).mean().unstack(),
        "by_month": y.groupby(idx.month).mean(),
        "by_year_month": y.resample("MS").mean(),
    }


def autocorrelation(y: pd.Series, nlags: int = 400) -> np.ndarray:
    """ACF with missing values handled pairwise ('conservative'): no imputation needed."""
    from statsmodels.tsa.stattools import acf
    return acf(y.to_numpy(), nlags=nlags, missing="conservative", fft=False)


def partial_autocorrelation(y: pd.Series, nlags: int = 72) -> np.ndarray:
    """PACF on the longest NaN-free segment (PACF does not accept gaps)."""
    from statsmodels.tsa.stattools import pacf
    seg = longest_complete_segment(y)
    return pacf(seg.to_numpy(), nlags=nlags, method="ywm")


def longest_complete_segment(y: pd.Series) -> pd.Series:
    ok = y.notna().to_numpy()
    edges = np.diff(np.r_[0, ok.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    i = int(np.argmax(ends - starts))
    return y.iloc[starts[i]:ends[i]]


def stationarity_tests(y: pd.Series) -> dict[str, Any]:
    """ADF (H0: unit root) and KPSS (H0: stationary) on the daily energy series.

    Daily aggregation removes the 24 h cycle so that the tests look at level/trend.
    The two tests are complementary: ADF rejects + KPSS does not reject -> stationary.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    daily = y.resample("1D").sum(min_count=20).dropna()
    adf = adfuller(daily, autolag="AIC")
    kp = kpss(daily, regression="c", nlags="auto")
    return {"n_days": int(len(daily)),
            "adf_stat": round(float(adf[0]), 3), "adf_pvalue": round(float(adf[1]), 4),
            "kpss_stat": round(float(kp[0]), 3), "kpss_pvalue": round(float(kp[1]), 4)}


# --------------------------------------------------------------------------- #
# 6. Reference baseline (written BEFORE any model)
# --------------------------------------------------------------------------- #
def seasonal_naive_scores(y: pd.Series, train_end: str, dev_end: str) -> dict[str, Any]:
    """MAE of y_hat(t) = y(t - m), m in {24, 168}, on the dev period; MASE scale from train.

    Only hours where both the target and the lagged value are observed are scored,
    so the comparison is on identical points for both baselines.
    """
    tr = y.index <= train_end
    dv = (y.index > train_end) & (y.index <= dev_end)
    lag24, lag168 = y.shift(24), y.shift(168)
    scale = (y - lag24)[tr].abs().mean()            # MASE scale: in-sample naive-24 MAE on train
    common = dv & y.notna().to_numpy() & lag24.notna().to_numpy() & lag168.notna().to_numpy()
    out = {"mase_scale_naive24_train_kwh": round(float(scale), 4), "n_dev_hours_scored": int(common.sum())}
    for name, pred in {"naive_24": lag24, "naive_168": lag168}.items():
        mae = (y - pred)[common].abs().mean()
        out[name] = {"MAE_kwh": round(float(mae), 4), "MASE": round(float(mae / scale), 4)}
    return out


def to_json(obj: Any, path: str | Path) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
