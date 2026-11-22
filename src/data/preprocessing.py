"""Phase 2 — reproducible, causal, logged preprocessing.

Pipeline (each step appends an entry to a JSON log):

1. minute grid check  -> the raw file already has a complete 1-min grid (phase 1);
   we re-check it and re-index defensively instead of assuming it.
2. hourly aggregation -> energy_kwh = mean(valid minutes of P [kW]) x 1 h, plus the
   number of valid minutes. Other variables: means (V, I, Q) or sums (sub-meters, Wh).
3. validity rule      -> an hour is OBSERVED if n_valid >= min_valid_minutes.
   Non-observed hours keep energy_kwh = NaN: the TARGET is never imputed.
4. causal fill        -> `energy_filled` is a copy of the target in which missing hours
   are replaced using ONLY past information. It is used to compute lag features,
   never as a training or evaluation target.

Two design notes:
* Using the valid minutes INSIDE an hour to estimate that hour is causal at hourly
  resolution: the hourly value only exists once the hour is over.
* Nothing here is fitted on future data. The only tuned choices (validity threshold,
  fill method by gap length) are selected on the TRAIN period by `select_*` functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


TARGET = "Global_active_power"
MEAN_COLS = ["Global_reactive_power", "Voltage", "Global_intensity"]     # instantaneous quantities -> mean
SUM_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]        # Wh per minute -> sum (Wh per hour)


@dataclass
class PrepLog:
    """Ordered log of every preprocessing decision and its effect."""
    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, **info: Any) -> None:
        self.steps.append({"step": step, **info})


# --------------------------------------------------------------------------- #
# 1-2. Minute grid and hourly aggregation
# --------------------------------------------------------------------------- #
def to_minute_grid(df: pd.DataFrame, log: PrepLog | None = None) -> pd.DataFrame:
    """Sort, drop exact duplicate timestamps (keep first) and re-index on a full 1-min grid."""
    n0, n_unique = len(df), int(df["timestamp"].nunique())
    d = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first").set_index("timestamp")
    grid = pd.date_range(d.index.min(), d.index.max(), freq="1min", name="timestamp")
    d = d.reindex(grid)
    if log:
        log.add("minute_grid", rows_in=n0, duplicates_dropped=n0 - n_unique, rows_out=len(d),
                timestamps_added=len(d) - n_unique, nan_rows=int(d[TARGET].isna().sum()))
    return d


def aggregate_hourly(m: pd.DataFrame, log: PrepLog | None = None) -> pd.DataFrame:
    """Minute grid -> hourly table. Partial hours keep their estimate + n_valid for later rules."""
    g = m.resample("1h")
    h = pd.DataFrame({"n_valid": g[TARGET].count(),
                      "energy_kwh": g[TARGET].mean()})          # mean kW over valid minutes x 1 h = kWh
    for c in MEAN_COLS:
        h[c] = g[c].mean()
    for c in SUM_COLS:
        # sum over valid minutes rescaled to 60 min (same unbiasedness argument as the target)
        h[c] = g[c].mean() * 60
    h.index.name = "timestamp"
    if log:
        n = h["n_valid"]
        log.add("hourly_aggregation", hours=len(h), complete=int((n == 60).sum()),
                partial=int(((n > 0) & (n < 60)).sum()), empty=int((n == 0).sum()),
                target_rule="energy_kwh = mean(P_kW over valid minutes) * 1 h",
                mean_cols=MEAN_COLS, sum_cols_rescaled_to_60min=SUM_COLS)
    return h


def apply_validity_rule(h: pd.DataFrame, min_valid: int, log: PrepLog | None = None) -> pd.DataFrame:
    """Hours with fewer than `min_valid` valid minutes become NOT observed (all values NaN)."""
    h = h.copy()
    h["observed"] = h["n_valid"] >= min_valid
    cols = ["energy_kwh"] + MEAN_COLS + SUM_COLS
    n_partial_dropped = int(((h["n_valid"] > 0) & ~h["observed"]).sum())
    h.loc[~h["observed"], cols] = np.nan
    if log:
        log.add("validity_rule", min_valid_minutes=min_valid, observed=int(h["observed"].sum()),
                not_observed=int((~h["observed"]).sum()), partial_hours_set_to_nan=n_partial_dropped)
    return h


# --------------------------------------------------------------------------- #
# 3. Experiment: how many valid minutes make an hour trustworthy? (train only)
# --------------------------------------------------------------------------- #
def partial_hour_error(m_train: pd.DataFrame, n_valid_grid: list[int], n_hours: int = 3000,
                       seed: int = 0) -> pd.DataFrame:
    """Simulate partial hours on COMPLETE train hours.

    For each sampled complete hour and each n_valid, keep a CONTIGUOUS block of n_valid
    minutes at a random position (real gaps are contiguous), estimate the hour by the mean
    of kept minutes, and compare with the true hourly energy.
    """
    rng = np.random.default_rng(seed)
    p = m_train[TARGET]
    g = p.groupby(p.index.floor("1h"))
    full = g.count() == 60
    hours = full[full].index.to_numpy()
    hours = rng.choice(hours, size=min(n_hours, len(hours)), replace=False)
    arr = p.loc[p.index.floor("1h").isin(hours)].to_numpy().reshape(-1, 60)   # sorted, complete -> reshape ok
    truth = arr.mean(axis=1)
    rows = []
    for k in n_valid_grid:
        starts = rng.integers(0, 60 - k + 1, size=len(arr))
        est = np.array([a[s:s + k].mean() for a, s in zip(arr, starts)])
        err = np.abs(est - truth)
        rows.append({"n_valid": k, "MAE_kwh": err.mean(), "p90_abs_err_kwh": np.quantile(err, 0.9),
                     "MAE_pct_of_mean": 100 * err.mean() / truth.mean()})
    return pd.DataFrame(rows).set_index("n_valid")


def select_min_valid(err: pd.DataFrame, reference_mae: float, max_ratio: float) -> int:
    """Smallest n_valid whose estimation MAE is below `max_ratio` x a reference error.

    The reference is the MAE of the naive-24 forecast on train: an hour is accepted as
    'observed' if estimating it from its own minutes is much more accurate than what any
    forecaster can hope to achieve, so it cannot distort model evaluation.
    """
    ok = err.index[err["MAE_kwh"] <= max_ratio * reference_mae]
    return int(ok.min()) if len(ok) else 60


# --------------------------------------------------------------------------- #
# 4. Causal fill methods for missing hours (used for FEATURES only)
# --------------------------------------------------------------------------- #
def fill_ffill(y: pd.Series) -> pd.Series:
    """Last observed value (zero-order hold)."""
    return y.ffill()


def fill_seasonal(y: pd.Series, period: int) -> pd.Series:
    """y(t) <- filled y(t - period). Sequential, so long gaps use already-filled values (causal)."""
    v = y.to_numpy().copy()
    for i in np.flatnonzero(np.isnan(v)):
        if i >= period:
            v[i] = v[i - period]
    out = pd.Series(v, index=y.index)
    return out.ffill()                      # beginning of series / still-missing -> last value


def fill_profile(y: pd.Series, weeks: int = 4) -> pd.Series:
    """Mean of the OBSERVED values at the same hour-of-week over the previous `weeks` weeks."""
    past = pd.concat([y.shift(168 * k) for k in range(1, weeks + 1)], axis=1).mean(axis=1, skipna=True)
    out = y.copy()
    miss = out.isna()
    out[miss] = past[miss]
    return out.ffill()


def fill_linear_offline(y: pd.Series) -> pd.Series:
    """NON-CAUSAL reference: linear interpolation uses the value AFTER the gap.

    Only reported to quantify what 'cheating' would gain; never used in the pipeline.
    """
    return y.interpolate("linear", limit_area="inside").ffill()


FILL_METHODS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "ffill": fill_ffill,
    "seasonal_24": lambda y: fill_seasonal(y, 24),
    "seasonal_168": lambda y: fill_seasonal(y, 168),
    "profile_4w": fill_profile,
    "linear_offline (non causal)": fill_linear_offline,
}
CAUSAL_METHODS = [k for k in FILL_METHODS if "non causal" not in k]


def evaluate_fill_methods(y_train: pd.Series, gap_lengths: list[int], n_blocks: int,
                          seed: int = 0) -> pd.DataFrame:
    """Mask artificial gaps of known length in fully observed stretches of the TRAIN series.

    Each artificial gap starts at least 4 weeks after the series start so that every method
    has history. MAE (kWh) is computed on masked hours only.
    """
    rng = np.random.default_rng(seed)
    v = y_train.to_numpy()
    rows = []
    for L in gap_lengths:
        # valid starts: block fully observed and at least 4 weeks of history
        obs = ~np.isnan(v)
        run = np.convolve(obs.astype(int), np.ones(L, int), "valid") == L
        starts = np.flatnonzero(run)
        starts = starts[(starts >= 4 * 168) & (starts + L < len(v))]
        chosen = rng.choice(starts, size=min(n_blocks, len(starts)), replace=False)
        errs = {k: [] for k in FILL_METHODS}
        for s in chosen:
            yy = y_train.copy()
            yy.iloc[s:s + L] = np.nan
            for name, f in FILL_METHODS.items():
                # only the window around the gap is needed: speeds up, same result for causal fills
                lo = max(0, s - 4 * 168 - 1)
                filled = f(yy.iloc[lo:s + L + 1])
                errs[name].append(np.abs(filled.iloc[s - lo:s - lo + L].to_numpy() - v[s:s + L]).mean())
        rows.append({"gap_h": L, "n_blocks": len(chosen), **{k: float(np.mean(e)) for k, e in errs.items()}})
    return pd.DataFrame(rows).set_index("gap_h")


def select_fill_policy(scores: pd.DataFrame) -> dict[int, str]:
    """Best CAUSAL method for each tested gap length."""
    return {int(L): str(scores.loc[L, CAUSAL_METHODS].astype(float).idxmin()) for L in scores.index}


def gap_lengths_of(y: pd.Series) -> pd.Series:
    """For each missing hour, the total length (h) of the gap it belongs to; NaN elsewhere.

    ANALYSIS ONLY — not causal (the length is known only when the gap ends).
    """
    miss = y.isna().to_numpy()
    edges = np.diff(np.r_[0, miss.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    out = np.full(len(y), np.nan)
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return pd.Series(out, index=y.index)


def causal_fill(y: pd.Series, short_max_h: int, short_method: str, long_method: str,
                log: PrepLog | None = None) -> tuple[pd.Series, pd.Series]:
    """Fill gaps <= short_max_h with `short_method`, longer ones with `long_method`.

    NOTE on causality of the switch: when a gap starts, its final length is unknown in real
    time. The rule is therefore applied PER HOUR using the time elapsed since the gap start
    (known in real time): hours 1..short_max_h of any gap use the short method, later hours
    the long method. This keeps the rule strictly causal.
    """
    miss = y.isna().to_numpy()
    # hours elapsed since gap start (1-based), computed causally
    elapsed = np.zeros(len(y), int)
    c = 0
    for i, m_ in enumerate(miss):
        c = c + 1 if m_ else 0
        elapsed[i] = c
    a, b = FILL_METHODS[short_method](y), FILL_METHODS[long_method](y)
    use_short = miss & (elapsed <= short_max_h)
    use_long = miss & (elapsed > short_max_h)
    filled = y.copy()
    filled[use_short] = a[use_short]
    filled[use_long] = b[use_long]
    method = pd.Series(np.where(use_short, short_method, np.where(use_long, long_method, "observed")), index=y.index)
    method[filled.isna()] = "unfilled"          # e.g. first hour of the file: no past to fill from
    if log:
        log.add("causal_fill", short_gap_max_hours=short_max_h, short_method=short_method, long_method=long_method,
                hours_filled_short=int(use_short.sum()), hours_filled_long=int(use_long.sum()),
                still_missing=int(filled.isna().sum()),
                note="filled values feed lag features only; target is never imputed")
    return filled, method


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_hourly(raw: pd.DataFrame, min_valid: int, short_max_h: int, short_method: str, long_method: str
                 ) -> tuple[pd.DataFrame, PrepLog]:
    """Full deterministic pipeline from the raw minute file to the processed hourly table."""
    log = PrepLog()
    m = to_minute_grid(raw, log)
    h = aggregate_hourly(m, log)
    h = apply_validity_rule(h, min_valid, log)
    h["energy_filled"], h["fill_method"] = causal_fill(h["energy_kwh"], short_max_h, short_method, long_method, log)
    # NB: total gap length (gap_lengths_of) is deliberately NOT stored: it is only known once the
    # gap is over, so it must never become a feature.
    return h, log


def save_processed(h: pd.DataFrame, log: PrepLog, processed_dir: str | Path, extra: dict | None = None) -> Path:
    from .diagnosis import to_json
    d = Path(processed_dir); d.mkdir(parents=True, exist_ok=True)
    path = d / "hourly.csv"
    h.to_csv(path, float_format="%.6f")
    to_json({"steps": log.steps, **(extra or {})}, d / "preprocessing_log.json")
    return path


def load_processed(processed_dir: str | Path) -> pd.DataFrame:
    h = pd.read_csv(Path(processed_dir) / "hourly.csv", index_col="timestamp", parse_dates=["timestamp"])
    return h
