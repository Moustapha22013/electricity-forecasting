"""Run the full phase-1 diagnosis: JSON of numbers + figures in reports/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import diagnosis as D  # noqa: E402
from .loader import download_raw, load_raw  # noqa: E402

MIN_VALID_MINUTES_DIAG = 45   # provisional rule for EDA only; the real policy is set in phase 2


def _save(fig: plt.Figure, figdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figdir / name, dpi=120)
    plt.close(fig)


def run(cfg: dict) -> dict:
    figdir = Path(cfg["reports"]["figures_dir"]); figdir.mkdir(parents=True, exist_ok=True)
    repdir = Path(cfg["reports"]["dir"])
    raw = download_raw(cfg["data"]["url"], cfg["data"]["raw_dir"], cfg["data"]["raw_file"], cfg["data"]["zip_sha256"])
    df = load_raw(raw)
    train_end, dev_end = cfg["split"]["train_end"], cfg["split"]["dev_end"]
    res: dict = {}

    # ---- quality checks on the whole file ----
    res["timestamps"] = D.timestamp_checks(df)
    res["dst"] = D.dst_check(df)
    res["missing_pattern"] = D.missing_pattern(df)
    runs = D.missing_runs(df)
    res["missing_runs"] = D.run_length_summary(runs)
    res["missing_by_period"] = D.missing_by_period(df)
    res["values_around_gaps"] = D.values_around_gaps(df, runs)
    res["describe"] = D.describe(df).round(4).to_dict(orient="index")
    res["physics"] = D.physical_consistency(df)
    res["outliers"] = D.robust_outliers(df)

    h = D.hourly_energy(df)
    res["hourly_completeness"] = D.hourly_completeness(h)
    res["hourly_mean_vs_sum_based_complete_hours_max_abs_diff"] = float(
        (h.loc[h.n_valid == 60, "energy_kwh_mean_based"] - h.loc[h.n_valid == 60, "energy_kwh_sum_based"]).abs().max())

    # ---- modelling-relevant analysis on train + dev only ----
    y_all = h["energy_kwh_mean_based"].where(h["n_valid"] >= MIN_VALID_MINUTES_DIAG)
    y = y_all[y_all.index <= dev_end]
    res["hourly_target_train_dev"] = {"n_hours": int(len(y)), "n_nan": int(y.isna().sum()),
                                      "mean_kwh": round(float(y.mean()), 4), "median_kwh": round(float(y.median()), 4),
                                      "std_kwh": round(float(y.std()), 4), "skew": round(float(y.skew()), 3),
                                      "q99_kwh": round(float(y.quantile(0.99)), 4)}
    prof = D.seasonal_profiles(y)
    rho = D.autocorrelation(y, nlags=400)
    res["acf"] = {int(l): round(float(rho[l]), 3) for l in (1, 2, 3, 6, 12, 23, 24, 25, 48, 144, 168, 336)}
    phi = D.partial_autocorrelation(y, nlags=72)
    res["pacf_segment_hours"] = int(len(D.longest_complete_segment(y)))
    res["pacf"] = {int(l): round(float(phi[l]), 3) for l in (1, 2, 3, 23, 24, 25, 48, 72)}
    res["stationarity_daily"] = D.stationarity_tests(y)
    res["baseline_dev"] = D.seasonal_naive_scores(y, train_end, dev_end)
    D.to_json(res, repdir / "01_diagnosis_results.json")
    runs.to_csv(repdir / "01_missing_runs.csv", index=False)

    # ---- figures ----
    s = df.set_index("timestamp")[D.TARGET]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    daily = h["energy_kwh_mean_based"].resample("1D").mean() * 24
    ax.plot(daily.index, daily, lw=0.7)
    for r in runs[runs.length_min >= 60].itertuples():
        ax.axvspan(r.start, r.end, color="red", alpha=0.35)
    for x, lab in ((train_end, "fin train"), (dev_end, "fin dev")):
        ax.axvline(pd.Timestamp(x), color="k", ls="--"); ax.text(pd.Timestamp(x), ax.get_ylim()[1] * 0.95, " " + lab, fontsize=8)
    ax.set_ylabel("kWh / jour"); ax.set_title("Énergie journalière (rouge : trous ≥ 1 h)")
    _save(fig, figdir, "01_daily_energy_gaps.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    L = runs["length_min"]
    ax[0].hist(np.log10(L), bins=40); ax[0].set_xlabel("log10(durée du trou, min)"); ax[0].set_ylabel("nombre de trous")
    ax[0].set_title(f"{len(L)} trous, {int(L.sum())} minutes")
    mbh = pd.Series(res["missing_by_period"]["by_hour_pct"])
    ax[1].bar(mbh.index, mbh.values); ax[1].set_xlabel("heure"); ax[1].set_ylabel("% minutes manquantes")
    ax[1].set_title("Manquants par heure de la journée")
    _save(fig, figdir, "01_missing_runs.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].hist(s.dropna(), bins=200); ax[0].set_yscale("log"); ax[0].set_xlabel("kW (minute)"); ax[0].set_title("Puissance active (minute)")
    ax[1].hist(y.dropna(), bins=100); ax[1].set_xlabel("kWh (heure)"); ax[1].set_title("Énergie horaire (train+dev)")
    ax[2].hist(df["Voltage"].dropna(), bins=100); ax[2].set_xlabel("V"); ax[2].set_title("Tension")
    _save(fig, figdir, "01_distributions.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    hw = prof["hour_x_weekday"]
    im = ax.imshow(hw, aspect="auto", cmap="viridis")
    ax.set_xticks(range(7)); ax.set_xticklabels(["lun", "mar", "mer", "jeu", "ven", "sam", "dim"])
    ax.set_ylabel("heure"); ax.grid(False); ax.set_title("kWh moyen par heure et jour (train+dev)")
    plt.colorbar(im)
    _save(fig, figdir, "01_profile_hour_weekday.png")

    fig, ax = plt.subplots(figsize=(13, 3.2))
    bym = prof["by_year_month"]
    ax.plot(bym.index, bym.values * 24, marker="o"); ax.set_ylabel("kWh / jour (moyenne mensuelle)")
    ax.set_title("Saisonnalité annuelle (train+dev)")
    _save(fig, figdir, "01_monthly.png")

    fig, ax = plt.subplots(2, 1, figsize=(13, 6))
    ax[0].bar(range(len(rho)), rho, width=1.0)
    for l in (24, 48, 168, 336):
        ax[0].axvline(l, color="r", ls=":", lw=1)
    ax[0].set_title("ACF de l'énergie horaire (train+dev, manquants exclus par paires)"); ax[0].set_xlabel("retard (h)")
    ax[1].bar(range(len(phi)), phi, width=0.8)
    ci = 1.96 / np.sqrt(res["pacf_segment_hours"])
    ax[1].axhline(ci, color="r", ls=":"); ax[1].axhline(-ci, color="r", ls=":")
    ax[1].set_title(f"PACF (plus long segment complet : {res['pacf_segment_hours']} h)"); ax[1].set_xlabel("retard (h)")
    _save(fig, figdir, "01_acf_pacf.png")

    nv = h["n_valid"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(nv[(nv > 0) & (nv < 60)], bins=59); ax.set_xlabel("minutes valides dans l'heure")
    ax.set_title("Heures partielles : nombre de minutes valides")
    _save(fig, figdir, "01_partial_hours.png")
    return res


if __name__ == "__main__":
    from ..config import load_config
    import pprint
    pprint.pprint(run(load_config()), width=140, sort_dicts=False)
