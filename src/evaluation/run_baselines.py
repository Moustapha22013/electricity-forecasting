"""Phase 3 — baselines: selection on an inner TRAIN validation, then evaluation on DEV.

Inner validation (selection) : last `inner_days` of train, one parameter fit, daily origins.
Dev evaluation               : daily origins over dev, refit every `fold_days`, gap `gap_hours`.
The test set is not touched.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..data.diagnosis import to_json  # noqa: E402
from ..data.preprocessing import load_processed  # noqa: E402
from ..models.naive import SeasonalNaive, SeasonalProfile  # noqa: E402
from ..models.sarima import Sarima  # noqa: E402
from . import metrics as M  # noqa: E402
from .validation import attach_truth, split_origins, walk_forward  # noqa: E402


def _score(fc: pd.DataFrame) -> float:
    return M.mae(fc["y"], fc["pred"])


def select_sarima(data: pd.DataFrame, origins: pd.DatetimeIndex, bc: dict, log: list) -> tuple[dict, pd.DataFrame]:
    rows = []
    grid = itertools.product(bc["sarima_grid"]["order"], bc["sarima_grid"]["seasonal"], bc["sarima_grid"]["transform"])
    for order, seas, tr in grid:
        m = Sarima(tuple(order), tuple(seas) + (24,), tr, fit_window_h=bc["sarima_fit_window_h"],
                   filter_window_h=bc["sarima_filter_window_h"])
        t = time.time()
        try:
            fc = attach_truth(walk_forward(m, data, origins, fold_days=len(origins), gap_hours=bc["gap_hours"]),
                              data["energy_kwh"])
            rows.append({"order": tuple(order), "seasonal": tuple(seas), "transform": tr, "MAE": _score(fc),
                         "AIC": m.aic, "coverage_80": M.coverage(fc["y"], fc["lo"], fc["hi"]),
                         "seconds": round(time.time() - t, 1)})
        except Exception as exc:                       # a non-converging candidate must not stop the run
            rows.append({"order": tuple(order), "seasonal": tuple(seas), "transform": tr, "MAE": np.nan,
                         "AIC": np.nan, "coverage_80": np.nan, "error": repr(exc)})
        log.append(rows[-1])
        print("  sarima", rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values("MAE")
    best = res.iloc[0]
    return {"order": best["order"], "seasonal": best["seasonal"], "transform": best["transform"]}, res


def select_profile(data: pd.DataFrame, origins: pd.DatetimeIndex, bc: dict) -> tuple[dict, pd.DataFrame]:
    rows = []
    for w, stat in itertools.product(bc["profile_grid"]["weeks"], bc["profile_grid"]["stat"]):
        fc = attach_truth(walk_forward(SeasonalProfile(w, stat=stat), data, origins, fold_days=len(origins),
                                       gap_hours=bc["gap_hours"]), data["energy_kwh"])
        rows.append({"weeks": w, "stat": stat, "MAE": _score(fc)})
    res = pd.DataFrame(rows).sort_values("MAE")
    return {"weeks": int(res.iloc[0]["weeks"]), "stat": str(res.iloc[0]["stat"])}, res


def paired_diff_ci(fc_a: pd.DataFrame, fc_b: pd.DataFrame) -> tuple[float, float, float]:
    """Mean of |e_a| - |e_b| on common hours with a week-block bootstrap CI (<0 -> a better)."""
    d = (fc_a["y"] - fc_a["pred"]).abs() - (fc_b["y"] - fc_b["pred"]).abs()
    d = d.dropna()
    lo, hi = M.block_bootstrap_ci(d)
    return float(d.mean()), lo, hi


def run(cfg: dict) -> dict:
    bc = cfg["baselines"]
    rep = Path(cfg["reports"]["dir"]); fig = Path(cfg["reports"]["figures_dir"])
    data = load_processed(cfg["data"]["processed_dir"]).asfreq("h")
    train_end, dev_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["dev_end"])
    y = data["energy_kwh"]
    scale = M.mase_scale(y.loc[:train_end], 24)
    log: list = []

    # ---------------- 1. selection on inner train validation ----------------
    inner, dev = split_origins(train_end, dev_end, bc["inner_days"])
    data_tr = data.loc[:train_end]
    print(f"[baselines] inner validation: {inner[0].date()} -> {inner[-1].date()} ({len(inner)} origins)", flush=True)
    prof_best, prof_grid = select_profile(data_tr, inner, bc)
    sar_best, sar_grid = select_sarima(data_tr, inner, bc, log)

    # ---------------- 2. evaluation on dev ----------------
    data_dev = data.loc[:dev_end]
    a = bc["alpha"]
    models = [SeasonalNaive(24, alpha=a), SeasonalNaive(168, alpha=a),
              SeasonalProfile(prof_best["weeks"], alpha=a, stat=prof_best["stat"]),
              Sarima(sar_best["order"], tuple(sar_best["seasonal"]) + (24,), sar_best["transform"],
                     fit_window_h=bc["sarima_fit_window_h"], filter_window_h=bc["sarima_filter_window_h"], alpha=a)]
    fcs: dict[str, pd.DataFrame] = {}
    for m in models:
        t = time.time()
        fcs[m.name] = attach_truth(walk_forward(m, data_dev, dev, fold_days=bc["fold_days"], gap_hours=bc["gap_hours"]), y)
        print(f"[baselines] dev {m.name}: {time.time() - t:.1f}s", flush=True)
    sarima_model = models[-1]

    # identical scoring points for all models
    common = pd.concat([f["pred"].rename(k) for k, f in fcs.items()], axis=1).notna().all(axis=1) & y.reindex(
        fcs["naive_24"].index).notna()
    fcs = {k: f[common.to_numpy()] for k, f in fcs.items()}
    names = list(fcs)

    table = pd.DataFrame({k: M.summary(f["y"], f["pred"], scale, f["lo"], f["hi"], alpha=a) for k, f in fcs.items()}).T
    for k, f in fcs.items():
        table.loc[k, ["under_share", "under_mae_contrib", "over_mae_contrib"]] = [
            M.under_over(f["y"], f["pred"])[c] for c in ("under_share", "under_mae_contrib", "over_mae_contrib")]
        d, lo, hi = paired_diff_ci(f, fcs["naive_24"])
        table.loc[k, ["dMAE_vs_naive24", "dMAE_ci_low", "dMAE_ci_high"]] = [d, lo, hi]
    by_h = pd.DataFrame({k: (f["y"] - f["pred"]).abs().groupby(f["horizon"]).mean() for k, f in fcs.items()})
    cov_h = pd.DataFrame({k: ((f["y"] >= f["lo"]) & (f["y"] <= f["hi"])).groupby(f["horizon"]).mean()
                          for k, f in fcs.items()})
    by_fold = pd.DataFrame({k: f.assign(ae=(f["y"] - f["pred"]).abs()).groupby("fold")["ae"].mean()
                            for k, f in fcs.items()})
    fold_start = fcs["naive_24"].groupby("fold")["origin"].min().dt.date.astype(str)
    by_fold.index = fold_start.to_numpy()

    # worst days (error analysis)
    worst = {}
    for k, f in fcs.items():
        dd = (f["y"] - f["pred"]).abs().groupby(f["origin"]).mean().sort_values(ascending=False).head(6)
        worst[k] = [{"day": str(i.date()), "weekday": i.day_name(), "MAE": round(float(v), 3),
                     "day_energy_kwh": round(float(f.loc[f["origin"] == i, "y"].sum()), 1),
                     "pred_energy_kwh": round(float(f.loc[f["origin"] == i, "pred"].sum()), 1)} for i, v in dd.items()]

    # SARIMA in-sample residual diagnostics (last fold fit)
    sar_diag = sarima_model.residual_diagnostics()

    out = {"mase_scale_kwh": scale, "inner_validation": [str(inner[0].date()), str(inner[-1].date())],
           "profile_selection": prof_grid.to_dict(orient="records"), "profile_best": prof_best,
           "sarima_selection": sar_grid.astype({"order": str, "seasonal": str}).to_dict(orient="records"),
           "sarima_best": {k: str(v) for k, v in sar_best.items()}, "sarima_residual_diagnostics": sar_diag,
           "dev_origins": [str(dev[0].date()), str(dev[-1].date())], "dev_table": table.to_dict(orient="index"),
           "dev_mae_by_horizon": by_h.to_dict(), "dev_coverage_by_horizon": cov_h.to_dict(),
           "dev_mae_by_fold": by_fold.to_dict(), "worst_days": worst}
    to_json(out, rep / "03_baselines_results.json")
    pd.concat([f.assign(model=k) for k, f in fcs.items()]).to_csv(rep / "03_forecasts_dev.csv")

    # ---------------- figures ----------------
    f1, ax = plt.subplots(1, 2, figsize=(14, 4))
    for k in names:
        ax[0].plot(by_h.index, by_h[k], marker=".", label=k)
        ax[1].plot(cov_h.index, cov_h[k], marker=".", label=k)
    ax[0].set_xlabel("horizon h (= heure + 1)"); ax[0].set_ylabel("MAE (kWh)"); ax[0].set_title("MAE par horizon (dev)")
    ax[1].axhline(0.8, color="k", ls="--"); ax[1].set_ylim(0.5, 1); ax[1].set_xlabel("horizon h")
    ax[1].set_title("Couverture de l'intervalle à 80 % par horizon (dev)")
    ax[0].legend(fontsize=8); f1.tight_layout(); f1.savefig(fig / "03_mae_coverage_by_horizon.png", dpi=120); plt.close(f1)

    wk = slice("2010-02-01", "2010-02-07 23:00")
    f2, ax = plt.subplots(len(names), 1, figsize=(14, 2.4 * len(names)), sharex=True)
    for a, k in zip(ax, names):
        f = fcs[k].loc[wk]
        a.fill_between(f.index, f["lo"], f["hi"], alpha=0.25, color="C1", label="intervalle 80 %")
        a.plot(f.index, f["y"], "k", lw=1, label="observé"); a.plot(f.index, f["pred"], "C1", lw=1.2, label="prévision")
        a.set_ylabel("kWh"); a.set_title(k, fontsize=9)
    ax[0].legend(fontsize=8, ncol=3); f2.tight_layout(); f2.savefig(fig / "03_example_week.png", dpi=120); plt.close(f2)

    f3, ax = plt.subplots(figsize=(10, 3.6))
    by_fold.plot(kind="bar", ax=ax); ax.set_ylabel("MAE (kWh)"); ax.set_title("MAE par pli de 28 jours (dev)")
    ax.tick_params(axis="x", rotation=0); f3.tight_layout(); f3.savefig(fig / "03_mae_by_fold.png", dpi=120); plt.close(f3)
    return out | {"table": table, "by_h": by_h, "by_fold": by_fold, "sarima_grid": sar_grid, "profile_grid": prof_grid}


if __name__ == "__main__":
    from ..config import load_config
    r = run(load_config())
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(r["profile_grid"].round(4).to_string()); print(r["sarima_grid"].round(4).to_string())
    print(r["table"].round(4).to_string()); print(r["by_h"].round(3).to_string()); print(r["by_fold"].round(3).to_string())
    print(r["sarima_residual_diagnostics"]); import pprint; pprint.pprint(r["worst_days"], width=160)
