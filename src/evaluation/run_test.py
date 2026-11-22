"""Phase 6 — single, pre-registered evaluation on the TEST period.

Safeguard: the SHA-256 of the `final` config section is stored in reports/06_test_lock.json
at the first run. A rerun with the same section recomputes the same numbers (deterministic:
reproducibility); a rerun with a modified section is REFUSED (it would be a second look at
the test set with a different pipeline).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..data.diagnosis import to_json  # noqa: E402
from ..data.preprocessing import load_processed  # noqa: E402
from ..features.engineering import build_features, calendar_features  # noqa: E402
from ..models.boosting import make_boosting  # noqa: E402
from ..models.naive import SeasonalNaive  # noqa: E402
from ..models.ridge import make_ridge  # noqa: E402
from ..models.sarima import Sarima  # noqa: E402
from . import metrics as M  # noqa: E402
from .run_baselines import paired_diff_ci  # noqa: E402
from .run_models import ABLATION  # noqa: E402
from .validation import attach_truth, daily_origins, walk_forward  # noqa: E402


class TestLockError(RuntimeError):
    pass


def _section_hash(final: dict) -> str:
    return hashlib.sha256(json.dumps(final, sort_keys=True).encode()).hexdigest()


def check_lock(final: dict, lock_path: Path) -> dict:
    h = _section_hash(final)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["final_sha256"] != h:
            raise TestLockError("The pre-registered `final` section changed after the test was run. "
                                "Refusing to re-evaluate on the test set with a different pipeline.")
        return lock | {"rerun": True}
    lock = {"final_sha256": h, "first_run": datetime.now().isoformat(timespec="seconds"), "final": final}
    lock_path.write_text(json.dumps(lock, indent=2))
    return lock | {"rerun": False}


def run(cfg: dict) -> dict:
    fc_cfg = cfg["final"]
    rep, fig = Path(cfg["reports"]["dir"]), Path(cfg["reports"]["figures_dir"])
    lock = check_lock(fc_cfg, rep / "06_test_lock.json")
    p = fc_cfg["protocol"]
    seed = cfg["seed"]
    data = load_processed(cfg["data"]["processed_dir"]).asfreq("h")
    y = data["energy_kwh"]
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    scale = M.mase_scale(y.loc[:train_end], 24)                     # same fixed scale as before
    origins = daily_origins(fc_cfg["test_start"], data.index[-1])
    print(f"[test] origins {origins[0].date()} -> {origins[-1].date()} ({len(origins)} days); lock rerun={lock['rerun']}",
          flush=True)

    feats_sel = ABLATION[fc_cfg["feature_config"]]
    r = fc_cfg["references"]
    models = {
        "naive_24": SeasonalNaive(24, alpha=p["alpha"]),
        "sarima": Sarima(tuple(r["sarima"]["order"]), tuple(r["sarima"]["seasonal"]) + (24,), r["sarima"]["transform"],
                         fit_window_h=cfg["baselines"]["sarima_fit_window_h"],
                         filter_window_h=cfg["baselines"]["sarima_filter_window_h"], alpha=p["alpha"]),
        "ridge": make_ridge(ABLATION[r["ridge"]["feature_config"]], alpha=r["ridge"]["alpha"], pi_alpha=p["alpha"],
                            calib_days=p["calib_days"]),
        "gbm (retenu)": make_boosting(feats_sel, seed=seed, calib_days=p["calib_days"], alpha=p["alpha"],
                                      **fc_cfg["gbm_params"]),
    }
    fcs = {}
    for name, m in models.items():
        t = time.time()
        fcs[name] = attach_truth(walk_forward(m, data, origins, fold_days=p["fold_days"], gap_hours=p["gap_hours"]), y)
        print(f"[test] {name}: {time.time() - t:.0f}s", flush=True)

    common = None
    for f in fcs.values():
        ok = f.index[f["y"].notna() & f["pred"].notna()]
        common = ok if common is None else common.intersection(ok)
    fcs = {k: f.loc[common] for k, f in fcs.items()}
    names = list(fcs)

    table = pd.DataFrame({k: M.summary(f["y"], f["pred"], scale, f["lo"], f["hi"], alpha=p["alpha"]) for k, f in fcs.items()}).T
    for k in names:
        for ref in ("sarima", "naive_24"):
            d, lo, hi = paired_diff_ci(fcs[k], fcs[ref])
            table.loc[k, [f"dMAE_vs_{ref}", f"ci_lo_{ref}", f"ci_hi_{ref}"]] = [d, lo, hi]
        e = fcs[k]["y"] - fcs[k]["pred"]
        table.loc[k, "daily_volume_bias_kwh"] = float(e.groupby(fcs[k]["origin"]).sum().mean())
    table["daily_energy_mean_kwh"] = float(fcs["naive_24"]["y"].groupby(fcs["naive_24"]["origin"]).sum().mean())

    def ae(k):
        return (fcs[k]["y"] - fcs[k]["pred"]).abs()

    by_h = pd.DataFrame({k: ae(k).groupby(fcs[k]["horizon"]).mean() for k in names})
    cov_h = pd.DataFrame({k: ((fcs[k]["y"] >= fcs[k]["lo"]) & (fcs[k]["y"] <= fcs[k]["hi"])).groupby(fcs[k]["horizon"]).mean()
                          for k in names})
    by_month = pd.DataFrame({k: ae(k).groupby(fcs[k].index.to_period("M").astype(str)).mean() for k in names})
    by_month["n_hours"] = ae("naive_24").groupby(fcs["naive_24"].index.to_period("M").astype(str)).size()

    # effect of the long gaps (Aug / Sep 2010): hours whose recent history was partly imputed
    X = build_features(data).loc[common]
    impacted = X["frac_filled_168"] > 0
    gap_effect = {k: {"MAE_history_imputed": float(ae(k)[impacted.to_numpy()].mean()),
                      "MAE_history_complete": float(ae(k)[~impacted.to_numpy()].mean()),
                      "n_hours_imputed": int(impacted.sum())} for k in names}
    # public holidays in the test period (descriptive only)
    cal = calendar_features(pd.DatetimeIndex(common))
    hol_mask = cal["is_holiday"].to_numpy() == 1
    holidays_test = {}
    for d in sorted(set(pd.DatetimeIndex(common)[hol_mask].normalize())):
        m = pd.DatetimeIndex(common).normalize() == d
        holidays_test[f"{d.date()} {d.day_name()}"] = {k: round(float(ae(k)[m].mean()), 3) for k in names}
    worst = ae("gbm (retenu)").groupby(fcs["gbm (retenu)"]["origin"]).mean().nlargest(8)

    out = {"lock": {k: v for k, v in lock.items() if k != "final"}, "n_origins": len(origins),
           "n_common_hours": int(len(common)), "test_table": table.to_dict(orient="index"),
           "mae_by_horizon": by_h.to_dict(), "coverage_by_horizon": cov_h.to_dict(), "mae_by_month": by_month.to_dict(),
           "gap_effect": gap_effect, "holidays_test": holidays_test,
           "gbm_worst_days": {f"{k.date()} {k.day_name()}": round(float(v), 3) for k, v in worst.items()}}
    to_json(out, rep / "06_test_results.json")
    pd.concat([f.assign(model=k) for k, f in fcs.items()]).to_csv(rep / "06_forecasts_test.csv")

    # ---- figures ----
    f1, ax = plt.subplots(1, 2, figsize=(14, 4))
    for k in names:
        ax[0].plot(by_h.index, by_h[k], marker=".", label=k); ax[1].plot(cov_h.index, cov_h[k], marker=".", label=k)
    ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("MAE (kWh)"); ax[0].set_title("MAE par horizon (TEST)"); ax[0].legend(fontsize=8)
    ax[1].axhline(0.8, color="k", ls="--"); ax[1].set_ylim(0.4, 1); ax[1].set_title("Couverture 80 % par horizon (TEST)")
    f1.tight_layout(); f1.savefig(fig / "06_test_by_horizon.png", dpi=120); plt.close(f1)

    f2, ax = plt.subplots(figsize=(10, 3.8))
    by_month[names].plot(kind="bar", ax=ax); ax.set_ylabel("MAE (kWh)"); ax.set_title("MAE par mois (TEST)")
    ax.tick_params(axis="x", rotation=0); f2.tight_layout(); f2.savefig(fig / "06_test_by_month.png", dpi=120); plt.close(f2)

    z = slice("2010-08-01", "2010-09-05 23:00")
    f3, ax = plt.subplots(figsize=(14, 3.6))
    g = fcs["gbm (retenu)"].loc[z]
    ax.fill_between(g.index, g["lo"], g["hi"], color="C3", alpha=0.2, label="intervalle 80 %")
    ax.plot(y.loc[z].index, y.loc[z], "k", lw=0.7, label="observé")
    ax.plot(g.index, g["pred"], "C3", lw=0.9, label="gbm (retenu)")
    ax.set_ylabel("kWh"); ax.legend(fontsize=8); ax.set_title("TEST : août 2010 (vacances et trou du 17 au 22/08)")
    f3.tight_layout(); f3.savefig(fig / "06_test_august.png", dpi=120); plt.close(f3)
    return out | {"table": table, "by_h": by_h, "cov_h": cov_h, "by_month": by_month}


if __name__ == "__main__":
    from ..config import load_config
    res = run(load_config())
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for k in ("table", "by_h", "cov_h", "by_month"):
        print(k); print(res[k].round(4).to_string())
    import pprint
    pprint.pprint({k: res[k] for k in ("gap_effect", "holidays_test", "gbm_worst_days", "lock")}, width=150)
