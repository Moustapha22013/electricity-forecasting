"""Phase 5 — Ridge and gradient boosting: tuning (inner train validation), dev evaluation
against the phase-3 baselines, ablation by feature group, and a labelled leakage demo.
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
from ..features.engineering import ALL_FEATURES, FEATURE_GROUPS, build_features, leaky_lag_1  # noqa: E402
from ..models.boosting import make_boosting  # noqa: E402
from ..models.ridge import make_ridge  # noqa: E402
from . import metrics as M  # noqa: E402
from .run_baselines import paired_diff_ci  # noqa: E402
from .validation import attach_truth, split_origins, walk_forward  # noqa: E402

G = FEATURE_GROUPS
ABLATION: dict[str, list[str]] = {
    "1_lags": G["lags"],
    "2_lags+calendar": G["lags"] + G["calendar"],
    "3_lags+rolling": G["lags"] + G["rolling"],
    "4_lags+rolling+calendar": G["lags"] + G["rolling"] + G["calendar"],
    "5_all": ALL_FEATURES,
    "6_all-without-lag_1": [f for f in ALL_FEATURES if f != "lag_1"],
}


def _wf(model, data, origins, fold_days, gap, y) -> pd.DataFrame:
    return attach_truth(walk_forward(model, data, origins, fold_days=fold_days, gap_hours=gap), y)


class _LeakyDemo:
    """DELIBERATELY BROKEN — reads a precomputed matrix that contains y(t-1) relative to the
    target, i.e. information from the target day. Only used to quantify what leakage would
    'gain'. Never used for model selection."""

    def __init__(self, inner, X_leak: pd.DataFrame, feats: list[str]):
        self.inner, self.X, self.feats, self.name = inner, X_leak, feats, "LEAK_demo_gbm"

    def fit(self, history):
        X = self.X.loc[: history.index[-1]]
        ok = X["observed"].astype(bool) & X[self.feats].notna().all(axis=1)
        self.inner.fit(X.loc[ok, self.feats], X.loc[ok, "y"]); return self

    def predict(self, history, origin, horizon=24):
        idx = pd.date_range(origin, periods=horizon, freq="h")
        p = np.maximum(self.inner.predict(self.X.loc[idx, self.feats]), 0)
        return pd.DataFrame({"pred": p, "lo": np.nan, "hi": np.nan}, index=idx)


def run(cfg: dict) -> dict:
    mc, bc = cfg["models"], cfg["baselines"]
    rep, fig = Path(cfg["reports"]["dir"]), Path(cfg["reports"]["figures_dir"])
    seed = cfg["seed"]
    data = load_processed(cfg["data"]["processed_dir"]).asfreq("h")
    train_end, dev_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["dev_end"])
    y = data["energy_kwh"]
    scale = M.mase_scale(y.loc[:train_end], 24)
    inner, dev = split_origins(train_end, dev_end, bc["inner_days"])
    data_tr, data_dev = data.loc[:train_end], data.loc[:dev_end]
    gap, fold = bc["gap_hours"], bc["fold_days"]
    out: dict = {"inner": [str(inner[0].date()), str(inner[-1].date())], "dev": [str(dev[0].date()), str(dev[-1].date())]}

    # ---------------- 1. tuning on inner train validation (all features, one fit) ----------------
    tune = []
    for a in mc["ridge_alpha"]:
        t = time.time()
        fc = _wf(make_ridge(ALL_FEATURES, alpha=a, calib_days=0), data_tr, inner, len(inner), gap, y)
        tune.append({"model": "ridge", "params": {"alpha": a}, "MAE": M.mae(fc["y"], fc["pred"]), "s": time.time() - t})
        print("  tune", tune[-1], flush=True)
    gg = mc["gbm_grid"]
    for loss, lr, it, lf in itertools.product(gg["loss"], gg["learning_rate"], gg["max_iter"], gg["max_leaf_nodes"]):
        t = time.time()
        p = {"loss": loss, "learning_rate": lr, "max_iter": it, "max_leaf_nodes": lf}
        fc = _wf(make_boosting(ALL_FEATURES, seed=seed, calib_days=0, **p), data_tr, inner, len(inner), gap, y)
        tune.append({"model": "gbm", "params": p, "MAE": M.mae(fc["y"], fc["pred"]), "s": time.time() - t})
        print("  tune", tune[-1], flush=True)
    tune_df = pd.DataFrame(tune)
    best = {m: tune_df[tune_df.model == m].sort_values("MAE").iloc[0]["params"] for m in ("ridge", "gbm")}
    out["tuning"] = tune_df.assign(params=tune_df.params.astype(str)).round(4).to_dict(orient="records")
    out["best_params"] = best

    def ridge(feats, **kw):
        return make_ridge(feats, alpha=best["ridge"]["alpha"], pi_alpha=bc["alpha"], calib_days=mc["calib_days"], **kw)

    def gbm(feats, **kw):
        return make_boosting(feats, seed=seed, calib_days=mc["calib_days"], alpha=bc["alpha"], **best["gbm"], **kw)

    # ---------------- 2. dev evaluation, all features ----------------
    fcs: dict[str, pd.DataFrame] = {}
    for name, m in (("ridge", ridge(ALL_FEATURES)), ("gbm", gbm(ALL_FEATURES))):
        t = time.time()
        fcs[name] = _wf(m, data_dev, dev, fold, gap, y)
        print(f"[models] dev {name}: {time.time() - t:.0f}s", flush=True)
    base = pd.read_csv(rep / "03_forecasts_dev.csv", index_col="target", parse_dates=["target", "origin"])
    for k, g in base.groupby("model"):
        fcs[k] = g.drop(columns="model")
    sarima_key = next(k for k in fcs if k.startswith("sarima"))

    common = None
    for f in fcs.values():
        ok = f.index[f["y"].notna() & f["pred"].notna()]
        common = ok if common is None else common.intersection(ok)
    fcs = {k: f.loc[common] for k, f in fcs.items()}
    order = ["naive_24", "naive_168", "profile_4w_median", sarima_key, "ridge", "gbm"]
    table = pd.DataFrame({k: M.summary(fcs[k]["y"], fcs[k]["pred"], scale, fcs[k]["lo"], fcs[k]["hi"], alpha=bc["alpha"])
                          for k in order}).T
    for k in order:
        for ref, tag in ((sarima_key, "sarima"), ("naive_24", "naive24")):
            d, lo, hi = paired_diff_ci(fcs[k], fcs[ref])
            table.loc[k, [f"dMAE_vs_{tag}", f"ci_lo_{tag}", f"ci_hi_{tag}"]] = [d, lo, hi]
        table.loc[k, ["under_mae_contrib", "over_mae_contrib"]] = [
            M.under_over(fcs[k]["y"], fcs[k]["pred"])[c] for c in ("under_mae_contrib", "over_mae_contrib")]
    by_h = pd.DataFrame({k: (fcs[k]["y"] - fcs[k]["pred"]).abs().groupby(fcs[k]["horizon"]).mean() for k in order})
    cov_h = pd.DataFrame({k: ((fcs[k]["y"] >= fcs[k]["lo"]) & (fcs[k]["y"] <= fcs[k]["hi"])).groupby(fcs[k]["horizon"]).mean()
                          for k in order})
    by_fold = pd.DataFrame({k: (fcs[k]["y"] - fcs[k]["pred"]).abs().groupby(fcs[k]["fold"]).mean() for k in order})
    worst = (fcs["gbm"]["y"] - fcs["gbm"]["pred"]).abs().groupby(fcs["gbm"]["origin"]).mean().nlargest(8)
    out |= {"n_common_hours": int(len(common)), "dev_table": table.to_dict(orient="index"),
            "mae_by_horizon": by_h.to_dict(), "coverage_by_horizon": cov_h.to_dict(), "mae_by_fold": by_fold.to_dict(),
            "gbm_worst_days": {str(k.date()) + " " + k.day_name(): round(float(v), 3) for k, v in worst.items()}}
    pd.concat([fcs[k].assign(model=k) for k in ("ridge", "gbm")]).to_csv(rep / "05_forecasts_dev.csv")

    # ---------------- 3. ablation (dev, same protocol, same scored hours) ----------------
    abl = []
    for cname, feats in ABLATION.items():
        for mname, mk in (("ridge", ridge), ("gbm", gbm)):
            t = time.time()
            f = _wf(mk(feats, ) if mname == "gbm" else mk(feats), data_dev, dev, fold, gap, y).loc[common]
            d, lo, hi = paired_diff_ci(f, fcs[sarima_key])
            abl.append({"config": cname, "model": mname, "n_features": len(feats), "MAE": M.mae(f["y"], f["pred"]),
                        "dMAE_vs_sarima": d, "ci_lo": lo, "ci_hi": hi,
                        "MAE_h1_3": float((f["y"] - f["pred"]).abs()[f["horizon"] <= 3].mean())})
            print("  ablation", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in abl[-1].items()},
                  f"{time.time() - t:.0f}s", flush=True)
    abl_df = pd.DataFrame(abl)
    out["ablation"] = abl_df.round(4).to_dict(orient="records")

    # ---------------- 4. labelled leakage demo (bypasses the firewall on purpose) ----------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xl = build_features(data_dev)
    Xl["lag_1_LEAK"] = leaky_lag_1(data_dev)
    feats_leak = [f for f in ALL_FEATURES if f != "lag_1"] + ["lag_1_LEAK"]
    est = HistGradientBoostingRegressor(loss=best["gbm"]["loss"], learning_rate=best["gbm"]["learning_rate"],
                                        max_iter=best["gbm"]["max_iter"], max_leaf_nodes=best["gbm"]["max_leaf_nodes"],
                                        min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=seed)
    fl = _wf(_LeakyDemo(est, Xl, feats_leak), data_dev, dev, fold, gap, y).loc[common]
    out["leak_demo"] = {"MAE_with_leaky_lag_1": M.mae(fl["y"], fl["pred"]),
                        "MAE_legal_gbm": M.mae(fcs["gbm"]["y"], fcs["gbm"]["pred"])}

    # ---------------- 5. permutation importance (fold-0 model, dev rows) ----------------
    from sklearn.inspection import permutation_importance
    m0 = gbm(ALL_FEATURES)
    m0.fit(data.loc[:dev[0] - pd.Timedelta(hours=gap + 1)])
    Xd = build_features(data_dev).loc[common, ALL_FEATURES]
    pi = permutation_importance(m0.model, Xd, y.loc[common], n_repeats=5, random_state=seed,
                                scoring="neg_mean_absolute_error")
    out["permutation_importance_gbm"] = pd.Series(pi.importances_mean, index=ALL_FEATURES).sort_values(
        ascending=False).round(4).to_dict()
    to_json(out, rep / "05_models_results.json")

    # ---------------- figures ----------------
    f1, ax = plt.subplots(1, 2, figsize=(14, 4))
    for k in order:
        ls = "-" if k in ("ridge", "gbm") else ":"
        ax[0].plot(by_h.index, by_h[k], ls, marker=".", label=k)
        ax[1].plot(cov_h.index, cov_h[k], ls, marker=".", label=k)
    ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("MAE (kWh)"); ax[0].set_title("MAE par horizon (dev)"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.8, color="k", ls="--"); ax[1].set_ylim(0.5, 1); ax[1].set_xlabel("horizon h")
    ax[1].set_title("Couverture 80 % par horizon (dev)")
    f1.tight_layout(); f1.savefig(fig / "05_mae_coverage_by_horizon.png", dpi=120); plt.close(f1)

    f2, ax = plt.subplots(figsize=(10, 4))
    piv = abl_df.pivot(index="config", columns="model", values="MAE")
    piv.plot(kind="bar", ax=ax); ax.axhline(table.loc[sarima_key, "MAE"], color="r", ls="--", label="SARIMA")
    ax.set_ylim(piv.min().min() * 0.9, piv.max().max() * 1.05); ax.set_ylabel("MAE dev (kWh)")
    ax.set_title("Ablation par groupes de variables (dev)"); ax.legend(); ax.tick_params(axis="x", rotation=20)
    f2.tight_layout(); f2.savefig(fig / "05_ablation.png", dpi=120); plt.close(f2)

    wk = slice("2010-02-01", "2010-02-07 23:00")
    f3, ax = plt.subplots(figsize=(14, 3.6))
    f = fcs["gbm"].loc[wk]
    ax.fill_between(f.index, f["lo"], f["hi"], alpha=0.25, color="C2", label="intervalle 80 % (conforme par heure)")
    ax.plot(f.index, f["y"], "k", lw=1, label="observé"); ax.plot(f.index, f["pred"], "C2", lw=1.3, label="gbm")
    ax.plot(f.index, fcs[sarima_key].loc[wk, "pred"], "C3", lw=0.9, alpha=0.7, label="SARIMA")
    ax.set_ylabel("kWh"); ax.legend(fontsize=8, ncol=4); ax.set_title("Semaine d'exemple (dev)")
    f3.tight_layout(); f3.savefig(fig / "05_example_week.png", dpi=120); plt.close(f3)
    return out | {"table": table, "by_h": by_h, "cov_h": cov_h, "by_fold": by_fold, "ablation_df": abl_df}


if __name__ == "__main__":
    from ..config import load_config
    r = run(load_config())
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for k in ("table", "by_h", "cov_h", "by_fold", "ablation_df"):
        print(k); print(r[k].round(4).to_string())
    import pprint
    pprint.pprint({k: r[k] for k in ("best_params", "leak_demo", "permutation_importance_gbm", "gbm_worst_days")}, width=150)
