"""Descriptive review of the 20 features of the PRE-REGISTERED final model (dev set only).

Not a feature-selection step: the final model is frozen (phase 6). This answers
"does each feature actually matter, and how many are needed?" for the article.

* Train: train period up to `gap_hours` + 1 h before the dev start; one fit (all variants
  are treated identically, so the comparison is fair). Evaluate: observed dev hours.
* The feature matrix is precomputed: allowed because every feature is causal by
  construction (tests/test_features.py). The test period is never read.
* Three importance measures:
    permutation   -> does the model RELY on the feature? (underestimates redundant ones)
    drop-column   -> does it add information BEYOND the others? (refit without it)
    drop-group    -> is a type of INFORMATION indispensable? (refit without the group)
* Uncertainty: paired per-hour differences, week-block bootstrap 95 % CI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from ..data.diagnosis import to_json  # noqa: E402
from ..data.preprocessing import load_processed  # noqa: E402
from ..features.engineering import build_features  # noqa: E402
from .metrics import block_bootstrap_ci  # noqa: E402
from .run_models import ABLATION  # noqa: E402

GROUPS: dict[str, list[str]] = {
    "heure (hour + sin/cos)": ["hour", "hour_sin", "hour_cos"],
    "jour (dayofweek, is_weekend, dow sin/cos)": ["dayofweek", "is_weekend", "dow_sin", "dow_cos"],
    "saison (doy sin/cos)": ["doy_sin", "doy_cos"],
    "retards 24/48/168/336": ["lag_24", "lag_48", "lag_168", "lag_336"],
    "profils (same_hour_7d + profile_4w)": ["same_hour_mean_7d", "profile_4w_median"],
    "fenêtres de niveau (roll_mean_24/168, std_24)": ["roll_mean_24", "roll_mean_168", "roll_std_24"],
}
REDUCED: dict[str, list[str]] = {
    "8 variables (une par information)": ["lag_1", "lag_168", "profile_4w_median", "same_hour_mean_7d",
                                          "roll_mean_24", "hour", "dayofweek", "doy_cos"],
    "5 variables": ["lag_1", "profile_4w_median", "same_hour_mean_7d", "hour", "dayofweek"],
    "3 variables (profils + heure)": ["profile_4w_median", "same_hour_mean_7d", "hour"],
}


def run(cfg: dict) -> dict:
    rep, fig = Path(cfg["reports"]["dir"]), Path(cfg["reports"]["figures_dir"])
    P, seed, gap = cfg["final"]["gbm_params"], cfg["seed"], cfg["baselines"]["gap_hours"]
    feats = ABLATION[cfg["final"]["feature_config"]]
    train_end, dev_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["dev_end"])

    data = load_processed(cfg["data"]["processed_dir"]).asfreq("h").loc[:dev_end]   # test never loaded
    X = build_features(data)
    ok = X["observed"].astype(bool) & X[feats].notna().all(axis=1)
    tr = ok & (X.index <= train_end - pd.Timedelta(hours=gap + 1))
    dv = ok & (X.index > train_end)
    y_dv = X.loc[dv, "y"]

    def fit(cols: list[str]) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss=P["loss"], learning_rate=P["learning_rate"], max_iter=P["max_iter"], max_leaf_nodes=P["max_leaf_nodes"],
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=seed,
        ).fit(X.loc[tr, cols], X.loc[tr, "y"])

    def abs_err(cols: list[str]) -> pd.Series:
        return (y_dv - fit(cols).predict(X.loc[dv, cols])).abs()

    def vs_ref(e: pd.Series) -> dict[str, float]:
        d = e - e_ref
        lo, hi = block_bootstrap_ci(d)
        return {"MAE": float(e.mean()), "dMAE": float(d.mean()), "ci_lo": lo, "ci_hi": hi}

    ref = fit(feats)
    e_ref = (y_dv - ref.predict(X.loc[dv, feats])).abs()
    print(f"[feature_review] reference MAE (dev, 20 features) = {e_ref.mean():.4f}", flush=True)

    pi = permutation_importance(ref, X.loc[dv, feats], y_dv, n_repeats=10, random_state=seed,
                                scoring="neg_mean_absolute_error")
    single = pd.DataFrame({f: {"permutation": float(pi.importances_mean[i]),
                               **{f"drop_{k}": v for k, v in vs_ref(abs_err([c for c in feats if c != f])).items()}}
                           for i, f in enumerate(feats)}).T.sort_values("drop_dMAE", ascending=False)
    groups = pd.DataFrame({g: vs_ref(abs_err([c for c in feats if c not in gs])) for g, gs in GROUPS.items()}).T
    reduced = pd.DataFrame({n: vs_ref(abs_err(cols)) | {"n_features": len(cols)} for n, cols in REDUCED.items()}).T

    out = {"reference_MAE": float(e_ref.mean()), "n_dev_hours": int(len(y_dv)), "features": feats,
           "single": single.to_dict(orient="index"), "groups": groups.to_dict(orient="index"),
           "reduced": reduced.to_dict(orient="index"),
           "significant_single_drops": single.index[single["drop_ci_lo"] > 0].tolist(),
           "never_used_by_model": single.index[(single["permutation"].abs() < 1e-9) & (single["drop_dMAE"].abs() < 1e-9)].tolist()}
    to_json(out, rep / "05b_feature_review.json")

    f1, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    s = single.sort_values("permutation")
    yy = np.arange(len(s))
    ax[0].barh(yy, s["permutation"], color="C0"); ax[0].set_yticks(yy); ax[0].set_yticklabels(s.index, fontsize=8)
    ax[0].set_title("Permutation : le modèle s'appuie-t-il dessus ?"); ax[0].set_xlabel("hausse de MAE (kWh)")
    ax[1].barh(yy, s["drop_dMAE"], color="C1",
               xerr=[s["drop_dMAE"] - s["drop_ci_lo"], s["drop_ci_hi"] - s["drop_dMAE"]], capsize=2)
    ax[1].axvline(0, color="k", lw=0.8); ax[1].set_title("Retrait : apporte-t-elle plus que les autres ? (IC 95 %)")
    ax[1].set_xlabel("hausse de MAE après réentraînement sans la variable (kWh)")
    f1.tight_layout(); f1.savefig(fig / "05b_feature_importance.png", dpi=120); plt.close(f1)
    return out | {"single_df": single, "groups_df": groups, "reduced_df": reduced}


if __name__ == "__main__":
    from ..config import load_config
    r = run(load_config())
    pd.set_option("display.width", 200)
    for k in ("single_df", "groups_df", "reduced_df"):
        print(r[k].round(4).to_string())
    print("significant:", r["significant_single_drops"], "| never used:", r["never_used_by_model"])
