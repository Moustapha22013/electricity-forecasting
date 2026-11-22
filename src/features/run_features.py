"""Phase 4 — build the feature matrix and analyse features on TRAIN only."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..data.diagnosis import to_json  # noqa: E402
from ..data.preprocessing import load_processed  # noqa: E402
from . import engineering as F  # noqa: E402


def run(cfg: dict) -> dict:
    rep, fig = Path(cfg["reports"]["dir"]), Path(cfg["reports"]["figures_dir"])
    data = load_processed(cfg["data"]["processed_dir"]).asfreq("h")
    X = F.build_features(data, cfg["forecast"]["issue_hour"])
    X.to_csv(Path(cfg["data"]["processed_dir"]) / "features.csv", float_format="%.6f")

    train_end = pd.Timestamp(cfg["split"]["train_end"])
    T = X.loc[:train_end]
    T = T[T["observed"] & T[F.ALL_FEATURES].notna().all(axis=1)]
    res: dict = {"n_rows_total": int(len(X)), "n_train_rows_usable": int(len(T)),
                 "first_complete_row": str(X[F.ALL_FEATURES].dropna().index[0]),
                 "nan_per_feature": X[F.ALL_FEATURES].isna().sum().astype(int).to_dict()}

    # 1. Spearman correlation with the target (monotone, robust to skew)
    num = [c for c in F.ALL_FEATURES if T[c].nunique() > 2]
    res["spearman_with_y"] = T[num].corrwith(T["y"], method="spearman").sort_values(ascending=False).round(3).to_dict()

    # 2. lag_1 usefulness by horizon: origin-relative (legal) vs target-relative (leaky)
    leak = F.leaky_lag_1(data).reindex(T.index)
    by_h = pd.DataFrame({
        "lag_1 (origine, légal)": T.groupby("horizon").apply(lambda g: g["lag_1"].corr(g["y"], method="spearman")),
        "y(t-1) (cible, FUITE)": T.assign(l=leak).groupby("horizon").apply(lambda g: g["l"].corr(g["y"], method="spearman")),
        "lag_24": T.groupby("horizon").apply(lambda g: g["lag_24"].corr(g["y"], method="spearman")),
        "profile_4w_median": T.groupby("horizon").apply(lambda g: g["profile_4w_median"].corr(g["y"], method="spearman")),
    })
    res["spearman_by_horizon"] = by_h.round(3).to_dict()

    # 3. Calendar effects (daily energy, train)
    daily = T["y"].resample("1D").sum(min_count=24).dropna()
    cal = F.calendar_features(daily.index)
    groups = {
        "jours ouvrés (lun-ven, non fériés)": (cal["dayofweek"] < 5) & (cal["is_holiday"] == 0),
        "samedis": (cal["dayofweek"] == 5) & (cal["is_holiday"] == 0),
        "dimanches": (cal["dayofweek"] == 6) & (cal["is_holiday"] == 0),
        "jours fériés": cal["is_holiday"] == 1,
        "veilles de férié": cal["is_day_before_holiday"] == 1,
        "24/12 et 31/12": cal["is_dec24_or_dec31"] == 1,
    }
    res["daily_energy_by_day_type"] = {k: {"n_days": int(m.sum()), "mean_kwh": round(float(daily[m].mean()), 2),
                                           "std_kwh": round(float(daily[m].std()), 2)} for k, m in groups.items()}

    # 4. Collinearity among lag / rolling features (relevant for Ridge)
    lr = F.FEATURE_GROUPS["lags"] + F.FEATURE_GROUPS["rolling"][:-1]
    corr = T[lr].corr(method="spearman")
    res["max_abs_corr_between_lag_features"] = round(float(corr.where(~np.eye(len(lr), dtype=bool)).abs().max().max()), 3)
    to_json(res, rep / "04_features_results.json")

    # ---- figures ----
    f1, ax = plt.subplots(figsize=(9, 4))
    for c, st in zip(by_h.columns, ["-o", "--x", "-s", "-^"]):
        ax.plot(by_h.index, by_h[c], st, ms=4, label=c)
    ax.set_xlabel("horizon h"); ax.set_ylabel("corrélation de Spearman avec y (train)")
    ax.set_title("Utilité d'un retard selon l'horizon : lag_1 légal vs y(t-1) (fuite)"); ax.legend(fontsize=8)
    f1.tight_layout(); f1.savefig(fig / "04_lag1_by_horizon.png", dpi=120); plt.close(f1)

    f2, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(lr))); ax.set_xticklabels(lr, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(lr))); ax.set_yticklabels(lr, fontsize=8); ax.grid(False)
    ax.set_title("Corrélations entre variables de retard (train)"); plt.colorbar(im)
    f2.tight_layout(); f2.savefig(fig / "04_lag_correlations.png", dpi=120); plt.close(f2)

    f3, ax = plt.subplots(figsize=(9, 3.6))
    d = res["daily_energy_by_day_type"]
    ax.bar(range(len(d)), [v["mean_kwh"] for v in d.values()], yerr=[v["std_kwh"] for v in d.values()], capsize=4)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{k}\n(n={v['n_days']})" for k, v in d.items()], fontsize=8)
    ax.set_ylabel("kWh / jour"); ax.set_title("Énergie journalière par type de jour (train, moyenne ± écart-type)")
    f3.tight_layout(); f3.savefig(fig / "04_day_types.png", dpi=120); plt.close(f3)
    return res | {"by_h": by_h}


if __name__ == "__main__":
    import pprint
    from ..config import load_config
    r = run(load_config())
    pd.set_option("display.width", 200)
    print(r.pop("by_h").round(3).to_string())
    pprint.pprint(r, width=150, sort_dicts=False)
