"""Run phase 2: select preprocessing rules on TRAIN, build and save the hourly table."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import preprocessing as P  # noqa: E402
from .diagnosis import to_json  # noqa: E402
from .loader import download_raw, load_raw  # noqa: E402


def _policy_from_scores(policy: dict[int, str]) -> tuple[int, str, str]:
    """Turn 'best method per gap length' into (short_max_h, short_method, long_method).

    short_method = best method for the shortest gaps; short_max_h = largest tested length
    up to which it stays best (contiguously); long_method = best method for the longest gaps.
    """
    lengths = sorted(policy)
    short = policy[lengths[0]]
    short_max = lengths[0]
    for L in lengths:
        if policy[L] != short:
            break
        short_max = L
    return short_max, short, policy[lengths[-1]]


def run(cfg: dict) -> dict:
    pc = cfg["preprocessing"]
    seed = cfg["seed"]
    figdir = Path(cfg["reports"]["figures_dir"]); figdir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(download_raw(cfg["data"]["url"], cfg["data"]["raw_dir"], cfg["data"]["raw_file"],
                                cfg["data"]["zip_sha256"]))
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    m = P.to_minute_grid(raw)
    m_train = m[m.index <= train_end]

    # --- A. validity threshold, selected on train ---
    h_tr_full = P.aggregate_hourly(m_train)
    y_tr_complete = h_tr_full["energy_kwh"].where(h_tr_full["n_valid"] == 60)
    ref_mae = float((y_tr_complete - y_tr_complete.shift(24)).abs().mean())        # naive-24 MAE on train
    err = P.partial_hour_error(m_train, pc["n_valid_grid"], n_hours=pc["n_hours_sim"], seed=seed)
    min_valid = P.select_min_valid(err, ref_mae, pc["max_ratio_to_naive_mae"])

    # --- B. fill method by gap length, selected on train (observed hours only) ---
    h_tr = P.apply_validity_rule(h_tr_full, min_valid)
    scores = P.evaluate_fill_methods(h_tr["energy_kwh"], pc["gap_lengths_h"], pc["n_blocks"], seed=seed)
    policy = P.select_fill_policy(scores)
    short_max, short_m, long_m = _policy_from_scores(policy)

    # --- C. build full hourly table with the selected rules (rules contain no future data) ---
    h, log = P.build_hourly(raw, min_valid, short_max, short_m, long_m)
    out = P.save_processed(h, log, cfg["data"]["processed_dir"], extra={
        "selection": {"reference_naive24_mae_train_kwh": ref_mae, "max_ratio": pc["max_ratio_to_naive_mae"],
                      "partial_hour_error": err.round(4).reset_index().to_dict(orient="records"),
                      "min_valid_minutes": min_valid,
                      "fill_scores_mae_kwh": scores.round(4).reset_index().to_dict(orient="records"),
                      "best_causal_method_by_gap": policy,
                      "policy": {"short_gap_max_hours": short_max, "short_method": short_m, "long_method": long_m}}})

    # --- summary by split (how much of each split is observed) ---
    dev_end = pd.Timestamp(cfg["split"]["dev_end"])
    split = pd.Series(pd.cut(h.index, [h.index.min() - pd.Timedelta("1h"), train_end, dev_end, h.index.max()],
                             labels=["train", "dev", "test"]), index=h.index)
    by_split = {s: {"hours": int((split == s).sum()), "observed": int(h["observed"][split == s].sum()),
                    "filled": int(((split == s) & ~h["observed"]).sum())} for s in ["train", "dev", "test"]}
    to_json({"by_split": by_split}, Path(cfg["reports"]["dir"]) / "02_split_summary.json")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(err.index, err["MAE_kwh"], marker="o", label="erreur d'estimation de l'heure")
    ax.axhline(pc["max_ratio_to_naive_mae"] * ref_mae, color="r", ls="--",
               label=f"{pc['max_ratio_to_naive_mae']:.0%} de la MAE naïf-24 (train)")
    ax.axvline(min_valid, color="k", ls=":"); ax.set_xlabel("minutes valides dans l'heure (bloc contigu)")
    ax.set_ylabel("MAE (kWh)"); ax.set_title("Heures partielles : seuil de validité (train)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "02_partial_hour_threshold.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in scores.columns.drop("n_blocks"):
        ax.plot(scores.index, scores[c], marker="o", ls="--" if "non causal" in c else "-", label=c)
    ax.set_xscale("log"); ax.set_xticks(scores.index); ax.set_xticklabels(scores.index)
    ax.set_xlabel("longueur du trou simulé (h)"); ax.set_ylabel("MAE sur les heures masquées (kWh)")
    ax.set_title("Méthodes de remplissage (train, trous simulés)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "02_fill_methods.png", dpi=120); plt.close(fig)

    z = slice("2009-06-08", "2009-06-20")
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.plot(h.loc[z].index, h.loc[z, "energy_filled"], color="C1", lw=1, label="valeur remplie (variables seulement)")
    ax.plot(h.loc[z].index, h.loc[z, "energy_kwh"], color="C0", lw=1, label="observé (cible)")
    ax.set_ylabel("kWh"); ax.legend(); ax.set_title("Exemple (train) : trou du 13-15/06/2009")
    fig.tight_layout(); fig.savefig(figdir / "02_fill_example.png", dpi=120); plt.close(fig)

    return {"output": str(out), "min_valid": min_valid, "ref_mae": ref_mae, "policy": (short_max, short_m, long_m),
            "partial_hour_error": err, "fill_scores": scores, "by_split": by_split, "log": log.steps}


if __name__ == "__main__":
    import pprint
    from ..config import load_config
    r = run(load_config())
    pd.set_option("display.width", 160)
    print(r["partial_hour_error"].round(4)); print(r["fill_scores"].round(4))
    pprint.pprint({k: v for k, v in r.items() if k not in ("partial_hour_error", "fill_scores")}, width=150, sort_dicts=False)
