"""Single entry point: `python main.py` reruns every implemented stage."""
from __future__ import annotations

import argparse
import random

import numpy as np

from src.config import load_config

STAGES = ["diagnose", "preprocess", "baselines", "features", "models", "feature_review", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Household electricity forecasting (AML313)")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    args = parser.parse_args()
    cfg = load_config()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    if args.stage in ("diagnose", "all"):
        from src.data.run_diagnosis import run
        res = run(cfg)
        b = res["baseline_dev"]
        print(f"[diagnose] {res['timestamps']['n_rows']} rows, {res['missing_pattern']['pct_rows_missing']} % missing; "
              f"dev naive-24 MAE {b['naive_24']['MAE_kwh']} kWh, naive-168 MAE {b['naive_168']['MAE_kwh']} kWh")
        print("[diagnose] outputs: reports/01_diagnosis_results.json, reports/figures/01_*.png")

    if args.stage in ("preprocess", "all"):
        from src.data.run_preprocessing import run as run_prep
        r = run_prep(cfg)
        print(f"[preprocess] min_valid={r['min_valid']} min; fill policy={r['policy']}; by split={r['by_split']}")
        print("[preprocess] outputs: data/processed/hourly.csv, data/processed/preprocessing_log.json, reports/figures/02_*.png")

    if args.stage in ("baselines", "all"):
        from src.evaluation.run_baselines import run as run_base
        r = run_base(cfg)
        print("[baselines] dev results:")
        print(r["table"][["MAE", "MAE_ci_low", "MAE_ci_high", "MASE", "coverage_80", "width_80"]].round(4).to_string())
        print("[baselines] outputs: reports/03_baselines_results.json, reports/03_forecasts_dev.csv, reports/figures/03_*.png")

    if args.stage in ("features", "all"):
        from src.features.run_features import run as run_feat
        r = run_feat(cfg)
        print(f"[features] {r['n_rows_total']} rows, {len(r['nan_per_feature'])} features, "
              f"{r['n_train_rows_usable']} usable train rows (first complete row {r['first_complete_row']})")
        print("[features] outputs: data/processed/features.csv, reports/04_features_results.json, reports/figures/04_*.png")

    if args.stage in ("models", "all"):
        from src.evaluation.run_models import run as run_mod
        r = run_mod(cfg)
        print("[models] dev results:")
        print(r["table"][["MAE", "MAE_ci_low", "MAE_ci_high", "MASE", "dMAE_vs_sarima", "coverage_80"]].round(4).to_string())
        print("[models] outputs: reports/05_models_results.json, reports/05_forecasts_dev.csv, reports/figures/05_*.png")

    if args.stage in ("feature_review", "all"):
        # descriptive analysis of the frozen final model's features (dev set only)
        from src.evaluation.run_feature_review import run as run_fr
        r = run_fr(cfg)
        print(f"[feature_review] significant single drops: {r['significant_single_drops']}; "
              f"never used: {r['never_used_by_model']}")
        print("[feature_review] outputs: reports/05b_feature_review.json, reports/figures/05b_feature_importance.png")

    if args.stage in ("test", "all"):
        # pre-registered single evaluation: refused if config `final` changed since the first run
        from src.evaluation.run_test import run as run_test
        r = run_test(cfg)
        print("[test] results (pre-registered, evaluated once):")
        print(r["table"][["MAE", "MAE_ci_low", "MAE_ci_high", "MASE", "dMAE_vs_sarima", "coverage_80"]].round(4).to_string())
        print("[test] outputs: reports/06_test_results.json, reports/06_forecasts_test.csv, reports/figures/06_*.png")


if __name__ == "__main__":
    main()
