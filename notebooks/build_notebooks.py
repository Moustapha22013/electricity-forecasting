"""Generate (and optionally execute) the 6 reading notebooks.

The notebooks contain NO modelling logic: they call `src/` and display the outputs written by
`python main.py`, so there is a single source of truth.

    python notebooks/build_notebooks.py            # write the .ipynb files
    python notebooks/build_notebooks.py --execute  # write and execute them (after main.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
from src.config import load_config
cfg = load_config()
R = ROOT / "reports"; FIG = R / "figures"
def load(name): return json.loads((R / name).read_text())
def fig(name): display(Image(filename=str(FIG / name)))"""


def md(t: str):
    return nbf.v4.new_markdown_cell(t)


def code(t: str):
    return nbf.v4.new_code_cell(t)


NOTEBOOKS: dict[str, list] = {
    "01_data_exploration.ipynb": [
        md("# 01 — Exploration et diagnostic des données\nRapport complet : `reports/01_data_quality_report.md`. "
           "Les chiffres viennent de `python main.py --stage diagnose`."),
        code(SETUP),
        md("## Le fichier brut (chargé par `src.data.loader`)"),
        code("from src.data.loader import load_raw\nraw = load_raw(Path(cfg['data']['raw_dir']) / cfg['data']['raw_file'])\n"
             "print(raw.shape); display(raw.head()); display(raw.dtypes.to_frame('dtype'))"),
        md("## Horodatage et valeurs manquantes"),
        code("d = load('01_diagnosis_results.json')\ndisplay(pd.Series(d['timestamps']).to_frame('valeur'))\n"
             "display(pd.Series(d['missing_pattern']).drop('nan_per_column').to_frame('valeur'))\n"
             "display(pd.DataFrame(d['missing_runs']['by_length']).T)"),
        code("fig('01_daily_energy_gaps.png'); fig('01_missing_runs.png')"),
        md("## Distributions et cohérence physique"),
        code("display(pd.DataFrame(d['describe']).T.round(3)); display(pd.Series(d['physics']).to_frame('valeur'))\n"
             "fig('01_distributions.png')"),
        md("## Saisonnalités, ACF/PACF, stationnarité (train + dev seulement)"),
        code("fig('01_profile_hour_weekday.png'); fig('01_monthly.png'); fig('01_acf_pacf.png')\n"
             "display(pd.Series(d['acf']).to_frame('ACF').T); display(pd.Series(d['stationarity_daily']).to_frame('valeur'))"),
    ],
    "02_preprocessing.ipynb": [
        md("# 02 — Préparation des données\nRapport : `reports/02_preprocessing_report.md`. La cible n'est jamais remplie ; "
           "`energy_filled` sert uniquement aux variables."),
        code(SETUP),
        code("from src.data.preprocessing import load_processed\nh = load_processed(cfg['data']['processed_dir'])\n"
             "display(h.head()); display(h['fill_method'].value_counts().to_frame('heures'))"),
        md("## Journal des étapes"),
        code("log = json.loads((Path(cfg['data']['processed_dir']) / 'preprocessing_log.json').read_text())\n"
             "for s in log['steps']: print(s)"),
        md("## Choix n°1 : seuil de validité d'une heure (train)"),
        code("sel = log['selection']; display(pd.DataFrame(sel['partial_hour_error']).set_index('n_valid').round(4))\n"
             "print('seuil retenu :', sel['min_valid_minutes'], 'minutes'); fig('02_partial_hour_threshold.png')"),
        md("## Choix n°2 : méthode de remplissage causal (train, trous simulés)"),
        code("display(pd.DataFrame(sel['fill_scores_mae_kwh']).set_index('gap_h').round(3)); print(sel['policy'])\n"
             "fig('02_fill_methods.png'); fig('02_fill_example.png')"),
    ],
    "03_baseline_models.ipynb": [
        md("# 03 — Baselines et protocole d'évaluation\nRapport : `reports/03_baselines_report.md`."),
        code(SETUP),
        md("## Démonstration : une prévision du naïf-24 passe par le pare-feu"),
        code("from src.data.preprocessing import load_processed\nfrom src.models.naive import SeasonalNaive\n"
             "data = load_processed(cfg['data']['processed_dir']).asfreq('h')\no = pd.Timestamp('2010-02-01')\n"
             "hist = data.loc[:o - pd.Timedelta('1h')]   # strictly before the origin\n"
             "m = SeasonalNaive(24).fit(hist); display(m.predict(hist, o).head(6))"),
        md("## Sélection (validation interne au train)"),
        code("b = load('03_baselines_results.json')\ndisplay(pd.DataFrame(b['profile_selection']).round(4))\n"
             "display(pd.DataFrame(b['sarima_selection']).sort_values('MAE').head(8).round(4))\n"
             "print('SARIMA retenu :', b['sarima_best']); print('Ljung-Box :', b['sarima_residual_diagnostics'])"),
        md("## Résultats sur le dev-set"),
        code("t = pd.DataFrame(b['dev_table']).T\ndisplay(t[['MAE','MAE_ci_low','MAE_ci_high','MASE','dMAE_vs_naive24','coverage_80','width_80']].round(4))\n"
             "fig('03_mae_coverage_by_horizon.png'); fig('03_example_week.png'); fig('03_mae_by_fold.png')"),
    ],
    "04_feature_engineering.ipynb": [
        md("# 04 — Variables causales\nRapport : `reports/04_features_report.md`."),
        code(SETUP),
        code("from src.features.engineering import FEATURE_GROUPS, ALL_FEATURES, build_features\n"
             "for g, f in FEATURE_GROUPS.items(): print(f'{g:<9} ({len(f)}):', f)"),
        md("## Démonstration de causalité : on modifie tout le jour J, ses variables ne changent pas"),
        code("from src.data.preprocessing import load_processed\ndata = load_processed(cfg['data']['processed_dir']).asfreq('h').loc[:'2010-03-01']\n"
             "D = pd.Timestamp('2010-02-10'); X1 = build_features(data)[ALL_FEATURES]\n"
             "d2 = data.copy(); m = d2.index >= D\n"
             "d2.loc[m, ['energy_kwh', 'energy_filled']] = d2.loc[m, ['energy_kwh', 'energy_filled']] * 7 + 3\n"
             "X2 = build_features(d2)[ALL_FEATURES]; day = (X1.index >= D) & (X1.index < D + pd.Timedelta('1D'))\n"
             "print('variables du jour J identiques :', X1[day].equals(X2[day]))"),
        md("## Analyse sur le train"),
        code("f = load('04_features_results.json'); display(pd.Series(f['spearman_with_y']).to_frame('Spearman avec y'))\n"
             "display(pd.DataFrame(f['daily_energy_by_day_type']).T)\n"
             "fig('04_lag1_by_horizon.png'); fig('04_day_types.png'); fig('04_lag_correlations.png')"),
    ],
    "05_model_training.ipynb": [
        md("# 05 — Ridge, gradient boosting, ablation\nRapport : `reports/05_models_report.md`."),
        code(SETUP),
        md("## Réglage (validation interne au train)"),
        code("r = load('05_models_results.json'); display(pd.DataFrame(r['tuning']).round(4)); print('retenus :', r['best_params'])"),
        md("## Dev-set : comparaison appariée avec SARIMA"),
        code("t = pd.DataFrame(r['dev_table']).T\ndisplay(t[['MAE','MAE_ci_low','MAE_ci_high','MASE','dMAE_vs_sarima','ci_lo_sarima','ci_hi_sarima','coverage_80','interval_score']].round(4))\n"
             "fig('05_mae_coverage_by_horizon.png'); fig('05_example_week.png')"),
        md("## Ablation par groupes et démonstration de fuite"),
        code("display(pd.DataFrame(r['ablation']).round(4)); print('fuite :', r['leak_demo']); fig('05_ablation.png')"),
        md("## Revue des 20 variables du modèle retenu (descriptive)"),
        code("v = load('05b_feature_review.json')\ndisplay(pd.DataFrame(v['single']).T.round(4)); display(pd.DataFrame(v['groups']).T.round(4))\n"
             "display(pd.DataFrame(v['reduced']).T.round(4)); fig('05b_feature_importance.png')"),
    ],
    "06_evaluation.ipynb": [
        md("# 06 — Évaluation finale unique sur le test-set\nRapport : `reports/06_test_report.md`. "
           "Pré-enregistrement : section `final` de `config.yaml`, verrouillée par SHA-256."),
        code(SETUP),
        code("lock = load('06_test_lock.json'); print({k: lock[k] for k in ('final_sha256', 'first_run')})\n"
             "from src.evaluation.run_test import _section_hash\nprint('section final inchangée :', _section_hash(cfg['final']) == lock['final_sha256'])"),
        code("t6 = load('06_test_results.json'); t = pd.DataFrame(t6['test_table']).T\n"
             "display(t[['MAE','MAE_ci_low','MAE_ci_high','MASE','dMAE_vs_sarima','ci_lo_sarima','ci_hi_sarima','coverage_80','interval_score','daily_volume_bias_kwh']].round(4))"),
        code("display(pd.DataFrame(t6['mae_by_month']).round(3)); display(pd.DataFrame(t6['gap_effect']).T.round(4))\n"
             "fig('06_test_by_horizon.png'); fig('06_test_by_month.png'); fig('06_test_august.png')"),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
        path = HERE / name
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, path)
        print("written", path.name, "(executed)" if args.execute else "")


if __name__ == "__main__":
    main()
