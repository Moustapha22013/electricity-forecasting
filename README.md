# Prévoir la consommation électrique d'un logement (AML313 · G1)

Prévision **J+1, heure par heure** de l'énergie consommée par un foyer (UCI *Individual Household Electric Power Consumption*, 2006-2010, pas de 1 min), avec un protocole **sans fuite temporelle**, vérifié par des tests et pré-enregistré.

**Auteurs :** Mohamedhen Vall Khaled (21026) · El Moustapha El Moustapha (22013) — Master SupNum.

## Résultat principal (test, 01/06 → 25/11/2010, évalué une seule fois)

| Modèle | MAE (kWh) [IC 95 %] | Δ vs SARIMA [IC 95 %] | Couverture 80 % |
|---|---|---|---|
| Naïf-24 (exigé) | 0,490 [0,425 ; 0,547] | +0,068 | 0,840 |
| SARIMA (1,0,1)(0,1,1)₂₄ (exigé) | 0,422 [0,387 ; 0,457] | — | 0,811 |
| Ridge | 0,416 [0,386 ; 0,448] | −0,006 (non significatif) | 0,749 |
| **Gradient boosting (retenu)** | **0,381 [0,337 ; 0,422]** | **−0,041 [−0,058 ; −0,024]** | 0,768 |

Limites rapportées sans correction : couverture de 76,8 % au lieu de 80 %, sous-prévision du volume journalier de 8,8 % (perte absolue = médiane), +18 % d'erreur après les longs trous.

## Lancer le projet (une seule commande)

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.lock.txt   # environnement exact (Python 3.14)
./.venv/bin/python main.py                          # ≈ 15 min : télécharge les données (SHA-256 vérifié) puis lance les 7 étapes
./.venv/bin/python -m pytest -q                     # 35 tests unitaires
```

Une étape seule : `python main.py --stage <nom>`.

| Étape | Contenu | Sorties |
|---|---|---|
| `diagnose` | phase 1 : diagnostic du fichier brut | `reports/01_*` |
| `preprocess` | phase 2 : table horaire, remplissage causal, journal | `data/processed/`, `reports/02_*` |
| `baselines` | phase 3 : naïfs, profil, SARIMA (≈ 10 min) | `reports/03_*` |
| `features` | phase 4 : 26 variables causales | `data/processed/features.csv`, `reports/04_*` |
| `models` | phase 5 : Ridge, boosting, ablation, démonstration de fuite (≈ 6 min) | `reports/05_*` |
| `feature_review` | revue descriptive des 20 variables retenues (dev) | `reports/05b_*` |
| `test` | phase 6 : évaluation **pré-enregistrée**, refusée si la section `final` de `config.yaml` a changé | `reports/06_*` |

Tous les réglages sont dans `config.yaml`, validés au chargement (`src/config.py`). Graine fixée : `seed: 2025`.

## Livrables

* **Article** (5 pages) : [`docs/article.pdf`](docs/article.pdf), source [`docs/article.md`](docs/article.md)
* **Soutenance** (12 diapositives, 10 min, notes de l'orateur) : [`docs/soutenance.pptx`](docs/soutenance.pptx)
* **Questions probables du jury** : [`docs/questions_jury.md`](docs/questions_jury.md)

Reconstruire l'article et les diapositives : `pip install -r requirements-docs.txt && python docs/build_docs.py`.

## Structure

```
config.yaml          réglages + pré-enregistrement du test (section `final`)
main.py              point d'entrée unique
src/data/            chargement, diagnostic, préparation causale
src/features/        variables causales
src/models/          naïfs, SARIMA, modèle à variables générique, Ridge, boosting
src/evaluation/      métriques, validation glissante (pare-feu), étapes 3, 5, 6
tests/               35 tests (causalité, pare-feu, verrou, métriques)
notebooks/           6 notebooks de lecture (appellent src/, aucun calcul propre)
reports/             rapports détaillés par phase, chiffres (JSON), figures
docs/                feuille de route, article, soutenance, questions du jury
```

## Données

* Source : https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
* Archive vérifiée par SHA-256 (`config.yaml`). Les données ne sont **pas** versionnées : `main.py` les télécharge.
* Découpage chronologique figé avant toute modélisation : train → 30/11/2009 · dev → 31/05/2010 · test → 26/11/2010.

## Rapports détaillés

* [Feuille de route et décisions](docs/ROADMAP.md)
* [Phase 1 : diagnostic des données](reports/01_data_quality_report.md)
* [Phase 2 : préparation des données](reports/02_preprocessing_report.md)
* [Phase 3 : baselines et protocole d'évaluation](reports/03_baselines_report.md)
* [Phase 4 : variables](reports/04_features_report.md)
* [Phase 5 : Ridge, boosting, ablation, revue des variables](reports/05_models_report.md)
* [Phase 6 : évaluation finale sur le test](reports/06_test_report.md)
