# Feuille de route technique

## 1. Analyse des exigences : contraintes dures

| Source | Contrainte | Conséquence technique |
|---|---|---|
| Sujet G1 | Horizon **J+1, horaire** (24 valeurs) | Définir l'**origine de prévision** et ce qui est connu à cet instant |
| Sujet G1 | Baselines **naïf-24** et **SARIMA (m = 24)** | Les deux sont écrites et évaluées **avant** Ridge / GB |
| Sujet G1 | MAE (kWh), **MASE**, **couverture d'un intervalle à 80 %** | Chaque modèle doit produire un intervalle, pas seulement une valeur |
| Cadre commun | **Éliminatoire :** toute fuite temporelle | Découpage chronologique, `gap`, imputation et normalisation apprises **sur le passé seulement** |
| Cadre commun | Test-set utilisé **une seule fois** | Dates de découpage figées dans `config.yaml` **dès la phase 1** |
| Cadre commun | Ablation, préparation **journalisée** | Journal JSON de chaque étape de préparation ; ablation par groupes de variables |
| Livrables | Dépôt reproductible, **une seule commande** | `python main.py` ; versions figées (`requirements.lock.txt`) ; graines fixées |

## 2. Décision clé : origine de prévision et `lag_1`

On fixe l'**origine** $o$ = 00:00 du jour J : la prévision est émise à minuit, avec toutes les données jusqu'à 23:59 de J−1, pour les heures 00 → 23 de J (horizon $h = 1 \dots 24$). Une variante plus réaliste pour un contrat (origine à midi la veille) pourra être testée ensuite en changeant `forecast.issue_hour`.

Pour une heure cible $t = o + h - 1$ :
* `lag_24` = $y_{t-24}$ et `lag_168` = $y_{t-168}$ sont **toujours connus** à l'origine, car $t-24 \le o-1$ pour tout $h \le 24$ ✅
* `lag_1` **par rapport à la cible**, $y_{t-1}$, n'est **pas connu** dès que $h \ge 2$ ❌ : ce serait une **fuite** (démontrée en TP2).
* **Choix :** `lag_1` = **dernière valeur connue à l'origine**, $y_{o-1}$. Il est commun aux 24 heures du jour, et le modèle reçoit l'horizon $h$ (ou l'heure) pour apprendre que son utilité **diminue avec h** (ρ(1) = 0,72, mais ρ(6) ≈ 0). C'est la stratégie **directe** à un seul modèle.
* Alternatives écartées : la stratégie *récursive* (on réinjecte $\hat y_{t-1}$, et les erreurs s'accumulent sur 24 pas) et 24 modèles séparés (24 fois plus coûteux). On pourra les comparer si le temps le permet.

## 3. Phases

| Phase | Contenu | Sorties | État |
|---|---|---|---|
| **1. Diagnostic** | Schéma, horodatage, manquants, mécanisme, physique, agrégation, saisonnalités, stationnarité, **baseline écrite** | `reports/01_data_quality_report.md`, figures, tests | ✅ |
| **2. Préparation** | Grille horaire, règle d'heure partielle (≥ 50 min), remplissage **causal** (ffill puis profil 4 semaines) pour les variables seulement, masque « observé », journal | `data/processed/hourly.csv`, `data/processed/preprocessing_log.json`, `reports/02_preprocessing_report.md` | ✅ |
| **3. Baselines** | Pare-feu walk-forward testé ; naïf-24, naïf-168, profil 4 sem., **SARIMA (1,0,1)(0,1,1)₂₄** choisi en validation interne au train ; IC par bootstrap par blocs ; couverture par horizon | `reports/03_baselines_report.md`, `reports/03_*` | ✅ |
| **4. Variables** | 26 variables en 4 groupes (retards, fenêtres arrêtées à l'origine, calendrier cyclique, fériés FR) ; `lag_1` = $y_{o-1}$ ; test de causalité « on modifie le jour J » ; effet des fériés mesuré à saison égale | `reports/04_features_report.md`, `data/processed/features.csv` | ✅ |
| **5. Modèles** | Ridge et boosting via le même pare-feu ; réglage en validation interne ; intervalles conformes par heure ; ablation en 6 configurations ; démonstration de fuite étiquetée ; modèle retenu : **boosting config. 4** | `reports/05_models_report.md` | ✅ |
| **6. Test final** | Pré-enregistrement (section `final` + verrou SHA-256) ; évaluation unique sur 178 jours ; boosting −9,6 % vs SARIMA [IC exclut 0] ; couverture 76,8 % ; biais de volume +8,8 % rapporté | `reports/06_test_report.md` | ✅ |
| **7. Livrables** | Relance complète vérifiée (8 fichiers de résultats identiques, 14 min 41 s) ; 6 notebooks exécutés ; article de 5 pages ; soutenance de 12 diapositives avec notes ; 22 questions du jury ; dépôt Git | `docs/`, `notebooks/` | ✅ |

## 4. Protocole de validation (fixé maintenant)

* **Train** : 16/12/2006 → 30/11/2009 · **Dev** : 01/12/2009 → 31/05/2010 · **Test** : 01/06/2010 → 26/11/2010 (scellé).
* Sélection des modèles : **walk-forward** sur train + dev, avec des origines successives (blocs de quelques semaines), **gap ≥ 24 h**, et un réentraînement à chaque origine.
* Tout ce qui est « appris » (paramètres d'imputation, profils, normalisation, ordres SARIMA, hyperparamètres, seuil de l'intervalle) est appris **sur les données d'entraînement du pli** uniquement.
* Score : uniquement sur les heures **observées**, avec **les mêmes points pour tous les modèles**. Résultats donnés **par horizon h** et en moyenne.

## 5. Architecture

```
electricity-forecasting/
├── config.yaml            # dates de découpage, horizon, graine, chemins
├── main.py                # point d'entrée unique : python main.py [--stage ...]
├── requirements.txt       # dépendances directes figées ; requirements.lock.txt = environnement complet
├── src/
│   ├── config.py
│   ├── data/      loader.py · diagnosis.py · run_diagnosis.py · preprocessing.py · run_preprocessing.py
│   ├── features/  engineering.py · run_features.py
│   ├── models/    naive.py · sarima.py · feature_model.py · ridge.py · boosting.py
│   ├── evaluation/ metrics.py · validation.py · run_baselines.py · run_models.py
│   └── pipeline.py
├── tests/                 # pytest : chargement, diagnostic, préparation (causalité), puis variables
├── notebooks/             # lecture et illustration des résultats (pas de logique métier)
├── reports/  figures/     # sorties générées
└── docs/                  # feuille de route, article, soutenance
```

Principe : **toute la logique est dans `src/`** et testée. Les notebooks ne font qu'appeler `src/` et montrer les résultats, pour qu'il n'y ait pas de code divergent.
