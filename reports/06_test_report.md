# Phase 6 — Évaluation finale unique sur le test-set

*Commande : `python main.py --stage test` · chiffres : `reports/06_test_results.json` · prévisions : `reports/06_forecasts_test.csv` · verrou : `reports/06_test_lock.json` · figures : `reports/figures/06_*.png`.*

## 1. Ce qui a été figé avant le test (pré-enregistrement)

Section `final` de `config.yaml`, écrite à partir des seuls résultats du dev-set. Son empreinte SHA-256 (`a0cf9ce2…`) a été enregistrée au **premier et unique** passage, le 22/11/2026 à 12:07. Toute modification de cette section fait désormais **refuser** l'évaluation (test `test_test_lock.py`).

| Élément | Valeur figée |
|---|---|
| Modèle retenu | gradient boosting, perte absolue, lr 0,05, 63 feuilles, 300 itérations |
| Variables | configuration 4 : retards + fenêtres + calendrier (20 variables) |
| Références | naïf-24 et SARIMA (1,0,1)(0,1,1)₂₄ (exigés) ; Ridge α = 0,1, configuration 4 |
| Protocole | origine 00:00, réapprentissage des **paramètres** tous les 28 jours sur tout le passé, gap 24 h, intervalle conforme par heure (91 jours) |
| Rapporté sans correction | biais, couverture |

**Une correction faite avant le test, et pourquoi.** La première version donnait à Ridge toutes les variables. Or la règle « à égalité, le plus simple » donnait aussi la configuration 4 pour Ridge (0,4775 contre 0,4785 sur le dev-set). La même règle a donc été appliquée aux deux modèles, **avant** tout accès au test.

**Hyperparamètres figés, paramètres réappris.** Pendant le test, les modèles sont réentraînés tous les 28 jours sur les données **déjà passées**, comme un système en service. Aucun hyperparamètre, aucune variable et aucun seuil n'est choisi sur le test.

## 2. Résultats (01/06 → 25/11/2010, 178 jours, 4 063 heures observées, les mêmes pour tous)

| Modèle | MAE (kWh) [IC 95 %] | MASE | Δ vs SARIMA [IC 95 %] | Δ vs naïf-24 [IC 95 %] | Couverture 80 % | Largeur | Score d'intervalle |
|---|---|---|---|---|---|---|---|
| naïf-24 | 0,490 [0,425 ; 0,547] | 0,739 | +0,068 | — | 0,840 | 1,68 | 2,41 |
| SARIMA | 0,422 [0,387 ; 0,457] | 0,636 | — | −0,068 [−0,102 ; −0,028] | 0,811 | 1,32 | 2,04 |
| Ridge | 0,416 [0,386 ; 0,448] | 0,627 | −0,006 [−0,018 ; +0,006] | −0,074 [−0,113 ; −0,027] | 0,749 | 1,21 | 1,98 |
| **Boosting (retenu)** | **0,381 [0,337 ; 0,422]** | **0,575** | **−0,041 [−0,058 ; −0,024]** | **−0,108 [−0,139 ; −0,078]** | 0,768 | **1,15** | **1,97** |

**Lecture :**
* **Le modèle retenu confirme son avantage** : −9,6 % de MAE par rapport à SARIMA et −22 % par rapport au naïf-24. Les deux IC excluent 0.
* **Les résultats sont cohérents avec le dev-set.** L'écart avec SARIMA était de −0,033 [−0,053 ; −0,015] sur le dev, il est de −0,041 [−0,058 ; −0,024] sur le test : les intervalles se recouvrent largement. Le gain reste dans la fourchette annoncée **avant** les modèles (5-10 % par rapport à SARIMA).
* **Ridge ne bat plus SARIMA** de façon significative (IC [−0,018 ; +0,006]). Son avantage sur le dev-set était déjà à la limite. On le rapporte tel quel.
* **Pourquoi les erreurs sont plus faibles que sur le dev-set (0,38 contre 0,46)** : le test couvre l'**été et l'automne**, avec moins de consommation (22,0 kWh/jour contre 29,2 sur le dev, qui couvrait l'hiver). L'erreur suit le niveau de consommation. La MASE baisse aussi, parce que son échelle (la MAE du naïf-24 sur le train) mélange toutes les saisons. Il ne faut **pas** en conclure que le modèle s'est amélioré : seule la **comparaison sur les mêmes heures** est valable.

## 3. Où le modèle gagne et où il perd

**Par horizon** (figure `06_test_by_horizon.png`) : le boosting est le meilleur, ou à égalité, pour **les 24 horizons**. Comme sur le dev-set, le gain le plus fort est à la **montée du matin** : h = 8 (7 h), 0,375 contre 0,543 pour SARIMA, soit −31 %. Viennent ensuite la nuit (h = 4-6 : 0,15 contre 0,19-0,20) et la fin de soirée (h = 22-23).

**Par mois :**

| Mois | naïf-24 | SARIMA | Ridge | Boosting |
|---|---|---|---|---|
| juin | 0,509 | 0,402 | 0,388 | **0,363** |
| juillet | 0,429 | 0,395 | 0,399 | **0,349** |
| **août** | 0,255 | 0,329 | 0,343 | **0,249** |
| septembre | 0,477 | 0,390 | 0,374 | **0,365** |
| octobre | 0,652 | 0,526 | 0,503 | **0,493** |
| novembre | 0,599 | 0,479 | 0,481 | **0,460** |

* Le boosting est le meilleur **chaque mois**.
* **Août (vacances)** : le logement consomme peu et de façon très régulière. Le **naïf-24** devient très bon (0,255), car hier ressemble exactement à aujourd'hui. **SARIMA et Ridge sont moins bons** (0,33-0,34) : ils continuent à prévoir une partie du profil habituel (réveil, soirée). Le boosting (0,249) s'adapte grâce aux fenêtres récentes (`roll_mean_24`, `same_hour_mean_7d`), qui captent le changement de **niveau**.
* **Octobre et novembre** sont les mois les plus difficiles pour tous : le chauffage reprend, la consommation monte et devient plus variable.

**Effet des longs trous (17-22/08 et 25-28/09).** 342 heures ont un historique de la dernière semaine **en partie rempli** :

| Modèle | MAE, historique complet | MAE, historique en partie rempli | Hausse |
|---|---|---|---|
| naïf-24 | 0,487 | 0,519 | +7 % |
| SARIMA | 0,418 | 0,459 | +10 % |
| Ridge | 0,414 | 0,432 | +4 % |
| **Boosting** | 0,375 | 0,445 | **+18 %** |

Le boosting est le **plus sensible aux trous** : il s'appuie beaucoup sur les fenêtres et les profils, donc sur des valeurs remplies. Il reste meilleur que SARIMA sur ces heures (0,445 contre 0,459), mais son avance fond. Ces heures suivent aussi souvent un **retour de vacances**, ce qui mélange deux effets. On ne peut pas les séparer avec ces données.

**Pires jours du boosting** : jeu. 04/11 (1,05 kWh/h), lun. 18/10 (0,92), **ven. 27/08** (0,91, retour de vacances, sous-estimé), dim. 21/11, sam. 30/10 et dim. 31/10 (week-end de la Toussaint), dim. 04/07. On retrouve des **week-ends** et des **changements de régime**, comme sur le dev-set.

**Jours fériés du test** (lecture descriptive, 4 jours) : 14/07, 15/08 (un dimanche), 01/11 et 11/11. Le boosting n'y est **pas meilleur** que les autres (par exemple 11/11 : 0,672 contre 0,585 pour SARIMA). C'est attendu : la configuration retenue ne contient pas les variables de fériés, et sur le dev-set elles n'apportaient rien de mesurable. **Avec 4 jours, on ne peut pas conclure** si elles auraient aidé ici. On ne le teste pas, car ce serait régler sur le test.

## 4. Intervalles à 80 %

| Modèle | Globale | Plage par heure | Lecture |
|---|---|---|---|
| naïf-24 | 0,840 | 0,75-0,91 | trop large : calibré sur 365 jours, qui incluent l'hiver plus variable |
| SARIMA | 0,811 | **0,66-0,98** | trop large la nuit (0,90-0,98), trop étroit en soirée (0,66-0,69 à 21-22 h), comme sur le dev-set |
| Ridge | 0,749 | 0,65-0,82 | trop étroit |
| **Boosting** | **0,768** | 0,67-0,83 | **un peu trop étroit**, surtout en milieu de journée (h = 11-15 : 0,67-0,75) |

* Le boosting a le **meilleur score d'intervalle** (1,97) et l'intervalle le **plus étroit** (1,15 kWh), mais sa couverture est **sous l'objectif** : 76,8 % au lieu de 80 %.
* **Cause** : l'intervalle conforme suppose que les erreurs futures ressemblent à celles des 91 jours de calibration. Quand la saison change, ce n'est plus vrai. Par exemple, la calibration du pli de juin vient de mars-mai. C'est la limite connue du conformal « split » quand les données ne sont pas échangeables dans le temps.
* Ce chiffre est **rapporté sans correction**, comme prévu. Piste pour un travail futur : conformal **adaptatif** (Gibbs & Candès, 2021), qui ajuste le quantile en continu selon les ratés récents.

## 5. Biais de volume (lecture métier)

| Modèle | Biais moyen par jour (réel − prévu) | En % de l'énergie journalière (22,0 kWh) |
|---|---|---|
| naïf-24 | +0,06 kWh | +0,3 % |
| SARIMA | −0,25 kWh | −1,1 % |
| Ridge | +0,19 kWh | +0,8 % |
| **Boosting** | **+1,93 kWh** | **+8,8 %** (sous-prévision) |

Le boosting **sous-prévoit le volume de 8,8 %** sur le test, contre 6,4 % sur le dev-set. C'est la conséquence attendue et **annoncée** de la perte absolue : elle vise la médiane, qui est sous la moyenne quand la distribution est asymétrique à droite. L'erreur horaire médiane est ≈ 0, ce qui en fait le meilleur modèle **heure par heure** (MAE). **Pour réserver un volume d'énergie, il ne faut pas l'utiliser tel quel** : il faudrait un modèle à perte quadratique, ou une correction de biais apprise **sur des données passées**. Comme prévu au pré-enregistrement, aucune correction n'est faite ici.

## 6. Résultat à citer (une phrase, avec son incertitude)

> Sur 178 jours de test (juin-novembre 2010, 4 063 heures), le gradient boosting à perte absolue sur 20 variables causales prévoit la consommation horaire du lendemain avec une MAE de **0,381 kWh [IC 95 % : 0,337-0,422]**, soit **−9,6 %** par rapport à SARIMA (Δ = −0,041 kWh [−0,058 ; −0,024]) et **−22 %** par rapport au naïf saisonnier exigé. Ses intervalles à 80 % couvrent **76,8 %** des heures, et il sous-prévoit le volume journalier de **8,8 %**.

## 7. Ce qu'on n'a plus le droit de faire

Changer un modèle, une variable, un hyperparamètre, l'intervalle ou la correction de biais **en fonction de ces résultats**, puis réévaluer sur juin-novembre 2010. Toute amélioration future (variables de fériés, vacances scolaires, conformal adaptatif, correction de volume) doit être évaluée sur des **données nouvelles**, ou présentée clairement comme **exploratoire**.
