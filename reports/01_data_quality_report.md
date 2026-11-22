# Phase 1 — Rapport de diagnostic des données

*Généré à partir de `python main.py --stage diagnose` · chiffres : `reports/01_diagnosis_results.json` · figures : `reports/figures/01_*.png`.*

**Règle suivie.** Les contrôles de **qualité** (format, trous, doublons, cohérence physique) portent sur tout le fichier : ils n'orientent pas le modèle. Les analyses qui **orientent la modélisation** (saisonnalité, ACF/PACF, stationnarité, baseline) portent **uniquement sur train + dev** (jusqu'au 31/05/2010). Le test-set (01/06/2010 → 26/11/2010) reste scellé.

## 1. Structure du fichier

| Élément | Constat |
|---|---|
| Fichier | `household_power_consumption.txt`, 133 Mo, séparateur `;`, fins de ligne CRLF |
| Lignes | **2 075 259** (+ en-tête) |
| Période | 16/12/2006 17:24 → 26/11/2010 21:02 (**1 441 jours**) |
| Colonnes | `Date` (jj/mm/aaaa), `Time`, 7 mesures numériques |
| Marqueurs de manquant | **deux** : `?` pour les 6 premières mesures, **champ vide** pour `Sub_metering_3` (même ligne) |

Unités (documentation UCI, vérifiées par les relations physiques, §4) : `Global_active_power` en kW (moyenne sur la minute), `Global_reactive_power` en kVAR, `Voltage` en V, `Global_intensity` en A, `Sub_metering_1/2/3` en **Wh consommés pendant la minute** (cuisine / buanderie / chauffe-eau + climatisation).

## 2. Cohérence de l'horodatage

| Contrôle | Résultat |
|---|---|
| Ordre chronologique | strictement croissant ✅ |
| Doublons de timestamp | **0** ✅ |
| Pas effectif | **1 min pour 100 % des écarts** ✅ |
| Timestamps absents | **0** : la grille minute est complète, les manquants sont des lignes présentes mais vides |
| Secondes ≠ 0 | 0 |

**Changement d'heure.** Les 8 nuits de changement d'heure (fin mars, fin octobre) ont exactement 60 lignes entre 02:00 et 03:00 : il n'y a ni heure manquante en mars, ni heure doublée en octobre. On a donc testé si l'horloge était **fixe** (sans changement d'heure) avec un comportement lié à l'heure légale : la **montée du matin en semaine** (réveil). Si l'horloge était fixe, cette montée sauterait de ±60 min au changement d'heure. On observe un décalage de **0 à 15 min seulement**, sur les 6 changements testés (2007-2009, train). **Conclusion :** l'index suit l'heure **locale** dans les comportements. Le traitement des nuits de changement d'heure par l'enregistreur reste inconnu (effet ≤ 1 h, 8 nuits sur 1 441). → On utilise un index **naïf en heure locale**, sans conversion de fuseau.

## 3. Valeurs manquantes

| Indicateur | Valeur |
|---|---|
| Lignes avec au moins un NaN | **25 979 (1,25 %)** |
| Lignes **entièrement** vides | 25 979 → **100 %** des manquants sont des lignes complètes |
| Nombre de trous (séquences contiguës) | **71** |

**Durée des trous**

| Durée | Trous | Minutes | Part des minutes manquantes |
|---|---|---|---|
| 1 min | 38 | 38 | 0,1 % |
| 2-10 min | 18 | 44 | 0,2 % |
| 11-60 min | 6 | 206 | 0,8 % |
| 1 h - 1 jour | 3 | 1 044 | 4,0 % |
| **> 1 jour** | **6** | **24 647** | **94,9 %** |

Les 6 trous les plus longs : 17-22/08/2010 (120 h), 25-28/09/2010 (87 h), 28-30/04/2007 (62 h), 13-15/06/2009 (55 h), 12-14/01/2010 (52 h), 20-21/03/2010 (34 h).

**Mécanisme probable (éléments de preuve, pas de certitude) :**
* **Panne d'acquisition, pas de capteur :** toutes les variables manquent ensemble. Un seul capteur défaillant laisserait les autres colonnes remplies.
* **Pas MCAR (pas complètement au hasard) :** le taux de manquants dépend du **calendrier**. Par année : 0,02 % (2006), 0,75 % (2007), 0,03 % (2008), 0,81 % (2009), **3,71 % (2010)**. Par jour : **2,3 % le samedi et le dimanche** contre 0,5-1,2 % en semaine. Par mois : **août 4,6 %**, septembre 3,0 %. En revanche, il dépend **peu de l'heure** (1,0 à 1,4 %).
* **Lien avec l'état du logement :** juste **après** un trou de plus de 10 min, la puissance médiane vaut **1,56 kW**, contre 0,60 kW en général et 0,64 kW juste avant (15 trous). Une reprise à forte consommation fait penser à une coupure de courant suivie du redémarrage des appareils (chauffe-eau, froid). Une coupure de courant signifie alors une **consommation réelle nulle**, et pas une valeur inconnue.
* **Lecture :** plutôt **MAR** (lié au calendrier observé), avec une part possible de **MNAR** (coupures, absences). On ne peut pas trancher : le compteur ne dit pas *pourquoi* il n'a pas enregistré.

**Conséquences pour la suite :**
1. Les trous de **plus d'un jour représentent 95 % des minutes manquantes** : aucune imputation n'y est fiable. On ne remplira pas ces jours pour **évaluer** : ces heures sont **exclues du score**, de la même façon pour tous les modèles.
2. **Le test-set contient les deux plus longs trous** (août et septembre 2010). La gestion des retards après un trou (lag_24 et lag_168 indisponibles) est donc **critique en test**.
3. Les petits trous (≤ 60 min, 288 minutes en tout) peuvent être comblés de façon **causale** (dernière valeur connue ou profil passé) sans effet mesurable. La politique exacte sera fixée et **journalisée** en phase 2.

## 4. Distributions, valeurs extrêmes et cohérence physique

| Variable | min | médiane | q99 % | max | % de zéros |
|---|---|---|---|---|---|
| Global_active_power (kW) | 0,076 | 0,602 | 4,85 | **11,12** | 0 |
| Global_reactive_power (kVAR) | 0 | 0,100 | 0,48 | 1,39 | 23,2 |
| Voltage (V) | 223,2 | 241,0 | 248,3 | 254,2 | 0 |
| Global_intensity (A) | 0,2 | 2,6 | 20,6 | 48,4 | 0 |
| Sub_metering_1 (Wh/min) | 0 | 0 | 38 | 88 | 90,6 |
| Sub_metering_2 (Wh/min) | 0 | 0 | 36 | 80 | 69,2 |
| Sub_metering_3 (Wh/min) | 0 | 1 | 20 | 31 | 41,1 |

**Contrôles physiques (vérification croisée des unités) :**
* Intensité recalculée $I \approx 1000\sqrt{P^2+Q^2}/V$ : **corrélation 0,9994** avec `Global_intensity`, erreur relative médiane 2 %. Les écarts > 10 % viennent des **faibles courants**, arrondis au pas de 0,2 A. Les unités kW / kVAR / V / A sont **confirmées**.
* Énergie de la minute $P \times 1000/60$ (Wh) comparée à la somme des sous-compteurs : les sous-compteurs couvrent **48,8 %** de l'énergie. La partie « autres usages » n'est négative que sur **33 minutes (0,002 %)**, ce qui est du bruit de mesure. Les sous-compteurs sont donc bien en **Wh par minute**.
* Tension hors de la plage 230 V ± 10 % : **0,003 %** du temps. Facteur de puissance médian : **0,993**.

**Valeurs extrêmes.** La distribution de la puissance est très **asymétrique à droite** (médiane 0,60 kW, moyenne 1,09 kW). Avec un score robuste (médiane / MAD), **4 %** des minutes ont |z| > 5 et le maximum est z = 17,9 (11,1 kW, soit 48 A). Ces valeurs sont **physiquement possibles** : four, plaques, chauffe-eau et lave-linge en même temps restent sous la puissance d'un abonnement domestique d'environ 12 kVA. **Décision : on ne supprime aucune valeur extrême.** Ce sont de vraies consommations, et justement celles que le contrat d'énergie doit prévoir. Aucune valeur négative ou bloquée à zéro n'a été trouvée.

## 5. Agrégation horaire : justification

* **Pourquoi l'heure ?** La question métier est horaire (24 valeurs pour le lendemain). La minute est très bruitée (allumages et extinctions d'appareils), et ce bruit n'est pas prévisible 24 h à l'avance. Enfin, l'heure réduit le volume par 60, ce qui rend SARIMA faisable.
* **Quelle fonction d'agrégation ?** La cible est l'**énergie de l'heure en kWh**. Comme `Global_active_power` est une puissance moyenne sur la minute en kW, $E_h = \sum_{m=1}^{60} P_m \cdot \tfrac{1}{60}$ kWh. Pour une heure complète, c'est **exactement la moyenne** des 60 valeurs (écart mesuré : 0,0). Pour une heure **incomplète**, la somme sous-estime l'énergie (elle compte les minutes manquantes comme 0). On utilise donc **la moyenne des minutes valides × 1 h**, ce qui est sans biais si les manquants sont représentatifs de l'heure. Test unitaire : `test_hourly_energy_mean_based_is_unbiased_for_partial_hour`.
* Pour les autres variables (si on les utilise un jour comme variables exogènes **passées**) : tension, intensité et puissance réactive → **moyenne** (grandeurs moyennes) ; sous-compteurs → **somme** en Wh (énergies).
* **Complétude horaire :** sur 34 589 heures, **34 085 sont complètes**, 83 sont partielles (médiane 58 minutes valides) et **421 sont vides**. Pour l'exploration, on a gardé les heures avec au moins 45 minutes valides. La règle définitive sera fixée en phase 2.

## 6. Saisonnalités, dépendance temporelle, stationnarité (train + dev)

Série horaire (30 295 h dont 229 manquantes) : moyenne **1,11 kWh**, médiane 0,82 kWh, écart-type 0,92 kWh, asymétrie +1,27.

* **Profil journalier :** nuit basse (≈ 0,4-0,5 kWh entre 1 h et 5 h), **pic du matin** vers 7 h en semaine, **pic du soir** entre 19 h et 21 h (≈ 1,8-2,2 kWh).
* **Profil hebdomadaire :** le **week-end est différent** (pas de pic à 7 h, consommation forte toute la journée, maximum le dimanche soir). C'est le motif d'un foyer qui **travaille en semaine**.
* **Saisonnalité annuelle :** environ 30-40 kWh/jour en hiver contre 15-20 kWh/jour en été, avec des **creux de vacances en août** (≈ 5 kWh/jour en août 2008). Ces creux sont des **changements de régime** que ni une baseline ni un modèle à retards courts ne prévoient au premier jour.
* **ACF :** ρ(1) = **0,72**, ρ(2) = 0,47, ρ(6) = 0,01, ρ(24) = **0,43**, ρ(48) = 0,40, ρ(144) = 0,39, ρ(168) = **0,46**, ρ(336) = 0,43. Il y a des pics à chaque multiple de 24 h, **légèrement plus forts à 168 h** (même jour de la semaine). Mais la saisonnalité est **faible** (environ 0,45) : la consommation d'un foyer dépend beaucoup de décisions humaines qu'on ne peut pas prévoir.
* **PACF** (plus long segment complet : 10 825 h) : φ(1) = **0,71**, φ(2) = −0,12, puis des pics vers **22-24 h** (φ(23) = 0,18, φ(24) = 0,10) et 47 h. Cela suggère une structure **AR(1) + saisonnière journalière**. Point de départ pour SARIMA : $(p,d,q) = (1\text{-}2,0,\cdot)$, $(P,D,Q)_{24}$ avec $P$ ou $Q = 1$, à valider par AIC et par validation glissante en phase 3.
* **Stationnarité** (série journalière, 1 250 jours) : l'**ADF** rejette la racine unitaire (stat −3,24, **p = 0,018**). Le **KPSS** ne rejette pas la stationnarité (stat 0,17, **p ≥ 0,10**). Les deux tests concordent : **pas de racine unitaire** → $d = 0$. La saisonnalité annuelle et les régimes de vacances restent des formes de **non-stationnarité déterministe**, à traiter par le calendrier et par des retards longs, pas par une différenciation.

**Point crucial pour la suite.** L'information la plus forte (ρ(1) = 0,72) disparaît en quelques heures (ρ(6) ≈ 0). Pour une prévision **J+1**, elle ne sert que pour les **premières heures de l'horizon**. Au-delà, il ne reste que la structure saisonnière (environ 0,45). **Hypothèse écrite avant tout modèle :** les modèles appris battront nettement la baseline sur les premières heures, mais seulement de **10-20 %** en moyenne sur 24 h. Des gains bien plus grands devront faire soupçonner une **fuite**.

## 7. Baseline de référence (écrite avant tout modèle)

Dev-set : 01/12/2009 → 31/05/2010. On évalue sur les **4 148 heures** où la cible, y(t−24) et y(t−168) sont toutes observées : ce sont les **mêmes points** pour les deux baselines. L'échelle de la MASE est la MAE du naïf-24 **sur le train** (0,663 kWh).

| Baseline | MAE (kWh) | MASE |
|---|---|---|
| Naïf saisonnier m = 24 : ŷ(t) = y(t−24) | **0,650** | 0,981 |
| Naïf saisonnier m = 168 : ŷ(t) = y(t−168) | **0,625** | 0,943 |

Le naïf-168 est un peu meilleur (−4 %), ce qui confirme ρ(168) > ρ(24). La baseline **exigée** par le sujet reste le naïf-24. **On la rapportera toujours**, et le naïf-168 servira de second repère plus exigeant.

## 8. Synthèse des risques identifiés

| Risque | Gravité | Mesure prévue |
|---|---|---|
| **`lag_1` incompatible avec un horizon de 24 h** si on le calcule par rapport à l'heure cible (fuite, éliminatoire) | 🔴 | Calculer `lag_1` par rapport à l'**origine de la prévision** (dernière heure connue) : stratégie *directe* avec l'horizon h en variable |
| Longs trous dans le **test-set** (août et septembre 2010) | 🔴 | Score uniquement sur les heures observées, identiques pour tous les modèles ; retards indisponibles → règle causale documentée |
| Imputation qui utilise le futur (interpolation, moyenne globale) | 🔴 | Dernière valeur connue ou profil **passé** uniquement ; test unitaire de causalité |
| Changements de régime (vacances) | 🟠 | Analyse des grosses erreurs ; indicateur de vacances scolaires (calendrier connu à l'avance), à évaluer par ablation |
| SARIMA saisonnier lent sur 30 000 h | 🟠 | Fenêtre d'apprentissage glissante limitée, mise à jour de l'état sans réestimation |
| Faible prévisibilité (ρ ≈ 0,45) | 🟡 | Attentes réalistes, intervalles de prévision étudiés sérieusement |
| Ambiguïté du changement d'heure | 🟢 | Index local naïf, effet ≤ 8 nuits |
