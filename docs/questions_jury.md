# Questions probables du jury — réponses courtes

## Protocole et fuites

**1. Pourquoi ne pas avoir utilisé un KFold ou un découpage aléatoire ?**
Un découpage aléatoire mettrait des heures du futur dans l'entraînement et entourerait chaque heure de test par ses voisines (ρ(1) = 0,72) : le score serait faussement bon. Nous utilisons une validation glissante chronologique avec un gap de 24 h. C'est une règle éliminatoire du sujet.

**2. Comment êtes-vous sûrs qu'il n'y a pas de fuite ?**
Ce ne sont pas des promesses, mais des **tests** :
- un « modèle espion » vérifie qu'aucun modèle ne voit de donnée à l'origine ou après ;
- on modifie tout le jour J et les variables de J ne changent pas ;
- modifier le futur ne change aucune valeur remplie ;
- un naïf avec m < 24 est refusé.

Il y a aussi un contrôle de vraisemblance : le gain obtenu (−9,6 %) est dans la fourchette écrite **avant** les modèles. Une fuite volontaire donnait −24 %.

**3. Le sujet demande lag_1. Pourquoi ne pas l'avoir pris comme y(t−1) ?**
À 24 h d'horizon, y(t−1) n'est connu que pour la première heure. Pour les autres heures, ce serait une fuite. Notre `lag_1` est la dernière valeur connue à minuit. Sa corrélation avec la cible passe de 0,65 (h = 1) à 0,07 (h = 8), alors que la version qui fuit reste à 0,6-0,8.

**4. Vous réentraînez pendant le test. N'est-ce pas utiliser le test ?**
Non. Les **hyperparamètres** sont figés (pré-enregistrés). Seuls les **paramètres** sont réappris tous les 28 jours, et uniquement sur des données **déjà passées**, avec un gap de 24 h. C'est ce que ferait un système en service.

**5. À quoi sert le verrou SHA-256 ?**
Il empêche de « regarder le test plusieurs fois ». L'empreinte de la section `final` est enregistrée au premier passage ; si la section change, le script refuse l'évaluation. Une relance identique est autorisée, car elle redonne les mêmes chiffres.

**6. Vous avez remplacé des valeurs manquantes. Ce n'est pas une fuite ?**
Seulement pour les **variables**, et de façon **causale** : la dernière valeur, puis la médiane des 4 semaines **passées**. La **cible n'est jamais remplie** : on ne note jamais un modèle sur une valeur inventée. L'interpolation linéaire, qui lit le futur, est interdite (un test le démontre).

## Données

**7. Pourquoi agréger à l'heure ?**
La question métier est horaire. La minute est trop bruitée (allumages d'appareils) pour être prévue 24 h à l'avance. L'agrégation divise aussi le volume par 60, ce qui rend SARIMA faisable.

**8. Pourquoi la moyenne et pas la somme des minutes ?**
P est une puissance moyenne sur la minute, donc E = moyenne × 1 h. Pour une heure incomplète, la somme compterait les minutes manquantes comme 0. La moyenne est sans biais si les minutes manquantes sont représentatives de l'heure.

**9. Les manquants sont-ils MCAR ?**
Non : 3,7 % en 2010 contre < 0,8 % avant, plus le week-end et en août, et une consommation forte juste après un trou (1,56 kW contre 0,60). C'est plutôt MAR, avec une part possible de MNAR (coupures). On ne peut pas trancher : le compteur ne dit pas *pourquoi* il n'a pas enregistré.

**10. Et les changements d'heure ?**
Il n'y a ni heure manquante ni heure doublée, mais la montée du matin ne saute pas de 60 min : l'horloge suit l'heure légale. Effet possible : moins d'une heure sur 8 nuits.

## Modèles

**11. Pourquoi SARIMA (0,1,1)₂₄ et pas un AR saisonnier ?**
Testé sur 24 candidats. Un AR saisonnier stationnaire ramène la prévision vers la moyenne globale. La différenciation saisonnière s'adapte au niveau de la saison (0,467 contre 0,525-0,660). (0,1,1)₂₄ est un lissage exponentiel du profil journalier.

**12. Pourquoi la perte absolue pour le boosting ?**
La MAE est minimisée par la **médiane** conditionnelle, et c'est ce que vise la perte absolue. Elle gagne en validation interne (0,405 contre 0,427).

**13. Mais alors le volume est sous-estimé ?**
Oui, de 8,8 % sur le test, et nous le disons. La médiane est sous la moyenne quand la distribution est asymétrique. **La métrique doit suivre la décision** : pour le profil horaire, la MAE (perte absolue) ; pour réserver un volume, la perte quadratique ou une correction de biais.

**14. Pourquoi le boosting bat-il SARIMA ?**
L'ablation le montre : avec les retards seuls, il ne fait pas mieux que SARIMA (0,500). Le gain vient du **calendrier**, c'est-à-dire de l'heure combinée au type de jour. À 7 h, l'erreur passe de 0,543 à 0,375 sur le test, car le réveil dépend du jour de la semaine, ce que SARIMA ne sait pas.

**15. Pourquoi 20 variables si 8 suffisent ?**
La sélection a été faite par **groupes**, avec une règle fixée à l'avance. L'analyse variable par variable a été faite **après** le test, à titre descriptif : on ne peut donc plus changer le modèle. Les variables redondantes ne nuisent pas, car le boosting les ignore.

**16. Pourquoi pas de deep learning (LSTM) ?**
Avec environ 25 000 heures d'entraînement et une saisonnalité faible, un modèle à variables bien construites est une référence solide et interprétable. Le sujet demande de préférer le simple quand il est compétitif. Un LSTM serait une piste, à comparer avec **le même protocole**.

## Résultats et incertitude

**17. Comment sont calculés les intervalles de confiance ?**
Par bootstrap par **blocs d'une semaine**, sur les différences d'erreur **appariées** (mêmes heures). Rééchantillonner heure par heure donnerait des IC trop étroits, car les erreurs sont autocorrélées.

**18. Pourquoi les erreurs sont-elles plus faibles sur le test que sur le dev ?**
Le test couvre l'été et l'automne (22 kWh/jour), le dev couvre l'hiver (29 kWh/jour), et l'erreur suit le niveau de consommation. Seule la comparaison **sur les mêmes heures** est valable : boosting −0,041 contre SARIMA sur le test, −0,033 sur le dev, avec des IC qui se recouvrent.

**19. Pourquoi la couverture fait-elle 76,8 % et pas 80 % ?**
Le conformal « split » suppose que les erreurs futures ressemblent à celles des 91 jours de calibration. Quand la saison change, ce n'est plus vrai. Nous le rapportons sans correction. Piste : conformal adaptatif (Gibbs & Candès, 2021). Le SARIMA à 81 % est trompeur : 90-98 % la nuit et 66 % en soirée.

**20. Ridge battait SARIMA sur le dev, plus sur le test. Comment l'expliquer ?**
L'avantage sur le dev était à la limite (IC [−0,026 ; −0,001]). Sur le test, l'IC contient 0. C'est exactement pourquoi on rapporte des IC : un petit gain sur une seule période n'est pas fiable.

## Limites et suite

**21. Quelles sont les principales limites ?**
- couverture sous 80 % ;
- biais de volume ;
- +18 % d'erreur après les longs trous ;
- pas de température ni de vacances scolaires ;
- un seul logement.

**22. Qu'ajouteriez-vous en priorité ?**
La **température** (chauffage), le **calendrier officiel des vacances scolaires**, un **conformal adaptatif**, et un modèle dédié au **volume**. Chaque ajout devra être évalué sur des **données nouvelles**, car le test actuel est consommé.
