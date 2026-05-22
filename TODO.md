# TennisPredictor — Roadmap

## Statut actuel (mai 2026)
- Modèle : XGBoost calibré (isotonic, cv=3) + blend base × mini-RG
- FEATURE_COLS : 29 features (Elo, clay stats, H2H, fatigue, RG historique)
- Backtesting expanding window RG 2017-2025 : ~82% accuracy Blend
- Cache persistant (parquet + joblib), qualifications exclues de l'entraînement

---

## Phase 1 — Optuna (en cours)

### Optimisation des hyperparamètres XGBoost
- [ ] Lancer Optuna avec ≥ 100 trials depuis le dashboard (bouton "Lancer Optuna")
  - CV sur 2017-2022 (holdout : 2023-2025)
  - Paramètres explorés : n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda,
    calibration_method, calibration_cv
  - Cible : score de Brier moyen pondéré (minimisation)
- [ ] Valider les meilleurs params sur 2023-2025 (bouton "Valider" dans le dashboard)
- [ ] Comparer Δ Brier et Δ Accuracy défaut vs Optuna
- [ ] Appliquer les params trouvés dans model.py (XGB_PARAMS) si amélioration confirmée
- [ ] Optimiser aussi les paramètres du blend :
  - BLEND_ALPHA (actuellement 0.70)
  - alpha floor pour SF/F (actuellement 0.80)
  - seuil de déclenchement du blend MIN_RG_FOR_BLEND (actuellement 20)
  - decay lo/hi (actuellement 20/200 matchs)

---

## Phase 2 — Feature engineering (+0.5–1.5% Acc estimé)

### Momentum clay récent
- [ ] Win streak sur terre battue (5 derniers matchs) : séquence victoires consécutives
- [ ] % victoires clay sur les 90 derniers jours (fenêtre glissante plus fine que 12m/6m)

### Performance RG spécifique
- [ ] Meilleur tour atteint aux 2-3 dernières éditions de RG (pas juste historique complet)
- [ ] Win rate RG sur fenêtre 3 ans glissante (déprécie les stats > 3 ans)

### Transition de surface
- [ ] Jours depuis le dernier match sur surface différente (gazon → terre = pénalité)
- [ ] Nombre de tournois consécutifs sur terre avant RG (continuité de rythme)

### Endurance / style de jeu
- [ ] % de matchs en 3 sets ou plus sur les 12 derniers mois (indicateur d'endurance)
- [ ] % de matchs en 5 sets au meilleur des cinq (style Grand Chelem)
- [ ] Durée moyenne des matchs / sets joués (fatigue accumulée vs endurance démontrée)

### Interaction non-linéaire âge × Elo
- [ ] Feature `age_clay_elo_interaction` = age_a × clay_elo_a (capte la courbe de carrière)
- [ ] Ou : écart de forme récente normalisé par âge (joueurs de 35 ans récupèrent moins vite)

---

## Phase 3 — Simulation de ROI avec données de cotes bookmakers

### Principe (Expected Value)
La stratégie : parier quand **EV = cote_bookmaker × P_modèle(victoire) − 1 > 0**

Notes importantes :
- La cote bookmaker encode déjà ~75-80% de l'info disponible (Elo, ranking, H2H publics).
  L'edge réel est l'**écart entre notre proba et la leur**, pas juste EV > 0.
- Formule complète : `EV = cote × P_modèle − 1` ; parier si `EV > seuil` (ex : > 0.03)
- Odds contiennent une marge bookmaker (~5-8%) à déduire : `P_bk_réelle = 1 / cote / (1 + marge)`

### Tours à cibler
- **R64 / R32** : meilleur rapport edge / liquidité, cotes moins surveillées sur outsiders
- **R128** : edge potentiel mais forte variance, mises réduites
- **QF** : mises réduites (moins de matchs, cotes plus efficientes)
- **SF / F** : éviter — forte liquidité, cotes ultra-efficientes, edge quasi nul

### À implémenter
- [ ] Collecter des données de cotes historiques RG 2017-2025
  - Source possible : [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php),
    OddsPortal (scraping), ou Betfair Exchange historical data
  - Colonnes nécessaires : match_id, winner, loser, cote_winner, cote_loser, tour
- [ ] Joindre les cotes aux prédictions du backtest (matching par joueurs + date)
- [ ] Calculer l'EV pour chaque match prédit par le modèle
- [ ] Simuler la stratégie Kelly partielle ou mise fixe par tranche d'EV
- [ ] Métriques à afficher :
  - ROI par tour (R128, R64, R32, R16, QF)
  - ROI global sur RG 2017-2025
  - Nombre de paris rentables vs perdants
  - Distribution des EV (pour calibrer le seuil de mise)
- [ ] Ajouter une page "Simulation Paris" dans le dashboard

### Gestion de portefeuille — Kelly Criterion
L'objectif : à partir d'un bankroll initial (ex : 1 000 €), maximiser sa croissance
sur l'ensemble des paris RG en allouant la mise optimale à chaque pari.

**Formule Kelly complète :**
```
f* = (EV) / (cote - 1)  =  (p × cote - 1) / (cote - 1)
mise_kelly = f* × bankroll_courant
```
- `p` = probabilité modèle, `cote` = cote décimale bookmaker
- Si `f* ≤ 0` → ne pas parier (EV négatif)

**Problème de Kelly pur :** il maximise la croissance log mais génère des drawdowns
violents (peut recommander 40% du bankroll sur un seul pari). En pratique :

**Stratégies à simuler et comparer :**
- [ ] **Kelly plein** : mise = f* × bankroll (référence théorique, très volatile)
- [ ] **Demi-Kelly** : mise = 0.5 × f* × bankroll (recommandé en pratique, drawdown ÷2)
- [ ] **Kelly plafonné** : mise = min(f* × bankroll, 5% du bankroll) (protection contre overbet)
- [ ] **Mise fixe par tranche d'EV** :
  - EV 0–3% → ne pas parier
  - EV 3–7% → 1% du bankroll
  - EV 7–15% → 2% du bankroll
  - EV > 15% → 3% du bankroll
- [ ] **Kelly par tour** (ta stratégie) :
  - R128 : 0.5× Kelly (variance élevée)
  - R64 / R32 : 1× Kelly (meilleur edge)
  - R16 / QF : 0.5× Kelly (mises réduites)
  - SF / F : 0× Kelly (ne pas parier)

**Métriques de comparaison entre stratégies :**
- [ ] Bankroll final sur RG 2017-2025 (croissance absolue)
- [ ] ROI annualisé
- [ ] Max drawdown (pire perte consécutive en % du portefeuille)
- [ ] Sharpe ratio des rendements par édition
- [ ] Nombre d'éditions rentables / perdantes

**À implémenter :**
- [ ] Classe `BankrollSimulator(strategy, initial_bankroll)` dans `src/betting.py`
- [ ] Méthode `.run(predictions_df, odds_df)` → retourne l'historique du bankroll match par match
- [ ] Graphique d'évolution du bankroll dans le dashboard (courbe par stratégie)
- [ ] Tableau comparatif des 5 stratégies sur RG 2017-2025

---

## Phase 4 — Gestion de l'incertitude (améliore Brier)

### Pondération temporelle des données d'entraînement
- [ ] Ajouter `sample_weight` à l'entraînement XGBoost selon l'ancienneté du match :
  - Match récent (< 1 an) : poids 1.0
  - 2015 : poids ≈ 0.6
  - 2010 : poids ≈ 0.3
  - Fonction : `weight = exp(-lambda × années_écoulées)` avec lambda ≈ 0.10-0.15
  - À optimiser via Optuna (lambda comme hyperparamètre)
- [ ] Tester l'impact sur Brier et Accuracy (un match 2010 ne prédit plus les mêmes joueurs)

### Calibration spécifique clay/RG
- [ ] Remplacer la calibration isotonic globale par une calibration dédiée sur données clay
- [ ] Tester une calibration séparée pour les tours avancés (SF/F sont différents)
- [ ] Comparer Brier avec calibration globale vs clay-specific

### Intervalles de confiance (Conformal Prediction)
- [ ] Implémenter un wrapper conformal prediction sur le modèle calibré
- [ ] Pour chaque match prédit, retourner `[P_low, P_high]` à 80% de couverture
- [ ] Utiliser cet intervalle dans la décision de pari :
  - Si `P_low × cote > 1` → pari à haute confiance
  - Si `P_high × cote > 1` mais `P_low < 1/cote` → pari risqué, mise réduite

---

## Backlog / idées futures
- Intégrer les données de surface indoor (certains joueurs clay performent différemment sous toit)
- Ajouter les stats ATP serve/return détaillées (% 1st serve, BP conversion) sur les 3 derniers mois
- Modèle de bracket complet : probabilité de parcours (atteindre QF, SF, F) pas juste match à match
- Export des prédictions RG 2026 en CSV depuis le dashboard
