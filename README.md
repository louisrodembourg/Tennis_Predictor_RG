# Roland Garros Match Predictor

Predicts the winner of every Roland Garros men's match from historical ATP data, with
a surface-specific Elo (WElo) and a calibrated XGBoost model trained in an expanding
window, so the model never sees the future.

**Result:** about **82 % match accuracy** in the expanding-window backtest over
Roland Garros 2017–2025, each edition predicted only from the matches played before it.
The backtest also reports the Brier score and log-loss per edition and per round, and
compares the model against ranking-based, clay-Elo and WElo baselines.

## Approach

- **Ratings:** four Elo variants are implemented and compared: standard Elo, a separate
  rating per surface weighted by tournament prestige, WElo (a "hot-hand" K-factor) and
  an adjusted Elo that blends overall and surface ratings. The blend weight is tuned on
  Roland Garros.
- **Features:** 41 pre-match features: recent and clay form, a clay win streak,
  fatigue over the previous 21 days (matches, sets, minutes; the most important group),
  clay head-to-head, Roland Garros history, ranking, age, serve and return statistics.
  Highly correlated Elo variants (r > 0.96) are dropped.
- **Model:** XGBoost with isotonic calibration, blended with a model specialised on
  Roland Garros matches. Hyperparameters are searched with Optuna (50 trials, Brier
  score objective, cross-validation on 2017–2022 with 2023–2025 held out).
- **Uncertainty:** split conformal prediction gives an interval around each win
  probability, with a guaranteed coverage level.
- **Extras:** SHAP feature analysis, a LightGBM + XGBoost stacking variant, Monte Carlo
  simulation of the full draw, and a betting simulator (expected value, Kelly sizing,
  ROI and drawdown) on historical odds.

## Leakage checks

The tests in `tests/` check that:

- only pre-match Elo is used;
- the clay rating is not updated on hard or grass matches;
- head-to-head is computed strictly before the match date;
- the dataset is symmetric, so the model has no player-position bias;
- retraining after a new result works.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh        # ATP results from JeffSackmann/tennis_atp → data/raw/
python scripts/run_backtest.py       # backtest 2017–2025 (add --save-csv to export)
streamlit run dashboard/app.py       # interactive dashboard
pytest tests/ -v
```

The dashboard has four pages: a Monte Carlo simulation of the draw with Elo profiles,
match-by-match predictions with the deciding features, a form to enter live results
(the ratings update and the model retrains), and the historical backtest.

From Python:

```python
from src.predictor import RolandGarrosPredictor

pred = RolandGarrosPredictor("data/raw")
pred.predict_match("Carlos Alcaraz", "Jannik Sinner", round_number=7)
# {'winner_predicted': ..., 'proba_a': ..., 'proba_b': ..., ...}

pred.add_result(winner="Carlos Alcaraz", loser="Jannik Sinner",
                score="6-3 7-5 6-1", round_number=7)   # Elo updated, model retrained
```

Results entered during the tournament are stored in `data/rg2026/results.jsonl`.

## Structure

```
src/
├── data_loader.py      # Sackmann ATP CSVs (2000–2025)
├── elo.py              # Standard, surface, WElo and adjusted Elo
├── features.py         # leakage-free feature engineering
├── model.py            # calibrated XGBoost, expanding-window training
├── ensemble.py         # LightGBM + XGBoost stacking
├── conformal.py        # split conformal prediction intervals
├── hyperopt.py         # Optuna search
├── betting.py          # odds, expected value, Kelly bankroll simulation
├── rg2026_bracket.py   # full-draw simulation
├── predictor.py        # main API (predict, add_result, simulate)
└── backtesting.py      # standalone backtest pipeline
```

## References

- Angelini et al. (2022), WElo: a weighted Elo rating system for tennis
- Williams et al., adjusted Elo for Grand Slams
- Kovalchik (2023), XGBoost and tennis features
- Data: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
