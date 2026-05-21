# Roland Garros 2026 Predictor

Système de prédiction de matchs basé sur WElo surface-spécifique + XGBoost, entraîné en expanding window sur les données ATP historiques (Jeff Sackmann).

## Installation

```bash
cd TennisPredictor

# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Télécharger les données

```bash
bash scripts/download_data.sh
```

Cela clone le repo [Jeff Sackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) et copie les fichiers CSV dans `data/raw/`.

## Backtesting (2017–2025)

```bash
python scripts/run_backtest.py
# Avec sauvegarde CSV :
python scripts/run_backtest.py --save-csv
```

Affiche dans le terminal :
- Accuracy / Brier Score / Log-Loss par édition Roland Garros
- Comparaison XGBoost vs baselines (ranking, Clay Elo, WElo)
- Décomposition par tour

## Dashboard interactif

```bash
streamlit run dashboard/app.py
```

Ouvre automatiquement `http://localhost:8501` avec 4 sections :

| Section | Description |
|---|---|
| **Vue d'ensemble** | Simulation Monte Carlo du tableau, profils Elo |
| **Match par match** | Prédiction avec radar comparatif et features décisives |
| **Saisie résultats** | Formulaire de saisie + réentraînement automatique |
| **Backtesting** | Graphes de performance historique 2017-2025 |

## Saisir un résultat pendant le tournoi

Via le dashboard (section "Saisie résultats"), ou en Python :

```python
from src.predictor import RolandGarrosPredictor

pred = RolandGarrosPredictor("data/raw")

# Prédire un match
result = pred.predict_match("Carlos Alcaraz", "Jannik Sinner", round_number=7)
print(result)
# {'winner_predicted': 'Carlos Alcaraz', 'proba_a': 0.64, 'proba_b': 0.36, ...}

# Enregistrer un résultat réel
pred.add_result(
    winner="Carlos Alcaraz",
    loser="Jannik Sinner",
    score="6-3 7-5 6-1",
    round_number=7,
)
# → Elo mis à jour + modèle réentraîné automatiquement
```

Les résultats saisis sont persistés dans `data/rg2026/results.jsonl` et rechargés à chaque démarrage.

## Architecture

```
src/
├── data_loader.py   # Chargement CSV Sackmann (2000-2025)
├── elo.py           # Standard Elo / Surface Elo / WElo / AdjustedElo
├── features.py      # Feature engineering sans data leakage
├── model.py         # XGBoost + expanding window backtesting
├── predictor.py     # API principale (predict, add_result, simulate)
└── backtesting.py   # Pipeline de backtesting standalone
```

## Modèles Elo implémentés

| Système | Description |
|---|---|
| **Standard Elo** | Elo classique toutes surfaces, K=32 |
| **Surface Elo** | Un rating séparé par surface (Clay/Hard/Grass), K pondéré par prestige du tournoi |
| **WElo** | Hot-hand : K multiplié par `1 + α·résultat_précédent` (α optimisé) |
| **AdjustedElo** | `(1−λ)·StandardElo + λ·SurfaceElo` (λ optimisé sur RG) |

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests vérifiés :
- No data leakage (Elo pré-match uniquement)
- Clay Elo non mis à jour sur hard/grass
- H2H calculé avant la date du match
- Dataset symétrique (pas de biais de position)
- Réentraînement du modèle

## Sources

- Angelini et al. (2022) — WElo : *ScienceDirect*
- Williams et al. — AdjustedElo pour Grands Chelems : *NTU*
- Kovalchik (2023) — XGBoost + features tennis : *PMLR*
- Buhamra et al. (2025) — Âge optimal 28-32 ans : *arXiv*
- Données : [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
