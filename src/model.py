"""
Modèle XGBoost + expanding window backtesting pour Roland Garros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from xgboost import XGBClassifier

from features import FEATURE_COLS, build_features, build_symmetric_dataset


XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class BacktestResult:
    year: int
    accuracy: float
    brier: float
    log_loss_val: float
    n_matches: int
    by_round: pd.DataFrame


def train_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    calibrate: bool = True,
) -> XGBClassifier | CalibratedClassifierCV:
    """Entraîne XGBoost sur feature_df (symétrisé)."""
    sym_df = build_symmetric_dataset(feature_df)
    X = sym_df[feature_cols].fillna(0)
    y = sym_df["target"]

    model = XGBClassifier(**XGB_PARAMS)
    if calibrate:
        model = CalibratedClassifierCV(model, cv=3, method="isotonic")

    model.fit(X, y)
    return model


def predict_proba_a(model, match_features: pd.Series, feature_cols: list[str] = FEATURE_COLS) -> float:
    """Probabilité que le joueur A gagne (features côté A)."""
    X = match_features[feature_cols].fillna(0).values.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    # classe 1 = A gagne
    return float(proba[1])


def expanding_window_backtest(
    all_feature_df: pd.DataFrame,
    rg_feature_df: pd.DataFrame,
    rg_years: list[int] | None = None,
    feature_cols: list[str] = FEATURE_COLS,
) -> list[BacktestResult]:
    """
    Pour chaque édition RG (2017-2025) :
      - Train : toutes les features clay AVANT le début de ce RG
      - Test  : features de ce RG uniquement
    """
    if rg_years is None:
        rg_years = list(range(2017, 2026))

    results = []

    pbar = tqdm(rg_years, desc="Backtest RG", unit="édition", ncols=80)
    for year in pbar:
        pbar.set_postfix({"année": year})
        rg_year_df = rg_feature_df[rg_feature_df["tourney_date"].dt.year == year]
        if rg_year_df.empty:
            tqdm.write(f"  [WARN] Pas de données RG {year}")
            continue

        rg_start = rg_year_df["tourney_date"].min()
        train_df = all_feature_df[all_feature_df["tourney_date"] < rg_start]

        if len(train_df) < 100:
            tqdm.write(f"  [WARN] Trop peu de données d'entraînement pour RG {year}")
            continue

        pbar.set_description(f"Backtest RG {year} (train={len(train_df):,})")
        model = train_model(train_df, feature_cols, calibrate=True)

        # Prédictions sur le RG de l'année
        X_test = rg_year_df[feature_cols].fillna(0)
        y_true = rg_year_df["target"].values
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.5).astype(int)

        acc = accuracy_score(y_true, preds)
        brier = brier_score_loss(y_true, probas)
        # sklearn raises if only one class is present in `y_true`.
        # Fournir explicitement les labels permet d'éviter l'erreur
        # ValueError: y_true contains only one label (1).
        ll = log_loss(y_true, np.column_stack([1 - probas, probas]), labels=[0, 1])

        # Décomposition par tour
        by_round_rows = []
        for rn in sorted(rg_year_df["round_number"].unique()):
            mask = rg_year_df["round_number"] == rn
            sub_X = rg_year_df[mask][feature_cols].fillna(0)
            sub_y = rg_year_df[mask]["target"].values
            sub_p = model.predict_proba(sub_X)[:, 1]
            by_round_rows.append({
                "round": rn,
                "accuracy": accuracy_score(sub_y, (sub_p >= 0.5).astype(int)),
                "brier": brier_score_loss(sub_y, sub_p),
                "n": int(mask.sum()),
            })

        results.append(BacktestResult(
            year=year,
            accuracy=acc,
            brier=brier,
            log_loss_val=ll,
            n_matches=len(rg_year_df),
            by_round=pd.DataFrame(by_round_rows),
        ))
        tqdm.write(f"  RG {year}: Acc={acc:.3f} | Brier={brier:.4f} | LogLoss={ll:.4f} | n={len(rg_year_df)}")

    return results


def compare_baselines(rg_feature_df: pd.DataFrame, all_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compare XGBoost vs baselines (ranking, Elo clay, WElo) sur RG 2017-2025."""
    rows = []

    for year in tqdm(range(2017, 2026), desc="Comparaison baselines", unit="année", ncols=80):
        rg_y = rg_feature_df[rg_feature_df["tourney_date"].dt.year == year]
        if rg_y.empty:
            continue

        rg_start = rg_y["tourney_date"].min()
        train_df = all_feature_df[all_feature_df["tourney_date"] < rg_start]
        y_true = rg_y["target"].values

        # Baseline 1 : ranking ATP
        rank_proba = 1.0 / (1.0 + np.exp(rg_y["ranking_diff"].fillna(0).values / 50.0))
        # Baseline 2 : clay Elo
        clay_proba = 1.0 / (1.0 + 10.0 ** (-rg_y["diff_clay_elo"].fillna(0).values / 400.0))
        # Baseline 3 : WElo ajusté
        welo_proba = 1.0 / (1.0 + 10.0 ** (-rg_y["diff_adjusted_elo"].fillna(0).values / 400.0))

        row = {
            "year": year,
            "n": len(rg_y),
            "baseline_rank_acc": accuracy_score(y_true, (rank_proba >= 0.5).astype(int)),
            "baseline_rank_brier": brier_score_loss(y_true, rank_proba),
            "baseline_clay_elo_acc": accuracy_score(y_true, (clay_proba >= 0.5).astype(int)),
            "baseline_clay_elo_brier": brier_score_loss(y_true, clay_proba),
            "baseline_welo_adj_acc": accuracy_score(y_true, (welo_proba >= 0.5).astype(int)),
            "baseline_welo_adj_brier": brier_score_loss(y_true, welo_proba),
        }

        if len(train_df) >= 100:
            model = train_model(train_df, FEATURE_COLS, calibrate=True)
            X_test = rg_y[FEATURE_COLS].fillna(0)
            xgb_proba = model.predict_proba(X_test)[:, 1]
            row["xgboost_acc"] = accuracy_score(y_true, (xgb_proba >= 0.5).astype(int))
            row["xgboost_brier"] = brier_score_loss(y_true, xgb_proba)
        else:
            row["xgboost_acc"] = np.nan
            row["xgboost_brier"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_importance(model, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """Retourne l'importance des features (compatible XGBoost natif et CalibratedClassifierCV)."""
    if hasattr(model, "estimator"):
        base = model.estimators_[0].estimator if hasattr(model, "estimators_") else model.estimator
    else:
        base = model
    importances = base.feature_importances_
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
