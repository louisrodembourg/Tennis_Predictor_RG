"""
Modèle XGBoost + expanding window backtesting pour Roland Garros.

Stratégie de prédiction :
  - Modèle de base  : XGBoost calibré entraîné sur toutes les données clay historiques
  - Mini-modèle RG  : XGBoost léger entraîné uniquement sur les matchs RG déjà joués
  - Prédiction finale = alpha × base + (1-alpha) × mini_rg
    (alpha décroît au fur et à mesure que les données RG s'accumulent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURE_COLS, build_features, build_symmetric_dataset


BLEND_ALPHA = 0.70       # poids du modèle historique (0.70 = 70 % base, 30 % RG)
MIN_RG_FOR_BLEND = 20    # nombre minimum d'exemples symétriques RG pour activer le blend

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

XGB_MINI_PARAMS = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

ROUND_NAMES = {
    -2: "Q1", -1: "Q2", 0: "Q3",
    1: "R128", 2: "R64", 3: "R32", 4: "R16", 5: "QF", 6: "SF", 7: "F",
}
# Tableau principal uniquement (pas les qualifications)
MAIN_DRAW_ROUNDS = {1, 2, 3, 4, 5, 6, 7}


@dataclass
class RoundStat:
    round_number: int
    round_name: str
    n: int
    acc_base: float
    brier_base: float
    acc_blend: float
    brier_blend: float
    rg_matches_used: int    # matchs RG ayant servi à entraîner le mini-modèle


@dataclass
class BacktestResult:
    year: int
    accuracy: float
    brier: float
    log_loss_val: float
    n_matches: int
    by_round: pd.DataFrame                    # colonnes : round, round_name, n, acc_base, brier_base, acc_blend, brier_blend
    accuracy_blend: float = 0.0
    brier_blend: float = 0.0
    log_loss_blend: float = 0.0


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    calibrate: bool = True,
    xgb_params: dict | None = None,
    calibration_method: str = "isotonic",
    calibration_cv: int = 3,
) -> XGBClassifier | CalibratedClassifierCV:
    """Entraîne le modèle de base (historique) sur feature_df symétrisé."""
    sym_df = build_symmetric_dataset(feature_df)
    X = sym_df[feature_cols].fillna(0)
    y = sym_df["target"]
    params = xgb_params if xgb_params is not None else XGB_PARAMS
    model = XGBClassifier(**params)
    if calibrate:
        model = CalibratedClassifierCV(model, cv=calibration_cv, method=calibration_method)
    model.fit(X, y)
    return model


def train_mini_model(
    rg_feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> Optional[XGBClassifier]:
    """
    Entraîne un mini-modèle léger sur les matchs RG déjà joués.
    Retourne None si trop peu de données.
    """
    sym = build_symmetric_dataset(rg_feature_df)
    if len(sym) < MIN_RG_FOR_BLEND:
        return None
    X = sym[feature_cols].fillna(0)
    y = sym["target"]
    model = XGBClassifier(**XGB_MINI_PARAMS)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def blend_probas(
    base_model,
    rg_model,
    X: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    alpha: float = BLEND_ALPHA,
) -> np.ndarray:
    """
    Retourne les probabilités blendées.
    Si rg_model est None, retourne les probas du modèle de base seul.
    alpha décroît dynamiquement avec la taille des données RG (cf. dynamic_alpha).
    """
    X_arr = X[feature_cols].fillna(0)
    base_p = base_model.predict_proba(X_arr)[:, 1]
    if rg_model is None:
        return base_p
    rg_p = rg_model.predict_proba(X_arr)[:, 1]
    return alpha * base_p + (1.0 - alpha) * rg_p


def dynamic_alpha(n_rg_matches: int, round_number: int = 1) -> float:
    """
    Alpha décroît de BLEND_ALPHA vers 0.55 selon le volume de données RG.
    SF/F : plancher à 0.80 car trop peu d'exemples, le mini-modèle overfit.
    """
    lo, hi = 20, 200
    if n_rg_matches <= lo:
        base = BLEND_ALPHA
    elif n_rg_matches >= hi:
        base = 0.55
    else:
        t = (n_rg_matches - lo) / (hi - lo)
        base = BLEND_ALPHA - t * (BLEND_ALPHA - 0.55)
    # SF et Finale : le mini-modèle a trop peu d'exemples → garder le modèle historique dominant
    if round_number >= 6:
        return max(0.80, base)
    return base


def predict_proba_a(
    model,
    match_features: pd.Series,
    rg_model=None,
    feature_cols: list[str] = FEATURE_COLS,
    alpha: float = BLEND_ALPHA,
) -> float:
    """Probabilité que le joueur A gagne, avec blend optionnel."""
    X = match_features[feature_cols].fillna(0).values.reshape(1, -1)
    base_p = float(model.predict_proba(X)[0][1])
    if rg_model is None:
        return base_p
    rg_p = float(rg_model.predict_proba(X)[0][1])
    return alpha * base_p + (1.0 - alpha) * rg_p


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def _intra_rg_sets_from_raw(rg_raw: pd.DataFrame) -> dict[str, int]:
    """Construit le dict {joueur: sets_joués} depuis les matchs bruts précédents."""
    if rg_raw.empty or "score" not in rg_raw.columns:
        return {}
    import re
    intra: dict[str, int] = {}
    scores = rg_raw["score"].fillna("").astype(str)
    for i, row in enumerate(rg_raw.itertuples(index=False)):
        n_sets = len(re.findall(r"\d+\s*-\s*\d+", scores.iloc[i]))
        for player in [row.winner_name, row.loser_name]:
            intra[player] = intra.get(player, 0) + n_sets
    return intra


def expanding_window_backtest(
    all_feature_df: pd.DataFrame,
    rg_feature_df: pd.DataFrame,
    rg_years: list[int] | None = None,
    feature_cols: list[str] = FEATURE_COLS,
    blend_alpha: float = BLEND_ALPHA,
    rg_raw_df: Optional[pd.DataFrame] = None,
    xgb_params: dict | None = None,
    calibration_method: str = "isotonic",
    calibration_cv: int = 3,
) -> list[BacktestResult]:
    """
    Pour chaque édition RG (2017-2025) :
      - Modèle de base  : toutes les features clay AVANT ce RG
      - Blend online    : pour chaque tour R, le mini-modèle est entraîné
                          sur les matchs des tours < R du MÊME tournoi.
      - Résultats par tour pour les deux approches.
    """
    if rg_years is None:
        rg_years = list(range(2017, 2026))

    results = []
    pbar = tqdm(rg_years, desc="Backtest RG", unit="édition", ncols=80)

    for year in pbar:
        pbar.set_postfix({"année": year})
        rg_year_df = rg_feature_df[rg_feature_df["tourney_date"].dt.year == year].copy()
        if rg_year_df.empty:
            tqdm.write(f"  [WARN] Pas de données RG {year}")
            continue

        rg_start = rg_year_df["tourney_date"].min()
        train_df = all_feature_df[all_feature_df["tourney_date"] < rg_start]

        if len(train_df) < 100:
            tqdm.write(f"  [WARN] Trop peu de données d'entraînement pour RG {year}")
            continue

        pbar.set_description(f"Backtest RG {year} (train={len(train_df):,})")
        base_model = train_model(
            train_df, feature_cols, calibrate=True,
            xgb_params=xgb_params,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
        )

        # --- Simulation online round par round (tableau principal uniquement) ---
        rounds = sorted(r for r in rg_year_df["round_number"].unique()
                        if r in MAIN_DRAW_ROUNDS)
        rg_seen: list[pd.DataFrame] = []   # matchs RG déjà joués (rounds précédents)
        by_round_rows: list[dict] = []
        all_base_proba: list[np.ndarray] = []
        all_blend_proba: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        # Matchs bruts de l'année pour diff_sets_played_rg (si fournis)
        rg_raw_year: pd.DataFrame = pd.DataFrame()
        if rg_raw_df is not None:
            rg_raw_year = rg_raw_df[rg_raw_df["tourney_date"].dt.year == year].copy()

        for rn in rounds:
            mask = rg_year_df["round_number"] == rn
            rn_df = rg_year_df[mask].copy()

            # Mise à jour de diff_sets_played_rg depuis les matchs bruts précédents
            if not rg_raw_year.empty:
                prev_raw = rg_raw_year[rg_raw_year["round_number"] < rn]
                intra = _intra_rg_sets_from_raw(prev_raw)
                if intra and "diff_sets_played_rg" in rn_df.columns:
                    sets_a = rn_df["player_a"].map(lambda p: intra.get(p, 0))
                    sets_b = rn_df["player_b"].map(lambda p: intra.get(p, 0))
                    rn_df["diff_sets_played_rg"] = sets_a.values - sets_b.values

            X_rn = rn_df[feature_cols].fillna(0)
            y_rn = rn_df["target"].values

            # Proba base
            base_p = base_model.predict_proba(X_rn)[:, 1]

            # Mini-modèle sur les rounds précédents
            rg_so_far = pd.concat(rg_seen, ignore_index=True) if rg_seen else pd.DataFrame()
            n_rg = len(rg_so_far)
            alpha = dynamic_alpha(n_rg * 2, round_number=rn)
            mini = train_mini_model(rg_so_far, feature_cols) if not rg_so_far.empty else None
            blend_p = blend_probas(base_model, mini, X_rn, feature_cols, alpha=alpha)

            # Stats
            acc_base  = accuracy_score(y_rn, (base_p  >= 0.5).astype(int))
            brier_base  = brier_score_loss(y_rn, base_p)
            acc_blend = accuracy_score(y_rn, (blend_p >= 0.5).astype(int))
            brier_blend = brier_score_loss(y_rn, blend_p)

            by_round_rows.append({
                "round_number":    rn,
                "round_name":      ROUND_NAMES.get(rn, str(rn)),
                "n":               int(mask.sum()),
                "acc_base":        acc_base,
                "brier_base":      brier_base,
                "acc_blend":       acc_blend,
                "brier_blend":     brier_blend,
                "rg_matches_used": n_rg,
            })

            all_base_proba.append(base_p)
            all_blend_proba.append(blend_p)
            all_y.append(y_rn)

            # Ajouter ce tour aux données RG vues
            rg_seen.append(rn_df)

        # Métriques globales
        y_all    = np.concatenate(all_y)
        base_all = np.concatenate(all_base_proba)
        blend_all = np.concatenate(all_blend_proba)

        acc   = accuracy_score(y_all, (base_all  >= 0.5).astype(int))
        brier = brier_score_loss(y_all, base_all)
        ll    = log_loss(y_all, np.column_stack([1 - base_all, base_all]), labels=[0, 1])
        acc_bl   = accuracy_score(y_all, (blend_all >= 0.5).astype(int))
        brier_bl = brier_score_loss(y_all, blend_all)
        ll_bl    = log_loss(y_all, np.column_stack([1 - blend_all, blend_all]), labels=[0, 1])

        results.append(BacktestResult(
            year=year,
            accuracy=acc, brier=brier, log_loss_val=ll,
            n_matches=len(y_all),   # matchs tableau principal uniquement (hors qualifs)
            by_round=pd.DataFrame(by_round_rows),
            accuracy_blend=acc_bl, brier_blend=brier_bl, log_loss_blend=ll_bl,
        ))
        tqdm.write(
            f"  RG {year}: "
            f"Base Acc={acc:.3f} Brier={brier:.4f} | "
            f"Blend Acc={acc_bl:.3f} Brier={brier_bl:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Comparaison baselines
# ---------------------------------------------------------------------------

def compare_baselines(rg_feature_df: pd.DataFrame, all_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compare XGBoost (base + blend) vs baselines sur RG 2017-2025."""
    rows = []

    for year in tqdm(range(2017, 2026), desc="Comparaison baselines", unit="année", ncols=80):
        rg_y = rg_feature_df[
            (rg_feature_df["tourney_date"].dt.year == year) &
            (rg_feature_df["round_number"].isin(MAIN_DRAW_ROUNDS))
        ]
        if rg_y.empty:
            continue

        rg_start = rg_y["tourney_date"].min()
        train_df = all_feature_df[all_feature_df["tourney_date"] < rg_start]
        y_true = rg_y["target"].values

        # Baselines
        rank_proba = 1.0 / (1.0 + np.exp(rg_y["ranking_diff"].fillna(0).values / 50.0))
        clay_proba = 1.0 / (1.0 + 10.0 ** (-rg_y["diff_clay_elo"].fillna(0).values / 400.0))
        welo_proba = 1.0 / (1.0 + 10.0 ** (-rg_y["diff_adjusted_elo"].fillna(0).values / 400.0))

        row: dict = {
            "year": year,
            "n": len(rg_y),
            "baseline_rank_acc":    accuracy_score(y_true, (rank_proba >= 0.5).astype(int)),
            "baseline_rank_brier":  brier_score_loss(y_true, rank_proba),
            "baseline_clay_elo_acc":  accuracy_score(y_true, (clay_proba >= 0.5).astype(int)),
            "baseline_clay_elo_brier": brier_score_loss(y_true, clay_proba),
            "baseline_welo_adj_acc":  accuracy_score(y_true, (welo_proba >= 0.5).astype(int)),
            "baseline_welo_adj_brier": brier_score_loss(y_true, welo_proba),
        }

        if len(train_df) >= 100:
            base_model = train_model(train_df, FEATURE_COLS, calibrate=True)
            X_test = rg_y[FEATURE_COLS].fillna(0)
            xgb_p = base_model.predict_proba(X_test)[:, 1]
            row["xgboost_acc"]   = accuracy_score(y_true, (xgb_p >= 0.5).astype(int))
            row["xgboost_brier"] = brier_score_loss(y_true, xgb_p)

            # Blend simulé : mini-modèle sur le RG de l'année précédente
            prev_rg = rg_feature_df[rg_feature_df["tourney_date"].dt.year == year - 1]
            if len(prev_rg) >= MIN_RG_FOR_BLEND // 2:
                mini = train_mini_model(prev_rg, FEATURE_COLS)
                if mini:
                    al = dynamic_alpha(len(prev_rg) * 2)
                    bl_p = blend_probas(base_model, mini, X_test, alpha=al)
                    row["blend_acc"]   = accuracy_score(y_true, (bl_p >= 0.5).astype(int))
                    row["blend_brier"] = brier_score_loss(y_true, bl_p)
        else:
            row["xgboost_acc"] = np.nan
            row["xgboost_brier"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_importance(model, feature_cols: list[str] = FEATURE_COLS) -> pd.DataFrame:
    """Retourne l'importance des features (compatible XGBoost natif et CalibratedClassifierCV)."""
    if hasattr(model, "estimators_"):
        base = model.estimators_[0].estimator
    elif hasattr(model, "estimator"):
        base = model.estimator
    else:
        base = model
    importances = base.feature_importances_
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
