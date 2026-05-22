"""
Optimisation des hyperparamètres XGBoost avec Optuna.
Validation croisée temporelle (expanding window) sur RG 2017-2022.
Cible : score de Brier moyen (minimisation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from features import FEATURE_COLS, build_symmetric_dataset
from model import MAIN_DRAW_ROUNDS


BEST_PARAMS_PATH = Path(__file__).parent / "best_params.json"

# Années utilisées pour l'optimisation ; 2023-2025 restent holdout
CV_YEARS = list(range(2017, 2023))


def _score_params(
    params: dict,
    all_feat: pd.DataFrame,
    rg_feat: pd.DataFrame,
    calibration_method: str = "isotonic",
    calibration_cv: int = 3,
    feature_cols: list[str] = FEATURE_COLS,
    years: list[int] = CV_YEARS,
) -> float:
    """
    Score de Brier moyen pondéré (expanding window) sur les années données.
    Évalue le modèle de base (sans blend) pour la vitesse.
    Retourne np.inf si l'entraînement échoue pour un trial.
    """
    total_brier = 0.0
    total_n = 0

    for year in years:
        rg_year = rg_feat[
            (rg_feat["tourney_date"].dt.year == year)
            & (rg_feat["round_number"].isin(MAIN_DRAW_ROUNDS))
        ]
        if rg_year.empty:
            continue

        rg_start = rg_year["tourney_date"].min()
        train_df = all_feat[all_feat["tourney_date"] < rg_start]
        if len(train_df) < 100:
            continue

        sym = build_symmetric_dataset(train_df)
        X_tr = sym[feature_cols].fillna(0)
        y_tr = sym["target"]

        try:
            xgb = XGBClassifier(**params, eval_metric="logloss", random_state=42, n_jobs=-1)
            model = CalibratedClassifierCV(xgb, cv=calibration_cv, method=calibration_method)
            model.fit(X_tr, y_tr)
        except Exception:
            return np.inf

        X_test = rg_year[feature_cols].fillna(0)
        y_test = rg_year["target"].values
        p = model.predict_proba(X_test)[:, 1]
        total_brier += brier_score_loss(y_test, p) * len(y_test)
        total_n += len(y_test)

    return total_brier / total_n if total_n > 0 else np.inf


def run_optuna(
    all_feat: pd.DataFrame,
    rg_feat: pd.DataFrame,
    n_trials: int = 50,
    on_trial_end: Optional[Callable[[int, float, dict], None]] = None,
) -> optuna.Study:
    """
    Lance l'optimisation Optuna et retourne le study.

    Parameters
    ----------
    all_feat     : features clay historiques (hors qualifications)
    rg_feat      : features Roland Garros (hors qualifications)
    n_trials     : nombre de trials
    on_trial_end : callback(trial_number, best_value, best_params) après chaque trial
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 3.0),
        }
        cal_method = trial.suggest_categorical("calibration_method", ["isotonic", "sigmoid"])
        cal_cv = trial.suggest_int("calibration_cv", 3, 5)

        return _score_params(
            params, all_feat, rg_feat,
            calibration_method=cal_method,
            calibration_cv=cal_cv,
        )

    def _cb(study: optuna.Study, trial: optuna.Trial) -> None:
        if on_trial_end is not None:
            on_trial_end(trial.number + 1, study.best_value, study.best_params)

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    return study


def save_best_params(study: optuna.Study) -> dict:
    """Sauvegarde les meilleurs paramètres dans best_params.json et retourne le dict."""
    best = study.best_params.copy()
    best["_best_brier"] = study.best_value
    best["_n_trials"] = len(study.trials)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    return best


def load_best_params() -> Optional[dict]:
    """Charge les meilleurs paramètres depuis best_params.json, None si absent."""
    if not BEST_PARAMS_PATH.exists():
        return None
    return json.loads(BEST_PARAMS_PATH.read_text())


def get_xgb_params(best: dict) -> dict:
    """Extrait les paramètres XGBoost purs depuis le dict Optuna (enlève les méta-clés)."""
    skip = {"calibration_method", "calibration_cv", "_best_brier", "_n_trials"}
    return {k: v for k, v in best.items() if k not in skip}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from backtesting import run_full_backtest

    DATA_DIR = str(Path(__file__).parent.parent / "data" / "raw")
    print("Chargement des données...")
    out = run_full_backtest(DATA_DIR)
    all_feat = out["all_features"]
    rg_feat  = out["rg_features"]

    n = 100
    print(f"\nLancement Optuna — {n} trials | CV {CV_YEARS[0]}-{CV_YEARS[-1]}")

    def _progress(n_done: int, best_val: float, _params: dict) -> None:
        print(f"  Trial {n_done:3d} | Best Brier : {best_val:.4f}")

    study = run_optuna(all_feat, rg_feat, n_trials=n, on_trial_end=_progress)
    best = save_best_params(study)

    print(f"\nMeilleurs paramètres (Brier = {best['_best_brier']:.4f}) :")
    for k, v in best.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")
    print(f"\nSauvegardés dans {BEST_PARAMS_PATH}")
