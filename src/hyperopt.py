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


BEST_PARAMS_PATH       = Path(__file__).parent / "best_params.json"
BEST_BLEND_PARAMS_PATH = Path(__file__).parent / "best_blend_params.json"

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
    temporal_lambda: float = 0.0,
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

        sample_weight = None
        if temporal_lambda > 0.0 and "tourney_date" in sym.columns:
            ref_date = pd.to_datetime(sym["tourney_date"]).max()
            years_ago = (ref_date - pd.to_datetime(sym["tourney_date"])).dt.days / 365.25
            sample_weight = np.exp(-temporal_lambda * years_ago.values)

        try:
            xgb = XGBClassifier(**params, eval_metric="logloss", random_state=42, n_jobs=-1)
            model = CalibratedClassifierCV(xgb, cv=calibration_cv, method=calibration_method)
            fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
            model.fit(X_tr, y_tr, **fit_kwargs)
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
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 3.0),
        }
        cal_method = trial.suggest_categorical("calibration_method", ["isotonic", "sigmoid"])
        cal_cv = trial.suggest_int("calibration_cv", 3, 5)
        temporal_lambda = trial.suggest_float("temporal_lambda", 0.0, 0.4)

        return _score_params(
            params, all_feat, rg_feat,
            calibration_method=cal_method,
            calibration_cv=cal_cv,
            temporal_lambda=temporal_lambda,
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
    skip = {"calibration_method", "calibration_cv", "temporal_lambda", "_best_brier", "_n_trials"}
    return {k: v for k, v in best.items() if k not in skip}


def get_temporal_lambda(best: dict) -> float:
    """Extrait temporal_lambda depuis le dict Optuna (0.0 si absent)."""
    return float(best.get("temporal_lambda", 0.0))


# ---------------------------------------------------------------------------
# Blend Optuna (Étape 2) — optimise les paramètres de blend
# ---------------------------------------------------------------------------

def _pretrain_base_models(
    all_feat: pd.DataFrame,
    rg_feat: pd.DataFrame,
    xgb_params: dict,
    calibration_method: str,
    calibration_cv: int,
    temporal_lambda: float,
    feature_cols: list[str],
    years: list[int],
) -> dict:
    """Pré-entraîne un modèle de base par année (exécuté une seule fois avant l'étude blend)."""
    from model import train_model, MAIN_DRAW_ROUNDS as MDR
    import tqdm as _tqdm
    models = {}
    for year in _tqdm.tqdm(years, desc="Pré-entraînement modèles de base", ncols=80):
        rg_year = rg_feat[
            (rg_feat["tourney_date"].dt.year == year) & (rg_feat["round_number"].isin(MDR))
        ]
        if rg_year.empty:
            continue
        rg_start = rg_year["tourney_date"].min()
        train_df = all_feat[all_feat["tourney_date"] < rg_start]
        if len(train_df) < 100:
            continue
        models[year] = train_model(
            train_df, feature_cols, calibrate=True,
            xgb_params=xgb_params,
            calibration_method=calibration_method,
            calibration_cv=calibration_cv,
            temporal_lambda=temporal_lambda,
        )
    return models


def _score_blend_params(
    blend_params: dict,
    rg_feat: pd.DataFrame,
    base_models: dict,
    feature_cols: list[str],
    years: list[int],
) -> float:
    """Score de Brier moyen sur les prédictions blendées (modèles de base pré-cachés)."""
    from model import train_mini_model, blend_probas, dynamic_alpha, MAIN_DRAW_ROUNDS as MDR

    ba      = blend_params["blend_alpha"]
    bat     = blend_params["blend_alpha_target"]
    d_lo    = int(blend_params["decay_lo"])
    d_hi    = int(blend_params["decay_hi"])
    afl     = blend_params["alpha_floor_late_round"]
    min_rg  = int(blend_params["min_rg_for_blend"])

    total_brier = 0.0
    total_n = 0

    for year in years:
        if year not in base_models:
            continue
        base_model = base_models[year]
        rg_year = rg_feat[
            (rg_feat["tourney_date"].dt.year == year) & (rg_feat["round_number"].isin(MDR))
        ].copy()
        if rg_year.empty:
            continue

        rounds = sorted(rg_year["round_number"].unique())
        rg_seen: list = []

        for rn in rounds:
            rn_df = rg_year[rg_year["round_number"] == rn]
            X_rn = rn_df[feature_cols].fillna(0)
            y_rn = rn_df["target"].values

            rg_so_far = pd.concat(rg_seen, ignore_index=True) if rg_seen else pd.DataFrame()
            n_rg = len(rg_so_far)
            alpha = dynamic_alpha(
                n_rg * 2, round_number=rn,
                blend_alpha=ba, blend_alpha_target=bat,
                decay_lo=d_lo, decay_hi=d_hi,
                alpha_floor_late_round=afl,
            )
            mini = train_mini_model(rg_so_far, feature_cols, min_rg=min_rg) if not rg_so_far.empty else None
            blend_p = blend_probas(base_model, mini, X_rn, feature_cols, alpha=alpha)

            total_brier += brier_score_loss(y_rn, blend_p) * len(y_rn)
            total_n += len(y_rn)
            rg_seen.append(rn_df)

    return total_brier / total_n if total_n > 0 else np.inf


def run_blend_optuna(
    all_feat: pd.DataFrame,
    rg_feat: pd.DataFrame,
    xgb_params: dict | None = None,
    calibration_method: str = "isotonic",
    calibration_cv: int = 4,
    temporal_lambda: float = 0.0,
    n_trials: int = 50,
    feature_cols: list[str] = FEATURE_COLS,
    on_trial_end: Optional[Callable[[int, float, dict], None]] = None,
) -> optuna.Study:
    """
    Optimise les paramètres de blend (alpha, decay, min_rg) avec Optuna.
    Les modèles XGBoost de base sont pré-entraînés une seule fois avant l'étude.
    """
    from model import XGB_PARAMS

    if xgb_params is None:
        xgb_params = XGB_PARAMS

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Pré-entraînement des {len(CV_YEARS)} modèles de base (fait une seule fois)...")
    base_models = _pretrain_base_models(
        all_feat, rg_feat, xgb_params, calibration_method, calibration_cv,
        temporal_lambda, feature_cols, CV_YEARS,
    )
    print(f"  {len(base_models)} modèles prêts. Lancement de l'étude blend...")

    study = optuna.create_study(
        study_name="blend_params",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial: optuna.Trial) -> float:
        blend_alpha = trial.suggest_float("blend_alpha", 0.50, 0.90)
        blend_alpha_target = trial.suggest_float("blend_alpha_target", 0.30, blend_alpha - 0.05)
        params = {
            "blend_alpha":             blend_alpha,
            "blend_alpha_target":      blend_alpha_target,
            "alpha_floor_late_round":  trial.suggest_float("alpha_floor_late_round", 0.60, 0.95),
            "decay_lo":                trial.suggest_int("decay_lo", 5, 40),
            "decay_hi":                trial.suggest_int("decay_hi", 50, 300),
            "min_rg_for_blend":        trial.suggest_int("min_rg_for_blend", 5, 40),
        }
        return _score_blend_params(params, rg_feat, base_models, feature_cols, CV_YEARS)

    def _cb(study: optuna.Study, trial: optuna.Trial) -> None:
        if on_trial_end is not None:
            on_trial_end(trial.number + 1, study.best_value, study.best_params)

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    return study


def save_best_blend_params(study: optuna.Study) -> dict:
    """Sauvegarde les meilleurs paramètres blend dans best_blend_params.json."""
    best = study.best_params.copy()
    best["_best_brier"] = study.best_value
    best["_n_trials"] = len(study.trials)
    BEST_BLEND_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    return best


def load_best_blend_params() -> Optional[dict]:
    """Charge les meilleurs paramètres blend, None si absent."""
    if not BEST_BLEND_PARAMS_PATH.exists():
        return None
    return json.loads(BEST_BLEND_PARAMS_PATH.read_text())


def get_blend_params(best: dict) -> dict:
    """Extrait les paramètres blend purs (enlève les méta-clés)."""
    skip = {"_best_brier", "_n_trials"}
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
