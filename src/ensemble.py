"""
Ensemble LightGBM + XGBoost avec meta-learner (stacking).

Usage:
    lgbm_model = train_lgbm(train_df, feature_cols)
    meta = train_stacked_ensemble(xgb_model, lgbm_model, cal_df, feature_cols)
    p = stacked_predict_proba(xgb_model, lgbm_model, meta, X, feature_cols)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from features import FEATURE_COLS, build_symmetric_dataset

LGBM_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def train_lgbm(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    calibrate: bool = True,
    lgbm_params: dict | None = None,
    calibration_cv: int = 4,
    temporal_lambda: float = 0.0,
) -> LGBMClassifier | CalibratedClassifierCV:
    """Train a calibrated LightGBM model. Same interface as model.train_model()."""
    sym_df = build_symmetric_dataset(feature_df)
    X = sym_df[feature_cols].fillna(0)
    y = sym_df["target"]

    sample_weight = None
    if temporal_lambda > 0.0 and "tourney_date" in sym_df.columns:
        ref_date = pd.to_datetime(sym_df["tourney_date"]).max()
        years_ago = (ref_date - pd.to_datetime(sym_df["tourney_date"])).dt.days / 365.25
        sample_weight = np.exp(-temporal_lambda * years_ago.values)

    params = lgbm_params if lgbm_params is not None else LGBM_PARAMS
    fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}

    if calibrate:
        model = CalibratedClassifierCV(LGBMClassifier(**params), cv=calibration_cv, method="isotonic")
        model.fit(X, y, **fit_kwargs)
    else:
        model = LGBMClassifier(**params)
        model.fit(X, y, **fit_kwargs)
    return model


def train_stacked_ensemble(
    xgb_model,
    lgbm_model,
    cal_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> Optional[LogisticRegression]:
    """
    Train a logistic regression meta-learner on held-out calibration data.
    Inputs: [p_xgb, p_lgbm]. Returns None if cal_df is too small.
    cal_df must NOT overlap with the data used to train xgb_model/lgbm_model.
    """
    sym_cal = build_symmetric_dataset(cal_df)
    if len(sym_cal) < 20:
        return None

    X_cal = sym_cal[feature_cols].fillna(0)
    y_cal = sym_cal["target"].values

    p_xgb  = xgb_model.predict_proba(X_cal)[:, 1]
    p_lgbm = lgbm_model.predict_proba(X_cal)[:, 1]
    meta_X  = np.column_stack([p_xgb, p_lgbm])

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(meta_X, y_cal)
    return meta


def stacked_predict_proba(
    xgb_model,
    lgbm_model,
    meta: Optional[LogisticRegression],
    X: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    """
    Return stacked probabilities. Falls back to simple average if meta is None.
    """
    X_arr = X[feature_cols].fillna(0)
    p_xgb  = xgb_model.predict_proba(X_arr)[:, 1]
    p_lgbm = lgbm_model.predict_proba(X_arr)[:, 1]

    if meta is None:
        return 0.5 * p_xgb + 0.5 * p_lgbm

    meta_X = np.column_stack([p_xgb, p_lgbm])
    return meta.predict_proba(meta_X)[:, 1]
