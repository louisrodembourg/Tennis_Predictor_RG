"""
SHAP feature importance analysis for the TennisPredictor XGBoost model.

Usage:
    cd /path/to/TennisPredictor
    python scripts/shap_analysis.py

Outputs a ranked table of mean |SHAP| per feature. Features with
mean |SHAP| < 0.002 are candidates for removal from FEATURE_COLS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, build_symmetric_dataset
from model import train_model, MAIN_DRAW_ROUNDS


THRESHOLD = 0.002   # features below this mean |SHAP| are flagged
CV_TRAIN_YEARS = list(range(2000, 2023))  # all clay data before 2023 RG


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-built features from cache or run full pipeline."""
    cache_dir = ROOT / "data" / "cache"
    all_feat_path  = cache_dir / "all_features_v3.parquet"
    rg_feat_path   = cache_dir / "rg_features_v3.parquet"

    if all_feat_path.exists() and rg_feat_path.exists():
        print("Loading features from cache...")
        all_feat = pd.read_parquet(all_feat_path)
        rg_feat  = pd.read_parquet(rg_feat_path)
        return all_feat, rg_feat

    print("Cache not found — running full feature pipeline (may take a few minutes)...")
    from backtesting import run_full_backtest
    bt = run_full_backtest(str(ROOT / "data" / "raw"))
    return bt["all_features"], bt["rg_features"]


def main() -> None:
    all_feat, rg_feat = load_data()

    # Train on all clay data before RG 2023 (latest year in CV_TRAIN_YEARS)
    rg_2023 = rg_feat[rg_feat["tourney_date"].dt.year == 2023]
    if rg_2023.empty:
        raise RuntimeError("No RG 2023 data found in rg_features.")
    cutoff = rg_2023["tourney_date"].min()
    train_df = all_feat[all_feat["tourney_date"] < cutoff]

    print(f"Training XGBoost on {len(train_df):,} clay matches (before RG 2023)...")
    model = train_model(train_df, FEATURE_COLS, calibrate=False)

    # Build training X for SHAP
    sym = build_symmetric_dataset(train_df)
    X_train = sym[FEATURE_COLS].fillna(0)

    # Use a representative sample to speed up SHAP (max 5000 rows)
    sample_size = min(5000, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)

    print(f"Computing SHAP values on {sample_size:,} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Feature Importance (mean |SHAP|) ===")
    print(importance_df.to_string(index=False))

    low_impact = importance_df[importance_df["mean_abs_shap"] < THRESHOLD]
    if low_impact.empty:
        print(f"\nAll features have mean |SHAP| >= {THRESHOLD}. No pruning candidates.")
    else:
        print(f"\n=== Pruning candidates (mean |SHAP| < {THRESHOLD}) ===")
        print(low_impact[["feature", "mean_abs_shap"]].to_string(index=False))
        print("\nConsider removing these from FEATURE_COLS in src/features.py")

    out_path = ROOT / "data" / "shap_importance.csv"
    importance_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
