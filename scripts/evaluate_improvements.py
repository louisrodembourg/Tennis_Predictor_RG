#!/usr/bin/env python3
"""
Evaluation script — measures the accuracy impact of each improvement track.

Tracks tested:
  Track 2 : New features (H2H recency, return stats, quality win rate)
  Track 3 : LightGBM ensemble (XGB + LGBM stacking)
  Track 1 : Optuna hyperparameter search (quick, 30 trials)
  Track 4 : SHAP feature importance (ranking + pruning candidates)

Usage:
    cd /path/to/TennisPredictor
    python scripts/evaluate_improvements.py [--optuna-trials 30] [--skip-optuna]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# Baseline reference (from last run before improvements)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE = {
    "acc_base":  0.9114,
    "brier_base": 0.0665,
    "acc_blend": 0.9150,
    "brier_blend": 0.0676,
}


def delta(new: float, ref: float, pct: bool = True) -> str:
    d = new - ref
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{sign}{d*100:.2f}pp"
    return f"{sign}{d:.4f}"


def separator(title: str = "", width: int = 72) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def load_pipeline(data_dir: str) -> dict:
    """Run the full data + feature pipeline (with cache)."""
    from backtesting import run_full_backtest
    return run_full_backtest(data_dir)


def run_backtest_config(
    all_feat: pd.DataFrame,
    rg_feat: pd.DataFrame,
    rg_raw: pd.DataFrame,
    xgb_params: dict | None,
    temporal_lambda: float,
    use_ensemble: bool,
    label: str,
) -> dict:
    from model import expanding_window_backtest, MAIN_DRAW_ROUNDS
    import numpy as np

    t0 = time.time()
    results = expanding_window_backtest(
        all_feat, rg_feat,
        rg_raw_df=rg_raw,
        xgb_params=xgb_params,
        temporal_lambda=temporal_lambda,
        use_ensemble=use_ensemble,
    )
    elapsed = time.time() - t0

    avg_acc   = np.mean([r.accuracy       for r in results])
    avg_brier = np.mean([r.brier          for r in results])
    avg_acc_bl  = np.mean([r.accuracy_blend for r in results])
    avg_brier_bl = np.mean([r.brier_blend   for r in results])
    avg_acc_st  = np.mean([r.accuracy_stack for r in results])
    avg_brier_st = np.mean([r.brier_stack   for r in results])

    return {
        "label": label,
        "elapsed": elapsed,
        "results": results,
        "avg_acc_base":   avg_acc,
        "avg_brier_base": avg_brier,
        "avg_acc_blend":  avg_acc_bl,
        "avg_brier_blend": avg_brier_bl,
        "avg_acc_stack":  avg_acc_st,
        "avg_brier_stack": avg_brier_st,
    }


def print_per_year(results, label: str) -> None:
    print(f"\n{'Year':<6} {'Acc Base':>9} {'Brier Base':>11} {'Acc Blend':>10} {'Brier Blend':>12} {'Acc Stack':>10}  ({label})")
    print("-" * 65)
    for r in results:
        print(f"{r.year:<6} {r.accuracy:>9.3f} {r.brier:>11.4f} {r.accuracy_blend:>10.3f} {r.brier_blend:>12.4f} {r.accuracy_stack:>10.3f}")


def print_comparison_table(configs: list[dict]) -> None:
    separator("SUMMARY COMPARISON TABLE")
    ref_b = BASELINE["acc_base"]
    ref_bl = BASELINE["acc_blend"]

    header = f"{'Config':<38} {'Acc Base':>9} {'Δ vs ref':>9} {'Brier Base':>11} {'Acc Blend':>10} {'Δ vs ref':>9} {'Acc Stack':>10}"
    print(header)
    print("-" * len(header))

    ref_row = f"{'[Baseline] XGB v2 features':38} {ref_b:>9.4f} {'':>9} {BASELINE['brier_base']:>11.4f} {ref_bl:>10.4f} {'':>9} {'N/A':>10}"
    print(ref_row)

    for c in configs:
        row = (
            f"{c['label']:<38} "
            f"{c['avg_acc_base']:>9.4f} "
            f"{delta(c['avg_acc_base'],  ref_b):>9} "
            f"{c['avg_brier_base']:>11.4f} "
            f"{c['avg_acc_blend']:>10.4f} "
            f"{delta(c['avg_acc_blend'], ref_bl):>9} "
            f"{c['avg_acc_stack']:>10.4f}"
        )
        print(row)


def run_shap_analysis(all_feat: pd.DataFrame, rg_feat: pd.DataFrame) -> pd.DataFrame:
    import shap
    from features import FEATURE_COLS, build_symmetric_dataset
    from model import train_model, MAIN_DRAW_ROUNDS

    rg_2023 = rg_feat[rg_feat["tourney_date"].dt.year == 2023]
    if rg_2023.empty:
        print("  [WARN] No RG 2023 data for SHAP. Skipping.")
        return pd.DataFrame()

    cutoff = rg_2023["tourney_date"].min()
    train_df = all_feat[all_feat["tourney_date"] < cutoff]
    model = train_model(train_df, FEATURE_COLS, calibrate=False,
                        xgb_params={"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                                    "random_state": 42, "n_jobs": -1, "eval_metric": "logloss"})
    sym = build_symmetric_dataset(train_df)
    X_sample = sym[FEATURE_COLS].fillna(0).sample(min(3000, len(sym)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    # ── Load pipeline ─────────────────────────────────────────────────────────
    separator("LOADING DATA + FEATURES")
    bt = load_pipeline(str(data_dir))
    all_feat = bt["all_features"]
    rg_feat  = bt["rg_features"]
    rg_raw   = bt.get("rg_with_elo", pd.DataFrame())

    # Try to get the raw RG matches for intra-tournament sets tracking
    try:
        from cache_manager import load_cache
        cached = load_cache(str(data_dir))
        if cached:
            df_elo = cached["df_with_elo"]
            if "tourney_name" in df_elo.columns:
                rg_raw = df_elo[df_elo["tourney_name"].str.contains("Roland Garros|French Open", case=False, na=False)].copy()
    except Exception:
        pass

    rg_raw_arg = rg_raw if not rg_raw.empty else None

    configs = []

    # ── Track 2: New features with current XGB params ─────────────────────────
    separator("TRACK 2 — New Features (H2H recency + Return stats + Top-50 win rate)")
    from model import XGB_PARAMS, TEMPORAL_LAMBDA
    print(f"Using current XGB_PARAMS (from model.py) + TEMPORAL_LAMBDA={TEMPORAL_LAMBDA}")
    c2 = run_backtest_config(all_feat, rg_feat, rg_raw_arg,
                             xgb_params=XGB_PARAMS, temporal_lambda=TEMPORAL_LAMBDA,
                             use_ensemble=False, label="[T2] New features, current XGB")
    print_per_year(c2["results"], "Track 2")
    print(f"\n  → Avg Acc Base:  {c2['avg_acc_base']:.4f}  ({delta(c2['avg_acc_base'],  BASELINE['acc_base'])} vs baseline)")
    print(f"  → Avg Brier Base: {c2['avg_brier_base']:.4f}  ({delta(c2['avg_brier_base'], BASELINE['brier_base'], pct=False)} vs baseline)")
    print(f"  → Avg Acc Blend: {c2['avg_acc_blend']:.4f}  ({delta(c2['avg_acc_blend'], BASELINE['acc_blend'])} vs baseline)")
    print(f"  ⏱  {c2['elapsed']:.0f}s")
    configs.append(c2)

    # ── Track 3: LightGBM ensemble ────────────────────────────────────────────
    if not args.skip_ensemble:
        separator("TRACK 3 — LightGBM Ensemble (XGB + LGBM + meta-learner)")
        c3 = run_backtest_config(all_feat, rg_feat, rg_raw_arg,
                                 xgb_params=XGB_PARAMS, temporal_lambda=TEMPORAL_LAMBDA,
                                 use_ensemble=True, label="[T2+T3] New features + ensemble")
        print_per_year(c3["results"], "Track 2+3")
        print(f"\n  → Avg Acc Stack: {c3['avg_acc_stack']:.4f}  ({delta(c3['avg_acc_stack'],  BASELINE['acc_base'])} vs baseline)")
        print(f"  → Avg Brier Stack: {c3['avg_brier_stack']:.4f}")
        print(f"  ⏱  {c3['elapsed']:.0f}s")
        configs.append(c3)
    else:
        print("  [SKIP] Track 3 (--skip-ensemble)")

    # ── Track 1: Optuna ───────────────────────────────────────────────────────
    if not args.skip_optuna:
        separator(f"TRACK 1 — Optuna ({args.optuna_trials} trials, CV 2017-2022)")
        from hyperopt import run_optuna, save_best_params, get_xgb_params, get_temporal_lambda

        n_trials = args.optuna_trials
        print(f"Running {n_trials} Optuna trials (CV years 2017-2022, holdout 2023-2025)...")
        best_brier_so_far: list[float] = []

        def on_trial_end(n_done: int, best_val: float, params: dict) -> None:
            best_brier_so_far.append(best_val)
            if n_done % 5 == 0 or n_done == 1:
                print(f"  Trial {n_done:3d} | Best Brier CV: {best_val:.4f}")

        t0 = time.time()
        study = run_optuna(all_feat, rg_feat, n_trials=n_trials, on_trial_end=on_trial_end)
        print(f"  ⏱  {time.time()-t0:.0f}s  |  Best Brier CV: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        best = save_best_params(study)
        xgb_params_opt = get_xgb_params(best)
        temporal_lambda_opt = get_temporal_lambda(best)
        print(f"  → temporal_lambda = {temporal_lambda_opt:.3f}")

        print("\n  Running holdout backtest with Optuna params (2023-2025)...")
        c1 = run_backtest_config(all_feat, rg_feat, rg_raw_arg,
                                 xgb_params=xgb_params_opt,
                                 temporal_lambda=temporal_lambda_opt,
                                 use_ensemble=False,
                                 label=f"[T1+T2] Optuna({n_trials}t) + new features")
        print_per_year(c1["results"], "Track 1+2")
        print(f"\n  → Avg Acc Base:  {c1['avg_acc_base']:.4f}  ({delta(c1['avg_acc_base'],  BASELINE['acc_base'])} vs baseline)")
        print(f"  → Avg Brier Base: {c1['avg_brier_base']:.4f}  ({delta(c1['avg_brier_base'], BASELINE['brier_base'], pct=False)} vs baseline)")
        print(f"  ⏱  {c1['elapsed']:.0f}s")
        configs.append(c1)

        if not args.skip_ensemble:
            print("\n  + ensemble on top of Optuna params...")
            c13 = run_backtest_config(all_feat, rg_feat, rg_raw_arg,
                                      xgb_params=xgb_params_opt,
                                      temporal_lambda=temporal_lambda_opt,
                                      use_ensemble=True,
                                      label=f"[T1+T2+T3] All tracks")
            print(f"  → Avg Acc Stack: {c13['avg_acc_stack']:.4f}  ({delta(c13['avg_acc_stack'], BASELINE['acc_base'])} vs baseline)")
            configs.append(c13)
    else:
        print("  [SKIP] Track 1 (--skip-optuna)")

    # ── Track 4: SHAP ─────────────────────────────────────────────────────────
    if not args.skip_shap:
        separator("TRACK 4 — SHAP Feature Importance")
        print("Computing SHAP values (3000-sample subset)...")
        try:
            t0 = time.time()
            shap_df = run_shap_analysis(all_feat, rg_feat)
            if not shap_df.empty:
                print(f"\n  Top 15 features:")
                print(shap_df.head(15).to_string(index=False))
                low = shap_df[shap_df["mean_abs_shap"] < 0.002]
                if not low.empty:
                    print(f"\n  Pruning candidates (mean |SHAP| < 0.002):")
                    print(low[["feature", "mean_abs_shap"]].to_string(index=False))
                else:
                    print("  No features below pruning threshold (0.002).")
                out = ROOT / "data" / "shap_importance.csv"
                shap_df.to_csv(out, index=False)
                print(f"  Saved to {out}")
            print(f"  ⏱  {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  [WARN] SHAP failed: {e}")
    else:
        print("  [SKIP] Track 4 (--skip-shap)")

    # ── Final summary ─────────────────────────────────────────────────────────
    separator()
    print_comparison_table(configs)
    separator()
    print("\nDone. Next step: run Optuna with 150+ trials from the dashboard or:")
    print("  python scripts/evaluate_improvements.py --optuna-trials 150 --skip-shap")


if __name__ == "__main__":
    main()
