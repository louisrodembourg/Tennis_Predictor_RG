"""
Module de backtesting standalone — évaluation sur RG 2017-2025.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import filter_clay, filter_roland_garros, load_matches
from elo import EloSystem
from features import FEATURE_COLS, HistoryIndex, build_features
from model import BacktestResult, compare_baselines, expanding_window_backtest


STEPS_TOTAL = 7


def _step(n: int, label: str) -> float:
    tqdm.write(f"\n[{n}/{STEPS_TOTAL}] {label}...")
    return time.time()


def _done(t0: float, detail: str = "") -> None:
    elapsed = time.time() - t0
    suffix = f"  ({detail})" if detail else ""
    tqdm.write(f"      ✓ {elapsed:.1f}s{suffix}")


def run_full_backtest(data_dir: str, year_start: int = 2000) -> dict:
    """
    Lance le pipeline complet de backtesting.
    Retourne un dict avec les résultats et le DataFrame de comparaison.
    """
    print("=" * 60)
    print("  BACKTESTING ROLAND GARROS 2017-2025")
    print("=" * 60)

    # [1/6] Chargement
    t = _step(1, "Chargement des données ATP")
    df = load_matches(data_dir, year_start=year_start)
    clay_df = filter_clay(df)
    rg_df = filter_roland_garros(df)
    _done(t, f"{len(df):,} matchs — {len(clay_df):,} clay — {len(rg_df):,} RG")

    # [2/6] Elo
    t = _step(2, "Calcul des ratings Elo (Standard / Surface / WElo / Adjusted)")
    elo = EloSystem(alpha=0.3, lambda_adj=0.5)
    df_elo = elo.compute(df)
    _done(t, f"{len(elo._ratings):,} joueurs indexés")

    # [3/6] Index historique (construit une seule fois, réutilisé pour clay + RG)
    t = _step(3, f"Construction de l'index historique ({len(df):,} matchs × 2 joueurs)")
    hist_index = HistoryIndex(df, rg_df)
    n_indexed = len(hist_index._players)
    _done(t, f"{n_indexed:,} joueurs indexés")

    # [4/6] Features clay
    clay_with_elo = df_elo[df_elo["surface"] == "Clay"].copy()
    t = _step(4, f"Construction features — matchs clay ({len(clay_with_elo):,} matchs)")
    all_feat = build_features(
        df=clay_with_elo,
        history=df,
        rg_history=rg_df,
        elo_df=df_elo,
        desc="  Features clay",
        index=hist_index,          # réutilise l'index pré-calculé
    )
    _done(t, f"{len(all_feat):,} lignes × {all_feat.shape[1]} colonnes")

    # [5/6] Features RG  (même index, pas de reconstruction)
    rg_with_elo = df_elo[
        df_elo["tourney_name"].str.contains("Roland Garros|French Open", case=False, na=False)
    ].copy()
    t = _step(5, f"Construction features — Roland Garros ({len(rg_with_elo):,} matchs)")
    rg_feat = build_features(
        df=rg_with_elo,
        history=df,
        rg_history=rg_df,
        elo_df=df_elo,
        desc="  Features RG  ",
        index=hist_index,          # réutilise le même index
    )
    _done(t, f"{len(rg_feat):,} matchs RG ({rg_feat['tourney_date'].dt.year.min()}–{rg_feat['tourney_date'].dt.year.max()})")

    # [6/7] Expanding window backtest
    t = _step(6, "Expanding window backtesting (RG 2017-2025)")
    results = expanding_window_backtest(all_feat, rg_feat)
    _done(t, f"{len(results)} éditions évaluées")

    # [7/7] Comparaison baselines
    t = _step(7, "Comparaison avec les baselines (Ranking / Clay Elo / WElo / XGBoost)")
    baseline_df = compare_baselines(rg_feat, all_feat)
    _done(t)

    tqdm.write("\n" + "=" * 60)
    tqdm.write("  Pipeline terminé.")
    tqdm.write("=" * 60)

    return {
        "backtest_results": results,
        "baseline_comparison": baseline_df,
        "all_features": all_feat,
        "rg_features": rg_feat,
    }


def print_summary(results: list[BacktestResult], baseline_df: pd.DataFrame) -> None:
    """Affiche le tableau récapitulatif dans le terminal."""
    print("\n" + "=" * 80)
    print("RÉSULTATS EXPANDING WINDOW (RG 2017-2025)")
    print("=" * 80)
    print(f"{'Année':<8} {'Acc':<8} {'Brier':<8} {'LogLoss':<10} {'N matchs':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r.year:<8} {r.accuracy:.3f}   {r.brier:.4f}  {r.log_loss_val:.4f}    {r.n_matches:<10}")

    if results:
        avg_acc = np.mean([r.accuracy for r in results])
        avg_brier = np.mean([r.brier for r in results])
        print("-" * 50)
        print(f"{'Moyenne':<8} {avg_acc:.3f}   {avg_brier:.4f}")

    print("\n" + "=" * 80)
    print("COMPARAISON BASELINES")
    print("=" * 80)
    if not baseline_df.empty:
        cols = ["year", "n", "baseline_rank_acc", "baseline_clay_elo_acc",
                "baseline_welo_adj_acc", "xgboost_acc"]
        available = [c for c in cols if c in baseline_df.columns]
        print(baseline_df[available].to_string(index=False, float_format="{:.3f}".format))

    print("\n" + "=" * 80)
    print("DÉCOMPOSITION PAR TOUR (toutes années)")
    print("=" * 80)
    all_by_round = []
    for r in results:
        df_r = r.by_round.copy()
        df_r["year"] = r.year
        all_by_round.append(df_r)

    if all_by_round:
        combined = pd.concat(all_by_round)
        round_names = {1: "R128", 2: "R64", 3: "R32", 4: "R16", 5: "QF", 6: "SF", 7: "F"}
        combined["round_name"] = combined["round"].map(round_names)
        pivot = combined.groupby("round_name")["accuracy"].agg(["mean", "std", "count"])
        print(pivot.to_string(float_format="{:.3f}".format))
