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

from cache_manager import load_cache
from data_loader import filter_clay, filter_roland_garros, load_matches
from elo import EloSystem
from features import FEATURE_COLS, HistoryIndex, build_features
from model import BacktestResult, compare_baselines, expanding_window_backtest


def _step(n: int, label: str) -> float:
    tqdm.write(f"\n[{n}] {label}...")
    return time.time()


def _done(t0: float, detail: str = "") -> None:
    elapsed = time.time() - t0
    suffix = f"  ({detail})" if detail else ""
    tqdm.write(f"      ✓ {elapsed:.1f}s{suffix}")


def _index_from_cache(cached: dict) -> HistoryIndex:
    """Construit un HistoryIndex minimal depuis les tables d'état cachées.
    build_features n'utilise que player_state/rg_state/h2h_state — pas les dicts _players/_rg."""
    idx = HistoryIndex.__new__(HistoryIndex)
    idx._players = {}
    idx._rg = {}
    idx.player_state = cached["player_state"]
    idx.rg_state     = cached["rg_state"]
    idx.h2h_state    = cached["h2h_state"]
    return idx


def run_full_backtest(data_dir: str, year_start: int = 2000) -> dict:
    """
    Lance le pipeline de backtesting.
    Utilise le cache persistant pour éviter de recalculer CSV + Elo + features clay.
    Retourne un dict avec les résultats et le DataFrame de comparaison.
    """
    print("=" * 60)
    print("  BACKTESTING ROLAND GARROS 2017-2025")
    print("=" * 60)

    cached = load_cache(data_dir)

    if cached:
        # ---------------------------------------------------------------
        # Chemin rapide : CSV + Elo + HistoryIndex + features clay depuis cache
        # ---------------------------------------------------------------
        tqdm.write("\n[1-4] Données chargées depuis le cache (CSV + Elo + features clay).")
        df       = cached["df"]
        df_elo   = cached["df_with_elo"]
        rg_df    = filter_roland_garros(df)
        all_feat = cached["all_features"]
        # Exclure les qualifications du jeu d'entraînement (dynamiques trop différentes)
        all_feat = all_feat[all_feat["round_number"] >= 1].copy()
        hist_index = _index_from_cache(cached)
        tqdm.write(f"       {len(df):,} matchs — features clay {len(all_feat):,} lignes (hors qualifs)")
    else:
        # ---------------------------------------------------------------
        # Chemin complet (premier lancement ou cache invalidé)
        # ---------------------------------------------------------------
        t = _step(1, "Chargement des données ATP")
        df = load_matches(data_dir, year_start=year_start)
        clay_df = filter_clay(df)
        rg_df = filter_roland_garros(df)
        _done(t, f"{len(df):,} matchs — {len(clay_df):,} clay — {len(rg_df):,} RG")

        t = _step(2, "Calcul des ratings Elo (Standard / Surface / WElo / Adjusted)")
        elo = EloSystem(alpha=0.3, lambda_adj=0.5)
        df_elo = elo.compute(df)
        _done(t, f"{len(elo._ratings):,} joueurs indexés")

        t = _step(3, f"Construction de l'index historique ({len(df):,} matchs × 2 joueurs)")
        hist_index = HistoryIndex(df, rg_df)
        _done(t, f"{len(hist_index._players):,} joueurs indexés")

        clay_with_elo = df_elo[df_elo["surface"] == "Clay"].copy()
        t = _step(4, f"Construction features clay ({len(clay_with_elo):,} matchs)")
        all_feat = build_features(
            df=clay_with_elo, history=df, rg_history=rg_df,
            elo_df=df_elo, desc="  Features clay", index=hist_index,
        )
        all_feat = all_feat[all_feat["round_number"] >= 1].copy()
        _done(t, f"{len(all_feat):,} lignes × {all_feat.shape[1]} colonnes (hors qualifs)")

    # [5] Features RG — toujours recalculées (légères, ~1 500 matchs)
    rg_with_elo = df_elo[
        df_elo["tourney_name"].str.contains("Roland Garros|French Open", case=False, na=False)
    ].copy()
    t = _step(5, f"Construction features Roland Garros ({len(rg_with_elo):,} matchs)")
    rg_feat = build_features(
        df=rg_with_elo, history=df, rg_history=rg_df,
        elo_df=df_elo, desc="  Features RG  ", index=hist_index,
    )
    _done(t, f"{len(rg_feat):,} matchs RG ({rg_feat['tourney_date'].dt.year.min()}–{rg_feat['tourney_date'].dt.year.max()})")

    # [6] Expanding window backtest
    t = _step(6, "Expanding window backtesting (RG 2017-2025)")
    results = expanding_window_backtest(all_feat, rg_feat, rg_raw_df=rg_with_elo)
    _done(t, f"{len(results)} éditions évaluées")

    # [7] Comparaison baselines
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
    print("\n" + "=" * 90)
    print("RÉSULTATS EXPANDING WINDOW (RG 2017-2025)")
    print("=" * 90)
    print(f"{'Année':<8} {'Acc Base':<10} {'Brier Base':<12} {'Acc Blend':<10} {'Brier Blend':<12} {'N matchs'}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.year:<8} {r.accuracy:.3f}      {r.brier:.4f}       "
            f"{r.accuracy_blend:.3f}      {r.brier_blend:.4f}       {r.n_matches}"
        )

    if results:
        avg_acc   = np.mean([r.accuracy for r in results])
        avg_brier = np.mean([r.brier for r in results])
        avg_acc_bl   = np.mean([r.accuracy_blend for r in results])
        avg_brier_bl = np.mean([r.brier_blend for r in results])
        print("-" * 65)
        print(f"{'Moyenne':<8} {avg_acc:.3f}      {avg_brier:.4f}       {avg_acc_bl:.3f}      {avg_brier_bl:.4f}")

    print("\n" + "=" * 90)
    print("COMPARAISON BASELINES")
    print("=" * 90)
    if not baseline_df.empty:
        cols = ["year", "n", "baseline_rank_acc", "baseline_clay_elo_acc",
                "baseline_welo_adj_acc", "xgboost_acc"]
        available = [c for c in cols if c in baseline_df.columns]
        print(baseline_df[available].to_string(index=False, float_format="{:.3f}".format))

    print("\n" + "=" * 90)
    print("DÉCOMPOSITION PAR TOUR (toutes années — moyenne Base vs Blend)")
    print("=" * 90)
    all_by_round = []
    for r in results:
        df_r = r.by_round.copy()
        df_r["year"] = r.year
        all_by_round.append(df_r)

    if all_by_round:
        combined = pd.concat(all_by_round, ignore_index=True)
        pivot = combined.groupby("round_name")[["acc_base", "acc_blend", "n"]].agg(
            {"acc_base": "mean", "acc_blend": "mean", "n": "sum"}
        )
        pivot.columns = ["Acc Base (moy)", "Acc Blend (moy)", "N matchs"]
        round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
        pivot = pivot.reindex([r for r in round_order if r in pivot.index])
        print(pivot.to_string(float_format="{:.3f}".format))
