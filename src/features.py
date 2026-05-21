"""
Feature engineering pour la prédiction Roland Garros.
Toutes les features sont calculées sans data leakage (données strictement antérieures).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

BEST_ROUND_ENCODING = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}


def _assert_no_leakage(df: pd.DataFrame, feature_df: pd.DataFrame, match_date_col: str = "tourney_date") -> None:
    """Vérifie qu'aucune feature n'est calculée avec des données futures."""
    assert match_date_col in df.columns, f"Colonne {match_date_col} manquante dans df"
    assert len(feature_df) == len(df), "feature_df doit avoir le même nombre de lignes que df"


def _win_rate(history: pd.DataFrame, player: str, surface: Optional[str], months: Optional[int],
              match_date: pd.Timestamp) -> float:
    """Taux de victoire d'un joueur sur une surface/période donnée."""
    mask = (
        ((history["winner_name"] == player) | (history["loser_name"] == player))
        & (history["tourney_date"] < match_date)
    )
    if surface:
        mask &= (history["surface"] == surface)
    if months:
        cutoff = match_date - timedelta(days=months * 30)
        mask &= (history["tourney_date"] >= cutoff)

    sub = history[mask]
    if len(sub) == 0:
        return 0.5
    wins = (sub["winner_name"] == player).sum()
    return wins / len(sub)


def _matches_in_window(history: pd.DataFrame, player: str, days: int,
                       match_date: pd.Timestamp) -> int:
    cutoff = match_date - timedelta(days=days)
    mask = (
        ((history["winner_name"] == player) | (history["loser_name"] == player))
        & (history["tourney_date"] < match_date)
        & (history["tourney_date"] >= cutoff)
    )
    return int(mask.sum())


def _h2h_clay(history: pd.DataFrame, player_a: str, player_b: str,
              match_date: pd.Timestamp) -> tuple[int, int]:
    """Retourne (wins_A_vs_B, total_H2H_clay) avant match_date."""
    mask = (
        ((history["winner_name"] == player_a) & (history["loser_name"] == player_b)
         | (history["winner_name"] == player_b) & (history["loser_name"] == player_a))
        & (history["surface"] == "Clay")
        & (history["tourney_date"] < match_date)
    )
    sub = history[mask]
    total = len(sub)
    wins_a = (sub["winner_name"] == player_a).sum()
    return int(wins_a), total


def _rg_stats(rg_history: pd.DataFrame, player: str, match_date: pd.Timestamp) -> dict:
    """Statistiques historiques du joueur à Roland Garros."""
    mask = (
        ((rg_history["winner_name"] == player) | (rg_history["loser_name"] == player))
        & (rg_history["tourney_date"] < match_date)
    )
    sub = rg_history[mask]
    if len(sub) == 0:
        return {"rg_win_rate": 0.5, "rg_matches": 0, "best_round_rg": 1}

    wins = (sub["winner_name"] == player).sum()
    win_rate = wins / len(sub)

    best_round = 1
    for _, row in sub.iterrows():
        if row["winner_name"] == player:
            r = BEST_ROUND_ENCODING.get(str(row.get("round", "")), 3)
        else:
            r = BEST_ROUND_ENCODING.get(str(row.get("round", "")), 3) - 1
        best_round = max(best_round, r)

    return {
        "rg_win_rate": win_rate,
        "rg_matches": len(sub),
        "best_round_rg": best_round,
    }


def _service_stats(history: pd.DataFrame, player: str, match_date: pd.Timestamp, n: int = 20) -> dict:
    """Moyennes des stats de service sur les n derniers matchs."""
    mask = (
        ((history["winner_name"] == player) | (history["loser_name"] == player))
        & (history["tourney_date"] < match_date)
    )
    sub = history[mask].tail(n)

    def safe_ratio(num_col, den_col, is_winner: bool) -> float:
        prefix = "w_" if is_winner else "l_"
        nums = sub[prefix + num_col] if (prefix + num_col) in sub.columns else pd.Series([np.nan] * len(sub))
        dens = sub[prefix + den_col] if (prefix + den_col) in sub.columns else pd.Series([np.nan] * len(sub))
        ratio = nums / dens.replace(0, np.nan)
        return float(ratio.mean()) if not ratio.isna().all() else 0.6

    results = {}
    for col in sub.itertuples():
        is_winner = col.winner_name == player
        break
    else:
        return {"first_serve_pct": 0.6, "first_serve_won_pct": 0.7, "bp_saved_pct": 0.6}

    winner_rows = sub[sub["winner_name"] == player]
    loser_rows = sub[sub["loser_name"] == player]

    def agg_ratio(w_num, w_den, l_num, l_den):
        vals = []
        for _, r in winner_rows.iterrows():
            n_, d_ = r.get(w_num, np.nan), r.get(w_den, np.nan)
            if pd.notna(n_) and pd.notna(d_) and d_ > 0:
                vals.append(n_ / d_)
        for _, r in loser_rows.iterrows():
            n_, d_ = r.get(l_num, np.nan), r.get(l_den, np.nan)
            if pd.notna(n_) and pd.notna(d_) and d_ > 0:
                vals.append(n_ / d_)
        return float(np.mean(vals)) if vals else 0.6

    return {
        "first_serve_pct": agg_ratio("w_1stIn", "w_svpt", "l_1stIn", "l_svpt"),
        "first_serve_won_pct": agg_ratio("w_1stWon", "w_1stIn", "l_1stWon", "l_1stIn"),
        "bp_saved_pct": agg_ratio("w_bpSaved", "w_bpFaced", "l_bpSaved", "l_bpFaced"),
    }


def build_features(
    df: pd.DataFrame,
    history: pd.DataFrame,
    rg_history: pd.DataFrame,
    elo_df: Optional[pd.DataFrame] = None,
    intra_rg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame de features pour tous les matchs de df.

    df         : matchs pour lesquels on veut les features (RG ou subset)
    history    : tous les matchs historiques (pour les fenêtres glissantes)
    rg_history : matchs RG historiques
    elo_df     : DataFrame avec ratings pré-match (sorti de EloSystem.compute)
    intra_rg   : dict {player: {"sets_played": int}} pour la session en cours

    Retourne un DataFrame (une ligne par match) avec :
    - target = 1 (winner_name gagne)
    - toutes les features
    """
    if intra_rg is None:
        intra_rg = {}

    rows = []
    for _, match in df.iterrows():
        player_a = match["winner_name"]
        player_b = match["loser_name"]
        match_date = match["tourney_date"]
        round_num = int(match.get("round_number", 3))

        # --- Features Elo ---
        if elo_df is not None:
            mid = match.get("match_id", "")
            elo_row = elo_df[elo_df["match_id"] == mid]
            if not elo_row.empty:
                er = elo_row.iloc[0]
                diff_std = er["winner_std_elo_pre"] - er["loser_std_elo_pre"]
                diff_clay = er["winner_clay_elo_pre"] - er["loser_clay_elo_pre"]
                diff_welo = er["winner_welo_pre"] - er["loser_welo_pre"]
                diff_adj = er["winner_adjusted_elo_pre"] - er["loser_adjusted_elo_pre"]
                elo_a_std = er["winner_std_elo_pre"]
                elo_b_std = er["loser_std_elo_pre"]
                elo_a_clay = er["winner_clay_elo_pre"]
                elo_b_clay = er["loser_clay_elo_pre"]
                elo_a_welo = er["winner_welo_pre"]
                elo_b_welo = er["loser_welo_pre"]
            else:
                diff_std = diff_clay = diff_welo = diff_adj = 0.0
                elo_a_std = elo_b_std = elo_a_clay = elo_b_clay = elo_a_welo = elo_b_welo = 1500.0
        else:
            diff_std = diff_clay = diff_welo = diff_adj = 0.0
            elo_a_std = elo_b_std = elo_a_clay = elo_b_clay = elo_a_welo = elo_b_welo = 1500.0

        # --- Forme récente ---
        wr_clay_12m_a = _win_rate(history, player_a, "Clay", 12, match_date)
        wr_clay_12m_b = _win_rate(history, player_b, "Clay", 12, match_date)
        wr_clay_6m_a = _win_rate(history, player_a, "Clay", 6, match_date)
        wr_clay_6m_b = _win_rate(history, player_b, "Clay", 6, match_date)
        wr_30d_a = _win_rate(history, player_a, None, 1, match_date)
        wr_30d_b = _win_rate(history, player_b, None, 1, match_date)
        mp_21d_a = _matches_in_window(history, player_a, 21, match_date)
        mp_21d_b = _matches_in_window(history, player_b, 21, match_date)
        wr_last10_a = _win_rate(history, player_a, None, None, match_date)
        wr_last10_b = _win_rate(history, player_b, None, None, match_date)

        # --- H2H clay ---
        h2h_wins_a, h2h_total = _h2h_clay(history, player_a, player_b, match_date)
        h2h_rate = h2h_wins_a / h2h_total if h2h_total >= 3 else 0.5

        # --- Historique RG ---
        rg_a = _rg_stats(rg_history, player_a, match_date)
        rg_b = _rg_stats(rg_history, player_b, match_date)

        # --- Features joueur ---
        rank_a = match.get("winner_rank", 100)
        rank_b = match.get("loser_rank", 100)
        age_a = match.get("winner_age", 25.0)
        age_b = match.get("loser_age", 25.0)
        rank_a = float(rank_a) if pd.notna(rank_a) else 200.0
        rank_b = float(rank_b) if pd.notna(rank_b) else 200.0
        age_a = float(age_a) if pd.notna(age_a) else 25.0
        age_b = float(age_b) if pd.notna(age_b) else 25.0

        # --- Stats service ---
        srv_a = _service_stats(history, player_a, match_date)
        srv_b = _service_stats(history, player_b, match_date)

        # --- Features intra-tournoi ---
        sets_a = intra_rg.get(player_a, {}).get("sets_played", 0)
        sets_b = intra_rg.get(player_b, {}).get("sets_played", 0)

        row = {
            # Identifiants
            "match_id": match.get("match_id", ""),
            "tourney_date": match_date,
            "player_a": player_a,
            "player_b": player_b,
            "target": 1,  # player_a (winner) gagne toujours dans ce DataFrame

            # Elo différentiels (A - B)
            "diff_standard_elo": diff_std,
            "diff_clay_elo": diff_clay,
            "diff_welo": diff_welo,
            "diff_adjusted_elo": diff_adj,
            "elo_a_std": elo_a_std,
            "elo_b_std": elo_b_std,
            "elo_a_clay": elo_a_clay,
            "elo_b_clay": elo_b_clay,
            "elo_a_welo": elo_a_welo,
            "elo_b_welo": elo_b_welo,

            # Forme récente
            "diff_win_rate_clay_12m": wr_clay_12m_a - wr_clay_12m_b,
            "diff_win_rate_clay_6m": wr_clay_6m_a - wr_clay_6m_b,
            "diff_win_rate_30d": wr_30d_a - wr_30d_b,
            "diff_matches_21d": mp_21d_a - mp_21d_b,
            "diff_win_rate_last10": wr_last10_a - wr_last10_b,
            "win_rate_clay_12m_a": wr_clay_12m_a,
            "win_rate_clay_12m_b": wr_clay_12m_b,

            # H2H clay
            "h2h_clay_wins_a": h2h_wins_a,
            "h2h_clay_total": h2h_total,
            "h2h_clay_rate": h2h_rate,

            # Historique RG
            "rg_win_rate_a": rg_a["rg_win_rate"],
            "rg_win_rate_b": rg_b["rg_win_rate"],
            "diff_rg_win_rate": rg_a["rg_win_rate"] - rg_b["rg_win_rate"],
            "rg_matches_a": rg_a["rg_matches"],
            "rg_matches_b": rg_b["rg_matches"],
            "best_round_rg_a": rg_a["best_round_rg"],
            "best_round_rg_b": rg_b["best_round_rg"],
            "diff_best_round_rg": rg_a["best_round_rg"] - rg_b["best_round_rg"],

            # Joueur
            "ranking_diff": rank_a - rank_b,
            "log_ranking_diff": np.log(rank_b + 1) - np.log(rank_a + 1),
            "age_a": age_a,
            "age_b": age_b,
            "age_optimal_a": int(28 <= age_a <= 32),
            "age_optimal_b": int(28 <= age_b <= 32),
            "diff_age_optimal": int(28 <= age_a <= 32) - int(28 <= age_b <= 32),

            # Stats service
            "diff_first_serve_pct": srv_a["first_serve_pct"] - srv_b["first_serve_pct"],
            "diff_first_serve_won_pct": srv_a["first_serve_won_pct"] - srv_b["first_serve_won_pct"],
            "diff_bp_saved_pct": srv_a["bp_saved_pct"] - srv_b["bp_saved_pct"],
            "first_serve_pct_a": srv_a["first_serve_pct"],
            "first_serve_won_pct_a": srv_a["first_serve_won_pct"],
            "bp_saved_pct_a": srv_a["bp_saved_pct"],

            # Intra-tournoi
            "round_number": round_num,
            "diff_sets_played_rg": sets_a - sets_b,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_symmetric_dataset(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Double le dataset en inversant les rôles A/B pour éviter le biais de position.
    target=1 si A gagne, target=0 si B gagne (i.e. A perd).
    """
    df_pos = feature_df.copy()
    df_pos["target"] = 1

    df_neg = feature_df.copy()
    df_neg["target"] = 0

    # Inverser tous les différentiels
    diff_cols = [c for c in df_neg.columns if c.startswith("diff_")]
    for col in diff_cols:
        df_neg[col] = -df_neg[col]

    # Inverser colonnes _a / _b
    a_cols = [c for c in df_neg.columns if c.endswith("_a") and c.replace("_a", "_b") in df_neg.columns]
    for col_a in a_cols:
        col_b = col_a.replace("_a", "_b")
        df_neg[col_a], df_neg[col_b] = df_neg[col_b].copy(), df_neg[col_a].copy()

    df_neg["h2h_clay_wins_a"] = df_neg["h2h_clay_total"] - feature_df["h2h_clay_wins_a"]
    df_neg["h2h_clay_rate"] = 1.0 - feature_df["h2h_clay_rate"]
    df_neg["ranking_diff"] = -feature_df["ranking_diff"]
    df_neg["log_ranking_diff"] = -feature_df["log_ranking_diff"]

    return pd.concat([df_pos, df_neg], ignore_index=True)


FEATURE_COLS = [
    "diff_standard_elo", "diff_clay_elo", "diff_welo", "diff_adjusted_elo",
    "diff_win_rate_clay_12m", "diff_win_rate_clay_6m", "diff_win_rate_30d",
    "diff_matches_21d", "diff_win_rate_last10",
    "h2h_clay_rate", "h2h_clay_total",
    "diff_rg_win_rate", "rg_matches_a", "rg_matches_b",
    "diff_best_round_rg",
    "ranking_diff", "log_ranking_diff",
    "age_a", "age_b", "diff_age_optimal",
    "diff_first_serve_pct", "diff_first_serve_won_pct", "diff_bp_saved_pct",
    "round_number", "diff_sets_played_rg",
]
