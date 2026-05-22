"""
Feature engineering pour la prédiction Roland Garros.

Optimisation : HistoryIndex pré-calcule un index {joueur → ses matchs} une seule fois.
Toutes les lookups travaillent sur ~300 lignes au lieu de 156 000 → gain ×50-100.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

BEST_ROUND_ENCODING = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
_EMPTY = pd.DataFrame()

# Colonnes de stats de service dans le format long
_SRV_W = ["w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpFaced", "w_bpSaved"]
_SRV_L = ["l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpFaced", "l_bpSaved"]
_SRV_NORM = ["svpt", "1stIn", "1stWon", "2ndWon", "bpFaced", "bpSaved"]


def _score_to_sets_played(score: pd.Series) -> pd.Series:
    """Estime le nombre de sets joués à partir de la colonne score."""
    text = score.fillna("").astype(str)
    return text.str.count(r"\d+\s*-\s*\d+").astype(float)


def _rolling_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calcule un ratio robuste en évitant les divisions par zéro."""
    denom = denominator.replace(0, np.nan)
    ratio = numerator / denom
    return ratio.fillna(np.nan)


def _merge_asof_by_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    by: list[str],
    on: str = "tourney_date",
    suffix: str = "",
) -> pd.DataFrame:
    """Merge asof stable sur plusieurs clés de groupe."""
    # merge_asof exige une clé temporelle monotone globale ; on trie d'abord par date.
    left_sorted = left.sort_values([on] + by).reset_index(drop=True)
    right_sorted = right.sort_values([on] + by).reset_index(drop=True)
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        by=by,
        on=on,
        direction="backward",
        allow_exact_matches=True,
    )
    if suffix:
        rename = {
            c: f"{c}{suffix}"
            for c in merged.columns
            if c not in left.columns and c != on and c not in by
        }
        merged = merged.rename(columns=rename)
    return merged


# ---------------------------------------------------------------------------
# Index pré-calculé
# ---------------------------------------------------------------------------

class HistoryIndex:
    """
    Construit une fois, utilisé pour tous les matchs du pipeline.

    Structures internes :
      _players  : {player -> DataFrame (date, surface, opponent, won, stats service)}
      _rg       : {player -> DataFrame (date, round, won)}

    Toutes les méthodes sont O(n_joueur) ≈ O(300) au lieu de O(N_total) ≈ O(156 000).
    """

    def __init__(self, history: pd.DataFrame, rg_history: pd.DataFrame):
        self._players: dict[str, pd.DataFrame] = {}
        self._rg: dict[str, pd.DataFrame] = {}
        self.player_state: pd.DataFrame = _EMPTY
        self.rg_state: pd.DataFrame = _EMPTY
        self.h2h_state: pd.DataFrame = _EMPTY
        self._build_player_index(history)
        self._build_rg_index(rg_history)
        self._build_vector_states(history, rg_history)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_player_index(self, history: pd.DataFrame) -> None:
        """
        Convertit l'historique en format long (une ligne par joueur par match).
        Chaque match → deux lignes : une pour le gagnant, une pour le perdant.
        """
        base = ["tourney_date", "surface"]

        # Sélectionner les colonnes service disponibles
        srv_w = [c for c in _SRV_W if c in history.columns]
        srv_l = [c for c in _SRV_L if c in history.columns]
        srv_norm = [c[2:] for c in srv_w]  # retire préfixe "w_" ou "l_"

        # Vue gagnant
        w_cols = base + ["winner_name", "loser_name"] + srv_w
        w = history[[c for c in w_cols if c in history.columns]].copy()
        w = w.rename(columns={"winner_name": "player", "loser_name": "opponent",
                               **{f"w_{s}": s for s in srv_norm}})
        w["won"] = True

        # Vue perdant
        l_cols = base + ["loser_name", "winner_name"] + srv_l
        l = history[[c for c in l_cols if c in history.columns]].copy()
        l = l.rename(columns={"loser_name": "player", "winner_name": "opponent",
                               **{f"l_{s}": s for s in srv_norm}})
        l["won"] = False

        # Colonnes finales communes
        keep = ["player", "opponent", "tourney_date", "surface", "won"] + srv_norm
        keep = [c for c in keep if c in w.columns and c in l.columns]

        long = pd.concat([w[keep], l[keep]], ignore_index=True)
        long = long.sort_values("tourney_date").reset_index(drop=True)

        # Grouper par joueur — c'est le cœur de l'optimisation
        for player, group in long.groupby("player", sort=False):
            self._players[str(player)] = group.reset_index(drop=True)

    def _build_rg_index(self, rg_history: pd.DataFrame) -> None:
        """Index léger sur les matchs Roland Garros uniquement."""
        cols = ["tourney_date", "winner_name", "loser_name", "round"]
        rg = rg_history[[c for c in cols if c in rg_history.columns]].copy()

        w = rg.rename(columns={"winner_name": "player", "loser_name": "opponent"})
        w["won"] = True
        l = rg.rename(columns={"loser_name": "player", "winner_name": "opponent"})
        l["won"] = False

        long = pd.concat([w, l], ignore_index=True)
        long = long.sort_values("tourney_date").reset_index(drop=True)

        for player, group in long.groupby("player", sort=False):
            self._rg[str(player)] = group.reset_index(drop=True)

    def _build_vector_states(self, history: pd.DataFrame, rg_history: pd.DataFrame) -> None:
        """Pré-calcule des tables d'état vectorisées pour les merges asof."""
        self.player_state = self._build_player_state(history)
        self.rg_state = self._build_rg_state(rg_history)
        self.h2h_state = self._build_h2h_state(history)

    def _build_player_state(self, history: pd.DataFrame) -> pd.DataFrame:
        base = ["player", "tourney_date", "surface", "won"]
        cols = ["tourney_date", "surface", "score", "minutes"] + _SRV_NORM
        available = [c for c in cols if c in history.columns]

        if not available:
            return _EMPTY

        long = []
        # Vue gagnant
        w_cols = ["tourney_date", "surface", "score", "minutes", "winner_name", "loser_name"] + [f"w_{c}" for c in _SRV_NORM]
        w = history[[c for c in w_cols if c in history.columns]].copy()
        if not w.empty:
            w = w.rename(columns={"winner_name": "player", "loser_name": "opponent", **{f"w_{c}": c for c in _SRV_NORM}})
            w["won"] = True
            long.append(w)

        # Vue perdant
        l_cols = ["tourney_date", "surface", "score", "minutes", "loser_name", "winner_name"] + [f"l_{c}" for c in _SRV_NORM]
        l = history[[c for c in l_cols if c in history.columns]].copy()
        if not l.empty:
            l = l.rename(columns={"loser_name": "player", "winner_name": "opponent", **{f"l_{c}": c for c in _SRV_NORM}})
            l["won"] = False
            long.append(l)

        if not long:
            return _EMPTY

        df = pd.concat(long, ignore_index=True)
        df = df.sort_values(["player", "tourney_date"]).reset_index(drop=True)

        df["won_i"] = df["won"].astype(int)
        df["match_i"] = 1
        surface_series = df["surface"] if "surface" in df.columns else pd.Series("", index=df.index)
        df["is_clay"] = surface_series.eq("Clay").astype(int)
        df["clay_win_i"] = df["won_i"] * df["is_clay"]
        df["clay_match_i"] = df["is_clay"]
        df["sets_played"] = _score_to_sets_played(df.get("score", pd.Series(index=df.index, dtype="object")))
        minutes_series = df["minutes"] if "minutes" in df.columns else pd.Series(0.0, index=df.index)
        df["minutes"] = pd.to_numeric(minutes_series, errors="coerce").fillna(0.0)

        numeric_srv = [c for c in _SRV_NORM if c in df.columns]
        for c in numeric_srv:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        frames: list[pd.DataFrame] = []
        for player, g in df.groupby("player", sort=False):
            g = g.sort_values("tourney_date").copy()
            g = g.set_index("tourney_date")

            g["matches_21d"] = g["match_i"].rolling("21D", closed="left").sum().fillna(0)
            g["sets_21d"] = g["sets_played"].rolling("21D", closed="left").sum().fillna(0)
            g["minutes_21d"] = g["minutes"].rolling("21D", closed="left").sum().fillna(0)

            clay_matches_12m = g["clay_match_i"].rolling("365D", closed="left").sum()
            clay_wins_12m = g["clay_win_i"].rolling("365D", closed="left").sum()
            clay_matches_6m = g["clay_match_i"].rolling("183D", closed="left").sum()
            clay_wins_6m = g["clay_win_i"].rolling("183D", closed="left").sum()
            recent_matches_30d = g["match_i"].rolling("30D", closed="left").sum()
            recent_wins_30d = g["won_i"].rolling("30D", closed="left").sum()

            g["win_rate_clay_12m"] = (clay_wins_12m / clay_matches_12m).fillna(0.5)
            g["win_rate_clay_6m"] = (clay_wins_6m / clay_matches_6m).fillna(0.5)
            g["win_rate_30d"] = (recent_wins_30d / recent_matches_30d).fillna(0.5)

            past_wins = g["won_i"].shift(1)
            g["win_rate_last10"] = past_wins.rolling(10, min_periods=1).mean().fillna(0.5)

            if "svpt" in g.columns and "1stIn" in g.columns and "1stWon" in g.columns and "bpFaced" in g.columns and "bpSaved" in g.columns:
                svpt = pd.to_numeric(g["svpt"], errors="coerce").fillna(0)
                first_in = pd.to_numeric(g["1stIn"], errors="coerce").fillna(0)
                first_won = pd.to_numeric(g["1stWon"], errors="coerce").fillna(0)
                bp_faced = pd.to_numeric(g["bpFaced"], errors="coerce").fillna(0)
                bp_saved = pd.to_numeric(g["bpSaved"], errors="coerce").fillna(0)

                svpt_20 = svpt.shift(1).rolling(20, min_periods=1).sum()
                first_in_20 = first_in.shift(1).rolling(20, min_periods=1).sum()
                first_won_20 = first_won.shift(1).rolling(20, min_periods=1).sum()
                bp_faced_20 = bp_faced.shift(1).rolling(20, min_periods=1).sum()
                bp_saved_20 = bp_saved.shift(1).rolling(20, min_periods=1).sum()

                g["first_serve_pct"] = _rolling_ratio(first_in_20, svpt_20).fillna(0.6)
                g["first_serve_won_pct"] = _rolling_ratio(first_won_20, first_in_20).fillna(0.7)
                g["bp_saved_pct"] = _rolling_ratio(bp_saved_20, bp_faced_20).fillna(0.6)
            else:
                g["first_serve_pct"] = 0.6
                g["first_serve_won_pct"] = 0.7
                g["bp_saved_pct"] = 0.6

            g["matches_before"] = g["match_i"].cumsum().shift(1).fillna(0)
            g["wins_before"] = g["won_i"].cumsum().shift(1).fillna(0)
            g["player"] = str(player)

            frames.append(g.reset_index())

        out = pd.concat(frames, ignore_index=True) if frames else _EMPTY
        cols = [
            "player", "tourney_date",
            "matches_21d", "sets_21d", "minutes_21d",
            "win_rate_clay_12m", "win_rate_clay_6m", "win_rate_30d", "win_rate_last10",
            "first_serve_pct", "first_serve_won_pct", "bp_saved_pct",
            "matches_before", "wins_before",
        ]
        return out[cols].sort_values(["player", "tourney_date"]).reset_index(drop=True)

    def _build_rg_state(self, rg_history: pd.DataFrame) -> pd.DataFrame:
        cols = ["tourney_date", "winner_name", "loser_name", "round"]
        rg = rg_history[[c for c in cols if c in rg_history.columns]].copy()
        if rg.empty:
            return _EMPTY

        w = rg.rename(columns={"winner_name": "player", "loser_name": "opponent"})
        w["won"] = True
        l = rg.rename(columns={"loser_name": "player", "winner_name": "opponent"})
        l["won"] = False
        long = pd.concat([w, l], ignore_index=True)
        long = long.sort_values(["player", "tourney_date"]).reset_index(drop=True)
        long["won_i"] = long["won"].astype(int)
        long["rg_match_i"] = 1
        round_series = long["round"] if "round" in long.columns else pd.Series("R32", index=long.index)
        long["round_num"] = round_series.map(BEST_ROUND_ENCODING).fillna(3).astype(int)

        frames: list[pd.DataFrame] = []
        for player, g in long.groupby("player", sort=False):
            g = g.sort_values("tourney_date").copy()
            g["rg_matches"] = g["rg_match_i"].cumsum().shift(1).fillna(0)
            g["rg_wins"] = g["won_i"].cumsum().shift(1).fillna(0)
            g["best_round_rg"] = g["round_num"].cummax().shift(1).fillna(1)
            g["rg_win_rate"] = (g["rg_wins"] / g["rg_matches"]).replace([np.inf, -np.inf], np.nan).fillna(0.5)
            g["player"] = str(player)
            frames.append(g[["player", "tourney_date", "rg_win_rate", "rg_matches", "best_round_rg"]])

        return pd.concat(frames, ignore_index=True) if frames else _EMPTY

    def _build_h2h_state(self, history: pd.DataFrame) -> pd.DataFrame:
        cols = ["tourney_date", "surface", "winner_name", "loser_name"]
        h = history[[c for c in cols if c in history.columns]].copy()
        if h.empty:
            return _EMPTY

        clay = h[h["surface"] == "Clay"].copy()
        if clay.empty:
            return _EMPTY

        w = clay.rename(columns={"winner_name": "player", "loser_name": "opponent"})
        w["won"] = True
        l = clay.rename(columns={"loser_name": "player", "winner_name": "opponent"})
        l["won"] = False
        long = pd.concat([w, l], ignore_index=True)
        long = long.sort_values(["player", "opponent", "tourney_date"]).reset_index(drop=True)
        long["won_i"] = long["won"].astype(int)

        frames: list[pd.DataFrame] = []
        for (player, opponent), g in long.groupby(["player", "opponent"], sort=False):
            g = g.sort_values("tourney_date").copy()
            g["h2h_clay_wins_a"] = g["won_i"].cumsum().shift(1).fillna(0)
            g["h2h_clay_total"] = np.arange(len(g))
            g["h2h_clay_rate"] = np.where(g["h2h_clay_total"] >= 3, g["h2h_clay_wins_a"] / g["h2h_clay_total"], 0.5)
            g["player"] = str(player)
            g["opponent"] = str(opponent)
            frames.append(g[["player", "opponent", "tourney_date", "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate"]])

        return pd.concat(frames, ignore_index=True) if frames else _EMPTY

    # ------------------------------------------------------------------
    # Lookups (toutes O(n_joueur))
    # ------------------------------------------------------------------

    def _get(self, player: str) -> pd.DataFrame:
        return self._players.get(player, _EMPTY)

    def win_rate(
        self,
        player: str,
        match_date: pd.Timestamp,
        surface: Optional[str] = None,
        months: Optional[int] = None,
    ) -> float:
        sub = self._get(player)
        if sub.empty:
            return 0.5

        # Slicing binaire sur date (sub est trié) puis filtre
        past = sub[sub["tourney_date"] < match_date]
        if surface:
            past = past[past["surface"] == surface]
        if months:
            cutoff = match_date - timedelta(days=months * 30)
            past = past[past["tourney_date"] >= cutoff]

        return float(past["won"].mean()) if len(past) > 0 else 0.5

    def win_rate_last_n(self, player: str, match_date: pd.Timestamp, n: int = 10) -> float:
        sub = self._get(player)
        if sub.empty:
            return 0.5
        past = sub[sub["tourney_date"] < match_date].tail(n)
        return float(past["won"].mean()) if len(past) > 0 else 0.5

    def matches_in_window(self, player: str, match_date: pd.Timestamp, days: int) -> int:
        sub = self._get(player)
        if sub.empty:
            return 0
        cutoff = match_date - timedelta(days=days)
        mask = (sub["tourney_date"] < match_date) & (sub["tourney_date"] >= cutoff)
        return int(mask.sum())

    def h2h_clay(
        self, player_a: str, player_b: str, match_date: pd.Timestamp
    ) -> tuple[int, int]:
        """
        Travaille sur les matchs de player_a uniquement,
        filtré par opponent=player_b — O(n_a) ≈ O(300).
        """
        sub = self._get(player_a)
        if sub.empty:
            return 0, 0
        mask = (
            (sub["opponent"] == player_b)
            & (sub["surface"] == "Clay")
            & (sub["tourney_date"] < match_date)
        )
        filtered = sub[mask]
        return int(filtered["won"].sum()), len(filtered)

    def rg_stats(self, player: str, match_date: pd.Timestamp) -> dict:
        sub = self._rg.get(player, _EMPTY)
        if sub.empty:
            return {"rg_win_rate": 0.5, "rg_matches": 0, "best_round_rg": 1}

        past = sub[sub["tourney_date"] < match_date]
        if past.empty:
            return {"rg_win_rate": 0.5, "rg_matches": 0, "best_round_rg": 1}

        win_rate = float(past["won"].mean())

        # Best round : vectorisé, plus d'iterrows
        round_num = past["round"].map(BEST_ROUND_ENCODING).fillna(3)
        # Perdant au tour T → il a atteint le tour T (pas T-1, c'est son résultat)
        best_round = int(round_num.max())

        return {
            "rg_win_rate": win_rate,
            "rg_matches": len(past),
            "best_round_rg": max(best_round, 1),
        }

    def service_stats(self, player: str, match_date: pd.Timestamp, n: int = 20) -> dict:
        sub = self._get(player)
        defaults = {"first_serve_pct": 0.6, "first_serve_won_pct": 0.7, "bp_saved_pct": 0.6}
        if sub.empty:
            return defaults

        recent = sub[sub["tourney_date"] < match_date].tail(n)
        if recent.empty:
            return defaults

        def ratio(num_col: str, den_col: str, fallback: float = 0.6) -> float:
            if num_col not in recent.columns or den_col not in recent.columns:
                return fallback
            den = recent[den_col].replace(0, np.nan)
            r = recent[num_col] / den
            valid = r.dropna()
            return float(valid.mean()) if len(valid) > 0 else fallback

        return {
            "first_serve_pct": ratio("1stIn", "svpt"),
            "first_serve_won_pct": ratio("1stWon", "1stIn", 0.7),
            "bp_saved_pct": ratio("bpSaved", "bpFaced"),
        }


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    history: pd.DataFrame,
    rg_history: pd.DataFrame,
    elo_df: Optional[pd.DataFrame] = None,
    intra_rg: Optional[dict] = None,
    desc: str = "Construction features",
    index: Optional[HistoryIndex] = None,
) -> pd.DataFrame:
    """Construit les features pré-match pour un ensemble de matchs.

    La logique est vectorisée :
    - états par joueur pré-calculés une seule fois dans `HistoryIndex`
    - jointures `merge_asof` pour récupérer l'état strictement antérieur au match
    - fatigue enrichie via `matches_21d`, `sets_21d` et `minutes_21d`
    """
    if intra_rg is None:
        intra_rg = {}

    if index is None:
        index = HistoryIndex(history, rg_history)

    base = df.copy()
    base["tourney_date"] = pd.to_datetime(base["tourney_date"])
    base = base.sort_values(["tourney_date", "tourney_id", "round_number"]).reset_index(drop=True)
    base["player_a"] = base["winner_name"]
    base["player_b"] = base["loser_name"]
    base["target"] = 1

    elo_cols = [
        "match_id",
        "winner_std_elo_pre", "winner_clay_elo_pre", "winner_welo_pre", "winner_adjusted_elo_pre",
        "loser_std_elo_pre", "loser_clay_elo_pre", "loser_welo_pre", "loser_adjusted_elo_pre",
    ]
    if elo_df is not None and not set(elo_cols[1:]).issubset(base.columns):
        elo_part = elo_df[[c for c in elo_cols if c in elo_df.columns]].copy()
        base = base.merge(elo_part, on="match_id", how="left")

    for col in [
        "winner_std_elo_pre", "winner_clay_elo_pre", "winner_welo_pre", "winner_adjusted_elo_pre",
        "loser_std_elo_pre", "loser_clay_elo_pre", "loser_welo_pre", "loser_adjusted_elo_pre",
    ]:
        if col not in base.columns:
            base[col] = 1500.0

    base["diff_standard_elo"] = base["winner_std_elo_pre"] - base["loser_std_elo_pre"]
    base["diff_clay_elo"] = base["winner_clay_elo_pre"] - base["loser_clay_elo_pre"]
    base["diff_welo"] = base["winner_welo_pre"] - base["loser_welo_pre"]
    base["diff_adjusted_elo"] = base["winner_adjusted_elo_pre"] - base["loser_adjusted_elo_pre"]
    base["elo_a_std"] = base["winner_std_elo_pre"]
    base["elo_b_std"] = base["loser_std_elo_pre"]
    base["elo_a_clay"] = base["winner_clay_elo_pre"]
    base["elo_b_clay"] = base["loser_clay_elo_pre"]
    base["elo_a_welo"] = base["winner_welo_pre"]
    base["elo_b_welo"] = base["loser_welo_pre"]

    static_cols = ["match_id", "player_a", "player_b", "tourney_date"]
    if "winner_rank" in base.columns:
        winner_rank = pd.to_numeric(base["winner_rank"], errors="coerce").fillna(200.0)
        loser_rank = pd.to_numeric(base["loser_rank"], errors="coerce").fillna(200.0) if "loser_rank" in base.columns else pd.Series(200.0, index=base.index)
        base["ranking_diff"] = winner_rank - loser_rank
        base["log_ranking_diff"] = np.log(loser_rank + 1) - np.log(winner_rank + 1)
    else:
        base["ranking_diff"] = 0.0
        base["log_ranking_diff"] = 0.0

    winner_age = base["winner_age"] if "winner_age" in base.columns else pd.Series(25.0, index=base.index)
    loser_age = base["loser_age"] if "loser_age" in base.columns else pd.Series(25.0, index=base.index)
    base["age_a"] = pd.to_numeric(winner_age, errors="coerce").fillna(25.0)
    base["age_b"] = pd.to_numeric(loser_age, errors="coerce").fillna(25.0)
    base["age_optimal_a"] = ((base["age_a"] >= 28) & (base["age_a"] <= 32)).astype(int)
    base["age_optimal_b"] = ((base["age_b"] >= 28) & (base["age_b"] <= 32)).astype(int)
    base["diff_age_optimal"] = base["age_optimal_a"] - base["age_optimal_b"]

    # Intra-tournoi RG: réutilisation live, mais sans coût pour le backtest si vide.
    if intra_rg:
        sets_a = base["player_a"].map(lambda p: intra_rg.get(p, {}).get("sets_played", 0)).astype(int)
        sets_b = base["player_b"].map(lambda p: intra_rg.get(p, {}).get("sets_played", 0)).astype(int)
    else:
        sets_a = pd.Series(0, index=base.index)
        sets_b = pd.Series(0, index=base.index)
    base["diff_sets_played_rg"] = sets_a - sets_b

    # Joindre les états joueur A/B
    player_state_cols = [
        "matches_21d", "sets_21d", "minutes_21d",
        "win_rate_clay_12m", "win_rate_clay_6m", "win_rate_30d", "win_rate_last10",
        "first_serve_pct", "first_serve_won_pct", "bp_saved_pct",
        "matches_before", "wins_before",
    ]
    if not index.player_state.empty:
        state_a = index.player_state.rename(columns={"player": "player_a", **{c: f"{c}_a" for c in player_state_cols}})
        state_b = index.player_state.rename(columns={"player": "player_b", **{c: f"{c}_b" for c in player_state_cols}})
        base = _merge_asof_by_keys(base, state_a[["player_a", "tourney_date"] + [f"{c}_a" for c in player_state_cols]], by=["player_a"])
        base = _merge_asof_by_keys(base, state_b[["player_b", "tourney_date"] + [f"{c}_b" for c in player_state_cols]], by=["player_b"])
    else:
        for c in player_state_cols:
            base[f"{c}_a"] = 0.0
            base[f"{c}_b"] = 0.0

    # RG historique A/B
    if not index.rg_state.empty:
        rg_a = index.rg_state.rename(columns={"player": "player_a", "rg_win_rate": "rg_win_rate_a", "rg_matches": "rg_matches_a", "best_round_rg": "best_round_rg_a"})
        rg_b = index.rg_state.rename(columns={"player": "player_b", "rg_win_rate": "rg_win_rate_b", "rg_matches": "rg_matches_b", "best_round_rg": "best_round_rg_b"})
        base = _merge_asof_by_keys(base, rg_a[["player_a", "tourney_date", "rg_win_rate_a", "rg_matches_a", "best_round_rg_a"]], by=["player_a"])
        base = _merge_asof_by_keys(base, rg_b[["player_b", "tourney_date", "rg_win_rate_b", "rg_matches_b", "best_round_rg_b"]], by=["player_b"])
    else:
        base["rg_win_rate_a"] = 0.5
        base["rg_win_rate_b"] = 0.5
        base["rg_matches_a"] = 0
        base["rg_matches_b"] = 0
        base["best_round_rg_a"] = 1
        base["best_round_rg_b"] = 1

    # H2H clay A/B
    if not index.h2h_state.empty:
        h2h = index.h2h_state.rename(columns={"player": "player_a", "opponent": "player_b"})
        base = _merge_asof_by_keys(base, h2h[["player_a", "player_b", "tourney_date", "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate"]], by=["player_a", "player_b"])
    else:
        base["h2h_clay_wins_a"] = 0
        base["h2h_clay_total"] = 0
        base["h2h_clay_rate"] = 0.5

    # Remplissage des NaN pour les premiers matchs de chaque joueur
    fill_defaults = {
        "matches_21d_a": 0, "sets_21d_a": 0, "minutes_21d_a": 0,
        "win_rate_clay_12m_a": 0.5, "win_rate_clay_6m_a": 0.5, "win_rate_30d_a": 0.5, "win_rate_last10_a": 0.5,
        "first_serve_pct_a": 0.6, "first_serve_won_pct_a": 0.7, "bp_saved_pct_a": 0.6, "matches_before_a": 0, "wins_before_a": 0,
        "matches_21d_b": 0, "sets_21d_b": 0, "minutes_21d_b": 0,
        "win_rate_clay_12m_b": 0.5, "win_rate_clay_6m_b": 0.5, "win_rate_30d_b": 0.5, "win_rate_last10_b": 0.5,
        "first_serve_pct_b": 0.6, "first_serve_won_pct_b": 0.7, "bp_saved_pct_b": 0.6, "matches_before_b": 0, "wins_before_b": 0,
        "rg_win_rate_a": 0.5, "rg_matches_a": 0, "best_round_rg_a": 1,
        "rg_win_rate_b": 0.5, "rg_matches_b": 0, "best_round_rg_b": 1,
        "h2h_clay_wins_a": 0, "h2h_clay_total": 0, "h2h_clay_rate": 0.5,
    }
    for c, default in fill_defaults.items():
        if c in base.columns:
            base[c] = base[c].fillna(default)

    # Diff features
    base["diff_win_rate_clay_12m"] = base["win_rate_clay_12m_a"] - base["win_rate_clay_12m_b"]
    base["diff_win_rate_clay_6m"] = base["win_rate_clay_6m_a"] - base["win_rate_clay_6m_b"]
    base["diff_win_rate_30d"] = base["win_rate_30d_a"] - base["win_rate_30d_b"]
    base["diff_matches_21d"] = base["matches_21d_a"] - base["matches_21d_b"]
    base["diff_sets_21d"] = base["sets_21d_a"] - base["sets_21d_b"]
    base["diff_minutes_21d"] = base["minutes_21d_a"] - base["minutes_21d_b"]
    base["diff_win_rate_last10"] = base["win_rate_last10_a"] - base["win_rate_last10_b"]
    base["diff_rg_win_rate"] = base["rg_win_rate_a"] - base["rg_win_rate_b"]
    base["diff_best_round_rg"] = base["best_round_rg_a"] - base["best_round_rg_b"]
    base["diff_first_serve_pct"] = base["first_serve_pct_a"] - base["first_serve_pct_b"]
    base["diff_first_serve_won_pct"] = base["first_serve_won_pct_a"] - base["first_serve_won_pct_b"]
    base["diff_bp_saved_pct"] = base["bp_saved_pct_a"] - base["bp_saved_pct_b"]

    output_cols = [
        "match_id", "tourney_date", "player_a", "player_b", "target",
        "diff_standard_elo", "diff_clay_elo", "diff_welo", "diff_adjusted_elo",
        "elo_a_std", "elo_b_std", "elo_a_clay", "elo_b_clay", "elo_a_welo", "elo_b_welo",
        "diff_win_rate_clay_12m", "diff_win_rate_clay_6m", "diff_win_rate_30d",
        "diff_matches_21d", "diff_sets_21d", "diff_minutes_21d", "diff_win_rate_last10",
        "matches_21d_a", "matches_21d_b", "sets_21d_a", "sets_21d_b", "minutes_21d_a", "minutes_21d_b",
        "win_rate_clay_12m_a", "win_rate_clay_12m_b",
        "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate",
        "rg_win_rate_a", "rg_win_rate_b", "diff_rg_win_rate", "rg_matches_a", "rg_matches_b",
        "best_round_rg_a", "best_round_rg_b", "diff_best_round_rg",
        "ranking_diff", "log_ranking_diff",
        "age_a", "age_b", "age_optimal_a", "age_optimal_b", "diff_age_optimal",
        "diff_first_serve_pct", "diff_first_serve_won_pct", "diff_bp_saved_pct",
        "first_serve_pct_a", "first_serve_won_pct_a", "bp_saved_pct_a",
        "round_number", "diff_sets_played_rg",
    ]

    for col in output_cols:
        if col not in base.columns:
            if col in {"target", "round_number"}:
                base[col] = 1 if col == "target" else base.get("round_number", 3)
            elif col.startswith(("age_", "elo_")):
                base[col] = 1500.0 if col.startswith("elo_") else 25.0
            elif col.startswith(("rg_matches", "matches_21d", "sets_21d", "minutes_21d", "h2h_clay_total")):
                base[col] = 0
            else:
                base[col] = np.nan

    return base[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset symétrique
# ---------------------------------------------------------------------------

def build_symmetric_dataset(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Double le dataset en inversant les rôles A/B.
    Élimine le biais de position : le modèle ne peut pas apprendre
    que "le joueur en colonne A gagne plus souvent".
    """
    df_pos = feature_df.copy()
    df_pos["target"] = 1

    df_neg = feature_df.copy()
    df_neg["target"] = 0

    # Inverser les différentiels
    diff_cols = [c for c in df_neg.columns if c.startswith("diff_")]
    df_neg[diff_cols] = -df_neg[diff_cols]

    # Inverser les paires _a / _b
    a_cols = [c for c in df_neg.columns if c.endswith("_a") and c.replace("_a", "_b") in df_neg.columns]
    for col_a in a_cols:
        col_b = col_a.replace("_a", "_b")
        df_neg[col_a], df_neg[col_b] = df_neg[col_b].copy(), df_neg[col_a].copy()

    # H2H : inverser manuellement
    if "h2h_clay_wins_a" in feature_df.columns and "h2h_clay_total" in feature_df.columns:
        h2h_wins_orig = feature_df["h2h_clay_wins_a"].copy()
        h2h_total_orig = feature_df["h2h_clay_total"].copy()
        df_neg["h2h_clay_wins_a"] = h2h_total_orig - h2h_wins_orig
    if "h2h_clay_rate" in feature_df.columns:
        df_neg["h2h_clay_rate"] = 1.0 - feature_df["h2h_clay_rate"]
    df_neg["ranking_diff"]    = -feature_df["ranking_diff"]
    df_neg["log_ranking_diff"] = -feature_df["log_ranking_diff"]

    return pd.concat([df_pos, df_neg], ignore_index=True)


# ---------------------------------------------------------------------------
# Colonnes de features pour le modèle
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "diff_standard_elo", "diff_clay_elo", "diff_welo", "diff_adjusted_elo",
    "diff_win_rate_clay_12m", "diff_win_rate_clay_6m", "diff_win_rate_30d",
    "diff_matches_21d", "diff_sets_21d", "diff_minutes_21d", "diff_win_rate_last10",
    "matches_21d_a", "matches_21d_b", "sets_21d_a", "sets_21d_b", "minutes_21d_a", "minutes_21d_b",
    "h2h_clay_rate", "h2h_clay_total",
    "diff_rg_win_rate", "rg_matches_a", "rg_matches_b",
    "diff_best_round_rg",
    "ranking_diff", "log_ranking_diff",
    "age_a", "age_b", "diff_age_optimal",
    "diff_first_serve_pct", "diff_first_serve_won_pct", "diff_bp_saved_pct",
    "round_number", "diff_sets_played_rg",
]
