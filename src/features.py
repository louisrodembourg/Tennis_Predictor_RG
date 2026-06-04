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

FEATURE_VERSION = "v3"  # increment to invalidate stale feature cache
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
        if rg.empty:
            return

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
        # --- Format long (gagnant + perdant) ---
        _RET_OPP_W = ["l_svpt", "l_1stWon", "l_2ndWon"]  # opponent (loser) serve stats → winner's return
        _RET_OPP_L = ["w_svpt", "w_1stWon", "w_2ndWon"]  # opponent (winner) serve stats → loser's return
        w_cols = ["tourney_date", "surface", "score", "minutes", "winner_name", "loser_name", "tourney_id", "best_of", "loser_rank"] + [f"w_{c}" for c in _SRV_NORM] + _RET_OPP_W
        l_cols = ["tourney_date", "surface", "score", "minutes", "loser_name", "winner_name", "tourney_id", "best_of", "winner_rank"] + [f"l_{c}" for c in _SRV_NORM] + _RET_OPP_L

        w = history[[c for c in w_cols if c in history.columns]].copy()
        l = history[[c for c in l_cols if c in history.columns]].copy()
        if w.empty and l.empty:
            return _EMPTY

        w = w.rename(columns={"winner_name": "player", "loser_name": "opponent",
                               **{f"w_{c}": c for c in _SRV_NORM},
                               "l_svpt": "opp_svpt", "l_1stWon": "opp_1stWon", "l_2ndWon": "opp_2ndWon",
                               "loser_rank": "opp_rank"})
        w["won"] = True
        l = l.rename(columns={"loser_name": "player", "winner_name": "opponent",
                               **{f"l_{c}": c for c in _SRV_NORM},
                               "w_svpt": "opp_svpt", "w_1stWon": "opp_1stWon", "w_2ndWon": "opp_2ndWon",
                               "winner_rank": "opp_rank"})
        l["won"] = False

        df = pd.concat([w, l], ignore_index=True)
        sort_cols = ["player", "tourney_date"]
        for extra in ["round_number", "match_num"]:
            if extra in df.columns:
                sort_cols.append(extra)
        df = df.sort_values(sort_cols).reset_index(drop=True)

        df["won_i"]      = df["won"].astype(int)
        df["is_clay"]    = df["surface"].eq("Clay").astype(int) if "surface" in df.columns else 0
        df["clay_win_i"] = df["won_i"] * df["is_clay"]
        df["sets_played"] = _score_to_sets_played(df.get("score", pd.Series(dtype="object")))
        _min_raw = df["minutes"] if "minutes" in df.columns else pd.Series(0.0, index=df.index)
        df["minutes"] = pd.to_numeric(_min_raw, errors="coerce").fillna(0.0)

        srv_cols = ["svpt", "1stIn", "1stWon", "bpFaced", "bpSaved"]
        for c in srv_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        srv_ok = all(c in df.columns for c in srv_cols)

        # --- Pré-allocation numpy (évite 20K DataFrames dans une boucle) ---
        DAY = np.timedelta64(1, "D")
        N = len(df)

        best_of_col = (
            pd.to_numeric(df["best_of"], errors="coerce").fillna(3.0).values
            if "best_of" in df.columns else np.full(N, 3.0)
        )
        has_tourney_id = "tourney_id" in df.columns
        tourney_id_col = df["tourney_id"].values if has_tourney_id else None

        out_matches_21d    = np.zeros(N)
        out_sets_21d       = np.zeros(N)
        out_minutes_21d    = np.zeros(N)
        out_wr_clay_12m    = np.full(N, 0.5)
        out_wr_clay_6m     = np.full(N, 0.5)
        out_wr_30d         = np.full(N, 0.5)
        out_wr_last10      = np.full(N, 0.5)
        out_fsp            = np.full(N, 0.6)
        out_fswp           = np.full(N, 0.7)
        out_bpsp           = np.full(N, 0.6)
        out_matches_before       = np.zeros(N)
        out_wins_before          = np.zeros(N)
        out_wr_clay_90d          = np.full(N, 0.5)
        out_win_streak_clay      = np.zeros(N)
        out_pct_3plus_sets_12m   = np.full(N, 0.5)
        out_pct_5sets_12m        = np.full(N, 0.4)
        out_avg_duration_12m     = np.full(N, 90.0)
        out_days_since_non_clay  = np.zeros(N)
        out_clay_streak_tourns   = np.zeros(N)
        out_return_pts_pct          = np.full(N, 0.35)  # default ~35% return pts won
        out_clay_win_vs_top50_12m   = np.full(N, 0.5)

        # df est déjà trié par (player, tourney_date)
        player_col = df["player"].values
        date_col   = df["tourney_date"].values.astype("datetime64[D]")
        won_col    = df["won_i"].values.astype(float)
        clay_w_col = df["clay_win_i"].values.astype(float)
        clay_m_col = df["is_clay"].values.astype(float)
        sets_col   = df["sets_played"].values.astype(float)
        mins_col   = df["minutes"].values.astype(float)

        # Opponent rank (for quality-adjusted win rate)
        if "opp_rank" in df.columns:
            df["opp_rank"] = pd.to_numeric(df["opp_rank"], errors="coerce").fillna(999.0)
            opp_rank_col = df["opp_rank"].values.astype(float)
        else:
            opp_rank_col = np.full(N, 999.0)

        # Return stats: opponent serve points available?
        _ret_cols = ["opp_svpt", "opp_1stWon", "opp_2ndWon"]
        for c in _ret_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        ret_ok = all(c in df.columns for c in _ret_cols)
        if ret_ok:
            opp_svpt_col  = df["opp_svpt"].values.astype(float)
            opp_1won_col  = df["opp_1stWon"].values.astype(float)
            opp_2won_col  = df["opp_2ndWon"].values.astype(float)

        # Colonnes service (optionnelles)
        if srv_ok:
            svpt_col  = df["svpt"].values.astype(float)
            fin_col   = df["1stIn"].values.astype(float)
            fwon_col  = df["1stWon"].values.astype(float)
            bpf_col   = df["bpFaced"].values.astype(float)
            bps_col   = df["bpSaved"].values.astype(float)

        # Bornes de groupe : une seule itération, O(n) total
        boundaries = np.where(np.concatenate([[True], player_col[1:] != player_col[:-1], [True]]))[0]
        for k in range(len(boundaries) - 1):
            s, e = int(boundaries[k]), int(boundaries[k + 1])
            n = e - s
            dates    = date_col[s:e]
            won_arr  = won_col[s:e]
            clay_w   = clay_w_col[s:e]
            clay_m   = clay_m_col[s:e]
            sets_arr = sets_col[s:e]
            mins_arr = mins_col[s:e]

            cs_w  = np.empty(n + 1); cs_w[0]  = 0.0; np.cumsum(won_arr,  out=cs_w[1:])
            cs_cw = np.empty(n + 1); cs_cw[0] = 0.0; np.cumsum(clay_w,   out=cs_cw[1:])
            cs_cm = np.empty(n + 1); cs_cm[0] = 0.0; np.cumsum(clay_m,   out=cs_cm[1:])
            cs_s  = np.empty(n + 1); cs_s[0]  = 0.0; np.cumsum(sets_arr, out=cs_s[1:])
            cs_mn = np.empty(n + 1); cs_mn[0] = 0.0; np.cumsum(mins_arr, out=cs_mn[1:])
            idx   = np.arange(n)  # end_idx in cs arrays

            def _ws(cs: np.ndarray, days: int) -> np.ndarray:
                st = np.searchsorted(dates, dates - days * DAY, side="left")
                return cs[idx] - cs[st]

            out_matches_21d[s:e]    = _ws(np.arange(n + 1, dtype=float), 21)
            out_sets_21d[s:e]       = _ws(cs_s, 21)
            out_minutes_21d[s:e]    = _ws(cs_mn, 21)

            cw12 = _ws(cs_cw, 365); cm12 = _ws(cs_cm, 365)
            cw6  = _ws(cs_cw, 183); cm6  = _ws(cs_cm, 183)
            w30  = _ws(cs_w, 30);   m30  = np.arange(n, dtype=float) - np.searchsorted(dates, dates - 30 * DAY, side="left")

            np.divide(cw12, cm12, out=out_wr_clay_12m[s:e], where=(cm12 > 0))
            np.divide(cw6,  cm6,  out=out_wr_clay_6m[s:e],  where=(cm6  > 0))
            np.divide(w30,  m30,  out=out_wr_30d[s:e],      where=(m30  > 0))
            out_matches_before[s:e] = idx.astype(float)
            out_wins_before[s:e]    = cs_w[idx]

            # win_rate_clay_90d
            cw90 = _ws(cs_cw, 90); cm90 = _ws(cs_cm, 90)
            np.divide(cw90, cm90, out=out_wr_clay_90d[s:e], where=(cm90 > 0))

            # win_streak_clay : consecutive clay wins before current match
            not_cw = (clay_w == 0)
            reset_cum = np.where(not_cw, np.arange(n), -1)
            last_reset_incl = np.maximum.accumulate(reset_cum)
            last_reset_excl = np.concatenate([[-1], last_reset_incl[:-1]])
            out_win_streak_clay[s:e] = np.maximum(0, np.arange(n) - last_reset_excl - 1)

            # pct_3plus_sets_12m
            cs_3p = np.empty(n + 1); cs_3p[0] = 0.0
            np.cumsum((sets_arr >= 3).astype(float), out=cs_3p[1:])
            s3p12 = _ws(cs_3p, 365)
            m12   = _ws(np.arange(n + 1, dtype=float), 365)
            np.divide(s3p12, m12, out=out_pct_3plus_sets_12m[s:e], where=(m12 > 0))

            # pct_5sets_12m (best_of==5 matches only)
            bo_arr = best_of_col[s:e]
            is_bo5 = (bo_arr == 5).astype(float)
            is_5s  = ((sets_arr >= 5) & (bo_arr == 5)).astype(float)
            cs_bo5 = np.empty(n + 1); cs_bo5[0] = 0.0; np.cumsum(is_bo5, out=cs_bo5[1:])
            cs_5s  = np.empty(n + 1); cs_5s[0]  = 0.0; np.cumsum(is_5s,  out=cs_5s[1:])
            bo5_12m = _ws(cs_bo5, 365); s5_12m = _ws(cs_5s, 365)
            np.divide(s5_12m, bo5_12m, out=out_pct_5sets_12m[s:e], where=(bo5_12m > 0))

            # avg_match_duration_12m (exclude matches where minutes==0)
            valid_min = (mins_arr > 0).astype(float)
            cs_vmin = np.empty(n + 1); cs_vmin[0] = 0.0; np.cumsum(valid_min, out=cs_vmin[1:])
            mn12    = _ws(cs_mn, 365); cnt_mn12 = _ws(cs_vmin, 365)
            np.divide(mn12, cnt_mn12, out=out_avg_duration_12m[s:e], where=(cnt_mn12 > 0))

            # days_since_non_clay
            not_clay_mask = (clay_m == 0)
            nc_reset = np.where(not_clay_mask, np.arange(n), -1)
            last_nc_incl = np.maximum.accumulate(nc_reset)
            last_nc_excl = np.concatenate([[-1], last_nc_incl[:-1]])
            has_nc = last_nc_excl >= 0
            safe_nc_idx = np.maximum(0, last_nc_excl).astype(int)
            out_days_since_non_clay[s:e] = np.where(
                has_nc, (dates - dates[safe_nc_idx]) / DAY, 0.0
            )

            # clay_streak_tournaments : consecutive clay tournaments before current
            if has_tourney_id:
                tid_arr = tourney_id_col[s:e]
                tid_change = np.concatenate([[True], tid_arr[1:] != tid_arr[:-1]])
                tourney_starts = np.where(tid_change)[0]
                n_t = len(tourney_starts)
                tourney_is_clay = clay_m[tourney_starts].astype(bool)
                clay_streak_t = np.zeros(n_t, dtype=np.int64)
                for t in range(1, n_t):
                    clay_streak_t[t] = clay_streak_t[t - 1] + 1 if tourney_is_clay[t - 1] else 0
                match_t_idx = np.searchsorted(tourney_starts, np.arange(n), side="right") - 1
                out_clay_streak_tourns[s:e] = clay_streak_t[match_t_idx]

            # win_rate_last10 via rolling numpy cumsum
            cs_w_full = np.concatenate([[0.0], np.cumsum(won_arr)])
            end10 = idx
            st10  = np.maximum(0, idx - 10)
            cnt10 = idx - st10
            np.divide(
                cs_w_full[end10] - cs_w_full[st10], cnt10,
                out=out_wr_last10[s:e], where=(cnt10 > 0),
            )

            # Service stats via rolling numpy cumsum (shifted: exclude current)
            if srv_ok:
                sv = svpt_col[s:e]; fi = fin_col[s:e]
                fw = fwon_col[s:e]; bf = bpf_col[s:e]; bs = bps_col[s:e]
                def _srv(arr: np.ndarray, w: int = 20) -> np.ndarray:
                    cs = np.empty(n + 1); cs[0] = 0.0; np.cumsum(arr, out=cs[1:])
                    st = np.maximum(0, idx - w)
                    return cs[idx] - cs[st]
                sv20 = _srv(sv); fi20 = _srv(fi); fw20 = _srv(fw); bf20 = _srv(bf); bs20 = _srv(bs)
                np.divide(fi20, sv20, out=out_fsp[s:e],  where=(sv20 > 0))
                np.divide(fw20, fi20, out=out_fswp[s:e], where=(fi20 > 0))
                np.divide(bs20, bf20, out=out_bpsp[s:e], where=(bf20 > 0))

            if ret_ok:
                osp = opp_svpt_col[s:e]
                o1w = opp_1won_col[s:e]
                o2w = opp_2won_col[s:e]
                ret_won = np.maximum(osp - o1w - o2w, 0.0)
                def _ret(arr: np.ndarray, w: int = 20) -> np.ndarray:
                    cs = np.empty(n + 1); cs[0] = 0.0; np.cumsum(arr, out=cs[1:])
                    st = np.maximum(0, idx - w)
                    return cs[idx] - cs[st]
                ret_won20  = _ret(ret_won)
                osp20      = _ret(osp)
                np.divide(ret_won20, osp20, out=out_return_pts_pct[s:e], where=(osp20 > 0))

            # clay_win_rate_vs_top50_12m: clay wins vs rank ≤ 50 in past 12 months
            opp_r = opp_rank_col[s:e]
            is_vs_top50_clay = ((opp_r <= 50) & (clay_m == 1)).astype(float)
            win_vs_top50_clay = (won_arr * is_vs_top50_clay).astype(float)
            cs_vt   = np.empty(n + 1); cs_vt[0]  = 0.0; np.cumsum(is_vs_top50_clay, out=cs_vt[1:])
            cs_wvt  = np.empty(n + 1); cs_wvt[0] = 0.0; np.cumsum(win_vs_top50_clay, out=cs_wvt[1:])
            st_12m  = np.searchsorted(dates, dates - 365 * DAY, side="left")
            cnt_vt  = cs_vt[idx] - cs_vt[st_12m]
            wins_vt = cs_wvt[idx] - cs_wvt[st_12m]
            np.divide(wins_vt, cnt_vt, out=out_clay_win_vs_top50_12m[s:e], where=(cnt_vt > 0))

        result = pd.DataFrame({
            "player":                  player_col,
            "tourney_date":            df["tourney_date"].values,
            "matches_21d":             out_matches_21d,
            "sets_21d":                out_sets_21d,
            "minutes_21d":             out_minutes_21d,
            "win_rate_clay_12m":       out_wr_clay_12m,
            "win_rate_clay_6m":        out_wr_clay_6m,
            "win_rate_30d":            out_wr_30d,
            "win_rate_last10":         out_wr_last10,
            "first_serve_pct":         out_fsp,
            "first_serve_won_pct":     out_fswp,
            "bp_saved_pct":            out_bpsp,
            "matches_before":          out_matches_before,
            "wins_before":             out_wins_before,
            "win_rate_clay_90d":       out_wr_clay_90d,
            "win_streak_clay":         out_win_streak_clay,
            "pct_3plus_sets_12m":      out_pct_3plus_sets_12m,
            "pct_5sets_12m":           out_pct_5sets_12m,
            "avg_match_duration_12m":  out_avg_duration_12m,
            "days_since_non_clay":          out_days_since_non_clay,
            "clay_streak_tournaments":      out_clay_streak_tourns,
            "return_pts_won_pct":           out_return_pts_pct,
            "clay_win_rate_vs_top50_12m":   out_clay_win_vs_top50_12m,
        })
        return result.sort_values(["player", "tourney_date"]).reset_index(drop=True)

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
        long["round_num"] = long.get("round", pd.Series(dtype=str)).map(BEST_ROUND_ENCODING).fillna(3).astype(int)
        long = long.sort_values(["player", "tourney_date", "round_num"]).reset_index(drop=True)
        long["won_i"] = long["won"].astype(int)

        grp = long.groupby("player", sort=False)
        long["rg_matches"]  = grp.cumcount()
        cum_wins_rg          = grp["won_i"].transform("cumsum")
        long["rg_wins"]     = (cum_wins_rg - long["won_i"]).clip(lower=0)
        # best_round_rg = max tour atteint avant ce match (12K lignes → lambda OK)
        long["best_round_rg"] = (
            grp["round_num"]
            .transform(lambda x: x.shift(1).expanding().max())
            .fillna(1)
        )
        long["rg_win_rate"] = (long["rg_wins"] / long["rg_matches"]).replace([np.inf, -np.inf], np.nan).fillna(0.5)

        # 3-year windowed RG stats
        N_rg = len(long)
        out_wr_rg_3yr   = np.full(N_rg, 0.5)
        out_br_rg_3yr   = np.ones(N_rg)
        _DAY = np.timedelta64(1, "D")
        _p_col   = long["player"].values
        _d_col   = long["tourney_date"].values.astype("datetime64[D]")
        _won_col = long["won_i"].values.astype(float)
        _rnd_col = long["round_num"].values.astype(float)
        _bounds  = np.where(np.concatenate([[True], _p_col[1:] != _p_col[:-1], [True]]))[0]
        for _k in range(len(_bounds) - 1):
            _s, _e = int(_bounds[_k]), int(_bounds[_k + 1])
            _n = _e - _s
            _dates = _d_col[_s:_e]
            _won   = _won_col[_s:_e]
            _rnds  = _rnd_col[_s:_e]
            _cs_w  = np.zeros(_n + 1); np.cumsum(_won, out=_cs_w[1:])
            _st3   = np.searchsorted(_dates, _dates - np.timedelta64(1095, "D"), side="left")
            for _i in range(_n):
                _st = int(_st3[_i])
                if _st < _i:
                    _m = float(_i - _st)
                    out_wr_rg_3yr[_s + _i] = (_cs_w[_i] - _cs_w[_st]) / _m
                    out_br_rg_3yr[_s + _i] = float(np.max(_rnds[_st:_i]))
        long["win_rate_rg_3yr"]  = out_wr_rg_3yr
        long["best_round_rg_3yr"] = out_br_rg_3yr

        return long[["player", "tourney_date", "rg_win_rate", "rg_matches", "best_round_rg",
                     "win_rate_rg_3yr", "best_round_rg_3yr"]].copy()

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

        # Vectorisé : Cython natif, pas de lambda Python
        long["h2h_clay_total"]    = long.groupby(["player", "opponent"]).cumcount()
        cum_wins                   = long.groupby(["player", "opponent"])["won_i"].transform("cumsum")
        long["h2h_clay_wins_a"]   = (cum_wins - long["won_i"]).clip(lower=0)
        total_safe                 = long["h2h_clay_total"].replace(0, np.nan)
        long["h2h_clay_rate"] = np.where(
            long["h2h_clay_total"] >= 3,
            long["h2h_clay_wins_a"] / total_safe,
            0.5,
        )

        # Exponentially-weighted H2H rate: recent matches count more (λ=0.3/year)
        _H2H_DECAY = 0.3
        _DAY_NS = np.timedelta64(1, "D")
        N_h = len(long)
        out_h2h_recent = np.full(N_h, 0.5)
        _p_h = long["player"].values
        _o_h = long["opponent"].values
        _d_h = long["tourney_date"].values.astype("datetime64[D]")
        _w_h = long["won_i"].values.astype(float)
        # Build per (player, opponent) groups
        pair_key = np.array([f"{p}|||{o}" for p, o in zip(_p_h, _o_h)])
        pair_bounds = np.where(np.concatenate([[True], pair_key[1:] != pair_key[:-1], [True]]))[0]
        for _k in range(len(pair_bounds) - 1):
            _s, _e = int(pair_bounds[_k]), int(pair_bounds[_k + 1])
            _dates = _d_h[_s:_e]
            _wins  = _w_h[_s:_e]
            for _i in range(_e - _s):
                if _i == 0:
                    continue  # no prior H2H
                ref = _dates[_i]
                years_ago = (_dates[:_i] - ref) / _DAY_NS / -365.25  # positive = past
                weights = np.exp(-_H2H_DECAY * np.maximum(years_ago, 0.0))
                w_sum = weights.sum()
                if w_sum > 0:
                    out_h2h_recent[_s + _i] = float(np.dot(weights, _wins[:_i]) / w_sum)
        long["h2h_clay_rate_recent"] = out_h2h_recent

        return long[["player", "opponent", "tourney_date",
                      "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate",
                      "h2h_clay_rate_recent"]].copy()

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
    sort_by = ["tourney_date"] + [c for c in ["tourney_id", "round_number"] if c in base.columns]
    base = base.sort_values(sort_by).reset_index(drop=True)
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
        "win_rate_clay_90d", "win_streak_clay",
        "pct_3plus_sets_12m", "pct_5sets_12m", "avg_match_duration_12m",
        "days_since_non_clay", "clay_streak_tournaments",
        "return_pts_won_pct", "clay_win_rate_vs_top50_12m",
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
        rg_a = index.rg_state.rename(columns={
            "player": "player_a", "rg_win_rate": "rg_win_rate_a", "rg_matches": "rg_matches_a",
            "best_round_rg": "best_round_rg_a", "win_rate_rg_3yr": "win_rate_rg_3yr_a",
            "best_round_rg_3yr": "best_round_rg_3yr_a",
        })
        rg_b = index.rg_state.rename(columns={
            "player": "player_b", "rg_win_rate": "rg_win_rate_b", "rg_matches": "rg_matches_b",
            "best_round_rg": "best_round_rg_b", "win_rate_rg_3yr": "win_rate_rg_3yr_b",
            "best_round_rg_3yr": "best_round_rg_3yr_b",
        })
        rg_a_cols = ["player_a", "tourney_date", "rg_win_rate_a", "rg_matches_a", "best_round_rg_a",
                     "win_rate_rg_3yr_a", "best_round_rg_3yr_a"]
        rg_b_cols = ["player_b", "tourney_date", "rg_win_rate_b", "rg_matches_b", "best_round_rg_b",
                     "win_rate_rg_3yr_b", "best_round_rg_3yr_b"]
        base = _merge_asof_by_keys(base, rg_a[[c for c in rg_a_cols if c in rg_a.columns]], by=["player_a"])
        base = _merge_asof_by_keys(base, rg_b[[c for c in rg_b_cols if c in rg_b.columns]], by=["player_b"])
    else:
        base["rg_win_rate_a"] = 0.5
        base["rg_win_rate_b"] = 0.5
        base["rg_matches_a"] = 0
        base["rg_matches_b"] = 0
        base["best_round_rg_a"] = 1
        base["best_round_rg_b"] = 1
        base["win_rate_rg_3yr_a"] = 0.5
        base["win_rate_rg_3yr_b"] = 0.5
        base["best_round_rg_3yr_a"] = 1
        base["best_round_rg_3yr_b"] = 1

    # H2H clay A/B
    if not index.h2h_state.empty:
        h2h = index.h2h_state.rename(columns={"player": "player_a", "opponent": "player_b"})
        h2h_cols = ["player_a", "player_b", "tourney_date", "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate"]
        if "h2h_clay_rate_recent" in h2h.columns:
            h2h_cols.append("h2h_clay_rate_recent")
        base = _merge_asof_by_keys(base, h2h[h2h_cols], by=["player_a", "player_b"])
    else:
        base["h2h_clay_wins_a"] = 0
        base["h2h_clay_total"] = 0
        base["h2h_clay_rate"] = 0.5
        base["h2h_clay_rate_recent"] = 0.5

    # Remplissage des NaN pour les premiers matchs de chaque joueur
    fill_defaults = {
        "matches_21d_a": 0, "sets_21d_a": 0, "minutes_21d_a": 0,
        "win_rate_clay_12m_a": 0.5, "win_rate_clay_6m_a": 0.5, "win_rate_30d_a": 0.5, "win_rate_last10_a": 0.5,
        "first_serve_pct_a": 0.6, "first_serve_won_pct_a": 0.7, "bp_saved_pct_a": 0.6, "matches_before_a": 0, "wins_before_a": 0,
        "win_rate_clay_90d_a": 0.5, "win_streak_clay_a": 0,
        "pct_3plus_sets_12m_a": 0.5, "pct_5sets_12m_a": 0.4, "avg_match_duration_12m_a": 90.0,
        "days_since_non_clay_a": 0, "clay_streak_tournaments_a": 0,
        "return_pts_won_pct_a": 0.35, "clay_win_rate_vs_top50_12m_a": 0.5,
        "matches_21d_b": 0, "sets_21d_b": 0, "minutes_21d_b": 0,
        "win_rate_clay_12m_b": 0.5, "win_rate_clay_6m_b": 0.5, "win_rate_30d_b": 0.5, "win_rate_last10_b": 0.5,
        "first_serve_pct_b": 0.6, "first_serve_won_pct_b": 0.7, "bp_saved_pct_b": 0.6, "matches_before_b": 0, "wins_before_b": 0,
        "win_rate_clay_90d_b": 0.5, "win_streak_clay_b": 0,
        "pct_3plus_sets_12m_b": 0.5, "pct_5sets_12m_b": 0.4, "avg_match_duration_12m_b": 90.0,
        "days_since_non_clay_b": 0, "clay_streak_tournaments_b": 0,
        "return_pts_won_pct_b": 0.35, "clay_win_rate_vs_top50_12m_b": 0.5,
        "rg_win_rate_a": 0.5, "rg_matches_a": 0, "best_round_rg_a": 1,
        "rg_win_rate_b": 0.5, "rg_matches_b": 0, "best_round_rg_b": 1,
        "win_rate_rg_3yr_a": 0.5, "best_round_rg_3yr_a": 1,
        "win_rate_rg_3yr_b": 0.5, "best_round_rg_3yr_b": 1,
        "h2h_clay_wins_a": 0, "h2h_clay_total": 0, "h2h_clay_rate": 0.5,
        "h2h_clay_rate_recent": 0.5,
    }
    for c, default in fill_defaults.items():
        if c in base.columns:
            base[c] = base[c].fillna(default)

    # Diff features
    age_clay_a = base["age_a"] * base["elo_a_clay"]
    age_clay_b = base["age_b"] * base["elo_b_clay"]

    extra = {
        "diff_win_rate_clay_12m":       base["win_rate_clay_12m_a"]                    - base["win_rate_clay_12m_b"],
        "diff_win_rate_clay_6m":        base["win_rate_clay_6m_a"]                     - base["win_rate_clay_6m_b"],
        "diff_win_rate_30d":            base["win_rate_30d_a"]                         - base["win_rate_30d_b"],
        "diff_matches_21d":             base["matches_21d_a"]                          - base["matches_21d_b"],
        "diff_sets_21d":                base["sets_21d_a"]                             - base["sets_21d_b"],
        "diff_minutes_21d":             base["minutes_21d_a"]                          - base["minutes_21d_b"],
        "diff_win_rate_last10":         base["win_rate_last10_a"]                      - base["win_rate_last10_b"],
        "diff_rg_win_rate":             base["rg_win_rate_a"]                          - base["rg_win_rate_b"],
        "diff_best_round_rg":           base["best_round_rg_a"]                        - base["best_round_rg_b"],
        "diff_first_serve_pct":         base["first_serve_pct_a"]                      - base["first_serve_pct_b"],
        "diff_first_serve_won_pct":     base["first_serve_won_pct_a"]                  - base["first_serve_won_pct_b"],
        "diff_bp_saved_pct":            base["bp_saved_pct_a"]                         - base["bp_saved_pct_b"],
        # Phase-2
        "diff_win_rate_clay_90d":       base.get("win_rate_clay_90d_a", 0.5)           - base.get("win_rate_clay_90d_b", 0.5),
        "diff_win_streak_clay":         base.get("win_streak_clay_a", 0.0)             - base.get("win_streak_clay_b", 0.0),
        "diff_pct_3plus_sets_12m":      base.get("pct_3plus_sets_12m_a", 0.5)         - base.get("pct_3plus_sets_12m_b", 0.5),
        "diff_pct_5sets_12m":           base.get("pct_5sets_12m_a", 0.4)              - base.get("pct_5sets_12m_b", 0.4),
        "diff_avg_match_duration_12m":  base.get("avg_match_duration_12m_a", 90.0)    - base.get("avg_match_duration_12m_b", 90.0),
        "diff_days_since_non_clay":     base.get("days_since_non_clay_a", 0.0)        - base.get("days_since_non_clay_b", 0.0),
        "diff_clay_streak_tournaments": base.get("clay_streak_tournaments_a", 0.0)    - base.get("clay_streak_tournaments_b", 0.0),
        "diff_win_rate_rg_3yr":         base.get("win_rate_rg_3yr_a", 0.5)            - base.get("win_rate_rg_3yr_b", 0.5),
        "diff_best_round_rg_3yr":       base.get("best_round_rg_3yr_a", 1.0)          - base.get("best_round_rg_3yr_b", 1.0),
        # Age × clay-Elo interaction
        "age_clay_elo_interaction_a":   age_clay_a,
        "age_clay_elo_interaction_b":   age_clay_b,
        "diff_age_clay_elo_interaction": age_clay_a - age_clay_b,
        # Return performance (Phase 3)
        "diff_return_pts_won_pct":      base.get("return_pts_won_pct_a", 0.35)        - base.get("return_pts_won_pct_b", 0.35),
        # Quality-adjusted win rate (Phase 3)
        "diff_clay_win_rate_vs_top50":  base.get("clay_win_rate_vs_top50_12m_a", 0.5) - base.get("clay_win_rate_vs_top50_12m_b", 0.5),
    }
    base = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1).copy()

    output_cols = [
        "match_id", "tourney_date", "player_a", "player_b", "target",
        "diff_standard_elo", "diff_clay_elo", "diff_welo", "diff_adjusted_elo",
        "elo_a_std", "elo_b_std", "elo_a_clay", "elo_b_clay", "elo_a_welo", "elo_b_welo",
        "diff_win_rate_clay_12m", "diff_win_rate_clay_6m", "diff_win_rate_30d",
        "diff_matches_21d", "diff_sets_21d", "diff_minutes_21d", "diff_win_rate_last10",
        "matches_21d_a", "matches_21d_b", "sets_21d_a", "sets_21d_b", "minutes_21d_a", "minutes_21d_b",
        "win_rate_clay_12m_a", "win_rate_clay_12m_b",
        "h2h_clay_wins_a", "h2h_clay_total", "h2h_clay_rate", "h2h_clay_rate_recent",
        "rg_win_rate_a", "rg_win_rate_b", "diff_rg_win_rate", "rg_matches_a", "rg_matches_b",
        "best_round_rg_a", "best_round_rg_b", "diff_best_round_rg",
        "ranking_diff", "log_ranking_diff",
        "age_a", "age_b", "age_optimal_a", "age_optimal_b", "diff_age_optimal",
        "diff_first_serve_pct", "diff_first_serve_won_pct", "diff_bp_saved_pct",
        "first_serve_pct_a", "first_serve_won_pct_a", "bp_saved_pct_a",
        "round_number", "diff_sets_played_rg",
        # Phase-2 new features
        "diff_win_rate_clay_90d", "diff_win_streak_clay",
        "diff_pct_3plus_sets_12m", "diff_pct_5sets_12m", "diff_avg_match_duration_12m",
        "diff_days_since_non_clay", "diff_clay_streak_tournaments",
        "diff_win_rate_rg_3yr", "diff_best_round_rg_3yr",
        "diff_age_clay_elo_interaction",
        "win_rate_rg_3yr_a", "win_rate_rg_3yr_b", "best_round_rg_3yr_a", "best_round_rg_3yr_b",
        "age_clay_elo_interaction_a", "age_clay_elo_interaction_b",
        # Phase-3 new features
        "diff_return_pts_won_pct", "diff_clay_win_rate_vs_top50",
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
    if "h2h_clay_rate_recent" in feature_df.columns:
        df_neg["h2h_clay_rate_recent"] = 1.0 - feature_df["h2h_clay_rate_recent"]
    df_neg["ranking_diff"]    = -feature_df["ranking_diff"]
    df_neg["log_ranking_diff"] = -feature_df["log_ranking_diff"]

    return pd.concat([df_pos, df_neg], ignore_index=True)


# ---------------------------------------------------------------------------
# Colonnes de features pour le modèle
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Elo : 2 features indépendantes (ajusté + welo), les autres sont redondantes (corr >0.96)
    "diff_adjusted_elo", "diff_welo",
    # Form récente
    "diff_win_rate_clay_12m", "diff_win_rate_clay_6m", "diff_win_rate_30d",
    "diff_win_rate_last10",
    # Momentum clay (Phase 2)
    "diff_win_rate_clay_90d", "diff_win_streak_clay",
    # Fatigue / charge de matchs (features les plus importantes)
    "diff_matches_21d", "diff_sets_21d", "diff_minutes_21d",
    "matches_21d_a", "matches_21d_b", "sets_21d_a", "sets_21d_b",
    "minutes_21d_a", "minutes_21d_b",
    # H2H clay
    "h2h_clay_rate", "h2h_clay_rate_recent", "h2h_clay_total",
    # Performance Roland Garros (carrière + 3 ans glissant)
    "diff_rg_win_rate", "rg_matches_a", "rg_matches_b", "diff_best_round_rg",
    "diff_win_rate_rg_3yr", "diff_best_round_rg_3yr",
    # Classement ATP
    "ranking_diff", "log_ranking_diff",
    # Âge + interaction non-linéaire (Phase 2)
    "age_a", "age_b",
    "diff_age_clay_elo_interaction",
    # Service / Return (signal faible mais non nul)
    "diff_first_serve_won_pct", "diff_bp_saved_pct", "diff_return_pts_won_pct",
    # Quality-adjusted clay win rate (vs top-50)
    "diff_clay_win_rate_vs_top50",
    # Endurance / style de jeu (Phase 2)
    "diff_pct_3plus_sets_12m", "diff_pct_5sets_12m", "diff_avg_match_duration_12m",
    # Transition de surface (Phase 2)
    "diff_days_since_non_clay", "diff_clay_streak_tournaments",
    # Contexte tournoi
    "round_number",
]
