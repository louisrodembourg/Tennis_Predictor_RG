"""
Classe principale RolandGarrosPredictor.
Gère le cycle complet : chargement → Elo → features → modèle → prédictions live.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cache_manager import load_cache, save_cache
from data_loader import filter_clay, filter_roland_garros, load_matches
from elo import EloSystem
from features import FEATURE_COLS, HistoryIndex, build_features, build_symmetric_dataset
from model import XGB_PARAMS, train_model, train_mini_model, dynamic_alpha


class RolandGarrosPredictor:
    def __init__(
        self,
        historical_data_path: str,
        rg2026_path: str = "data/rg2026/results.jsonl",
        alpha: float = 0.3,
        lambda_adj: float = 0.5,
    ):
        self.data_path = historical_data_path
        self.rg2026_path = Path(rg2026_path)
        self.rg2026_path.parent.mkdir(parents=True, exist_ok=True)

        cached = load_cache(historical_data_path)
        if cached:
            # --- Chargement depuis le cache (quelques secondes) ---
            self.df          = cached["df"]
            self.df_with_elo = cached["df_with_elo"]
            self.clay_df     = filter_clay(self.df)
            self.rg_df       = filter_roland_garros(self.df)
            self._feature_df = cached["all_features"]
            self._feature_df = self._feature_df[self._feature_df["round_number"] >= 1].copy()
            self.model       = cached["model"]

            # Reconstruire EloSystem à partir de l'état persisté
            print("Reconstruction des ratings Elo (depuis cache)...")
            self.elo = EloSystem(alpha=alpha, lambda_adj=lambda_adj)
            self.elo.compute(self.df)  # rapide grâce au cache data — rejoue depuis zéro

            # Reconstruire HistoryIndex depuis les tables pré-calculées
            print("Reconstruction de l'index historique (depuis cache)...")
            self._hist_index = HistoryIndex.__new__(HistoryIndex)
            self._hist_index._players = {}
            self._hist_index._rg = {}
            self._hist_index._build_player_index(self.df)
            self._hist_index._build_rg_index(self.rg_df)
            self._hist_index.player_state = cached["player_state"]
            self._hist_index.rg_state     = cached["rg_state"]
            self._hist_index.h2h_state    = cached["h2h_state"]
        else:
            # --- Calcul complet + sauvegarde dans le cache ---
            print("Chargement des données historiques...")
            self.df = load_matches(historical_data_path)
            self.clay_df = filter_clay(self.df)
            self.rg_df = filter_roland_garros(self.df)

            print("Calcul des ratings Elo...")
            self.elo = EloSystem(alpha=alpha, lambda_adj=lambda_adj)
            self.df_with_elo = self.elo.compute(self.df)

            print("Construction de l'index historique...")
            self._hist_index = HistoryIndex(self.df, self.rg_df)

            print("Construction des features...")
            self._feature_df = self._build_all_features()
            # Exclure les qualifications de l'entraînement
            self._feature_df = self._feature_df[self._feature_df["round_number"] >= 1].copy()

            print("Entraînement du modèle initial...")
            self.model = train_model(self._feature_df, FEATURE_COLS)

            save_cache(historical_data_path, {
                "df":           self.df,
                "df_with_elo":  self.df_with_elo,
                "player_state": self._hist_index.player_state,
                "rg_state":     self._hist_index.rg_state,
                "h2h_state":    self._hist_index.h2h_state,
                "all_features": self._feature_df,
                "model":        self.model,
            })

        # État intra-tournoi RG 2026 (toujours rechargé depuis le JSONL)
        self._intra_rg: dict[str, dict] = {}
        self._rg_model = None                                # mini-modèle RG 2026
        self._rg2026_results: list[dict] = self._load_rg2026_results()

        if self._rg2026_results:
            print(f"Réintégration de {len(self._rg2026_results)} résultats RG 2026...")
            for result in self._rg2026_results:
                self._apply_result(result, retrain=False)
            self._retrain()

        print("Prédicteur prêt.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_match(self, player_a: str, player_b: str, round_number: int = 1) -> dict:
        """
        Prédit le vainqueur entre player_a et player_b.
        Retourne : winner_predicted, proba_a, proba_b, confidence, top_features.
        """
        features = self._compute_match_features(player_a, player_b, round_number)
        if features is None:
            return {
                "winner_predicted": player_a,
                "proba_a": 0.5,
                "proba_b": 0.5,
                "confidence": "faible",
                "top_features": [],
            }

        X = features[FEATURE_COLS].fillna(0).values.reshape(1, -1)
        base_p = float(self.model.predict_proba(X)[0][1])
        if self._rg_model is not None:
            rg_p = float(self._rg_model.predict_proba(X)[0][1])
            alpha = dynamic_alpha(len(self._rg2026_results) * 2, round_number=round_number)
            proba_a = alpha * base_p + (1.0 - alpha) * rg_p
        else:
            proba_a = base_p
        proba_b = 1.0 - proba_a

        winner = player_a if proba_a >= 0.5 else player_b

        gap = abs(proba_a - 0.5)
        if gap > 0.2:
            confidence = "haute"
        elif gap > 0.1:
            confidence = "moyenne"
        else:
            confidence = "faible"

        # Score numérique 0-100 : 0 = match parfaitement incertain, 100 = certitude absolue
        confidence_score = round(gap * 200, 1)

        top_features = self._get_top_features(features, player_a, player_b)

        return {
            "winner_predicted": winner,
            "proba_a": round(proba_a, 4),
            "proba_b": round(proba_b, 4),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "top_features": top_features,
        }

    def add_result(
        self,
        winner: str,
        loser: str,
        score: str,
        round_number: int,
        match_date: str = "2026-05-25",
    ) -> dict:
        """
        Enregistre un résultat RG 2026, met à jour les Elo et réentraîne le modèle.
        round_number < 1 = qualifications (fatigue trackée, pas dans le tableau principal).
        """
        result = {
            "winner": winner,
            "loser": loser,
            "score": score,
            "round_number": round_number,
            "tourney_date": match_date,
        }
        self._apply_result(result, retrain=True)
        self._save_rg2026_result(result)

        print(f"  Résultat enregistré : {winner} bat {loser} ({score}) — Tour {round_number}")
        return {"status": "ok", "winner": winner, "loser": loser}

    def predict_tournament(self, draw: dict) -> pd.DataFrame:
        """
        draw = {1: [("Joueur A", "Joueur B"), ...], 2: [...], ...}
        Retourne un DataFrame des prédictions par match.
        """
        rows = []
        for round_num, matches in draw.items():
            for pa, pb in matches:
                pred = self.predict_match(pa, pb, int(round_num))
                rows.append({
                    "round": round_num,
                    "player_a": pa,
                    "player_b": pb,
                    "winner_predicted": pred["winner_predicted"],
                    "proba_a": pred["proba_a"],
                    "proba_b": pred["proba_b"],
                    "confidence": pred["confidence"],
                })
        return pd.DataFrame(rows)

    def simulate_tournament(
        self,
        remaining_players: list[str],
        n_simulations: int = 10000,
        round_start: int = 1,
    ) -> pd.DataFrame:
        """
        Monte Carlo : simule n fois le tournoi avec les joueurs restants.
        Retourne probabilité de victoire finale par joueur.
        """
        if len(remaining_players) == 0:
            return pd.DataFrame()

        n = len(remaining_players)
        # Pad à la prochaine puissance de 2
        size = 1
        while size < n:
            size *= 2
        players = remaining_players + ["BYE"] * (size - n)

        win_counts = {p: 0 for p in remaining_players}

        for _ in range(n_simulations):
            bracket = players[:]
            round_num = round_start

            while len(bracket) > 1:
                next_round = []
                for i in range(0, len(bracket), 2):
                    pa, pb = bracket[i], bracket[i + 1]
                    if pb == "BYE":
                        next_round.append(pa)
                        continue
                    if pa == "BYE":
                        next_round.append(pb)
                        continue
                    pred = self.predict_match(pa, pb, round_num)
                    r = np.random.random()
                    winner = pa if r < pred["proba_a"] else pb
                    next_round.append(winner)
                bracket = next_round
                round_num += 1

            champion = bracket[0]
            if champion in win_counts:
                win_counts[champion] += 1

        rows = [
            {"player": p, "win_proba": win_counts[p] / n_simulations}
            for p in remaining_players
        ]
        return pd.DataFrame(rows).sort_values("win_proba", ascending=False).reset_index(drop=True)

    def get_player_profile(self, player: str) -> dict:
        """Retourne le profil complet d'un joueur (Elo + stats)."""
        ratings = self.elo.get_player_ratings(player, "Clay")

        # Dernière ligne du joueur dans les données
        mask = (self.df["winner_name"] == player) | (self.df["loser_name"] == player)
        recent = self.df[mask].tail(1)
        rank = None
        if not recent.empty:
            row = recent.iloc[0]
            rank = row["winner_rank"] if row["winner_name"] == player else row["loser_rank"]

        return {
            "player": player,
            "rank": rank,
            **ratings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_all_features(self) -> pd.DataFrame:
        """Construit les features pour tous les matchs clay historiques."""
        clay_with_elo = self.df_with_elo[self.df_with_elo["surface"] == "Clay"].copy()
        return build_features(
            df=clay_with_elo,
            history=self.df,
            rg_history=self.rg_df,
            elo_df=self.df_with_elo,
            index=self._hist_index,
        )

    def _compute_match_features(
        self, player_a: str, player_b: str, round_number: int
    ) -> Optional[pd.Series]:
        """Calcule les features d'un match futur pour la prédiction live."""
        import hashlib

        pseudo_match = pd.DataFrame([{
            "match_id": hashlib.md5(f"{player_a}{player_b}live".encode()).hexdigest()[:12],
            "tourney_date": pd.Timestamp.now(),
            "tourney_name": "Roland Garros",
            "surface": "Clay",
            "round": {1: "R128", 2: "R64", 3: "R32", 4: "R16", 5: "QF", 6: "SF", 7: "F"}.get(round_number, "R32"),
            "round_number": round_number,
            "winner_name": player_a,
            "loser_name": player_b,
            "winner_rank": self._get_rank(player_a),
            "loser_rank": self._get_rank(player_b),
            "winner_age": self._get_age(player_a),
            "loser_age": self._get_age(player_b),
            "k_factor": 40,
        }])

        ratings_a = self.elo.get_player_ratings(player_a, "Clay")
        ratings_b = self.elo.get_player_ratings(player_b, "Clay")

        # Injecter les Elo actuels dans elo_df temporaire
        elo_row = pd.DataFrame([{
            "match_id": pseudo_match.iloc[0]["match_id"],
            "winner_std_elo_pre": ratings_a["std_elo"],
            "winner_clay_elo_pre": ratings_a["clay_elo"],
            "winner_hard_elo_pre": ratings_a["hard_elo"],
            "winner_grass_elo_pre": ratings_a["grass_elo"],
            "winner_welo_pre": ratings_a["welo"],
            "winner_adjusted_elo_pre": ratings_a["adjusted_elo"],
            "winner_surface_elo_pre": ratings_a["clay_elo"],
            "loser_std_elo_pre": ratings_b["std_elo"],
            "loser_clay_elo_pre": ratings_b["clay_elo"],
            "loser_hard_elo_pre": ratings_b["hard_elo"],
            "loser_grass_elo_pre": ratings_b["grass_elo"],
            "loser_welo_pre": ratings_b["welo"],
            "loser_adjusted_elo_pre": ratings_b["adjusted_elo"],
            "loser_surface_elo_pre": ratings_b["clay_elo"],
        }])

        feat_df = build_features(
            df=pseudo_match,
            history=self.df,
            rg_history=self.rg_df,
            elo_df=elo_row,
            intra_rg=self._intra_rg,
            index=self._hist_index,
        )

        if feat_df.empty:
            return None
        return feat_df.iloc[0]

    def _get_top_features(
        self, features: pd.Series, player_a: str, player_b: str, top_n: int = 5
    ) -> list[dict]:
        """Top features les plus influentes avec direction (faveur A ou B)."""
        _FEATURE_LABELS = {
            "diff_standard_elo": "Elo standard",
            "diff_clay_elo": "Elo clay",
            "diff_welo": "WElo",
            "diff_adjusted_elo": "Elo ajusté",
            "diff_win_rate_clay_12m": "Win rate clay 12 mois",
            "diff_win_rate_clay_6m": "Win rate clay 6 mois",
            "diff_win_rate_30d": "Win rate 30 jours",
            "diff_matches_21d": "Matchs 21 jours (fatigue)",
            "diff_sets_21d": "Sets 21 jours (fatigue)",
            "diff_minutes_21d": "Minutes 21 jours (fatigue)",
            "diff_win_rate_last10": "Win rate 10 derniers matchs",
            "h2h_clay_rate": "H2H clay",
            "diff_rg_win_rate": "Win rate Roland Garros",
            "diff_best_round_rg": "Meilleur tour RG",
            "ranking_diff": "Classement ATP",
            "log_ranking_diff": "Classement (log)",
            "diff_first_serve_pct": "% 1er service",
            "diff_first_serve_won_pct": "% pts gagnés sur 1er service",
            "diff_bp_saved_pct": "% balles de break sauvées",
            "round_number": "Tour du tournoi",
            "diff_sets_played_rg": "Sets joués ce RG (intra-tournoi)",
        }
        try:
            base_model = self.model
            if hasattr(base_model, "estimators_"):
                importances = base_model.estimators_[0].estimator.feature_importances_
            elif hasattr(base_model, "estimator"):
                importances = base_model.estimator.feature_importances_
            else:
                importances = base_model.feature_importances_

            top_idx = np.argsort(importances)[::-1][:top_n]
            result = []
            for i in top_idx:
                if i >= len(FEATURE_COLS):
                    continue
                col = FEATURE_COLS[i]
                val = float(features.get(col, 0) or 0)
                # Pour les features "diff_*" et ranking_diff : positif = avantage A
                # Pour h2h_clay_rate : > 0.5 = avantage A
                # Pour round_number : neutre
                if col in ("round_number", "h2h_clay_total", "rg_matches_a", "rg_matches_b",
                           "matches_21d_a", "matches_21d_b", "sets_21d_a", "sets_21d_b",
                           "age_a", "age_b"):
                    favors = "neutre"
                elif col == "h2h_clay_rate":
                    favors = player_a if val > 0.5 else (player_b if val < 0.5 else "neutre")
                elif col == "diff_matches_21d":
                    # Plus de matchs = plus de fatigue = désavantage
                    favors = player_b if val > 0 else (player_a if val < 0 else "neutre")
                elif col == "diff_sets_21d" or col == "diff_minutes_21d":
                    favors = player_b if val > 0 else (player_a if val < 0 else "neutre")
                elif col == "ranking_diff":
                    # ranking_diff = rank_a - rank_b ; rang plus bas = meilleur
                    favors = player_b if val > 0 else (player_a if val < 0 else "neutre")
                else:
                    favors = player_a if val > 0 else (player_b if val < 0 else "neutre")

                result.append({
                    "feature": _FEATURE_LABELS.get(col, col),
                    "value": round(val, 4),
                    "importance": round(float(importances[i]), 4),
                    "favors": favors,
                })
            return result
        except Exception:
            return []

    def _apply_result(self, result: dict, retrain: bool = True) -> None:
        """Met à jour Elo, historique RG et features intra-tournoi."""
        winner = result["winner"]
        loser = result["loser"]
        round_num = result["round_number"]

        # Mise à jour Elo
        self.elo.update_single_match(winner, loser, "Clay", k=40.0)

        # Mise à jour état intra-tournoi
        score = result.get("score", "6-4 6-4")
        sets_played = score.count("-")
        for player in [winner, loser]:
            if player not in self._intra_rg:
                self._intra_rg[player] = {"sets_played": 0, "matches_played": 0}
            self._intra_rg[player]["sets_played"] += sets_played
            self._intra_rg[player]["matches_played"] += 1

        if retrain:
            self._retrain()

    def _retrain(self) -> None:
        """Entraîne le mini-modèle RG sur les résultats RG 2026 accumulés (modèle de base inchangé)."""
        rg2026_matches = self._build_rg2026_feature_df()
        if rg2026_matches is not None and not rg2026_matches.empty:
            self._rg_model = train_mini_model(rg2026_matches, FEATURE_COLS)
        else:
            self._rg_model = None

    def _build_rg2026_feature_df(self) -> Optional[pd.DataFrame]:
        if not self._rg2026_results:
            return None

        rows = []
        for r in self._rg2026_results:
            rows.append({
                "match_id": f"rg2026_{r['winner']}_{r['loser']}",
                "tourney_date": pd.Timestamp("2026-05-25"),
                "tourney_name": "Roland Garros",
                "surface": "Clay",
                "round_number": r["round_number"],
                "round": "F",
                "winner_name": r["winner"],
                "loser_name": r["loser"],
                "winner_rank": self._get_rank(r["winner"]),
                "loser_rank": self._get_rank(r["loser"]),
                "winner_age": self._get_age(r["winner"]),
                "loser_age": self._get_age(r["loser"]),
                "k_factor": 40,
            })

        df_2026 = pd.DataFrame(rows)
        return build_features(
            df=df_2026,
            history=self.df,
            rg_history=self.rg_df,
            elo_df=None,
            intra_rg=self._intra_rg,
            index=self._hist_index,
        )

    def _get_rank(self, player: str) -> float:
        mask = (self.df["winner_name"] == player) | (self.df["loser_name"] == player)
        recent = self.df[mask].tail(1)
        if recent.empty:
            return 100.0
        row = recent.iloc[0]
        rank = row["winner_rank"] if row["winner_name"] == player else row["loser_rank"]
        return float(rank) if pd.notna(rank) else 100.0

    def _get_age(self, player: str) -> float:
        mask = (self.df["winner_name"] == player) | (self.df["loser_name"] == player)
        recent = self.df[mask].tail(1)
        if recent.empty:
            return 25.0
        row = recent.iloc[0]
        age = row["winner_age"] if row["winner_name"] == player else row["loser_age"]
        return float(age) if pd.notna(age) else 25.0

    def _save_rg2026_result(self, result: dict) -> None:
        with open(self.rg2026_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _load_rg2026_results(self) -> list[dict]:
        if not self.rg2026_path.exists():
            return []
        results = []
        with open(self.rg2026_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results
