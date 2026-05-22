"""
Calcul des systèmes Elo : Standard, Surface-specific, WElo, AdjustedElo.
Tous les ratings sont calculés AVANT chaque match (pas de data leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


ELO_INIT = 1500.0
ELO_K_DEFAULT = 32.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass
class PlayerElo:
    std: float = ELO_INIT
    clay: float = ELO_INIT
    hard: float = ELO_INIT
    grass: float = ELO_INIT
    carpet: float = ELO_INIT
    welo: float = ELO_INIT
    last_result: Optional[int] = None  # 1 = win, 0 = loss (pour WElo)
    matches_played: int = 0


class EloSystem:
    """
    Calcule et maintient 4 systèmes Elo pour chaque joueur.
    Appeler compute(df) retourne df enrichi avec les ratings pré-match.
    """

    def __init__(self, alpha: float = 0.3, lambda_adj: float = 0.5):
        """
        alpha  : hyperparamètre WElo (poids du hot-hand)
        lambda : mix AdjustedElo = (1-λ)*StandardElo + λ*SurfaceElo
        """
        self.alpha = alpha
        self.lambda_adj = lambda_adj
        self._ratings: dict[str, PlayerElo] = {}

    def _get(self, player: str) -> PlayerElo:
        if player not in self._ratings:
            self._ratings[player] = PlayerElo()
        return self._ratings[player]

    def _surface_attr(self, surface: str) -> str:
        return {"Clay": "clay", "Hard": "hard", "Grass": "grass", "Carpet": "carpet"}.get(surface, "hard")

    def _update_std(self, winner: str, loser: str, k: float) -> None:
        rw = self._get(winner)
        rl = self._get(loser)
        exp_w = _expected(rw.std, rl.std)
        delta = k * (1.0 - exp_w)
        rw.std += delta
        rl.std -= delta

    def _update_surface(self, winner: str, loser: str, surface: str, k: float) -> None:
        attr = self._surface_attr(surface)
        rw = self._get(winner)
        rl = self._get(loser)
        rw_s = getattr(rw, attr)
        rl_s = getattr(rl, attr)
        exp_w = _expected(rw_s, rl_s)
        delta = k * (1.0 - exp_w)
        setattr(rw, attr, rw_s + delta)
        setattr(rl, attr, rl_s - delta)

    def _update_welo(self, winner: str, loser: str, k: float) -> None:
        rw = self._get(winner)
        rl = self._get(loser)
        # K pondéré par hot-hand
        k_w = k * (1.0 + self.alpha * (rw.last_result or 0))
        k_l = k * (1.0 + self.alpha * (rl.last_result or 0))
        exp_w = _expected(rw.welo, rl.welo)
        rw.welo += k_w * (1.0 - exp_w)
        rl.welo -= k_l * exp_w
        rw.last_result = 1
        rl.last_result = 0

    def _adjusted_elo(self, player: str, surface: str) -> float:
        r = self._get(player)
        surf_elo = getattr(r, self._surface_attr(surface))
        return (1.0 - self.lambda_adj) * r.std + self.lambda_adj * surf_elo

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parcourt df chronologiquement et retourne df avec colonnes :
          winner_{std,clay,hard,grass,welo,adjusted}_elo_pre
          loser_{std,clay,hard,grass,welo,adjusted}_elo_pre
        """
        self._ratings.clear()

        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Calcul Elo", unit="match", ncols=80):
            w, l = row["winner_name"], row["loser_name"]
            surface = row.get("surface", "Hard")
            k = float(row.get("k_factor", ELO_K_DEFAULT))

            rw = self._get(w)
            rl = self._get(l)
            surf_attr = self._surface_attr(surface)

            # Snapshot pré-match
            record = {
                "match_id": row.get("match_id", ""),
                "winner_std_elo_pre": rw.std,
                "winner_clay_elo_pre": rw.clay,
                "winner_hard_elo_pre": rw.hard,
                "winner_grass_elo_pre": rw.grass,
                "winner_welo_pre": rw.welo,
                "winner_adjusted_elo_pre": self._adjusted_elo(w, surface),
                "winner_surface_elo_pre": getattr(rw, surf_attr),
                "loser_std_elo_pre": rl.std,
                "loser_clay_elo_pre": rl.clay,
                "loser_hard_elo_pre": rl.hard,
                "loser_grass_elo_pre": rl.grass,
                "loser_welo_pre": rl.welo,
                "loser_adjusted_elo_pre": self._adjusted_elo(l, surface),
                "loser_surface_elo_pre": getattr(rl, surf_attr),
            }
            records.append(record)

            # Mise à jour post-match
            self._update_std(w, l, k)
            self._update_surface(w, l, surface, k)
            self._update_welo(w, l, k)

            rw.matches_played += 1
            rl.matches_played += 1

        elo_df = pd.DataFrame(records)
        return df.merge(elo_df, on="match_id", how="left")

    def get_player_ratings(self, player: str, surface: str = "Clay") -> dict:
        """Retourne les ratings actuels d'un joueur (pour prédiction live)."""
        r = self._get(player)
        return {
            "std_elo": r.std,
            "clay_elo": r.clay,
            "hard_elo": r.hard,
            "grass_elo": r.grass,
            "welo": r.welo,
            "adjusted_elo": self._adjusted_elo(player, surface),
        }

    def update_single_match(self, winner: str, loser: str, surface: str, k: float = 40.0) -> None:
        """Mise à jour incrémentale après saisie d'un résultat RG 2026."""
        self._update_std(winner, loser, k)
        self._update_surface(winner, loser, surface, k)
        self._update_welo(winner, loser, k)
        self._get(winner).matches_played += 1
        self._get(loser).matches_played += 1


def optimize_hyperparams(df: pd.DataFrame, rg_df: pd.DataFrame) -> dict:
    """
    Grid search sur alpha et lambda en minimisant le Brier score sur RG 2017-2024.
    Retourne les meilleurs hyperparamètres.
    """
    from sklearn.metrics import brier_score_loss

    best_score = float("inf")
    best_params = {"alpha": 0.3, "lambda_adj": 0.5}

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for lam in [0.3, 0.4, 0.5, 0.6, 0.7]:
            elo = EloSystem(alpha=alpha, lambda_adj=lam)
            enriched = elo.compute(df)

            # Évaluer sur RG uniquement
            rg_enriched = enriched[enriched["match_id"].isin(rg_df["match_id"])]
            if rg_enriched.empty:
                continue

            exp_winner = 1.0 / (
                1.0 + 10.0 ** (
                    (rg_enriched["loser_adjusted_elo_pre"] - rg_enriched["winner_adjusted_elo_pre"]) / 400.0
                )
            )
            score = brier_score_loss(np.ones(len(rg_enriched)), exp_winner)
            if score < best_score:
                best_score = score
                best_params = {"alpha": alpha, "lambda_adj": lam}

    print(f"Meilleurs hyperparams Elo: {best_params} (Brier={best_score:.4f})")
    return best_params
