"""
Module de simulation de paris sportifs sur Roland Garros.

Fonctionnalités :
  - Téléchargement des cotes historiques depuis tennis-data.co.uk
  - Calcul de l'Expected Value (EV)
  - BankrollSimulator avec 5 stratégies Kelly / mise fixe
  - Métriques : ROI, drawdown, Sharpe, paris gagnants
"""

from __future__ import annotations

import difflib
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ODDS_DIR = Path(__file__).parent.parent / "data" / "odds"

ROUND_MAP = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Quarterfinals": 5, "Semifinals": 6, "Final": 7, "The Final": 7,
    "QF": 5, "SF": 6, "F": 7,
}

ROUND_KELLY_FACTOR = {1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.5, 6: 0.0, 7: 0.0}

EV_TIER_STAKES = [
    (0.15, 0.03),
    (0.07, 0.02),
    (0.03, 0.01),
]


# ---------------------------------------------------------------------------
# Normalisation des noms de joueurs
# ---------------------------------------------------------------------------

def _ascii_norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def _sackmann_to_abbrev(name: str) -> str:
    """Convertit "Novak Djokovic" → "Djokovic N." (format tennis-data.co.uk)."""
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    first, rest = parts[0], parts[1:]
    last = " ".join(rest)
    return f"{last} {first[0]}."


def _build_name_lookup(sackmann_names: list[str]) -> dict[str, str]:
    """Construit {abbrev_normalized: original_name} pour le matching fuzzy."""
    lookup: dict[str, str] = {}
    for name in sackmann_names:
        abbrev = _sackmann_to_abbrev(name)
        key = _ascii_norm(abbrev)
        lookup[key] = name
    return lookup


def match_player_name(
    td_name: str,
    lookup: dict[str, str],
    cutoff: float = 0.80,
) -> Optional[str]:
    """
    Retourne le nom Sackmann correspondant à un nom tennis-data.co.uk.
    Essaie d'abord un match exact (normalisé ASCII), puis fuzzy si nécessaire.
    """
    key = _ascii_norm(td_name.strip())
    if key in lookup:
        return lookup[key]
    matches = difflib.get_close_matches(key, lookup.keys(), n=1, cutoff=cutoff)
    if matches:
        return lookup[matches[0]]
    return None


# ---------------------------------------------------------------------------
# Téléchargement des cotes
# ---------------------------------------------------------------------------

def download_rg_odds(
    years: list[int] | None = None,
    dest_dir: Path | str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Télécharge les fichiers de cotes Roland Garros depuis tennis-data.co.uk.
    Sauvegarde en CSV dans data/odds/rg_odds_YYYY.csv.
    Retourne un DataFrame combiné.
    """
    import requests

    if years is None:
        years = list(range(2017, 2026))
    if dest_dir is None:
        dest_dir = ODDS_DIR
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dfs: list[pd.DataFrame] = []
    for year in years:
        csv_path = dest_dir / f"rg_odds_{year}.csv"
        if csv_path.exists() and not force:
            dfs.append(pd.read_csv(csv_path))
            continue

        # tennis-data.co.uk stocke toute l'année dans un seul fichier {year}.xlsx
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"  [WARN] Impossible de télécharger les cotes {year}: {e}")
            continue

        from io import BytesIO
        try:
            df = pd.read_excel(BytesIO(resp.content))
        except Exception as e2:
            logger.warning(f"  [WARN] Impossible de lire le fichier {year}: {e2}")
            continue

        # Filtrer pour ne garder que Roland Garros
        if "Tournament" in df.columns:
            df = df[df["Tournament"] == "French Open"].copy()
        if df.empty:
            logger.warning(f"  [WARN] Aucun match French Open trouvé dans les données {year}")
            continue

        df = _normalize_odds_df(df, year)
        df.to_csv(csv_path, index=False)
        dfs.append(df)
        logger.info(f"  Cotes {year} téléchargées ({len(df)} matchs)")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_rg_odds(
    years: list[int] | None = None,
    dest_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Charge les cotes depuis les CSV locaux."""
    if years is None:
        years = list(range(2017, 2026))
    if dest_dir is None:
        dest_dir = ODDS_DIR
    dest_dir = Path(dest_dir)

    dfs: list[pd.DataFrame] = []
    for year in years:
        csv_path = dest_dir / f"rg_odds_{year}.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _normalize_odds_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Normalise les colonnes d'un fichier tennis-data.co.uk."""
    df = df.copy()
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date", "tournament_date"):
            col_map[c] = "date"
        elif cl in ("winner", "winner1"):
            col_map[c] = "winner_td"
        elif cl in ("loser", "loser1"):
            col_map[c] = "loser_td"
        elif cl == "round":
            col_map[c] = "round_str"
        elif cl in ("b365w", "b365_w"):
            col_map[c] = "odds_winner"
        elif cl in ("b365l", "b365_l"):
            col_map[c] = "odds_loser"
        elif cl in ("psw", "ps_w") and "odds_winner" not in col_map.values():
            col_map[c] = "odds_winner"
        elif cl in ("psl", "ps_l") and "odds_loser" not in col_map.values():
            col_map[c] = "odds_loser"
        elif cl in ("avgw", "avg_w") and "odds_winner" not in col_map.values():
            col_map[c] = "odds_winner"
        elif cl in ("avgl", "avg_l") and "odds_loser" not in col_map.values():
            col_map[c] = "odds_loser"

    df = df.rename(columns=col_map)
    keep = [c for c in ["date", "winner_td", "loser_td", "round_str", "odds_winner", "odds_loser"] if c in df.columns]
    df = df[keep].copy()

    df["year"] = year
    if "round_str" in df.columns:
        df["round_number"] = df["round_str"].map(ROUND_MAP)
    if "odds_winner" in df.columns:
        df["odds_winner"] = pd.to_numeric(df["odds_winner"], errors="coerce")
    if "odds_loser" in df.columns:
        df["odds_loser"] = pd.to_numeric(df["odds_loser"], errors="coerce")

    return df.dropna(subset=["winner_td", "loser_td"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Jointure prédictions ↔ cotes
# ---------------------------------------------------------------------------

def join_predictions_to_odds(
    preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    cutoff: float = 0.80,
) -> pd.DataFrame:
    """
    Joint les prédictions du backtest aux cotes bookmaker.

    preds_df doit avoir : year, round_number, player_a (winner), player_b (loser), p_blend, target(=1)
    odds_df doit avoir  : year, round_number, winner_td, loser_td, odds_winner, odds_loser

    Retourne TWO rows per matched match (one bet on each player) :
      player_bet, opponent, p_model, odds, won (1/0), year, round_number
    """
    if preds_df.empty or odds_df.empty:
        return pd.DataFrame()

    all_names = pd.concat([preds_df["player_a"], preds_df["player_b"]]).unique().tolist()
    lookup = _build_name_lookup(all_names)

    odds_df = odds_df.copy()
    odds_df["winner_sack"] = odds_df["winner_td"].apply(
        lambda n: match_player_name(str(n), lookup, cutoff)
    )
    odds_df["loser_sack"] = odds_df["loser_td"].apply(
        lambda n: match_player_name(str(n), lookup, cutoff)
    )

    unmatched = odds_df[odds_df["winner_sack"].isna() | odds_df["loser_sack"].isna()]
    if not unmatched.empty:
        logger.warning(f"  {len(unmatched)} matchs sans correspondance de nom (sur {len(odds_df)})")

    odds_df = odds_df.dropna(subset=["winner_sack", "loser_sack"])

    # Merge sur winner=player_a, loser=player_b (le sens naturel du backtest)
    merged = preds_df.merge(
        odds_df[["year", "round_number", "winner_sack", "loser_sack", "odds_winner", "odds_loser"]],
        left_on=["year", "round_number", "player_a", "player_b"],
        right_on=["year", "round_number", "winner_sack", "loser_sack"],
        how="inner",
    )

    # Génère deux lignes par match : pari sur le vainqueur ET pari sur le perdant
    rows_winner = merged.assign(
        player_bet=merged["player_a"],
        opponent=merged["player_b"],
        p_model=merged["p_blend"],
        odds=merged["odds_winner"],
        won=1,                          # player_a a gagné → pari sur vainqueur = gagnant
    )
    rows_loser = merged.assign(
        player_bet=merged["player_b"],
        opponent=merged["player_a"],
        p_model=1.0 - merged["p_blend"],
        odds=merged["odds_loser"],
        won=0,                          # player_b a perdu → pari sur perdant = perdant
    )

    keep = ["year", "round_number", "player_bet", "opponent", "p_model", "odds", "won"]
    result = pd.concat([rows_winner[keep], rows_loser[keep]], ignore_index=True)
    result = result.sort_values(["year", "round_number"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Calcul EV
# ---------------------------------------------------------------------------

def calculate_ev(p_model: float, odds_decimal: float) -> float:
    """EV = odds * p_model - 1. Positif = pari avantageux."""
    return odds_decimal * p_model - 1.0


# ---------------------------------------------------------------------------
# BankrollSimulator
# ---------------------------------------------------------------------------

class BankrollSimulator:
    """
    Simule une gestion de bankroll sur les prédictions du backtest.

    Stratégies disponibles :
      - "full_kelly"    : mise = f* × bankroll
      - "half_kelly"    : mise = 0.5 × f* × bankroll
      - "capped_kelly"  : mise = min(f* × bankroll, 5% × bankroll)
      - "fixed_ev_tier" : mise par tranche d'EV (1%, 2%, 3% du bankroll)
      - "kelly_by_round": facteur Kelly par tour (R128=0.5×, R64/R32=1×, R16/QF=0.5×, SF/F=0×)

    Parameters
    ----------
    strategy : str
        Nom de la stratégie parmi celles listées ci-dessus.
    initial_bankroll : float
        Bankroll initiale (défaut 1000 €).
    min_ev_threshold : float
        EV minimum pour placer un pari (défaut 0.03 = 3%).
    kelly_cap : float
        Plafond de mise pour la stratégie capped_kelly (en fraction du bankroll, défaut 0.05).
    """

    def __init__(
        self,
        strategy: str = "half_kelly",
        initial_bankroll: float = 1000.0,
        min_ev_threshold: float = 0.03,
        kelly_cap: float = 0.05,
    ):
        if strategy not in ("full_kelly", "half_kelly", "capped_kelly", "fixed_ev_tier", "kelly_by_round"):
            raise ValueError(f"Stratégie inconnue: {strategy}")
        self.strategy = strategy
        self.initial_bankroll = initial_bankroll
        self.min_ev_threshold = min_ev_threshold
        self.kelly_cap = kelly_cap

    def run(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simule la stratégie sur toutes les prédictions.

        predictions_df doit avoir : year, round_number, player_bet, p_model, odds, won
          (won = 1 si le pari est gagnant, 0 sinon — généré par join_predictions_to_odds)

        Retourne un DataFrame match par match avec les colonnes :
          year, round_number, player_bet, ev, odds, stake, result, bankroll_after, pnl
        """
        df = predictions_df.copy().sort_values(["year", "round_number"]).reset_index(drop=True)
        bankroll = self.initial_bankroll
        rows: list[dict] = []

        for _, row in df.iterrows():
            p    = float(row["p_model"])
            odds = float(row.get("odds", np.nan))
            rn   = int(row.get("round_number", 1))

            if np.isnan(odds) or odds <= 1.0:
                continue

            ev = calculate_ev(p, odds)
            if ev <= self.min_ev_threshold:
                continue

            stake = self._compute_stake(ev, odds, bankroll, rn)
            stake = min(stake, bankroll)
            if stake <= 0:
                continue

            result = int(row["won"])
            pnl    = stake * (odds - 1.0) if result == 1 else -stake
            bankroll += pnl

            rows.append({
                "year":           int(row["year"]),
                "round_number":   rn,
                "player_bet":     row["player_bet"],
                "p_model":        p,
                "ev":             ev,
                "odds":           odds,
                "stake":          stake,
                "result":         result,
                "pnl":            pnl,
                "bankroll_after": bankroll,
            })

        return pd.DataFrame(rows)

    def _kelly_fraction(self, ev: float, odds: float) -> float:
        """f* = EV / (odds - 1). Retourne 0 si négatif."""
        denom = odds - 1.0
        if denom <= 0:
            return 0.0
        return max(0.0, ev / denom)

    def _compute_stake(self, ev: float, odds: float, bankroll: float, round_number: int) -> float:
        f_star = self._kelly_fraction(ev, odds)
        if self.strategy == "full_kelly":
            return f_star * bankroll
        elif self.strategy == "half_kelly":
            return 0.5 * f_star * bankroll
        elif self.strategy == "capped_kelly":
            return min(f_star * bankroll, self.kelly_cap * bankroll)
        elif self.strategy == "fixed_ev_tier":
            for ev_threshold, fraction in EV_TIER_STAKES:
                if ev >= ev_threshold:
                    return fraction * bankroll
            return 0.0
        elif self.strategy == "kelly_by_round":
            factor = ROUND_KELLY_FACTOR.get(round_number, 0.0)
            return factor * f_star * bankroll
        return 0.0


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(history_df: pd.DataFrame, initial_bankroll: float = 1000.0) -> dict:
    """
    Calcule les métriques de performance sur l'historique du bankroll.

    Retourne un dict avec : roi, max_drawdown, sharpe, profitable_years,
    n_bets, n_wins, roi_by_round
    """
    if history_df.empty:
        return {}

    final = float(history_df["bankroll_after"].iloc[-1])
    roi   = (final - initial_bankroll) / initial_bankroll

    # Max drawdown
    running_max = history_df["bankroll_after"].cummax()
    drawdown    = (running_max - history_df["bankroll_after"]) / running_max
    max_dd      = float(drawdown.max())

    # Sharpe par année
    if "year" in history_df.columns:
        pnl_by_year = history_df.groupby("year")["pnl"].sum()
        n_years = len(pnl_by_year)
        sharpe = (
            float(pnl_by_year.mean() / pnl_by_year.std() * np.sqrt(n_years))
            if pnl_by_year.std() > 0 and n_years > 1 else float("nan")
        )
        profitable_years = int((pnl_by_year > 0).sum())
    else:
        sharpe = float("nan")
        profitable_years = int(history_df["pnl"].sum() > 0)

    # ROI par tour
    if "round_number" in history_df.columns:
        grp = history_df.groupby("round_number")
        invested = grp["stake"].sum()
        pnl_r    = grp["pnl"].sum()
        roi_by_round = (pnl_r / invested.replace(0, np.nan)).to_dict()
    else:
        roi_by_round = {}

    return {
        "roi":              roi,
        "final_bankroll":   final,
        "max_drawdown":     max_dd,
        "sharpe":           sharpe,
        "profitable_years": profitable_years,
        "n_bets":           len(history_df),
        "n_wins":           int((history_df["result"] == 1).sum()),
        "roi_by_round":     roi_by_round,
    }


# ---------------------------------------------------------------------------
# Simulation complète toutes stratégies
# ---------------------------------------------------------------------------

def run_betting_backtest(
    backtest_preds_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    strategies: list[str] | None = None,
    initial_bankroll: float = 1000.0,
    min_ev_threshold: float = 0.03,
) -> dict:
    """
    Joint les prédictions aux cotes et exécute toutes les stratégies.

    Retourne un dict {strategy_name: {"history": DataFrame, "metrics": dict}}
    """
    if strategies is None:
        strategies = ["full_kelly", "half_kelly", "capped_kelly", "fixed_ev_tier", "kelly_by_round"]

    matched = join_predictions_to_odds(backtest_preds_df, odds_df)
    if matched.empty:
        logger.warning("Aucun match trouvé entre prédictions et cotes.")
        return {}

    results: dict = {}
    for strat in strategies:
        sim = BankrollSimulator(
            strategy=strat,
            initial_bankroll=initial_bankroll,
            min_ev_threshold=min_ev_threshold,
        )
        history = sim.run(matched)
        metrics = compute_metrics(history, initial_bankroll=initial_bankroll)
        results[strat] = {"history": history, "metrics": metrics}
        logger.info(
            f"  {strat}: ROI={metrics.get('roi', float('nan')):.1%} "
            f"MaxDD={metrics.get('max_drawdown', float('nan')):.1%} "
            f"Bets={metrics.get('n_bets', 0)}"
        )

    return results
