"""
Chargement et nettoyage des données ATP (Jeff Sackmann).
"""

import os
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


COLS_TO_KEEP = [
    "tourney_id", "tourney_name", "surface", "tourney_date", "tourney_level",
    "match_num", "round",
    "winner_id", "winner_name", "winner_rank", "winner_rank_points", "winner_age",
    "loser_id", "loser_name", "loser_rank", "loser_rank_points", "loser_age",
    "score", "best_of",
    "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpFaced", "w_bpSaved",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpFaced", "l_bpSaved",
    "minutes",
]

SURFACE_MAP = {
    "Clay": "Clay",
    "Hard": "Hard",
    "Grass": "Grass",
    "Carpet": "Carpet",
}


def _normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    return str(name).strip().title()


def _make_match_id(row: pd.Series) -> str:
    key = f"{row['tourney_id']}_{row['match_num']}_{row['winner_name']}_{row['loser_name']}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_matches(data_dir: str, year_start: int = 2000, year_end: int = 2025) -> pd.DataFrame:
    """
    Charge et consolide les fichiers atp_matches_YYYY.csv de Sackmann.
    Retourne un DataFrame trié chronologiquement, sans walkovers.
    """
    data_path = Path(data_dir)
    pattern = str(data_path / "atp_matches_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier atp_matches_*.csv trouvé dans {data_dir}.\n"
            "Lancez d'abord : bash scripts/download_data.sh"
        )

    files_in_range = [f for f in files if year_start <= int(Path(f).stem.split("_")[-1]) <= year_end]
    dfs = []
    for f in tqdm(files_in_range, desc="Chargement CSV", unit="fichier", ncols=80):
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = int(Path(f).stem.split("_")[-1])
            dfs.append(df)
        except Exception as e:
            tqdm.write(f"  [WARN] Impossible de charger {f}: {e}")

    if not dfs:
        raise ValueError(f"Aucune donnée chargée pour {year_start}-{year_end}.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Données brutes : {len(df):,} matchs ({year_start}-{year_end})")

    # Garder uniquement les colonnes disponibles
    cols = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[cols + ["year"]].copy()

    # Exclure walkovers
    mask_wo = df["score"].astype(str).str.contains("W/O|walkover|DEF|RET", case=False, na=False)
    df = df[~mask_wo].reset_index(drop=True)

    # Normaliser les noms
    df["winner_name"] = df["winner_name"].apply(_normalize_name)
    df["loser_name"] = df["loser_name"].apply(_normalize_name)

    # Convertir tourney_date (format YYYYMMDD → datetime)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"])

    # Normaliser surface
    df["surface"] = df["surface"].map(SURFACE_MAP).fillna("Unknown")

    # Encoder le tour en entier (1=R128, 7=F ; qualifications = -2/-1/0)
    round_order = {
        "Q1": -2, "Q2": -1, "Q3": 0,           # qualifications
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7, "RR": 3,
    }
    df["round_number"] = df["round"].map(round_order).fillna(3).astype(int)

    # Normaliser tourney_level en string (certains CSV anciens ont des entiers)
    df["tourney_level"] = df["tourney_level"].astype(str).str.strip().replace("nan", "")

    # Encoder le niveau de tournoi (pour K factor)
    level_k = {"G": 40, "M": 32, "A": 24, "D": 20, "C": 20, "F": 30}
    df["k_factor"] = df["tourney_level"].map(level_k).fillna(24).astype(int)

    # Créer match_id
    df["match_num"] = df.get("match_num", pd.Series(range(len(df)))).fillna(0).astype(int)
    df["match_id"] = df.apply(_make_match_id, axis=1)

    # Trier chronologiquement
    df = df.sort_values(["tourney_date", "tourney_id", "round_number"]).reset_index(drop=True)

    print(f"Après nettoyage : {len(df):,} matchs")
    return df


def filter_clay(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les matchs sur terre battue."""
    return df[df["surface"] == "Clay"].copy()


def filter_roland_garros(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les matchs de Roland Garros."""
    mask = df["tourney_name"].str.contains("Roland Garros|Roland-Garros|French Open", case=False, na=False)
    return df[mask].copy()


def get_player_list(df: pd.DataFrame) -> list[str]:
    """Liste de tous les joueurs uniques (triés alphabétiquement)."""
    players = set(df["winner_name"].tolist()) | set(df["loser_name"].tolist())
    return sorted(players - {""})
