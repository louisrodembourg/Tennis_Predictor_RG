"""
Cache persistant pour les objets coûteux à calculer.

Stratégie :
  - Clé = hash des mtime de tous les atp_matches_*.csv
  - Si la clé correspond, on charge depuis data/cache/ (parquet + joblib)
  - Sinon, on recalcule tout et on sauvegarde

Ce qui est caché :
  df_processed      : DataFrame nettoyé (toutes surfaces)
  df_with_elo       : df + colonnes Elo pré-match
  player_state      : table d'état vectorisée par joueur
  rg_state          : idem pour Roland Garros
  h2h_state         : idem pour H2H clay
  all_features      : features clay (input du modèle)
  model             : XGBoost calibré final
"""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# Clé de cache
# ---------------------------------------------------------------------------

def _csv_hash(data_dir: str) -> str:
    """Hash stable basé sur les mtime de tous les CSV ATP + la version des features."""
    from features import FEATURE_VERSION
    files = sorted(glob.glob(str(Path(data_dir) / "atp_matches_*.csv")))
    h = hashlib.md5()
    h.update(FEATURE_VERSION.encode())
    for f in files:
        h.update(f.encode())
        h.update(str(Path(f).stat().st_mtime).encode())
    return h.hexdigest()[:16]


def _key_path() -> Path:
    return CACHE_DIR / "cache_key.txt"


def _current_key(data_dir: str) -> str:
    return _csv_hash(data_dir)


def cache_is_valid(data_dir: str) -> bool:
    kp = _key_path()
    if not kp.exists():
        return False
    if kp.read_text().strip() != _current_key(data_dir):
        return False
    required = [
        "df_processed.parquet",
        "df_with_elo.parquet",
        "player_state.parquet",
        "rg_state.parquet",
        "h2h_state.parquet",
        "all_features.parquet",
        "model.joblib",
    ]
    return all((CACHE_DIR / f).exists() for f in required)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_cache(data_dir: str) -> Optional[dict]:
    """Retourne un dict avec tous les objets si le cache est valide, sinon None."""
    if not cache_is_valid(data_dir):
        return None
    print("Cache trouvé — chargement depuis data/cache/ ...")
    try:
        return {
            "df":            pd.read_parquet(CACHE_DIR / "df_processed.parquet"),
            "df_with_elo":   pd.read_parquet(CACHE_DIR / "df_with_elo.parquet"),
            "player_state":  pd.read_parquet(CACHE_DIR / "player_state.parquet"),
            "rg_state":      pd.read_parquet(CACHE_DIR / "rg_state.parquet"),
            "h2h_state":     pd.read_parquet(CACHE_DIR / "h2h_state.parquet"),
            "all_features":  pd.read_parquet(CACHE_DIR / "all_features.parquet"),
            "model":         joblib.load(CACHE_DIR / "model.joblib"),
        }
    except Exception as e:
        print(f"  [WARN] Échec chargement cache : {e} — recalcul complet.")
        return None


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Homogénéise les colonnes object avec types mixtes pour éviter ArrowTypeError."""
    df = df.copy()
    for col in df.columns[df.dtypes == object]:
        if df[col].notna().any() and not df[col].apply(type).eq(str).all():
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def save_cache(data_dir: str, cache: dict) -> None:
    """Sauvegarde tous les objets et met à jour la clé."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Sauvegarde du cache dans data/cache/ ...")
    _sanitize_df(cache["df"]).to_parquet(CACHE_DIR / "df_processed.parquet", index=False)
    cache["df_with_elo"].to_parquet(CACHE_DIR / "df_with_elo.parquet", index=False)
    cache["player_state"].to_parquet(CACHE_DIR / "player_state.parquet", index=False)
    cache["rg_state"].to_parquet(CACHE_DIR / "rg_state.parquet", index=False)
    cache["h2h_state"].to_parquet(CACHE_DIR / "h2h_state.parquet", index=False)
    cache["all_features"].to_parquet(CACHE_DIR / "all_features.parquet", index=False)
    joblib.dump(cache["model"], CACHE_DIR / "model.joblib", compress=3)
    _key_path().write_text(_current_key(data_dir))
    print("  Cache sauvegardé.")


def invalidate_cache() -> None:
    """Force un recalcul complet au prochain démarrage."""
    kp = _key_path()
    if kp.exists():
        kp.unlink()
