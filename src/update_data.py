"""
Mise à jour des fichiers de données ATP depuis le dépôt Jeff Sackmann.
Usage : python src/update_data.py
        python src/update_data.py --years 2026
        python src/update_data.py --years 2025 2026 --clear-cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

FILE_PATTERNS = [
    "atp_matches_{year}.csv",
    "atp_matches_qual_chall_{year}.csv",
    "atp_matches_futures_{year}.csv",
]


def download_year(year: int, force: bool = False) -> list[str]:
    """Télécharge tous les fichiers pour une année. Retourne la liste des fichiers mis à jour."""
    updated = []
    for pattern in FILE_PATTERNS:
        filename = pattern.format(year=year)
        dest = RAW_DIR / filename
        url  = f"{BASE_URL}/{filename}"

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                content = resp.read()

            if dest.exists() and not force:
                old = dest.read_bytes()
                if old == content:
                    print(f"  {filename}  (inchangé)")
                    continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            rows = content.count(b"\n")
            print(f"  {filename}  → {rows:,} lignes  ✓")
            updated.append(filename)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                pass   # file doesn't exist yet for this year — normal
            else:
                print(f"  {filename}  [WARN] HTTP {e.code}")
        except Exception as e:
            print(f"  {filename}  [WARN] {e}")

    return updated


def clear_cache() -> None:
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        for f in CACHE_DIR.glob("*.txt"):
            f.unlink()
        for f in CACHE_DIR.glob("*.joblib"):
            f.unlink()
        print("Cache invalidé.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Met à jour les données ATP depuis Sackmann/GitHub")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Années à mettre à jour (défaut : 2025 et 2026)")
    parser.add_argument("--force", action="store_true",
                        help="Réécrit même si le fichier n'a pas changé")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Supprime le cache parquet après la mise à jour")
    parser.add_argument("--all", action="store_true",
                        help="Télécharge toutes les années depuis 2000")
    args = parser.parse_args()

    if args.all:
        import datetime
        years = list(range(2000, datetime.date.today().year + 1))
    elif args.years:
        years = args.years
    else:
        import datetime
        cur = datetime.date.today().year
        years = [cur - 1, cur]

    print(f"Mise à jour des données ATP ({', '.join(str(y) for y in years)})...")
    all_updated = []
    for y in years:
        print(f"\nAnnée {y}")
        all_updated.extend(download_year(y, force=args.force))

    if all_updated:
        print(f"\n{len(all_updated)} fichier(s) mis à jour.")
        if args.clear_cache:
            clear_cache()
        else:
            print("Pensez à vider le cache (--clear-cache) pour forcer la reconstruction des features.")
    else:
        print("\nAucun fichier modifié.")


if __name__ == "__main__":
    main()
