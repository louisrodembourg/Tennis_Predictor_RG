#!/bin/bash
# Télécharge les données ATP depuis le repo Jeff Sackmann

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_RAW="$PROJECT_ROOT/data/raw"
TMP_DIR="/tmp/tennis_atp_$$"

echo "Clonage du repo Jeff Sackmann..."
git clone --depth=1 https://github.com/JeffSackmann/tennis_atp.git "$TMP_DIR"

echo "Copie des fichiers dans $DATA_RAW..."
mkdir -p "$DATA_RAW"
cp "$TMP_DIR"/atp_matches_20*.csv "$DATA_RAW/" 2>/dev/null || true
cp "$TMP_DIR"/atp_matches_1[0-9][0-9][0-9].csv "$DATA_RAW/" 2>/dev/null || true

# Données récentes (si disponibles)
cp "$TMP_DIR"/atp_matches_futures_*.csv "$DATA_RAW/" 2>/dev/null || echo "  (pas de futures)"
cp "$TMP_DIR"/atp_matches_qual_chall_*.csv "$DATA_RAW/" 2>/dev/null || echo "  (pas de qualif/chall)"

echo "Nettoyage..."
rm -rf "$TMP_DIR"

N=$(ls "$DATA_RAW"/atp_matches_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Données téléchargées : $N fichiers dans $DATA_RAW"
echo "OK — Vous pouvez maintenant lancer :"
echo "  python scripts/run_backtest.py"
echo "  streamlit run dashboard/app.py"
