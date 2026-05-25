"""
Simulation du tableau complet Roland Garros 2026.

Usage :
    from rg2026_bracket import load_draw, simulate_bracket, get_player_path
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Name resolution: abbreviated ATP format → full Sackmann names
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """ASCII-fold + lower + strip punctuation for fuzzy matching."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def build_name_index(df: pd.DataFrame) -> dict:
    """
    Build a lookup index from all player names in a DataFrame (winner_name / loser_name).
    Returns {last_word_normalized: [full_name, ...]}
    """
    names: set[str] = set()
    for col in ("winner_name", "loser_name"):
        if col in df.columns:
            names.update(df[col].dropna().astype(str).unique())

    index: dict[str, list[str]] = {}
    for name in names:
        words = name.strip().split()
        if not words:
            continue
        key = _normalize(words[-1])          # last word of full name
        index.setdefault(key, []).append(name)

    return index


def _initials_match(initials_str: str, first_part: str) -> bool:
    """
    Check whether the initials string matches the first part of the full name.
    "J"   vs "Jannik"           → True
    "JM"  vs "Juan Manuel"      → True
    "PH"  vs "Pierre-Hugues"    → True  (P + H from hyphenated)
    "FAA" vs "Felix Auger"      → False
    """
    # Extract capital letters from the full first-part tokens
    tokens = re.split(r"[\s\-]", first_part)
    first_letters = [t[0].upper() for t in tokens if t]

    initials = [c.upper() for c in initials_str if c.isalpha()]
    if not initials:
        return True   # no initials → accept anything

    # Check that each initial appears in order among first_letters
    pos = 0
    for init in initials:
        found = False
        while pos < len(first_letters):
            if first_letters[pos] == init:
                pos += 1
                found = True
                break
            pos += 1
        if not found:
            return False
    return True


def resolve_abbrev_name(abbrev: str, name_index: dict) -> tuple[str, str]:
    """
    Resolve an abbreviated ATP name to the full Sackmann name.

    Examples:
        "J.Sinner"             → ("Jannik Sinner",        "exact")
        "JM.Cerundolo"         → ("Juan Manuel Cerundolo","exact")
        "F.Auger-Aliassime"    → ("Felix Auger Aliassime","exact")
        "A.De Minaur"          → ("Alex De Minaur",       "exact")
        "PH.Herbert"           → ("Pierre-Hugues Herbert","exact")
        "T.Macha"              → ("Tomas Machac",         "fuzzy")
        "Unknown Player"       → ("Unknown Player",       "unresolved")

    Returns (resolved_name, confidence: "exact" | "fuzzy" | "unresolved")
    """
    abbrev = abbrev.strip()

    dot_pos = abbrev.find(".")

    # Full name already (no dot, or space before the dot): try direct lookup first
    if dot_pos < 0 or (dot_pos > 0 and " " in abbrev[:dot_pos]):
        # Try exact match in index
        key = _normalize(abbrev.strip().split()[-1]) if abbrev.strip() else ""
        for candidate in name_index.get(key, []):
            if _normalize(candidate) == _normalize(abbrev):
                return candidate, "exact"
        # Fuzzy fallback on the full string
        best_score, best_name = 0.0, abbrev
        for names_list in name_index.values():
            for full in names_list:
                sim = difflib.SequenceMatcher(None, _normalize(abbrev), _normalize(full)).ratio()
                if sim > best_score:
                    best_score, best_name = sim, full
        if best_score >= 0.90:
            return best_name, "exact"
        return abbrev, "unresolved"

    initials_str = abbrev[:dot_pos]           # e.g. "J", "JM", "PH", "AD"
    surname_part = abbrev[dot_pos + 1:].strip()  # e.g. "Sinner", "De Minaur", "Auger-Aliassime"

    # Normalise surname_part: hyphens → spaces, get last word as key
    surname_norm = _normalize(surname_part)
    surname_words = surname_norm.split()
    if not surname_words:
        return abbrev, "unresolved"

    last_word = surname_words[-1]

    # --- Exact surname-last-word match ---
    candidates = name_index.get(last_word, [])

    # If empty, try every word of the surname (for compound surnames searched by first word)
    if not candidates:
        for w in surname_words[:-1]:
            cands = name_index.get(w, [])
            if cands:
                candidates = cands
                break

    # Filter: surname_part must be a suffix of the candidate's full name (normalised)
    def _surname_matches(full: str) -> bool:
        words = full.strip().split()
        # The surname_part should appear as the last N words
        n = len(surname_part.replace("-", " ").split())
        tail = " ".join(words[-n:])
        return _normalize(tail) == _normalize(surname_part.replace("-", " "))

    exact = [c for c in candidates if _surname_matches(c)]

    # Among exact surname matches, filter by initials
    def _first_part(full: str, surname_len: int) -> str:
        words = full.strip().split()
        return " ".join(words[:-surname_len]) if surname_len < len(words) else words[0]

    surname_word_count = len(surname_part.replace("-", " ").split())
    matched = [c for c in exact if _initials_match(initials_str, _first_part(c, surname_word_count))]

    if len(matched) == 1:
        return matched[0], "exact"
    if len(matched) > 1:
        # Multiple matches — prefer by initials strength
        return matched[0], "exact"

    # --- Fuzzy fallback on surname ---
    all_names: list[str] = []
    for names_list in name_index.values():
        all_names.extend(names_list)

    # Score each name: surname similarity + initials bonus
    scored: list[tuple[float, str]] = []
    for full in all_names:
        words = full.strip().split()
        tail = " ".join(words[-surname_word_count:])
        sim = difflib.SequenceMatcher(None, _normalize(surname_part), _normalize(tail)).ratio()
        if sim < 0.80:   # strict threshold to avoid false positives (e.g. Merida → Almeida)
            continue
        init_ok = _initials_match(initials_str, _first_part(full, surname_word_count))
        score = sim + (0.3 if init_ok else 0.0)
        scored.append((score, full))

    if scored:
        scored.sort(reverse=True)
        best_score, best_name = scored[0]
        if best_score >= 0.9:
            return best_name, "fuzzy"

    return abbrev, "unresolved"


def resolve_draw_names(
    draw_lines: list[str],
    df: pd.DataFrame,
) -> list[dict]:
    """
    Resolve all abbreviated names in a draw text (one name per line) to full names.

    Returns list of dicts:
        {"original": ..., "resolved": ..., "confidence": "exact"|"fuzzy"|"unresolved"}
    """
    index = build_name_index(df)
    results = []
    for line in draw_lines:
        line = line.strip()
        if not line:
            continue
        # Strip seed prefix like "[1] " or "[WC] "
        m = re.match(r"^\[([^\]]*)\]\s*(.+)$", line)
        seed_str = ""
        name_part = line
        if m:
            seed_str = m.group(1).strip()
            name_part = m.group(2).strip()

        resolved, confidence = resolve_abbrev_name(name_part, index)
        results.append({
            "original": name_part,
            "seed": seed_str,
            "resolved": resolved,
            "confidence": confidence,
        })
    return results


DRAW_PATH = Path(__file__).parent.parent / "data" / "rg2026" / "draw.csv"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "rg2026" / "results.jsonl"

ROUND_NAMES = {1: "R128", 2: "R64", 3: "R32", 4: "R16", 5: "QF", 6: "SF", 7: "F"}

# Standard positions of seeds in a 128-player draw (per ATP convention)
SEED_POSITIONS: dict[int, int] = {
    1: 1, 2: 128,
    3: 65, 4: 64,
    5: 33, 6: 96, 7: 97, 8: 32,
    9: 17, 10: 48, 11: 80, 12: 113,
    13: 112, 14: 81, 15: 49, 16: 16,
    17: 9, 18: 24, 19: 40, 20: 56, 21: 72, 22: 88, 23: 104, 24: 120,
    25: 121, 26: 105, 27: 89, 28: 73, 29: 57, 30: 41, 31: 25, 32: 8,
}


# ---------------------------------------------------------------------------
# Draw management
# ---------------------------------------------------------------------------

def generate_draw_template(ranked_players: list[dict], out_path: Path = DRAW_PATH) -> Path:
    """
    Génère un fichier draw.csv template depuis une liste de joueurs classés.

    ranked_players : list of {"player_name", "atp_rank", "ioc", "seed"(optional)}
    Retourne le chemin du fichier créé.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Assign seeds 1-32 to top 32 by rank
    players_by_seed: dict[int, dict] = {}
    unseeded = []

    for i, p in enumerate(ranked_players):
        seed = i + 1 if i < 32 else None
        entry = {
            "draw_pos": None,
            "player_name": p.get("player_name", f"TBD Rank {p.get('atp_rank', '?')}"),
            "seed": seed if seed else "",
            "atp_rank": p.get("atp_rank", ""),
            "nationality": p.get("ioc", ""),
            "status": "seeded" if seed else "ranked",
        }
        if seed:
            players_by_seed[seed] = entry
        else:
            unseeded.append(entry)

    # Fill 128 positions
    draw: list[dict] = [None] * 128  # type: ignore

    # Place seeds in their standard positions
    for seed, pos in SEED_POSITIONS.items():
        if seed in players_by_seed:
            entry = players_by_seed[seed].copy()
            entry["draw_pos"] = pos
            draw[pos - 1] = entry

    # Fill remaining positions with unseeded ranked players, then TBD
    unseeded_iter = iter(unseeded)
    for i in range(128):
        if draw[i] is None:
            try:
                entry = next(unseeded_iter).copy()
            except StopIteration:
                entry = {
                    "draw_pos": i + 1,
                    "player_name": f"TBD / Qualifier {i + 1}",
                    "seed": "",
                    "atp_rank": "",
                    "nationality": "",
                    "status": "qualifier",
                }
            entry["draw_pos"] = i + 1
            draw[i] = entry

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["draw_pos", "player_name", "seed", "atp_rank", "nationality", "status"])
        writer.writeheader()
        writer.writerows(draw)

    return out_path


def load_draw(path: Path = DRAW_PATH) -> list[dict]:
    """Charge le tableau depuis draw.csv. Retourne une liste de 128 dicts triés par draw_pos."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows.sort(key=lambda r: int(r.get("draw_pos") or 0))
    return rows


def load_confirmed_results() -> list[dict]:
    """Charge les résultats déjà joués depuis results.jsonl."""
    if not RESULTS_PATH.exists():
        return []
    results = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

def _pairs(players: list) -> list[tuple]:
    """Découpe une liste en paires (match 1v2, 3v4, …)."""
    return [(players[i], players[i + 1]) for i in range(0, len(players), 2)]


def simulate_bracket(
    draw: list[dict],
    predictor,
    confirmed_results: Optional[list[dict]] = None,
) -> dict:
    """
    Simule le tableau complet RG 2026 round par round.

    draw : list de 128 dicts avec 'draw_pos' et 'player_name'
    predictor : instance de RolandGarrosPredictor (pour predict_match)
    confirmed_results : résultats déjà connus (écrase les prédictions)

    Retourne :
    {
        "rounds": {
            1: [{"a": ..., "b": ..., "winner": ..., "proba_a": ..., "confirmed": bool}],
            2: [...],  ...  7: [...]
        },
        "champion": str,
        "probas": {player: float}  # proba de gagner le tournoi
    }
    """
    if len(draw) < 128:
        return {"error": "Draw incomplet (besoin de 128 joueurs)"}

    names = [p.get("player_name", f"TBD {i+1}") for i, p in enumerate(draw)]

    # Index confirmed results by (round, players)
    confirmed_map: dict[tuple, str] = {}
    if confirmed_results:
        for r in confirmed_results:
            rn = int(r.get("round_number", 0))
            w  = r.get("winner", "")
            l  = r.get("loser", "")
            confirmed_map[(rn, w, l)] = w
            confirmed_map[(rn, l, w)] = w

    def _get_winner(player_a: str, player_b: str, round_number: int) -> tuple[str, float, bool]:
        """Retourne (winner, proba_a, is_confirmed)."""
        # Check confirmed results
        key = (round_number, player_a, player_b)
        if key in confirmed_map:
            winner = confirmed_map[key]
            return winner, (1.0 if winner == player_a else 0.0), True

        # Skip TBD matches
        if "TBD" in player_a or "Qualifier" in player_a:
            return player_b, 0.05, False
        if "TBD" in player_b or "Qualifier" in player_b:
            return player_a, 0.95, False

        try:
            pred = predictor.predict_match(player_a, player_b, round_number=round_number)
            proba_a = pred.get("proba_a", 0.5)
        except Exception:
            proba_a = 0.5

        winner = player_a if proba_a >= 0.5 else player_b
        return winner, proba_a, False

    rounds_output = {}
    current = names[:]

    for rnd in range(1, 8):  # R128 → F
        matches = []
        next_round = []
        pairs = _pairs(current)

        for player_a, player_b in pairs:
            winner, proba_a, confirmed = _get_winner(player_a, player_b, rnd)
            matches.append({
                "a": player_a,
                "b": player_b,
                "winner": winner,
                "proba_a": round(proba_a, 4),
                "proba_b": round(1.0 - proba_a, 4),
                "confirmed": confirmed,
                "round_name": ROUND_NAMES.get(rnd, str(rnd)),
            })
            next_round.append(winner)

        rounds_output[rnd] = matches
        current = next_round

    champion = current[0] if current else "?"

    return {
        "rounds": rounds_output,
        "champion": champion,
    }


def get_player_path(
    player_name: str,
    bracket_result: dict,
    draw: list[dict],
) -> list[dict]:
    """
    Retourne le chemin prédit du joueur dans le tableau (liste d'adversaires par tour).
    """
    rounds = bracket_result.get("rounds", {})
    path = []

    for rnd in sorted(rounds.keys()):
        for match in rounds[rnd]:
            if player_name in (match["a"], match["b"]):
                opponent = match["b"] if match["a"] == player_name else match["a"]
                is_winner = (match["winner"] == player_name)
                path.append({
                    "round": rnd,
                    "round_name": match["round_name"],
                    "opponent": opponent,
                    "player_proba": match["proba_a"] if match["a"] == player_name else match["proba_b"],
                    "predicted_winner": match["winner"],
                    "player_wins": is_winner,
                    "confirmed": match["confirmed"],
                })
                break

    return path


def bracket_summary_df(bracket_result: dict) -> pd.DataFrame:
    """Retourne un DataFrame de tous les matchs du tableau pour affichage."""
    rows = []
    for rnd, matches in sorted(bracket_result.get("rounds", {}).items()):
        for m in matches:
            rows.append({
                "Tour": m["round_name"],
                "Joueur A": m["a"],
                "Prob A": f"{m['proba_a']:.1%}",
                "Joueur B": m["b"],
                "Prob B": f"{m['proba_b']:.1%}",
                "Vainqueur prédit": m["winner"],
                "Confirmé": "✓" if m["confirmed"] else "",
            })
    return pd.DataFrame(rows)
