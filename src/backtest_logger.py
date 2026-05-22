"""
Persistance des résultats de backtesting.
Chaque run est ajouté au fichier results/backtest_history.txt avec une
ligne de titre décrivant la configuration (date, features, hyperparamètres).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path(__file__).parent.parent / "results" / "backtest_history.txt"

SEP  = "=" * 80
SEP2 = "-" * 80


def save_backtest_results(
    results: list,
    baseline_df: Optional[pd.DataFrame],
    config: dict,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Ajoute (append) un bloc complet de résultats dans le fichier historique."""
    path = Path(output_path) if output_path else DEFAULT_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        _write_header(f, config)
        _write_results_by_year(f, results)
        _write_results_by_round(f, results)
        if baseline_df is not None and not baseline_df.empty:
            _write_baselines(f, baseline_df)
        f.write("\n")

    return path


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _write_header(f, config: dict) -> None:
    f.write(f"\n{SEP}\n")
    f.write(f"  RUN  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{SEP}\n")
    for key, val in config.items():
        if isinstance(val, list):
            val = f"[{len(val)} features]"
        f.write(f"  {key:<28} {val}\n")
    f.write("\n")


def _write_results_by_year(f, results: list) -> None:
    f.write("  RÉSULTATS PAR ÉDITION\n")
    f.write(SEP2 + "\n")

    hdr = (
        f"{'Année':<8} {'Acc Base':>9} {'Brier Base':>11} {'LL Base':>9}"
        f" {'Acc Blend':>10} {'Brier Blend':>12} {'LL Blend':>9} {'N matchs':>9}"
    )
    f.write(hdr + "\n")
    f.write("-" * 82 + "\n")

    accs, briers, lls = [], [], []
    accs_bl, briers_bl, lls_bl = [], [], []

    for r in results:
        acc_bl   = getattr(r, "accuracy_blend",  float("nan"))
        brier_bl = getattr(r, "brier_blend",     float("nan"))
        ll_bl    = getattr(r, "log_loss_blend",  float("nan"))
        ll_base  = getattr(r, "log_loss_val",    float("nan"))

        f.write(
            f"{r.year:<8} {r.accuracy:>9.3f} {r.brier:>11.4f} {ll_base:>9.4f}"
            f" {acc_bl:>10.3f} {brier_bl:>12.4f} {ll_bl:>9.4f} {r.n_matches:>9}\n"
        )
        accs.append(r.accuracy);  briers.append(r.brier);  lls.append(ll_base)
        if not np.isnan(acc_bl):
            accs_bl.append(acc_bl); briers_bl.append(brier_bl); lls_bl.append(ll_bl)

    if accs:
        f.write("-" * 82 + "\n")
        _m = lambda vals, fmt: f"{np.mean(vals):{fmt}}" if vals else "—"
        f.write(
            f"{'Moyenne':<8} {_m(accs,'>9.3f')} {_m(briers,'>11.4f')} {_m(lls,'>9.4f')}"
            f" {_m(accs_bl,'>10.3f')} {_m(briers_bl,'>12.4f')} {_m(lls_bl,'>9.4f')}\n"
        )
    f.write("\n")


def _write_results_by_round(f, results: list) -> None:
    """Décomposition par phase du tournoi (moyenne sur toutes les éditions)."""
    all_by_round = []
    for r in results:
        if not hasattr(r, "by_round") or r.by_round is None or r.by_round.empty:
            continue
        df_r = r.by_round.copy()
        df_r["year"] = r.year
        all_by_round.append(df_r)

    if not all_by_round:
        return

    combined = pd.concat(all_by_round, ignore_index=True)

    agg_cols = {c: "mean" for c in ["acc_base", "brier_base", "acc_blend", "brier_blend"]
                if c in combined.columns}
    agg_cols["n"] = "sum"
    pivot = combined.groupby("round_name").agg(agg_cols).reset_index()

    round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "Q1", "Q2", "Q3"]
    pivot["_ord"] = pivot["round_name"].map({r: i for i, r in enumerate(round_order)})
    pivot = pivot.sort_values("_ord").drop(columns="_ord")

    f.write("  DÉCOMPOSITION PAR PHASE DU TOURNOI (moyenne 2017-2025)\n")
    f.write(SEP2 + "\n")

    has_blend = "acc_blend" in pivot.columns

    if has_blend:
        hdr = f"{'Tour':<8} {'Acc Base':>9} {'Brier Base':>11} {'Acc Blend':>10} {'Brier Blend':>12} {'N matchs':>9}"
    else:
        hdr = f"{'Tour':<8} {'Acc Base':>9} {'Brier Base':>11} {'N matchs':>9}"
    f.write(hdr + "\n")
    f.write("-" * (65 if has_blend else 42) + "\n")

    for _, row in pivot.iterrows():
        if has_blend:
            f.write(
                f"{row['round_name']:<8} {row['acc_base']:>9.3f} {row['brier_base']:>11.4f}"
                f" {row['acc_blend']:>10.3f} {row['brier_blend']:>12.4f} {int(row['n']):>9}\n"
            )
        else:
            f.write(
                f"{row['round_name']:<8} {row['acc_base']:>9.3f} {row['brier_base']:>11.4f}"
                f" {int(row['n']):>9}\n"
            )
    f.write("\n")


def _write_baselines(f, baseline_df: pd.DataFrame) -> None:
    f.write("  COMPARAISON BASELINES\n")
    f.write(SEP2 + "\n")

    # Colonnes accuracy + brier pour chaque méthode
    acc_cols = [
        ("baseline_rank_acc",     "baseline_rank_brier",     "Ranking"),
        ("baseline_clay_elo_acc", "baseline_clay_elo_brier", "Clay Elo"),
        ("baseline_welo_adj_acc", "baseline_welo_adj_brier", "WElo Adj."),
        ("xgboost_acc",           "xgboost_brier",           "XGBoost"),
        ("blend_acc",             "blend_brier",             "Blend"),
    ]
    present = [(a, b, lbl) for a, b, lbl in acc_cols
               if a in baseline_df.columns]

    # Header
    hdr = f"{'Année':>6} {'N':>5}"
    for _, _, lbl in present:
        hdr += f"  {lbl+' Acc':>11} {lbl+' Brier':>12}"
    f.write(hdr + "\n")
    f.write("-" * len(hdr) + "\n")

    for _, row in baseline_df.iterrows():
        line = f"{int(row['year']):>6} {int(row['n']):>5}"
        for a_col, b_col, _ in present:
            acc_v   = row.get(a_col, float("nan"))
            brier_v = row.get(b_col, float("nan"))
            acc_s   = f"{acc_v:>11.3f}"   if pd.notna(acc_v)   else f"{'N/A':>11}"
            brier_s = f"{brier_v:>12.4f}" if pd.notna(brier_v) else f"{'N/A':>12}"
            line += f"  {acc_s} {brier_s}"
        f.write(line + "\n")

    # Moyennes
    f.write("-" * len(hdr) + "\n")
    line = f"{'Moy':>6} {'':>5}"
    for a_col, b_col, _ in present:
        col_a = pd.to_numeric(baseline_df.get(a_col), errors="coerce").dropna()
        col_b = pd.to_numeric(baseline_df.get(b_col), errors="coerce").dropna()
        acc_s   = f"{col_a.mean():>11.3f}" if len(col_a) else f"{'—':>11}"
        brier_s = f"{col_b.mean():>12.4f}" if len(col_b) else f"{'—':>12}"
        line += f"  {acc_s} {brier_s}"
    f.write(line + "\n")
