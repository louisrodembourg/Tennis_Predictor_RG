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


def save_backtest_results(
    results: list,
    baseline_df: Optional[pd.DataFrame],
    config: dict,
    output_path: Optional[Path | str] = None,
) -> Path:
    """
    Ajoute (append) un bloc de résultats dans le fichier historique.

    Paramètres
    ----------
    results      : liste de BacktestResult (year, accuracy, brier, log_loss_val, n_matches)
    baseline_df  : DataFrame de comparaison baselines (peut être None)
    config       : dict décrivant la configuration du run
    output_path  : chemin du fichier de sortie (défaut : results/backtest_history.txt)

    Retourne le chemin absolu du fichier écrit.
    """
    path = Path(output_path) if output_path else DEFAULT_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        _write_header(f, config)
        _write_results(f, results)
        if baseline_df is not None and not baseline_df.empty:
            _write_baselines(f, baseline_df)
        f.write("\n")

    return path


# ---------------------------------------------------------------------------
# Helpers privés
# ---------------------------------------------------------------------------

def _write_header(f, config: dict) -> None:
    sep = "=" * 72
    f.write(f"\n{sep}\n")
    f.write(f"  RUN  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{sep}\n")
    for key, val in config.items():
        if isinstance(val, list):
            val = f"[{len(val)} features]"
        f.write(f"  {key:<28} {val}\n")
    f.write("\n")


def _write_results(f, results: list) -> None:
    header = f"{'Année':<8} {'Accuracy':>10} {'Brier':>10} {'LogLoss':>10} {'N matchs':>10}"
    f.write(header + "\n")
    f.write("-" * 52 + "\n")

    accs, briers = [], []
    for r in results:
        f.write(
            f"{r.year:<8} {r.accuracy:>10.3f} {r.brier:>10.4f}"
            f" {r.log_loss_val:>10.4f} {r.n_matches:>10}\n"
        )
        accs.append(r.accuracy)
        briers.append(r.brier)

    if accs:
        f.write("-" * 52 + "\n")
        f.write(
            f"{'Moyenne':<8} {np.mean(accs):>10.3f} {np.mean(briers):>10.4f}\n"
        )


def _write_baselines(f, baseline_df: pd.DataFrame) -> None:
    f.write("\n  COMPARAISON BASELINES\n")
    f.write("-" * 72 + "\n")
    cols_order = [
        "year", "n",
        "baseline_rank_acc", "baseline_clay_elo_acc",
        "baseline_welo_adj_acc", "xgboost_acc",
        "xgboost_brier",
    ]
    avail = [c for c in cols_order if c in baseline_df.columns]
    disp = baseline_df[avail].copy()
    float_cols = [c for c in avail if c not in ("year", "n")]
    for c in float_cols:
        disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    f.write(disp.to_string(index=False) + "\n")
