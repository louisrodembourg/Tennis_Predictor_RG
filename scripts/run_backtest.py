#!/usr/bin/env python3
"""
Script standalone de backtesting Roland Garros 2017-2025.
Usage : python scripts/run_backtest.py [--data-dir data/raw] [--year-start 2000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Backtesting RG 2017-2025")
    parser.add_argument("--data-dir", default="data/raw", help="Répertoire des CSV Sackmann")
    parser.add_argument("--year-start", type=int, default=2000, help="Première année à charger")
    parser.add_argument("--save-csv", action="store_true", help="Sauvegarder les résultats en CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir

    if not data_dir.exists():
        print(f"ERREUR : répertoire {data_dir} introuvable.")
        print("Lancez d'abord : bash scripts/download_data.sh")
        sys.exit(1)

    from backtesting import print_summary, run_full_backtest

    bt = run_full_backtest(str(data_dir), year_start=args.year_start)
    print_summary(bt["backtest_results"], bt["baseline_comparison"])

    if args.save_csv:
        out_dir = Path(__file__).parent.parent / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in bt["backtest_results"]:
            rows.append({
                "year": r.year,
                "accuracy": r.accuracy,
                "brier": r.brier,
                "log_loss": r.log_loss_val,
                "n_matches": r.n_matches,
            })
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / "backtest_results.csv", index=False)
        bt["baseline_comparison"].to_csv(out_dir / "baseline_comparison.csv", index=False)
        print(f"\nRésultats sauvegardés dans {out_dir}/")


if __name__ == "__main__":
    main()
