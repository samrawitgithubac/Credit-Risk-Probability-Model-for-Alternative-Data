"""
Convenience runner for Task 3 (Feature Engineering).

Usage (from repo root):
  python scripts/run_task3.py
  python scripts/run_task3.py --raw data/raw/data.csv --out data/processed/processed.csv --head 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task 3 feature engineering and preview the output.")
    parser.add_argument("--raw", default="data/raw/data.csv", help="Path to raw CSV")
    parser.add_argument("--out", default="data/processed/processed.csv", help="Path to write processed CSV")
    parser.add_argument("--head", type=int, default=5, help="How many rows to preview")
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from src.data_processing import write_processed_dataset  # noqa: WPS433 (runtime import)

    raw_path = root / args.raw
    out_path = root / args.out

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {raw_path}\n"
            "Expected something like data/raw/data.csv. Put the Kaggle CSV there first."
        )

    written = write_processed_dataset(raw_path, out_path)
    print(f"\nWrote processed dataset to: {written}")

    df = pd.read_csv(written)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print("\nPreview:")
    print(df.head(args.head).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


