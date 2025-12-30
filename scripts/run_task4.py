"""
Convenience runner for Task 4 (Proxy Target Engineering).

This will:
1) Load raw data (transactions)
2) Compute RFM per customer
3) Cluster customers into 3 groups (KMeans)
4) Label the least-engaged cluster as is_high_risk=1
5) Merge into the Task 3 processed feature table
6) Write data/processed/processed_with_target.csv

Usage (from repo root):
  python scripts/run_task4.py
  python scripts/run_task4.py --snapshot-date 2019-01-01 --head 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Task 4 proxy target engineering and preview results.")
    parser.add_argument("--raw", default="data/raw/data.csv", help="Path to raw CSV")
    parser.add_argument("--features", default="data/processed/processed.csv", help="Task 3 output CSV path")
    parser.add_argument("--out", default="data/processed/processed_with_target.csv", help="Output CSV path")
    parser.add_argument("--snapshot-date", default=None, help="Optional snapshot date (e.g., 2019-01-01)")
    parser.add_argument("--head", type=int, default=5, help="How many rows to preview")
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from src.target_engineering import RFMConfig, add_proxy_target, merge_target_into_features  # noqa: WPS433

    raw_path = root / args.raw
    feat_path = root / args.features
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path}")
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Task 3 features not found at: {feat_path}\n"
            "Run: python scripts/run_task3.py"
        )

    df_raw = pd.read_csv(raw_path)
    df_feat = pd.read_csv(feat_path)

    labeled, summary, high_risk_cluster = add_proxy_target(df_raw, config=RFMConfig(snapshot_date=args.snapshot_date))

    print("\nCluster summary (means):")
    print(summary.to_string(index=False))
    print(f"\nHigh-risk cluster chosen: {high_risk_cluster}")

    merged = merge_target_into_features(df_feat, labeled)
    merged.to_csv(out_path, index=False)

    print(f"\nWrote dataset with target to: {out_path}")
    print(f"Shape: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    print("\nPreview:")
    print(merged.head(args.head).to_string(index=False))
    print("\nis_high_risk distribution:")
    print(merged["is_high_risk"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


