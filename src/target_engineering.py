"""
Task 4 - Proxy Target Variable Engineering (RFM + KMeans)

We do not have a ground-truth "default" label. This module creates a *proxy*
label `is_high_risk` by identifying the least-engaged customers using RFM:

- Recency: days since last transaction (higher => worse / more disengaged)
- Frequency: number of transactions (higher => better engaged)
- Monetary: total spend proxy (sum of Value) (higher => better engaged)

We then cluster customers into 3 groups with KMeans on scaled RFM features and
label the "least engaged" cluster as high-risk (is_high_risk = 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RFM_REQUIRED_COLUMNS = ["CustomerId", "TransactionStartTime", "TransactionId", "Value"]


@dataclass(frozen=True)
class RFMConfig:
    n_clusters: int = 3
    random_state: int = 42
    snapshot_date: Optional[str] = None
    # If None, snapshot_date will be set to (max(TransactionStartTime) + 1 day)


def compute_rfm(df: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute RFM metrics per CustomerId.

    Returns a dataframe:
      CustomerId, Recency, Frequency, Monetary
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("compute_rfm expects a pandas DataFrame.")

    missing = [c for c in RFM_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()
    d["TransactionStartTime"] = pd.to_datetime(d["TransactionStartTime"], errors="coerce")

    # Drop rows without a valid timestamp, because Recency depends on time
    d = d.dropna(subset=["TransactionStartTime"])
    if d.empty:
        raise ValueError("No valid TransactionStartTime values after parsing; cannot compute RFM.")

    if snapshot_date is None:
        snapshot_date = d["TransactionStartTime"].max().normalize() + pd.Timedelta(days=1)

    gb = d.groupby("CustomerId", dropna=False)
    last_txn = gb["TransactionStartTime"].max()
    recency = (snapshot_date - last_txn).dt.days.astype(int)

    frequency = gb["TransactionId"].count().astype(int)
    monetary = gb["Value"].sum().astype(float)

    rfm = pd.DataFrame(
        {
            "CustomerId": recency.index,
            "Recency": recency.values,
            "Frequency": frequency.reindex(recency.index).values,
            "Monetary": monetary.reindex(recency.index).values,
        }
    )
    return rfm


def _choose_high_risk_cluster(rfm_with_clusters: pd.DataFrame) -> int:
    """
    Decide which cluster is "least engaged" (proxy high-risk).

    We compute a risk score:
      risk = z(Recency) - z(Frequency) - z(Monetary)
    The cluster with the highest mean risk score is labeled high-risk.
    """
    needed = {"cluster", "Recency", "Frequency", "Monetary"}
    if not needed.issubset(rfm_with_clusters.columns):
        raise ValueError(f"Expected columns {sorted(needed)} in clustered RFM table.")

    # Standardize for fair combination
    X = rfm_with_clusters[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X)
    risk_score = Z[:, 0] - Z[:, 1] - Z[:, 2]
    tmp = rfm_with_clusters[["cluster"]].copy()
    tmp["risk_score"] = risk_score

    cluster_scores = tmp.groupby("cluster")["risk_score"].mean()
    return int(cluster_scores.idxmax())


def add_proxy_target(
    df_raw: pd.DataFrame,
    config: RFMConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Create proxy target `is_high_risk` and return:
      - labeled_customers: CustomerId + RFM + cluster + is_high_risk
      - cluster_summary: per-cluster mean RFM + counts + is_high_risk_cluster flag
      - high_risk_cluster_id
    """
    config = config or RFMConfig()
    snapshot = pd.to_datetime(config.snapshot_date) if config.snapshot_date else None

    rfm = compute_rfm(df_raw, snapshot_date=snapshot)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float))

    km = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init="auto")
    clusters = km.fit_predict(X)

    labeled = rfm.copy()
    labeled["cluster"] = clusters.astype(int)

    high_risk_cluster = _choose_high_risk_cluster(labeled)
    labeled["is_high_risk"] = (labeled["cluster"] == high_risk_cluster).astype(int)

    summary = (
        labeled.groupby("cluster")
        .agg(
            customers=("CustomerId", "count"),
            Recency_mean=("Recency", "mean"),
            Frequency_mean=("Frequency", "mean"),
            Monetary_mean=("Monetary", "mean"),
        )
        .reset_index()
    )
    summary["is_high_risk_cluster"] = (summary["cluster"] == high_risk_cluster).astype(int)

    return labeled, summary, high_risk_cluster


def merge_target_into_features(
    features_df: pd.DataFrame,
    labeled_customers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge `is_high_risk` into an existing customer-level feature table.
    """
    if "CustomerId" not in features_df.columns:
        raise ValueError("features_df must contain CustomerId.")
    if not {"CustomerId", "is_high_risk"}.issubset(labeled_customers_df.columns):
        raise ValueError("labeled_customers_df must contain CustomerId and is_high_risk.")

    out = features_df.merge(
        labeled_customers_df[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left",
        validate="one_to_one",
    )
    # Customers without valid timestamps would have been dropped from RFM; mark them as unknown/low-risk proxy (0)
    out["is_high_risk"] = out["is_high_risk"].fillna(0).astype(int)
    return out


def write_processed_with_target(
    raw_csv_path: str | Path,
    processed_features_csv_path: str | Path,
    out_csv_path: str | Path,
    config: RFMConfig | None = None,
) -> Path:
    """
    Read raw + processed feature table and write a new CSV including `is_high_risk`.
    """
    raw_csv_path = Path(raw_csv_path)
    processed_features_csv_path = Path(processed_features_csv_path)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_csv_path)
    df_features = pd.read_csv(processed_features_csv_path)

    labeled, _, _ = add_proxy_target(df_raw, config=config)
    merged = merge_target_into_features(df_features, labeled)
    merged.to_csv(out_csv_path, index=False)
    return out_csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create proxy target is_high_risk using RFM + KMeans.")
    parser.add_argument("--raw", required=True, help="Path to raw CSV (e.g., data/raw/data.csv)")
    parser.add_argument(
        "--features",
        default="data/processed/processed.csv",
        help="Path to processed feature CSV from Task 3",
    )
    parser.add_argument(
        "--out",
        default="data/processed/processed_with_target.csv",
        help="Output CSV path (default: data/processed/processed_with_target.csv)",
    )
    parser.add_argument("--snapshot-date", default=None, help="Optional snapshot date (e.g., 2019-01-01)")
    args = parser.parse_args()

    cfg = RFMConfig(snapshot_date=args.snapshot_date)
    out = write_processed_with_target(args.raw, args.features, args.out, config=cfg)
    print(f"Wrote processed dataset with target to: {out}")


