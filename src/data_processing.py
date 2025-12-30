"""
Task 3 - Feature Engineering

This module builds a reproducible, model-ready dataset from the raw Xente
transaction data using sklearn Pipelines.

Design choice:
- We build a *customer-level* feature table (1 row per CustomerId) because the
  project goal is to score "a new customer".
- Transaction-level categorical fields are aggregated to "most frequent" (mode)
  per customer, then One-Hot Encoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


RAW_REQUIRED_COLUMNS = [
    "CustomerId",
    "TransactionId",
    "Amount",
    "Value",
    "TransactionStartTime",
]


def _safe_mode(series: pd.Series) -> Optional[object]:
    """Return the most frequent value (mode); None if empty or all null."""
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if s.empty:
        return None
    modes = s.mode(dropna=True)
    if modes.empty:
        return None
    return modes.iloc[0]


class CustomerFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build customer-level features from raw transaction-level dataframe.

    Output: pandas.DataFrame with one row per CustomerId.
    """

    def __init__(self, customer_id_col: str = "CustomerId"):
        self.customer_id_col = customer_id_col

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CustomerFeatureBuilder expects a pandas DataFrame as input.")

        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = X.copy()

        # Parse datetime safely (invalid parses become NaT)
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionYear"] = df["TransactionStartTime"].dt.year

        # Choose categorical columns (object dtype) EXCLUDING IDs that are too high-cardinality
        # TransactionId is excluded; CustomerId is the group key.
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        excluded = {self.customer_id_col, "TransactionId", "BatchId", "AccountId", "SubscriptionId"}
        cat_cols = [c for c in object_cols if c not in excluded]

        # Core numeric aggregates requested by Task 3
        gb = df.groupby(self.customer_id_col, dropna=False)

        agg_dict = {
            "Amount": ["sum", "mean", "std"],
            "TransactionId": ["count"],
            "TransactionHour": ["mean"],
            "TransactionDay": ["mean"],
            "TransactionMonth": [_safe_mode],
            "TransactionYear": [_safe_mode],
        }

        # Add mode aggregations for categorical fields (per customer)
        for c in cat_cols:
            agg_dict[c] = [_safe_mode]

        feats = gb.agg(agg_dict)

        # Flatten multi-index columns
        feats.columns = [f"{col}_{fn}" for col, fn in feats.columns.to_flat_index()]

        # Rename to match spec more explicitly
        rename_map = {
            "Amount_sum": "TotalTransactionAmount",
            "Amount_mean": "AverageTransactionAmount",
            "Amount_std": "StdTransactionAmount",
            "TransactionId_count": "TransactionCount",
            "TransactionMonth__safe_mode": "TransactionMonth",
            "TransactionYear__safe_mode": "TransactionYear",
        }
        feats = feats.rename(columns=rename_map)

        # Make std deterministic when there is only one transaction (pandas returns NaN)
        if "StdTransactionAmount" in feats.columns:
            feats["StdTransactionAmount"] = feats["StdTransactionAmount"].fillna(0.0)

        # Ensure index is a real column for downstream joins
        feats = feats.reset_index()
        return feats


@dataclass(frozen=True)
class FeaturePipelineSpec:
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numeric: bool = True


def build_feature_pipeline(spec: FeaturePipelineSpec | None = None) -> Pipeline:
    """
    Returns a pipeline that:
    1) builds customer-level features
    2) imputes missing values
    3) one-hot encodes categorical features
    4) scales numeric features

    Output is a numpy array suitable for modeling.
    """
    spec = spec or FeaturePipelineSpec()

    feature_builder = CustomerFeatureBuilder(customer_id_col="CustomerId")

    # We don't know the final columns until after aggregation, so we rely on pandas dtypes
    # using a ColumnTransformer with "remainder=drop".
    numeric_pipe_steps = [
        ("imputer", SimpleImputer(strategy=spec.numeric_impute_strategy)),
    ]
    if spec.scale_numeric:
        numeric_pipe_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_pipe_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=spec.categorical_impute_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Column selectors using pandas dtype
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, lambda df: df.select_dtypes(include=[np.number]).columns),
            ("cat", categorical_pipe, lambda df: df.select_dtypes(include=["object"]).columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_builder", feature_builder),
            ("preprocess", preprocess),
        ]
    )


def build_feature_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: returns the aggregated (non-encoded) customer feature table."""
    return CustomerFeatureBuilder(customer_id_col="CustomerId").transform(df_raw)


def write_processed_dataset(
    raw_csv_path: str | Path,
    out_csv_path: str | Path,
) -> Path:
    """
    Create the customer-level feature table and write to CSV.

    This writes the *aggregated* feature table (human-readable), not the one-hot/scaled matrix.
    """
    raw_csv_path = Path(raw_csv_path)
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_csv_path)
    features = build_feature_table(df_raw)
    features.to_csv(out_csv_path, index=False)
    return out_csv_path


if __name__ == "__main__":
    # Basic CLI usage (no extra dependency like click/typer)
    # Example:
    #   python -m src.data_processing --raw data/raw/data.csv --out data/processed/processed.csv
    import argparse

    parser = argparse.ArgumentParser(description="Build processed dataset for credit-risk model.")
    parser.add_argument("--raw", required=True, help="Path to raw CSV (e.g., data/raw/data.csv)")
    parser.add_argument(
        "--out",
        default="data/processed/processed.csv",
        help="Output CSV path (default: data/processed/processed.csv)",
    )
    args = parser.parse_args()

    out_path = write_processed_dataset(args.raw, args.out)
    print(f"Wrote processed dataset to: {out_path}")


