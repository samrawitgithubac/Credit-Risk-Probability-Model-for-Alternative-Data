"""
Task 5 - Model Training & Tracking (MLflow)

This script trains and evaluates at least two models on the processed dataset
produced by Task 3 + Task 4 (customer-level features + proxy target).

Input:  data/processed/processed_with_target.csv
Target: is_high_risk (1 = high-risk, 0 = low-risk)

It logs:
- parameters
- metrics (accuracy, precision, recall, f1, roc_auc)
- artifacts (confusion matrix png)

Runs are stored in the local `mlruns/` folder by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET_COL = "is_high_risk"
ID_COL = "CustomerId"


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = "data/processed/processed_with_target.csv"
    experiment_name: str = "credit-risk-task5"
    test_size: float = 0.2
    random_state: int = 42
    scoring: str = "roc_auc"


def load_training_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at: {path}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in training data.")
    return df


def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Exclude ID column from modeling features
    feature_df = X.drop(columns=[ID_COL], errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_binary_classifier(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _plot_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def train_and_log_models(cfg: TrainConfig) -> Dict[str, Any]:
    df = load_training_data(cfg.data_path)
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    # Model 1: Logistic Regression (interpretable baseline)
    lr_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", penalty="l2")),
        ]
    )
    lr_grid = {
        "model__C": [0.1, 1.0, 10.0],
    }

    # Model 2: Random Forest (non-linear baseline)
    rf_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1, class_weight="balanced")),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_split": [2, 10],
    }

    candidates = [
        ("logreg", lr_pipe, lr_grid),
        ("random_forest", rf_pipe, rf_grid),
    ]

    mlflow.set_experiment(cfg.experiment_name)

    best = {"model_name": None, "roc_auc": -np.inf, "run_id": None, "model_uri": None}

    for name, pipe, grid in candidates:
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, grid, scoring=cfg.scoring, cv=3, n_jobs=-1)
            search.fit(X_train, y_train)

            best_est = search.best_estimator_
            y_proba = best_est.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            metrics = evaluate_binary_classifier(y_test.to_numpy(), y_proba, y_pred)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)

            cm = confusion_matrix(y_test, y_pred)
            fig = _plot_confusion_matrix(cm, title=f"Confusion Matrix ({name})")
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                cm_path = Path(td) / f"confusion_matrix_{name}.png"
                fig.savefig(cm_path, dpi=150)
                mlflow.log_artifact(str(cm_path))
            plt.close(fig)

            # Log the model itself
            mlflow.sklearn.log_model(best_est, artifact_path="model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

            if metrics["roc_auc"] > best["roc_auc"]:
                best = {
                    "model_name": name,
                    "roc_auc": metrics["roc_auc"],
                    "run_id": mlflow.active_run().info.run_id,
                    "model_uri": model_uri,
                    "metrics": metrics,
                    "classification_report": classification_report(y_test, y_pred, zero_division=0),
                }

    # Attempt to register the best model (works when registry backend is available)
    try:
        if best["model_uri"]:
            registered = mlflow.register_model(best["model_uri"], "credit_risk_model")
            best["registered_model"] = {"name": registered.name, "version": registered.version}
    except Exception as e:  # noqa: BLE001
        best["registered_model_error"] = str(e)

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 5: Train models and log to MLflow.")
    parser.add_argument("--data", default=TrainConfig.data_path, help="Path to processed_with_target.csv")
    parser.add_argument("--experiment", default=TrainConfig.experiment_name, help="MLflow experiment name")
    parser.add_argument("--test-size", type=float, default=TrainConfig.test_size, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=TrainConfig.random_state, help="Random seed")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data,
        experiment_name=args.experiment,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    best = train_and_log_models(cfg)
    print("\nBest model:")
    print(best["model_name"], "roc_auc=", best["roc_auc"])
    print("\nMetrics:")
    print(best.get("metrics"))
    print("\nClassification report:")
    print(best.get("classification_report"))


