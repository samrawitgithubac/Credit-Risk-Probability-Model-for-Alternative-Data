from __future__ import annotations

import os
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictRequest, PredictResponse


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def load_model() -> tuple[Any, str]:
    """
    Load best model from MLflow.

    Defaults:
    - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    - MODEL_URI=models:/credit_risk_model/1
      (you can also use models:/credit_risk_model/latest)
    """
    tracking_uri = _get_env("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = _get_env("MODEL_URI", "models:/credit_risk_model/1")
    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri


app = FastAPI(title="Credit Risk Model API", version="0.1.0")

_MODEL = None
_MODEL_URI = None
_MODEL_LOAD_ERROR = None


@app.on_event("startup")
def _startup() -> None:
    global _MODEL, _MODEL_URI, _MODEL_LOAD_ERROR  # noqa: PLW0603
    try:
        _MODEL, _MODEL_URI = load_model()
        _MODEL_LOAD_ERROR = None
    except Exception as e:  # noqa: BLE001
        _MODEL = None
        _MODEL_URI = None
        _MODEL_LOAD_ERROR = str(e)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _MODEL is not None else "error",
        "model_loaded": _MODEL is not None,
        "model_uri": _MODEL_URI,
        "error": _MODEL_LOAD_ERROR,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_MODEL_LOAD_ERROR}")

    if req.features is not None:
        records: List[Dict[str, Any]] = [req.features]
    else:
        records = req.records or []

    df = pd.DataFrame(records)

    # Preferred: probability of class 1 (high-risk)
    if hasattr(_MODEL, "predict_proba"):
        probs = _MODEL.predict_proba(df)[:, 1].astype(float).tolist()
    else:
        # Fallback: treat predicted label as probability
        preds = _MODEL.predict(df)
        probs = [float(x) for x in preds]

    return PredictResponse(model_uri=_MODEL_URI or "", n_records=len(records), risk_probabilities=probs)


