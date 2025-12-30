from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """
    Flexible request schema: you can send either:
      - a single record in `features`
      - multiple records in `records`
    """

    features: Optional[Dict[str, Any]] = Field(default=None, description="Single customer feature dict")
    records: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of customer feature dicts")

    @model_validator(mode="after")
    def _validate_one_of(self):
        if self.features is None and self.records is None:
            raise ValueError("Provide either `features` or `records`.")
        if self.features is not None and self.records is not None:
            raise ValueError("Provide only one of `features` or `records`, not both.")
        if self.features is not None and len(self.features) == 0:
            raise ValueError("`features` cannot be empty.")
        if self.records is not None and len(self.records) == 0:
            raise ValueError("`records` cannot be empty.")
        return self


class PredictResponse(BaseModel):
    model_uri: str
    n_records: int
    risk_probabilities: List[float]


