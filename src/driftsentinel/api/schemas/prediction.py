
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request containing one or more transactions."""

    transactions: list[dict[str, Any]] = Field(min_length=1)


class TransactionPrediction(BaseModel):
    """Prediction output for a single transaction."""

    TransactionID: int | float | str | None = None
    raw_probability: float
    calibrated_probability: float
    action: str
    expected_cost: float


class PredictionResponse(BaseModel):
    """Response containing predictions."""

    predictions: list[TransactionPrediction]
