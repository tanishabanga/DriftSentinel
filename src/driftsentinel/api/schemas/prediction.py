from typing import Any

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 1000


class PredictionRequest(BaseModel):
    """Request containing a bounded batch of transactions."""

    transactions: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


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
