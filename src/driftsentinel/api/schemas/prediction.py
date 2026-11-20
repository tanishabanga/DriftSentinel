from typing import Any

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 1000
MAX_HISTORY_LIMIT = 100


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
    """Response containing predictions and their persisted batch ID."""

    predictions: list[TransactionPrediction]
    batch_id: str | None = None


class PredictionHistoryListResponse(BaseModel):
    """Response containing recent individual prediction records."""

    predictions: list[dict[str, Any]]


class PredictionBatchResponse(BaseModel):
    """Response containing one persisted prediction batch."""

    batch: dict[str, Any]

class PredictionBatchListResponse(BaseModel):
    """Response containing recent batch summaries."""

    batches: list[dict[str, Any]]


class PredictionBatchTopRisksResponse(BaseModel):
    """Response containing the highest-risk predictions in one batch."""

    batch: dict[str, Any]


class PredictionBatchPageResponse(BaseModel):
    """Response containing one page of batch prediction records."""

    batch: dict[str, Any]
