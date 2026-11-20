import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from driftsentinel.api.schemas.prediction import (
    MAX_HISTORY_LIMIT,
    PredictionBatchListResponse,
    PredictionBatchPageResponse,
    PredictionBatchTopRisksResponse,
    PredictionBatchResponse,
    PredictionHistoryListResponse,
    PredictionRequest,
    PredictionResponse,
    TransactionPrediction,
)
from driftsentinel.models.exceptions import InferenceInputError
from driftsentinel.monitoring.prediction_storage import PredictionStorageError

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_transactions(
    payload: PredictionRequest,
    request: Request,
):
    """Generate predictions and persist their minimal output records."""

    inference_service = request.app.state.inference_service
    prediction_store = request.app.state.prediction_store

    try:
        transactions = pd.DataFrame(payload.transactions)

        predictions = inference_service.predict(transactions)

        results = [
            TransactionPrediction(**record)
            for record in predictions.to_dict(orient="records")
        ]

        batch_id = prediction_store.save_batch(
            [prediction.model_dump() for prediction in results]
        )

        return PredictionResponse(
            predictions=results,
            batch_id=batch_id,
        )

    except InferenceInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Predictions were generated but could not be persisted.",
        ) from exc


@router.get(
    "/predictions",
    response_model=PredictionHistoryListResponse,
)
def list_prediction_history(
    request: Request,
    limit: int = Query(default=100, ge=1, le=MAX_HISTORY_LIMIT),
):
    """Return the most recent persisted individual predictions."""

    prediction_store = request.app.state.prediction_store

    try:
        predictions = prediction_store.list_predictions(limit=limit)

        return PredictionHistoryListResponse(
            predictions=predictions,
        )

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction history storage is unavailable.",
        ) from exc


@router.get(
    "/predictions/batches",
    response_model=PredictionBatchListResponse,
)
def list_prediction_batches(
    request: Request,
    limit: int = Query(default=100, ge=1, le=100),
):
    """Return recent prediction batches with aggregate decision counts."""

    prediction_store = request.app.state.prediction_store

    try:
        batches = prediction_store.list_batches(limit=limit)
        return PredictionBatchListResponse(batches=batches)

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction history storage is unavailable.",
        ) from exc


@router.get(
    "/predictions/batches/{batch_id}/page",
    response_model=PredictionBatchPageResponse,
)
def get_prediction_batch_page(
    batch_id: str,
    request: Request,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=200),
    action: str | None = Query(
        default=None,
        pattern="^(approve|review|block)$",
    ),
):
    """Return one filtered page of records from a prediction batch."""

    prediction_store = request.app.state.prediction_store

    try:
        batch = prediction_store.get_batch_page(
            batch_id=batch_id,
            limit=limit,
            offset=offset,
            search=search,
            action=action,
        )

        if batch is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction batch not found.",
            )

        return PredictionBatchPageResponse(batch=batch)

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction history storage is unavailable.",
        ) from exc


@router.get(
    "/predictions/batches/{batch_id}/top-risk",
    response_model=PredictionBatchTopRisksResponse,
)
def get_prediction_batch_top_risks(
    batch_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=20),
    action: str | None = Query(
        default=None,
        pattern="^(approve|review|block)$",
    ),
):
    """Return the highest calibrated probabilities in a prediction batch."""

    prediction_store = request.app.state.prediction_store

    try:
        batch = prediction_store.get_batch_top_risks(
            batch_id=batch_id,
            limit=limit,
            action=action,
        )

        if batch is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction batch not found.",
            )

        return PredictionBatchTopRisksResponse(batch=batch)

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction history storage is unavailable.",
        ) from exc


@router.get(
    "/predictions/batches/{batch_id}",
    response_model=PredictionBatchResponse,
)
def get_prediction_batch(
    batch_id: str,
    request: Request,
):
    """Retrieve all persisted predictions belonging to one batch."""

    prediction_store = request.app.state.prediction_store

    try:
        batch = prediction_store.get_batch(batch_id)

        if batch is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction batch not found.",
            )

        return PredictionBatchResponse(batch=batch)

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction history storage is unavailable.",
        ) from exc