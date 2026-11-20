import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from driftsentinel.api.schemas.monitoring import (
    MonitoringSnapshotListResponse,
    MonitoringSnapshotRequest,
    MonitoringSnapshotResponse,
)
from driftsentinel.api.schemas.prediction import TransactionPrediction
from driftsentinel.models.exceptions import InferenceInputError
from driftsentinel.monitoring.prediction_storage import PredictionStorageError
from driftsentinel.monitoring.storage import (
    MonitoringSnapshotStore,
    MonitoringStorageError,
)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.post(
    "/snapshots",
    response_model=MonitoringSnapshotResponse,
)
def create_monitoring_snapshot(
    payload: MonitoringSnapshotRequest,
    request: Request,
):
    """Generate predictions, evaluate drift, and persist a snapshot."""

    inference_service = request.app.state.inference_service
    monitoring_service = request.app.state.monitoring_service
    prediction_store = request.app.state.prediction_store
    batch_id = None

    try:
        MonitoringSnapshotStore._validate_window(
            payload.window_start_utc,
            payload.window_end_utc,
        )

        transactions = pd.DataFrame(payload.transactions)

        predictions = inference_service.predict(transactions)

        prediction_records = [
            TransactionPrediction(**record)
            for record in predictions.to_dict(orient="records")
        ]

        batch_id = prediction_store.save_batch(
            [prediction.model_dump() for prediction in prediction_records]
        )

        feature_names = inference_service.feature_names
        features = transactions.loc[:, feature_names].copy()

        snapshot = monitoring_service.record_batch(
            features=features,
            predictions=predictions,
            window_start_utc=payload.window_start_utc,
            window_end_utc=payload.window_end_utc,
            metadata=payload.metadata,
            prediction_batch_id=batch_id,
        )

        return MonitoringSnapshotResponse(snapshot=snapshot)

    except InferenceInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except (ValueError, TypeError) as exc:
        if batch_id is not None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Predictions were saved, but monitoring snapshot "
                        "processing failed."
                    ),
                    "prediction_batch_id": batch_id,
                    "reason": str(exc),
                },
            ) from exc

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PredictionStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Predictions were generated but could not be persisted.",
        ) from exc

    except MonitoringStorageError as exc:
        detail = "Monitoring snapshot storage is unavailable."

        if batch_id is not None:
            detail = {
                "message": (
                    "Predictions were saved, but the monitoring snapshot "
                    "could not be persisted."
                ),
                "prediction_batch_id": batch_id,
            }

        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc


@router.get(
    "/snapshots",
    response_model=MonitoringSnapshotListResponse,
)
def list_monitoring_snapshots(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
):
    """List the most recent monitoring snapshots."""

    monitoring_store = request.app.state.monitoring_store

    try:
        snapshots = monitoring_store.list_snapshots(limit=limit)
        return MonitoringSnapshotListResponse(snapshots=snapshots)

    except MonitoringStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Monitoring snapshot storage is unavailable.",
        ) from exc


@router.get(
    "/snapshots/{snapshot_id}",
    response_model=MonitoringSnapshotResponse,
)
def get_monitoring_snapshot(
    snapshot_id: str,
    request: Request,
):
    """Retrieve one monitoring snapshot by its ID."""

    monitoring_store = request.app.state.monitoring_store

    try:
        snapshot = monitoring_store.get_snapshot(snapshot_id)

        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail="Monitoring snapshot not found.",
            )

        return MonitoringSnapshotResponse(snapshot=snapshot)

    except MonitoringStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Monitoring snapshot storage is unavailable.",
        ) from exc
