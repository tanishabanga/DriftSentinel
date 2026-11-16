import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from driftsentinel.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    TransactionPrediction,
)
from driftsentinel.models.exceptions import InferenceInputError

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_transactions(
    payload: PredictionRequest,
    request: Request,
):
    """Generate fraud predictions for submitted transactions."""

    inference_service = request.app.state.inference_service

    try:
        transactions = pd.DataFrame(payload.transactions)

        predictions = inference_service.predict(transactions)

        results = [
            TransactionPrediction(**record)
            for record in predictions.to_dict(orient="records")
        ]

        return PredictionResponse(predictions=results)

    except InferenceInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
