from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/health")
def health_check(request: Request):
    """Check API, model, and calibrator readiness."""

    inference_service = getattr(
        request.app.state,
        "inference_service",
        None,
    )

    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "service": "DriftSentinel API",
                "model_loaded": False,
                "calibrator_loaded": False,
            },
        )

    model_loaded = getattr(
        inference_service,
        "model",
        None,
    ) is not None

    calibrator_loaded = getattr(
        inference_service,
        "calibrator",
        None,
    ) is not None

    if not model_loaded or not calibrator_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "service": "DriftSentinel API",
                "model_loaded": model_loaded,
                "calibrator_loaded": calibrator_loaded,
            },
        )

    return {
        "status": "healthy",
        "service": "DriftSentinel API",
        "model_loaded": True,
        "calibrator_loaded": True,
    }
