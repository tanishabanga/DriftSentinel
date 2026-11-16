
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """Check whether the API process is running."""

    return {
        "status": "healthy",
        "service": "DriftSentinel API"
    }
