from contextlib import asynccontextmanager

from fastapi import FastAPI

from driftsentinel.api.routes.health import router as health_router
from driftsentinel.api.routes.prediction import router as prediction_router
from driftsentinel.models.inference import FraudInferenceService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the inference service once during application startup."""

    app.state.inference_service = FraudInferenceService()
    yield
    app.state.inference_service = None


app = FastAPI(
    title="DriftSentinel API",
    description="Temporal fraud intelligence and adaptive decision platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(prediction_router)