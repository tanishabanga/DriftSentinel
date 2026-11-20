import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from driftsentinel.api.routes.health import router as health_router
from driftsentinel.api.routes.monitoring import router as monitoring_router
from driftsentinel.api.routes.prediction import router as prediction_router
from driftsentinel.models.inference import FraudInferenceService
from driftsentinel.monitoring.prediction_storage import PredictionLedgerStore
from driftsentinel.monitoring.service import MonitoringService
from driftsentinel.monitoring.storage import MonitoringSnapshotStore

DEFAULT_DATABASE_PATH = Path("artifacts/monitoring/monitoring.db")
DEFAULT_REFERENCE_PROFILE_PATH = Path(
    "artifacts/monitoring/reference_profile.json"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application services during startup."""

    app.state.inference_service = FraudInferenceService()

    database_path = Path(
        os.getenv(
            "DRIFTSENTINEL_DB_PATH",
            str(DEFAULT_DATABASE_PATH),
        )
    )

    monitoring_store = MonitoringSnapshotStore(database_path)
    monitoring_store.initialize()

    prediction_store = PredictionLedgerStore(database_path)
    prediction_store.initialize()

    reference_profile_path = Path(
        os.getenv(
            "DRIFTSENTINEL_REFERENCE_PROFILE_PATH",
            str(DEFAULT_REFERENCE_PROFILE_PATH),
        )
    )

    if not reference_profile_path.is_file():
        raise FileNotFoundError(
            f"Monitoring reference profile not found: "
            f"{reference_profile_path}"
        )

    reference_profile_json = await asyncio.to_thread(
        reference_profile_path.read_text,
        encoding="utf-8",
    )
    reference_profile = json.loads(reference_profile_json)

    monitoring_service = MonitoringService(
        reference_profile=reference_profile,
        store=monitoring_store,
    )

    app.state.monitoring_store = monitoring_store
    app.state.prediction_store = prediction_store
    app.state.monitoring_service = monitoring_service

    yield

    app.state.inference_service = None
    app.state.monitoring_store = None
    app.state.prediction_store = None
    app.state.monitoring_service = None


app = FastAPI(
    title="DriftSentinel API",
    description="Temporal fraud intelligence and adaptive decision platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(monitoring_router)