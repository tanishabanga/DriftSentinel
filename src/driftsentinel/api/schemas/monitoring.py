from typing import Any

from pydantic import BaseModel, Field

from driftsentinel.api.schemas.prediction import MAX_BATCH_SIZE


class MonitoringSnapshotRequest(BaseModel):
    """Request to generate and persist a monitoring snapshot."""

    transactions: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )
    window_start_utc: str | None = None
    window_end_utc: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitoringSnapshotListResponse(BaseModel):
    """Response containing recent monitoring snapshots."""

    snapshots: list[dict[str, Any]]


class MonitoringSnapshotResponse(BaseModel):
    """Response containing one monitoring snapshot."""

    snapshot: dict[str, Any]
