import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MonitoringStorageError(Exception):
    """Raised when monitoring snapshot storage fails."""


class MonitoringSnapshotStore:
    """Persist aggregate monitoring snapshots in a SQLite database."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        """Create the database directory and monitoring table."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        created_at_utc TEXT NOT NULL,
                        window_start_utc TEXT,
                        window_end_utc TEXT,
                        prediction_count INTEGER NOT NULL
                            CHECK (prediction_count >= 0),
                        prediction_summary_json TEXT NOT NULL,
                        drift_summary_json TEXT,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )

                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(monitoring_snapshots)"
                    ).fetchall()
                }

                if "prediction_batch_id" not in columns:
                    connection.execute(
                        """
                        ALTER TABLE monitoring_snapshots
                        ADD COLUMN prediction_batch_id TEXT
                        """
                    )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_monitoring_snapshots_created_at
                    ON monitoring_snapshots(created_at_utc)
                    """
                )

        except sqlite3.Error as exc:
            raise MonitoringStorageError(
                "Could not initialize monitoring database."
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Manage a connection transaction and reliably close it."""

        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _serialize_json(value: Any, field_name: str) -> str:
        """Serialize JSON-compatible data and reject non-finite values."""

        try:
            return json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must contain valid JSON-compatible data."
            ) from exc

    @staticmethod
    def _validate_window(
        window_start_utc: str | None,
        window_end_utc: str | None,
    ) -> None:
        """Validate optional ISO-formatted event-time boundaries."""

        parsed_values = []

        for value, field_name in (
            (window_start_utc, "window_start_utc"),
            (window_end_utc, "window_end_utc"),
        ):
            if value is None:
                parsed_values.append(None)
                continue

            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or None.")

            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be a valid ISO timestamp."
                ) from exc

            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(
                    f"{field_name} must include a timezone."
                )

            parsed_values.append(parsed)

        start, end = parsed_values

        if start is not None and end is not None and start > end:
            raise ValueError(
                "window_start_utc cannot be later than window_end_utc."
            )

    def save_snapshot(
        self,
        prediction_summary: dict[str, Any],
        drift_summary: dict[str, Any] | None = None,
        window_start_utc: str | None = None,
        window_end_utc: str | None = None,
        metadata: dict[str, Any] | None = None,
        prediction_batch_id: str | None = None,
    ) -> str:
        """Save one aggregate monitoring snapshot and return its ID."""

        if not isinstance(prediction_summary, dict):
            raise TypeError("prediction_summary must be a dictionary.")

        if drift_summary is not None and not isinstance(
            drift_summary,
            dict,
        ):
            raise TypeError("drift_summary must be a dictionary or None.")

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        if prediction_batch_id is not None:
            if not isinstance(prediction_batch_id, str) or not prediction_batch_id.strip():
                raise ValueError(
                    "prediction_batch_id must be a non-empty string or None."
                )

        prediction_count = prediction_summary.get("prediction_count")

        if (
            isinstance(prediction_count, bool)
            or not isinstance(prediction_count, int)
            or prediction_count < 0
        ):
            raise ValueError(
                "prediction_summary must contain a non-negative integer "
                "'prediction_count'."
            )

        self._validate_window(
            window_start_utc,
            window_end_utc,
        )

        prediction_json = self._serialize_json(
            prediction_summary,
            "prediction_summary",
        )

        drift_json = (
            self._serialize_json(drift_summary, "drift_summary")
            if drift_summary is not None
            else None
        )

        metadata_json = self._serialize_json(
            metadata,
            "metadata",
        )

        snapshot_id = str(uuid.uuid4())
        created_at_utc = datetime.now(UTC).isoformat()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO monitoring_snapshots (
                        snapshot_id,
                        created_at_utc,
                        window_start_utc,
                        window_end_utc,
                        prediction_count,
                        prediction_summary_json,
                        drift_summary_json,
                        metadata_json,
                        prediction_batch_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        created_at_utc,
                        window_start_utc,
                        window_end_utc,
                        prediction_count,
                        prediction_json,
                        drift_json,
                        metadata_json,
                        prediction_batch_id,
                    ),
                )

        except sqlite3.Error as exc:
            raise MonitoringStorageError(
                "Could not save monitoring snapshot."
            ) from exc

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        """Retrieve one snapshot, returning None when it does not exist."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT *
                    FROM monitoring_snapshots
                    WHERE snapshot_id = ?
                    """,
                    (snapshot_id,),
                ).fetchone()

        except sqlite3.Error as exc:
            raise MonitoringStorageError(
                "Could not retrieve monitoring snapshot."
            ) from exc

        if row is None:
            return None

        return self._deserialize_row(row)

    def list_snapshots(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the most recent snapshots."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM monitoring_snapshots
                    ORDER BY created_at_utc DESC, snapshot_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        except sqlite3.Error as exc:
            raise MonitoringStorageError(
                "Could not list monitoring snapshots."
            ) from exc

        return [
            self._deserialize_row(row)
            for row in rows
        ]

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row into a Python snapshot."""

        return {
            "snapshot_id": row["snapshot_id"],
            "created_at_utc": row["created_at_utc"],
            "window_start_utc": row["window_start_utc"],
            "window_end_utc": row["window_end_utc"],
            "prediction_count": row["prediction_count"],
            "prediction_summary": json.loads(
                row["prediction_summary_json"]
            ),
            "drift_summary": (
                json.loads(row["drift_summary_json"])
                if row["drift_summary_json"] is not None
                else None
            ),
            "metadata": json.loads(row["metadata_json"]),
            "prediction_batch_id": row["prediction_batch_id"],
        }
