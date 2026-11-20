import json
import math
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PredictionStorageError(Exception):
    """Raised when prediction ledger storage fails."""


class PredictionLedgerStore:
    """Persist individual prediction outputs without raw transaction features."""

    ALLOWED_ACTIONS = {"approve", "review", "block"}

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        """Create the database directory, ledger table, and indexes."""

        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_records (
                        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        predicted_at_utc TEXT NOT NULL,
                        transaction_id_json TEXT,
                        raw_probability REAL NOT NULL
                            CHECK (raw_probability >= 0 AND raw_probability <= 1),
                        calibrated_probability REAL NOT NULL
                            CHECK (
                                calibrated_probability >= 0
                                AND calibrated_probability <= 1
                            ),
                        action TEXT NOT NULL
                            CHECK (action IN ('approve', 'review', 'block')),
                        expected_cost REAL NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_prediction_records_created
                    ON prediction_records(predicted_at_utc DESC, prediction_id DESC)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_prediction_records_batch
                    ON prediction_records(batch_id, prediction_id)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_prediction_records_action
                    ON prediction_records(action)
                    """
                )

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not initialize prediction ledger."
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Manage a SQLite transaction and reliably close the connection."""

        connection = sqlite3.connect(self.database_path, timeout=10.0)
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
    def _validate_probability(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a finite number.")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} must be a finite number."
            ) from exc

        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(
                f"{field_name} must be finite and between 0 and 1."
            )

        return number

    @staticmethod
    def _validate_expected_cost(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("expected_cost must be a finite number.")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "expected_cost must be a finite number."
            ) from exc

        if not math.isfinite(number):
            raise ValueError("expected_cost must be a finite number.")

        return number

    @staticmethod
    def _serialize_transaction_id(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(
                "TransactionID must be a string, integer, float, or None."
            )

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("TransactionID must be finite.")

        try:
            return json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("TransactionID is not JSON-compatible.") from exc

    @classmethod
    def _validate_prediction(cls, prediction: dict[str, Any]) -> tuple:
        if not isinstance(prediction, dict):
            raise TypeError("Each prediction must be a dictionary.")

        raw_probability = cls._validate_probability(
            prediction.get("raw_probability"),
            "raw_probability",
        )
        calibrated_probability = cls._validate_probability(
            prediction.get("calibrated_probability"),
            "calibrated_probability",
        )
        expected_cost = cls._validate_expected_cost(
            prediction.get("expected_cost")
        )

        action = prediction.get("action")
        if not isinstance(action, str) or action not in cls.ALLOWED_ACTIONS:
            raise ValueError(
                "action must be one of: approve, review, block."
            )

        transaction_id = cls._serialize_transaction_id(
            prediction.get("TransactionID")
        )

        return (
            transaction_id,
            raw_probability,
            calibrated_probability,
            action,
            expected_cost,
        )

    def save_batch(self, predictions: list[dict[str, Any]]) -> str:
        """Validate and atomically persist one batch of prediction outputs."""

        if not isinstance(predictions, list) or not predictions:
            raise ValueError("predictions must be a non-empty list.")

        # Validate the batch structure before validating individual fields.
        if any(not isinstance(prediction, dict) for prediction in predictions):
            raise TypeError("Each prediction must be a dictionary.")

        # Validate the entire batch before opening a write transaction.
        validated_predictions = [
            self._validate_prediction(prediction)
            for prediction in predictions
        ]

        batch_id = str(uuid.uuid4())
        predicted_at_utc = datetime.now(UTC).isoformat()

        rows = [
            (
                batch_id,
                predicted_at_utc,
                transaction_id,
                raw_probability,
                calibrated_probability,
                action,
                expected_cost,
            )
            for (
                transaction_id,
                raw_probability,
                calibrated_probability,
                action,
                expected_cost,
            ) in validated_predictions
        ]

        try:
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO prediction_records (
                        batch_id,
                        predicted_at_utc,
                        transaction_id_json,
                        raw_probability,
                        calibrated_probability,
                        action,
                        expected_cost
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not save prediction batch."
            ) from exc

        return batch_id

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> dict[str, Any]:
        transaction_id = (
            json.loads(row["transaction_id_json"])
            if row["transaction_id_json"] is not None
            else None
        )

        return {
            "prediction_id": row["prediction_id"],
            "batch_id": row["batch_id"],
            "predicted_at_utc": row["predicted_at_utc"],
            "TransactionID": transaction_id,
            "raw_probability": row["raw_probability"],
            "calibrated_probability": row["calibrated_probability"],
            "action": row["action"],
            "expected_cost": row["expected_cost"],
        }

    def list_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent individual predictions."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM prediction_records
                    ORDER BY predicted_at_utc DESC, prediction_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not list prediction records."
            ) from exc

        return [self._deserialize_row(row) for row in rows]

    def list_batches(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent batches with full-batch decision aggregates."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100.")

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        batch_id,
                        MIN(predicted_at_utc) AS predicted_at_utc,
                        COUNT(*) AS prediction_count,
                        SUM(CASE WHEN action = 'approve' THEN 1 ELSE 0 END)
                            AS approve_count,
                        SUM(CASE WHEN action = 'review' THEN 1 ELSE 0 END)
                            AS review_count,
                        SUM(CASE WHEN action = 'block' THEN 1 ELSE 0 END)
                            AS block_count,
                        SUM(
                            CASE WHEN calibrated_probability < 0.4
                            THEN 1 ELSE 0 END
                        ) AS lower_probability_count,
                        SUM(
                            CASE WHEN calibrated_probability >= 0.4
                                AND calibrated_probability < 0.8
                            THEN 1 ELSE 0 END
                        ) AS elevated_probability_count,
                        SUM(
                            CASE WHEN calibrated_probability >= 0.8
                            THEN 1 ELSE 0 END
                        ) AS highest_probability_count,
                        MAX(prediction_id) AS latest_prediction_id
                    FROM prediction_records
                    GROUP BY batch_id
                    ORDER BY predicted_at_utc DESC, latest_prediction_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not list prediction batches."
            ) from exc

        return [
            {
                "batch_id": row["batch_id"],
                "predicted_at_utc": row["predicted_at_utc"],
                "prediction_count": row["prediction_count"],
                "approve_count": row["approve_count"],
                "review_count": row["review_count"],
                "block_count": row["block_count"],
                "lower_probability_count": row["lower_probability_count"],
                "elevated_probability_count": row["elevated_probability_count"],
                "highest_probability_count": row["highest_probability_count"],
            }
            for row in rows
        ]

    def get_batch_page(
        self,
        batch_id: str,
        limit: int = 25,
        offset: int = 0,
        search: str | None = None,
        action: str | None = None,
    ) -> dict[str, Any] | None:
        """Return one filtered page of records from a prediction batch."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100.")

        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")

        if search is not None and not isinstance(search, str):
            raise ValueError("search must be a string.")

        if action is not None and action not in self.ALLOWED_ACTIONS:
            raise ValueError("action must be approve, review, or block.")

        conditions = ["batch_id = ?"]
        parameters: list[Any] = [batch_id]

        if action is not None:
            conditions.append("action = ?")
            parameters.append(action)

        if search:
            escaped_search = (
                search.replace("!", "!!")
                .replace("%", "!%")
                .replace("_", "!_")
            )
            pattern = f"%{escaped_search}%"
            conditions.append(
                "(CAST(prediction_id AS TEXT) LIKE ? "
                "ESCAPE '!' OR COALESCE(transaction_id_json, '') LIKE ? "
                "ESCAPE '!')"
            )
            parameters.extend([pattern, pattern])

        where_clause = " AND ".join(conditions)

        try:
            with self._connect() as connection:
                batch = connection.execute(
                    """
                    SELECT
                        MIN(predicted_at_utc) AS predicted_at_utc,
                        COUNT(*) AS prediction_count
                    FROM prediction_records
                    WHERE batch_id = ?
                    """,
                    (batch_id,),
                ).fetchone()

                if batch["prediction_count"] == 0:
                    return None

                total = connection.execute(
                    f"""
                    SELECT COUNT(*) AS total
                    FROM prediction_records
                    WHERE {where_clause}
                    """,
                    parameters,
                ).fetchone()["total"]

                rows = connection.execute(
                    f"""
                    SELECT *
                    FROM prediction_records
                    WHERE {where_clause}
                    ORDER BY prediction_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [*parameters, limit, offset],
                ).fetchall()

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not retrieve prediction batch page."
            ) from exc

        return {
            "batch_id": batch_id,
            "predicted_at_utc": batch["predicted_at_utc"],
            "prediction_count": batch["prediction_count"],
            "total": total,
            "limit": limit,
            "offset": offset,
            "predictions": [self._deserialize_row(row) for row in rows],
        }

    def get_batch_top_risks(
        self,
        batch_id: str,
        limit: int = 20,
        action: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the highest calibrated probabilities in one batch."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer between 1 and 20.")

        if action is not None and action not in self.ALLOWED_ACTIONS:
            raise ValueError("action must be approve, review, or block.")

        try:
            with self._connect() as connection:
                batch = connection.execute(
                    """
                    SELECT MIN(predicted_at_utc) AS predicted_at_utc,
                           COUNT(*) AS prediction_count
                    FROM prediction_records
                    WHERE batch_id = ?
                    """,
                    (batch_id,),
                ).fetchone()

                if batch["prediction_count"] == 0:
                    return None

                conditions = ["batch_id = ?"]
                parameters: list[Any] = [batch_id]

                if action is not None:
                    conditions.append("action = ?")
                    parameters.append(action)

                rows = connection.execute(
                    f"""
                    SELECT *
                    FROM prediction_records
                    WHERE {" AND ".join(conditions)}
                    ORDER BY calibrated_probability DESC, prediction_id ASC
                    LIMIT ?
                    """,
                    [*parameters, limit],
                ).fetchall()

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not retrieve highest-risk predictions."
            ) from exc

        return {
            "batch_id": batch_id,
            "predicted_at_utc": batch["predicted_at_utc"],
            "prediction_count": batch["prediction_count"],
            "predictions": [self._deserialize_row(row) for row in rows],
        }

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Retrieve one prediction batch and its individual outputs."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM prediction_records
                    WHERE batch_id = ?
                    ORDER BY prediction_id ASC
                    """,
                    (batch_id,),
                ).fetchall()

        except sqlite3.Error as exc:
            raise PredictionStorageError(
                "Could not retrieve prediction batch."
            ) from exc

        if not rows:
            return None

        predictions = [self._deserialize_row(row) for row in rows]

        return {
            "batch_id": batch_id,
            "predicted_at_utc": rows[0]["predicted_at_utc"],
            "prediction_count": len(predictions),
            "predictions": predictions,
        }