"""Orchestration service for DriftSentinel monitoring."""

import numpy as np
import pandas as pd

from driftsentinel.monitoring.drift import evaluate_feature_drift
from driftsentinel.monitoring.predictions import summarize_predictions
from driftsentinel.monitoring.storage import (
    MonitoringSnapshotStore,
    MonitoringStorageError,
)


class MonitoringService:
    """Combine feature drift and prediction monitoring into snapshots."""

    def __init__(
        self,
        reference_profile: dict,
        store: MonitoringSnapshotStore,
    ):
        if not isinstance(reference_profile, dict):
            raise TypeError("reference_profile must be a dictionary.")

        if not isinstance(store, MonitoringSnapshotStore):
            raise TypeError("store must be a MonitoringSnapshotStore.")

        self.reference_profile = reference_profile
        self.store = store

    @staticmethod
    def _build_drift_summary(
        drift_results: pd.DataFrame,
    ) -> dict:
        """Build aggregate and per-feature drift metrics."""

        if not isinstance(drift_results, pd.DataFrame):
            raise TypeError("drift_results must be a pandas DataFrame.")

        if drift_results.empty:
            raise ValueError("Drift results cannot be empty.")

        required_columns = {
            "feature",
            "feature_type",
            "psi",
            "reference_missing_rate",
            "current_missing_rate",
            "missing_rate_change",
        }

        missing = required_columns.difference(drift_results.columns)

        if missing:
            raise ValueError(
                f"Drift results are missing columns: {sorted(missing)}"
            )

        results = drift_results.copy()

        numeric_columns = [
            "psi",
            "reference_missing_rate",
            "current_missing_rate",
            "missing_rate_change",
        ]

        numeric_metrics = results[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )

        metric_values = numeric_metrics.to_numpy(dtype=float)

        if not np.isfinite(metric_values).all():
            raise ValueError("Drift metrics must contain only finite values.")

        if (numeric_metrics["psi"] < 0).any():
            raise ValueError("PSI values must be non-negative.")

        missing_rate_columns = [
            "reference_missing_rate",
            "current_missing_rate",
        ]

        for column in missing_rate_columns:
            if not numeric_metrics[column].between(0, 1).all():
                raise ValueError(
                    f"{column} values must be between 0 and 1."
                )

        if results["feature"].isna().any():
            raise ValueError("Feature names cannot be missing.")

        if results["feature"].duplicated().any():
            raise ValueError("Feature names must be unique.")

        results[numeric_columns] = numeric_metrics

        results = results.sort_values(
            "psi",
            ascending=False,
            kind="stable",
        )

        feature_records = []

        for row in results.to_dict(orient="records"):
            feature_records.append(
                {
                    "feature": str(row["feature"]),
                    "feature_type": str(row["feature_type"]),
                    "psi": float(row["psi"]),
                    "reference_missing_rate": float(
                        row["reference_missing_rate"]
                    ),
                    "current_missing_rate": float(
                        row["current_missing_rate"]
                    ),
                    "missing_rate_change": float(
                        row["missing_rate_change"]
                    ),
                }
            )

        psi_values = results["psi"].astype(float)
        missing_changes = results["missing_rate_change"].astype(float)

        return {
            "feature_count": len(results),
            "mean_psi": float(psi_values.mean()),
            "median_psi": float(psi_values.median()),
            "max_psi": float(psi_values.max()),
            "features_psi_above_0_10": int(
                (psi_values > 0.10).sum()
            ),
            "features_psi_above_0_25": int(
                (psi_values > 0.25).sum()
            ),
            "mean_absolute_missing_rate_change": float(
                missing_changes.abs().mean()
            ),
            "features": feature_records,
        }

    def record_batch(
        self,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        window_start_utc: str | None = None,
        window_end_utc: str | None = None,
        metadata: dict | None = None,
        prediction_batch_id: str | None = None,
    ) -> dict:
        """Calculate and persist one aggregate monitoring snapshot."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predictions must be a pandas DataFrame.")

        if features.empty:
            raise ValueError("Feature batch cannot be empty.")

        if predictions.empty:
            raise ValueError("Prediction batch cannot be empty.")

        if len(features) != len(predictions):
            raise ValueError(
                "Feature and prediction batch sizes must match."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        prediction_summary = summarize_predictions(predictions)

        drift_results = evaluate_feature_drift(
            features,
            self.reference_profile,
        )

        drift_summary = self._build_drift_summary(drift_results)

        snapshot_metadata = {
            "monitoring_version": 1,
            "reference_source": self.reference_profile.get(
                "source",
                "unknown",
            ),
            "reference_row_count": self.reference_profile.get(
                "row_count"
            ),
            "reference_feature_count": self.reference_profile.get(
                "feature_count"
            ),
        }

        if metadata:
            snapshot_metadata.update(metadata)

        snapshot_id = self.store.save_snapshot(
            prediction_summary=prediction_summary,
            drift_summary=drift_summary,
            window_start_utc=window_start_utc,
            window_end_utc=window_end_utc,
            metadata=snapshot_metadata,
            prediction_batch_id=prediction_batch_id,
        )

        snapshot = self.store.get_snapshot(snapshot_id)

        if snapshot is None:
            raise MonitoringStorageError(
                "Saved monitoring snapshot could not be retrieved."
            )

        return snapshot
