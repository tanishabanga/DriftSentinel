import pandas as pd
import pytest

from driftsentinel.monitoring.service import MonitoringService
from driftsentinel.monitoring.storage import MonitoringSnapshotStore


@pytest.fixture
def reference_profile():
    return {
        "source": "training_reference",
        "row_count": 100,
        "feature_count": 2,
        "numerical": {
            "amount": {
                "cut_points": [10.0, 20.0],
                "bin_proportions": [0.3, 0.4, 0.3],
                "missing_rate": 0.0,
            }
        },
        "categorical": {
            "category": {
                "top_categories": {"A": 60, "B": 40},
                "category_proportions": {"A": 0.6, "B": 0.4},
                "other_proportion": 0.0,
                "missing_rate": 0.0,
            }
        },
    }


@pytest.fixture
def service(tmp_path, reference_profile):
    store = MonitoringSnapshotStore(tmp_path / "monitoring.db")
    store.initialize()
    return MonitoringService(reference_profile, store)


def sample_features():
    return pd.DataFrame(
        {
            "amount": [5.0, 15.0, 25.0, 12.0],
            "category": ["A", "A", "B", "C"],
        }
    )


def sample_predictions():
    return pd.DataFrame(
        {
            "raw_probability": [0.01, 0.15, 0.75, 0.40],
            "calibrated_probability": [0.02, 0.12, 0.80, 0.35],
            "action": ["approve", "approve", "block", "review"],
            "expected_cost": [0.2, 0.5, 0.1, 1.5],
        }
    )


def test_record_batch_persists_combined_snapshot(service):
    snapshot = service.record_batch(
        sample_features(),
        sample_predictions(),
    )

    assert snapshot["prediction_count"] == 4
    assert snapshot["prediction_summary"]["prediction_count"] == 4
    assert snapshot["drift_summary"]["feature_count"] == 2
    assert snapshot["metadata"]["reference_source"] == "training_reference"


def test_snapshot_contains_per_feature_drift_metrics(service):
    snapshot = service.record_batch(
        sample_features(),
        sample_predictions(),
    )

    features = snapshot["drift_summary"]["features"]

    assert len(features) == 2
    assert all("psi" in feature for feature in features)
    assert all("missing_rate_change" in feature for feature in features)
    assert features[0]["psi"] >= features[1]["psi"]


def test_snapshot_contains_prediction_action_counts(service):
    snapshot = service.record_batch(
        sample_features(),
        sample_predictions(),
    )

    assert snapshot["prediction_summary"]["action_counts"] == {
        "approve": 2,
        "block": 1,
        "review": 1,
    }


def test_snapshot_preserves_window_and_metadata(service):
    snapshot = service.record_batch(
        sample_features(),
        sample_predictions(),
        window_start_utc="2026-11-17T10:00:00+00:00",
        window_end_utc="2026-11-17T11:00:00+00:00",
        metadata={"batch_source": "api"},
    )

    assert snapshot["window_start_utc"] == (
        "2026-11-17T10:00:00+00:00"
    )
    assert snapshot["window_end_utc"] == (
        "2026-11-17T11:00:00+00:00"
    )
    assert snapshot["metadata"]["batch_source"] == "api"


def test_record_batch_does_not_store_transaction_ids(service):
    features = sample_features()
    predictions = sample_predictions()
    predictions["TransactionID"] = [1001, 1002, 1003, 1004]

    snapshot = service.record_batch(features, predictions)

    assert "TransactionID" not in snapshot["prediction_summary"]
    assert "TransactionID" not in snapshot["drift_summary"]


def test_record_batch_rejects_non_dataframe_features(service):
    with pytest.raises(TypeError, match="features"):
        service.record_batch([], sample_predictions())


def test_record_batch_rejects_non_dataframe_predictions(service):
    with pytest.raises(TypeError, match="predictions"):
        service.record_batch(sample_features(), [])


def test_record_batch_rejects_empty_features(service):
    with pytest.raises(ValueError, match="Feature batch"):
        service.record_batch(
            pd.DataFrame(columns=["amount", "category"]),
            sample_predictions(),
        )


def test_record_batch_rejects_empty_predictions(service):
    with pytest.raises(ValueError, match="Prediction batch"):
        service.record_batch(
            sample_features(),
            pd.DataFrame(
                columns=[
                    "raw_probability",
                    "calibrated_probability",
                    "action",
                    "expected_cost",
                ]
            ),
        )


def test_record_batch_rejects_mismatched_batch_sizes(service):
    with pytest.raises(ValueError, match="batch sizes"):
        service.record_batch(
            sample_features(),
            sample_predictions().iloc[:2],
        )


def test_record_batch_rejects_missing_reference_feature(service):
    features = sample_features().drop(columns=["category"])

    with pytest.raises(ValueError, match="missing reference features"):
        service.record_batch(features, sample_predictions())


def test_record_batch_rejects_invalid_metadata(service):
    with pytest.raises(TypeError, match="metadata"):
        service.record_batch(
            sample_features(),
            sample_predictions(),
            metadata=[],
        )


def test_service_rejects_invalid_reference_profile(tmp_path):
    store = MonitoringSnapshotStore(tmp_path / "monitoring.db")

    with pytest.raises(TypeError, match="reference_profile"):
        MonitoringService([], store)


def test_service_rejects_invalid_store(reference_profile):
    with pytest.raises(TypeError, match="store"):
        MonitoringService(reference_profile, object())


def sample_drift_results():
    return pd.DataFrame(
        {
            "feature": ["amount", "category"],
            "feature_type": ["numerical", "categorical"],
            "psi": [0.12, 0.04],
            "reference_missing_rate": [0.0, 0.1],
            "current_missing_rate": [0.05, 0.2],
            "missing_rate_change": [0.05, 0.1],
        }
    )


def test_drift_summary_rejects_infinite_psi():
    results = sample_drift_results()
    results.loc[0, "psi"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        MonitoringService._build_drift_summary(results)


def test_drift_summary_rejects_nan_missing_rate():
    results = sample_drift_results()
    results.loc[0, "current_missing_rate"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        MonitoringService._build_drift_summary(results)


def test_drift_summary_rejects_missing_rate_above_one():
    results = sample_drift_results()
    results.loc[0, "reference_missing_rate"] = 1.2

    with pytest.raises(ValueError, match="between 0 and 1"):
        MonitoringService._build_drift_summary(results)


def test_drift_summary_rejects_negative_psi():
    results = sample_drift_results()
    results.loc[0, "psi"] = -0.01

    with pytest.raises(ValueError, match="non-negative"):
        MonitoringService._build_drift_summary(results)


def test_drift_summary_rejects_duplicate_feature_names():
    results = sample_drift_results()
    results.loc[1, "feature"] = "amount"

    with pytest.raises(ValueError, match="unique"):
        MonitoringService._build_drift_summary(results)
