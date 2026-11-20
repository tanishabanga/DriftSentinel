import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftsentinel.api.main import app
from driftsentinel.models.exceptions import InferenceInputError
from driftsentinel.monitoring.reference import build_reference_profile
from driftsentinel.monitoring.prediction_storage import PredictionStorageError
from driftsentinel.monitoring.storage import MonitoringStorageError


class FakeMonitoringInferenceService:
    """Lightweight inference service for monitoring API tests."""

    def __init__(self):
        self.feature_names = ["amount", "velocity"]
        self.calls = []

    def predict(self, transactions):
        self.calls.append(transactions.copy())

        missing = set(self.feature_names) - set(transactions.columns)
        if missing:
            raise InferenceInputError(
                f"Missing required model features: {sorted(missing)}"
            )

        return pd.DataFrame(
            {
                "TransactionID": transactions.get(
                    "TransactionID",
                    pd.Series(range(len(transactions))),
                ).to_numpy(),
                "raw_probability": [0.75] * len(transactions),
                "calibrated_probability": [0.70] * len(transactions),
                "action": ["block"] * len(transactions),
                "expected_cost": [0.0] * len(transactions),
            }
        )


@pytest.fixture
def monitoring_client(monkeypatch, tmp_path):
    fake_service = FakeMonitoringInferenceService()

    reference_data = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 30.0, 40.0],
            "velocity": [1.0, 2.0, 3.0, 4.0],
        }
    )

    schema = {
        "feature_names": ["amount", "velocity"],
        "numerical_columns": ["amount", "velocity"],
        "categorical_columns": [],
    }

    reference_profile = build_reference_profile(
        reference_data,
        schema,
        source="monitoring_api_test",
        schema_source="test_schema",
    )

    reference_path = tmp_path / "reference_profile.json"
    reference_path.write_text(
        json.dumps(reference_profile),
        encoding="utf-8",
    )

    database_path = tmp_path / "monitoring-test.db"

    monkeypatch.setenv(
        "DRIFTSENTINEL_DB_PATH",
        str(database_path),
    )
    monkeypatch.setenv(
        "DRIFTSENTINEL_REFERENCE_PROFILE_PATH",
        str(reference_path),
    )

    monkeypatch.setattr(
        "driftsentinel.api.main.FraudInferenceService",
        lambda: fake_service,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, fake_service


def sample_transactions():
    return [
        {
            "TransactionID": 1001,
            "amount": 15.0,
            "velocity": 1.5,
        },
        {
            "TransactionID": 1002,
            "amount": 35.0,
            "velocity": 3.5,
        },
    ]


def test_create_monitoring_snapshot(monitoring_client):
    client, fake_service = monitoring_client

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": sample_transactions()},
    )

    assert response.status_code == 200

    snapshot = response.json()["snapshot"]

    assert snapshot["prediction_count"] == 2
    assert snapshot["prediction_summary"]["prediction_count"] == 2
    assert snapshot["drift_summary"]["feature_count"] == 2
    assert snapshot["metadata"]["reference_source"] == (
        "monitoring_api_test"
    )
    assert snapshot["prediction_batch_id"]

    batch_response = client.get(
        f"/predictions/batches/{snapshot['prediction_batch_id']}"
    )

    assert batch_response.status_code == 200

    batch = batch_response.json()["batch"]
    assert batch["batch_id"] == snapshot["prediction_batch_id"]
    assert batch["prediction_count"] == 2
    assert len(batch["predictions"]) == 2

    saved_ids = [
        record["TransactionID"]
        for record in batch["predictions"]
    ]
    assert saved_ids == [1001, 1002]

    for record in batch["predictions"]:
        assert "amount" not in record
        assert "velocity" not in record

    assert len(fake_service.calls) == 1


def test_snapshot_is_persisted_and_retrievable(monitoring_client):
    client, _ = monitoring_client

    create_response = client.post(
        "/monitoring/snapshots",
        json={"transactions": sample_transactions()},
    )

    assert create_response.status_code == 200

    created_snapshot = create_response.json()["snapshot"]
    snapshot_id = created_snapshot["snapshot_id"]

    retrieve_response = client.get(
        f"/monitoring/snapshots/{snapshot_id}"
    )

    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["snapshot"] == created_snapshot


def test_list_monitoring_snapshots(monitoring_client):
    client, _ = monitoring_client

    for _ in range(2):
        response = client.post(
            "/monitoring/snapshots",
            json={"transactions": sample_transactions()},
        )
        assert response.status_code == 200

    list_response = client.get("/monitoring/snapshots")

    assert list_response.status_code == 200
    assert len(list_response.json()["snapshots"]) == 2


def test_snapshot_list_limit_is_respected(monitoring_client):
    client, _ = monitoring_client

    for _ in range(2):
        client.post(
            "/monitoring/snapshots",
            json={"transactions": sample_transactions()},
        )

    response = client.get("/monitoring/snapshots?limit=1")

    assert response.status_code == 200
    assert len(response.json()["snapshots"]) == 1


def test_missing_snapshot_returns_404(monitoring_client):
    client, _ = monitoring_client

    response = client.get(
        "/monitoring/snapshots/nonexistent-snapshot"
    )

    assert response.status_code == 404


def test_missing_model_feature_returns_422(monitoring_client):
    client, fake_service = monitoring_client

    response = client.post(
        "/monitoring/snapshots",
        json={
            "transactions": [
                {
                    "TransactionID": 1001,
                    "amount": 15.0,
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "velocity" in response.json()["detail"]
    assert len(fake_service.calls) == 1


def test_invalid_window_returns_422(monitoring_client):
    client, _ = monitoring_client

    response = client.post(
        "/monitoring/snapshots",
        json={
            "transactions": sample_transactions(),
            "window_start_utc": "2026-11-17T12:00:00+00:00",
            "window_end_utc": "2026-11-16T12:00:00+00:00",
        },
    )

    assert response.status_code == 422


def test_invalid_snapshot_list_limit_returns_422(monitoring_client):
    client, _ = monitoring_client

    response = client.get("/monitoring/snapshots?limit=101")

    assert response.status_code == 422


def test_snapshot_storage_failure_returns_503(
    monitoring_client,
    monkeypatch,
):
    client, _ = monitoring_client

    def fail_save(*args, **kwargs):
        raise MonitoringStorageError("Database unavailable.")

    monkeypatch.setattr(
        "driftsentinel.monitoring.storage.MonitoringSnapshotStore.save_snapshot",
        fail_save,
    )

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": sample_transactions()},
    )

    assert response.status_code == 503

    detail = response.json()["detail"]
    assert "Predictions were saved" in detail["message"]

    batch_id = detail["prediction_batch_id"]
    assert batch_id

    batch_response = client.get(
        f"/predictions/batches/{batch_id}"
    )
    assert batch_response.status_code == 200
    assert batch_response.json()["batch"]["prediction_count"] == 2

    snapshots_response = client.get("/monitoring/snapshots")
    assert snapshots_response.status_code == 200
    assert snapshots_response.json()["snapshots"] == []


def test_prediction_ledger_failure_prevents_snapshot_creation(
    monitoring_client,
    monkeypatch,
):
    client, _ = monitoring_client

    def fail_save(*args, **kwargs):
        raise PredictionStorageError("Database unavailable.")

    monkeypatch.setattr(
        "driftsentinel.monitoring.prediction_storage.PredictionLedgerStore.save_batch",
        fail_save,
    )

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": sample_transactions()},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Predictions were generated but could not be persisted."
    )

    snapshots_response = client.get("/monitoring/snapshots")
    assert snapshots_response.status_code == 200
    assert snapshots_response.json()["snapshots"] == []


def test_snapshot_list_storage_failure_returns_503(
    monitoring_client,
    monkeypatch,
):
    client, _ = monitoring_client

    def fail_list(*args, **kwargs):
        raise MonitoringStorageError("Database unavailable.")

    monkeypatch.setattr(
        "driftsentinel.monitoring.storage.MonitoringSnapshotStore.list_snapshots",
        fail_list,
    )

    response = client.get("/monitoring/snapshots")

    assert response.status_code == 503



def test_empty_monitoring_batch_returns_422(monitoring_client):
    client, fake_service = monitoring_client

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": []},
    )

    assert response.status_code == 422
    assert len(fake_service.calls) == 0


def test_oversized_monitoring_batch_returns_422(monitoring_client):
    client, fake_service = monitoring_client

    transactions = [
        {"TransactionID": i, "amount": 15.0, "velocity": 1.5}
        for i in range(1001)
    ]

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": transactions},
    )

    assert response.status_code == 422
    assert len(fake_service.calls) == 0


def test_maximum_monitoring_batch_is_accepted(monitoring_client):
    client, fake_service = monitoring_client

    transactions = [
        {"TransactionID": i, "amount": 15.0, "velocity": 1.5}
        for i in range(1000)
    ]

    response = client.post(
        "/monitoring/snapshots",
        json={"transactions": transactions},
    )

    assert response.status_code == 200
    assert response.json()["snapshot"]["prediction_count"] == 1000
    assert len(fake_service.calls) == 1


def test_snapshot_retrieval_storage_failure_returns_503(
    monitoring_client,
    monkeypatch,
):
    client, _ = monitoring_client

    def fail_get(*args, **kwargs):
        raise MonitoringStorageError("Database unavailable.")

    monkeypatch.setattr(
        "driftsentinel.monitoring.storage.MonitoringSnapshotStore.get_snapshot",
        fail_get,
    )

    response = client.get("/monitoring/snapshots/test-snapshot")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Monitoring snapshot storage is unavailable."
    )
