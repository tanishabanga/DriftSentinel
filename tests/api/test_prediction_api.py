
import pytest
from fastapi.testclient import TestClient

from driftsentinel.api.main import app


class FakeInferenceService:
    """Mock inference service for API tests."""

    def __init__(self):
        self.calls = []
        self.model = object()
        self.calibrator = object()

    def predict(self, transactions):
        import pandas as pd

        self.calls.append(transactions.copy())

        return pd.DataFrame({
            "TransactionID": transactions["TransactionID"].to_numpy(),
            "raw_probability": [0.75] * len(transactions),
            "calibrated_probability": [0.70] * len(transactions),
            "action": ["block"] * len(transactions),
            "expected_cost": [0.0] * len(transactions),
        })


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake_service = FakeInferenceService()
    database_path = tmp_path / "monitoring-test.db"

    monkeypatch.setenv(
        "DRIFTSENTINEL_DB_PATH",
        str(database_path),
    )

    monkeypatch.setattr(
        "driftsentinel.api.main.FraudInferenceService",
        lambda: fake_service,
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client, fake_service


def test_single_transaction_prediction(client):
    test_client, fake_service = client

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5}
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()["predictions"]

    assert len(data) == 1
    assert data[0]["TransactionID"] == 1001
    assert data[0]["raw_probability"] == 0.75
    assert data[0]["calibrated_probability"] == 0.70
    assert data[0]["action"] == "block"
    assert data[0]["expected_cost"] == 0.0

    assert len(fake_service.calls) == 1


def test_batch_transaction_prediction(client):
    test_client, fake_service = client

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
                {"TransactionID": 1002, "feature": 20.0},
            ]
        },
    )

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2
    assert predictions[0]["TransactionID"] == 1001
    assert predictions[1]["TransactionID"] == 1002
    assert len(fake_service.calls) == 1


def test_empty_transaction_list_rejected(client):
    test_client, _ = client

    response = test_client.post(
        "/predict",
        json={"transactions": []},
    )

    assert response.status_code == 422


def test_missing_transactions_field_rejected(client):
    test_client, _ = client

    response = test_client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_transaction_order_preserved(client):
    test_client, _ = client

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 3003, "feature": 1.0},
                {"TransactionID": 1001, "feature": 2.0},
                {"TransactionID": 2002, "feature": 3.0},
            ]
        },
    )

    assert response.status_code == 200

    ids = [
        item["TransactionID"]
        for item in response.json()["predictions"]
    ]

    assert ids == [3003, 1001, 2002]


def test_inference_service_initialized_once(client):
    _, fake_service = client

    assert isinstance(fake_service, FakeInferenceService)



def test_inference_service_created_once_per_lifespan(
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    from driftsentinel.api import main

    monkeypatch.setenv(
        "DRIFTSENTINEL_DB_PATH",
        str(tmp_path / "lifecycle-test.db"),
    )

    initialization_count = 0
    fake_service = FakeInferenceService()

    def fake_constructor():
        nonlocal initialization_count
        initialization_count += 1
        return fake_service

    monkeypatch.setattr(
        main,
        "FraudInferenceService",
        fake_constructor,
    )

    with TestClient(main.app) as test_client:
        assert initialization_count == 1

        monitoring_store = test_client.app.state.monitoring_store
        monitoring_service = test_client.app.state.monitoring_service

        assert monitoring_store is not None
        assert monitoring_store.database_path == (
            tmp_path / "lifecycle-test.db"
        )

        assert monitoring_service is not None
        assert monitoring_service.store is monitoring_store
        assert monitoring_service.reference_profile["feature_count"] == 431
        assert monitoring_service.reference_profile["row_count"] == 413379

    assert initialization_count == 1
    assert main.app.state.inference_service is None
    assert main.app.state.monitoring_store is None
    assert main.app.state.monitoring_service is None



def test_health_reports_loaded_components(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "DriftSentinel API",
        "model_loaded": True,
        "calibrator_loaded": True,
    }


def test_health_returns_503_when_inference_service_is_missing(client):
    test_client, _ = client
    test_client.app.state.inference_service = None

    response = test_client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "unavailable",
        "service": "DriftSentinel API",
        "model_loaded": False,
        "calibrator_loaded": False,
    }


@pytest.mark.parametrize("component", ["model", "calibrator"])
def test_health_returns_503_when_component_is_missing(client, component):
    test_client, fake_service = client
    setattr(fake_service, component, None)

    response = test_client.get("/health")

    assert response.status_code == 503

    detail = response.json()["detail"]
    assert detail["status"] == "unavailable"
    assert detail["model_loaded"] is (component != "model")
    assert detail["calibrator_loaded"] is (component != "calibrator")


def test_oversized_transaction_batch_rejected(client):
    test_client, fake_service = client

    transactions = [
        {"TransactionID": i, "feature": 1.0}
        for i in range(1001)
    ]

    response = test_client.post(
        "/predict",
        json={"transactions": transactions},
    )

    assert response.status_code == 422
    assert len(fake_service.calls) == 0



def test_invalid_inference_input_returns_422(client):
    from driftsentinel.models.exceptions import InferenceInputError

    test_client, fake_service = client

    def reject_input(transactions):
        raise InferenceInputError(
            "Input is missing required features: ['amount']"
        )

    fake_service.predict = reject_input

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5}
            ]
        },
    )

    assert response.status_code == 422
    assert "missing required features" in response.json()["detail"]


def test_unexpected_value_error_returns_500(client):
    test_client, fake_service = client

    def fail_internally(transactions):
        raise ValueError("Internal model failure")

    fake_service.predict = fail_internally

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5}
            ]
        },
    )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"


def test_maximum_transaction_batch_accepted(client):
    test_client, fake_service = client

    transactions = [
        {"TransactionID": i, "feature": 1.0}
        for i in range(1000)
    ]

    response = test_client.post(
        "/predict",
        json={"transactions": transactions},
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1000
    assert len(fake_service.calls) == 1


def test_startup_fails_when_reference_profile_is_missing(
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    from driftsentinel.api import main

    monkeypatch.setenv(
        "DRIFTSENTINEL_DB_PATH",
        str(tmp_path / "missing-reference-test.db"),
    )
    monkeypatch.setenv(
        "DRIFTSENTINEL_REFERENCE_PROFILE_PATH",
        str(tmp_path / "missing-reference.json"),
    )

    monkeypatch.setattr(
        main,
        "FraudInferenceService",
        FakeInferenceService,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Monitoring reference profile not found",
    ), TestClient(main.app):
        pass


def test_prediction_batch_is_persisted(client):
    test_client, _ = client

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
                {"TransactionID": 1002, "feature": 20.0},
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()
    batch_id = data["batch_id"]

    assert batch_id
    assert len(data["predictions"]) == 2

    store = test_client.app.state.prediction_store
    batch = store.get_batch(batch_id)

    assert batch is not None
    assert batch["prediction_count"] == 2
    assert [item["TransactionID"] for item in batch["predictions"]] == [
        1001,
        1002,
    ]


def test_prediction_storage_failure_returns_503(client, monkeypatch):
    test_client, fake_service = client

    from driftsentinel.monitoring.prediction_storage import (
        PredictionStorageError,
    )

    def fail_save(predictions):
        raise PredictionStorageError("Database unavailable.")

    monkeypatch.setattr(
        test_client.app.state.prediction_store,
        "save_batch",
        fail_save,
    )

    response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
            ]
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Predictions were generated but could not be persisted."
    )
    assert len(fake_service.calls) == 1

def test_list_prediction_history(client):
    test_client, _ = client

    create_response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
                {"TransactionID": 1002, "feature": 20.0},
            ]
        },
    )

    assert create_response.status_code == 200

    response = test_client.get("/predictions")

    assert response.status_code == 200

    predictions = response.json()["predictions"]

    assert len(predictions) == 2
    assert {item["TransactionID"] for item in predictions} == {1001, 1002}
    assert all(item["action"] == "block" for item in predictions)


def test_prediction_history_limit_is_respected(client):
    test_client, _ = client

    for transaction_id in (1001, 1002, 1003):
        response = test_client.post(
            "/predict",
            json={
                "transactions": [
                    {"TransactionID": transaction_id, "feature": 12.5},
                ]
            },
        )
        assert response.status_code == 200

    response = test_client.get("/predictions?limit=2")

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_invalid_prediction_history_limit_returns_422(client):
    test_client, _ = client

    response = test_client.get("/predictions?limit=101")

    assert response.status_code == 422


def test_prediction_batch_endpoint_returns_complete_batch(client):
    test_client, _ = client

    create_response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
                {"TransactionID": 1002, "feature": 20.0},
            ]
        },
    )

    batch_id = create_response.json()["batch_id"]

    response = test_client.get(
        f"/predictions/batches/{batch_id}"
    )

    assert response.status_code == 200

    batch = response.json()["batch"]

    assert batch["batch_id"] == batch_id
    assert batch["prediction_count"] == 2
    assert [item["TransactionID"] for item in batch["predictions"]] == [
        1001,
        1002,
    ]


def test_missing_prediction_batch_returns_404(client):
    test_client, _ = client

    response = test_client.get(
        "/predictions/batches/nonexistent-batch"
    )

    assert response.status_code == 404


def test_prediction_history_storage_failure_returns_503(
    client,
    monkeypatch,
):
    test_client, _ = client

    from driftsentinel.monitoring.prediction_storage import (
        PredictionStorageError,
    )

    def fail_list(*args, **kwargs):
        raise PredictionStorageError("Database unavailable.")

    monkeypatch.setattr(
        test_client.app.state.prediction_store,
        "list_predictions",
        fail_list,
    )

    response = test_client.get("/predictions")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction history storage is unavailable."
    )


def test_prediction_batch_storage_failure_returns_503(
    client,
    monkeypatch,
):
    test_client, _ = client

    from driftsentinel.monitoring.prediction_storage import (
        PredictionStorageError,
    )

    def fail_get(*args, **kwargs):
        raise PredictionStorageError("Database unavailable.")

    monkeypatch.setattr(
        test_client.app.state.prediction_store,
        "get_batch",
        fail_get,
    )

    response = test_client.get(
        "/predictions/batches/test-batch"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction history storage is unavailable."
    )

def test_list_prediction_batches(client):
    test_client, _ = client

    create_response = test_client.post(
        "/predict",
        json={
            "transactions": [
                {"TransactionID": 1001, "feature": 12.5},
                {"TransactionID": 1002, "feature": 20.0},
            ],
        },
    )
    assert create_response.status_code == 200

    response = test_client.get("/predictions/batches")

    assert response.status_code == 200
    batches = response.json()["batches"]

    assert len(batches) == 1
    assert batches[0]["batch_id"] == create_response.json()["batch_id"]
    assert batches[0]["prediction_count"] == 2
    assert batches[0]["block_count"] == 2
    assert batches[0]["approve_count"] == 0
    assert batches[0]["review_count"] == 0


def test_prediction_batch_page_paginates_records(client):
    test_client, _ = client

    predictions = [
        {
            "TransactionID": transaction_id,
            "raw_probability": 0.75,
            "calibrated_probability": 0.70,
            "action": "block",
            "expected_cost": 0.0,
        }
        for transaction_id in range(1001, 1006)
    ]
    batch_id = test_client.app.state.prediction_store.save_batch(predictions)

    response = test_client.get(
        f"/predictions/batches/{batch_id}/page?limit=2&offset=2"
    )

    assert response.status_code == 200
    batch = response.json()["batch"]

    assert batch["batch_id"] == batch_id
    assert batch["prediction_count"] == 5
    assert batch["total"] == 5
    assert batch["limit"] == 2
    assert batch["offset"] == 2
    assert [item["TransactionID"] for item in batch["predictions"]] == [
        1003,
        1004,
    ]


def test_prediction_batch_page_searches_transaction_id(client):
    test_client, _ = client

    batch_id = test_client.app.state.prediction_store.save_batch([
        {
            "TransactionID": 1001,
            "raw_probability": 0.75,
            "calibrated_probability": 0.70,
            "action": "block",
            "expected_cost": 0.0,
        },
        {
            "TransactionID": "TX-1002",
            "raw_probability": 0.20,
            "calibrated_probability": 0.15,
            "action": "review",
            "expected_cost": 1.5,
        },
    ])

    response = test_client.get(
        f"/predictions/batches/{batch_id}/page?search=TX-1002"
    )

    assert response.status_code == 200
    batch = response.json()["batch"]

    assert batch["total"] == 1
    assert batch["predictions"][0]["TransactionID"] == "TX-1002"


def test_prediction_batch_page_filters_by_action(client):
    test_client, _ = client

    batch_id = test_client.app.state.prediction_store.save_batch([
        {
            "TransactionID": 1001,
            "raw_probability": 0.75,
            "calibrated_probability": 0.70,
            "action": "block",
            "expected_cost": 0.0,
        },
        {
            "TransactionID": 1002,
            "raw_probability": 0.20,
            "calibrated_probability": 0.15,
            "action": "review",
            "expected_cost": 1.5,
        },
    ])

    response = test_client.get(
        f"/predictions/batches/{batch_id}/page?action=review"
    )

    assert response.status_code == 200
    batch = response.json()["batch"]

    assert batch["total"] == 1
    assert batch["predictions"][0]["action"] == "review"


def test_prediction_batch_page_returns_404_for_missing_batch(client):
    test_client, _ = client

    response = test_client.get(
        "/predictions/batches/missing-batch/page"
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
        "action=unknown",
        f"search={'x' * 201}",
    ],
)
def test_prediction_batch_page_rejects_invalid_query_parameters(
    client,
    query,
):
    test_client, _ = client

    response = test_client.get(
        f"/predictions/batches/test-batch/page?{query}"
    )

    assert response.status_code == 422

def test_prediction_batch_top_risks_returns_descending_probabilities(client):
    test_client, _ = client
    store = test_client.app.state.prediction_store

    predictions = []
    for index, probability in enumerate([0.25, 0.95, 0.60, 0.80, 0.99]):
        predictions.append({
            "TransactionID": 2000 + index,
            "raw_probability": probability,
            "calibrated_probability": probability,
            "action": "block" if probability >= 0.8 else "review",
            "expected_cost": 0.0,
        })

    batch_id = store.save_batch(predictions)

    response = test_client.get(
        f"/predictions/batches/{batch_id}/top-risk"
    )

    assert response.status_code == 200
    batch = response.json()["batch"]
    assert batch["batch_id"] == batch_id
    assert batch["prediction_count"] == 5

    probabilities = [
        item["calibrated_probability"]
        for item in batch["predictions"]
    ]
    assert probabilities == [0.99, 0.95, 0.80, 0.60, 0.25]


def test_prediction_batch_top_risks_respects_limit(client):
    test_client, _ = client
    store = test_client.app.state.prediction_store

    predictions = [
        {
            "TransactionID": index,
            "raw_probability": index / 10,
            "calibrated_probability": index / 10,
            "action": "review",
            "expected_cost": 0.0,
        }
        for index in range(1, 11)
    ]
    batch_id = store.save_batch(predictions)

    response = test_client.get(
        f"/predictions/batches/{batch_id}/top-risk?limit=3"
    )

    assert response.status_code == 200
    assert len(response.json()["batch"]["predictions"]) == 3
    assert [
        item["calibrated_probability"]
        for item in response.json()["batch"]["predictions"]
    ] == [1.0, 0.9, 0.8]


def test_prediction_batch_top_risks_filters_by_action(client):
    test_client, _ = client
    store = test_client.app.state.prediction_store

    batch_id = store.save_batch([
        {
            "TransactionID": 1,
            "raw_probability": 0.95,
            "calibrated_probability": 0.95,
            "action": "block",
            "expected_cost": 0.0,
        },
        {
            "TransactionID": 2,
            "raw_probability": 0.85,
            "calibrated_probability": 0.85,
            "action": "review",
            "expected_cost": 0.0,
        },
    ])

    response = test_client.get(
        f"/predictions/batches/{batch_id}/top-risk?action=review"
    )

    assert response.status_code == 200
    predictions = response.json()["batch"]["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["action"] == "review"
    assert predictions[0]["TransactionID"] == 2


def test_prediction_batch_top_risks_returns_404_for_missing_batch(client):
    test_client, _ = client

    response = test_client.get(
        "/predictions/batches/missing-batch/top-risk"
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=21",
        "action=unknown",
    ],
)
def test_prediction_batch_top_risks_rejects_invalid_query(client, query):
    test_client, _ = client

    response = test_client.get(
        f"/predictions/batches/test-batch/top-risk?{query}"
    )

    assert response.status_code == 422
