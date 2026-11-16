
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
def client(monkeypatch):
    fake_service = FakeInferenceService()

    monkeypatch.setattr(
        "driftsentinel.api.main.FraudInferenceService",
        lambda: fake_service,
    )

    with TestClient(app) as test_client:
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



def test_inference_service_created_once_per_lifespan(monkeypatch):
    from fastapi.testclient import TestClient

    from driftsentinel.api import main

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

    with TestClient(main.app):
        assert initialization_count == 1

    assert initialization_count == 1



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
