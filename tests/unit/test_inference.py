
import json

import numpy as np
import pandas as pd
import pytest
import copy

import driftsentinel.models.inference as inference


class FakeModel:
    classes_ = np.array([0, 1])

    def load_model(self, path):
        self.loaded = True

    def predict_proba(self, features):
        probabilities = features["amount"].to_numpy(dtype=float)
        probabilities = np.clip(probabilities, 0, 1)

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


class FakeCalibrator:
    def predict(self, probabilities):
        return np.clip(
            probabilities * 0.9 + 0.05,
            0,
            1
        )


@pytest.fixture
def service_paths(tmp_path, monkeypatch):
    schema = {
        "schema_version": 1,
        "feature_count": 2,
        "feature_names": ["amount", "category"],
        "numerical_columns": ["amount"],
        "categorical_columns": ["category"],
        "categorical_vocabularies": {
            "category": ["A", "B"]
        },
        "missing_value_policy": "preserve_missing_values",
        "unknown_category_policy": "map_to_missing"
    }

    schema_path = tmp_path / "schema.json"
    model_path = tmp_path / "model.json"
    calibrator_path = tmp_path / "calibrator.joblib"

    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    model_path.write_text("fake model", encoding="utf-8")
    calibrator_path.write_text("fake calibrator", encoding="utf-8")

    monkeypatch.setattr(
        inference,
        "XGBClassifier",
        FakeModel
    )

    monkeypatch.setattr(
        inference.joblib,
        "load",
        lambda path: FakeCalibrator()
    )

    return model_path, schema_path, calibrator_path


@pytest.fixture
def service(service_paths):
    model_path, schema_path, calibrator_path = service_paths

    return inference.FraudInferenceService(
        model_path=model_path,
        schema_path=schema_path,
        calibrator_path=calibrator_path
    )


def test_service_initializes_successfully(service):
    assert service.model.loaded is True
    assert service.feature_names == ["amount", "category"]


def test_feature_preparation_preserves_schema_order(service):
    data = pd.DataFrame({
        "category": ["A", "B"],
        "amount": [0.2, 0.8],
        "TransactionID": [1001, 1002]
    })

    prepared = service._prepare_features(data)

    assert prepared.columns.tolist() == [
        "amount",
        "category"
    ]

    assert str(prepared["amount"].dtype) == "float32"
    assert isinstance(prepared["category"].dtype, pd.CategoricalDtype)


def test_unknown_categories_become_missing(service):
    data = pd.DataFrame({
        "amount": [0.5],
        "category": ["UNKNOWN"]
    })

    prepared = service._prepare_features(data)

    assert pd.isna(prepared.loc[0, "category"])


def test_invalid_numeric_values_become_missing(service):
    data = pd.DataFrame({
        "amount": ["invalid"],
        "category": ["A"]
    })

    prepared = service._prepare_features(data)

    assert pd.isna(prepared.loc[0, "amount"])


def test_missing_required_feature_raises_error(service):
    data = pd.DataFrame({
        "amount": [0.5]
    })

    with pytest.raises(ValueError, match="missing required features"):
        service.predict_raw(data)


def test_empty_input_raises_error(service):
    data = pd.DataFrame(columns=["amount", "category"])

    with pytest.raises(ValueError, match="cannot be empty"):
        service.predict_raw(data)


def test_raw_prediction_uses_fraud_class(service):
    data = pd.DataFrame({
        "amount": [0.2, 0.8],
        "category": ["A", "B"]
    })

    probabilities = service.predict_raw(data)

    np.testing.assert_allclose(
        probabilities,
        [0.2, 0.8]
    )


def test_calibrated_prediction(service):
    raw = np.array([0.2, 0.8])

    calibrated = service.predict_calibrated(raw)

    np.testing.assert_allclose(
        calibrated,
        [0.23, 0.77]
    )


def test_complete_inference_pipeline(service):
    data = pd.DataFrame({
        "amount": [0.01, 0.5, 0.9],
        "category": ["A", "B", "A"]
    })

    results = service.predict(data)

    assert len(results) == 3

    assert results.columns.tolist() == [
        "raw_probability",
        "calibrated_probability",
        "action",
        "expected_cost"
    ]

    assert np.isfinite(
        results["raw_probability"]
    ).all()

    assert np.isfinite(
        results["calibrated_probability"]
    ).all()

    assert results["action"].isin(
        ["approve", "review", "block"]
    ).all()

    assert (results["expected_cost"] >= 0).all()


@pytest.mark.parametrize(
    "probabilities",
    [
        [-0.1],
        [1.1],
        [np.nan],
        [np.inf],
        [[0.2, 0.8]]
    ]
)
def test_invalid_calibration_probabilities_raise_error(
    service,
    probabilities
):
    with pytest.raises(ValueError):
        service.predict_calibrated(probabilities)


def test_series_input_is_supported(service):
    transaction = pd.Series({
        "amount": 0.8,
        "category": "A"
    })

    results = service.predict(transaction)

    assert len(results) == 1
    assert results.iloc[0]["raw_probability"] == pytest.approx(0.8)


def test_duplicate_input_columns_are_rejected(service):
    data = pd.DataFrame(
        [[0.5, 0.6, "A"]],
        columns=["amount", "amount", "category"]
    )

    with pytest.raises(ValueError, match="Duplicate input columns"):
        service.predict_raw(data)


def test_reversed_model_class_order_is_supported(service):
    service.model.classes_ = np.array([1, 0])

    def reversed_predict_proba(features):
        probabilities = features["amount"].to_numpy(dtype=float)
        return np.column_stack([probabilities, 1 - probabilities])

    service.model.predict_proba = reversed_predict_proba

    data = pd.DataFrame({
        "amount": [0.2, 0.8],
        "category": ["A", "B"]
    })

    np.testing.assert_allclose(
        service.predict_raw(data),
        [0.2, 0.8]
    )


def test_prediction_output_preserves_input_row_order(service):
    data = pd.DataFrame(
        {
            "amount": [0.8, 0.2, 0.5],
            "category": ["A", "B", "A"]
        },
        index=[300, 100, 200]
    )

    results = service.predict(data)

    np.testing.assert_allclose(
        results["raw_probability"],
        [0.8, 0.2, 0.5]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "incorrect_feature_count",
        "duplicate_feature_names",
        "overlapping_columns",
        "incomplete_feature_coverage",
        "missing_vocabulary",
        "unsupported_missing_policy",
        "unsupported_unknown_category_policy"
    ]
)
def test_invalid_schema_is_rejected(service, mutation):
    schema = copy.deepcopy(service.schema)

    if mutation == "incorrect_feature_count":
        schema["feature_count"] = 3
    elif mutation == "duplicate_feature_names":
        schema["feature_names"] = ["amount", "amount"]
    elif mutation == "overlapping_columns":
        schema["categorical_columns"] = ["amount", "category"]
    elif mutation == "incomplete_feature_coverage":
        schema["numerical_columns"] = []
    elif mutation == "missing_vocabulary":
        schema["categorical_vocabularies"] = {}
    elif mutation == "unsupported_missing_policy":
        schema["missing_value_policy"] = "fill_with_zero"
    elif mutation == "unsupported_unknown_category_policy":
        schema["unknown_category_policy"] = "ignore"

    service.schema = schema

    with pytest.raises(ValueError):
        service._validate_schema()



def test_transaction_id_is_preserved(service):
    data = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "amount": [0.2, 0.8],
        "category": ["A", "B"]
    })

    results = service.predict(data)

    assert results["TransactionID"].tolist() == [1001, 1002]


def test_original_dataframe_index_is_preserved(service):
    data = pd.DataFrame(
        {
            "amount": [0.2, 0.8],
            "category": ["A", "B"]
        },
        index=[500, 200]
    )

    results = service.predict(data)

    assert results.index.tolist() == [500, 200]


def test_transaction_metadata_does_not_change_predictions(service):
    data = pd.DataFrame({
        "amount": [0.2, 0.8],
        "category": ["A", "B"]
    })

    data_with_metadata = data.copy()
    data_with_metadata.insert(
        0,
        "TransactionID",
        [1001, 1002]
    )

    original_results = service.predict(data)
    metadata_results = service.predict(data_with_metadata)

    np.testing.assert_allclose(
        original_results["raw_probability"],
        metadata_results["raw_probability"]
    )

    np.testing.assert_allclose(
        original_results["calibrated_probability"],
        metadata_results["calibrated_probability"]
    )

    assert (
        original_results["action"].tolist()
        == metadata_results["action"].tolist()
    )
