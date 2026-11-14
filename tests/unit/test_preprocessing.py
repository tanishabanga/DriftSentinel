
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from driftsentinel.features.preprocessing import (
    build_preprocessor,
    categorical_to_string,
    select_feature_columns,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "amount": [100.0, 200.0, np.nan, 400.0, 500.0],
        "card1": [101, 102, 101, 103, 102],
        "category": ["A", "B", None, "A", "C"],
        "isFraud": [0, 1, 0, 0, 1],
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [10, 20, 30, 40, 50],
    })


def test_feature_classification(sample_data):
    numerical, categorical = select_feature_columns(sample_data)

    assert numerical == ["amount"]
    assert "card1" in categorical
    assert "category" in categorical


def test_identifiers_timestamp_and_target_are_excluded(sample_data):
    numerical, categorical = select_feature_columns(sample_data)

    selected = set(numerical + categorical)

    assert "TransactionID" not in selected
    assert "TransactionDT" not in selected
    assert "isFraud" not in selected


def test_categorical_missing_values_are_preserved_as_sentinel():
    data = pd.DataFrame({
        "category": ["A", None, np.nan, 101]
    })

    result = categorical_to_string(data)

    assert result["category"].tolist() == [
        "A",
        "__MISSING__",
        "__MISSING__",
        "101",
    ]


def test_preprocessor_returns_sparse_output(sample_data):
    X = sample_data.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert issparse(transformed)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]


def test_validation_with_unseen_categories(sample_data):
    X_train = sample_data.drop(
        columns=["isFraud", "TransactionID", "TransactionDT"]
    )

    X_validation = pd.DataFrame({
        "amount": [250.0, np.nan],
        "card1": [999, 101],
        "category": ["UNSEEN_CATEGORY", None],
    })

    preprocessor = build_preprocessor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    assert issparse(X_train_transformed)
    assert issparse(X_validation_transformed)
    assert (
        X_train_transformed.shape[1]
        == X_validation_transformed.shape[1]
    )
    assert np.isfinite(X_validation_transformed.data).all()


def test_duplicate_columns_are_rejected():
    X = pd.DataFrame(
        [[1, 2]],
        columns=["amount", "amount"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        select_feature_columns(X)
