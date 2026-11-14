
"""Reusable preprocessing pipelines for DriftSentinel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET_COLUMN = "isFraud"

EXCLUDED_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    TARGET_COLUMN,
}

# Numeric columns that represent codes or identifiers, not measurements.
NUMERIC_CATEGORICAL_COLUMNS = {
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "id_18",
    "id_32",
}


def select_feature_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numerical and categorical predictor columns."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")

    available_columns = [
        column
        for column in X.columns
        if column not in EXCLUDED_COLUMNS
    ]

    categorical_columns = [
        column
        for column in available_columns
        if (
            pd.api.types.is_object_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[column])
            or column in NUMERIC_CATEGORICAL_COLUMNS
        )
    ]

    numerical_columns = [
        column
        for column in available_columns
        if (
            column not in categorical_columns
            and pd.api.types.is_numeric_dtype(X[column])
        )
    ]

    unsupported_columns = [
        column
        for column in available_columns
        if (
            column not in categorical_columns
            and column not in numerical_columns
        )
    ]

    if unsupported_columns:
        raise TypeError(
            f"Unsupported feature columns: {unsupported_columns}"
        )

    if not numerical_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found.")

    return numerical_columns, categorical_columns

def categorical_to_string(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical values to strings while preserving missingness."""
    return (
        X.astype(object)
        .where(X.notna(), "__MISSING__")
        .astype(str)
    )

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline using the columns in X."""

    numerical_columns, categorical_columns = select_feature_columns(X)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
    steps=[
        (
            "to_string",
            FunctionTransformer(
                 categorical_to_string,
                feature_names_out="one-to-one",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.001,
                sparse_output=True,
                dtype=np.float32,
            ),
        ),
    ]
)

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
