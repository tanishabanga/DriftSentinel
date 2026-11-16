
"""Feature schema utilities for DriftSentinel inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from driftsentinel.features.preprocessing import (
    select_feature_columns,
)

EXCLUDED_COLUMNS = {
    "isFraud",
    "TransactionID",
    "TransactionDT",
}


def build_feature_schema(training_path: Path) -> dict:
    """Build an inference schema using the training partition only."""

    train_df = pd.read_parquet(training_path)

    X_train = train_df.drop(
        columns=list(EXCLUDED_COLUMNS),
    )

    numerical_columns, categorical_columns = (
        select_feature_columns(X_train)
    )

    categorical_vocabularies = {}

    for column in categorical_columns:
        values = X_train[column].astype("string")
        categories = values.dropna().unique()

        categorical_vocabularies[column] = [
            str(value) for value in categories
        ]

    return {
        "schema_version": 1,
        "feature_count": len(X_train.columns),
        "feature_names": list(X_train.columns),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "categorical_vocabularies": categorical_vocabularies,
        "missing_value_policy": "preserve_missing_values",
        "unknown_category_policy": "map_to_missing",
    }


def save_feature_schema(
    schema: dict,
    output_path: Path,
) -> None:
    """Save the feature schema as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(schema, indent=4),
        encoding="utf-8",
    )
