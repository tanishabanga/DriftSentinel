import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NUMERIC_BINS = 10
DEFAULT_CATEGORICAL_TOP_K = 50


def validate_schema(df: pd.DataFrame, schema: dict) -> None:
    """Validate feature schema consistency against the input dataframe."""

    required_keys = {
        "feature_names",
        "numerical_columns",
        "categorical_columns",
    }

    missing_keys = required_keys - set(schema)

    if missing_keys:
        raise ValueError(
            f"Schema is missing required keys: {sorted(missing_keys)}"
        )

    feature_names = schema["feature_names"]
    numerical = schema["numerical_columns"]
    categorical = schema["categorical_columns"]

    for name, columns in (
        ("feature_names", feature_names),
        ("numerical_columns", numerical),
        ("categorical_columns", categorical),
    ):
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate columns found in {name}.")

    if set(numerical) & set(categorical):
        raise ValueError("Numerical and categorical features overlap.")

    if set(feature_names) != set(numerical) | set(categorical):
        raise ValueError(
            "Feature names do not exactly match numerical and categorical columns."
        )

    missing_features = sorted(set(feature_names) - set(df.columns))

    if missing_features:
        raise ValueError(
            f"Dataset is missing model features: {missing_features[:10]}"
        )

    if df.empty:
        raise ValueError("Cannot build a reference profile from empty data.")


def numeric_profile(
    series: pd.Series,
    num_bins: int = DEFAULT_NUMERIC_BINS,
) -> dict:
    """Build a numerical distribution profile with a missing-value bucket."""

    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    finite_mask = np.isfinite(values)
    valid = values[finite_mask]
    missing_count = int((~finite_mask).sum())

    if len(valid) == 0:
        cut_points = []
        bin_counts = []
    else:
        quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]

        cut_points = np.unique(
            np.quantile(valid, quantiles)
        ).tolist()

        bin_indices = np.searchsorted(
            cut_points,
            valid,
            side="right",
        )

        bin_counts = np.bincount(
            bin_indices,
            minlength=len(cut_points) + 1,
        ).astype(int).tolist()

    total = len(series)

    return {
        "cut_points": cut_points,
        "bin_counts": bin_counts,
        "missing_count": missing_count,
        "missing_rate": missing_count / total,
        "bin_proportions": [
            count / total for count in bin_counts
        ],
    }


def categorical_profile(
    series: pd.Series,
    top_k: int = DEFAULT_CATEGORICAL_TOP_K,
) -> dict:
    """Build a categorical profile with top categories and an other bucket."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    values = series.dropna().astype(str)
    frequencies = values.value_counts()

    top_categories = frequencies.head(top_k)

    top_counts = {
        str(category): int(count)
        for category, count in top_categories.items()
    }

    missing_count = int(series.isna().sum())
    other_count = int(len(values) - top_categories.sum())
    total = len(series)

    return {
        "top_categories": top_counts,
        "other_count": other_count,
        "missing_count": missing_count,
        "missing_rate": missing_count / total,
        "category_proportions": {
            category: count / total
            for category, count in top_counts.items()
        },
        "other_proportion": other_count / total,
    }


def build_reference_profile(
    df: pd.DataFrame,
    schema: dict,
    source: str = "unknown",
    schema_source: str = "unknown",
    num_bins: int = DEFAULT_NUMERIC_BINS,
    categorical_top_k: int = DEFAULT_CATEGORICAL_TOP_K,
) -> dict:
    """Build a reusable reference profile from training data."""

    validate_schema(df, schema)

    numerical_columns = schema["numerical_columns"]
    categorical_columns = schema["categorical_columns"]
    feature_names = schema["feature_names"]

    numerical_profiles = {
        column: numeric_profile(df[column], num_bins)
        for column in numerical_columns
    }

    categorical_profiles = {
        column: categorical_profile(df[column], categorical_top_k)
        for column in categorical_columns
    }

    return {
        "profile_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "schema_source": schema_source,
        "row_count": len(df),
        "feature_count": len(feature_names),
        "numerical_feature_count": len(numerical_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_top_k": categorical_top_k,
        "numerical": numerical_profiles,
        "categorical": categorical_profiles,
    }


def save_reference_profile(
    profile: dict,
    output_path: str | Path,
) -> None:
    """Save a reference profile as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, allow_nan=False)
