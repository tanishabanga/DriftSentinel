import numpy as np
import pandas as pd
import pytest

from driftsentinel.monitoring.reference import (
    build_reference_profile,
    categorical_profile,
    numeric_profile,
    save_reference_profile,
    validate_schema,
)


def test_numeric_profile_tracks_missing_values():
    series = pd.Series([1.0, 2.0, np.nan, 4.0])

    result = numeric_profile(series, num_bins=2)

    assert result["missing_count"] == 1
    assert result["missing_rate"] == 0.25
    assert sum(result["bin_counts"]) == 3
    assert sum(result["bin_proportions"]) + result["missing_rate"] == pytest.approx(1.0)


def test_numeric_profile_handles_non_finite_values():
    series = pd.Series([1.0, np.inf, -np.inf, np.nan])

    result = numeric_profile(series)

    assert result["missing_count"] == 3
    assert result["missing_rate"] == 0.75


def test_numeric_profile_handles_all_missing_values():
    result = numeric_profile(pd.Series([np.nan, np.inf]))

    assert result["cut_points"] == []
    assert result["bin_counts"] == []
    assert result["missing_rate"] == 1.0


def test_numeric_profile_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        numeric_profile(pd.Series([1, 2]), num_bins=0)


def test_categorical_profile_tracks_other_and_missing():
    series = pd.Series(["A", "A", "B", "C", None])

    result = categorical_profile(series, top_k=1)

    assert result["top_categories"] == {"A": 2}
    assert result["other_count"] == 2
    assert result["missing_count"] == 1
    assert result["missing_rate"] == 0.2
    assert result["other_proportion"] == 0.4


def test_categorical_profile_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        categorical_profile(pd.Series(["A"]), top_k=0)


def test_schema_validation_accepts_complete_schema():
    df = pd.DataFrame({
        "amount": [10.0, 20.0],
        "device": ["mobile", "desktop"],
    })

    schema = {
        "feature_names": ["amount", "device"],
        "numerical_columns": ["amount"],
        "categorical_columns": ["device"],
    }

    validate_schema(df, schema)


def test_schema_validation_rejects_missing_features():
    df = pd.DataFrame({"amount": [10.0]})

    schema = {
        "feature_names": ["amount", "device"],
        "numerical_columns": ["amount"],
        "categorical_columns": ["device"],
    }

    with pytest.raises(ValueError, match="missing model features"):
        validate_schema(df, schema)


def test_schema_validation_rejects_overlapping_features():
    df = pd.DataFrame({"amount": [10.0]})

    schema = {
        "feature_names": ["amount"],
        "numerical_columns": ["amount"],
        "categorical_columns": ["amount"],
    }

    with pytest.raises(ValueError, match="overlap"):
        validate_schema(df, schema)


def test_schema_validation_rejects_empty_data():
    df = pd.DataFrame({"amount": pd.Series(dtype=float)})

    schema = {
        "feature_names": ["amount"],
        "numerical_columns": ["amount"],
        "categorical_columns": [],
    }

    with pytest.raises(ValueError, match="empty data"):
        validate_schema(df, schema)


def test_build_reference_profile_preserves_expected_structure():
    df = pd.DataFrame({
        "amount": [10.0, 20.0, np.nan],
        "device": ["mobile", "desktop", None],
    })

    schema = {
        "feature_names": ["amount", "device"],
        "numerical_columns": ["amount"],
        "categorical_columns": ["device"],
    }

    profile = build_reference_profile(
        df,
        schema,
        source="test.parquet",
        schema_source="test_schema.json",
    )

    assert profile["profile_version"] == 1
    assert profile["row_count"] == 3
    assert profile["feature_count"] == 2
    assert profile["numerical_feature_count"] == 1
    assert profile["categorical_feature_count"] == 1
    assert profile["source"] == "test.parquet"
    assert profile["numerical"]["amount"]["missing_rate"] == pytest.approx(1 / 3)
    assert profile["categorical"]["device"]["missing_rate"] == pytest.approx(1 / 3)


def test_save_reference_profile_creates_json(tmp_path):
    profile = {"profile_version": 1, "row_count": 10}
    output = tmp_path / "nested" / "profile.json"

    save_reference_profile(profile, output)

    assert output.exists()

    import json

    with output.open(encoding="utf-8") as file:
        assert json.load(file) == profile
