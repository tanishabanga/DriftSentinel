import numpy as np
import pandas as pd
import pytest

from driftsentinel.monitoring.drift import (
    calculate_psi,
    evaluate_feature_drift,
)


def test_psi_is_zero_for_identical_distributions():
    result = calculate_psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    assert result == pytest.approx(0.0)


def test_psi_is_positive_for_shifted_distributions():
    result = calculate_psi([0.8, 0.2], [0.2, 0.8])
    assert result > 0


def test_psi_handles_zero_frequency_buckets():
    result = calculate_psi([1.0, 0.0], [0.0, 1.0])
    assert np.isfinite(result)
    assert result > 0


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        ([], []),
        ([0.5, 0.5], [1.0]),
        ([0.5, -0.5], [0.5, 0.5]),
        ([np.nan, 1.0], [0.5, 0.5]),
        ([np.inf, 1.0], [0.5, 0.5]),
        ([0.0, 0.0], [0.5, 0.5]),
    ],
)
def test_psi_rejects_invalid_distributions(reference, current):
    with pytest.raises(ValueError):
        calculate_psi(reference, current)


def test_psi_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        calculate_psi([0.5, 0.5], [0.5, 0.5], epsilon=0)


def test_evaluate_feature_drift_for_numeric_and_categorical_features():
    profile = {
        "row_count": 4,
        "numerical": {
            "amount": {
                "cut_points": [10.0],
                "bin_proportions": [0.5, 0.5],
                "missing_rate": 0.0,
            }
        },
        "categorical": {
            "channel": {
                "top_categories": {"web": 2, "app": 2},
                "category_proportions": {"web": 0.5, "app": 0.5},
                "other_proportion": 0.0,
                "missing_rate": 0.0,
            }
        },
    }
    current = pd.DataFrame(
        {
            "amount": [1.0, 5.0, 12.0, 15.0],
            "channel": ["web", "app", "web", "app"],
        }
    )

    result = evaluate_feature_drift(current, profile)

    assert set(result["feature"]) == {"amount", "channel"}
    assert set(result["feature_type"]) == {"numerical", "categorical"}
    assert result["psi"].max() == pytest.approx(0.0)
    assert (result["current_missing_rate"] == 0).all()


def test_numeric_drift_uses_reference_cut_points():
    profile = {
        "row_count": 4,
        "numerical": {
            "amount": {
                "cut_points": [10.0],
                "bin_proportions": [0.5, 0.5],
                "missing_rate": 0.0,
            }
        },
        "categorical": {},
    }
    current = pd.DataFrame({"amount": [20.0, 21.0, 22.0, 23.0]})

    result = evaluate_feature_drift(current, profile)

    assert result.loc[0, "psi"] > 0


def test_numeric_non_finite_values_are_counted_as_missing():
    profile = {
        "row_count": 4,
        "numerical": {
            "amount": {
                "cut_points": [10.0],
                "bin_proportions": [0.5, 0.5],
                "missing_rate": 0.0,
            }
        },
        "categorical": {},
    }
    current = pd.DataFrame({"amount": [1.0, 2.0, np.inf, np.nan]})

    result = evaluate_feature_drift(current, profile)

    assert result.loc[0, "current_missing_rate"] == pytest.approx(0.5)
    assert result.loc[0, "missing_rate_change"] == pytest.approx(0.5)


def test_categorical_unknown_values_are_counted_as_other():
    profile = {
        "row_count": 4,
        "numerical": {},
        "categorical": {
            "channel": {
                "top_categories": {"web": 2, "app": 2},
                "category_proportions": {"web": 0.5, "app": 0.5},
                "other_proportion": 0.0,
                "missing_rate": 0.0,
            }
        },
    }
    current = pd.DataFrame({"channel": ["web", "new", "new", None]})

    result = evaluate_feature_drift(current, profile)

    assert result.loc[0, "current_missing_rate"] == pytest.approx(0.25)
    assert result.loc[0, "psi"] > 0


def test_evaluate_rejects_missing_reference_features():
    profile = {
        "row_count": 1,
        "numerical": {
            "amount": {
                "cut_points": [],
                "bin_proportions": [1.0],
                "missing_rate": 0.0,
            }
        },
        "categorical": {},
    }

    with pytest.raises(ValueError, match="missing reference features"):
        evaluate_feature_drift(pd.DataFrame({"other": [1]}), profile)


def test_evaluate_rejects_empty_dataframe():
    profile = {"numerical": {}, "categorical": {}}

    with pytest.raises(ValueError, match="empty DataFrame"):
        evaluate_feature_drift(pd.DataFrame(), profile)


def test_evaluate_rejects_invalid_numeric_profile():
    profile = {
        "numerical": {
            "amount": {
                "cut_points": [10.0],
                "bin_proportions": [1.0],
                "missing_rate": 0.0,
            }
        },
        "categorical": {},
    }

    with pytest.raises(ValueError, match="one more bin"):
        evaluate_feature_drift(pd.DataFrame({"amount": [1.0]}), profile)
