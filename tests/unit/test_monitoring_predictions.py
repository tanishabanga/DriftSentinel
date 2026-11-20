import numpy as np
import pandas as pd
import pytest

from driftsentinel.monitoring.predictions import summarize_predictions


@pytest.fixture
def prediction_frame():
    return pd.DataFrame(
        {
            "TransactionID": [101, 102, 103, 104],
            "raw_probability": [0.05, 0.20, 0.70, 0.95],
            "calibrated_probability": [0.04, 0.18, 0.75, 0.96],
            "action": ["approve", "approve", "review", "block"],
            "expected_cost": [0.2, 1.0, 2.5, 0.1],
        }
    )


def test_prediction_summary_has_expected_structure(prediction_frame):
    result = summarize_predictions(prediction_frame)

    assert result["prediction_count"] == 4
    assert "raw_probability" in result
    assert "calibrated_probability" in result
    assert "action_counts" in result
    assert "action_proportions" in result
    assert "expected_cost" in result


def test_prediction_summary_counts_actions(prediction_frame):
    result = summarize_predictions(prediction_frame)

    assert result["action_counts"] == {
        "approve": 2,
        "block": 1,
        "review": 1,
    }


def test_action_proportions_sum_to_one(prediction_frame):
    result = summarize_predictions(prediction_frame)

    assert sum(result["action_proportions"].values()) == pytest.approx(1.0)


def test_probability_histogram_counts_all_predictions(prediction_frame):
    result = summarize_predictions(prediction_frame)

    histogram = result["calibrated_probability"]["histogram"]

    assert sum(histogram["counts"]) == 4
    assert sum(histogram["proportions"]) == pytest.approx(1.0)


def test_probability_histogram_includes_zero_and_one():
    predictions = pd.DataFrame(
        {
            "raw_probability": [0.0, 1.0],
            "calibrated_probability": [0.0, 1.0],
            "action": ["approve", "block"],
            "expected_cost": [0.0, 1.0],
        }
    )

    result = summarize_predictions(predictions)

    assert sum(result["raw_probability"]["histogram"]["counts"]) == 2


def test_mean_calibration_adjustment(prediction_frame):
    result = summarize_predictions(prediction_frame)

    expected = np.mean(
        prediction_frame["calibrated_probability"]
        - prediction_frame["raw_probability"]
    )

    assert result["mean_calibration_adjustment"] == pytest.approx(expected)


def test_expected_cost_summary(prediction_frame):
    result = summarize_predictions(prediction_frame)

    assert result["expected_cost"]["mean"] == pytest.approx(0.95)
    assert result["expected_cost"]["min"] == pytest.approx(0.1)
    assert result["expected_cost"]["max"] == pytest.approx(2.5)


def test_summary_does_not_include_transaction_ids(prediction_frame):
    result = summarize_predictions(prediction_frame)

    assert "TransactionID" not in result
    assert "transactions" not in result


def test_custom_probability_bin_count(prediction_frame):
    result = summarize_predictions(prediction_frame, probability_bins=5)

    assert len(
        result["raw_probability"]["histogram"]["counts"]
    ) == 5


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, None])
def test_invalid_probability_bin_count(prediction_frame, bins):
    with pytest.raises(ValueError):
        summarize_predictions(prediction_frame, probability_bins=bins)


def test_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        summarize_predictions(pd.DataFrame())


def test_rejects_missing_required_columns():
    predictions = pd.DataFrame(
        {"raw_probability": [0.2]}
    )

    with pytest.raises(ValueError, match="missing required columns"):
        summarize_predictions(predictions)


@pytest.mark.parametrize(
    "probability",
    [
        np.nan,
        np.inf,
        -0.1,
        1.1,
    ],
)
def test_rejects_invalid_raw_probabilities(probability):
    predictions = pd.DataFrame(
        {
            "raw_probability": [probability],
            "calibrated_probability": [0.2],
            "action": ["approve"],
            "expected_cost": [1.0],
        }
    )

    with pytest.raises(ValueError):
        summarize_predictions(predictions)


def test_rejects_missing_actions():
    predictions = pd.DataFrame(
        {
            "raw_probability": [0.2],
            "calibrated_probability": [0.2],
            "action": [None],
            "expected_cost": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        summarize_predictions(predictions)


def test_rejects_non_finite_expected_cost():
    predictions = pd.DataFrame(
        {
            "raw_probability": [0.2],
            "calibrated_probability": [0.2],
            "action": ["approve"],
            "expected_cost": [np.inf],
        }
    )

    with pytest.raises(ValueError, match="non-finite"):
        summarize_predictions(predictions)


def test_rejects_duplicate_columns():
    predictions = pd.DataFrame(
        [[0.2, 0.3, "approve", 1.0]],
        columns=[
            "raw_probability",
            "raw_probability",
            "action",
            "expected_cost",
        ],
    )

    with pytest.raises(ValueError, match="Duplicate"):
        summarize_predictions(predictions)


def test_rejects_non_dataframe_input():
    with pytest.raises(TypeError):
        summarize_predictions([0.2, 0.3])
