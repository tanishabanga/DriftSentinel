
import numpy as np
import pytest

from driftsentinel.policies.costs import DecisionCosts
from scripts.analyze_action_calibration import analyze_action_calibration


@pytest.fixture
def costs():
    return DecisionCosts()


def test_action_calibration_creates_three_action_groups(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.10, 0.90])

    calibration_rows, cost_rows = analyze_action_calibration(
        y_true,
        probabilities,
        "test_strategy",
        costs,
    )

    assert len(calibration_rows) == 3
    assert len(cost_rows) == 3

    assert {
        row["action"] for row in calibration_rows
    } == {"approve", "review", "block"}


def test_action_calibration_calculates_observed_and_predicted_rates(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.10, 0.90])

    calibration_rows, _ = analyze_action_calibration(
        y_true,
        probabilities,
        "test_strategy",
        costs,
    )

    results = {
        row["action"]: row for row in calibration_rows
    }

    assert results["approve"]["mean_predicted_risk"] == pytest.approx(0.01)
    assert results["approve"]["observed_fraud_rate"] == pytest.approx(0.0)

    assert results["review"]["mean_predicted_risk"] == pytest.approx(0.10)
    assert results["review"]["observed_fraud_rate"] == pytest.approx(1.0)

    assert results["block"]["mean_predicted_risk"] == pytest.approx(0.90)
    assert results["block"]["observed_fraud_rate"] == pytest.approx(0.0)


def test_empty_action_groups_return_nan_rates(costs):
    y_true = np.array([0, 1])
    probabilities = np.array([0.01, 0.02])

    calibration_rows, cost_rows = analyze_action_calibration(
        y_true,
        probabilities,
        "test_strategy",
        costs,
    )

    results = {
        row["action"]: row for row in calibration_rows
    }

    assert results["review"]["transactions"] == 0
    assert np.isnan(results["review"]["mean_predicted_risk"])
    assert np.isnan(results["review"]["observed_fraud_rate"])
    assert np.isnan(results["review"]["calibration_gap"])

    cost_results = {
        row["action"]: row for row in cost_rows
    }

    assert cost_results["review"]["expected_cost"] == 0.0
    assert cost_results["review"]["realized_cost"] == 0.0
    assert cost_results["review"]["independent_gap_contribution"] == 0.0


def test_cost_gap_matches_independent_contribution(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.10, 0.90])

    _, cost_rows = analyze_action_calibration(
        y_true,
        probabilities,
        "test_strategy",
        costs,
    )

    for row in cost_rows:
        assert row["cost_gap"] == pytest.approx(
            row["independent_gap_contribution"]
        )


def test_custom_cost_matrix_is_used():
    costs = DecisionCosts(
        approve_fraud=200,
        review_legitimate=4,
        review_fraud=9,
        block_legitimate=20,
        block_fraud=0,
    )

    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.005, 0.10, 0.90])

    _, cost_rows = analyze_action_calibration(
        y_true,
        probabilities,
        "custom_cost_test",
        costs,
    )

    for row in cost_rows:
        assert row["cost_gap"] == pytest.approx(
            row["independent_gap_contribution"]
        )


def test_invalid_probabilities_are_rejected(costs):
    with pytest.raises(ValueError):
        analyze_action_calibration(
            np.array([0, 1]),
            np.array([0.1, 1.5]),
            "invalid_test",
            costs,
        )


def test_invalid_labels_are_rejected(costs):
    with pytest.raises(ValueError):
        analyze_action_calibration(
            np.array([0, 2]),
            np.array([0.1, 0.5]),
            "invalid_test",
            costs,
        )
