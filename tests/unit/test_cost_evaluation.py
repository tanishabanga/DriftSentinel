
import numpy as np
import pytest

from driftsentinel.policies.cost_evaluation import (
    calculate_realized_cost,
    evaluate_cost_policy,
)
from driftsentinel.policies.costs import DecisionCosts


@pytest.fixture
def costs():
    return DecisionCosts()


def test_calculate_realized_cost_uses_actual_labels(costs):
    y_true = np.array([0, 1, 0])
    actions = np.array(["approve", "review", "block"])

    result = calculate_realized_cost(y_true, actions, costs)

    np.testing.assert_array_equal(result, [0, 5, 10])


def test_unconstrained_policy_calculates_both_costs(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.50, 0.99])

    result = evaluate_cost_policy(
        y_true,
        probabilities,
        costs,
    )

    assert result["policy"] == "unconstrained"
    assert result["total_expected_cost"] == pytest.approx(4.6)
    assert result["total_realized_cost"] == pytest.approx(15.0)
    assert result["review_count"] == 1
    assert result["review_utilization"] is None


def test_capacity_policy_respects_review_limit(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.50, 0.99])

    result = evaluate_cost_policy(
        y_true,
        probabilities,
        costs,
        review_capacity=1,
    )

    assert result["review_count"] <= 1
    assert result["review_count"] == 1
    assert result["review_utilization"] == pytest.approx(1.0)


def test_zero_capacity_prevents_manual_reviews(costs):
    y_true = np.array([0, 1, 0])
    probabilities = np.array([0.01, 0.50, 0.99])

    result = evaluate_cost_policy(
        y_true,
        probabilities,
        costs,
        review_capacity=0,
    )

    assert result["review_count"] == 0
    assert result["review_utilization"] == 0


def test_invalid_labels_are_rejected(costs):
    with pytest.raises(ValueError):
        evaluate_cost_policy(
            np.array([0, 2, 1]),
            np.array([0.1, 0.5, 0.9]),
            costs,
        )


def test_invalid_actions_are_rejected(costs):
    with pytest.raises(ValueError):
        calculate_realized_cost(
            np.array([0, 1]),
            np.array(["approve", "unknown"]),
            costs,
        )
