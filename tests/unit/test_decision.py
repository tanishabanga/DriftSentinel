
import numpy as np
import pytest

from driftsentinel.policies.costs import DecisionCosts
from driftsentinel.policies.decision import make_decisions


def test_low_probability_is_approved():
    result = make_decisions([0.01])

    assert result.iloc[0]["action"] == "approve"


def test_medium_probability_is_reviewed():
    result = make_decisions([0.50])

    assert result.iloc[0]["action"] == "review"


def test_high_probability_is_blocked():
    result = make_decisions([0.80])

    assert result.iloc[0]["action"] == "block"


def test_expected_cost_is_minimum():
    result = make_decisions([0.50])

    assert result.iloc[0]["expected_cost"] == pytest.approx(3.5)


def test_multiple_probabilities():
    result = make_decisions([0.01, 0.50, 0.80])

    assert result["action"].tolist() == [
        "approve",
        "review",
        "block",
    ]


@pytest.mark.parametrize(
    "probabilities",
    [
        [-0.1],
        [1.1],
        [np.nan],
        [np.inf],
        [],
        [[0.2, 0.8]],
    ],
)
def test_invalid_probabilities_raise_error(probabilities):
    with pytest.raises(ValueError):
        make_decisions(probabilities)


def test_negative_cost_raises_error():
    with pytest.raises(ValueError):
        DecisionCosts(approve_fraud=-10)



@pytest.mark.parametrize(
    "cost",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_cost_raises_error(cost):
    with pytest.raises(ValueError):
        DecisionCosts(approve_fraud=cost)
