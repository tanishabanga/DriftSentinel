
import pytest

from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)


def test_review_count_respects_capacity():
    probabilities = [0.01, 0.10, 0.30, 0.50, 0.80]

    result = make_capacity_constrained_decisions(
        probabilities,
        review_capacity=2,
    )

    assert (result["action"] == "review").sum() <= 2


def test_zero_capacity_produces_no_reviews():
    result = make_capacity_constrained_decisions(
        [0.01, 0.50, 0.80],
        review_capacity=0,
    )

    assert (result["action"] == "review").sum() == 0


def test_capacity_larger_than_dataset():
    probabilities = [0.01, 0.50, 0.80]

    result = make_capacity_constrained_decisions(
        probabilities,
        review_capacity=10,
    )

    assert (result["action"] == "review").sum() <= len(probabilities)


def test_reviewed_transactions_have_positive_savings():
    result = make_capacity_constrained_decisions(
        [0.01, 0.10, 0.30, 0.50, 0.80],
        review_capacity=5,
    )

    reviewed = result[result["action"] == "review"]

    assert (reviewed["review_savings"] > 0).all()


def test_invalid_negative_capacity():
    with pytest.raises(ValueError):
        make_capacity_constrained_decisions(
            [0.1, 0.5],
            review_capacity=-1,
        )


def test_invalid_non_integer_capacity():
    with pytest.raises(TypeError):
        make_capacity_constrained_decisions(
            [0.1, 0.5],
            review_capacity=1.5,
        )



def test_no_transactions_are_reviewed_when_review_is_never_beneficial():
    from driftsentinel.policies.costs import DecisionCosts

    costs = DecisionCosts(
        approve_legitimate=0,
        approve_fraud=1,
        review_legitimate=10,
        review_fraud=10,
        block_legitimate=10,
        block_fraud=0,
    )

    result = make_capacity_constrained_decisions(
        [0.1, 0.5, 0.9],
        review_capacity=3,
        costs=costs,
    )

    assert (result["action"] == "review").sum() == 0


def test_sufficient_capacity_reviews_every_eligible_transaction():
    result = make_capacity_constrained_decisions(
        [0.01, 0.10, 0.30, 0.50, 0.80],
        review_capacity=100,
    )

    eligible_count = (result["review_savings"] > 0).sum()
    reviewed_count = (result["action"] == "review").sum()

    assert reviewed_count == eligible_count
