
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
