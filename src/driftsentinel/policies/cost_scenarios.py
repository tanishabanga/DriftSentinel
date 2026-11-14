
"""Illustrative cost scenarios for cost-sensitivity experiments."""

from driftsentinel.policies.costs import DecisionCosts

COST_SCENARIOS: dict[str, DecisionCosts] = {
    "baseline": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=100,
        review_legitimate=2,
        review_fraud=5,
        block_legitimate=10,
        block_fraud=0,
    ),

    "higher_fraud_loss": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=200,
        review_legitimate=2,
        review_fraud=5,
        block_legitimate=10,
        block_fraud=0,
    ),

    "severe_fraud_loss": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=500,
        review_legitimate=2,
        review_fraud=5,
        block_legitimate=10,
        block_fraud=0,
    ),

    "higher_review_expense": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=100,
        review_legitimate=5,
        review_fraud=8,
        block_legitimate=10,
        block_fraud=0,
    ),

    "expensive_review": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=100,
        review_legitimate=10,
        review_fraud=13,
        block_legitimate=10,
        block_fraud=0,
    ),

    "higher_false_block_cost": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=100,
        review_legitimate=2,
        review_fraud=5,
        block_legitimate=25,
        block_fraud=0,
    ),

    "severe_false_block_cost": DecisionCosts(
        approve_legitimate=0,
        approve_fraud=100,
        review_legitimate=2,
        review_fraud=5,
        block_legitimate=50,
        block_fraud=0,
    ),
}
