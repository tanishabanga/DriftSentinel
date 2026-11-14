
"""Evaluation utilities for cost-sensitive fraud decisions."""

import numpy as np

from driftsentinel.policies.capacity import (
    make_capacity_constrained_decisions,
)
from driftsentinel.policies.costs import DecisionCosts
from driftsentinel.policies.decision import make_decisions


def calculate_realized_cost(
    y_true,
    actions,
    costs: DecisionCosts,
) -> np.ndarray:
    """Calculate realized cost using actual labels and selected actions."""

    y_true = np.asarray(y_true)
    actions = np.asarray(actions)

    if y_true.ndim != 1 or actions.ndim != 1:
        raise ValueError("Labels and actions must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(actions):
        raise ValueError("Labels and actions must have equal, non-zero length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    valid_actions = {"approve", "review", "block"}

    if not np.isin(actions, list(valid_actions)).all():
        raise ValueError("Actions must be approve, review, or block.")

    realized_costs = {
        "approve": np.where(
            y_true == 0,
            costs.approve_legitimate,
            costs.approve_fraud,
        ),
        "review": np.where(
            y_true == 0,
            costs.review_legitimate,
            costs.review_fraud,
        ),
        "block": np.where(
            y_true == 0,
            costs.block_legitimate,
            costs.block_fraud,
        ),
    }

    return np.select(
        [actions == "approve", actions == "review", actions == "block"],
        [
            realized_costs["approve"],
            realized_costs["review"],
            realized_costs["block"],
        ],
    )


def evaluate_cost_policy(
    y_true,
    probabilities,
    costs: DecisionCosts,
    review_capacity: int | None = None,
) -> dict:
    """Evaluate expected and realized costs for a decision policy."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal, non-zero length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if review_capacity is None:
        decisions = make_decisions(probabilities, costs)
        policy_name = "unconstrained"
    else:
        decisions = make_capacity_constrained_decisions(
            probabilities,
            review_capacity,
            costs,
        )
        policy_name = "capacity_constrained"

    actions = decisions["action"].to_numpy()

    realized_costs = calculate_realized_cost(
        y_true,
        actions,
        costs,
    )

    fraud = y_true == 1
    legitimate = y_true == 0

    review = actions == "review"
    approve = actions == "approve"
    block = actions == "block"

    total_transactions = len(y_true)
    fraud_count = int(fraud.sum())
    legitimate_count = int(legitimate.sum())
    review_count = int(review.sum())

    return {
        "policy": policy_name,
        "review_capacity": review_capacity,
        "total_transactions": total_transactions,
        "total_expected_cost": float(decisions["expected_cost"].sum()),
        "total_realized_cost": float(realized_costs.sum()),
        "realized_cost_per_transaction": float(realized_costs.mean()),
        "fraud_approved": int((fraud & approve).sum()),
        "fraud_reviewed": int((fraud & review).sum()),
        "fraud_blocked": int((fraud & block).sum()),
        "legitimate_approved": int((legitimate & approve).sum()),
        "legitimate_reviewed": int((legitimate & review).sum()),
        "legitimate_blocked": int((legitimate & block).sum()),
        "review_count": review_count,
        "review_rate": float(review.mean()),
        "review_utilization": (
            (
                float(review_count / review_capacity)
                if review_capacity > 0
                else 0.0
            )
            if review_capacity is not None
            else None
        ),
        "fraud_approval_rate": (
            float((fraud & approve).sum() / fraud_count)
            if fraud_count else 0.0
        ),
        "fraud_intervention_rate": (
            float((fraud & (review | block)).sum() / fraud_count)
            if fraud_count else 0.0
        ),
        "legitimate_block_rate": (
            float((legitimate & block).sum() / legitimate_count)
            if legitimate_count else 0.0
        ),
    }
