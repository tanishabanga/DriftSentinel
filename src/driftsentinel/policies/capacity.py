
"""Capacity-constrained fraud review policy."""

import numpy as np
import pandas as pd

from driftsentinel.policies.costs import DecisionCosts


def make_capacity_constrained_decisions(
    probabilities,
    review_capacity: int,
    costs: DecisionCosts | None = None,
) -> pd.DataFrame:
    """Choose decisions while respecting a manual-review capacity."""

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not isinstance(review_capacity, (int, np.integer)):
        raise TypeError("Review capacity must be an integer.")

    if review_capacity < 0:
        raise ValueError("Review capacity cannot be negative.")

    if costs is None:
        costs = DecisionCosts()

    p = probabilities

    approve_cost = (
        (1 - p) * costs.approve_legitimate
        + p * costs.approve_fraud
    )

    review_cost = (
        (1 - p) * costs.review_legitimate
        + p * costs.review_fraud
    )

    block_cost = (
        (1 - p) * costs.block_legitimate
        + p * costs.block_fraud
    )

    # Best alternative if manual review is unavailable.
    non_review_cost = np.minimum(approve_cost, block_cost)

    # Expected savings from reviewing instead of the best alternative.
    review_savings = non_review_cost - review_cost

    # Initially assign every transaction its cheapest non-review action.
    actions = np.where(
        approve_cost <= block_cost,
        "approve",
        "block",
    )

    # Only transactions with positive review savings are eligible.
    eligible_indices = np.flatnonzero(review_savings > 0)

    # Rank eligible transactions by expected savings, highest first.
    ranked_indices = eligible_indices[
        np.argsort(-review_savings[eligible_indices], kind="stable")
    ]

    # Allocate available review capacity.
    selected_indices = ranked_indices[:review_capacity]
    actions[selected_indices] = "review"

    selected_costs = np.where(
        actions == "approve",
        approve_cost,
        np.where(actions == "review", review_cost, block_cost),
    )

    return pd.DataFrame({
        "probability": p,
        "action": actions,
        "expected_cost": selected_costs,
        "review_savings": review_savings,
    })
