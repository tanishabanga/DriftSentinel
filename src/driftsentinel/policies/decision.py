
"""Cost-sensitive fraud decision engine."""

import numpy as np
import pandas as pd

from driftsentinel.policies.costs import DecisionCosts


def make_decisions(
    probabilities,
    costs: DecisionCosts | None = None,
) -> pd.DataFrame:
    """Choose the action with the lowest expected cost."""

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Probabilities must be a non-empty 1D array.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if costs is None:
        costs = DecisionCosts()

    p = probabilities

    expected_costs = {
        "approve": (
            (1 - p) * costs.approve_legitimate
            + p * costs.approve_fraud
        ),
        "review": (
            (1 - p) * costs.review_legitimate
            + p * costs.review_fraud
        ),
        "block": (
            (1 - p) * costs.block_legitimate
            + p * costs.block_fraud
        ),
    }

    cost_matrix = np.column_stack(
        list(expected_costs.values())
    )

    actions = np.array(["approve", "review", "block"])

    selected_indices = np.argmin(cost_matrix, axis=1)

    return pd.DataFrame({
        "probability": p,
        "action": actions[selected_indices],
        "expected_cost": cost_matrix[
            np.arange(len(p)), selected_indices
        ],
    })
