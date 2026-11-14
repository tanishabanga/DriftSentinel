
"""Metrics for evaluating probability calibration."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error using equal-width bins."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Inputs must have equal, non-zero length.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.minimum(
        np.digitize(probabilities, bin_edges[1:-1]),
        n_bins - 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index

        if not mask.any():
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        ece += mask.mean() * abs(accuracy - confidence)

    return float(ece)


def evaluate_calibration(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> dict:
    """Evaluate probability quality and ranking performance."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    # Reuse ECE validation.
    ece = expected_calibration_error(
        y_true,
        probabilities,
        n_bins=n_bins,
    )

    if len(np.unique(y_true)) < 2:
        raise ValueError("Both classes must be present for evaluation.")

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "ece": ece,
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
    }
