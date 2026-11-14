
"""Threshold analysis for binary fraud classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def analyze_thresholds(
    y_true,
    probabilities,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    """Compare binary classification metrics across probability thresholds."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(list(thresholds), dtype=float)

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

    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("Provide at least one threshold.")

    if not np.isfinite(thresholds).all():
        raise ValueError("Thresholds must be finite.")

    if ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Thresholds must be between 0 and 1.")

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        true_positives = int(((y_true == 1) & (predictions == 1)).sum())
        false_positives = int(((y_true == 0) & (predictions == 1)).sum())
        true_negatives = int(((y_true == 0) & (predictions == 0)).sum())
        false_negatives = int(((y_true == 1) & (predictions == 0)).sum())

        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "flagged_count": int(predictions.sum()),
            "flagged_rate": float(predictions.mean()),
        })

    return pd.DataFrame(rows)
