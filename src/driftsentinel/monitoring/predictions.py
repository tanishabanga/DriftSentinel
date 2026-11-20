"""Aggregate prediction monitoring for DriftSentinel."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "raw_probability",
    "calibrated_probability",
    "action",
    "expected_cost",
}


def _validate_probability_column(series: pd.Series, column_name: str) -> np.ndarray:
    """Convert and validate a probability column."""
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column_name} must contain numeric values.") from exc

    if not np.isfinite(values).all():
        raise ValueError(f"{column_name} contains non-finite values.")

    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{column_name} must contain values between 0 and 1.")

    return values


def _distribution_summary(
    values: np.ndarray,
    bins: int,
) -> dict:
    """Summarize probability values and their fixed-width distribution."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)

    quantiles = {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": quantiles,
        "histogram": {
            "bin_edges": edges.tolist(),
            "counts": counts.astype(int).tolist(),
            "proportions": (counts / len(values)).astype(float).tolist(),
        },
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    probability_bins: int = 10,
) -> dict:
    """Create privacy-conscious aggregate monitoring metrics.

    The function consumes the output of FraudInferenceService.predict().
    It does not retain transaction IDs or individual prediction records.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")

    if predictions.empty:
        raise ValueError("Cannot summarize an empty prediction DataFrame.")

    if predictions.columns.duplicated().any():
        raise ValueError("Duplicate prediction columns detected.")

    if (
        isinstance(probability_bins, bool)
        or not isinstance(probability_bins, int)
        or probability_bins < 1
    ):
        raise ValueError("probability_bins must be a positive integer.")

    missing = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction output is missing required columns: {sorted(missing)}"
        )

    raw = _validate_probability_column(
        predictions["raw_probability"],
        "raw_probability",
    )
    calibrated = _validate_probability_column(
        predictions["calibrated_probability"],
        "calibrated_probability",
    )

    if len(raw) != len(calibrated):
        raise ValueError("Raw and calibrated prediction counts differ.")

    actions = predictions["action"]
    if actions.isna().any():
        raise ValueError("Action column contains missing values.")

    action_values = actions.astype(str)
    action_counts = action_values.value_counts().sort_index()
    action_proportions = action_counts / len(predictions)

    try:
        expected_cost = pd.to_numeric(
            predictions["expected_cost"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_cost must contain numeric values.") from exc

    if not np.isfinite(expected_cost).all():
        raise ValueError("expected_cost contains non-finite values.")

    return {
        "prediction_count": len(predictions),
        "raw_probability": _distribution_summary(raw, probability_bins),
        "calibrated_probability": _distribution_summary(
            calibrated,
            probability_bins,
        ),
        "mean_calibration_adjustment": float(
            np.mean(calibrated - raw)
        ),
        "action_counts": {
            str(action): int(count)
            for action, count in action_counts.items()
        },
        "action_proportions": {
            str(action): float(proportion)
            for action, proportion in action_proportions.items()
        },
        "expected_cost": {
            "mean": float(np.mean(expected_cost)),
            "median": float(np.median(expected_cost)),
            "p95": float(np.quantile(expected_cost, 0.95)),
            "min": float(np.min(expected_cost)),
            "max": float(np.max(expected_cost)),
        },
    }
