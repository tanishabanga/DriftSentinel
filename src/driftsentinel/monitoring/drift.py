"""Feature drift calculations using a saved reference profile."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_EPSILON = 1e-6


def calculate_psi(
    reference_proportions: Sequence[float],
    current_proportions: Sequence[float],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Calculate Population Stability Index for two bucket distributions."""
    reference = np.asarray(reference_proportions, dtype=np.float64)
    current = np.asarray(current_proportions, dtype=np.float64)

    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("Distributions must be one-dimensional.")
    if reference.size == 0 or current.size == 0:
        raise ValueError("Distributions must not be empty.")
    if reference.size != current.size:
        raise ValueError("Distributions must have the same number of buckets.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Distributions must contain only finite values.")
    if (reference < 0).any() or (current < 0).any():
        raise ValueError("Distribution proportions cannot be negative.")
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("Each distribution must have a positive total.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number.")

    reference = reference / reference.sum()
    current = current / current.sum()

    reference = np.clip(reference, epsilon, None)
    current = np.clip(current, epsilon, None)

    reference = reference / reference.sum()
    current = current / current.sum()

    psi = np.sum(
        (current - reference) * np.log(current / reference)
    )

    if not np.isfinite(psi):
        raise ValueError("PSI calculation produced a non-finite result.")

    return float(psi)


def _validate_numeric_profile(feature_profile: dict) -> tuple[list[float], list[float]]:
    """Validate and return numeric cut points and reference proportions."""
    if not isinstance(feature_profile, dict):
        raise TypeError("Numeric feature profile must be a dictionary.")

    required = {"cut_points", "bin_proportions", "missing_rate"}
    missing = required.difference(feature_profile)
    if missing:
        raise ValueError(f"Numeric profile is missing keys: {sorted(missing)}")

    cut_points = np.asarray(feature_profile["cut_points"], dtype=np.float64)
    bin_proportions = np.asarray(
        feature_profile["bin_proportions"], dtype=np.float64
    )
    missing_rate = feature_profile["missing_rate"]

    if cut_points.ndim != 1 or bin_proportions.ndim != 1:
        raise ValueError("Numeric cut points and proportions must be one-dimensional.")
    if bin_proportions.size != cut_points.size + 1:
        raise ValueError("Numeric profile must have exactly one more bin than cut points.")
    if not np.isfinite(cut_points).all():
        raise ValueError("Numeric cut points must be finite.")
    if cut_points.size and not np.all(np.diff(cut_points) > 0):
        raise ValueError("Numeric cut points must be strictly increasing.")
    if not np.isfinite(bin_proportions).all() or (bin_proportions < 0).any():
        raise ValueError("Numeric bin proportions must be finite and non-negative.")
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("Numeric missing_rate must be between 0 and 1.")

    reference = np.append(bin_proportions, float(missing_rate))
    if reference.sum() <= 0:
        raise ValueError("Numeric reference distribution has no observations.")

    return cut_points.tolist(), reference.tolist()


def _numeric_distribution(
    series: pd.Series,
    cut_points: list[float],
) -> tuple[list[float], float]:
    """Bin current numeric values using the reference cut points."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)
    valid_values = values[finite_mask]

    edges = np.asarray([-np.inf, *cut_points, np.inf], dtype=np.float64)
    counts, _ = np.histogram(valid_values, bins=edges)

    missing_count = int((~finite_mask).sum())
    total = len(series)

    if total == 0:
        raise ValueError("Cannot evaluate drift on an empty DataFrame.")

    proportions = counts.astype(np.float64) / total
    missing_rate = missing_count / total

    return np.append(proportions, missing_rate).tolist(), missing_rate


def _validate_categorical_profile(
    feature_profile: dict,
) -> tuple[list[str], list[float]]:
    """Validate and return category labels and reference proportions."""
    if not isinstance(feature_profile, dict):
        raise TypeError("Categorical feature profile must be a dictionary.")

    required = {"top_categories", "category_proportions", "other_proportion", "missing_rate"}
    missing = required.difference(feature_profile)
    if missing:
        raise ValueError(
            f"Categorical profile is missing keys: {sorted(missing)}"
        )

    top_categories = feature_profile["top_categories"]
    category_proportions = feature_profile["category_proportions"]

    if not isinstance(top_categories, dict) or not isinstance(category_proportions, dict):
        raise TypeError("Categorical profile categories and proportions must be dictionaries.")

    labels = list(top_categories.keys())
    if set(labels) != set(category_proportions.keys()):
        raise ValueError("Categorical counts and proportions have different categories.")

    proportions = [category_proportions[label] for label in labels]
    other_proportion = feature_profile["other_proportion"]
    missing_rate = feature_profile["missing_rate"]

    values = np.asarray(
        [*proportions, other_proportion, missing_rate],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Categorical proportions must be finite and non-negative.")
    if missing_rate > 1 or other_proportion > 1:
        raise ValueError("Categorical proportions cannot exceed 1.")
    if values.sum() <= 0:
        raise ValueError("Categorical reference distribution has no observations.")

    return labels, values.tolist()


def _categorical_distribution(
    series: pd.Series,
    labels: list[str],
) -> tuple[list[float], float]:
    """Count current known, other, and missing categories."""
    total = len(series)
    if total == 0:
        raise ValueError("Cannot evaluate drift on an empty DataFrame.")

    known = set(labels)
    counts = {label: 0 for label in labels}
    other_count = 0
    missing_count = 0

    for value in series:
        if pd.isna(value):
            missing_count += 1
            continue

        category = str(value)
        if category in known:
            counts[category] += 1
        else:
            other_count += 1

    proportions = [counts[label] / total for label in labels]
    proportions.extend([other_count / total, missing_count / total])

    return proportions, missing_count / total


def evaluate_feature_drift(
    df: pd.DataFrame,
    reference_profile: dict,
) -> pd.DataFrame:
    """Evaluate numerical and categorical feature drift against a reference."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot evaluate drift on an empty DataFrame.")
    if not isinstance(reference_profile, dict):
        raise TypeError("reference_profile must be a dictionary.")

    numerical = reference_profile.get("numerical")
    categorical = reference_profile.get("categorical")

    if not isinstance(numerical, dict) or not isinstance(categorical, dict):
        raise TypeError(
            "Reference profile must contain numerical and categorical dictionaries."
        )

    expected_features = set(numerical) | set(categorical)
    missing_features = expected_features.difference(df.columns)
    if missing_features:
        raise ValueError(
            f"DataFrame is missing reference features: {sorted(missing_features)}"
        )

    rows = []

    for feature, feature_profile in numerical.items():
        cut_points, reference = _validate_numeric_profile(feature_profile)
        current, current_missing_rate = _numeric_distribution(
            df[feature], cut_points
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "numerical",
                "psi": calculate_psi(reference, current),
                "reference_missing_rate": float(feature_profile["missing_rate"]),
                "current_missing_rate": current_missing_rate,
                "missing_rate_change": (
                    current_missing_rate - float(feature_profile["missing_rate"])
                ),
                "reference_rows": reference_profile.get("row_count"),
                "current_rows": len(df),
            }
        )

    for feature, feature_profile in categorical.items():
        labels, reference = _validate_categorical_profile(feature_profile)
        current, current_missing_rate = _categorical_distribution(
            df[feature], labels
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "psi": calculate_psi(reference, current),
                "reference_missing_rate": float(feature_profile["missing_rate"]),
                "current_missing_rate": current_missing_rate,
                "missing_rate_change": (
                    current_missing_rate - float(feature_profile["missing_rate"])
                ),
                "reference_rows": reference_profile.get("row_count"),
                "current_rows": len(df),
            }
        )

    return pd.DataFrame(rows)
