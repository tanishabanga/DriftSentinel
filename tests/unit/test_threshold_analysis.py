
import numpy as np
import pytest

from driftsentinel.policies.threshold_analysis import analyze_thresholds


def test_threshold_metrics_are_correct():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    result = analyze_thresholds(
        y_true,
        probabilities,
        thresholds=[0.5],
    )

    row = result.iloc[0]

    assert row["true_positives"] == 2
    assert row["false_positives"] == 0
    assert row["true_negatives"] == 2
    assert row["false_negatives"] == 0
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0


def test_higher_threshold_does_not_increase_flagged_count():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.6, 0.9])

    result = analyze_thresholds(
        y_true,
        probabilities,
        thresholds=[0.2, 0.5, 0.8],
    )

    counts = result["flagged_count"].tolist()

    assert counts == sorted(counts, reverse=True)


def test_invalid_probability_raises_error():
    y_true = np.array([0, 1])
    probabilities = np.array([0.2, 1.5])

    with pytest.raises(ValueError):
        analyze_thresholds(y_true, probabilities)


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        analyze_thresholds(
            y_true=[0, 1],
            probabilities=[0.2],
        )
