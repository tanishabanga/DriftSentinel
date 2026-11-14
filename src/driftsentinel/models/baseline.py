
"""Baseline fraud classification experiments."""

from time import perf_counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from driftsentinel.features.preprocessing import build_preprocessor

EXCLUDED_COLUMNS = [
    "isFraud",
    "TransactionID",
    "TransactionDT",
]


def build_logistic_regression_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a reusable Logistic Regression classification pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=None,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )



def train_logistic_regression(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """Train and evaluate a Logistic Regression baseline."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    X_train = train_df.drop(columns=EXCLUDED_COLUMNS)
    y_train = train_df["isFraud"].astype(int)

    X_validation = validation_df.drop(columns=EXCLUDED_COLUMNS)
    y_validation = validation_df["isFraud"].astype(int)

    pipeline = build_logistic_regression_pipeline(X_train)

    start_time = perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = perf_counter() - start_time

    start_time = perf_counter()
    probabilities = pipeline.predict_proba(X_validation)[:, 1]
    inference_time = perf_counter() - start_time

    predictions = (probabilities >= threshold).astype(int)

    results = {
        "model": "Logistic Regression",
        "threshold": threshold,
        "average_precision": average_precision_score(
            y_validation, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_validation, probabilities
        ),
        "precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_validation, predictions
        ).tolist(),
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "validation_rows": len(validation_df),
        "training_rows": len(train_df),
    }

    return {
        "pipeline": pipeline,
        "results": results,
        "probabilities": probabilities,
    }
