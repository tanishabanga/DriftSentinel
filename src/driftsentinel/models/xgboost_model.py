
"""XGBoost model and preprocessing for fraud classification."""

import pandas as pd
from xgboost import XGBClassifier

from driftsentinel.features.preprocessing import (
    select_feature_columns,
)


def prepare_xgboost_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numerical and categorical features for XGBoost."""

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns must match."
        )

    numerical_columns, categorical_columns = select_feature_columns(
        X_train
    )

    # Convert numerical features to float32 to reduce memory usage.
    for column in numerical_columns:
        X_train[column] = X_train[column].astype("float32")
        X_validation[column] = X_validation[column].astype("float32")

    # Learn categorical vocabularies from training data only.
    for column in categorical_columns:
        train_values = X_train[column].astype("string")
        validation_values = X_validation[column].astype("string")

        categories = train_values.dropna().unique()

        X_train[column] = pd.Categorical(
            train_values,
            categories=categories,
        )

        X_validation[column] = pd.Categorical(
            validation_values,
            categories=categories,
        )

    return X_train, X_validation


def build_xgboost_classifier() -> XGBClassifier:
    """Build a reproducible CPU-based XGBoost classifier."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        max_bin=256,
        scale_pos_weight=1,
        n_jobs=2,
        random_state=42,
        early_stopping_rounds=50,
        verbosity=1,
    )
