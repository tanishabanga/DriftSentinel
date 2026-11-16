"""Exceptions used by the DriftSentinel inference service."""


class InferenceInputError(ValueError):
    """Raised when transaction input violates the inference contract."""
