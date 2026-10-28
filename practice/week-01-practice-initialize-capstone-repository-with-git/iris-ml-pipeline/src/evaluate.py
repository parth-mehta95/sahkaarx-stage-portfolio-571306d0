"""
Model evaluation module for Iris ML Pipeline.

Computes classification metrics including accuracy, precision, recall, and F1-score.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

__all__ = ["evaluate_model", "format_evaluation_report"]


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, float]:
    """Evaluate a trained classifier against a test dataset.

    Parameters
    ----------
    model : Any
        Fitted scikit-learn estimator implementing `predict(X)`.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples,)
        True target labels.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision (macro), recall (macro), and f1 (macro).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def format_evaluation_report(
    model: Any,
    X_test: Any,
    y_test: Any,
    target_names: Any = None,
) -> str:
    """Generate a formatted classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)
