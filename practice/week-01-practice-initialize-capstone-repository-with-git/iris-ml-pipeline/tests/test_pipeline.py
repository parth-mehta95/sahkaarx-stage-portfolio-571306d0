"""Unit tests verifying pipeline components and conflict resolution stability."""

import os
import tempfile
import pandas as pd
import pytest

from src.data_loader import load_iris, preprocess
from src.evaluate import evaluate_model
from src.train import train_model


def test_load_iris_contract():
    """Verify load_iris returns proper structures and shapes."""
    X, y = load_iris(as_frame=True, return_X_y=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_preprocess_scaling_and_split():
    """Verify preprocessing correctly scales data and splits train/test sets."""
    X, y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = preprocess(X, y, test_size=0.2, random_state=42)

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert y_train.shape == (120,)
    assert y_test.shape == (30,)

    # Scaled training data should have approximately zero mean
    assert abs(X_train.mean().mean()) < 0.1


def test_train_model_logistic_regression():
    """Verify baseline model training and artifact generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "lr_model.pkl")
        model, metrics = train_model(
            model_type="logistic_regression",
            model_output_path=model_path,
        )
        assert os.path.exists(model_path)
        assert metrics["accuracy"] >= 0.90


def test_train_model_random_forest():
    """Verify ensemble model training and artifact generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "rf_model.pkl")
        model, metrics = train_model(
            model_type="random_forest",
            model_output_path=model_path,
        )
        assert os.path.exists(model_path)
        assert metrics["accuracy"] >= 0.90
