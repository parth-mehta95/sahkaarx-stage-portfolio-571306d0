"""
Unit tests for the Iris data validation functions module.

Verifies schema compliance, dimension verification, type adherence,
physical range checking, missing value guards, and error handling.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

try:
    from validation_functions import (
        DatasetMismatchError,
        FeatureValidationError,
        TargetValidationError,
        ValidationError,
        validate_dataset_pair,
        validate_iris_features,
        validate_prediction_probabilities,
        validate_target_labels,
        validate_train_test_split,
    )
except ImportError:
    from .validation_functions import (
        DatasetMismatchError,
        FeatureValidationError,
        TargetValidationError,
        ValidationError,
        validate_dataset_pair,
        validate_iris_features,
        validate_prediction_probabilities,
        validate_target_labels,
        validate_train_test_split,
    )


@pytest.fixture
def canonical_iris_data():
    """Fixture providing canonical scikit-learn Iris dataset."""
    X, y = load_iris(as_frame=True, return_X_y=True)
    return X, y


class TestValidateIrisFeatures:
    """Test suite for validate_iris_features."""

    def test_valid_dataframe_passes(self, canonical_iris_data):
        X, _ = canonical_iris_data
        result = validate_iris_features(X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (150, 4)

    def test_valid_ndarray_passes(self, canonical_iris_data):
        X, _ = canonical_iris_data
        X_arr = X.to_numpy()
        result = validate_iris_features(X_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (150, 4)

    def test_invalid_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected features as pd.DataFrame or np.ndarray"):
            validate_iris_features([[5.1, 3.5, 1.4, 0.2]])  # List instead of DF or ndarray

    def test_wrong_dimensions_raises_feature_error(self):
        # 1D array instead of 2D
        with pytest.raises(FeatureValidationError, match="must be 2-dimensional"):
            validate_iris_features(np.array([5.1, 3.5, 1.4, 0.2]))

    def test_wrong_column_count_raises_feature_error(self, canonical_iris_data):
        X, _ = canonical_iris_data
        # Drop one column
        with pytest.raises(FeatureValidationError, match="expected exactly 4 features"):
            validate_iris_features(X.iloc[:, :3])

    def test_nan_values_rejected_by_default(self, canonical_iris_data):
        X, _ = canonical_iris_data
        corrupted = X.copy()
        corrupted.iloc[0, 0] = np.nan
        with pytest.raises(FeatureValidationError, match="null/NaN"):
            validate_iris_features(corrupted)

    def test_infinite_values_rejected(self, canonical_iris_data):
        X, _ = canonical_iris_data
        corrupted = X.to_numpy().copy()
        corrupted[0, 0] = np.inf
        with pytest.raises(FeatureValidationError, match="non-finite"):
            validate_iris_features(corrupted)

    def test_negative_values_rejected(self, canonical_iris_data):
        X, _ = canonical_iris_data
        corrupted = X.copy()
        corrupted.iloc[0, 0] = -1.5
        with pytest.raises(FeatureValidationError, match="strictly positive"):
            validate_iris_features(corrupted)

    def test_non_numeric_dtype_rejected(self, canonical_iris_data):
        X, _ = canonical_iris_data
        corrupted = X.copy()
        corrupted["sepal length (cm)"] = "invalid_string"
        with pytest.raises(FeatureValidationError, match="non-numeric"):
            validate_iris_features(corrupted)

    def test_botany_physical_bounds(self, canonical_iris_data):
        X, _ = canonical_iris_data
        corrupted = X.copy()
        corrupted.iloc[0, 0] = 50.0  # Absurdly large sepal length (50cm)
        with pytest.raises(FeatureValidationError, match="outside physical bounds"):
            validate_iris_features(corrupted, check_bounds=True)


class TestValidateTargetLabels:
    """Test suite for validate_target_labels."""

    def test_valid_numeric_series_passes(self, canonical_iris_data):
        _, y = canonical_iris_data
        result = validate_target_labels(y)
        assert len(result) == 150

    def test_valid_string_labels_passes(self):
        labels = ["setosa", "versicolor", "virginica"] * 10
        result = validate_target_labels(labels)
        assert len(result) == 30

    def test_invalid_label_raises_error(self):
        corrupted = np.array([0, 1, 99])
        with pytest.raises(TargetValidationError, match="unrecognized class labels"):
            validate_target_labels(corrupted)

    def test_nan_target_raises_error(self):
        corrupted = pd.Series([0, 1, None])
        with pytest.raises(TargetValidationError, match="null or NaN"):
            validate_target_labels(corrupted)

    def test_2d_multiclass_target_rejected(self):
        corrupted = np.zeros((10, 3))  # One-hot encoded matrix
        with pytest.raises(TargetValidationError, match="1-dimensional"):
            validate_target_labels(corrupted)


class TestValidateDatasetPair:
    """Test suite for validate_dataset_pair."""

    def test_valid_pair_passes(self, canonical_iris_data):
        X, y = canonical_iris_data
        v_X, v_y = validate_dataset_pair(X, y)
        assert len(v_X) == len(v_y) == 150

    def test_mismatched_sample_counts_raises_error(self, canonical_iris_data):
        X, y = canonical_iris_data
        with pytest.raises(DatasetMismatchError, match="Sample count mismatch"):
            validate_dataset_pair(X, y.iloc[:100])


class TestValidateTrainTestSplit:
    """Test suite for validate_train_test_split."""

    def test_valid_split_passes(self, canonical_iris_data):
        X, y = canonical_iris_data
        X_train, X_test = X.iloc[:120], X.iloc[120:]
        y_train, y_test = y.iloc[:120], y.iloc[120:]
        # Note: Canonical iris is sorted, so make sure all classes exist in train
        y_train = pd.Series([0, 1, 2] * 40)
        y_test = pd.Series([0, 1, 2] * 10)
        assert validate_train_test_split(X_train, X_test, y_train, y_test) is True

    def test_unseen_class_in_test_raises_error(self, canonical_iris_data):
        X, _ = canonical_iris_data
        X_train, X_test = X.iloc[:100], X.iloc[100:]
        y_train = pd.Series([0, 1] * 50)
        y_test = pd.Series([2] * 50)  # Class 2 absent from train
        with pytest.raises(TargetValidationError, match="absent from training"):
            validate_train_test_split(X_train, X_test, y_train, y_test)


class TestValidatePredictionProbabilities:
    """Test suite for validate_prediction_probabilities."""

    def test_valid_probabilities_pass(self):
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.05, 0.05, 0.90],
        ])
        result = validate_prediction_probabilities(probs)
        assert result.shape == (3, 3)

    def test_row_sum_not_one_raises_error(self):
        corrupted = np.array([[0.5, 0.5, 0.5]])  # Sum is 1.5
        with pytest.raises(ValidationError, match="must equal 1.0"):
            validate_prediction_probabilities(corrupted)

    def test_out_of_range_raises_error(self):
        corrupted = np.array([[-0.1, 0.9, 0.2]])
        with pytest.raises(ValidationError, match="within the \\[0.0, 1.0\\] interval"):
            validate_prediction_probabilities(corrupted)
