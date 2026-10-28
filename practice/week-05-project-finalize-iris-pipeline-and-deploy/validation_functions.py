"""
Validation functions module for the Iris ML pipeline.

Provides strict, type-hinted validation utilities ensuring schema compliance,
numeric range validity, missing value prevention, and dataset pair consistency
for both continuous feature matrices and categorical target vectors.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

__all__ = [
    "ValidationError",
    "FeatureValidationError",
    "TargetValidationError",
    "DatasetMismatchError",
    "validate_iris_features",
    "validate_feature_matrix",
    "validate_target_labels",
    "validate_target_vector",
    "validate_dataset_pair",
    "validate_train_test_split",
    "validate_prediction_probabilities",
]

# Convenience Aliases
validate_feature_matrix = validate_iris_features
validate_target_vector = validate_target_labels
validate_iris_dataframe = validate_iris_features

# Canonical Iris Dataset Constants
EXPECTED_IRIS_FEATURE_COUNT: int = 4
EXPECTED_IRIS_FEATURE_NAMES: Tuple[str, ...] = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)
EXPECTED_IRIS_NUMERIC_CLASSES: Set[int] = {0, 1, 2}
EXPECTED_IRIS_STRING_CLASSES: Set[str] = {"setosa", "versicolor", "virginica"}

# Plausible physical dimension bounds for Iris species (in centimeters)
FEATURE_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "sepal length (cm)": (3.5, 9.0),
    "sepal width (cm)": (1.5, 5.0),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.5),
}


class ValidationError(ValueError):
    """Base exception for all pipeline data validation failures."""
    pass


class FeatureValidationError(ValidationError):
    """Raised when feature matrix schema, dimension, or value checks fail."""
    pass


class TargetValidationError(ValidationError):
    """Raised when target vector labels, shape, or class distributions fail."""
    pass


class DatasetMismatchError(ValidationError):
    """Raised when features and targets have inconsistent sample lengths."""
    pass


def validate_iris_features(
    features: Union[pd.DataFrame, np.ndarray],
    expected_dim: int = EXPECTED_IRIS_FEATURE_COUNT,
    allow_nan: bool = False,
    min_samples: int = 1,
    check_bounds: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """Validate Iris feature matrix dimensions, data types, and value sanity.

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Input feature matrix, either as a pandas DataFrame or 2D NumPy ndarray.
    expected_dim : int, default=4
        Expected number of feature columns (dimensions). Default is 4 for Iris.
    allow_nan : bool, default=False
        If False, rejects data containing NaN, None, or infinite values.
    min_samples : int, default=1
        Minimum number of observation rows required.
    check_bounds : bool, default=False
        If True and features is a DataFrame with standard column names,
        verifies that values lie within realistic physical botany boundaries.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The validated feature matrix, identical to input.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame or NumPy ndarray.
    FeatureValidationError
        If dimensions, sample count, null values, or numeric bounds fail checks.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> X, _ = load_iris(as_frame=True, return_X_y=True)
    >>> X_valid = validate_iris_features(X)
    >>> X_valid.shape
    (150, 4)
    """
    # 1. Type validation
    if not isinstance(features, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"Expected features as pd.DataFrame or np.ndarray, got {type(features).__name__}"
        )

    # 2. Dimensionality check (must be 2D matrix)
    if features.ndim != 2:
        raise FeatureValidationError(
            f"Feature matrix must be 2-dimensional (samples, features). Received ndim={features.ndim}."
        )

    num_samples, num_features = features.shape

    # 3. Minimum sample count check
    if num_samples < min_samples:
        raise FeatureValidationError(
            f"Feature matrix contains {num_samples} samples, but minimum required is {min_samples}."
        )

    # 4. Column count check
    if num_features != expected_dim:
        raise FeatureValidationError(
            f"Feature matrix has {num_features} columns; expected exactly {expected_dim} features."
        )

    # 5. Missing and infinite value check
    if not allow_nan:
        if isinstance(features, pd.DataFrame):
            if features.isnull().any().any():
                null_cols = features.columns[features.isnull().any()].tolist()
                raise FeatureValidationError(
                    f"Feature DataFrame contains null/NaN values in columns: {null_cols}"
                )
            if not np.isfinite(features.to_numpy(dtype=float)).all():
                raise FeatureValidationError("Feature DataFrame contains non-finite (infinite) values.")
        else:
            if not np.isfinite(features).all():
                raise FeatureValidationError(
                    "Feature NumPy ndarray contains NaN or non-finite (infinite) values."
                )

    # 6. Numeric dtype check & non-negativity
    if isinstance(features, pd.DataFrame):
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                raise FeatureValidationError(
                    f"Feature column '{col}' has non-numeric dtype: {features[col].dtype}"
                )
        arr_to_check = features.to_numpy(dtype=float)
    else:
        if not np.issubdtype(features.dtype, np.number):
            raise FeatureValidationError(
                f"Feature ndarray has non-numeric dtype: {features.dtype}"
            )
        arr_to_check = features

    if (arr_to_check <= 0).any():
        raise FeatureValidationError(
            "Feature values must be strictly positive measurements (> 0.0) for flower morphology."
        )

    # 7. Optional botany range verification
    if check_bounds and isinstance(features, pd.DataFrame):
        for col, (low, high) in FEATURE_PHYSICAL_BOUNDS.items():
            if col in features.columns:
                out_of_bounds = (features[col] < low) | (features[col] > high)
                if out_of_bounds.any():
                    num_violations = int(out_of_bounds.sum())
                    raise FeatureValidationError(
                        f"Column '{col}' has {num_violations} values outside physical bounds [{low}, {high}]."
                    )

    return features


def validate_target_labels(
    target: Union[pd.Series, np.ndarray, Sequence[Any]],
    expected_classes: Optional[Set[Any]] = None,
    min_samples: int = 1,
) -> Union[pd.Series, np.ndarray]:
    """Validate Iris classification target labels, class domain, and 1D shape.

    Parameters
    ----------
    target : Union[pd.Series, np.ndarray, Sequence[Any]]
        Target label array or series representing class indices or class names.
    expected_classes : Optional[Set[Any]], default=None
        Set of valid allowed classes (e.g. {0, 1, 2} or {"setosa", "versicolor", "virginica"}).
        If None, automatically permits {0, 1, 2} or {"setosa", "versicolor", "virginica"}.
    min_samples : int, default=1
        Minimum number of target observations required.

    Returns
    -------
    Union[pd.Series, np.ndarray]
        The validated target series or array.

    Raises
    ------
    TypeError
        If target is not a pd.Series, np.ndarray, list, or tuple.
    TargetValidationError
        If target contains null values, wrong dimensions, or unrecognized classes.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 2, 1, 0])
    >>> y_valid = validate_target_labels(y)
    >>> len(y_valid)
    5
    """
    # 1. Type validation
    if not isinstance(target, (pd.Series, np.ndarray, list, tuple)):
        raise TypeError(
            f"Expected target as pd.Series, np.ndarray, or sequence, got {type(target).__name__}"
        )

    if isinstance(target, (list, tuple)):
        target = np.asarray(target)

    # 2. Dimensionality check (must be 1D)
    if isinstance(target, pd.Series):
        ndim = 1
        num_samples = len(target)
    else:
        ndim = target.ndim
        num_samples = target.shape[0] if ndim > 0 else 0

    if ndim != 1:
        # Allow (N, 1) column vectors by flattening or reject
        if ndim == 2 and target.shape[1] == 1:
            target = target.ravel()
            num_samples = target.shape[0]
        else:
            raise TargetValidationError(
                f"Target must be a 1-dimensional array/series of labels. Received ndim={ndim}."
            )

    # 3. Minimum sample count
    if num_samples < min_samples:
        raise TargetValidationError(
            f"Target contains {num_samples} samples; minimum required is {min_samples}."
        )

    # 4. Null value check
    if isinstance(target, pd.Series):
        if target.isnull().any():
            raise TargetValidationError("Target Series contains null or NaN values.")
        unique_labels = set(target.unique())
    else:
        if pd.isna(target).any():
            raise TargetValidationError("Target array contains null or NaN values.")
        unique_labels = set(np.unique(target))

    # 5. Class domain validation
    if expected_classes is None:
        # If numeric, check {0, 1, 2}; if string, check {'setosa', 'versicolor', 'virginica'}
        is_numeric = all(isinstance(x, (int, np.integer)) for x in unique_labels)
        if is_numeric:
            expected_classes = EXPECTED_IRIS_NUMERIC_CLASSES
        else:
            expected_classes = EXPECTED_IRIS_STRING_CLASSES

    invalid_labels = unique_labels - expected_classes
    if invalid_labels:
        raise TargetValidationError(
            f"Target contains unrecognized class labels: {invalid_labels}. "
            f"Allowed classes are: {expected_classes}."
        )

    return target


def validate_dataset_pair(
    features: Union[pd.DataFrame, np.ndarray],
    target: Union[pd.Series, np.ndarray, Sequence[Any]],
    expected_dim: int = EXPECTED_IRIS_FEATURE_COUNT,
    min_samples: int = 1,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Validate alignment and integrity between feature matrix and target labels.

    Ensures both inputs satisfy individual schema constraints and share an
    identical number of observation instances.

    Parameters
    ----------
    features : Union[pd.DataFrame, np.ndarray]
        Input feature matrix.
    target : Union[pd.Series, np.ndarray, Sequence[Any]]
        Target label array or series.
    expected_dim : int, default=4
        Expected number of feature columns.
    min_samples : int, default=1
        Minimum number of instances required.

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]
        Tuple containing the validated (features, target) objects.

    Raises
    ------
    DatasetMismatchError
        If row count of features does not match length of target.
    """
    valid_features = validate_iris_features(
        features=features,
        expected_dim=expected_dim,
        min_samples=min_samples,
    )
    valid_target = validate_target_labels(
        target=target,
        min_samples=min_samples,
    )

    len_x = len(valid_features)
    len_y = len(valid_target)

    if len_x != len_y:
        raise DatasetMismatchError(
            f"Sample count mismatch: features matrix has {len_x} rows, "
            f"but target vector has {len_y} labels."
        )

    return valid_features, valid_target


def validate_train_test_split(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    min_train_ratio: float = 0.5,
) -> bool:
    """Validate partitioned train and test datasets for schema and class consistency.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training feature partition.
    X_test : Union[pd.DataFrame, np.ndarray]
        Test feature partition.
    y_train : Union[pd.Series, np.ndarray]
        Training labels.
    y_test : Union[pd.Series, np.ndarray]
        Test labels.
    min_train_ratio : float, default=0.5
        Minimum expected ratio of train samples to total samples.

    Returns
    -------
    bool
        True if all train-test consistency checks pass.

    Raises
    ------
    ValidationError
        If split proportions, feature dimensions, or target classes are mismatched.
    """
    # 1. Validate both partitions independently
    validate_dataset_pair(X_train, y_train)
    validate_dataset_pair(X_test, y_test)

    # 2. Check column dimension consistency
    if X_train.shape[1] != X_test.shape[1]:
        raise FeatureValidationError(
            f"Train feature count ({X_train.shape[1]}) differs from test feature count ({X_test.shape[1]})."
        )

    # 3. Check split ratio
    total_samples = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total_samples
    if train_ratio < min_train_ratio:
        raise ValidationError(
            f"Training partition ratio ({train_ratio:.2f}) is below minimum threshold ({min_train_ratio:.2f})."
        )

    # 4. Check class representation (test classes should not have unseen labels)
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    unseen_classes = test_classes - train_classes
    if unseen_classes:
        raise TargetValidationError(
            f"Test partition contains classes absent from training partition: {unseen_classes}"
        )

    return True


def validate_prediction_probabilities(
    probabilities: Union[np.ndarray, pd.DataFrame],
    num_classes: int = 3,
    tol: float = 1e-4,
) -> np.ndarray:
    """Validate predicted class probability matrices output by ML models.

    Verifies 2D matrix shape, value bounds in [0.0, 1.0], and row sum closure to 1.0.

    Parameters
    ----------
    probabilities : Union[np.ndarray, pd.DataFrame]
        Probability distribution matrix (samples, classes).
    num_classes : int, default=3
        Expected number of class probability outputs.
    tol : float, default=1e-4
        Tolerance for row sum equality to 1.0.

    Returns
    -------
    np.ndarray
        Validated probability matrix as a NumPy ndarray.

    Raises
    ------
    ValidationError
        If shape, individual bounds, or row normalization criteria fail.
    """
    if isinstance(probabilities, pd.DataFrame):
        probs_arr = probabilities.to_numpy(dtype=float)
    elif isinstance(probabilities, np.ndarray):
        probs_arr = probabilities.astype(float)
    else:
        raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(probabilities).__name__}")

    if probs_arr.ndim != 2:
        raise ValidationError(
            f"Probability array must be 2D (samples, classes); received ndim={probs_arr.ndim}."
        )

    if probs_arr.shape[1] != num_classes:
        raise ValidationError(
            f"Probability matrix has {probs_arr.shape[1]} columns; expected {num_classes} classes."
        )

    if (probs_arr < 0.0).any() or (probs_arr > 1.0).any():
        raise ValidationError("Probabilities must strictly fall within the [0.0, 1.0] interval.")

    row_sums = probs_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise ValidationError(
            f"Probability row sums must equal 1.0; maximum deviation observed was {max_deviation:.6f}."
        )

    return probs_arr


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print("==========================================================")
    print(" Executing Iris Pipeline Validation Functions Sanity Check ")
    print("==========================================================")

    # 1. Test canonical Iris data
    raw_X, raw_y = load_iris(as_frame=True, return_X_y=True)
    valid_X, valid_y = validate_dataset_pair(raw_X, raw_y)
    print(f"[✓] validate_dataset_pair passed: X shape {valid_X.shape}, y shape {valid_y.shape}")

    # 2. Test feature bounds check
    validate_iris_features(valid_X, check_bounds=True)
    print("[✓] validate_iris_features botany bounds check passed")

    # 3. Test string class validation
    target_names = np.array(["setosa", "versicolor", "virginica"] * 50)
    validate_target_labels(target_names)
    print("[✓] validate_target_labels string labels check passed")

    # 4. Demonstrate expected error catching
    print("\nDemonstrating error handling on corrupt inputs:")
    try:
        corrupted_features = valid_X.iloc[:, :3]  # Only 3 features instead of 4
        validate_iris_features(corrupted_features)
    except FeatureValidationError as e:
        print(f"[✓] Caught expected dimension error: {e}")

    try:
        mismatched_target = valid_y.iloc[:100]  # Only 100 rows instead of 150
        validate_dataset_pair(valid_X, mismatched_target)
    except DatasetMismatchError as e:
        print(f"[✓] Caught expected sample mismatch error: {e}")

    try:
        invalid_labels = np.array([0, 1, 99])  # Label 99 is invalid
        validate_target_labels(invalid_labels)
    except TargetValidationError as e:
        print(f"[✓] Caught expected invalid class label error: {e}")

    print("\nAll validation function sanity tests passed successfully!")
