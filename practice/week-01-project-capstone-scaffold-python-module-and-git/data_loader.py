"""
Data loader and preprocessing module for ML pipelines.

Provides type-hinted, reusable functions for loading and preprocessing
the Iris dataset for downstream machine learning tasks.
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

__all__ = ["load_iris", "preprocess"]


def load_iris(
    as_frame: bool = True,
    return_X_y: bool = True,
) -> Union[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[np.ndarray, np.ndarray],
    pd.DataFrame,
    Bunch,
]:
    """Load the classic Iris dataset with complete type annotations.

    Parameters
    ----------
    as_frame : bool, default=True
        If True, returns the dataset using pandas DataFrame and Series objects.
        If False, returns NumPy ndarrays (when return_X_y=True) or a scikit-learn
        Bunch container (when return_X_y=False).
    return_X_y : bool, default=True
        If True, returns a (features, target) tuple.
        If False, returns a unified container:
        - When as_frame=True: A pandas DataFrame containing all 4 feature columns
          plus the 'target' column.
        - When as_frame=False: The scikit-learn Bunch object.

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray], pd.DataFrame, Bunch]
        - If return_X_y=True and as_frame=True:
            Tuple[pd.DataFrame, pd.Series] containing features (150, 4) and target labels (150,).
        - If return_X_y=True and as_frame=False:
            Tuple[np.ndarray, np.ndarray] containing features array and target array.
        - If return_X_y=False and as_frame=True:
            pd.DataFrame of shape (150, 5) with feature columns and target column.
        - If return_X_y=False and as_frame=False:
            sklearn.utils.Bunch object with dataset attributes.

    Examples
    --------
    >>> # Default usage: loads X as DataFrame, y as Series
    >>> X, y = load_iris()
    >>> type(X), type(y)
    (<class 'pandas.core.frame.DataFrame'>, <class 'pandas.core.series.Series'>)

    >>> # Unified DataFrame
    >>> df = load_iris(return_X_y=False)
    >>> df.shape
    (150, 5)
    """
    if return_X_y:
        return sklearn_load_iris(as_frame=as_frame, return_X_y=True)

    bunch: Bunch = sklearn_load_iris(as_frame=as_frame, return_X_y=False)
    if as_frame:
        return bunch.frame
    return bunch


def preprocess(
    data: Union[pd.DataFrame, np.ndarray],
    target: Optional[Union[pd.Series, np.ndarray]] = None,
    scaler: Optional[StandardScaler] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = 42,
) -> Union[
    pd.DataFrame,
    np.ndarray,
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]],
    Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ],
]:
    """Preprocess Iris dataset features by applying standard scaling.

    Standardizes features to zero mean and unit variance using StandardScaler.
    Preserves pandas DataFrame and Series formatting (including column names and
    index) when pandas objects are supplied as input.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input feature matrix to be scaled, or combined DataFrame containing
        features and an optional 'target' column.
    target : Optional[Union[pd.Series, np.ndarray]], default=None
        Target label vector corresponding to the feature matrix.
        If provided along with test_size, stratified train-test splitting is performed.
        If provided without test_size, a tuple (scaled_features, target) is returned.
    scaler : Optional[StandardScaler], default=None
        An optional pre-configured StandardScaler instance. If None, a new
        StandardScaler is initialized and fitted.
    test_size : Optional[float], default=None
        Proportion of the dataset to allocate to the test split (e.g. 0.2).
        If provided along with target, splits and standardizes data without data leakage.
        If None, scales the entire input feature set.
    random_state : Optional[int], default=42
        Random seed used for reproducible train-test splitting.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray, Tuple[...], Tuple[...]]
        - If test_size is provided and target is provided:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test).
        - If target is provided (and test_size is None):
            Tuple of (scaled_features, target).
        - If target is None (and test_size is None):
            Scaled feature matrix matching the input type (pd.DataFrame or np.ndarray).
            If a DataFrame containing 'target' column was passed, returns DataFrame with
            features scaled and 'target' preserved.

    Examples
    --------
    >>> X, y = load_iris()
    >>> # 1. Scale feature matrix only
    >>> X_scaled = preprocess(X)
    >>> type(X_scaled)
    <class 'pandas.core.frame.DataFrame'>

    >>> # 2. Scale features and keep target paired
    >>> X_scaled, y = preprocess(X, y)

    >>> # 3. Split and scale train/test sets
    >>> X_train, X_test, y_train, y_test = preprocess(X, y, test_size=0.2)
    """
    if scaler is None:
        scaler = StandardScaler()

    # Case 1: Train-Test Split requested
    if test_size is not None and target is not None:
        stratify_target = target if hasattr(target, "__len__") else None
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )

        X_train_scaled_arr: np.ndarray = scaler.fit_transform(X_train)
        X_test_scaled_arr: np.ndarray = scaler.transform(X_test)

        if isinstance(data, pd.DataFrame):
            X_train_scaled = pd.DataFrame(
                X_train_scaled_arr,
                columns=data.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                X_test_scaled_arr,
                columns=data.columns,
                index=X_test.index,
            )
        else:
            X_train_scaled = X_train_scaled_arr
            X_test_scaled = X_test_scaled_arr

        return X_train_scaled, X_test_scaled, y_train, y_test

    # Case 2: Target supplied without split -> return (scaled_data, target)
    if target is not None:
        scaled_arr: np.ndarray = scaler.fit_transform(data)
        if isinstance(data, pd.DataFrame):
            scaled_features = pd.DataFrame(
                scaled_arr,
                columns=data.columns,
                index=data.index,
            )
        else:
            scaled_features = scaled_arr
        return scaled_features, target

    # Case 3: Target is None, data is a DataFrame that might have a 'target' column
    if isinstance(data, pd.DataFrame) and "target" in data.columns:
        feature_cols = [col for col in data.columns if col != "target"]
        scaled_feature_arr: np.ndarray = scaler.fit_transform(data[feature_cols])
        result_df = pd.DataFrame(
            scaled_feature_arr,
            columns=feature_cols,
            index=data.index,
        )
        result_df["target"] = data["target"]
        return result_df

    # Case 4: Pure feature matrix scaling
    scaled_matrix: np.ndarray = scaler.fit_transform(data)
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(
            scaled_matrix,
            columns=data.columns,
            index=data.index,
        )

    return scaled_matrix


if __name__ == "__main__":
    print("=== Testing load_iris() ===")
    X_df, y_ser = load_iris()
    print(f"X type: {type(X_df)}, shape: {X_df.shape}")
    print(f"y type: {type(y_ser)}, shape: {y_ser.shape}")
    print(f"Features: {list(X_df.columns)}")

    print("\n=== Testing preprocess(X) ===")
    X_scaled = preprocess(X_df)
    print(f"X_scaled type: {type(X_scaled)}, shape: {X_scaled.shape}")
    print(f"Feature means (should be ~0): {X_scaled.mean().round(4).to_dict()}")
    print(f"Feature stds (should be ~1): {X_scaled.std().round(4).to_dict()}")

    print("\n=== Testing preprocess(X, y) ===")
    X_s, y_s = preprocess(X_df, y_ser)
    print(f"Scaled features shape: {X_s.shape}, target shape: {y_s.shape}")

    print("\n=== Testing preprocess(X, y, test_size=0.2) ===")
    X_tr, X_te, y_tr, y_te = preprocess(X_df, y_ser, test_size=0.2)
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")
    print(f"y_train: {y_tr.shape}, y_test: {y_te.shape}")
    print("\nAll data loader checks passed successfully!")
