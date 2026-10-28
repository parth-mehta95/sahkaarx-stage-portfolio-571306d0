"""
Comprehensive Unit Test Suite for Iris Machine Learning Models Module.

Tests Decision Tree, Random Forest, and K-Means training, Hungarian bipartite
cluster label alignment, evaluation metric extraction, CSV metric export, and model
serialization.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Add repository root to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from models import (
    CLASS_NAMES,
    FEATURE_NAMES,
    ModelEvaluationResult,
    align_cluster_labels,
    evaluate_classification_model,
    evaluate_clustering_model,
    export_models,
    generate_comparison_dataframe,
    save_metrics_table,
    train_and_evaluate_all_models,
    train_decision_tree,
    train_kmeans,
    train_random_forest,
)


@pytest.fixture(scope="module")
def iris_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture providing standardized stratified train/test partitions of Iris."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


# ==============================================================================
# Model Training Tests
# ==============================================================================


def test_train_decision_tree_success(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify Decision Tree trains cleanly and achieves benchmark accuracy."""
    X_train, X_test, y_train, y_test = iris_data
    model, latency = train_decision_tree(X_train, y_train, max_depth=3, random_state=42)

    assert isinstance(model, DecisionTreeClassifier)
    assert hasattr(model, "tree_")
    assert latency > 0
    assert len(model.classes_) == 3

    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    acc = np.mean(preds == y_test)
    assert acc >= 0.90, f"Expected Decision Tree accuracy >= 0.90, got {acc}"


def test_train_random_forest_success(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify Random Forest trains ensemble and achieves benchmark accuracy."""
    X_train, X_test, y_train, y_test = iris_data
    model, latency = train_random_forest(X_train, y_train, n_estimators=50, max_depth=3, random_state=42)

    assert isinstance(model, RandomForestClassifier)
    assert len(model.estimators_) == 50
    assert latency > 0

    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    acc = np.mean(preds == y_test)
    assert acc >= 0.90, f"Expected Random Forest accuracy >= 0.90, got {acc}"


def test_train_kmeans_success(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify K-Means forms 3 distinct clusters with correct centroid geometry."""
    X_train, _, _, _ = iris_data
    model, latency = train_kmeans(X_train, n_clusters=3, random_state=42)

    assert isinstance(model, KMeans)
    assert model.cluster_centers_.shape == (3, 4)
    assert model.inertia_ > 0
    assert latency > 0


# ==============================================================================
# Cluster Alignment Tests (Hungarian / Bipartite Matching)
# ==============================================================================


def test_align_cluster_labels_identity() -> None:
    """Verify cluster alignment preserves identical labels."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])

    aligned, mapping = align_cluster_labels(y_true, cluster_labels)
    np.testing.assert_array_equal(aligned, y_true)
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_align_cluster_labels_permutation() -> None:
    """Verify Hungarian algorithm inverts arbitrary cluster permutations."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    cluster_labels = np.array([2, 2, 0, 0, 1, 1])

    aligned, mapping = align_cluster_labels(y_true, cluster_labels)
    np.testing.assert_array_equal(aligned, y_true)
    assert mapping == {2: 0, 0: 1, 1: 2}


def test_align_cluster_labels_error_handling() -> None:
    """Verify alignment raises ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        align_cluster_labels(np.array([]), np.array([]))

    with pytest.raises(ValueError, match="Length mismatch"):
        align_cluster_labels(np.array([0, 1]), np.array([0, 1, 2]))


# ==============================================================================
# Evaluation Metric Function Tests
# ==============================================================================


def test_evaluate_classification_model(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify classification evaluation yields complete metrics."""
    X_train, X_test, y_train, y_test = iris_data
    model, lat = train_decision_tree(X_train, y_train, random_state=42)
    res = evaluate_classification_model(model, X_test, y_test, model_name="DT-Test", train_time_ms=lat)

    assert isinstance(res, ModelEvaluationResult)
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.precision_macro <= 1.0
    assert 0.0 <= res.recall_macro <= 1.0
    assert 0.0 <= res.f1_macro <= 1.0
    assert len(res.confusion_matrix) == 3
    assert len(res.confusion_matrix[0]) == 3
    assert res.inference_time_ms >= 0


def test_evaluate_clustering_model(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify clustering evaluation yields both intrinsic and aligned extrinsic metrics."""
    X_train, X_test, _, y_test = iris_data
    km, lat = train_kmeans(X_train, n_clusters=3, random_state=42)
    res, mapping = evaluate_clustering_model(km, X_test, y_test, train_time_ms=lat)

    assert isinstance(res, ModelEvaluationResult)
    assert len(mapping) == 3
    assert res.accuracy >= 0.75  # Aligned accuracy on Iris test set
    assert "silhouette_score" in res.additional_metrics
    assert "adjusted_rand_index" in res.additional_metrics
    assert "v_measure" in res.additional_metrics


# ==============================================================================
# End-to-End Pipeline & Artifact Generation Tests
# ==============================================================================


def test_train_and_evaluate_all_models(iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verify orchestrator trains all 3 models and builds the comparison table."""
    X_train, X_test, y_train, y_test = iris_data
    models_dict, comp_df = train_and_evaluate_all_models(X_train, y_train, X_test, y_test, random_state=42)

    assert set(models_dict.keys()) == {"decision_tree", "random_forest", "kmeans"}
    assert comp_df.shape[0] == 3
    required_cols = [
        "Model",
        "Paradigm",
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1-Score (Macro)",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
    ]
    for col in required_cols:
        assert col in comp_df.columns, f"Missing required column: {col}"


def test_save_metrics_table_and_export_models(
    iris_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verify saving metrics to CSV and exporting trained models via joblib."""
    X_train, X_test, y_train, y_test = iris_data
    models_dict, comp_df = train_and_evaluate_all_models(X_train, y_train, X_test, y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "test_metrics.csv"
        save_metrics_table(comp_df, csv_file)

        assert csv_file.exists()
        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 3
        assert "Accuracy" in loaded_df.columns

        # Test model serialization
        export_dir = tmp_path / "models_out"
        saved = export_models(models_dict, export_dir)
        assert len(saved) == 3
        for name, file_path in saved.items():
            p = Path(file_path)
            assert p.exists()
            loaded_est = joblib.load(p)
            assert loaded_est is not None
