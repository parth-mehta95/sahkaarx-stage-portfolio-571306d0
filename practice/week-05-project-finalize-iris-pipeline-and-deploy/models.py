"""
Machine Learning Model Training, Evaluation, and Comparison Module.

Provides type-hinted, production-grade functions to train, evaluate, and benchmark
supervised classifiers (Decision Trees, Random Forest) and unsupervised clustering
(K-Means) on the Iris dataset. Includes optimal bipartite matching (Hungarian
algorithm) to map unsupervised cluster assignments to ground-truth labels for direct
metric comparison (Accuracy, Precision, Recall, F1-score).
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    homogeneity_completeness_v_measure,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

__all__ = [
    "ModelEvaluationResult",
    "train_decision_tree",
    "train_random_forest",
    "train_kmeans",
    "align_cluster_labels",
    "evaluate_classification_model",
    "evaluate_clustering_model",
    "train_and_evaluate_all_models",
    "generate_comparison_dataframe",
    "save_metrics_table",
    "export_models",
    "CLASS_NAMES",
    "FEATURE_NAMES",
]

# Canonical Iris Botanical Metadata
CLASS_NAMES: Tuple[str, ...] = ("setosa", "versicolor", "virginica")
FEATURE_NAMES: Tuple[str, ...] = (
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
)


@dataclass
class ModelEvaluationResult:
    """Container for model performance metrics and training metadata."""

    model_name: str
    paradigm: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    train_time_ms: float
    inference_time_ms: float
    hyperparameters: Dict[str, Any]
    confusion_matrix: List[List[int]]
    additional_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation results to dictionary format."""
        return asdict(self)


def align_cluster_labels(
    y_true: Union[pd.Series, np.ndarray],
    cluster_labels: Union[pd.Series, np.ndarray],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Align unsupervised cluster labels to true ground-truth class labels.

    Uses the Kuhn-Munkres (Hungarian) algorithm on the negative contingency matrix
    to solve the maximum weight bipartite matching problem. This establishes an
    optimal, mathematically sound 1-to-1 mapping from arbitrary cluster IDs to
    botanical target classes, enabling direct calculation of supervised classification
    metrics (Accuracy, Precision, Recall, F1).

    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth class targets (integers 0, 1, ..., K-1).
    cluster_labels : Union[pd.Series, np.ndarray]
        Cluster labels predicted by an unsupervised clustering algorithm.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, int]]
        - aligned_labels: np.ndarray of cluster labels mapped to true class space.
        - mapping: Dict mapping each raw cluster ID to its assigned ground-truth label.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths or contain empty sequences.
    """
    y_true_arr = np.asarray(y_true).ravel()
    cluster_arr = np.asarray(cluster_labels).ravel()

    if y_true_arr.size == 0 or cluster_arr.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if y_true_arr.size != cluster_arr.size:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.size} elements, "
            f"cluster_labels has {cluster_arr.size} elements."
        )

    unique_clusters = np.unique(cluster_arr)
    unique_targets = np.unique(y_true_arr)
    n_clusters = max(len(unique_clusters), len(unique_targets))

    # Construct contingency matrix: rows = cluster ID, columns = true class ID
    contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for c, t in zip(cluster_arr, y_true_arr):
        if c < n_clusters and t < n_clusters:
            contingency[c, t] += 1

    # Solve linear sum assignment for maximum overlap (negate matrix for minimization)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    # Map clusters
    aligned_labels = np.array([mapping.get(int(c), int(c)) for c in cluster_arr], dtype=np.int64)
    return aligned_labels, mapping


def train_decision_tree(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    max_depth: Optional[int] = 3,
    criterion: str = "gini",
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[DecisionTreeClassifier, float]:
    """Train a scikit-learn Decision Tree Classifier on the training split.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training feature matrix of shape (n_samples, n_features).
    y_train : Union[pd.Series, np.ndarray]
        Training target vector of shape (n_samples,).
    max_depth : Optional[int], default=3
        Maximum tree depth to prevent over-specialization and overfitting.
    criterion : str, default='gini'
        Function to measure split quality ('gini' or 'entropy').
    random_state : int, default=42
        Determines random number generation for reproducibility.
    **kwargs : Any
        Additional keyword arguments forwarded to DecisionTreeClassifier.

    Returns
    -------
    Tuple[DecisionTreeClassifier, float]
        The fitted estimator and training latency in milliseconds.
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs,
    )
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return model, train_latency_ms


def train_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[RandomForestClassifier, float]:
    """Train a scikit-learn Random Forest Classifier ensemble.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training feature matrix of shape (n_samples, n_features).
    y_train : Union[pd.Series, np.ndarray]
        Training target vector of shape (n_samples,).
    n_estimators : int, default=100
        The number of trees in the forest ensemble.
    max_depth : Optional[int], default=3
        Maximum depth of each decision tree estimator.
    random_state : int, default=42
        Random state seed ensuring reproducible bootstrap sampling.
    **kwargs : Any
        Additional keyword arguments forwarded to RandomForestClassifier.

    Returns
    -------
    Tuple[RandomForestClassifier, float]
        The fitted ensemble estimator and training latency in milliseconds.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs,
    )
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return model, train_latency_ms


def train_kmeans(
    X: Union[pd.DataFrame, np.ndarray],
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    **kwargs: Any,
) -> Tuple[KMeans, float]:
    """Train an unsupervised K-Means clustering algorithm.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int, default=3
        The number of clusters to form (matching 3 Iris species).
    random_state : int, default=42
        Determines random number generation for centroid initialization.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    **kwargs : Any
        Additional keyword arguments forwarded to KMeans.

    Returns
    -------
    Tuple[KMeans, float]
        The fitted clustering model and fit latency in milliseconds.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        **kwargs,
    )
    start_time = time.perf_counter()
    model.fit(X)
    train_latency_ms = (time.perf_counter() - start_time) * 1000.0
    return model, train_latency_ms


def evaluate_classification_model(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model_name: str,
    train_time_ms: float = 0.0,
    paradigm: str = "Supervised",
) -> ModelEvaluationResult:
    """Evaluate a supervised classification model on test data.

    Computes Accuracy, Macro/Weighted Precision, Macro/Weighted Recall,
    Macro/Weighted F1-score, inference latency, and the confusion matrix.

    Parameters
    ----------
    model : Union[DecisionTreeClassifier, RandomForestClassifier]
        Fitted scikit-learn classification estimator.
    X_test : Union[pd.DataFrame, np.ndarray]
        Test feature matrix.
    y_test : Union[pd.Series, np.ndarray]
        True class labels for the test partition.
    model_name : str
        Human-readable name of the evaluated model.
    train_time_ms : float, default=0.0
        Model training duration in milliseconds.
    paradigm : str, default='Supervised'
        Learning paradigm description.

    Returns
    -------
    ModelEvaluationResult
        Structured container with all calculated metrics.
    """
    start_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_ms = (time.perf_counter() - start_inf) * 1000.0

    acc = float(accuracy_score(y_test, y_pred))
    p_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    p_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()

    params: Dict[str, Any] = {}
    if hasattr(model, "get_params"):
        all_params = model.get_params()
        keys_of_interest = ["criterion", "max_depth", "n_estimators", "min_samples_split"]
        params = {k: all_params[k] for k in keys_of_interest if k in all_params}

    return ModelEvaluationResult(
        model_name=model_name,
        paradigm=paradigm,
        accuracy=acc,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f1_m,
        precision_weighted=p_weighted,
        recall_weighted=r_weighted,
        f1_weighted=f1_w,
        train_time_ms=round(train_time_ms, 3),
        inference_time_ms=round(inference_time_ms, 3),
        hyperparameters=params,
        confusion_matrix=cm,
        additional_metrics={},
    )


def evaluate_clustering_model(
    model: KMeans,
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.Series, np.ndarray],
    model_name: str = "K-Means (Cluster-Aligned)",
    train_time_ms: float = 0.0,
) -> Tuple[ModelEvaluationResult, Dict[int, int]]:
    """Evaluate an unsupervised K-Means clustering model.

    Computes intrinsic clustering metrics (Silhouette Score, Inertia) and
    extrinsic comparison metrics (Adjusted Rand Index, Homogeneity, Completeness,
    V-Measure). Furthermore, aligns cluster assignments to ground-truth labels
    via optimal Hungarian bipartite matching to produce directly comparable
    Accuracy, Precision, Recall, and F1-score.

    Parameters
    ----------
    model : KMeans
        Fitted scikit-learn KMeans estimator.
    X : Union[pd.DataFrame, np.ndarray]
        Feature matrix used for evaluation.
    y_true : Union[pd.Series, np.ndarray]
        Ground-truth target labels.
    model_name : str, default='K-Means (Cluster-Aligned)'
        Human-readable name of the evaluated model.
    train_time_ms : float, default=0.0
        Model training duration in milliseconds.

    Returns
    -------
    Tuple[ModelEvaluationResult, Dict[int, int]]
        Evaluation result container and the optimal cluster-to-label mapping dictionary.
    """
    start_inf = time.perf_counter()
    raw_clusters = model.predict(X)
    inference_time_ms = (time.perf_counter() - start_inf) * 1000.0

    aligned_preds, mapping = align_cluster_labels(y_true, raw_clusters)

    acc = float(accuracy_score(y_true, aligned_preds))
    p_macro = float(precision_score(y_true, aligned_preds, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, aligned_preds, average="macro", zero_division=0))
    f1_m = float(f1_score(y_true, aligned_preds, average="macro", zero_division=0))

    p_weighted = float(precision_score(y_true, aligned_preds, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_true, aligned_preds, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, aligned_preds, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, aligned_preds).tolist()

    # Intrinsic & Extrinsic Clustering metrics
    sil = float(silhouette_score(X, raw_clusters))
    ari = float(adjusted_rand_score(y_true, raw_clusters))
    homo, comp, v_meas = homogeneity_completeness_v_measure(y_true, raw_clusters)

    extra_metrics = {
        "silhouette_score": round(sil, 4),
        "adjusted_rand_index": round(ari, 4),
        "homogeneity": round(float(homo), 4),
        "completeness": round(float(comp), 4),
        "v_measure": round(float(v_meas), 4),
        "inertia": round(float(model.inertia_), 2),
    }

    params = {
        "n_clusters": model.n_clusters,
        "n_init": model.n_init,
        "max_iter": model.max_iter,
    }

    eval_result = ModelEvaluationResult(
        model_name=model_name,
        paradigm="Unsupervised (Bipartite Aligned)",
        accuracy=acc,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f1_m,
        precision_weighted=p_weighted,
        recall_weighted=r_weighted,
        f1_weighted=f1_w,
        train_time_ms=round(train_time_ms, 3),
        inference_time_ms=round(inference_time_ms, 3),
        hyperparameters=params,
        confusion_matrix=cm,
        additional_metrics=extra_metrics,
    )

    return eval_result, mapping


def train_and_evaluate_all_models(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train and evaluate Decision Tree, Random Forest, and K-Means models.

    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training features.
    y_train : Union[pd.Series, np.ndarray]
        Training targets.
    X_test : Union[pd.DataFrame, np.ndarray]
        Testing features.
    y_test : Union[pd.Series, np.ndarray]
        Testing targets.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        Dictionary with trained models and evaluation summaries, plus comparison DataFrame.
    """
    # 1. Decision Tree
    dt_model, dt_train_time = train_decision_tree(
        X_train, y_train, max_depth=3, random_state=random_state
    )
    dt_eval = evaluate_classification_model(
        dt_model, X_test, y_test, model_name="Decision Tree", train_time_ms=dt_train_time
    )

    # 2. Random Forest
    rf_model, rf_train_time = train_random_forest(
        X_train, y_train, n_estimators=100, max_depth=3, random_state=random_state
    )
    rf_eval = evaluate_classification_model(
        rf_model, X_test, y_test, model_name="Random Forest", train_time_ms=rf_train_time
    )

    # 3. K-Means (Evaluated on full dataset or test partition for rigorous comparison)
    # Fit K-Means on training features and evaluate on test features with true labels
    km_model, km_train_time = train_kmeans(
        X_train, n_clusters=3, random_state=random_state
    )
    km_eval, km_mapping = evaluate_clustering_model(
        km_model, X_test, y_test, model_name="K-Means (Cluster-Aligned)", train_time_ms=km_train_time
    )

    results = {
        "decision_tree": {"model": dt_model, "eval": dt_eval},
        "random_forest": {"model": rf_model, "eval": rf_eval},
        "kmeans": {"model": km_model, "eval": km_eval, "cluster_mapping": km_mapping},
    }

    comparison_df = generate_comparison_dataframe([dt_eval, rf_eval, km_eval])
    return results, comparison_df


def generate_comparison_dataframe(
    eval_results: Sequence[ModelEvaluationResult],
) -> pd.DataFrame:
    """Generate a clean pandas DataFrame summarizing model benchmarks.

    Parameters
    ----------
    eval_results : Sequence[ModelEvaluationResult]
        Collection of evaluation results.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    records: List[Dict[str, Any]] = []
    for res in eval_results:
        rec = {
            "Model": res.model_name,
            "Paradigm": res.paradigm,
            "Accuracy": round(res.accuracy, 4),
            "Precision (Macro)": round(res.precision_macro, 4),
            "Recall (Macro)": round(res.recall_macro, 4),
            "F1-Score (Macro)": round(res.f1_macro, 4),
            "Precision (Weighted)": round(res.precision_weighted, 4),
            "Recall (Weighted)": round(res.recall_weighted, 4),
            "F1-Score (Weighted)": round(res.f1_weighted, 4),
            "Train Latency (ms)": res.train_time_ms,
            "Inference Latency (ms)": res.inference_time_ms,
        }
        if "adjusted_rand_index" in res.additional_metrics:
            rec["ARI"] = res.additional_metrics["adjusted_rand_index"]
            rec["Silhouette"] = res.additional_metrics["silhouette_score"]
        else:
            rec["ARI"] = "N/A"
            rec["Silhouette"] = "N/A"
        records.append(rec)

    return pd.DataFrame(records)


def save_metrics_table(
    df: pd.DataFrame,
    filepath: Union[str, Path] = "metrics.csv",
) -> None:
    """Save model performance comparison table to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Comparison metrics DataFrame.
    filepath : Union[str, Path], default='metrics.csv'
        Destination filepath.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_models(
    models_dict: Dict[str, Any],
    output_dir: Union[str, Path] = "models",
) -> Dict[str, str]:
    """Serialize trained model estimators using joblib.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Dictionary mapping model names to model instances.
    output_dir : Union[str, Path], default='models'
        Output directory for serialized artifacts.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names to saved artifact file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    for name, item in models_dict.items():
        estimator = item["model"] if isinstance(item, dict) and "model" in item else item
        dest = out_path / f"{name}.joblib"
        joblib.dump(estimator, dest)
        saved_paths[name] = str(dest)

    return saved_paths


def main() -> None:
    """Execute end-to-end Iris model training, evaluation, and serialization pipeline."""
    print("=" * 70)
    print(" 🌸 Iris Machine Learning Model Training & Comparative Evaluation ")
    print("=" * 70)

    # 1. Load Iris data
    iris = sklearn_load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = iris.target

    print(f"\n[+] Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes.")

    # 2. Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Partitioned data: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples (stratified).")

    # 3. Standard scaling (fit on train, transform test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train and benchmark all 3 models
    print("\n[+] Training Decision Tree, Random Forest, and K-Means models...")
    models_dict, comparison_df = train_and_evaluate_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test, random_state=42
    )

    print("\n[✓] Performance Benchmark Summary Table:")
    print("-" * 70)
    print(comparison_df.to_string(index=False))
    print("-" * 70)

    # 5. Export metrics to CSV
    metrics_csv_path = Path(__file__).resolve().parent / "metrics.csv"
    save_metrics_table(comparison_df, metrics_csv_path)
    print(f"\n[✓] Saved comparison metrics to: {metrics_csv_path}")

    # 6. Export models
    models_dir = Path(__file__).resolve().parent / "models"
    saved_paths = export_models(models_dict, models_dir)
    print(f"[✓] Serialized trained model binaries to: {models_dir}")
    for name, p in saved_paths.items():
        print(f"    - {name}: {p}")


if __name__ == "__main__":
    main()
