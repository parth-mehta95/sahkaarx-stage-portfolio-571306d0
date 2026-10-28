"""
Comprehensive Integration Test Suite for End-to-End Iris ML Pipeline.

Verifies that all capstone modules work together seamlessly:
- Data Ingestion (data_loader.py)
- Data Validation (validation_functions.py)
- Relational Database Operations (sql_queries.py)
- SQL Feature Engineering (sql_features.py)
- Model Training & Alignment (models.py)
- End-to-End Pipeline Execution (pipeline.py)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest

# Add repository root to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_loader import load_iris, preprocess
from models import (
    CLASS_NAMES,
    FEATURE_NAMES,
    export_models,
    save_metrics_table,
    train_and_evaluate_all_models,
)
from pipeline import run_full_pipeline
from sql_features import (
    aggregate_features_by_size_tier,
    aggregate_species_features,
    compute_feature_separability_summary,
    engineer_observation_features,
)
from sql_queries import (
    get_database_connection,
    initialize_iris_db,
    join_measurements_with_species,
    select_iris_measurements,
    select_species_summary_statistics,
)
from validation_functions import (
    validate_feature_matrix,
    validate_iris_dataframe,
    validate_target_vector,
)


def test_data_loader_and_validation_integration() -> None:
    """Verify data_loader output conforms to validation_functions constraints."""
    X, y = load_iris(as_frame=True, return_X_y=True)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (150, 4)
    assert len(y) == 150

    # Validation should pass without raising exceptions
    validate_feature_matrix(X.values)
    validate_target_vector(y.values)


def test_sql_queries_and_schema_integration() -> None:
    """Verify SQL database initialization, schema integrity, and query operations."""
    conn = initialize_iris_db(":memory:")

    # 1. Verify schema tables exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "iris_species" in tables
    assert "iris_measurements" in tables

    # 2. Verify record counts
    cursor.execute("SELECT COUNT(*) FROM iris_species;")
    assert cursor.fetchone()[0] == 3

    cursor.execute("SELECT COUNT(*) FROM iris_measurements;")
    assert cursor.fetchone()[0] == 150

    # 3. Test parameterized SELECT
    df_setosa = select_iris_measurements(conn, species_name="setosa")
    assert len(df_setosa) == 50
    assert (df_setosa["species_name"] == "setosa").all()

    # 4. Test relational JOIN with computed petal area
    df_join = join_measurements_with_species(conn, min_petal_length=4.0)
    assert len(df_join) > 0
    assert "petal_area_cm2" in df_join.columns
    assert "native_region" in df_join.columns
    assert (df_join["petal_length"] >= 4.0).all()

    # 5. Test analytical summary stats
    df_summary = select_species_summary_statistics(conn)
    assert len(df_summary) == 3
    assert set(df_summary["species_name"]) == {"setosa", "versicolor", "virginica"}
    conn.close()


def test_sql_features_integration() -> None:
    """Verify SQL feature engineering transforms raw data into rich ML features."""
    conn = initialize_iris_db(":memory:")

    # 1. Engineered observation features with ratios & CTE benchmarks
    features_df = engineer_observation_features(conn)
    assert len(features_df) == 150
    expected_cols = [
        "sepal_aspect_ratio",
        "petal_aspect_ratio",
        "petal_to_sepal_length_ratio",
        "petal_surface_area",
        "sepal_surface_area",
        "delta_from_species_avg_petal_length",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature column: {col}"

    # 2. Size tiers aggregation
    tiers_df = aggregate_features_by_size_tier(conn)
    assert len(tiers_df) > 0
    assert "size_tier" in tiers_df.columns

    # 3. Class separability summary
    sep_df = compute_feature_separability_summary(conn)
    assert len(sep_df) > 0
    assert "f_ratio_proxy" in sep_df.columns

    conn.close()


def test_models_and_hungarian_alignment_integration() -> None:
    """Verify model training, Hungarian cluster alignment, and benchmarking."""
    X, y = load_iris(as_frame=False, return_X_y=True)
    X_train, X_test, y_train, y_test = preprocess(X, y, test_size=0.2, random_state=42)

    models_dict, comp_df = train_and_evaluate_all_models(
        X_train, y_train, X_test, y_test, random_state=42
    )

    assert "decision_tree" in models_dict
    assert "random_forest" in models_dict
    assert "kmeans" in models_dict

    # Check accuracy benchmarks
    dt_acc = comp_df.loc[comp_df["Model"] == "Decision Tree", "Accuracy"].values[0]
    rf_acc = comp_df.loc[comp_df["Model"] == "Random Forest", "Accuracy"].values[0]
    km_acc = comp_df.loc[comp_df["Model"] == "K-Means (Cluster-Aligned)", "Accuracy"].values[0]

    assert dt_acc >= 0.90, f"Decision Tree accuracy too low: {dt_acc}"
    assert rf_acc >= 0.90, f"Random Forest accuracy too low: {rf_acc}"
    assert km_acc >= 0.80, f"K-Means aligned accuracy too low: {km_acc}"


def test_full_pipeline_run() -> None:
    """Verify complete end-to-end pipeline executes cleanly and exports artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        metrics_csv = tmp_path / "test_metrics.csv"
        models_dir = tmp_path / "test_models"

        res = run_full_pipeline(
            random_state=42,
            test_size=0.2,
            output_metrics_path=str(metrics_csv),
            output_models_dir=str(models_dir),
        )

        assert metrics_csv.exists()
        df = pd.read_csv(metrics_csv)
        assert len(df) == 3
        assert set(df["Model"]) == {"Decision Tree", "Random Forest", "K-Means (Cluster-Aligned)"}

        # Verify exported models
        assert models_dir.exists()
        for name in ["decision_tree", "random_forest", "kmeans"]:
            artifact_file = models_dir / f"{name}.joblib"
            assert artifact_file.exists()
            loaded = joblib.load(artifact_file)
            assert loaded is not None
