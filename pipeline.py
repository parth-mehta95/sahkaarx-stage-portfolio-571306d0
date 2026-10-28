"""
End-to-End Iris Machine Learning Pipeline Runner.

Integrates all capstone lifecycle phases:
1. Data Ingestion & Loading (data_loader.py)
2. Data Validation & Type Safety (validation_functions.py)
3. Relational Database Persistence & SQL Queries (sql_queries.py)
4. Advanced SQL Feature Engineering (sql_features.py)
5. Preprocessing & Leak-Free Feature Scaling (data_loader.preprocess)
6. Model Training, Bipartite Alignment, Evaluation & Serialization (models.py)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

# Core Pipeline Modules
from data_loader import load_iris, preprocess
from models import (
    CLASS_NAMES,
    FEATURE_NAMES,
    export_models,
    generate_comparison_dataframe,
    save_metrics_table,
    train_and_evaluate_all_models,
)
from sql_features import (
    aggregate_features_by_size_tier,
    aggregate_species_features,
    compute_feature_separability_summary,
    engineer_observation_features,
)
from sql_queries import (
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("iris_pipeline")


def run_full_pipeline(
    random_state: int = 42,
    test_size: float = 0.2,
    output_metrics_path: str = "metrics.csv",
    output_models_dir: str = "models",
) -> Dict[str, Any]:
    """Execute the complete Iris Machine Learning Pipeline.

    Parameters
    ----------
    random_state : int, default=42
        Reproducibility seed for data partitioning and stochastic models.
    test_size : float, default=0.2
        Stratified test split fraction.
    output_metrics_path : str, default='metrics.csv'
        Destination file path for model comparison metrics table.
    output_models_dir : str, default='models'
        Output directory for serialized joblib model artifacts.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing pipeline execution artifacts and metrics.
    """
    logger.info("=" * 70)
    logger.info("🌸 Starting End-to-End Iris Machine Learning Pipeline")
    logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # Step 1: Data Ingestion & Loading
    # --------------------------------------------------------------------------
    logger.info("[Phase 2] Loading canonical Iris dataset...")
    X, y = load_iris(as_frame=True, return_X_y=True)
    logger.info("Successfully loaded Iris data: %d samples, %d features.", X.shape[0], X.shape[1])

    # --------------------------------------------------------------------------
    # Step 2: Data Validation
    # --------------------------------------------------------------------------
    logger.info("[Phase 2] Running validation contracts...")
    validate_feature_matrix(X.values if hasattr(X, "values") else X)
    validate_target_vector(y.values if hasattr(y, "values") else y)
    logger.info("Validation checks passed: feature matrix and target vector are sound.")

    # --------------------------------------------------------------------------
    # Step 3: Relational SQL Persistence & Querying
    # --------------------------------------------------------------------------
    logger.info("[Phase 3] Ingesting observations into SQLite relational database...")
    db_conn = initialize_iris_db(":memory:")
    
    # Verify parameterized SELECT query
    filtered_df = select_iris_measurements(db_conn, species_name="setosa", limit=5)
    logger.info("Executed parameterized SELECT query (found %d setosa records).", len(filtered_df))

    # Verify relational JOIN query
    joined_df = join_measurements_with_species(db_conn, min_petal_length=4.0, limit=5)
    logger.info("Executed relational INNER JOIN query (found %d records with petal >= 4.0cm).", len(joined_df))

    # Verify analytical summary statistics
    summary_df = select_species_summary_statistics(db_conn)
    logger.info("Executed species summary aggregation (%d taxonomy classes).", len(summary_df))

    # --------------------------------------------------------------------------
    # Step 4: SQL Feature Engineering
    # --------------------------------------------------------------------------
    logger.info("[Phase 3] Engineering advanced SQL features & ratios...")
    features_df = engineer_observation_features(db_conn)
    logger.info("Engineered observation features with geometric ratios: %s", features_df.columns.tolist()[:6])

    tier_df = aggregate_features_by_size_tier(db_conn)
    logger.info("Computed morphological size tiers: %d tiers.", len(tier_df))

    sep_df = compute_feature_separability_summary(db_conn)
    logger.info("Computed class separability summary across features.")

    # --------------------------------------------------------------------------
    # Step 5: Data Preprocessing & Leak-Free Feature Scaling
    # --------------------------------------------------------------------------
    logger.info("[Phase 3] Partitioning & scaling features without data leakage...")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Partitioned data: Train=%d samples, Test=%d samples (stratified).",
        len(X_train_scaled),
        len(X_test_scaled),
    )

    # --------------------------------------------------------------------------
    # Step 6: Model Training & Bipartite Cluster Alignment
    # --------------------------------------------------------------------------
    logger.info("[Phase 4 & 5] Training and benchmarking Decision Tree, Random Forest & K-Means...")
    models_dict, comparison_df = train_and_evaluate_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test, random_state=random_state
    )

    logger.info("Comparative Performance Benchmarks:\n%s", comparison_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # Step 7: Export Metrics & Serialized Models
    # --------------------------------------------------------------------------
    metrics_path = Path(output_metrics_path)
    save_metrics_table(comparison_df, metrics_path)
    logger.info("Saved comparison benchmark metrics to: %s", metrics_path)

    models_dir = Path(output_models_dir)
    saved_paths = export_models(models_dict, models_dir)
    logger.info("Serialized trained model artifacts to: %s", models_dir)
    for name, path_str in saved_paths.items():
        logger.info("  - %s: %s", name, path_str)

    db_conn.close()
    logger.info("=" * 70)
    logger.info("✓ Full Iris ML Pipeline Executed Successfully!")
    logger.info("=" * 70)

    return {
        "models": models_dict,
        "metrics_df": comparison_df,
        "saved_model_paths": saved_paths,
        "metrics_csv_path": str(metrics_path),
    }


if __name__ == "__main__":
    run_full_pipeline()
