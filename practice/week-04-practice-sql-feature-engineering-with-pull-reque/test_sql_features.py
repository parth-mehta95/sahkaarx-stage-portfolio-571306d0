"""
Unit test suite for sql_features.py module.

Verifies SQL aggregation queries, GROUP BY operations, multi-table JOINs,
derived geometric features, relative benchmark deltas, and SQL injection protections.
"""

from __future__ import annotations

import sqlite3
import pytest
import pandas as pd
import numpy as np

from sql_features import (
    get_database_connection,
    managed_database_connection,
    create_iris_tables,
    populate_iris_tables,
    initialize_iris_db,
    aggregate_species_features,
    engineer_observation_features,
    aggregate_features_by_size_tier,
    compute_feature_separability_summary,
    get_feature_matrix_and_target,
    ALLOWED_ORDER_COLUMNS,
    ALLOWED_DIMENSION_COLUMNS,
)


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """Fixture providing an in-memory SQLite database seeded with Iris data."""
    conn = initialize_iris_db(":memory:")
    yield conn
    conn.close()


class TestTaxonomicSpeciesAggregations:
    """Tests for aggregate_species_features (GROUP BY and JOIN queries)."""

    def test_aggregate_all_species_returns_three_rows(self, db_conn: sqlite3.Connection) -> None:
        """Verify summary returns exactly 3 species with 50 samples each."""
        df = aggregate_species_features(db_conn)
        assert len(df) == 3
        assert set(df["species_name"]) == {"setosa", "versicolor", "virginica"}
        assert (df["sample_count"] == 50).all()

    def test_aggregate_columns_present_and_valid(self, db_conn: sqlite3.Connection) -> None:
        """Verify presence and validity of all required aggregation metrics."""
        df = aggregate_species_features(db_conn)
        expected_cols = {
            "species_id",
            "species_name",
            "botanical_family",
            "native_region",
            "conservation_status",
            "sample_count",
            "avg_sepal_length",
            "avg_sepal_width",
            "min_sepal_length",
            "max_sepal_length",
            "sepal_length_range",
            "avg_petal_length",
            "avg_petal_width",
            "min_petal_length",
            "max_petal_length",
            "petal_length_range",
            "avg_petal_surface_area",
            "avg_sepal_surface_area",
            "avg_petal_aspect_ratio",
            "avg_sepal_aspect_ratio",
            "avg_petal_to_sepal_area_ratio",
        }
        for col in expected_cols:
            assert col in df.columns, f"Missing required aggregate column: {col}"

    def test_botanical_dimensions_hierarchy(self, db_conn: sqlite3.Connection) -> None:
        """Verify known morphological properties: setosa has smallest petal area."""
        df = aggregate_species_features(db_conn).set_index("species_name")
        setosa_petal_area = df.loc["setosa", "avg_petal_surface_area"]
        versicolor_petal_area = df.loc["versicolor", "avg_petal_surface_area"]
        virginica_petal_area = df.loc["virginica", "avg_petal_surface_area"]

        assert setosa_petal_area < versicolor_petal_area < virginica_petal_area

    def test_having_clause_filtering(self, db_conn: sqlite3.Connection) -> None:
        """Verify HAVING min_sample_count threshold filters correctly."""
        # Iris has 50 samples per class
        df_50 = aggregate_species_features(db_conn, min_sample_count=50)
        assert len(df_50) == 3

        df_51 = aggregate_species_features(db_conn, min_sample_count=51)
        assert len(df_51) == 0

    def test_species_name_filtering(self, db_conn: sqlite3.Connection) -> None:
        """Verify species_name parameter correctly filters to a single species."""
        df = aggregate_species_features(db_conn, species_name="setosa")
        assert len(df) == 1
        assert df.iloc[0]["species_name"] == "setosa"

    def test_invalid_min_sample_count_raises(self, db_conn: sqlite3.Connection) -> None:
        """Verify ValueError raised when min_sample_count < 1."""
        with pytest.raises(ValueError, match="min_sample_count must be a positive integer"):
            aggregate_species_features(db_conn, min_sample_count=0)


class TestObservationFeatureEngineering:
    """Tests for engineer_observation_features (Multi-table JOIN and CTE features)."""

    def test_full_dataset_row_count(self, db_conn: sqlite3.Connection) -> None:
        """Verify function returns all 150 observation rows."""
        df = engineer_observation_features(db_conn)
        assert len(df) == 150

    def test_surface_area_and_ratio_computations(self, db_conn: sqlite3.Connection) -> None:
        """Verify SQL calculated features match exact numerical products and quotients."""
        df = engineer_observation_features(db_conn)

        for _, row in df.head(10).iterrows():
            expected_petal_area = round(row["petal_length"] * row["petal_width"], 4)
            assert row["petal_surface_area"] == pytest.approx(expected_petal_area, abs=1e-4)

            expected_petal_ratio = round(row["petal_length"] / row["petal_width"], 4)
            assert row["petal_aspect_ratio"] == pytest.approx(expected_petal_ratio, abs=1e-4)

    def test_relative_species_benchmark_deltas(self, db_conn: sqlite3.Connection) -> None:
        """Verify that deltas from species mean sum to zero within each species."""
        df = engineer_observation_features(db_conn, include_species_diffs=True)
        assert "petal_len_diff_from_species_mean" in df.columns
        assert "sepal_len_diff_from_species_mean" in df.columns

        # Sum of deviations from group mean should be near zero for each species
        for species in ("setosa", "versicolor", "virginica"):
            subset = df[df["species_name"] == species]
            assert subset["petal_len_diff_from_species_mean"].sum() == pytest.approx(0.0, abs=1e-2)
            assert subset["sepal_len_diff_from_species_mean"].sum() == pytest.approx(0.0, abs=1e-2)

    def test_pagination_and_sorting(self, db_conn: sqlite3.Connection) -> None:
        """Verify limit and offset pagination work properly."""
        df_paged = engineer_observation_features(db_conn, limit=10, offset=20)
        assert len(df_paged) == 10
        assert df_paged.iloc[0]["sample_id"] == 21

    def test_invalid_order_by_raises_value_error(self, db_conn: sqlite3.Connection) -> None:
        """Verify SQL injection guard rejects non-whitelisted order_by columns."""
        with pytest.raises(ValueError, match="Invalid order_by column"):
            engineer_observation_features(db_conn, order_by="untrusted_col; DROP TABLE iris_measurements;--")


class TestSizeTierBinningAggregation:
    """Tests for aggregate_features_by_size_tier (Discretization & GROUP BY)."""

    def test_binned_aggregation_structure(self, db_conn: sqlite3.Connection) -> None:
        """Verify grouping on species_name and size_tier."""
        df = aggregate_features_by_size_tier(db_conn, dimension="petal_length")
        assert "species_name" in df.columns
        assert "size_tier" in df.columns
        assert "tier_sample_count" in df.columns
        assert set(df["size_tier"]).issubset({"compact", "medium", "elongated"})

    def test_setosa_mostly_compact(self, db_conn: sqlite3.Connection) -> None:
        """Verify setosa petal lengths (< 2.5cm) fall entirely into 'compact' tier."""
        df = aggregate_features_by_size_tier(db_conn, dimension="petal_length")
        setosa_tiers = df[df["species_name"] == "setosa"]
        assert len(setosa_tiers) == 1
        assert setosa_tiers.iloc[0]["size_tier"] == "compact"
        assert setosa_tiers.iloc[0]["tier_sample_count"] == 50

    def test_invalid_dimension_raises_value_error(self, db_conn: sqlite3.Connection) -> None:
        """Verify whitelist prevents SQL injection on dimension binning."""
        with pytest.raises(ValueError, match="Invalid dimension"):
            aggregate_features_by_size_tier(db_conn, dimension="injected_dimension")


class TestSeparabilityAndMLPipeline:
    """Tests for separability summary and scikit-learn feature matrix extraction."""

    def test_separability_metrics(self, db_conn: sqlite3.Connection) -> None:
        """Verify between-class divergence metrics are calculated for 3 species."""
        df = compute_feature_separability_summary(db_conn)
        assert len(df) == 3
        assert "petal_len_class_divergence" in df.columns
        assert "petal_area_class_divergence" in df.columns

    def test_get_feature_matrix_and_target(self, db_conn: sqlite3.Connection) -> None:
        """Verify extracted X and y matrices are properly shaped and typed."""
        X, y = get_feature_matrix_and_target(db_conn, include_engineered=True)
        assert len(X) == 150
        assert len(y) == 150
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        # Features should not contain identifier strings or IDs
        assert "sample_id" not in X.columns
        assert "species_name" not in X.columns
        # Verify engineered features are present
        assert "petal_surface_area" in X.columns
        assert "petal_aspect_ratio" in X.columns
        assert "petal_len_diff_from_species_mean" in X.columns

    def test_managed_database_connection(self) -> None:
        """Verify context manager commits on normal completion and closes cleanly."""
        with managed_database_connection(":memory:") as conn:
            populate_iris_tables(conn)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM iris_measurements;")
            count = cur.fetchone()[0]
            assert count == 150


class TestZeroDivisionResilience:
    """Tests ensuring NULLIF guards prevent ZeroDivisionError in SQL operations."""

    def test_nullif_guard_with_zero_dimension(self, db_conn: sqlite3.Connection) -> None:
        """Insert a hypothetical test measurement with width=0 and verify query handles it gracefully."""
        with db_conn:
            db_conn.execute(
                """
                INSERT INTO iris_measurements (sepal_length, sepal_width, petal_length, petal_width, species_id)
                VALUES (5.0, 3.0, 1.5, 0.0000000001, 0);
                """
            )
        # Function should execute without raising ZeroDivisionError or SQLite exceptions
        df = engineer_observation_features(db_conn)
        assert len(df) == 151
