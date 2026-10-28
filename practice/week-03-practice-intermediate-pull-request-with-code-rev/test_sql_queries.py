"""
Unit test suite for the Iris SQL query module.

Covers:
- Database connection lifecycle and foreign key enforcement
- Relational schema creation and index verification
- Data population and integrity constraint checks
- Parameterized SELECT queries with filtering and pagination
- Multi-table INNER JOIN queries with calculated columns
- Analytical aggregation queries (GROUP BY & summary stats)
- SQL injection immunity and malicious input handling
- Boundary validation and domain error handling
"""

import sqlite3
import pytest
import pandas as pd

from sql_queries import (
    get_database_connection,
    managed_database_connection,
    create_iris_tables,
    populate_iris_tables,
    initialize_iris_db,
    select_iris_measurements,
    join_measurements_with_species,
    select_species_summary_statistics,
    execute_custom_query,
    CANONICAL_SPECIES_METADATA,
    ALLOWED_ORDER_COLUMNS,
)


@pytest.fixture
def memory_db() -> sqlite3.Connection:
    """Fixture providing an initialized in-memory Iris SQLite database."""
    conn = initialize_iris_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def empty_db() -> sqlite3.Connection:
    """Fixture providing a blank in-memory database with tables created."""
    conn = get_database_connection(":memory:")
    create_iris_tables(conn)
    yield conn
    conn.close()


# -----------------------------------------------------------------------------
# 1. Connection & Schema Tests
# -----------------------------------------------------------------------------

def test_database_connection_enforces_foreign_keys():
    """Verify that get_database_connection enables SQLite PRAGMA foreign_keys."""
    conn = get_database_connection(":memory:")
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys;")
    result = cursor.fetchone()
    conn.close()
    assert result[0] == 1, "Foreign keys must be enabled on every connection."


def test_invalid_db_path_raises_value_error():
    """Verify that invalid database paths trigger ValueError."""
    with pytest.raises(ValueError):
        get_database_connection("")
    with pytest.raises(ValueError):
        get_database_connection("   ")


def test_managed_database_connection_commits_and_closes():
    """Verify that managed_database_connection commits transactions and closes cleanly."""
    with managed_database_connection(":memory:") as conn:
        create_iris_tables(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "iris_species" in tables
        assert "iris_measurements" in tables


def test_schema_creates_tables_and_indexes(empty_db: sqlite3.Connection):
    """Verify that create_iris_tables defines the required tables and performance indexes."""
    cursor = empty_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "iris_species" in tables
    assert "iris_measurements" in tables

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = {row[0] for row in cursor.fetchall()}
    assert "idx_measurements_species" in indexes
    assert "idx_measurements_petal_len" in indexes


def test_foreign_key_constraint_prevents_orphan_measurements(empty_db: sqlite3.Connection):
    """Verify that inserting a measurement with non-existent species_id raises IntegrityError."""
    # Attempting to insert measurement when species_id=99 does not exist
    invalid_insert = """
    INSERT INTO iris_measurements (sepal_length, sepal_width, petal_length, petal_width, species_id)
    VALUES (5.1, 3.5, 1.4, 0.2, 99);
    """
    with pytest.raises(sqlite3.IntegrityError):
        with empty_db:
            empty_db.execute(invalid_insert)


# -----------------------------------------------------------------------------
# 2. Population & Integrity Tests
# -----------------------------------------------------------------------------

def test_populate_iris_tables_counts(memory_db: sqlite3.Connection):
    """Verify default population loads 3 species and 150 canonical observations."""
    cursor = memory_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM iris_species;")
    species_count = cursor.fetchone()[0]
    assert species_count == 3

    cursor.execute("SELECT COUNT(*) FROM iris_measurements;")
    measurement_count = cursor.fetchone()[0]
    assert measurement_count == 150


def test_custom_dataframe_population():
    """Verify populating database from a custom subset DataFrame."""
    custom_df = pd.DataFrame({
        "sepal_length": [5.0, 6.0],
        "sepal_width": [3.0, 3.2],
        "petal_length": [1.5, 4.5],
        "petal_width": [0.2, 1.3],
        "species_id": [0, 1],
    })
    conn = get_database_connection(":memory:")
    sp_count, m_count = populate_iris_tables(conn, custom_df)
    assert sp_count == 3
    assert m_count == 2
    conn.close()


def test_populate_missing_columns_raises_error():
    """Verify that incomplete DataFrame raises ValueError."""
    bad_df = pd.DataFrame({"sepal_length": [5.0], "species_id": [0]})
    conn = get_database_connection(":memory:")
    with pytest.raises(ValueError):
        populate_iris_tables(conn, bad_df)
    conn.close()


# -----------------------------------------------------------------------------
# 3. Parameterized SELECT Query Tests (Query 1)
# -----------------------------------------------------------------------------

def test_select_all_measurements(memory_db: sqlite3.Connection):
    """Verify unconditional SELECT returns all 150 records."""
    df = select_iris_measurements(memory_db)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert "species_name" in df.columns
    assert "sepal_length" in df.columns


def test_select_filter_by_species(memory_db: sqlite3.Connection):
    """Verify parameterized filtering by species returns exactly 50 samples each."""
    for species in ["setosa", "versicolor", "virginica"]:
        df = select_iris_measurements(memory_db, species_name=species)
        assert len(df) == 50
        assert (df["species_name"] == species).all()


def test_select_filter_numeric_thresholds(memory_db: sqlite3.Connection):
    """Verify compound numeric filters for sepal length and petal width."""
    df = select_iris_measurements(
        memory_db, min_sepal_length=6.0, max_petal_width=1.5
    )
    assert not df.empty
    assert (df["sepal_length"] >= 6.0).all()
    assert (df["petal_width"] <= 1.5).all()


def test_select_pagination_limit_and_offset(memory_db: sqlite3.Connection):
    """Verify LIMIT and OFFSET pagination functioning."""
    df_first5 = select_iris_measurements(memory_db, limit=5, offset=0)
    df_next5 = select_iris_measurements(memory_db, limit=5, offset=5)

    assert len(df_first5) == 5
    assert len(df_next5) == 5
    # IDs must be sequential and disjoint
    assert list(df_first5["sample_id"]) == [1, 2, 3, 4, 5]
    assert list(df_next5["sample_id"]) == [6, 7, 8, 9, 10]


# -----------------------------------------------------------------------------
# 4. Multi-Table Relational JOIN Query Tests (Query 2)
# -----------------------------------------------------------------------------

def test_join_measurements_with_species_attributes(memory_db: sqlite3.Connection):
    """Verify INNER JOIN returns enriched metadata and computed petal_area_cm2."""
    df = join_measurements_with_species(memory_db)
    assert len(df) == 150

    expected_cols = {
        "sample_id",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "petal_area_cm2",
        "species_id",
        "species_name",
        "botanical_family",
        "native_region",
        "conservation_status",
    }
    assert expected_cols.issubset(set(df.columns))

    # Check calculated column correctness
    first_row = df.iloc[0]
    expected_area = round(first_row["petal_length"] * first_row["petal_width"], 2)
    assert first_row["petal_area_cm2"] == expected_area


def test_join_with_species_and_threshold_filter(memory_db: sqlite3.Connection):
    """Verify JOIN filtering with min_petal_length and specific species."""
    df = join_measurements_with_species(
        memory_db, species_name="virginica", min_petal_length=5.0
    )
    assert not df.empty
    assert (df["species_name"] == "virginica").all()
    assert (df["petal_length"] >= 5.0).all()
    assert (df["native_region"] == "Southeastern United States").all()


def test_join_sorting_whitelist(memory_db: sqlite3.Connection):
    """Verify safe column sorting via ORDER BY whitelist."""
    df_desc = join_measurements_with_species(
        memory_db, order_by="petal_length", ascending=False, limit=10
    )
    petal_lens = list(df_desc["petal_length"])
    assert petal_lens == sorted(petal_lens, reverse=True)


def test_join_invalid_order_by_rejected(memory_db: sqlite3.Connection):
    """Verify unapproved column in order_by raises ValueError (anti-injection defense)."""
    with pytest.raises(ValueError, match="Invalid order_by column"):
        join_measurements_with_species(
            memory_db, order_by="non_existent_column; DROP TABLE iris_species;"
        )


# -----------------------------------------------------------------------------
# 5. Analytical Aggregation Query Tests (Query 3)
# -----------------------------------------------------------------------------

def test_select_species_summary_statistics(memory_db: sqlite3.Connection):
    """Verify analytical summary aggregation computes accurate per-species metrics."""
    df = select_species_summary_statistics(memory_db)
    assert len(df) == 3
    assert list(df["species_name"]) == ["setosa", "versicolor", "virginica"]
    assert (df["sample_count"] == 50).all()

    # Botanical reality sanity checks
    # Setosa petals are known to be significantly shorter on average than Virginica
    setosa_petal = df[df["species_name"] == "setosa"]["avg_petal_length"].values[0]
    virginica_petal = df[df["species_name"] == "virginica"]["avg_petal_length"].values[0]
    assert setosa_petal < virginica_petal
    assert 1.0 <= setosa_petal <= 2.0
    assert 5.0 <= virginica_petal <= 6.0


# -----------------------------------------------------------------------------
# 6. SQL Injection Immunity & Validation Boundary Tests
# -----------------------------------------------------------------------------

def test_sql_injection_attempt_in_species_filter(memory_db: sqlite3.Connection):
    """Verify malicious SQL injection string is safely sanitized by parameter binding."""
    malicious_input = "' OR '1'='1"
    df = select_iris_measurements(memory_db, species_name=malicious_input)
    # Since no species is named "' OR '1'='1", query must safely return 0 rows without error
    assert len(df) == 0


def test_sql_injection_attempt_in_join_filter(memory_db: sqlite3.Connection):
    """Verify drop table injection fails harmlessly."""
    malicious_input = "setosa'; DROP TABLE iris_species; --"
    df = join_measurements_with_species(memory_db, species_name=malicious_input)
    assert len(df) == 0

    # Ensure table was NOT dropped
    cursor = memory_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM iris_species;")
    assert cursor.fetchone()[0] == 3


def test_negative_numeric_thresholds_raise_value_error(memory_db: sqlite3.Connection):
    """Verify that negative filter values trigger descriptive ValueError."""
    with pytest.raises(ValueError, match="min_sepal_length must be non-negative"):
        select_iris_measurements(memory_db, min_sepal_length=-1.0)

    with pytest.raises(ValueError, match="max_petal_width must be non-negative"):
        select_iris_measurements(memory_db, max_petal_width=-0.5)

    with pytest.raises(ValueError, match="limit must be a positive integer"):
        select_iris_measurements(memory_db, limit=0)

    with pytest.raises(ValueError, match="offset must be non-negative"):
        select_iris_measurements(memory_db, offset=-5)


def test_execute_custom_query_safely(memory_db: sqlite3.Connection):
    """Verify generic custom query execution with parameter binding."""
    df = execute_custom_query(
        memory_db,
        "SELECT COUNT(*) AS total_samples FROM iris_measurements WHERE petal_width > ?",
        params=(1.0,),
    )
    assert "total_samples" in df.columns
    assert df["total_samples"].iloc[0] > 0


if __name__ == "__main__":
    import sys
    print("Running test_sql_queries.py test suite...")
    test_database_connection_enforces_foreign_keys()
    test_invalid_db_path_raises_value_error()
    test_managed_database_connection_commits_and_closes()

    conn = initialize_iris_db(":memory:")
    test_populate_iris_tables_counts(conn)
    test_select_all_measurements(conn)
    test_select_filter_by_species(conn)
    test_select_filter_numeric_thresholds(conn)
    test_select_pagination_limit_and_offset(conn)
    test_join_measurements_with_species_attributes(conn)
    test_join_with_species_and_threshold_filter(conn)
    test_join_sorting_whitelist(conn)
    test_join_invalid_order_by_rejected(conn)
    test_select_species_summary_statistics(conn)
    test_sql_injection_attempt_in_species_filter(conn)
    test_sql_injection_attempt_in_join_filter(conn)
    test_negative_numeric_thresholds_raise_value_error(conn)
    test_execute_custom_query_safely(conn)
    conn.close()

    print("ALL TESTS PASSED! (14/14 self-tests succeeded)")
