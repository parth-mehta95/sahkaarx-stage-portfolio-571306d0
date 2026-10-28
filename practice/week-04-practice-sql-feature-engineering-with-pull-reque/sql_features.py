"""
SQL Feature Engineering module for the Iris ML pipeline.

Provides type-hinted, high-performance SQL aggregation queries for feature
engineering, including GROUP BY taxonomic summaries, multi-table JOINs,
derived geometric ratios, and relative benchmark deviations.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris as sklearn_load_iris

__all__ = [
    "get_database_connection",
    "managed_database_connection",
    "create_iris_tables",
    "populate_iris_tables",
    "initialize_iris_db",
    "create_feature_indexes",
    "aggregate_species_features",
    "engineer_observation_features",
    "aggregate_features_by_size_tier",
    "compute_feature_separability_summary",
    "get_feature_matrix_and_target",
    "ALLOWED_ORDER_COLUMNS",
    "ALLOWED_DIMENSION_COLUMNS",
]

# Canonical Iris Botanical Metadata for Relational Taxonomy Dimension Table
CANONICAL_SPECIES_METADATA: Tuple[Dict[str, Any], ...] = (
    {
        "species_id": 0,
        "species_name": "setosa",
        "botanical_family": "Iridaceae",
        "native_region": "North America / Arctic",
        "conservation_status": "Least Concern",
    },
    {
        "species_id": 1,
        "species_name": "versicolor",
        "botanical_family": "Iridaceae",
        "native_region": "Eastern North America",
        "conservation_status": "Least Concern",
    },
    {
        "species_id": 2,
        "species_name": "virginica",
        "botanical_family": "Iridaceae",
        "native_region": "Southeastern United States",
        "conservation_status": "Vulnerable",
    },
)

# Whitelisted columns for sorting and dimension binning (SQL injection prevention)
ALLOWED_ORDER_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species_id",
    "species_name",
    "petal_surface_area",
    "sepal_surface_area",
    "petal_aspect_ratio",
    "sepal_aspect_ratio",
    "petal_to_sepal_length_ratio",
    "petal_to_sepal_width_ratio",
    "petal_to_sepal_area_ratio",
    "petal_len_diff_from_species_mean",
    "sepal_len_diff_from_species_mean",
)

ALLOWED_DIMENSION_COLUMNS: Tuple[str, ...] = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)


def get_database_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Establish and configure an SQLite database connection.

    Enforces foreign key constraints and configures thread-safe connection.

    Parameters
    ----------
    db_path : str, default=":memory:"
        Path to SQLite database file or ':memory:' for transient in-memory storage.

    Returns
    -------
    sqlite3.Connection
        Active SQLite connection with foreign keys enabled.

    Raises
    ------
    ValueError
        If db_path is empty or not a string.
    sqlite3.DatabaseError
        If connection cannot be opened or configured.
    """
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValueError("Database path must be a non-empty string.")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextlib.contextmanager
def managed_database_connection(
    db_path: str = ":memory:",
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe SQLite database connection lifecycle management.

    Ensures transactions are committed on success, rolled back on error,
    and connections are reliably closed upon exiting the context block.

    Parameters
    ----------
    db_path : str, default=":memory:"
        Path to database or ':memory:'.

    Yields
    ------
    sqlite3.Connection
        Active SQLite connection.
    """
    conn = get_database_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_iris_tables(conn: sqlite3.Connection) -> None:
    """Create normalized relational tables and foundational indexes.

    Creates two normalized tables:
    1. `iris_species`: Dimension table containing botanical taxonomy and metadata.
    2. `iris_measurements`: Fact table containing morphological measurements
       with foreign key referencing `iris_species(species_id)`.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    """
    species_table_ddl = """
    CREATE TABLE IF NOT EXISTS iris_species (
        species_id INTEGER PRIMARY KEY,
        species_name TEXT NOT NULL UNIQUE,
        botanical_family TEXT NOT NULL,
        native_region TEXT NOT NULL,
        conservation_status TEXT NOT NULL
    );
    """

    measurements_table_ddl = """
    CREATE TABLE IF NOT EXISTS iris_measurements (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sepal_length REAL NOT NULL CHECK(sepal_length > 0),
        sepal_width REAL NOT NULL CHECK(sepal_width > 0),
        petal_length REAL NOT NULL CHECK(petal_length > 0),
        petal_width REAL NOT NULL CHECK(petal_width > 0),
        species_id INTEGER NOT NULL,
        FOREIGN KEY (species_id) REFERENCES iris_species(species_id)
            ON UPDATE CASCADE
            ON DELETE RESTRICT
    );
    """

    indexes_ddl = """
    CREATE INDEX IF NOT EXISTS idx_measurements_species
        ON iris_measurements(species_id);
    CREATE INDEX IF NOT EXISTS idx_measurements_petal_len
        ON iris_measurements(petal_length);
    CREATE INDEX IF NOT EXISTS idx_measurements_sepal_len
        ON iris_measurements(sepal_length);
    """

    with conn:
        conn.execute(species_table_ddl)
        conn.execute(measurements_table_ddl)
        conn.executescript(indexes_ddl)


def create_feature_indexes(conn: sqlite3.Connection) -> None:
    """Create compound and covering indexes to accelerate ML feature queries.

    Optimizes GROUP BY and JOIN operations involving species_id along with
    petal and sepal dimensions.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    """
    feature_indexes_ddl = """
    CREATE INDEX IF NOT EXISTS idx_features_species_petals
        ON iris_measurements(species_id, petal_length, petal_width);
    CREATE INDEX IF NOT EXISTS idx_features_species_sepals
        ON iris_measurements(species_id, sepal_length, sepal_width);
    """
    with conn:
        conn.executescript(feature_indexes_ddl)


def populate_iris_tables(
    conn: sqlite3.Connection,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[int, int]:
    """Populate Iris dimension and measurement tables with canonical data.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    df : Optional[pd.DataFrame], default=None
        Optional input DataFrame containing Iris measurements. If None,
        the canonical scikit-learn Iris dataset is automatically loaded.

    Returns
    -------
    Tuple[int, int]
        Count of (species_records_inserted, measurement_records_inserted).
    """
    create_iris_tables(conn)
    create_feature_indexes(conn)

    # 1. Insert species metadata dimension rows
    species_insert_sql = """
    INSERT OR IGNORE INTO iris_species (
        species_id, species_name, botanical_family, native_region, conservation_status
    ) VALUES (:species_id, :species_name, :botanical_family, :native_region, :conservation_status);
    """

    with conn:
        conn.executemany(species_insert_sql, CANONICAL_SPECIES_METADATA)

    # 2. Prepare measurement records
    if df is None:
        bunch = sklearn_load_iris(as_frame=True)
        raw_df: pd.DataFrame = bunch.frame.copy()
        raw_df.rename(
            columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
                "target": "species_id",
            },
            inplace=True,
        )
    else:
        raw_df = df.copy()

    measurements_data: List[Tuple[float, float, float, float, int]] = []
    for _, row in raw_df.iterrows():
        measurements_data.append(
            (
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
                int(row["species_id"]),
            )
        )

    # Check if measurements table already populated
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM iris_measurements;")
    existing_count = cursor.fetchone()[0]

    if existing_count == 0:
        insert_measurement_sql = """
        INSERT INTO iris_measurements (
            sepal_length, sepal_width, petal_length, petal_width, species_id
        ) VALUES (?, ?, ?, ?, ?);
        """
        with conn:
            conn.executemany(insert_measurement_sql, measurements_data)
        inserted_measurements = len(measurements_data)
    else:
        inserted_measurements = 0

    return len(CANONICAL_SPECIES_METADATA), inserted_measurements


def initialize_iris_db(db_path: str = ":memory:") -> sqlite3.Connection:
    """Initialize an Iris SQLite database with tables, indexes, and seeded data.

    Parameters
    ----------
    db_path : str, default=":memory:"
        Database path or ':memory:'.

    Returns
    -------
    sqlite3.Connection
        Ready-to-query SQLite connection with 150 Iris observations and taxonomy metadata.
    """
    conn = get_database_connection(db_path)
    populate_iris_tables(conn)
    return conn


# ==============================================================================
# Feature Engineering Query 1: Species-Level GROUP BY Aggregations with JOIN
# ==============================================================================

def aggregate_species_features(
    conn: sqlite3.Connection,
    min_sample_count: int = 1,
    species_name: Optional[str] = None,
) -> pd.DataFrame:
    """Compute taxonomic summary statistics and benchmark features via GROUP BY and JOIN.

    Aggregates observation counts, means, min/max bounds, ranges, and geometric
    surface benchmarks for each Iris species. Uses parameter binding and HAVING
    clause filtering.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    min_sample_count : int, default=1
        Minimum number of samples required per group (HAVING threshold).
    species_name : Optional[str], default=None
        Optional species name filter ('setosa', 'versicolor', 'virginica').

    Returns
    -------
    pd.DataFrame
        Aggregated summary metrics indexed by species_id.

    Raises
    ------
    ValueError
        If min_sample_count is less than 1.
    """
    if not isinstance(min_sample_count, int) or min_sample_count < 1:
        raise ValueError(f"min_sample_count must be a positive integer, got {min_sample_count}")

    where_clauses: List[str] = []
    params: List[Any] = []

    if species_name is not None:
        where_clauses.append("LOWER(s.species_name) = ?")
        params.append(species_name.strip().lower())

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # HAVING clause parameter
    having_sql = "HAVING sample_count >= ?"
    params.append(int(min_sample_count))

    query = f"""
    SELECT
        s.species_id,
        s.species_name,
        s.botanical_family,
        s.native_region,
        s.conservation_status,
        COUNT(m.sample_id) AS sample_count,
        -- Sepal Mean Dimensions
        ROUND(AVG(m.sepal_length), 4) AS avg_sepal_length,
        ROUND(AVG(m.sepal_width), 4) AS avg_sepal_width,
        ROUND(MIN(m.sepal_length), 4) AS min_sepal_length,
        ROUND(MAX(m.sepal_length), 4) AS max_sepal_length,
        ROUND(MAX(m.sepal_length) - MIN(m.sepal_length), 4) AS sepal_length_range,
        -- Petal Mean Dimensions
        ROUND(AVG(m.petal_length), 4) AS avg_petal_length,
        ROUND(AVG(m.petal_width), 4) AS avg_petal_width,
        ROUND(MIN(m.petal_length), 4) AS min_petal_length,
        ROUND(MAX(m.petal_length), 4) AS max_petal_length,
        ROUND(MAX(m.petal_length) - MIN(m.petal_length), 4) AS petal_length_range,
        -- Engineered Morphological Benchmarks (Guarded against division by zero)
        ROUND(AVG(m.petal_length * m.petal_width), 4) AS avg_petal_surface_area,
        ROUND(AVG(m.sepal_length * m.sepal_width), 4) AS avg_sepal_surface_area,
        ROUND(AVG(m.petal_length / NULLIF(m.petal_width, 0)), 4) AS avg_petal_aspect_ratio,
        ROUND(AVG(m.sepal_length / NULLIF(m.sepal_width, 0)), 4) AS avg_sepal_aspect_ratio,
        ROUND(AVG((m.petal_length * m.petal_width) / NULLIF(m.sepal_length * m.sepal_width, 0)), 4) AS avg_petal_to_sepal_area_ratio
    FROM iris_species AS s
    INNER JOIN iris_measurements AS m
        ON s.species_id = m.species_id
    {where_sql}
    GROUP BY
        s.species_id,
        s.species_name,
        s.botanical_family,
        s.native_region,
        s.conservation_status
    {having_sql}
    ORDER BY s.species_id ASC;
    """

    return pd.read_sql_query(query, conn, params=params)


# ==============================================================================
# Feature Engineering Query 2: Observation-Level Feature Engineering with JOINs
# ==============================================================================

def engineer_observation_features(
    conn: sqlite3.Connection,
    species_name: Optional[str] = None,
    include_species_diffs: bool = True,
    order_by: str = "sample_id",
    ascending: bool = True,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> pd.DataFrame:
    """Engineer per-observation morphological features via multi-table JOIN and CTE.

    Computes:
    1. Surface area approximations: petal_surface_area, sepal_surface_area
    2. Dimension aspect ratios: petal_aspect_ratio, sepal_aspect_ratio
    3. Cross-organ interaction ratios: petal_to_sepal_length_ratio, petal_to_sepal_area_ratio
    4. Relative benchmark deviations: delta from species-level mean calculated via
       an inner CTE aggregation join.
    5. Taxonomy metadata: botanical_family, native_region, conservation_status.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    species_name : Optional[str], default=None
        Filter by species name ('setosa', 'versicolor', 'virginica').
    include_species_diffs : bool, default=True
        Whether to calculate relative deviations from species group averages.
    order_by : str, default="sample_id"
        Column name to sort by (must be in ALLOWED_ORDER_COLUMNS).
    ascending : bool, default=True
        Sort direction: True for ASC, False for DESC.
    limit : Optional[int], default=None
        Maximum rows to return.
    offset : Optional[int], default=None
        Pagination row offset.

    Returns
    -------
    pd.DataFrame
        Engineered feature table ready for ML model training.

    Raises
    ------
    ValueError
        If order_by is not whitelisted or limit/offset are invalid.
    """
    if order_by not in ALLOWED_ORDER_COLUMNS:
        raise ValueError(
            f"Invalid order_by column '{order_by}'. Must be one of: {ALLOWED_ORDER_COLUMNS}"
        )
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    if offset is not None and offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    direction = "ASC" if ascending else "DESC"
    params: List[Any] = []
    where_clauses: List[str] = []

    if species_name is not None:
        where_clauses.append("LOWER(s.species_name) = ?")
        params.append(species_name.strip().lower())

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    pagination_sql = ""
    if limit is not None:
        pagination_sql += " LIMIT ?"
        params.append(int(limit))
        if offset is not None:
            pagination_sql += " OFFSET ?"
            params.append(int(offset))
    elif offset is not None:
        pagination_sql += " LIMIT -1 OFFSET ?"
        params.append(int(offset))

    if include_species_diffs:
        # CTE joining individual observations with taxonomic group averages
        query = f"""
        WITH species_benchmarks AS (
            SELECT
                species_id,
                AVG(sepal_length) AS sp_avg_sepal_len,
                AVG(sepal_width) AS sp_avg_sepal_wid,
                AVG(petal_length) AS sp_avg_petal_len,
                AVG(petal_width) AS sp_avg_petal_wid,
                AVG(petal_length * petal_width) AS sp_avg_petal_area,
                AVG(sepal_length * sepal_width) AS sp_avg_sepal_area
            FROM iris_measurements
            GROUP BY species_id
        )
        SELECT
            m.sample_id,
            m.species_id,
            s.species_name,
            s.botanical_family,
            s.native_region,
            s.conservation_status,
            -- Raw Measured Dimensions
            m.sepal_length,
            m.sepal_width,
            m.petal_length,
            m.petal_width,
            -- Engineered Geometric Surface Features
            ROUND(m.sepal_length * m.sepal_width, 4) AS sepal_surface_area,
            ROUND(m.petal_length * m.petal_width, 4) AS petal_surface_area,
            -- Engineered Aspect Ratios (Safe NULLIF guards)
            ROUND(m.sepal_length / NULLIF(m.sepal_width, 0), 4) AS sepal_aspect_ratio,
            ROUND(m.petal_length / NULLIF(m.petal_width, 0), 4) AS petal_aspect_ratio,
            ROUND(m.petal_length / NULLIF(m.sepal_length, 0), 4) AS petal_to_sepal_length_ratio,
            ROUND(m.petal_width / NULLIF(m.sepal_width, 0), 4) AS petal_to_sepal_width_ratio,
            ROUND((m.petal_length * m.petal_width) / NULLIF(m.sepal_length * m.sepal_width, 0), 4) AS petal_to_sepal_area_ratio,
            -- Relative Deviations From Species Benchmark (ML Contextual Features)
            ROUND(m.sepal_length - sb.sp_avg_sepal_len, 4) AS sepal_len_diff_from_species_mean,
            ROUND(m.sepal_width - sb.sp_avg_sepal_wid, 4) AS sepal_wid_diff_from_species_mean,
            ROUND(m.petal_length - sb.sp_avg_petal_len, 4) AS petal_len_diff_from_species_mean,
            ROUND(m.petal_width - sb.sp_avg_petal_wid, 4) AS petal_wid_diff_from_species_mean,
            ROUND((m.petal_length * m.petal_width) / NULLIF(sb.sp_avg_petal_area, 0), 4) AS petal_area_ratio_to_species_avg,
            ROUND((m.sepal_length * m.sepal_width) / NULLIF(sb.sp_avg_sepal_area, 0), 4) AS sepal_area_ratio_to_species_avg
        FROM iris_measurements AS m
        INNER JOIN iris_species AS s
            ON m.species_id = s.species_id
        INNER JOIN species_benchmarks AS sb
            ON m.species_id = sb.species_id
        {where_sql}
        ORDER BY {order_by} {direction}
        {pagination_sql};
        """
    else:
        query = f"""
        SELECT
            m.sample_id,
            m.species_id,
            s.species_name,
            s.botanical_family,
            s.native_region,
            s.conservation_status,
            -- Raw Measured Dimensions
            m.sepal_length,
            m.sepal_width,
            m.petal_length,
            m.petal_width,
            -- Engineered Geometric Surface Features
            ROUND(m.sepal_length * m.sepal_width, 4) AS sepal_surface_area,
            ROUND(m.petal_length * m.petal_width, 4) AS petal_surface_area,
            -- Engineered Aspect Ratios (Safe NULLIF guards)
            ROUND(m.sepal_length / NULLIF(m.sepal_width, 0), 4) AS sepal_aspect_ratio,
            ROUND(m.petal_length / NULLIF(m.petal_width, 0), 4) AS petal_aspect_ratio,
            ROUND(m.petal_length / NULLIF(m.sepal_length, 0), 4) AS petal_to_sepal_length_ratio,
            ROUND(m.petal_width / NULLIF(m.sepal_width, 0), 4) AS petal_to_sepal_width_ratio,
            ROUND((m.petal_length * m.petal_width) / NULLIF(m.sepal_length * m.sepal_width, 0), 4) AS petal_to_sepal_area_ratio
        FROM iris_measurements AS m
        INNER JOIN iris_species AS s
            ON m.species_id = s.species_id
        {where_sql}
        ORDER BY {order_by} {direction}
        {pagination_sql};
        """

    return pd.read_sql_query(query, conn, params=params)


# ==============================================================================
# Feature Engineering Query 3: Multi-Category Binning Aggregation (GROUP BY & JOIN)
# ==============================================================================

def aggregate_features_by_size_tier(
    conn: sqlite3.Connection,
    dimension: str = "petal_length",
) -> pd.DataFrame:
    """Discretize continuous measurements into size tiers and aggregate with JOIN.

    Uses SQL CASE expressions to discretize measurements into:
    - 'compact': < 2.5 cm
    - 'medium': 2.5 cm to 5.0 cm
    - 'elongated': >= 5.0 cm

    Then executes a GROUP BY on (species_name, size_tier) to calculate tier distributions
    and sub-group morphological means.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    dimension : str, default="petal_length"
        Continuous feature dimension to bin. Must be in ALLOWED_DIMENSION_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Binned aggregation metrics.

    Raises
    ------
    ValueError
        If dimension column is not permitted.
    """
    if dimension not in ALLOWED_DIMENSION_COLUMNS:
        raise ValueError(
            f"Invalid dimension '{dimension}'. Must be one of: {ALLOWED_DIMENSION_COLUMNS}"
        )

    # Thresholds tailored for Iris morphology
    if dimension in ("petal_length", "sepal_length"):
        low_thresh, high_thresh = 2.5, 5.0
    else:
        low_thresh, high_thresh = 1.0, 2.0

    query = f"""
    SELECT
        s.species_name,
        CASE
            WHEN m.{dimension} < {low_thresh} THEN 'compact'
            WHEN m.{dimension} >= {low_thresh} AND m.{dimension} < {high_thresh} THEN 'medium'
            ELSE 'elongated'
        END AS size_tier,
        COUNT(m.sample_id) AS tier_sample_count,
        ROUND(AVG(m.sepal_length), 4) AS avg_sepal_length,
        ROUND(AVG(m.petal_length), 4) AS avg_petal_length,
        ROUND(AVG(m.petal_length * m.petal_width), 4) AS avg_petal_surface_area,
        ROUND(AVG(m.petal_length / NULLIF(m.petal_width, 0)), 4) AS avg_petal_aspect_ratio
    FROM iris_measurements AS m
    INNER JOIN iris_species AS s
        ON m.species_id = s.species_id
    GROUP BY
        s.species_name,
        size_tier
    ORDER BY
        s.species_name ASC,
        tier_sample_count DESC;
    """

    return pd.read_sql_query(query, conn)


# ==============================================================================
# Feature Engineering Query 4: Class Separability & Feature Importance Metrics
# ==============================================================================

def compute_feature_separability_summary(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Compute SQL-driven between-class separability metrics for feature selection.

    Compares global feature means against per-species group means to quantify
    discriminative potential (between-class sum of squares proxy) entirely in SQL.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.

    Returns
    -------
    pd.DataFrame
        Summary table with between-class separation metrics for each feature.
    """
    query = """
    WITH global_averages AS (
        SELECT
            AVG(sepal_length) AS glob_sepal_len,
            AVG(sepal_width) AS glob_sepal_wid,
            AVG(petal_length) AS glob_petal_len,
            AVG(petal_width) AS glob_petal_wid,
            AVG(petal_length * petal_width) AS glob_petal_area,
            AVG(petal_length / NULLIF(petal_width, 0)) AS glob_petal_aspect
        FROM iris_measurements
    ),
    species_aggregates AS (
        SELECT
            s.species_id,
            s.species_name,
            COUNT(m.sample_id) AS n_k,
            AVG(m.sepal_length) AS sp_sepal_len,
            AVG(m.sepal_width) AS sp_sepal_wid,
            AVG(m.petal_length) AS sp_petal_len,
            AVG(m.petal_width) AS sp_petal_wid,
            AVG(m.petal_length * m.petal_width) AS sp_petal_area,
            AVG(m.petal_length / NULLIF(m.petal_width, 0)) AS sp_petal_aspect
        FROM iris_species AS s
        INNER JOIN iris_measurements AS m ON s.species_id = m.species_id
        GROUP BY s.species_id, s.species_name
    )
    SELECT
        sa.species_id,
        sa.species_name,
        sa.n_k AS sample_count,
        -- Absolute difference between species mean and dataset global mean
        ROUND(sa.sp_sepal_len - ga.glob_sepal_len, 4) AS sepal_len_class_divergence,
        ROUND(sa.sp_sepal_wid - ga.glob_sepal_wid, 4) AS sepal_wid_class_divergence,
        ROUND(sa.sp_petal_len - ga.glob_petal_len, 4) AS petal_len_class_divergence,
        ROUND(sa.sp_petal_wid - ga.glob_petal_wid, 4) AS petal_wid_class_divergence,
        ROUND(sa.sp_petal_area - ga.glob_petal_area, 4) AS petal_area_class_divergence,
        ROUND(sa.sp_petal_aspect - ga.glob_petal_aspect, 4) AS petal_aspect_class_divergence
    FROM species_aggregates AS sa
    CROSS JOIN global_averages AS ga
    ORDER BY sa.species_id ASC;
    """

    return pd.read_sql_query(query, conn)


# ==============================================================================
# High-Level ML Feature Matrix Extractor
# ==============================================================================

def get_feature_matrix_and_target(
    conn: sqlite3.Connection,
    include_engineered: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract ready-to-train ML feature matrix X and target label Series y.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    include_engineered : bool, default=True
        If True, includes all SQL-engineered surface areas, ratios, and relative
        deviations. If False, returns only raw physical measurements.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X_features, y_target) tuple for scikit-learn estimators.
    """
    df = engineer_observation_features(conn, include_species_diffs=include_engineered)

    # Target series (species_id or species_name)
    y = df["species_id"].copy()

    # Feature columns exclusion list
    non_feature_cols = [
        "sample_id",
        "species_id",
        "species_name",
        "botanical_family",
        "native_region",
        "conservation_status",
    ]

    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    X = df[feature_cols].copy()

    return X, y


# ==============================================================================
# Module Entrypoint & Demonstrator
# ==============================================================================

def main() -> None:
    """Demonstrate SQL feature engineering queries."""
    print("=" * 70)
    print("Iris ML Feature Engineering: SQL Aggregations Demonstration")
    print("=" * 70)

    with managed_database_connection() as conn:
        populate_iris_tables(conn)

        print("\n1. Taxonomic Feature Aggregations (GROUP BY & JOIN):")
        species_aggs = aggregate_species_features(conn, min_sample_count=10)
        display_cols = [
            "species_name",
            "sample_count",
            "avg_sepal_length",
            "avg_petal_length",
            "avg_petal_surface_area",
            "avg_petal_aspect_ratio",
        ]
        print(species_aggs[display_cols].to_string(index=False))

        print("\n2. Observation-Level Engineered Features (JOIN + CTE Relational Benchmarks):")
        obs_features = engineer_observation_features(conn, limit=5)
        obs_cols = [
            "sample_id",
            "species_name",
            "petal_surface_area",
            "petal_aspect_ratio",
            "petal_len_diff_from_species_mean",
            "petal_area_ratio_to_species_avg",
        ]
        print(obs_features[obs_cols].to_string(index=False))

        print("\n3. Binned Size Tiers (GROUP BY & Discretized JOIN):")
        binned = aggregate_features_by_size_tier(conn, dimension="petal_length")
        print(binned.to_string(index=False))

        print("\n4. Feature Divergence vs Global Mean (Separability Query):")
        sep = compute_feature_separability_summary(conn)
        print(sep.to_string(index=False))

        print("\n5. Ready-to-Train ML Feature Matrix:")
        X, y = get_feature_matrix_and_target(conn)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Engineered feature names ({len(X.columns)}): {list(X.columns)}")

    print("\n[✓] SQL Feature Engineering functions executed successfully!")


if __name__ == "__main__":
    main()
