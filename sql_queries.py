"""
SQL query functions module for the Iris ML pipeline.

Provides type-hinted, secure, parameterized SQL query interfaces for storing,
retrieving, joining, and aggregating Iris dataset measurements and botanical
metadata within an SQLite relational database.
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
    "create_iris_tables",
    "populate_iris_tables",
    "initialize_iris_db",
    "select_iris_measurements",
    "join_measurements_with_species",
    "select_species_summary_statistics",
    "execute_custom_query",
    "managed_database_connection",
]

# Canonical Iris Botanical Metadata for Relational Taxonomy Table
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

# Allowed columns for ordering in queries (prevents SQL injection on ORDER BY)
ALLOWED_ORDER_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species_id",
    "species_name",
    "petal_area_cm2",
)


def get_database_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Establish and configure an SQLite database connection.

    Enforces foreign key constraints and configures the row factory for
    dictionary-style access when needed.

    Parameters
    ----------
    db_path : str, default=":memory:"
        Path to SQLite database file or ':memory:' for transient in-memory storage.

    Returns
    -------
    sqlite3.Connection
        Active SQLite connection with PRAGMA foreign_keys enabled.

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

    Examples
    --------
    >>> with managed_database_connection() as conn:
    ...     create_iris_tables(conn)
    ...     # transaction automatically committed upon exit
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
    """Create normalized relational tables and performance indexes for Iris data.

    Creates two normalized tables:
    1. `iris_species`: Dimension table containing botanical taxonomy and metadata.
    2. `iris_measurements`: Fact table containing morphological measurements
       with foreign key referencing `iris_species(species_id)`.

    Creates performance indexes on foreign keys and frequently filtered columns.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.

    Raises
    ------
    sqlite3.DatabaseError
        If table creation or indexing fails.
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
        Must contain feature columns and target/species identifier.

    Returns
    -------
    Tuple[int, int]
        Count of (species_records_inserted, measurement_records_inserted).

    Raises
    ------
    ValueError
        If provided DataFrame has invalid shape or missing required columns.
    sqlite3.DatabaseError
        If database insertion fails.
    """
    create_iris_tables(conn)

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
        # Normalize column names if sklearn standard naming was supplied
        column_map = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "species_id",
            "species": "species_id",
        }
        raw_df.rename(columns=column_map, inplace=True)

    required_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species_id"}
    if not required_cols.issubset(raw_df.columns):
        missing = required_cols - set(raw_df.columns)
        raise ValueError(f"DataFrame missing required measurement columns: {missing}")

    records: List[Tuple[float, float, float, float, int]] = []
    for _, row in raw_df.iterrows():
        records.append((
            float(row["sepal_length"]),
            float(row["sepal_width"]),
            float(row["petal_length"]),
            float(row["petal_width"]),
            int(row["species_id"]),
        ))

    measurement_insert_sql = """
    INSERT INTO iris_measurements (
        sepal_length, sepal_width, petal_length, petal_width, species_id
    ) VALUES (?, ?, ?, ?, ?);
    """

    with conn:
        conn.executemany(measurement_insert_sql, records)

    species_count = len(CANONICAL_SPECIES_METADATA)
    measurement_count = len(records)
    return species_count, measurement_count


def initialize_iris_db(db_path: str = ":memory:") -> sqlite3.Connection:
    """Convenience factory function: connect, create schema, and populate Iris tables.

    Parameters
    ----------
    db_path : str, default=":memory:"
        SQLite database path.

    Returns
    -------
    sqlite3.Connection
        Ready-to-query SQLite connection with 150 Iris observations and taxonomy metadata.
    """
    conn = get_database_connection(db_path)
    populate_iris_tables(conn)
    return conn


# ==============================================================================
# SQL Query 1: Parameterized SELECT Queries with Filtering & Pagination
# ==============================================================================

def select_iris_measurements(
    conn: sqlite3.Connection,
    species_name: Optional[str] = None,
    min_sepal_length: Optional[float] = None,
    max_petal_width: Optional[float] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> pd.DataFrame:
    """Execute a parameterized SELECT query filtering Iris measurements.

    Protects against SQL injection by using parameter placeholders (?) for all
    filter criteria and validated pagination bounds.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    species_name : Optional[str], default=None
        Optional species name filter ('setosa', 'versicolor', 'virginica').
    min_sepal_length : Optional[float], default=None
        Minimum sepal length threshold filter (must be non-negative).
    max_petal_width : Optional[float], default=None
        Maximum petal width threshold filter (must be non-negative).
    limit : Optional[int], default=None
        Maximum number of records to return (must be positive integer).
    offset : Optional[int], default=None
        Pagination offset (must be non-negative integer).

    Returns
    -------
    pd.DataFrame
        Query result set as a pandas DataFrame.

    Raises
    ------
    ValueError
        If threshold parameters or limits are invalid.
    """
    # Parameter boundary validation
    if min_sepal_length is not None and min_sepal_length < 0:
        raise ValueError(f"min_sepal_length must be non-negative, got {min_sepal_length}")
    if max_petal_width is not None and max_petal_width < 0:
        raise ValueError(f"max_petal_width must be non-negative, got {max_petal_width}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    if offset is not None and offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    clauses: List[str] = []
    params: List[Any] = []

    if species_name is not None:
        clauses.append("s.species_name = ?")
        params.append(species_name.strip().lower())

    if min_sepal_length is not None:
        clauses.append("m.sepal_length >= ?")
        params.append(float(min_sepal_length))

    if max_petal_width is not None:
        clauses.append("m.petal_width <= ?")
        params.append(float(max_petal_width))

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    pagination_clause = ""
    if limit is not None:
        pagination_clause += " LIMIT ?"
        params.append(int(limit))
        if offset is not None:
            pagination_clause += " OFFSET ?"
            params.append(int(offset))
    elif offset is not None:
        pagination_clause += " LIMIT -1 OFFSET ?"
        params.append(int(offset))

    query = f"""
    SELECT
        m.sample_id,
        m.sepal_length,
        m.sepal_width,
        m.petal_length,
        m.petal_width,
        m.species_id,
        s.species_name
    FROM iris_measurements AS m
    INNER JOIN iris_species AS s ON m.species_id = s.species_id
    {where_clause}
    ORDER BY m.sample_id ASC
    {pagination_clause};
    """

    return pd.read_sql_query(query, conn, params=params)


# ==============================================================================
# SQL Query 2: Multi-Table Relational JOIN Queries with Enriched Metadata
# ==============================================================================

def join_measurements_with_species(
    conn: sqlite3.Connection,
    species_name: Optional[str] = None,
    min_petal_length: Optional[float] = None,
    order_by: str = "sample_id",
    ascending: bool = True,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Execute a relational INNER JOIN combining measurements with botanical metadata.

    Enriches raw morphology observations with botanical family, native geographic
    distribution, conservation status, and a computed SQL feature:
    `petal_area_cm2` = ROUND(petal_length * petal_width, 2).

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    species_name : Optional[str], default=None
        Filter by species name ('setosa', 'versicolor', 'virginica').
    min_petal_length : Optional[float], default=None
        Minimum petal length threshold (must be non-negative).
    order_by : str, default="sample_id"
        Column to sort by. Must be in ALLOWED_ORDER_COLUMNS.
    ascending : bool, default=True
        Sort direction: True for ASC, False for DESC.
    limit : Optional[int], default=None
        Maximum records to return (must be positive).

    Returns
    -------
    pd.DataFrame
        Joined result set containing measurement attributes and taxonomy metadata.

    Raises
    ------
    ValueError
        If parameters fail validation or invalid order_by column is specified.
    """
    if order_by not in ALLOWED_ORDER_COLUMNS:
        raise ValueError(
            f"Invalid order_by column '{order_by}'. Allowed: {ALLOWED_ORDER_COLUMNS}"
        )
    if min_petal_length is not None and min_petal_length < 0:
        raise ValueError(f"min_petal_length must be non-negative, got {min_petal_length}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {limit}")

    clauses: List[str] = []
    params: List[Any] = []

    if species_name is not None:
        clauses.append("LOWER(s.species_name) = ?")
        params.append(species_name.strip().lower())

    if min_petal_length is not None:
        clauses.append("m.petal_length >= ?")
        params.append(float(min_petal_length))

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    direction = "ASC" if ascending else "DESC"

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(int(limit))

    query = f"""
    SELECT
        m.sample_id,
        m.sepal_length,
        m.sepal_width,
        m.petal_length,
        m.petal_width,
        ROUND(m.petal_length * m.petal_width, 2) AS petal_area_cm2,
        s.species_id,
        s.species_name,
        s.botanical_family,
        s.native_region,
        s.conservation_status
    FROM iris_measurements AS m
    INNER JOIN iris_species AS s
        ON m.species_id = s.species_id
    {where_clause}
    ORDER BY {order_by} {direction}
    {limit_clause};
    """

    return pd.read_sql_query(query, conn, params=params)


# ==============================================================================
# SQL Query 3: Multi-Table Analytical Aggregation (GROUP BY & JOIN)
# ==============================================================================

def select_species_summary_statistics(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Execute an analytical JOIN with GROUP BY computing per-species summary stats.

    Aggregates observation counts, mean dimensions, standard deviation proxies,
    and dimension ranges for each flower taxonomy category.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by species with aggregate morphological statistics.
    """
    query = """
    SELECT
        s.species_id,
        s.species_name,
        s.botanical_family,
        s.native_region,
        s.conservation_status,
        COUNT(m.sample_id) AS sample_count,
        ROUND(AVG(m.sepal_length), 2) AS avg_sepal_length,
        ROUND(AVG(m.sepal_width), 2) AS avg_sepal_width,
        ROUND(AVG(m.petal_length), 2) AS avg_petal_length,
        ROUND(AVG(m.petal_width), 2) AS avg_petal_width,
        ROUND(MIN(m.petal_length), 2) AS min_petal_length,
        ROUND(MAX(m.petal_length), 2) AS max_petal_length,
        ROUND(AVG(m.petal_length * m.petal_width), 2) AS avg_petal_area_cm2
    FROM iris_species AS s
    LEFT JOIN iris_measurements AS m
        ON s.species_id = m.species_id
    GROUP BY
        s.species_id,
        s.species_name,
        s.botanical_family,
        s.native_region,
        s.conservation_status
    ORDER BY s.species_id ASC;
    """

    return pd.read_sql_query(query, conn)


def execute_custom_query(
    conn: sqlite3.Connection,
    query: str,
    params: Sequence[Any] = (),
) -> pd.DataFrame:
    """Safely execute an arbitrary read query with parameter binding.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active SQLite database connection.
    query : str
        SQL query string containing parameter placeholders (?).
    params : Sequence[Any], default=()
        Sequence of bound parameters.

    Returns
    -------
    pd.DataFrame
        Result set formatted as a pandas DataFrame.

    Raises
    ------
    ValueError
        If query is empty or not a string.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    return pd.read_sql_query(query, conn, params=params)


if __name__ == "__main__":
    print("Initializing in-memory Iris SQLite database...")
    db_conn = initialize_iris_db()

    print("\n--- 1. Parameterized SELECT Query (species='setosa', min_sepal_length=5.0) ---")
    df_select = select_iris_measurements(
        db_conn, species_name="setosa", min_sepal_length=5.0, limit=5
    )
    print(df_select.to_string())

    print("\n--- 2. Relational JOIN Query with Botanical Metadata ---")
    df_join = join_measurements_with_species(
        db_conn, min_petal_length=4.5, order_by="petal_length", ascending=False, limit=5
    )
    print(df_join[["sample_id", "species_name", "petal_length", "petal_area_cm2", "native_region"]].to_string())

    print("\n--- 3. Analytical Summary Statistics (GROUP BY + JOIN) ---")
    df_summary = select_species_summary_statistics(db_conn)
    print(df_summary[["species_name", "sample_count", "avg_sepal_length", "avg_petal_length", "avg_petal_area_cm2"]].to_string())

    db_conn.close()
    print("\nAll SQL query functions executed successfully!")
