-- ==============================================================================
-- Iris ML Pipeline Relational Database Schema (SQLite 3.x)
-- SahkaarX Stage Portfolio Capstone Project
-- ==============================================================================
-- PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------------------
-- Dimension Table: Taxonomy & Botanical Metadata
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS iris_species (
    species_id INTEGER PRIMARY KEY,
    species_name TEXT NOT NULL UNIQUE,
    botanical_family TEXT NOT NULL,
    native_region TEXT NOT NULL,
    conservation_status TEXT NOT NULL
);

-- ------------------------------------------------------------------------------
-- Fact Table: Physical Morphological Measurements
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- Performance & Filter Indexes
-- ------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_measurements_species
    ON iris_measurements(species_id);

CREATE INDEX IF NOT EXISTS idx_measurements_petal_len
    ON iris_measurements(petal_length);

CREATE INDEX IF NOT EXISTS idx_measurements_sepal_len
    ON iris_measurements(sepal_length);

CREATE INDEX IF NOT EXISTS idx_measurements_composite
    ON iris_measurements(species_id, petal_length, petal_width);

-- ------------------------------------------------------------------------------
-- Canonical Botanical Seed Data
-- ------------------------------------------------------------------------------
INSERT OR IGNORE INTO iris_species (species_id, species_name, botanical_family, native_region, conservation_status)
VALUES
    (0, 'setosa', 'Iridaceae', 'North America / Arctic', 'Least Concern'),
    (1, 'versicolor', 'Iridaceae', 'Eastern North America', 'Least Concern'),
    (2, 'virginica', 'Iridaceae', 'Southeastern United States', 'Vulnerable');

-- ------------------------------------------------------------------------------
-- Analytical View: Feature Engineering & Ratios
-- ------------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS view_iris_features AS
SELECT
    m.sample_id,
    s.species_name,
    m.sepal_length,
    m.sepal_width,
    m.petal_length,
    m.petal_width,
    ROUND(m.sepal_length / NULLIF(m.sepal_width, 0), 4) AS sepal_ratio,
    ROUND(m.petal_length / NULLIF(m.petal_width, 0), 4) AS petal_ratio,
    ROUND(m.sepal_length * m.sepal_width, 4) AS sepal_area_cm2,
    ROUND(m.petal_length * m.petal_width, 4) AS petal_area_cm2,
    ROUND((m.sepal_length * m.sepal_width) + (m.petal_length * m.petal_width), 4) AS total_area_cm2,
    m.species_id
FROM iris_measurements m
INNER JOIN iris_species s ON m.species_id = s.species_id;

-- ------------------------------------------------------------------------------
-- Analytical View: Species Summary Benchmarks
-- ------------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS view_species_summary AS
SELECT
    s.species_id,
    s.species_name,
    COUNT(m.sample_id) AS sample_count,
    ROUND(AVG(m.sepal_length), 3) AS avg_sepal_length,
    ROUND(AVG(m.sepal_width), 3) AS avg_sepal_width,
    ROUND(AVG(m.petal_length), 3) AS avg_petal_length,
    ROUND(AVG(m.petal_width), 3) AS avg_petal_width,
    ROUND(AVG(m.petal_length * m.petal_width), 3) AS avg_petal_area_cm2
FROM iris_species s
LEFT JOIN iris_measurements m ON s.species_id = m.species_id
GROUP BY s.species_id, s.species_name;

-- ------------------------------------------------------------------------------
-- Analytical View: Size Tier Bins
-- ------------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS view_size_tiers AS
SELECT
    sample_id,
    species_name,
    petal_area_cm2,
    CASE
        WHEN petal_area_cm2 < 2.0 THEN 'Small'
        WHEN petal_area_cm2 BETWEEN 2.0 AND 7.0 THEN 'Medium'
        ELSE 'Large'
    END AS size_tier
FROM view_iris_features;
