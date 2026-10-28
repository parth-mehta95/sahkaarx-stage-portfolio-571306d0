# Project: Finalize Iris Pipeline and Deploy 🌸🚀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/sqlite-3.x%20%7C%20relational-lightgrey.svg)](https://www.sqlite.org/)
[![Type Checked](https://img.shields.io/badge/types-mypy%20%7C%20strict-green.svg)](https://mypy-lang.org/)
[![Testing](https://img.shields.io/badge/pytest-100%25%20passing-brightgreen.svg)](https://docs.pytest.org/)
[![PR Status](https://img.shields.io/badge/PR%20%236-merged%20into%20main-purple.svg)](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Task Brief

- **Project Title:** Capstone Finalization: End-to-End Iris ML Pipeline & Deployment
- **Scenario:** Complete the capstone milestone: merge all feature branches, finalize SQL queries and schema DDL, integrate models with Kuhn-Munkres Hungarian cluster alignment, verify all modules work together, write comprehensive documentation, and deploy to GitHub.
- **Repository Context:** **SahkaarX Stage Portfolio Capstone Project** — Final Integration & Capstone Milestone.
- **Key Modules Integrated:**
  1. [`data_loader.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/data_loader.py): Type-hinted data loading and leak-free stratified preprocessing.
  2. [`validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/validation_functions.py): Strict schema, dimension, botany bounds, and dataset pair consistency checks.
  3. [`sql_queries.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/sql_queries.py): Parameterized SELECT queries, relational INNER JOINs, and taxonomic summary statistics.
  4. [`sql_features.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/sql_features.py): High-performance SQL feature engineering, geometric aspect ratios, species benchmark CTEs, and size tier binning.
  5. [`models.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/models.py): Decision Tree, Random Forest, and K-Means training, Hungarian bipartite matching, and artifact serialization.
  6. [`schema.sql`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/schema.sql): Normalized relational DDL schema with tables, performance indexes, analytical views, and seed data.
  7. [`pipeline.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/pipeline.py): Unified end-to-end execution runner tying all pipeline phases together.
  8. [`test_pipeline.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/test_pipeline.py): Comprehensive integration test suite verifying end-to-end functionality.

---

## 🔗 Submission Details

| Field | Detail |
| :--- | :--- |
| **Public GitHub Repository URL** | [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0) |
| **Primary Branch** | `main` |
| **Final Pull Request** | `PR #6: feat(capstone): finalize end-to-end Iris ML pipeline, schema, models, and deployment` |
| **PR Merge Strategy** | Non-fast-forward (`--no-ff`) 3-way merge into `main` |
| **Minimum Artifacts** | `README.md`, `models/` directory, `sql_queries.py` (all verified present & functional) |
| **SQL Schema Artifact** | [`schema.sql`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-05-project-finalize-iris-pipeline-and-deploy/schema.sql) |
| **Model Artifacts Directory** | [`models/`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-05-project-finalize-iris-pipeline-and-deploy/models/) containing `model_manifest.json`, `.gitkeep`, `README.md`, `export_models.py` |
| **Benchmark Metrics Table** | [`metrics.csv`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-05-project-finalize-iris-pipeline-and-deploy/metrics.csv) |
| **Interactive Notebook** | [`iris_model_comparison.ipynb`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-05-project-finalize-iris-pipeline-and-deploy/iris_model_comparison.ipynb) |

---

## 🏗️ End-to-End Pipeline Architecture

The Iris ML Capstone implements a fully integrated, modular, production-grade machine learning lifecycle:

```mermaid
flowchart TD
    subgraph Phase 1 [Phase 1: Scaffolding & Git Control]
        P1A[Git Repository Setup] --> P1B[Pinned requirements.txt]
        P1B --> P1C[.gitignore Artifact Rules]
    end

    subgraph Phase 2 [Phase 2: Ingestion & Validation]
        P2A[data_loader.py: load_iris] --> P2B[Type Contracts & Bunches]
        P2B --> P2C[validation_functions.py: Range & Schema Checks]
    end

    subgraph Phase 3 [Phase 3: Relational Persistence & Feature Engineering]
        P3A[schema.sql: SQLite DDL] --> P3B[sql_queries.py: Parameterized SELECT & JOIN]
        P3B --> P3C[sql_features.py: Ratios, Benchmark CTE & Size Tiers]
        P3C --> P3D[data_loader.preprocess: Leak-Free Scaling]
    end

    subgraph Phase 4 [Phase 4: Model Exploration & Training]
        P4A[Decision Tree max_depth=3]
        P4B[Random Forest n=100]
        P4C[K-Means n_clusters=3]
    end

    subgraph Phase 5 [Phase 5: Evaluation & Bipartite Alignment]
        P5A[Supervised Metrics: Accuracy, Precision, Recall, F1]
        P5B[Kuhn-Munkres Hungarian Algorithm Bipartite Alignment]
        P5C[metrics.csv Benchmark Export]
        P5D[models/ Serialization via joblib]
    end

    subgraph Phase 6 [Phase 6: Integration & Deployment]
        P6A[pipeline.py: Unified Pipeline Runner]
        P6B[test_pipeline.py: 100% Passing Integration Tests]
        P6C[GitHub Repository Deployment on main]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
    Phase 3 --> Phase 4
    Phase 4 --> Phase 5
    Phase 5 --> Phase 6
```

---

## 🗄️ Relational SQL Schema & Database Architecture

The relational storage layer enforces database normalization, referential integrity, and rapid querying:

### Relational Schema Diagram (ERD)

```mermaid
erDiagram
    iris_species ||--o{ iris_measurements : "classifies (1:N)"
    iris_species {
        INTEGER species_id PK "0: setosa, 1: versicolor, 2: virginica"
        TEXT species_name UK "Botanical taxon name"
        TEXT botanical_family "Taxonomic family (Iridaceae)"
        TEXT native_region "Geographic provenance"
        TEXT conservation_status "IUCN status"
    }
    iris_measurements {
        INTEGER sample_id PK "Auto-incrementing surrogate key"
        REAL sepal_length "Sepal length in cm (> 0.0)"
        REAL sepal_width "Sepal width in cm (> 0.0)"
        REAL petal_length "Petal length in cm (> 0.0)"
        REAL petal_width "Petal width in cm (> 0.0)"
        INTEGER species_id FK "References iris_species(species_id)"
    }
```

### DDL Implementation (`schema.sql`)

```sql
-- Dimension Table: Taxonomy & Botanical Metadata
CREATE TABLE IF NOT EXISTS iris_species (
    species_id INTEGER PRIMARY KEY,
    species_name TEXT NOT NULL UNIQUE,
    botanical_family TEXT NOT NULL,
    native_region TEXT NOT NULL,
    conservation_status TEXT NOT NULL
);

-- Fact Table: Physical Morphological Measurements
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

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_measurements_species ON iris_measurements(species_id);
CREATE INDEX IF NOT EXISTS idx_measurements_petal_len ON iris_measurements(petal_length);
CREATE INDEX IF NOT EXISTS idx_measurements_sepal_len ON iris_measurements(sepal_length);
CREATE INDEX IF NOT EXISTS idx_measurements_composite ON iris_measurements(species_id, petal_length, petal_width);
```

### Relational Features & Views
1. **`view_iris_features`**: Computes `sepal_ratio`, `petal_ratio`, `sepal_area_cm2`, `petal_area_cm2`, and `total_area_cm2` using safe `NULLIF` guards against zero division.
2. **`view_species_summary`**: Analytical grouping aggregating sample counts, dimensional means, and average petal surface areas per species.
3. **`view_size_tiers`**: Categorizes observation records into `'Small'`, `'Medium'`, and `'Large'` morphological size tiers via SQL `CASE` expressions.

---

## 📊 Model Training, Evaluation & Benchmarking

Three candidate architectures were trained and evaluated on stratified 80/20 train/test partitions using standard scaling fitted strictly on training data:

### Performance Comparison Table (`metrics.csv`)

| Model | Learning Paradigm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Train Latency (ms) | Inference Latency (ms) | ARI | Silhouette |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | Supervised (Classification) | **96.67%** | 0.9697 | 0.9667 | 0.9666 | 0.9697 | 0.9667 | 0.9666 | 2.145 ms | 0.282 ms | N/A | N/A |
| **Random Forest** | Supervised (Ensemble) | **100.00%** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 74.832 ms | 2.415 ms | N/A | N/A |
| **K-Means (Aligned)** | Unsupervised (Bipartite Aligned) | **86.67%** | 0.8788 | 0.8667 | 0.8658 | 0.8788 | 0.8667 | 0.8658 | 18.420 ms | 0.345 ms | **0.7163** | **0.4599** |

---

## 🧮 Kuhn-Munkres Hungarian Bipartite Cluster Alignment

Unsupervised clustering models like K-Means assign arbitrary cluster identifiers ($\{0, 1, 2\}$) without semantic awareness of true botanical taxa. To compare unsupervised K-Means performance directly against supervised Decision Trees and Random Forests, we formulate the **Maximum Weight Bipartite Matching Problem**:

Given contingency matrix $C \in \mathbb{R}^{K \times K}$ where $C_{i, j}$ represents the count of observations assigned cluster index $i$ whose ground-truth class is $j$:

$$\max_{\pi \in S_K} \sum_{i=1}^K C_{i, \pi(i)}$$

By passing the negated contingency matrix $-C$ into `scipy.optimize.linear_sum_assignment(-contingency)`, we compute the optimal bijection $\pi^*: \text{Cluster} \to \text{Species}$ in $O(K^3)$ time:
- **Cluster 0** $\to$ *setosa* (Class 0, 100% purity)
- **Cluster 1** $\to$ *versicolor* (Class 1)
- **Cluster 2** $\to$ *virginica* (Class 2)

This yields an aligned accuracy of **86.67%**, proving that morphological geometry strongly captures biological taxonomy even without supervision.

---

## 📦 Trained Model Artifacts Directory (`models/`)

The [`models/`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-05-project-finalize-iris-pipeline-and-deploy/models/) directory contains:

1. **`model_manifest.json`**: Machine-readable specification containing model architectures, hyperparameters, feature names, train/test split parameters, and benchmark metric scores.
2. **`README.md`**: Guide explaining model deserialization, input feature scaling requirements, and inference execution.
3. **`export_models.py`**: Automated script to retrain and serialize `.joblib` model estimators.
4. **`.gitkeep`**: Ensures directory persistence in Git version control.

---

## 🔄 Pull Request Review & Merge History

All development followed structured feature branching and formal pull request peer review:

```mermaid
gitGraph
   commit id: "Initial Scaffold"
   branch feat/data-loader
   checkout feat/data-loader
   commit id: "PR #1: typed data loader"
   checkout main
   merge feat/data-loader id: "Merge PR #1"
   branch branch-conflict
   checkout branch-conflict
   commit id: "PR #2: validation module"
   checkout main
   merge branch-conflict id: "Merge PR #2"
   branch sql-queries
   checkout sql-queries
   commit id: "PR #3: relational queries"
   checkout main
   merge sql-queries id: "Merge PR #3"
   branch sql-features
   checkout sql-features
   commit id: "PR #4: feature engineering"
   checkout main
   merge sql-features id: "Merge PR #4"
   branch feat/ml-models
   checkout feat/ml-models
   commit id: "PR #5: train & compare models"
   checkout main
   merge feat/ml-models id: "Merge PR #5"
   branch feat/finalize-capstone-pipeline
   checkout feat/finalize-capstone-pipeline
   commit id: "PR #6: full pipeline & deploy"
   checkout main
   merge feat/finalize-capstone-pipeline id: "Merge PR #6 (Final)"
```

### Summary of Merged Pull Requests:
- **PR #1 (`feat/data-loader`):** Type-hinted `load_iris()` and leak-free `preprocess()` functions.
- **PR #2 (`branch-conflict`):** Robust input validators and simulated Git conflict resolution.
- **PR #3 (`sql-queries`):** Parameterized SQLite queries, relational inner joins, and transaction management.
- **PR #4 (`sql-features`):** SQL aggregation functions, geometric ratios, species benchmark CTE, and size tiers.
- **PR #5 (`feat/ml-models`):** Decision Tree, Random Forest, K-Means, and Kuhn-Munkres cluster alignment.
- **PR #6 (`feat/finalize-capstone-pipeline`):** Final pipeline integration, `schema.sql`, `models/` manifest, unified `pipeline.py`, and complete portfolio deployment.

---

## 🧪 Verification & Testing

To reproduce all results and verify pipeline integrity:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end pipeline runner
python pipeline.py

# 3. Execute the comprehensive integration test suite
pytest test_pipeline.py -v

# 4. Execute unit test suites for individual modules
pytest practice/week-02-practice-feature-branch-with-type-hinted-validat/test_validation_functions.py -v
pytest practice/week-03-practice-intermediate-pull-request-with-code-rev/test_sql_queries.py -v
pytest practice/week-04-practice-sql-feature-engineering-with-pull-reque/test_sql_features.py -v
pytest practice/week-05-practice-train-and-compare-ml-models/test_models.py -v
```

---

## ✅ Deliverables Checklist & Success Criteria

- [x] **Public GitHub Repo with full pipeline:** [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0)
- [x] **All code merged to `main` branch:** Clean Git history with all 6 PRs merged via 3-way merge commits.
- [x] **Trained model artifacts:** `models/` directory with `model_manifest.json`, serialization script, and documentation.
- [x] **SQL schema:** `schema.sql` providing normalized SQLite DDL, performance indexes, and analytical views.
- [x] **Models and SQL queries functional:** 100% test pass rate across all unit and integration test suites.
- [x] **Minimum Artifacts Present:**
  - `README.md` (root and week-05-project)
  - `models` directory (root and week-05-project)
  - `sql_queries.py` (root and week-05-project)