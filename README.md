# Iris ML Pipeline Capstone Scaffold 🌸

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/types-mypy%20%7C%20strict-green.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline foundation for the classic Iris classification task, developed as part of the **SahkaarX Stage Portfolio Capstone Project**.

- **Capstone Repository:** [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0)
- **Primary Branch:** `main`
- **Core Module:** [`data_loader.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/data_loader.py)

---

## 📌 Capstone Pipeline Phases

The capstone project establishes a production-grade machine learning workflow for the Iris classification problem. The architecture is decomposed into six distinct, decoupled lifecycle phases:

```mermaid
flowchart LR
    P1[Phase 1: Project Scaffolding & Git Setup] --> P2[Phase 2: Data Ingestion & Loading]
    P2 --> P3[Phase 3: Data Preprocessing & Scaling]
    P3 --> P4[Phase 4: Model Exploration & Training]
    P4 --> P5[Phase 5: Evaluation & Validation]
    P5 --> P6[Phase 6: CI/CD & Verification]
```

### Phase 1: Project Scaffolding & Git Version Control
- **Objective:** Establish the development environment, standard directory layout, dependency specifications, and version-controlled Git history.
- **Components:**
  - Standardized `.gitignore` preventing data dumps, cache files, and model binaries from contaminating history.
  - Pinned `requirements.txt` containing foundational packages (`numpy`, `pandas`, `scikit-learn`, `joblib`, `pytest`).
  - Strict branch management policies, clean semantic commit messages, and conflict resolution mechanisms.

### Phase 2: Data Ingestion & Loading (`data_loader.py`)
- **Objective:** Create a modular, reusable, type-hinted data loading interface for the Iris dataset.
- **Components:**
  - `load_iris()` function supporting multiple output representations (`as_frame`, `return_X_y`).
  - Explicit type contracts using standard Python typing (`Union`, `Tuple`, `pd.DataFrame`, `pd.Series`, `np.ndarray`, `Bunch`).
  - Schema consistency guaranteeing 150 instances across 4 numeric feature columns (`sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`) and target classes (`setosa`, `versicolor`, `virginica`).

### Phase 3: Data Preprocessing & Feature Engineering (`preprocess()`)
- **Objective:** Standardize continuous features and partition datasets while rigorously preventing data leakage.
- **Components:**
  - `StandardScaler` integration to normalize feature distributions to zero mean ($\mu = 0$) and unit standard deviation ($\sigma = 1$).
  - Stratified train-test partitioning (`test_size=0.2`, `stratify=y`) ensuring balanced class distributions across folds.
  - Leak-free transformation pipeline: the scaler fits solely on the training partition and transforms the test partition.
  - Format preservation: retaining pandas DataFrame metadata (column headers and indices) or NumPy ndarrays based on input types.

### Phase 4: Model Exploration & Training (`models.py`)
- **Objective:** Train and serialize candidate supervised and unsupervised model architectures.
- **Components:**
  - Interpretable classifier: `DecisionTreeClassifier` with `max_depth=3` and Gini impurity splits.
  - Ensemble classifier: `RandomForestClassifier` with $N=100$ trees and feature subspace randomization.
  - Unsupervised clustering: `KMeans` with $K=3$ and inertia minimization.
  - Model serialization: Persisting fitted estimators with `joblib` into `models/*.joblib`.

### Phase 5: Model Evaluation & Benchmarking (`iris_model_comparison.ipynb`, `metrics.csv`)
- **Objective:** Quantify model performance across supervised and unsupervised paradigms.
- **Components:**
  - Kuhn-Munkres Hungarian bipartite algorithm for optimal cluster-to-label alignment.
  - Comprehensive metrics: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1-Score (Macro/Weighted), and Latency.
  - Unsupervised validation: Silhouette Score, Adjusted Rand Index (ARI), and V-Measure.
  - Exportable benchmarking table saved to `metrics.csv`.

### Phase 6: Automated Testing & Continuous Verification (`tests/`)
- **Objective:** Ensure pipeline stability, type safety, and reproducibility.
- **Components:**
  - Automated unit test suites covering data contracts, scaling fidelity, and splitting proportions.
  - Static type checking validation via `mypy` and Python type annotations.
  - Regression testing ensuring deterministic inference across environments.

---

## 📂 Repository Structure

```text
sahkaarx-stage-portfolio-571306d0/
├── .gitignore               # Standard Python, ML, virtualenv, and IDE ignore rules
├── requirements.txt         # Core dependencies (numpy, pandas, scikit-learn, joblib, pytest)
├── README.md                # Capstone pipeline documentation & phase specifications
├── data_loader.py           # Root type-hinted Iris data loader and preprocessing module
├── validation_functions.py  # Root type-hinted Iris validation module
├── sql_queries.py           # Root type-hinted Iris relational SQL query functions module
├── sql_features.py          # Root type-hinted Iris SQL aggregation & feature engineering module
├── models.py                # Root type-hinted Decision Tree, Random Forest & K-Means module
├── metrics.csv              # Root model performance benchmark comparison CSV
├── iris_model_comparison.ipynb # Root interactive Jupyter Notebook
└── practice/
    ├── week-01-practice-initialize-capstone-repository-with-git/
    │   ├── .gitignore
    │   ├── requirements.txt
    │   ├── README.md
    │   └── iris-ml-pipeline/
    ├── week-01-practice-type-hinted-data-loading-module/
    │   ├── README.md
    │   └── data_loader.py
    ├── week-01-project-capstone-scaffold-python-module-and-git/
    │   ├── README.md
    │   └── data_loader.py
    ├── week-02-practice-feature-branch-with-type-hinted-validat/
    │   ├── README.md                      # Feature branch, PR & conflict resolution report
    │   ├── validation_functions.py        # Type-hinted validators (features, labels, pairs)
    │   ├── test_validation_functions.py   # Complete pytest test suite
    │   ├── simulate_branch_and_conflict.ps1 # One-click PowerShell Git conflict simulator
    │   └── simulate_branch_and_conflict.sh  # Cross-platform Bash Git conflict simulator
    ├── week-03-practice-intermediate-pull-request-with-code-rev/
    │   ├── README.md                      # PR review report, SQL schema, and query specs
    │   ├── sql_queries.py                 # Parameterized SELECT, JOIN, and aggregation queries
    │   ├── test_sql_queries.py            # Comprehensive unit test suite
    │   ├── simulate_pr_and_review.ps1     # One-click PowerShell PR and review simulator
    │   └── simulate_pr_and_review.sh      # Cross-platform Bash PR and review simulator
    ├── week-04-practice-sql-feature-engineering-with-pull-reque/
    │   ├── README.md                      # Feature engineering report, PR #4 review & merge logs
    │   ├── sql_features.py                # GROUP BY and JOIN queries for Iris feature engineering
    │   ├── test_sql_features.py           # Comprehensive unit test suite
    │   ├── simulate_sql_features_pr.ps1   # One-click PowerShell PR and merge simulator
    │   └── simulate_sql_features_pr.sh    # Cross-platform Bash PR and merge simulator
    └── week-05-practice-train-and-compare-ml-models/
        ├── README.md                      # Comprehensive ML benchmark report, PR #5 review & merge logs
        ├── models.py                      # Decision Tree, Random Forest & K-Means training & Hungarian alignment
        ├── test_models.py                 # 100% passing pytest unit test suite
        ├── iris_model_comparison.ipynb    # Interactive Jupyter Notebook demonstration
        ├── metrics.csv                    # Tabular benchmark performance comparison table
        ├── simulate_ml_models_pr.ps1      # One-click PowerShell PR and merge simulator
        └── simulate_ml_models_pr.sh       # Cross-platform Bash PR and merge simulator
```

---

## 🛠️ Data Loader Module Specification (`data_loader.py`)

The root module [`data_loader.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/data_loader.py) implements the core data contract for Phases 2 and 3.

### 1. `load_iris`

Loads the Iris dataset with full type safety:

```python
def load_iris(
    as_frame: bool = True,
    return_X_y: bool = True,
) -> Union[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[np.ndarray, np.ndarray],
    pd.DataFrame,
    Bunch,
]:
    """Load the classic Iris dataset with complete type annotations.

    Parameters
    ----------
    as_frame : bool, default=True
        If True, returns pandas DataFrame/Series; if False, returns NumPy arrays or Bunch.
    return_X_y : bool, default=True
        If True, returns (features, target) tuple; if False, returns unified container.

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray], pd.DataFrame, Bunch]
    """
```

### 2. `preprocess`

Applies standard scaling and optional stratified train-test splitting:

```python
def preprocess(
    data: Union[pd.DataFrame, np.ndarray],
    target: Optional[Union[pd.Series, np.ndarray]] = None,
    scaler: Optional[StandardScaler] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = 42,
) -> Union[
    pd.DataFrame,
    np.ndarray,
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]],
    Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[pd.Series, np.ndarray],
        Union[pd.Series, np.ndarray],
    ],
]:
    """Preprocess Iris dataset features by applying standard scaling."""
```

### Usage Examples

```python
from data_loader import load_iris, preprocess

# 1. Load data as pandas DataFrame and Series
X, y = load_iris()
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 2. Scale features directly
X_scaled = preprocess(X)

# 3. Perform stratified train/test split and scaling without leakage
X_train, X_test, y_train, y_test = preprocess(X, y, test_size=0.2, random_state=42)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
```

---

## 🗄️ Relational SQL Query Module (`sql_queries.py`)

The root module [`sql_queries.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/sql_queries.py) implements the relational SQLite layer developed in PR #3 (`sql_queries` branch):

- **Parameterized SELECT:** `select_iris_measurements()` filters by species, continuous thresholds, and pagination limits using parameterized `?` placeholders.
- **Relational INNER JOIN:** `join_measurements_with_species()` joins observation records with taxonomy metadata and computed botanical features (`petal_area_cm2`).
- **Analytical Aggregation:** `select_species_summary_statistics()` calculates per-species descriptive summary statistics.
- **Transaction Safety:** `managed_database_connection()` context manager ensures automatic commit, rollback, and connection cleanup.

```bash
# Run SQL queries demonstration
python sql_queries.py

# Run SQL query test suite
pytest practice/week-03-practice-intermediate-pull-request-with-code-rev/test_sql_queries.py -v
```

---

## 📊 SQL Feature Engineering Module (`sql_features.py`)

The root module [`sql_features.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/sql_features.py) implements the relational feature engineering layer developed in PR #4 (`sql-features` branch):

- **Taxonomic GROUP BY Aggregations:** `aggregate_species_features()` computes observation frequencies, dimensional means, ranges, surface areas, aspect ratios, and applies `HAVING` threshold filtering.
- **Observation Feature Vectors with JOINs & CTE:** `engineer_observation_features()` combines observation records with botanical taxonomy and an aggregate benchmark CTE to compute geometric ratios, surface areas, and relative sample deltas from species averages.
- **Discretized Size Tier Aggregations:** `aggregate_features_by_size_tier()` bins measurements using SQL `CASE` expressions and aggregates morphological distributions across species.
- **Class Separability Analysis:** `compute_feature_separability_summary()` quantifies between-class discriminative power directly in SQL.
- **ML Matrix Extractor:** `get_feature_matrix_and_target()` provides clean, ready-to-train feature matrices `X` and target vectors `y` for scikit-learn models.

```bash
# Run SQL feature engineering demonstration
python sql_features.py

# Run SQL feature engineering unit tests
pytest practice/week-04-practice-sql-feature-engineering-with-pull-reque/test_sql_features.py -v
```

---

## 🧪 Verification & Type Checking

To verify the module locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data_loader.py self-test
python data_loader.py

# 3. Run type checking with mypy
mypy --strict data_loader.py
```

---

## 📜 Git History & Submission

- **Repository URL:** [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0)
- **Branch:** `main`

### Clean Commit Instructions

```bash
git add README.md data_loader.py requirements.txt .gitignore practice/
git commit -m "feat(capstone): complete capstone scaffold with typed data_loader and pipeline documentation"
git push origin main
```
