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

### Phase 4: Model Exploration & Training (`train.py`)
- **Objective:** Train and serialize baseline and ensemble classification models.
- **Components:**
  - Baseline model: Multiclass `LogisticRegression` using `lbfgs` solver.
  - Ensemble model: `RandomForestClassifier` with hyperparameter configuration (`n_estimators`, `max_depth`).
  - Serialization of fitted estimators using `joblib` into the `models/` registry for reproducible inference.

### Phase 5: Model Evaluation & Benchmarking (`evaluate.py`)
- **Objective:** Quantify model performance across multiple classification criteria.
- **Components:**
  - Comprehensive performance metrics: Accuracy, Precision, Recall, and Macro-F1 score.
  - Confusion matrix generation and scikit-learn classification reporting.
  - Comparative benchmark evaluation between baseline and ensemble architectures.

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
└── practice/
    ├── week-01-practice-initialize-capstone-repository-with-git/
    │   ├── .gitignore
    │   ├── requirements.txt
    │   ├── README.md
    │   └── iris-ml-pipeline/
    │       ├── setup_git_repo.ps1
    │       ├── setup_git_repo.sh
    │       ├── data/
    │       ├── models/
    │       ├── notebooks/
    │       ├── src/
    │       │   ├── data_loader.py
    │       │   ├── evaluate.py
    │       │   └── train.py
    │       └── tests/
    ├── week-01-practice-type-hinted-data-loading-module/
    │   ├── README.md
    │   └── data_loader.py
    └── week-01-project-capstone-scaffold-python-module-and-git/
        ├── README.md        # Milestone project brief, deliverables & phase breakdown
        └── data_loader.py   # Type-hinted module mirror for practice evaluation
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
