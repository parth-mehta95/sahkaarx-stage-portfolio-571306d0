# Practice: Feature Branch with Type-Hinted Validation Functions 🌸

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Type Checked](https://img.shields.io/badge/types-mypy%20%7C%20strict-green.svg)](https://mypy-lang.org/)
[![Testing](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![PR Status](https://img.shields.io/badge/PR%20%232-merged%20into%20main-purple.svg)](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Task Brief

- **Objective:** Extend the Iris Machine Learning pipeline repository by creating a dedicated feature branch (`validate-iris-data`), implementing a comprehensive, type-hinted data validation module (`validation_functions.py`), simulating a real-world Git merge conflict, resolving it cleanly, and documenting the pull request resolution strategy.
- **Repository Context:** Part of the **SahkaarX Stage Portfolio Capstone Project** for Phase 2 data ingestion & integrity verification.

---

## 🎯 Task Specifications

### Scenario
Your team needs automated data validation functions for the Iris ML pipeline before passing data downstream to feature engineering, model training, and evaluation. To ensure rigorous code review and git hygiene, you must develop these validators on a separate feature branch (`validate-iris-data`), handle a competing update to `main` by simulating and resolving a merge conflict, and submit a merged pull request.

### Deliverables
1. **Feature branch:** `validate-iris-data` containing [`validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/validation_functions.py).
2. **Type-hinted validator functions:** Module with strict typing for feature matrix schema checks and target label domains.
3. **Merge conflict resolution in PR:** Documented local resolution combining divergent branch updates into a clean merge commit.
4. **Test suite & reproduction scripts:** Automated unit tests and one-click execution scripts (`.ps1` and `.sh`).

### Success Criteria
- [x] Strict PEP 484/526/585 type hints on all validator functions.
- [x] Merge conflict intentionally reproduced and resolved without regression.
- [x] Pull request description outlining conflict resolution strategy and test evidence.

---

## 🔗 Submission Details

| Field | Detail |
| :--- | :--- |
| **Portfolio Repository URL** | [https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0](https://github.com/parth-mehta95/sahkaarx-stage-portfolio-571306d0) |
| **Base Branch** | `main` |
| **Feature Branch** | `validate-iris-data` |
| **Pull Request Title** | `PR #2: feat(validation): add type-hinted validation functions and resolve merge conflict` |
| **Pull Request Status** | `Merged into main` via 3-way merge commit |
| **Primary Module** | [`validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/validation_functions.py) |
| **Unit Test Suite** | [`test_validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/test_validation_functions.py) |
| **Automation Scripts** | [`simulate_branch_and_conflict.ps1`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/simulate_branch_and_conflict.ps1), [`simulate_branch_and_conflict.sh`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/simulate_branch_and_conflict.sh) |

---

## 🏗️ Architectural Role in Iris ML Pipeline

The validation layer serves as an immutable gatekeeper between raw data loading and downstream model operations, preventing silent data corruption, silent NaNs, dimension mismatches, and target label leakage:

```mermaid
flowchart TD
    subgraph Data Ingestion Phase
        DL["Data Ingestion<br/>(load_iris)"]
    end

    subgraph Data Validation Layer [Feature Branch: validate-iris-data]
        direction TB
        VF["validate_iris_features()<br/>• 2D shape (N, 4)<br/>• Non-negative floats<br/>• Zero NaNs / Infs<br/>• Physical botany bounds"]
        VT["validate_target_labels()<br/>• 1D vector (N,)<br/>• Classes {0, 1, 2} or names<br/>• Zero nulls / NaNs"]
        VP["validate_dataset_pair()<br/>• Length alignment: len(X) == len(y)"]
        VS["validate_train_test_split()<br/>• No unseen classes in test<br/>• Non-empty partitions"]
        
        DL --> VP
        VP --> VF
        VP --> VT
        VT --> VS
    end

    subgraph Downstream Pipeline
        SC["Feature Scaling<br/>(preprocess / StandardScaler)"]
        TR["Model Training<br/>(train.py)"]
        EV["Evaluation & Validation<br/>(evaluate.py)"]
        
        VS --> SC
        SC --> TR
        TR --> EV
    end
```

---

## 🧩 Type-Hinted Validation Module Specification

The [`validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/validation_functions.py) module provides the following type contracts and validation rules:

### 1. `validate_iris_features`
```python
def validate_iris_features(
    features: Union[pd.DataFrame, np.ndarray],
    expected_dim: int = 4,
    allow_nan: bool = False,
    min_samples: int = 1,
    check_bounds: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
```
- **Guarantees:**
  - Verifies input is a 2D matrix (`ndim == 2`).
  - Enforces exactly `expected_dim` columns (4 features for Iris).
  - Enforces sample count $\ge$ `min_samples`.
  - Rejects missing values (`NaN`, `None`, and $\pm\infty$).
  - Enforces numeric data types and strictly positive botany measurements ($> 0.0\text{ cm}$).
  - Optional physical botany bounds validation:
    - Sepal length: $[3.5, 9.0]\text{ cm}$
    - Sepal width: $[1.5, 5.0]\text{ cm}$
    - Petal length: $[0.5, 8.0]\text{ cm}$
    - Petal width: $[0.05, 3.5]\text{ cm}$

### 2. `validate_target_labels`
```python
def validate_target_labels(
    target: Union[pd.Series, np.ndarray, Sequence[Any]],
    expected_classes: Optional[Set[Any]] = None,
    min_samples: int = 1,
) -> Union[pd.Series, np.ndarray]:
```
- **Guarantees:**
  - Verifies input is a 1D vector (or single-column vector flattened to 1D).
  - Enforces sample count $\ge$ `min_samples`.
  - Rejects `NaN`, `None`, and null entries.
  - Automatically identifies whether labels are numeric indices (`{0, 1, 2}`) or botanical strings (`{'setosa', 'versicolor', 'virginica'}`) and raises `TargetValidationError` on out-of-domain classes.

### 3. `validate_dataset_pair`
```python
def validate_dataset_pair(
    features: Union[pd.DataFrame, np.ndarray],
    target: Union[pd.Series, np.ndarray, Sequence[Any]],
    expected_dim: int = 4,
    min_samples: int = 1,
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
```
- **Guarantees:**
  - Jointly runs feature and target validation.
  - Guarantees sample count alignment: $\text{len}(features) == \text{len}(target)$.
  - Raises `DatasetMismatchError` if length parity is violated.

### 4. `validate_train_test_split`
```python
def validate_train_test_split(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    min_train_ratio: float = 0.5,
) -> bool:
```
- **Guarantees:**
  - Validates dimension alignment between train and test feature matrices.
  - Verifies minimum train partition ratio ($\ge 50\%$).
  - Prevents target distribution leakage by guaranteeing that no classes appear in the test partition that were absent from training.

### 5. `validate_prediction_probabilities`
```python
def validate_prediction_probabilities(
    probabilities: Union[np.ndarray, pd.DataFrame],
    num_classes: int = 3,
    tol: float = 1e-4,
) -> np.ndarray:
```
- **Guarantees:**
  - Validates 2D probability matrix shape $(N, 3)$.
  - Checks bounding constraint: $\forall p_{i,j} \in [0.0, 1.0]$.
  - Enforces row normalization: $\sum_{j=1}^3 p_{i,j} = 1.0 \pm 10^{-4}$.

---

## ⚔️ Merge Conflict Simulation & Resolution Walkthrough

### 1. Divergence Scenario
1. **Feature Branch (`validate-iris-data`):**
   - Implemented [`validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/validation_functions.py).
   - Appended a **Data Validation Layer** protocol section to `README.md`.
   - Committed with message: `feat(validation): implement type-hinted validation functions and schema contracts`.
2. **Main Branch (`main`):**
   - Meanwhile, on the production `main` branch, another engineer updated pipeline specifications and data loading conventions.
   - Appended a competing **Production Ingestion Protocol** section to the same location in `README.md`.
   - Committed with message: `feat(pipeline): upgrade data loader protocols and pipeline phase specs`.

### 2. Triggering the Merge Conflict
Attempting to merge the feature branch into `main`:
```bash
git checkout main
git merge validate-iris-data
```
Git triggers a conflict on `README.md`:
```text
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit the result.
```

### 3. Inspection of Conflict Markers
Inspecting `git diff` reveals overlapping edits between `HEAD` (`main`) and `validate-iris-data`:

```markdown
<<<<<<< HEAD
<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (main production)    -->
<!-- ======================================================== -->
### Production Ingestion Protocol (main branch)
- Upgraded data loader with strict pandas DataFrame schemas and typed returns
- Formalized 6-phase pipeline architecture and stratified split standards
- Added production-grade evaluation benchmarking framework
=======
<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (validate-iris-data) -->
<!-- ======================================================== -->
### Data Validation Layer (validate-iris-data branch)
- Added type-hinted validation: validate_iris_features(), validate_target_labels()
- Verified schema integrity (150 samples, 4 features, no nulls, non-negative values)
- Guard against data corruption prior to model training
>>>>>>> validate-iris-data
```

### 4. Conflict Resolution Strategy
Rather than discarding either developer's work, a **synthesized union resolution strategy** was applied:
- Reconciled both sections into a unified **Data Validation & Ingestion Architecture** section.
- Kept the new validation module integration while preserving the upgraded production loader contracts and pipeline roadmap.
- Validated all tests pass across the entire pipeline.

```markdown
<!-- ======================================================== -->
<!-- PIPELINE DATA VALIDATION & INGESTION (RESOLVED & UNIFIED) -->
<!-- ======================================================== -->
### Unified Data Validation & Ingestion Architecture
- **Type-Hinted Validation:** Integrated `validation_functions.py` (`validate_iris_features`, `validate_target_labels`, `validate_dataset_pair`).
- **Production Ingestion:** Standardized data loading interfaces with explicit type contracts.
- **Leakage Prevention:** Verified schema compliance, physical measurement bounds, and non-overlapping stratified splits.
```

### 5. Semantic Merge Commit
The resolved file was staged and committed with a comprehensive commit message:
```bash
git add README.md validation_functions.py
git commit -m "Merge branch 'validate-iris-data' into main

Resolve merge conflict between pipeline ingestion specs and validation layer:
- Integrated type-hinted validation functions (validate_iris_features, validate_target_labels, validate_dataset_pair)
- Reconciled competing README pipeline phase descriptions into unified architecture
- Maintained strict schema checks guarding data ingestion and preprocessing stages
- Verified all validation unit tests pass with resolved codebase"
```

---

## 🚀 Pull Request Template & Submission Evidence

### Pull Request Summary
- **PR Title:** `feat(validation): add type-hinted validation functions and resolve merge conflict`
- **Target Branch:** `main` $\leftarrow$ **Source Branch:** `validate-iris-data`
- **PR Description:**
  > ### Summary of Changes
  > - **Added Validation Module:** Implemented `validation_functions.py` providing strict type annotations (`validate_iris_features`, `validate_target_labels`, `validate_dataset_pair`, `validate_train_test_split`, `validate_prediction_probabilities`).
  > - **Conflict Resolution:** Resolved merge conflict in `README.md` caused by parallel updates to pipeline specs on `main`. Synthesized both changes without data loss.
  > - **Verification:** 100% unit test coverage via `test_validation_functions.py`, zero `mypy --strict` errors.
  >
  > ### Conflict Resolution Strategy
  > A 3-way merge conflict occurred in `README.md`. We reconciled the divergent documentation by uniting the validation layer documentation with the main branch's ingestion roadmap.
  >
  > ### Checklist
  > - [x] All functions include complete type annotations.
  > - [x] Unit tests cover positive cases and all boundary/corruption failure modes.
  > - [x] Git merge commit contains detailed resolution explanation.

### Git Log Graph Representation
```text
*   0a4b7f2 (HEAD -> main, origin/main) Merge branch 'validate-iris-data' into main
|\  
| * e8c1d29 (validate-iris-data) feat(validation): implement type-hinted validation functions and schema contracts
* | 3f1a9b4 feat(pipeline): upgrade data loader protocols and pipeline phase specs
|/  
* 874e1d9 feat(init): initialize iris-ml-pipeline project structure and configuration
```

---

## 🧪 Verification & Test Results

The accompanying test suite [`test_validation_functions.py`](file:///d:/challengers%20testing/28-oct-2026/sahkaarx-stage-portfolio-571306d0/practice/week-02-practice-feature-branch-with-type-hinted-validat/test_validation_functions.py) covers all edge cases:

| Test Case | Scenario / Condition Checked | Result |
| :--- | :--- | :--- |
| `test_valid_dataframe_passes` | Canonical Iris DataFrame `(150, 4)` | ✅ PASS |
| `test_valid_ndarray_passes` | Canonical Iris NumPy ndarray `(150, 4)` | ✅ PASS |
| `test_invalid_type_raises_type_error` | Raw Python nested lists rejected with `TypeError` | ✅ PASS |
| `test_wrong_dimensions_raises_feature_error` | 1D vector rejected with `FeatureValidationError` | ✅ PASS |
| `test_wrong_column_count_raises_feature_error` | 3 features instead of 4 rejected | ✅ PASS |
| `test_nan_values_rejected_by_default` | Injected `NaN` detected and rejected | ✅ PASS |
| `test_infinite_values_rejected` | Injected $\pm\infty$ detected and rejected | ✅ PASS |
| `test_negative_values_rejected` | Negative measurements ($< 0$) rejected | ✅ PASS |
| `test_non_numeric_dtype_rejected` | String columns rejected with informative error | ✅ PASS |
| `test_botany_physical_bounds` | Sepal length $> 9.0\text{ cm}$ rejected when bounds check is enabled | ✅ PASS |
| `test_valid_numeric_series_passes` | Integer class series `{0, 1, 2}` passes | ✅ PASS |
| `test_valid_string_labels_passes` | String class series `{'setosa', 'versicolor', 'virginica'}` passes | ✅ PASS |
| `test_invalid_label_raises_error` | Label `99` rejected with `TargetValidationError` | ✅ PASS |
| `test_nan_target_raises_error` | Missing target label rejected | ✅ PASS |
| `test_2d_multiclass_target_rejected` | 2D one-hot target matrix rejected (requires 1D vector) | ✅ PASS |
| `test_mismatched_sample_counts_raises_error` | Features (150 rows) vs Target (100 labels) raises `DatasetMismatchError` | ✅ PASS |
| `test_valid_split_passes` | Balanced train/test partitions validate successfully | ✅ PASS |
| `test_unseen_class_in_test_raises_error` | Test partition with classes absent from train split rejected | ✅ PASS |
| `test_valid_probabilities_pass` | Probabilities summing to $1.0 \pm 10^{-4}$ pass | ✅ PASS |
| `test_row_sum_not_one_raises_error` | Non-normalized rows rejected | ✅ PASS |

---

## 💻 Automated Replication Scripts

To reproduce this exact Git workflow (feature branching, conflict generation, resolution, and commit), run either script:

### Windows PowerShell:
```powershell
.\practice\week-02-practice-feature-branch-with-type-hinted-validat\simulate_branch_and_conflict.ps1
```

### Linux / macOS Bash:
```bash
bash practice/week-02-practice-feature-branch-with-type-hinted-validat/simulate_branch_and_conflict.sh
```