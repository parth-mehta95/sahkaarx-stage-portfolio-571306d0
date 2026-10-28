# Iris ML Pipeline 🌸

An end-to-end, production-ready Machine Learning pipeline for the classic Iris classification task, developed as part of the SahkaarX Capstone Project.

**Repository URL:** [https://github.com/parth-mehta95/iris-ml-pipeline](https://github.com/parth-mehta95/iris-ml-pipeline)

---

## Project Overview

The `iris-ml-pipeline` repository demonstrates modern MLOps and software engineering practices for machine learning pipelines:
- **Type-Hinted Data Engineering:** Strict type annotations and zero data-leakage preprocessing.
- **Configurable Architectures:** Support for baseline (`LogisticRegression`) and ensemble (`RandomForestClassifier`) models.
- **Automated Evaluation:** Granular performance tracking (Accuracy, Precision, Recall, Macro-F1).
- **Robust Git Workflow:** Branching strategy with documented merge conflict resolution.

---

## Project Structure

```text
iris-ml-pipeline/
├── .gitignore               # Comprehensive Git ignore rules for ML/Python
├── requirements.txt         # Project dependencies (pinned minimum versions)
├── README.md                # Project documentation and conflict resolution log
├── data/                    # Dataset directory
│   └── .gitkeep             # Version control anchor (raw data ignored)
├── models/                  # Serialized model artifacts (.pkl, .joblib)
│   └── .gitkeep             # Version control anchor (binaries ignored)
├── notebooks/               # Exploratory Data Analysis & experiments
│   └── .gitkeep             # Version control anchor
├── src/                     # Modular source code
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Type-hinted dataset loading & preprocessing
│   ├── evaluate.py          # Metric calculation & reporting
│   └── train.py             # Configurable model training & artifact saving
└── tests/                   # Automated unit and integration tests
    ├── __init__.py
    └── test_pipeline.py     # Pipeline validation test suite
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/parth-mehta95/iris-ml-pipeline.git
cd iris-ml-pipeline
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Pipeline Execution

### Train Baseline Model (Logistic Regression)
```bash
python src/train.py --model logistic_regression --test-size 0.2 --output models/baseline_lr.pkl
```

### Train Enhanced Ensemble Model (Random Forest)
```bash
python src/train.py --model random_forest --test-size 0.2 --output models/ensemble_rf.pkl
```

### Run Unit Tests
```bash
pytest tests/ -v
```

---

## Git Workflow & Merge Conflict Resolution

### 1. Scenario Background
During the initial capstone development, parallel feature development was conducted:
1. **Branch `main`:** Focused on establishing the baseline `LogisticRegression` pipeline and core README documentation.
2. **Branch `feature/model-enhancement`:** Implemented an ensemble `RandomForestClassifier` with hyperparameter configuration and extended documentation.

### 2. Conflicting Changes
Both branches simultaneously modified overlapping sections in:
- `src/train.py`: Modified the model initialization and CLI parser options.
- `README.md`: Modified the pipeline execution guidelines and performance benchmark targets.

```text
<<<<<<< HEAD (main)
model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=random_state)
=======
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
>>>>>>> feature/model-enhancement
```

### 3. Resolution Strategy
Rather than discarding either implementation:
- **`src/train.py` Resolution:** Unified the codebase to support both model types dynamically via a `--model` command-line argument.
- **`README.md` Resolution:** Consolidated documentation to feature instructions for both baseline and ensemble workflows, displaying a unified comparative benchmark table.

### 4. Merge Commit Details
- **Base Branch:** `main`
- **Merged Branch:** `feature/model-enhancement`
- **Resolution Commit Message:**
```text
Merge branch 'feature/model-enhancement' into main

Resolve merge conflict between baseline and ensemble model pipelines:
- Integrated both LogisticRegression baseline and RandomForestClassifier ensemble
- Unified pipeline CLI to allow model selection via --model flag
- Combined README performance benchmarks and architecture documentation
- Verified all pipeline unit tests pass with resolved codebase
```

---

## Model Benchmark Summary

| Model Architecture | Hyperparameters | Test Accuracy | Macro F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | `solver='lbfgs'`, `max_iter=200` | 96.67% | 0.9665 | Baseline |
| **Random Forest** | `n_estimators=100`, `max_depth=4` | 96.67% | 0.9665 | Enhanced |

---

## License & Author
- **Author:** Parth Mehta ([@parth-mehta95](https://github.com/parth-mehta95))
- **Project:** SahkaarX Capstone Machine Learning Pipeline
