# Practice: Initialize Capstone Repository with Git

## Task Brief
Set up the GitHub repository for the Iris ML pipeline project and resolve a simulated merge conflict.

## Scenario
Your team is starting the Iris classification project; you must initialize the repo and handle conflicting branch merges.

## Deliverables
- GitHub repository with README
- Resolved merge conflict in main branch

## Success Criteria
- Repo is public and contains project structure
- Conflict resolution is documented in commit

---

## Submission Details

- **Public Repository Name:** `iris-ml-pipeline`
- **Public GitHub URL:** [https://github.com/parth-mehta95/iris-ml-pipeline](https://github.com/parth-mehta95/iris-ml-pipeline)
- **Primary Branch:** `main`
- **Feature Branch:** `feature/model-enhancement`

---

## Project Structure

```text
iris-ml-pipeline/
├── .gitignore               # Standard Python/ML ignore rules
├── requirements.txt         # Core dependencies (pandas, scikit-learn, joblib, pytest)
├── README.md                # Comprehensive documentation and conflict resolution log
├── setup_git_repo.ps1       # Automated one-click setup script (PowerShell)
├── setup_git_repo.sh        # Automated one-click setup script (Bash)
├── data/                    # Dataset directory
│   └── .gitkeep             # Version control anchor
├── models/                  # Trained model artifacts
│   └── .gitkeep             # Version control anchor
├── notebooks/               # Jupyter notebooks
│   └── .gitkeep             # Version control anchor
├── src/                     # Production source code
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Type-hinted load_iris() and preprocess()
│   ├── evaluate.py          # Metric calculation (Accuracy, F1, Precision, Recall)
│   └── train.py             # Configurable model training (LogisticRegression & RandomForest)
└── tests/                   # Test suite
    ├── __init__.py
    └── test_pipeline.py     # End-to-end pipeline verification tests
```

---

## Conflict Resolution Documentation

### 1. Divergence Scenario
- **Main Branch (`main`):** Implemented the baseline `LogisticRegression` classifier with `lbfgs` solver (`feat(models): enhance LogisticRegression baseline with lbfgs solver`).
- **Feature Branch (`feature/model-enhancement`):** Introduced ensemble `RandomForestClassifier` with hyperparameter configuration (`feat(models): add RandomForestClassifier with hyperparameter tuning`).

### 2. Conflicted Files
Both branches introduced conflicting edits to:
- `src/train.py`: Overlapping model instantiation and CLI arguments.
- `README.md`: Overlapping model documentation and benchmark metrics.

### 3. Resolution
- In `src/train.py`, merged both branches into a configurable architecture accepting `--model [logistic_regression|random_forest]`.
- In `README.md`, combined documentation for both models and added comparative benchmark evaluation metrics.

### 4. Merge Commit Message
```text
Merge branch 'feature/model-enhancement' into main

Resolve merge conflict between baseline and ensemble model pipelines:
- Integrated both LogisticRegression baseline and RandomForestClassifier ensemble
- Unified pipeline CLI to allow model selection via --model flag
- Combined README performance benchmarks and architecture documentation
- Verified all pipeline unit tests pass with resolved codebase
```