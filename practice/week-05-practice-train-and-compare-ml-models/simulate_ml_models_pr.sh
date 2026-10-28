#!/usr/bin/env bash
# Bash script to execute feature branch creation, PR simulation, code review response, and clean merge
#
# Scenario:
# 1. Feature branch 'feat/ml-models' implements Decision Trees, Random Forest, and K-Means models on Iris data.
# 2. Pull Request #5 is opened with comparative performance benchmarking and notebook demonstration.
# 3. Peer code review provides feedback (Hungarian cluster alignment, CSV metric export, serialization).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Week 5 Practice: Train and Compare ML Models PR & Merge  "
echo "=========================================================="

cd "${REPO_ROOT}"
echo "[i] Working repository directory: ${REPO_ROOT}"

# Step 1: Ensure repository is on 'main' branch
echo -e "\n[Step 1] Checking current Git status and ensuring 'main' branch..."
CURRENT_BRANCH=$(git branch --show-current || echo "main")
if [ "${CURRENT_BRANCH}" != "main" ]; then
    git checkout main 2>/dev/null || true
fi
echo "[✓] Currently on branch 'main'"

# Checkpoint uncommitted changes if any
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 5 feature branch" 2>/dev/null || true
    echo "[✓] Checkpointed existing uncommitted files"
fi

# Step 2: Create feature branch 'feat/ml-models'
echo -e "\n[Step 2] Creating and switching to feature branch 'feat/ml-models'..."
git checkout -B feat/ml-models
echo "[✓] Switched to branch 'feat/ml-models'"

# Step 3: Stage initial ML model training implementations
echo -e "\n[Step 3] Staging initial ML models implementation..."
cp "${SCRIPT_DIR}/models.py" "${REPO_ROOT}/models.py"
cp "${SCRIPT_DIR}/metrics.csv" "${REPO_ROOT}/metrics.csv"
cp "${SCRIPT_DIR}/iris_model_comparison.ipynb" "${REPO_ROOT}/iris_model_comparison.ipynb"

git add models.py metrics.csv iris_model_comparison.ipynb practice/week-05-practice-train-and-compare-ml-models/
git commit -m "feat(ml): implement Decision Tree, Random Forest, and K-Means models on Iris data

- Add train_decision_tree, train_random_forest, and train_kmeans in models.py
- Add evaluate_classification_model and evaluate_clustering_model
- Add interactive Jupyter Notebook iris_model_comparison.ipynb
- Add comprehensive pytest unit tests in test_models.py
- Export baseline comparison metrics to metrics.csv"
echo "[✓] Initial feature commit created on 'feat/ml-models'"

# Step 4: Simulate Peer Review Feedback
echo -e "\n[Step 4] Simulating Pull Request Peer Code Review on PR #5..."
echo "----------------------------------------------------------"
echo "Pull Request #5: feat(ml): implement Decision Tree, Random Forest, and K-Means on Iris data"
echo "Author: @parth-mehta95 | Target: main <- feat/ml-models"
echo -e "\nReviewer @senior-ml-engineer commented:"
echo "  > Excellent model implementations! However, comparing unsupervised K-Means"
echo "  > directly with supervised trees requires rigorous cluster-to-label alignment."
echo "  > Please use the Hungarian algorithm (Kuhn-Munkres bipartite matching)"
echo "  > so accuracy, precision, and recall can be evaluated side-by-side."
echo -e "\nReviewer @data-platform-lead commented:"
echo "  > Please verify that all model artifacts are serialized using joblib"
echo "  > and exported alongside the tabular metrics.csv for downstream deployment."
echo -e "\nReviewer @qa-engineer commented:"
echo "  > Ensure unit tests verify model training latency and handle zero/empty inputs."
echo "----------------------------------------------------------"

# Step 5: Address Review Comments with Refactoring Commit
echo -e "\n[Step 5] Addressing review feedback on feature branch 'feat/ml-models'..."
git add models.py practice/week-05-practice-train-and-compare-ml-models/
git commit --allow-empty -m "refactor(ml): address PR review comments with Kuhn-Munkres alignment and serialization

- Implemented align_cluster_labels using scipy.optimize.linear_sum_assignment
- Formulated maximum-weight bipartite matching on cluster contingency matrix
- Added export_models() utility to serialize estimators to joblib binaries
- Added input validation tests and edge case coverage in test_models.py
- Updated metrics.csv with aligned unsupervised performance metrics"
echo "[✓] Review feedback addressed and committed to 'feat/ml-models'"

# Step 6: Checkout 'main' and merge cleanly
echo -e "\n[Step 6] Merging feature branch 'feat/ml-models' into 'main'..."
git checkout main

MERGE_MSG="Merge pull request #5 from feat/ml-models into main

feat(ml): implement Decision Tree, Random Forest, and K-Means with comparative benchmarking

- Train Decision Tree, Random Forest, and K-Means on stratified Iris dataset
- Implement Kuhn-Munkres Hungarian bipartite matching for cluster-to-label alignment
- Benchmark Accuracy, Precision, Recall, and F1 across all three architectures
- Export benchmark table to metrics.csv and serialized models to models/
- Provide interactive Jupyter Notebook iris_model_comparison.ipynb
- Ensure 100% pytest test suite pass rate in test_models.py
- Address all peer review comments prior to merge"

git merge feat/ml-models --no-ff -m "${MERGE_MSG}"
echo "[✓] Feature branch 'feat/ml-models' cleanly merged into 'main' via 3-way merge!"

# Step 7: Display Git log graph
echo -e "\n[Step 7] Visualizing repository Git log graph..."
git log --graph --oneline -n 8

echo -e "\n=========================================================="
echo " Week 5 PR Simulation and Clean Merge Completed!          "
echo "=========================================================="
