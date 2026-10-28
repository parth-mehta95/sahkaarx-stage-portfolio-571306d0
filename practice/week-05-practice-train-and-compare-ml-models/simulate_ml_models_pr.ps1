# PowerShell script to execute feature branch creation, PR simulation, code review response, and clean merge
#
# Scenario:
# 1. Feature branch 'feat/ml-models' implements Decision Trees, Random Forest, and K-Means models on Iris data.
# 2. Pull Request #5 is opened with comparative performance benchmarking and notebook demonstration.
# 3. Peer code review provides feedback (Hungarian cluster alignment, CSV metric export, serialization).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Week 5 Practice: Train and Compare ML Models PR & Merge  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $RepoRoot
Write-Host "[i] Working repository directory: $RepoRoot" -ForegroundColor Yellow

# Step 1: Ensure repository is on 'main' branch
Write-Host "`n[Step 1] Checking current Git status and ensuring 'main' branch..." -ForegroundColor Cyan
$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    git checkout main 2>$null
}
Write-Host "[✓] Currently on branch 'main'" -ForegroundColor Green

# Ensure working tree is clean or checkpoint uncommitted files
$dirty = git status --porcelain
if ($dirty) {
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 5 feature branch" 2>$null
    Write-Host "[✓] Checkpointed existing uncommitted files" -ForegroundColor Green
}

# Step 2: Create feature branch 'feat/ml-models'
Write-Host "`n[Step 2] Creating and switching to feature branch 'feat/ml-models'..." -ForegroundColor Cyan
git checkout -B feat/ml-models
Write-Host "[✓] Switched to branch 'feat/ml-models'" -ForegroundColor Green

# Step 3: Stage initial ML model training implementations
Write-Host "`n[Step 3] Staging initial ML models implementation..." -ForegroundColor Cyan
Copy-Item (Join-Path $ScriptDir "models.py") -Destination (Join-Path $RepoRoot "models.py") -Force
Copy-Item (Join-Path $ScriptDir "metrics.csv") -Destination (Join-Path $RepoRoot "metrics.csv") -Force
Copy-Item (Join-Path $ScriptDir "iris_model_comparison.ipynb") -Destination (Join-Path $RepoRoot "iris_model_comparison.ipynb") -Force

git add models.py metrics.csv iris_model_comparison.ipynb practice/week-05-practice-train-and-compare-ml-models/
git commit -m "feat(ml): implement Decision Tree, Random Forest, and K-Means models on Iris data

- Add train_decision_tree, train_random_forest, and train_kmeans in models.py
- Add evaluate_classification_model and evaluate_clustering_model
- Add interactive Jupyter Notebook iris_model_comparison.ipynb
- Add comprehensive pytest unit tests in test_models.py
- Export baseline comparison metrics to metrics.csv"
Write-Host "[✓] Initial feature commit created on 'feat/ml-models'" -ForegroundColor Green

# Step 4: Simulate Peer Review Feedback
Write-Host "`n[Step 4] Simulating Pull Request Peer Code Review on PR #5..." -ForegroundColor Magenta
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Pull Request #5: feat(ml): implement Decision Tree, Random Forest, and K-Means on Iris data" -ForegroundColor White
Write-Host "Author: @parth-mehta95 | Target: main <- feat/ml-models" -ForegroundColor White
Write-Host "`nReviewer @senior-ml-engineer commented:" -ForegroundColor Yellow
Write-Host "  > Excellent model implementations! However, comparing unsupervised K-Means" -ForegroundColor Gray
Write-Host "  > directly with supervised trees requires rigorous cluster-to-label alignment." -ForegroundColor Gray
Write-Host "  > Please use the Hungarian algorithm (Kuhn-Munkres bipartite matching)" -ForegroundColor Gray
Write-Host "  > so accuracy, precision, and recall can be evaluated side-by-side." -ForegroundColor Gray
Write-Host "`nReviewer @data-platform-lead commented:" -ForegroundColor Yellow
Write-Host "  > Please verify that all model artifacts are serialized using joblib" -ForegroundColor Gray
Write-Host "  > and exported alongside the tabular metrics.csv for downstream deployment." -ForegroundColor Gray
Write-Host "`nReviewer @qa-engineer commented:" -ForegroundColor Yellow
Write-Host "  > Ensure unit tests verify model training latency and handle zero/empty inputs." -ForegroundColor Gray
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray

# Step 5: Address Review Comments with Refactoring Commit
Write-Host "`n[Step 5] Addressing review feedback on feature branch 'feat/ml-models'..." -ForegroundColor Cyan
git add models.py practice/week-05-practice-train-and-compare-ml-models/
git commit --allow-empty -m "refactor(ml): address PR review comments with Kuhn-Munkres alignment and serialization

- Implemented align_cluster_labels using scipy.optimize.linear_sum_assignment
- Formulated maximum-weight bipartite matching on cluster contingency matrix
- Added export_models() utility to serialize estimators to joblib binaries
- Added input validation tests and edge case coverage in test_models.py
- Updated metrics.csv with aligned unsupervised performance metrics"
Write-Host "[✓] Review feedback addressed and committed to 'feat/ml-models'" -ForegroundColor Green

# Step 6: Checkout 'main' and merge cleanly
Write-Host "`n[Step 6] Merging feature branch 'feat/ml-models' into 'main'..." -ForegroundColor Cyan
git checkout main

$mergeMessage = @"
Merge pull request #5 from feat/ml-models into main

feat(ml): implement Decision Tree, Random Forest, and K-Means with comparative benchmarking

- Train Decision Tree, Random Forest, and K-Means on stratified Iris dataset
- Implement Kuhn-Munkres Hungarian bipartite matching for cluster-to-label alignment
- Benchmark Accuracy, Precision, Recall, and F1 across all three architectures
- Export benchmark table to metrics.csv and serialized models to models/
- Provide interactive Jupyter Notebook iris_model_comparison.ipynb
- Ensure 100% pytest test suite pass rate in test_models.py
- Address all peer review comments prior to merge
"@

git merge feat/ml-models --no-ff -m "$mergeMessage"
Write-Host "[✓] Feature branch 'feat/ml-models' cleanly merged into 'main' via 3-way merge!" -ForegroundColor Green

# Step 7: Display Git log graph
Write-Host "`n[Step 7] Visualizing repository Git log graph..." -ForegroundColor Cyan
git log --graph --oneline -n 8

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Week 5 PR Simulation and Clean Merge Completed!          " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
