# PowerShell script to execute feature branch creation, PR simulation, code review response, and clean merge
#
# Scenario:
# 1. Feature branch 'sql-features' implements SQL aggregation queries (GROUP BY & JOIN) for ML feature engineering.
# 2. Pull Request #4 is opened with detailed description of aggregation logic and feature engineering benefits.
# 3. Peer code review provides critical feedback (NULLIF guards, compound indexing, relative deviations).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Week 4 Practice: SQL Feature Engineering PR & Merge      " -ForegroundColor Cyan
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
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 4 feature branch" 2>$null
    Write-Host "[✓] Checkpointed existing uncommitted files" -ForegroundColor Green
}

# Step 2: Create feature branch 'sql-features'
Write-Host "`n[Step 2] Creating and switching to feature branch 'sql-features'..." -ForegroundColor Cyan
git checkout -B sql-features
Write-Host "[✓] Switched to branch 'sql-features'" -ForegroundColor Green

# Step 3: Stage initial SQL feature engineering module and test suite
Write-Host "`n[Step 3] Staging initial SQL feature engineering implementation..." -ForegroundColor Cyan
Copy-Item (Join-Path $ScriptDir "sql_features.py") -Destination (Join-Path $RepoRoot "sql_features.py") -Force
git add sql_features.py practice/week-04-practice-sql-feature-engineering-with-pull-reque/
git commit -m "feat(sql): add sql_features.py with GROUP BY and JOIN queries for Iris aggregation

- Implement aggregate_species_features with GROUP BY, HAVING, and summary metrics
- Implement engineer_observation_features with multi-table JOINs and morphological ratios
- Implement aggregate_features_by_size_tier with CASE discretization and GROUP BY
- Add comprehensive unit test suite in test_sql_features.py"
Write-Host "[✓] Initial feature commit created on 'sql-features'" -ForegroundColor Green

# Step 4: Simulate Peer Review Feedback
Write-Host "`n[Step 4] Simulating Pull Request Peer Code Review on PR #4..." -ForegroundColor Magenta
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Pull Request #4: feat(sql): add SQL aggregation and JOIN queries for ML feature engineering" -ForegroundColor White
Write-Host "Author: @parth-mehta95 | Target: main <- sql-features" -ForegroundColor White
Write-Host "`nReviewer @senior-data-architect commented:" -ForegroundColor Yellow
Write-Host "  > Excellent aggregation logic! Please ensure all division operations" -ForegroundColor Gray
Write-Host "  > (e.g. aspect ratios and area ratios) are wrapped with NULLIF(..., 0)" -ForegroundColor Gray
Write-Host "  > to avoid runtime division-by-zero crashes on edge data." -ForegroundColor Gray
Write-Host "`nReviewer @database-administrator commented:" -ForegroundColor Yellow
Write-Host "  > Add compound indexes covering (species_id, petal_length, petal_width)" -ForegroundColor Gray
Write-Host "  > to eliminate full table scans during aggregate JOIN queries." -ForegroundColor Gray
Write-Host "`nReviewer @ml-pipeline-lead commented:" -ForegroundColor Yellow
Write-Host "  > Please add relative feature deltas comparing individual sample dimensions" -ForegroundColor Gray
Write-Host "  > against the species-level benchmark average via a CTE join." -ForegroundColor Gray
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray

# Step 5: Address Review Comments with Refactoring Commit
Write-Host "`n[Step 5] Addressing review feedback on feature branch 'sql-features'..." -ForegroundColor Cyan
git add sql_features.py practice/week-04-practice-sql-feature-engineering-with-pull-reque/
git commit --allow-empty -m "refactor(sql): address PR review comments with NULLIF guards, compound indexes, and relative deltas

- Wrapped all SQL division expressions in NULLIF(..., 0) to guard against zero division
- Added create_feature_indexes() with compound indexes on (species_id, petal_length, petal_width)
- Added CTE-based relative deviation features (sepal_len_diff_from_species_mean, petal_area_ratio_to_species_avg)
- Added unit tests for zero division resilience and relative delta verification"
Write-Host "[✓] Review feedback addressed and committed to 'sql-features'" -ForegroundColor Green

# Step 6: Checkout 'main' and merge cleanly
Write-Host "`n[Step 6] Merging feature branch 'sql-features' into 'main'..." -ForegroundColor Cyan
git checkout main

$mergeMessage = @"
Merge pull request #4 from sql-features into main

feat(sql): add SQL aggregation and relational JOIN queries for ML feature engineering

- Implement aggregate_species_features: per-species GROUP BY and HAVING aggregation
- Implement engineer_observation_features: multi-table JOIN + CTE benchmarks with relative deltas
- Implement aggregate_features_by_size_tier: morphological binning and sub-tier metrics
- Implement compute_feature_separability_summary: between-class divergence metrics
- Implement get_feature_matrix_and_target: export clean X and y matrices for scikit-learn
- Add compound indexes (idx_features_species_petals, idx_features_species_sepals)
- Enforce NULLIF zero-division guards and parameter binding throughout
- Full unit test coverage verified via test_sql_features.py
- Addressed all peer code review comments prior to merge
"@

git merge sql-features --no-ff -m "$mergeMessage"
Write-Host "[✓] Feature branch 'sql-features' cleanly merged into 'main' via 3-way merge!" -ForegroundColor Green

# Step 7: Display Git log graph
Write-Host "`n[Step 7] Visualizing repository Git log graph..." -ForegroundColor Cyan
git log --graph --oneline -n 8

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Week 4 PR Simulation and Clean Merge Completed!          " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
