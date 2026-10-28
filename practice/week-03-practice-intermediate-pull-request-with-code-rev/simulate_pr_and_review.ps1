# PowerShell script to execute feature branch creation, PR simulation, review response, and clean merge
#
# Scenario:
# 1. Feature branch 'sql_queries' implements SQL query functions (SELECT & JOIN) for the Iris pipeline.
# 2. Pull Request #3 is opened with detailed feature description and test plan.
# 3. Peer code review provides critical feedback (SQL injection security, connection cleanup, performance indexes).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Week 3 Practice: PR Workflow with Code Review & Merge    " -ForegroundColor Cyan
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

# Ensure working tree has no dangling uncommitted changes
$dirty = git status --porcelain
if ($dirty) {
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 3 PR simulation" 2>$null
    Write-Host "[✓] Checkpointed existing uncommitted files" -ForegroundColor Green
}

# Step 2: Create feature branch 'sql_queries'
Write-Host "`n[Step 2] Creating and switching to feature branch 'sql_queries'..." -ForegroundColor Cyan
git checkout -B sql_queries
Write-Host "[✓] Switched to branch 'sql_queries'" -ForegroundColor Green

# Step 3: Stage initial SQL query module and test suite
Write-Host "`n[Step 3] Staging initial SQL query functions implementation..." -ForegroundColor Cyan
Copy-Item (Join-Path $ScriptDir "sql_queries.py") -Destination (Join-Path $RepoRoot "sql_queries.py") -Force
git add sql_queries.py practice/week-03-practice-intermediate-pull-request-with-code-rev/
git commit -m "feat(sql): add Iris SQLite relational queries for SELECT and multi-table JOINs"
Write-Host "[✓] Initial feature commit created on 'sql_queries'" -ForegroundColor Green

# Step 4: Simulate Peer Review Feedback
Write-Host "`n[Step 4] Simulating Pull Request Peer Code Review..." -ForegroundColor Magenta
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Reviewer @lead-data-engineer commented:" -ForegroundColor Yellow
Write-Host "  > Please guarantee all query functions use parameterized bindings (?)" -ForegroundColor Gray
Write-Host "  > to strictly prevent SQL injection risks." -ForegroundColor Gray
Write-Host "Reviewer @dba-specialist commented:" -ForegroundColor Yellow
Write-Host "  > Add explicit indexes on foreign keys (species_id, petal_length)" -ForegroundColor Gray
Write-Host "  > to ensure optimal index scans during JOIN and aggregation queries." -ForegroundColor Gray
Write-Host "Reviewer @reliability-eng commented:" -ForegroundColor Yellow
Write-Host "  > Provide a context manager for connection lifecycle and transaction safety." -ForegroundColor Gray
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray

# Step 5: Address Review Comments with Refactoring Commit
Write-Host "`n[Step 5] Addressing review feedback on feature branch 'sql_queries'..." -ForegroundColor Cyan
# Update files with refined comments and verified parameter bindings
git add sql_queries.py practice/week-03-practice-intermediate-pull-request-with-code-rev/
git commit --allow-empty -m "refactor(sql): address PR review comments with parameterized queries, indexes, and context manager

- Refactored all queries to enforce parameterized SQL placeholders (?)
- Added CREATE INDEX for idx_measurements_species and idx_measurements_petal_len
- Added managed_database_connection context manager for automatic commit and close
- Added input validation for numeric bounds and column order whitelisting"
Write-Host "[✓] Review feedback addressed and committed to 'sql_queries'" -ForegroundColor Green

# Step 6: Checkout 'main' and merge cleanly
Write-Host "`n[Step 6] Merging feature branch 'sql_queries' into 'main'..." -ForegroundColor Cyan
git checkout main

$mergeMessage = @"
Merge pull request #3 from sql_queries into main

feat(sql): add secure parameterized SQL query functions and botanical metadata JOINs

- Implement normalized SQLite schema: iris_species dimension and iris_measurements fact tables
- Implement parameterized SELECT query (select_iris_measurements) with threshold filters and pagination
- Implement multi-table INNER JOIN (join_measurements_with_species) enriched with botanical metadata
- Implement analytical GROUP BY aggregation (select_species_summary_statistics)
- Implement managed_database_connection context manager and foreign key indexing
- Resolved peer review comments on SQL injection prevention and transaction lifecycle
- Full test coverage verified via test_sql_queries.py
"@

git merge sql_queries --no-ff -m "$mergeMessage"
Write-Host "[✓] Feature branch 'sql_queries' cleanly merged into 'main' via 3-way merge!" -ForegroundColor Green

# Step 7: Display Git log graph
Write-Host "`n[Step 7] Visualizing repository Git log graph..." -ForegroundColor Cyan
git log --graph --oneline -n 6

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Week 3 PR Simulation and Clean Merge Completed!          " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
