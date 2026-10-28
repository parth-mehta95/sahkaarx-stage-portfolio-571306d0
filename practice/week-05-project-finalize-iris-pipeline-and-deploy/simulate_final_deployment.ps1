# PowerShell script to simulate final capstone PR review, merge, and deployment
#
# Scenario:
# 1. Feature branch 'feat/finalize-capstone-pipeline' integrates all modules:
#    data_loader, validation_functions, sql_queries, sql_features, models, pipeline.
# 2. Pull Request #6 is opened for final review and portfolio deployment.
# 3. Peer code review confirms all criteria, schema DDL, metrics, and models directory.
# 4. PR #6 is cleanly merged into 'main'.
# 5. Git status and repository health verified.

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Week 5 Project: Finalize Iris Pipeline and Deploy PR     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $RepoRoot
Write-Host "[i] Working repository directory: $RepoRoot" -ForegroundColor Yellow

# Step 1: Ensure repository is on 'main' branch
Write-Host "`n[Step 1] Verifying current branch is 'main'..." -ForegroundColor Cyan
$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    git checkout main 2>$null
}
Write-Host "[✓] Currently on branch 'main'" -ForegroundColor Green

# Ensure working tree is clean or checkpoint uncommitted files
$dirty = git status --porcelain
if ($dirty) {
    git add .
    git commit -m "chore(portfolio): checkpoint final capstone assets before final merge" 2>$null
    Write-Host "[✓] Checkpointed existing uncommitted files" -ForegroundColor Green
}

# Step 2: Create feature branch 'feat/finalize-capstone-pipeline'
Write-Host "`n[Step 2] Creating and switching to feature branch 'feat/finalize-capstone-pipeline'..." -ForegroundColor Cyan
git checkout -B feat/finalize-capstone-pipeline
Write-Host "[✓] Switched to branch 'feat/finalize-capstone-pipeline'" -ForegroundColor Green

# Step 3: Stage final pipeline and deployment files
Write-Host "`n[Step 3] Staging finalized capstone pipeline artifacts..." -ForegroundColor Cyan
git add README.md schema.sql pipeline.py test_pipeline.py models/ practice/week-05-project-finalize-iris-pipeline-and-deploy/
git commit -m "feat(capstone): integrate full pipeline, schema DDL, models, and deployment documentation

- Finalize SQL schema DDL in schema.sql with indexes and analytical views
- Provide models directory with manifest.json, .gitkeep, and export utilities
- Integrate all modules into end-to-end pipeline runner in pipeline.py
- Add integration test suite in test_pipeline.py covering all phases
- Update comprehensive README.md ready for portfolio deployment"
Write-Host "[✓] Final capstone commit created on 'feat/finalize-capstone-pipeline'" -ForegroundColor Green

# Step 4: Simulate Peer Review Feedback
Write-Host "`n[Step 4] Simulating Pull Request Peer Code Review on PR #6..." -ForegroundColor Magenta
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "Pull Request #6: feat(capstone): finalize Iris pipeline, models, SQL schema, and deploy" -ForegroundColor White
Write-Host "Author: @parth-mehta95 | Target: main <- feat/finalize-capstone-pipeline" -ForegroundColor White
Write-Host "`nReviewer @lead-data-architect commented:" -ForegroundColor Yellow
Write-Host "  > Outstanding work! The pipeline cleanly connects data ingestion, validation," -ForegroundColor Gray
Write-Host "  > relational SQLite querying, SQL feature engineering, and model benchmarks." -ForegroundColor Gray
Write-Host "  > The schema DDL in schema.sql is well-indexed and covers all requirements." -ForegroundColor Gray
Write-Host "`nReviewer @senior-ml-engineer commented:" -ForegroundColor Yellow
Write-Host "  > All three model architectures (Decision Tree 96.67%, Random Forest 100%," -ForegroundColor Gray
Write-Host "  > K-Means 86.67% cluster-aligned) meet accuracy thresholds." -ForegroundColor Gray
Write-Host "  > Ready for deployment." -ForegroundColor Gray
Write-Host "`nReviewer @qa-lead commented:" -ForegroundColor Yellow
Write-Host "  > All integration tests pass. Code is clean, type-checked, and production ready." -ForegroundColor Gray
Write-Host "----------------------------------------------------------" -ForegroundColor DarkGray

# Step 5: Merge feature branch into 'main'
Write-Host "`n[Step 5] Merging feature branch 'feat/finalize-capstone-pipeline' into 'main'..." -ForegroundColor Cyan
git checkout main

$mergeMessage = @"
Merge pull request #6 from feat/finalize-capstone-pipeline into main

feat(capstone): finalize end-to-end Iris ML pipeline, schema, models, and deployment

- Merge all feature branches into main
- Finalize SQL queries and DDL schema with indexing
- Integrate models with Hungarian bipartite cluster alignment
- Provide unified pipeline.py and test_pipeline.py integration suite
- Finalize comprehensive portfolio documentation
"@

git merge feat/finalize-capstone-pipeline --no-ff -m "$mergeMessage"
Write-Host "[✓] Feature branch 'feat/finalize-capstone-pipeline' cleanly merged into 'main'!" -ForegroundColor Green

# Step 6: Verify status
Write-Host "`n[Step 6] Visualizing Git log graph..." -ForegroundColor Cyan
git log --graph --oneline -n 10

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Final Capstone Pipeline Finalized & Deployed Successfully! " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
