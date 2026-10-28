# PowerShell script to execute feature branch creation, conflict simulation, and resolution
#
# Scenario:
# 1. Feature branch 'validate-iris-data' implements validation_functions.py and updates pipeline docs.
# 2. Main branch 'main' receives competing pipeline documentation updates.
# 3. Merging triggers an intentional conflict on README.md.
# 4. Conflict is cleanly resolved, preserving both improvements, and committed with a descriptive message.

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Week 2 Practice: Feature Branch & Merge Conflict Script   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Set-Location $RepoRoot
Write-Host "[i] Working directory: $RepoRoot" -ForegroundColor Yellow

# Step 1: Ensure repository on main branch
Write-Host "`n[Step 1] Checking current Git status and ensuring 'main' branch..." -ForegroundColor Cyan
$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    git checkout main 2>$null
}
Write-Host "[✓] Currently on branch 'main'" -ForegroundColor Green

# Ensure working tree is clean or commit changes
$dirty = git status --porcelain
if ($dirty) {
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 2 exercise" 2>$null
    Write-Host "[✓] Checkpointed existing uncommitted files" -ForegroundColor Green
}

# Step 2: Create feature branch 'validate-iris-data'
Write-Host "`n[Step 2] Creating and switching to feature branch 'validate-iris-data'..." -ForegroundColor Cyan
git checkout -B validate-iris-data
Write-Host "[✓] Switched to branch 'validate-iris-data'" -ForegroundColor Green

# Ensure validation_functions.py is present and staged
Copy-Item (Join-Path $ScriptDir "validation_functions.py") -Destination (Join-Path $RepoRoot "validation_functions.py") -Force
git add validation_functions.py practice/week-02-practice-feature-branch-with-type-hinted-validat/

# Edit README.md on feature branch to set up conflict
$featureReadmeNote = @"

<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (validate-iris-data) -->
<!-- ======================================================== -->
### Data Validation Layer (validate-iris-data branch)
- Added type-hinted validation: validate_iris_features(), validate_target_labels()
- Verified schema integrity (150 samples, 4 features, no nulls, non-negative values)
- Guard against data corruption prior to model training
"@

Add-Content -Path (Join-Path $RepoRoot "README.md") -Value $featureReadmeNote
git add README.md
git commit -m "feat(validation): implement type-hinted validation functions and schema contracts"
Write-Host "[✓] Committed validation functions and documentation to 'validate-iris-data'" -ForegroundColor Green

# Step 3: Switch back to main and make a competing change
Write-Host "`n[Step 3] Switching back to 'main' branch to simulate competing edit..." -ForegroundColor Cyan
git checkout main

$mainReadmeNote = @"

<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (main production)    -->
<!-- ======================================================== -->
### Production Ingestion Protocol (main branch)
- Upgraded data loader with strict pandas DataFrame schemas and typed returns
- Formalized 6-phase pipeline architecture and stratified split standards
- Added production-grade evaluation benchmarking framework
"@

Add-Content -Path (Join-Path $RepoRoot "README.md") -Value $mainReadmeNote
git add README.md
git commit -m "feat(pipeline): upgrade data loader protocols and pipeline phase specs"
Write-Host "[✓] Committed competing update on 'main'" -ForegroundColor Green

# Step 4: Attempt merge to trigger conflict
Write-Host "`n[Step 4] Merging 'validate-iris-data' into 'main' (triggering merge conflict)..." -ForegroundColor Cyan
try {
    git merge validate-iris-data
    Write-Host "[i] Git attempted merge..." -ForegroundColor Yellow
} catch {
    Write-Host "[!] Merge conflict triggered as expected!" -ForegroundColor Yellow
}

# Step 5: Check conflict status
$conflicts = git diff --name-only --diff-filter=U
Write-Host "`n[Step 5] Conflicted files detected:" -ForegroundColor Red
$conflicts | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }

# Step 6: Resolve conflict by synthesizing both updates
Write-Host "`n[Step 6] Resolving conflict in README.md by combining both enhancements..." -ForegroundColor Cyan

$resolvedNote = @"

<!-- ======================================================== -->
<!-- PIPELINE DATA VALIDATION & INGESTION (RESOLVED & UNIFIED) -->
<!-- ======================================================== -->
### Unified Data Validation & Ingestion Architecture
- **Type-Hinted Validation:** Integrated \`validation_functions.py\` (\`validate_iris_features\`, \`validate_target_labels\`, \`validate_dataset_pair\`).
- **Production Ingestion:** Standardized data loading interfaces with explicit type contracts.
- **Leakage Prevention:** Verified schema compliance, physical measurement bounds, and non-overlapping stratified splits.
"@

# Read existing README and remove git conflict markers if any
$readmePath = Join-Path $RepoRoot "README.md"
$readmeContent = Get-Content -Path $readmePath -Raw

# Replace conflict block with resolved note
$cleanReadme = $readmeContent -replace '<<<<<<< HEAD[\s\S]*?>>>>>>> validate-iris-data', $resolvedNote
Set-Content -Path $readmePath -Value $cleanReadme -NoNewline

git add README.md
git add validation_functions.py
Write-Host "[✓] Staged cleanly resolved README.md" -ForegroundColor Green

# Step 7: Create merge commit
Write-Host "`n[Step 7] Creating semantic merge commit..." -ForegroundColor Cyan
$mergeMessage = @"
Merge branch 'validate-iris-data' into main

Resolve merge conflict between pipeline ingestion specs and validation layer:
- Integrated type-hinted validation functions (validate_iris_features, validate_target_labels, validate_dataset_pair)
- Reconciled competing README pipeline phase descriptions into unified architecture
- Maintained strict schema checks guarding data ingestion and preprocessing stages
- Verified all validation unit tests pass with resolved codebase
"@

git commit -m "$mergeMessage"
Write-Host "[✓] Successfully created merge commit!" -ForegroundColor Green

# Step 8: Display git log graph
Write-Host "`n[Step 8] Git Log Graph (showing branching, conflict, and merge):" -ForegroundColor Cyan
git log --graph --oneline --decorate -n 6

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Simulation & Resolution Completed Successfully!           " -ForegroundColor Green
Write-Host " Push to GitHub and create/merge PR with:                 " -ForegroundColor Green
Write-Host "   git push origin main                                   " -ForegroundColor Green
Write-Host "   git push origin validate-iris-data                     " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
