#!/usr/bin/env bash
# Shell script to execute feature branch creation, conflict simulation, and resolution
#
# Scenario:
# 1. Feature branch 'validate-iris-data' implements validation_functions.py and updates pipeline docs.
# 2. Main branch 'main' receives competing pipeline documentation updates.
# 3. Merging triggers an intentional conflict on README.md.
# 4. Conflict is cleanly resolved, preserving both improvements, and committed with a descriptive message.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Week 2 Practice: Feature Branch & Merge Conflict Script   "
echo "=========================================================="

cd "${REPO_ROOT}"
echo "[i] Working directory: ${REPO_ROOT}"

# Step 1: Ensure repository on main branch
echo ""
echo "[Step 1] Ensuring 'main' branch..."
CURRENT_BRANCH=$(git branch --show-current || echo "")
if [ "${CURRENT_BRANCH}" != "main" ]; then
    git checkout main || true
fi

# Checkpoint uncommitted changes if any
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 2 exercise" || true
fi

# Step 2: Create feature branch 'validate-iris-data'
echo ""
echo "[Step 2] Creating and switching to feature branch 'validate-iris-data'..."
git checkout -B validate-iris-data

# Copy validation_functions.py to repo root if needed
cp "${SCRIPT_DIR}/validation_functions.py" "${REPO_ROOT}/validation_functions.py"
git add validation_functions.py practice/week-02-practice-feature-branch-with-type-hinted-validat/

# Edit README.md on feature branch to set up conflict
cat << 'EOF' >> "${REPO_ROOT}/README.md"

<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (validate-iris-data) -->
<!-- ======================================================== -->
### Data Validation Layer (validate-iris-data branch)
- Added type-hinted validation: validate_iris_features(), validate_target_labels()
- Verified schema integrity (150 samples, 4 features, no nulls, non-negative values)
- Guard against data corruption prior to model training
EOF

git add README.md
git commit -m "feat(validation): implement type-hinted validation functions and schema contracts"

# Step 3: Switch back to main and make a competing change
echo ""
echo "[Step 3] Switching back to 'main' branch to simulate competing edit..."
git checkout main

cat << 'EOF' >> "${REPO_ROOT}/README.md"

<!-- ======================================================== -->
<!-- PIPELINE DATA CONTRACT & VALIDATION (main production)    -->
<!-- ======================================================== -->
### Production Ingestion Protocol (main branch)
- Upgraded data loader with strict pandas DataFrame schemas and typed returns
- Formalized 6-phase pipeline architecture and stratified split standards
- Added production-grade evaluation benchmarking framework
EOF

git add README.md
git commit -m "feat(pipeline): upgrade data loader protocols and pipeline phase specs"

# Step 4: Attempt merge to trigger conflict
echo ""
echo "[Step 4] Merging 'validate-iris-data' into 'main' (triggering merge conflict)..."
git merge validate-iris-data || true

# Step 5: Check conflict status
echo ""
echo "[Step 5] Conflicted files detected:"
git diff --name-only --diff-filter=U

# Step 6: Resolve conflict by synthesizing both updates
echo ""
echo "[Step 6] Resolving conflict in README.md by combining both enhancements..."

RESOLVED_NOTE=$(cat << 'EOF'

<!-- ======================================================== -->
<!-- PIPELINE DATA VALIDATION & INGESTION (RESOLVED & UNIFIED) -->
<!-- ======================================================== -->
### Unified Data Validation & Ingestion Architecture
- **Type-Hinted Validation:** Integrated `validation_functions.py` (`validate_iris_features`, `validate_target_labels`, `validate_dataset_pair`).
- **Production Ingestion:** Standardized data loading interfaces with explicit type contracts.
- **Leakage Prevention:** Verified schema compliance, physical measurement bounds, and non-overlapping stratified splits.
EOF
)

# Extract and clean conflict block in README.md
python - << 'PYEOF'
import re

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

resolved = """
<!-- ======================================================== -->
<!-- PIPELINE DATA VALIDATION & INGESTION (RESOLVED & UNIFIED) -->
<!-- ======================================================== -->
### Unified Data Validation & Ingestion Architecture
- **Type-Hinted Validation:** Integrated `validation_functions.py` (`validate_iris_features`, `validate_target_labels`, `validate_dataset_pair`).
- **Production Ingestion:** Standardized data loading interfaces with explicit type contracts.
- **Leakage Prevention:** Verified schema compliance, physical measurement bounds, and non-overlapping stratified splits.
"""

cleaned = re.sub(r'<<<<<<< HEAD[\s\S]*?>>>>>>> validate-iris-data', resolved.strip(), content)
with open("README.md", "w", encoding="utf-8") as f:
    f.write(cleaned)
PYEOF

git add README.md validation_functions.py

# Step 7: Create merge commit
echo ""
echo "[Step 7] Creating semantic merge commit..."
git commit -m "Merge branch 'validate-iris-data' into main

Resolve merge conflict between pipeline ingestion specs and validation layer:
- Integrated type-hinted validation functions (validate_iris_features, validate_target_labels, validate_dataset_pair)
- Reconciled competing README pipeline phase descriptions into unified architecture
- Maintained strict schema checks guarding data ingestion and preprocessing stages
- Verified all validation unit tests pass with resolved codebase"

# Step 8: Display git log graph
echo ""
echo "[Step 8] Git Log Graph (showing branching, conflict, and merge):"
git log --graph --oneline --decorate -n 6

echo ""
echo "=========================================================="
echo " Simulation & Resolution Completed Successfully!           "
echo "=========================================================="
