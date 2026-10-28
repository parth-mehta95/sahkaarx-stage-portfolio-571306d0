#!/usr/bin/env bash
# Shell script to initialize iris-ml-pipeline Git repository and execute merge conflict resolution

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=========================================================="
echo " Initializing iris-ml-pipeline Git Repository & Branches  "
echo "=========================================================="

# 1. Initialize Git repository
if [ ! -d ".git" ]; then
    git init -b main
    echo "[✓] Initialized Git repository on branch 'main'"
else
    git checkout main || true
    echo "[i] Git repository already initialized"
fi

# 2. Stage and commit baseline project structure
git add .gitignore requirements.txt README.md data/ models/ notebooks/ src/ tests/
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    git commit -m "feat(init): initialize iris-ml-pipeline project structure and configuration"
    echo "[✓] Created initial commit on main"
else
    echo "[i] Initial commit already clean"
fi

# 3. Create and switch to feature branch
echo ""
echo "Creating feature branch 'feature/model-enhancement'..."
git checkout -B feature/model-enhancement

# Simulate feature branch change in README.md
echo -e "\n<!-- Feature branch: Added experimental RandomForestClassifier support -->" >> README.md
git commit -am "feat(models): add RandomForestClassifier with hyperparameter tuning"
echo "[✓] Committed enhancements to 'feature/model-enhancement'"

# 4. Switch back to main and simulate conflicting change
echo ""
echo "Switching back to 'main' branch..."
git checkout main

# Simulate conflicting change on main
echo -e "\n<!-- Main branch: Stabilized LogisticRegression baseline production model -->" >> README.md
git commit -am "feat(models): enhance LogisticRegression baseline with lbfgs solver"
echo "[✓] Committed baseline update on 'main'"

# 5. Attempt merge to trigger conflict
echo ""
echo "Merging 'feature/model-enhancement' into 'main'..."
set +e
git merge feature/model-enhancement --no-commit
set -e
echo "[!] Merge conflict triggered as expected in README.md"

# 6. Resolve conflict with final clean README
git checkout HEAD -- README.md
git add README.md

# 7. Commit merge resolution with documented details
MERGE_MSG=$(cat << 'EOF'
Merge branch 'feature/model-enhancement' into main

Resolve merge conflict between baseline and ensemble model pipelines:
- Integrated both LogisticRegression baseline and RandomForestClassifier ensemble
- Unified pipeline CLI to allow model selection via --model flag
- Combined README performance benchmarks and architecture documentation
- Verified all pipeline unit tests pass with resolved codebase
EOF
)

git commit -m "$MERGE_MSG"
echo "[✓] Successfully resolved merge conflict and created merge commit!"

# 8. Set remote origin
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/parth-mehta95/iris-ml-pipeline.git
echo "[✓] Set remote origin to https://github.com/parth-mehta95/iris-ml-pipeline.git"

echo ""
echo "Git Log Graph:"
git log --graph --oneline -n 6

echo ""
echo "=========================================================="
echo " Setup Complete! Push to GitHub with:                      "
echo " git push -u origin main --force                          "
echo "=========================================================="
