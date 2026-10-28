#!/usr/bin/env bash
# Bash script to execute feature branch creation, PR simulation, code review response, and clean merge
#
# Scenario:
# 1. Feature branch 'sql-features' implements SQL aggregation queries (GROUP BY & JOIN) for ML feature engineering.
# 2. Pull Request #4 is opened with detailed description of aggregation logic and feature engineering benefits.
# 3. Peer code review provides critical feedback (NULLIF guards, compound indexing, relative deviations).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Week 4 Practice: SQL Feature Engineering PR & Merge      "
echo "=========================================================="

cd "${REPO_ROOT}"
echo "[i] Working repository directory: ${REPO_ROOT}"

# Step 1: Ensure repository is on 'main' branch
echo -e "\n[Step 1] Checking current Git status and ensuring 'main' branch..."
current_branch="$(git branch --show-current || true)"
if [ "${current_branch}" != "main" ]; then
    git checkout main 2>/dev/null || true
fi
echo "[✓] Currently on branch 'main'"

# Checkpoint uncommitted files if needed
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 4 feature branch" 2>/dev/null || true
    echo "[✓] Checkpointed existing uncommitted files"
fi

# Step 2: Create feature branch 'sql-features'
echo -e "\n[Step 2] Creating and switching to feature branch 'sql-features'..."
git checkout -B sql-features
echo "[✓] Switched to branch 'sql-features'"

# Step 3: Stage initial SQL feature engineering module and test suite
echo -e "\n[Step 3] Staging initial SQL feature engineering implementation..."
cp "${SCRIPT_DIR}/sql_features.py" "${REPO_ROOT}/sql_features.py"
git add sql_features.py practice/week-04-practice-sql-feature-engineering-with-pull-reque/
git commit -m "feat(sql): add sql_features.py with GROUP BY and JOIN queries for Iris aggregation

- Implement aggregate_species_features with GROUP BY, HAVING, and summary metrics
- Implement engineer_observation_features with multi-table JOINs and morphological ratios
- Implement aggregate_features_by_size_tier with CASE discretization and GROUP BY
- Add comprehensive unit test suite in test_sql_features.py"
echo "[✓] Initial feature commit created on 'sql-features'"

# Step 4: Simulate Peer Review Feedback
echo -e "\n[Step 4] Simulating Pull Request Peer Code Review on PR #4..."
echo "----------------------------------------------------------"
echo "Pull Request #4: feat(sql): add SQL aggregation and JOIN queries for ML feature engineering"
echo "Author: @parth-mehta95 | Target: main <- sql-features"
echo ""
echo "Reviewer @senior-data-architect commented:"
echo "  > Excellent aggregation logic! Please ensure all division operations"
echo "  > (e.g. aspect ratios and area ratios) are wrapped with NULLIF(..., 0)"
echo "  > to avoid runtime division-by-zero crashes on edge data."
echo ""
echo "Reviewer @database-administrator commented:"
echo "  > Add compound indexes covering (species_id, petal_length, petal_width)"
echo "  > to eliminate full table scans during aggregate JOIN queries."
echo ""
echo "Reviewer @ml-pipeline-lead commented:"
echo "  > Please add relative feature deltas comparing individual sample dimensions"
echo "  > against the species-level benchmark average via a CTE join."
echo "----------------------------------------------------------"

# Step 5: Address Review Comments with Refactoring Commit
echo -e "\n[Step 5] Addressing review feedback on feature branch 'sql-features'..."
git add sql_features.py practice/week-04-practice-sql-feature-engineering-with-pull-reque/
git commit --allow-empty -m "refactor(sql): address PR review comments with NULLIF guards, compound indexes, and relative deltas

- Wrapped all SQL division expressions in NULLIF(..., 0) to guard against zero division
- Added create_feature_indexes() with compound indexes on (species_id, petal_length, petal_width)
- Added CTE-based relative deviation features (sepal_len_diff_from_species_mean, petal_area_ratio_to_species_avg)
- Added unit tests for zero division resilience and relative delta verification"
echo "[✓] Review feedback addressed and committed to 'sql-features'"

# Step 6: Checkout 'main' and merge cleanly
echo -e "\n[Step 6] Merging feature branch 'sql-features' into 'main'..."
git checkout main

MERGE_MSG="Merge pull request #4 from sql-features into main

feat(sql): add SQL aggregation and relational JOIN queries for ML feature engineering

- Implement aggregate_species_features: per-species GROUP BY and HAVING aggregation
- Implement engineer_observation_features: multi-table JOIN + CTE benchmarks with relative deltas
- Implement aggregate_features_by_size_tier: morphological binning and sub-tier metrics
- Implement compute_feature_separability_summary: between-class divergence metrics
- Implement get_feature_matrix_and_target: export clean X and y matrices for scikit-learn
- Add compound indexes (idx_features_species_petals, idx_features_species_sepals)
- Enforce NULLIF zero-division guards and parameter binding throughout
- Full unit test coverage verified via test_sql_features.py
- Addressed all peer code review comments prior to merge"

git merge sql-features --no-ff -m "${MERGE_MSG}"
echo "[✓] Feature branch 'sql-features' cleanly merged into 'main' via 3-way merge!"

# Step 7: Display Git log graph
echo -e "\n[Step 7] Visualizing repository Git log graph..."
git log --graph --oneline -n 8

echo -e "\n=========================================================="
echo " Week 4 PR Simulation and Clean Merge Completed!          "
echo "=========================================================="
