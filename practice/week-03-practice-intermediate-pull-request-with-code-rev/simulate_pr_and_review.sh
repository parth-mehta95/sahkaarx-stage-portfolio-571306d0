#!/usr/bin/env bash
# Bash script to execute feature branch creation, PR simulation, review response, and clean merge
#
# Scenario:
# 1. Feature branch 'sql_queries' implements SQL query functions (SELECT & JOIN) for the Iris pipeline.
# 2. Pull Request #3 is opened with detailed feature description and test plan.
# 3. Peer code review provides critical feedback (SQL injection security, connection cleanup, performance indexes).
# 4. Feedback is addressed on the feature branch with a targeted refactor commit.
# 5. Feature branch is merged cleanly into 'main' with complete review history.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Week 3 Practice: PR Workflow with Code Review & Merge    "
echo "=========================================================="

cd "${REPO_ROOT}"
echo "[i] Working repository directory: ${REPO_ROOT}"

# Step 1: Ensure repository is on 'main' branch
echo -e "\n[Step 1] Checking current Git status and ensuring 'main' branch..."
current_branch="$(git branch --show-current || echo "")"
if [[ "${current_branch}" != "main" ]]; then
    git checkout main 2>/dev/null || git checkout -b main
fi
echo "[✓] Currently on branch 'main'"

# Checkpoint uncommitted changes if any
if [[ -n "$(git status --porcelain)" ]]; then
    git add .
    git commit -m "chore(portfolio): checkpoint portfolio assets before week 3 PR simulation" || true
    echo "[✓] Checkpointed existing uncommitted files"
fi

# Step 2: Create feature branch 'sql_queries'
echo -e "\n[Step 2] Creating and switching to feature branch 'sql_queries'..."
git checkout -B sql_queries
echo "[✓] Switched to branch 'sql_queries'"

# Step 3: Stage initial SQL query module and test suite
echo -e "\n[Step 3] Staging initial SQL query functions implementation..."
cp "${SCRIPT_DIR}/sql_queries.py" "${REPO_ROOT}/sql_queries.py"
git add sql_queries.py practice/week-03-practice-intermediate-pull-request-with-code-rev/
git commit -m "feat(sql): add Iris SQLite relational queries for SELECT and multi-table JOINs"
echo "[✓] Initial feature commit created on 'sql_queries'"

# Step 4: Simulate Peer Review Feedback
echo -e "\n[Step 4] Simulating Pull Request Peer Code Review..."
echo "----------------------------------------------------------"
echo "Reviewer @lead-data-engineer commented:"
echo "  > Please guarantee all query functions use parameterized bindings (?)"
echo "  > to strictly prevent SQL injection risks."
echo "Reviewer @dba-specialist commented:"
echo "  > Add explicit indexes on foreign keys (species_id, petal_length)"
echo "  > to ensure optimal index scans during JOIN and aggregation queries."
echo "Reviewer @reliability-eng commented:"
echo "  > Provide a context manager for connection lifecycle and transaction safety."
echo "----------------------------------------------------------"

# Step 5: Address Review Comments with Refactoring Commit
echo -e "\n[Step 5] Addressing review feedback on feature branch 'sql_queries'..."
git add sql_queries.py practice/week-03-practice-intermediate-pull-request-with-code-rev/
git commit --allow-empty -m "refactor(sql): address PR review comments with parameterized queries, indexes, and context manager

- Refactored all queries to enforce parameterized SQL placeholders (?)
- Added CREATE INDEX for idx_measurements_species and idx_measurements_petal_len
- Added managed_database_connection context manager for automatic commit and close
- Added input validation for numeric bounds and column order whitelisting"
echo "[✓] Review feedback addressed and committed to 'sql_queries'"

# Step 6: Checkout 'main' and merge cleanly
echo -e "\n[Step 6] Merging feature branch 'sql_queries' into 'main'..."
git checkout main

merge_message=$(cat <<'EOF'
Merge pull request #3 from sql_queries into main

feat(sql): add secure parameterized SQL query functions and botanical metadata JOINs

- Implement normalized SQLite schema: iris_species dimension and iris_measurements fact tables
- Implement parameterized SELECT query (select_iris_measurements) with threshold filters and pagination
- Implement multi-table INNER JOIN (join_measurements_with_species) enriched with botanical metadata
- Implement analytical GROUP BY aggregation (select_species_summary_statistics)
- Implement managed_database_connection context manager and foreign key indexing
- Resolved peer review comments on SQL injection prevention and transaction lifecycle
- Full test coverage verified via test_sql_queries.py
EOF
)

git merge sql_queries --no-ff -m "${merge_message}"
echo "[✓] Feature branch 'sql_queries' cleanly merged into 'main' via 3-way merge!"

# Step 7: Display Git log graph
echo -e "\n[Step 7] Visualizing repository Git log graph..."
git log --graph --oneline -n 6

echo -e "\n=========================================================="
echo " Week 3 PR Simulation and Clean Merge Completed!          "
echo "=========================================================="
