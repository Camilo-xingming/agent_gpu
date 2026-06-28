#!/bin/bash
set -e

# Default to the main repository unless overridden
REPO=${1:-"ssql2014/RalphGPU"}

echo "Fetching milestones for $REPO..."
# Find max sprint number by parsing titles like 'Sprint 60'
MAX_SPRINT=$(gh api --paginate "repos/$REPO/milestones?state=all&per_page=100" -q '.[].title' | grep -Eo '^Sprint [0-9]+$' | awk '{print $2}' | sort -nr | head -n1 || true)

if [ -z "$MAX_SPRINT" ]; then
    echo "No existing Sprint milestones found."
    MAX_SPRINT=0
fi

NEXT_SPRINT=$((MAX_SPRINT + 1))
NEXT_TITLE="Sprint $NEXT_SPRINT"

echo "Current max sprint: $MAX_SPRINT"
echo "Target next sprint: $NEXT_SPRINT ($NEXT_TITLE)"

# Check if target milestone already exists
EXISTING=$(gh api --paginate "repos/$REPO/milestones?state=all&per_page=100" -q '.[] | select(.title=="'"$NEXT_TITLE"'") | .title' || true)

if [ "$EXISTING" == "$NEXT_TITLE" ]; then
    echo "Milestone '$NEXT_TITLE' already exists. Idempotent check passed."
else
    echo "Creating milestone '$NEXT_TITLE'..."
    gh api "repos/$REPO/milestones" -X POST -f title="$NEXT_TITLE" -f state="open" --silent
    echo "Milestone '$NEXT_TITLE' created successfully."
fi
