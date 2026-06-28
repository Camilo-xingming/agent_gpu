#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || pwd)"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/phase3-test.XXXXXX")"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

BIN_DIR="$TMP_ROOT/bin"
FIXTURE_DIR="$TMP_ROOT/fixtures"
mkdir -p "$BIN_DIR" "$FIXTURE_DIR"

GH_LOG="$TMP_ROOT/gh.log"
OPENCLAW_LOG="$TMP_ROOT/openclaw.log"
: > "$GH_LOG"
: > "$OPENCLAW_LOG"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_jq() {
  local file="$1"
  local expr="$2"
  local desc="$3"
  if ! jq -e "$expr" "$file" >/dev/null 2>&1; then
    fail "$desc (file=$file expr=$expr)"
  fi
}

assert_file() {
  local file="$1"
  [[ -s "$file" ]] || fail "missing file: $file"
}

cat > "$FIXTURE_DIR/milestones-open-sprint60.json" << 'JSON'
[
  {
    "title": "Sprint 60",
    "number": 60,
    "state": "open",
    "open_issues": 1,
    "created_at": "2026-03-07T00:00:00Z"
  }
]
JSON

cat > "$FIXTURE_DIR/milestones-all-sprint60.json" << 'JSON'
[
  {
    "title": "Sprint 59",
    "number": 59,
    "state": "closed",
    "closed_at": "2026-03-07T00:00:00Z"
  },
  {
    "title": "Sprint 60",
    "number": 60,
    "state": "open",
    "created_at": "2026-03-08T00:00:00Z"
  }
]
JSON

cat > "$FIXTURE_DIR/issues-open-watchdog.json" << 'JSON'
[
  {
    "number": 42,
    "title": "watchdog sample",
    "assignees": [{"login": "alice"}],
    "createdAt": "2026-03-01T00:00:00Z",
    "url": "https://example.invalid/issues/42"
  }
]
JSON

cat > "$FIXTURE_DIR/issues-open-process.json" << 'JSON'
[
  {
    "number": 42,
    "title": "process recurrence item",
    "url": "https://example.invalid/issues/42",
    "labels": [{"name": "process"}],
    "assignees": [],
    "milestone": null,
    "state": "OPEN"
  }
]
JSON

cat > "$FIXTURE_DIR/issues-all.json" << 'JSON'
[
  {
    "number": 42,
    "state": "OPEN",
    "title": "process recurrence item",
    "url": "https://example.invalid/issues/42"
  }
]
JSON

cat > "$FIXTURE_DIR/matching-refs.json" << 'JSON'
[
  {"ref": "refs/heads/issue-42/codex"}
]
JSON

cat > "$FIXTURE_DIR/pr-list-empty.json" << 'JSON'
[]
JSON

cat > "$FIXTURE_DIR/run-list-empty.json" << 'JSON'
[]
JSON

cat > "$FIXTURE_DIR/memory-stale.md" << 'EOF_MD'
# MEMORY
## Current State (Sprint 50)
EOF_MD

cat > "$FIXTURE_DIR/retro-recurrence.md" << 'EOF_MD'
# Sprint Retrospective Sprint 58
## Action Items
- [ ] Follow up #42 in planning

# Sprint Retrospective Sprint 59
## Action Items
- [ ] Follow up #42 in planning

# Sprint Retrospective Sprint 60
## Action Items
- [ ] Follow up #42 in planning
EOF_MD

cat > "$FIXTURE_DIR/gh" << 'EOF_GH'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >> "${GH_LOG_FILE:-/dev/null}"

cmd="${1:-}"
sub="${2:-}"

if [[ "$cmd" == "api" ]]; then
  route="${2:-}"
  case "$route" in
    repos/*/milestones\?state=all*)
      cat "$GH_STUB_MILESTONES_ALL_JSON"
      exit 0
      ;;
    repos/*/milestones\?state=open*)
      cat "$GH_STUB_MILESTONES_OPEN_JSON"
      exit 0
      ;;
    repos/*/git/matching-refs/heads/issue-*/)
      cat "$GH_STUB_MATCHING_REFS_JSON"
      exit 0
      ;;
    repos/*/issues\?state=all\&milestone=*)
      echo '[]'
      exit 0
      ;;
    repos/*/milestones/*)
      jq -cn '{state:"closed"}'
      exit 0
      ;;
    *)
      echo "unsupported gh api route: $route" >&2
      exit 2
      ;;
  esac
fi

if [[ "$cmd" == "issue" && "$sub" == "list" ]]; then
  shift 2
  state=''
  label=''
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --state)
        state="${2:-}"
        shift 2
        ;;
      --label)
        label="${2:-}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  if [[ "$state" == "all" ]]; then
    cat "$GH_STUB_ISSUES_ALL_JSON"
    exit 0
  fi

  if [[ "$label" == "process" && -n "${GH_STUB_ISSUES_PROCESS_JSON:-}" ]]; then
    cat "$GH_STUB_ISSUES_PROCESS_JSON"
  else
    cat "$GH_STUB_ISSUES_OPEN_JSON"
  fi
  exit 0
fi

if [[ "$cmd" == "issue" && "$sub" == "view" ]]; then
  issue_num="${3:-0}"
  jq -cn --arg n "$issue_num" '{number: ($n|tonumber), state:"OPEN", title:"fallback", url:("https://example.invalid/issues/"+$n), comments: []}'
  exit 0
fi

if [[ "$cmd" == "issue" && "$sub" == "create" ]]; then
  printf '%s\n' "${GH_STUB_ISSUE_CREATE_URL:-https://example.invalid/issues/999}"
  exit 0
fi

if [[ "$cmd" == "pr" && "$sub" == "list" ]]; then
  cat "$GH_STUB_PR_LIST_JSON"
  exit 0
fi

if [[ "$cmd" == "run" && "$sub" == "list" ]]; then
  cat "$GH_STUB_RUN_LIST_JSON"
  exit 0
fi

echo "unsupported gh command: $*" >&2
exit 2
EOF_GH
chmod +x "$FIXTURE_DIR/gh"

cat > "$FIXTURE_DIR/openclaw" << 'EOF_OC'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${OPENCLAW_LOG_FILE:-/dev/null}"
exit 0
EOF_OC
chmod +x "$FIXTURE_DIR/openclaw"

run_watchdog_memory_stale_test() {
  local out_file="$TMP_ROOT/watchdog-memory.json"
  local rc=0

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all-sprint60.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open-sprint60.json" \
  GH_STUB_MATCHING_REFS_JSON="$FIXTURE_DIR/matching-refs.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open-watchdog.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list-empty.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list-empty.json" \
  GH_BIN="$FIXTURE_DIR/gh" \
  OPENCLAW_BIN="$FIXTURE_DIR/openclaw" \
  MEMORY_FILE="$FIXTURE_DIR/memory-stale.md" \
  DISCORD_NOTIFY=false \
  "$SCRIPT_DIR/sprint-watchdog.sh" --repo ssql2014/RalphGPU --json --no-discord-notify > "$out_file" || rc=$?

  [[ "$rc" -eq 1 ]] || fail "watchdog should return WARN exit 1 on stale memory, got $rc"
  assert_jq "$out_file" '.status == "WARN"' "watchdog status should be WARN"
  assert_jq "$out_file" '.memory_staleness.stale == true' "memory staleness should be true"
  assert_jq "$out_file" '.memory_staleness.active_sprint == 60' "active sprint parsed"
  assert_jq "$out_file" '.memory_staleness.memory_sprint == 50' "memory sprint parsed"
}

run_planning_recurrence_test() {
  local shared_dir="$TMP_ROOT/planning-shared"
  mkdir -p "$shared_dir"

  local today
  today="$(date '+%Y-%m-%d')"
  jq -cn --arg d "$today" --rawfile retro "$FIXTURE_DIR/retro-recurrence.md" '{date:$d, retro_md:$retro, action_items:{github:[], discord:[]}}' > "$shared_dir/retro-data.json"

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all-sprint60.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open-sprint60.json" \
  GH_STUB_MATCHING_REFS_JSON="$FIXTURE_DIR/matching-refs.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open-watchdog.json" \
  GH_STUB_ISSUES_PROCESS_JSON="$FIXTURE_DIR/issues-open-process.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list-empty.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list-empty.json" \
  GH_BIN="$FIXTURE_DIR/gh" \
  OPENCLAW_BIN="$FIXTURE_DIR/openclaw" \
  OPENCLAW_LOG_FILE="$OPENCLAW_LOG" \
  SHARED_DIR="$shared_dir" \
  RETRO_DATA_FILE="$shared_dir/retro-data.json" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro-recurrence.md" \
  "$SCRIPT_DIR/sprint-planning-gather.sh" >/dev/null

  local out_file="$shared_dir/sprint-planning-data.json"
  assert_file "$out_file"
  assert_jq "$out_file" '.retro_recurrence.checked == true' "recurrence check executed"
  assert_jq "$out_file" '(.retro_recurrence.matches | length) == 1' "one repeated process issue"
  assert_jq "$out_file" '(.retro_recurrence.escalations | length) == 1' "one escalation issue"
  grep -q 'Repeated retro process items' "$OPENCLAW_LOG" || fail "escalation discord message not sent"
}

run_dev_system_review_test() {
  local shared_dir="$TMP_ROOT/dev-review-shared"
  mkdir -p "$shared_dir/status"
  local out_file="$shared_dir/dev-system-review.json"

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all-sprint60.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open-sprint60.json" \
  GH_STUB_MATCHING_REFS_JSON="$FIXTURE_DIR/matching-refs.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open-watchdog.json" \
  GH_STUB_ISSUES_PROCESS_JSON="$FIXTURE_DIR/issues-open-process.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list-empty.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list-empty.json" \
  GH_STUB_ISSUE_CREATE_URL="https://example.invalid/issues/777" \
  GH_BIN="$FIXTURE_DIR/gh" \
  OPENCLAW_BIN="$FIXTURE_DIR/openclaw" \
  OPENCLAW_LOG_FILE="$OPENCLAW_LOG" \
  SHARED_DIR="$shared_dir" \
  MEMORY_FILE="$FIXTURE_DIR/memory-stale.md" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro-recurrence.md" \
  "$SCRIPT_DIR/dev-system-review.sh" >/dev/null

  assert_file "$out_file"
  assert_jq "$out_file" '.trigger.triggered == true' "dev review should trigger at sprint 60"
  assert_jq "$out_file" '.memory_staleness.stale == true' "dev review should detect stale memory"
  assert_jq "$out_file" '(.findings | length) >= 1' "dev review should emit findings"
  assert_jq "$out_file" '.created_issue.url == "https://example.invalid/issues/777"' "dev review should create process issue"

  [[ "$(cat "$shared_dir/status/dev-system-review.last-sprint")" == "60" ]] || fail "state file should pin sprint 60"

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all-sprint60.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open-sprint60.json" \
  GH_STUB_MATCHING_REFS_JSON="$FIXTURE_DIR/matching-refs.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open-watchdog.json" \
  GH_STUB_ISSUES_PROCESS_JSON="$FIXTURE_DIR/issues-open-process.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list-empty.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list-empty.json" \
  GH_STUB_ISSUE_CREATE_URL="https://example.invalid/issues/778" \
  GH_BIN="$FIXTURE_DIR/gh" \
  OPENCLAW_BIN="$FIXTURE_DIR/openclaw" \
  OPENCLAW_LOG_FILE="$OPENCLAW_LOG" \
  SHARED_DIR="$shared_dir" \
  MEMORY_FILE="$FIXTURE_DIR/memory-stale.md" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro-recurrence.md" \
  "$SCRIPT_DIR/dev-system-review.sh" >/dev/null

  assert_jq "$out_file" '.trigger.triggered == false' "second run should skip same sprint"
}

run_watchdog_memory_stale_test
run_planning_recurrence_test
run_dev_system_review_test

echo "PASS: phase3 self-bootstrap tests completed"
