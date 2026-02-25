#!/usr/bin/env bash
# Targeted checks for cron optimization scripts.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || pwd)"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cron-opt-test.XXXXXX")"
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

cat > "$FIXTURE_DIR/milestones-all.json" << 'JSON'
[
  {
    "title": "Sprint 2026-02-20",
    "number": 7,
    "state": "closed",
    "closed_at": "2026-02-20T18:00:00Z",
    "due_on": "2026-02-20",
    "created_at": "2026-02-19T12:00:00Z",
    "open_issues": 0
  },
  {
    "title": "Sprint 2026-02-25",
    "number": 8,
    "state": "open",
    "due_on": "2026-02-26",
    "created_at": "2026-02-25T00:00:00Z",
    "open_issues": 1
  }
]
JSON

cat > "$FIXTURE_DIR/milestones-open.json" << 'JSON'
[
  {
    "title": "Sprint 2026-02-25",
    "number": 8,
    "state": "open",
    "due_on": "2026-02-26",
    "created_at": "2026-02-25T00:00:00Z",
    "open_issues": 1
  }
]
JSON

cat > "$FIXTURE_DIR/milestone-issues.json" << 'JSON'
[
  {
    "number": 42,
    "state": "open",
    "title": "Follow up item",
    "assignees": [],
    "labels": []
  },
  {
    "number": 43,
    "state": "closed",
    "title": "Completed item",
    "assignees": [],
    "labels": []
  }
]
JSON

cat > "$FIXTURE_DIR/run-list.json" << 'JSON'
[
  {
    "name": "CI",
    "status": "completed",
    "conclusion": "success",
    "headBranch": "master",
    "createdAt": "2026-02-25T00:00:00Z"
  },
  {
    "name": "Nightly",
    "status": "completed",
    "conclusion": "failure",
    "headBranch": "master",
    "createdAt": "2026-02-25T01:00:00Z"
  }
]
JSON

cat > "$FIXTURE_DIR/issues-open.json" << 'JSON'
[
  {
    "number": 50,
    "title": "P0 sample",
    "labels": [{"name": "P0-high"}],
    "assignees": [],
    "milestone": null,
    "state": "OPEN"
  },
  {
    "number": 51,
    "title": "P1 sample",
    "labels": [{"name": "P1-medium"}],
    "assignees": [{"login": "alice"}],
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
    "title": "Follow up item",
    "url": "https://example.invalid/issues/42"
  },
  {
    "number": 43,
    "state": "CLOSED",
    "title": "Completed item",
    "url": "https://example.invalid/issues/43"
  },
  {
    "number": 50,
    "state": "OPEN",
    "title": "P0 sample",
    "url": "https://example.invalid/issues/50"
  }
]
JSON

cat > "$FIXTURE_DIR/pr-list.json" << 'JSON'
[
  {
    "number": 300,
    "title": "Open PR",
    "headRefName": "feature/cron-optimization",
    "author": {"login": "bob"},
    "createdAt": "2026-02-25T00:30:00Z",
    "url": "https://example.invalid/pr/300"
  }
]
JSON

cat > "$FIXTURE_DIR/retro.md" << 'EOF_MD'
# Sprint Retrospective

## Action Items
- [ ] Follow up #42 in planning
EOF_MD

cat > "$FIXTURE_DIR/discord-dev.txt" << 'EOF_DC'
Action: sync #42 status in dev channel
EOF_DC

cat > "$FIXTURE_DIR/discord-main.txt" << 'EOF_DC'
TODO: review sprint health summary
EOF_DC

cat > "$BIN_DIR/gh" << 'EOF_GH'
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
    repos/*/issues\?state=all\&milestone=*)
      cat "$GH_STUB_MILESTONE_ISSUES_JSON"
      exit 0
      ;;
    *)
      echo "unsupported gh api route: $route" >&2
      exit 2
      ;;
  esac
fi

if [[ "$cmd" == "run" && "$sub" == "list" ]]; then
  cat "$GH_STUB_RUN_LIST_JSON"
  exit 0
fi

if [[ "$cmd" == "issue" && "$sub" == "list" ]]; then
  shift 2
  state=''
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --state)
        state="${2:-}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  if [[ "$state" == "open" ]]; then
    cat "$GH_STUB_ISSUES_OPEN_JSON"
  else
    cat "$GH_STUB_ISSUES_ALL_JSON"
  fi
  exit 0
fi

if [[ "$cmd" == "issue" && "$sub" == "view" ]]; then
  issue_num="${3:-0}"
  jq -cn --arg n "$issue_num" '{number: ($n | tonumber), state: "OPEN", title: "fallback", url: ("https://example.invalid/issues/" + $n)}'
  exit 0
fi

if [[ "$cmd" == "pr" && "$sub" == "list" ]]; then
  cat "$GH_STUB_PR_LIST_JSON"
  exit 0
fi

echo "unsupported gh command: $*" >&2
exit 2
EOF_GH
chmod +x "$BIN_DIR/gh"

cat > "$BIN_DIR/openclaw" << 'EOF_OC'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >> "${OPENCLAW_LOG_FILE:-/dev/null}"

cmd="${1:-}"
sub="${2:-}"

if [[ "$cmd" == "message" && "$sub" == "read" ]]; then
  if printf '%s' "$*" | grep -q '1475083010968649778'; then
    cat "$OPENCLAW_STUB_DEV_TEXT"
  else
    cat "$OPENCLAW_STUB_MAIN_TEXT"
  fi
  exit 0
fi

if [[ "$cmd" == "cron" && "$sub" == "list" ]]; then
  printf 'retro-gather: ok\n'
  exit 0
fi

if [[ "$cmd" == "message" && ( "$sub" == "send" || "$sub" == "create" ) ]]; then
  exit 0
fi

echo "unsupported openclaw command: $*" >&2
exit 2
EOF_OC
chmod +x "$BIN_DIR/openclaw"

run_warning_vs_critical_test() {
  local warn_dir="$TMP_ROOT/warn-shared"
  mkdir -p "$warn_dir"

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open.json" \
  GH_STUB_MILESTONE_ISSUES_JSON="$FIXTURE_DIR/milestone-issues.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list.json" \
  SHARED_DIR="$warn_dir" \
  GH_BIN="$BIN_DIR/gh" \
  OPENCLAW_BIN="/nonexistent/openclaw" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro.md" \
  "$SCRIPT_DIR/retro-gather.sh" >/dev/null

  local warn_retro="$warn_dir/retro-data.json"
  assert_file "$warn_retro"
  assert_jq "$warn_retro" '.status == "ok"' "warning run should stay ok"
  assert_jq "$warn_retro" '(.warnings | length) > 0' "warning run should collect warnings"
  assert_jq "$warn_retro" '(.errors | length) == 0' "warning run should not collect critical errors"

  local critical_dir="$TMP_ROOT/critical-shared"
  mkdir -p "$critical_dir"

  SHARED_DIR="$critical_dir" \
  GH_BIN="/nonexistent/gh" \
  OPENCLAW_BIN="/nonexistent/openclaw" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro.md" \
  "$SCRIPT_DIR/retro-gather.sh" >/dev/null || true

  local critical_retro="$critical_dir/retro-data.json"
  assert_file "$critical_retro"
  assert_jq "$critical_retro" '.status == "error"' "critical run should report error"
  assert_jq "$critical_retro" '(.errors | length) > 0' "critical run should collect errors"
}

run_zero_token_and_coverage_test() {
  local shared_dir="$TMP_ROOT/shared"
  mkdir -p "$shared_dir"
  : > "$GH_LOG"
  : > "$OPENCLAW_LOG"

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open.json" \
  GH_STUB_MILESTONE_ISSUES_JSON="$FIXTURE_DIR/milestone-issues.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list.json" \
  OPENCLAW_LOG_FILE="$OPENCLAW_LOG" \
  OPENCLAW_STUB_DEV_TEXT="$FIXTURE_DIR/discord-dev.txt" \
  OPENCLAW_STUB_MAIN_TEXT="$FIXTURE_DIR/discord-main.txt" \
  SHARED_DIR="$shared_dir" \
  GH_BIN="$BIN_DIR/gh" \
  OPENCLAW_BIN="$BIN_DIR/openclaw" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro.md" \
  "$SCRIPT_DIR/retro-gather.sh" >/dev/null

  GH_LOG_FILE="$GH_LOG" \
  GH_STUB_MILESTONES_ALL_JSON="$FIXTURE_DIR/milestones-all.json" \
  GH_STUB_MILESTONES_OPEN_JSON="$FIXTURE_DIR/milestones-open.json" \
  GH_STUB_MILESTONE_ISSUES_JSON="$FIXTURE_DIR/milestone-issues.json" \
  GH_STUB_RUN_LIST_JSON="$FIXTURE_DIR/run-list.json" \
  GH_STUB_ISSUES_OPEN_JSON="$FIXTURE_DIR/issues-open.json" \
  GH_STUB_ISSUES_ALL_JSON="$FIXTURE_DIR/issues-all.json" \
  GH_STUB_PR_LIST_JSON="$FIXTURE_DIR/pr-list.json" \
  OPENCLAW_LOG_FILE="$OPENCLAW_LOG" \
  OPENCLAW_STUB_DEV_TEXT="$FIXTURE_DIR/discord-dev.txt" \
  OPENCLAW_STUB_MAIN_TEXT="$FIXTURE_DIR/discord-main.txt" \
  SHARED_DIR="$shared_dir" \
  GH_BIN="$BIN_DIR/gh" \
  OPENCLAW_BIN="$BIN_DIR/openclaw" \
  RETRO_MD_PATH="$FIXTURE_DIR/retro.md" \
  "$SCRIPT_DIR/sprint-planning-gather.sh" >/dev/null

  local health_json
  health_json="$(SHARED_DIR="$shared_dir" "$SCRIPT_DIR/scrum-health.sh")"

  local coverage
  coverage="$(printf '%s\n' "$health_json" | jq -r '.coverage_pct')"
  [[ "$coverage" -ge 95 ]] || fail "scrum-health coverage below threshold: $coverage"

  local retro_file="$shared_dir/retro-data.json"
  local planning_file="$shared_dir/sprint-planning-data.json"
  local retro_status="$shared_dir/status/retro-gather.status.json"
  local planning_status="$shared_dir/status/sprint-planning-gather.status.json"

  assert_file "$retro_file"
  assert_file "$planning_file"
  assert_file "$retro_status"
  assert_file "$planning_status"

  assert_jq "$retro_file" '.scrum_health.coverage_pct >= 95' "retro scrum_health coverage"
  assert_jq "$planning_file" '.scrum_health.coverage_pct >= 95' "planning scrum_health coverage"
  assert_jq "$retro_status" '.component == "retro-gather" and (.state | type == "string")' "retro status file"
  assert_jq "$planning_status" '.component == "sprint-planning-gather" and (.state | type == "string")' "planning status file"

  grep -q '^message read ' "$OPENCLAW_LOG" || fail "missing openclaw message read calls"
  grep -q '^cron list' "$OPENCLAW_LOG" || fail "missing openclaw cron list call"

  if grep -Eq '(^| )agent( |$)|(^| )chat( |$)|(^| )ask( |$)' "$OPENCLAW_LOG"; then
    fail "non-zero-token openclaw command detected"
  fi

  if grep -q '^issue view ' "$GH_LOG"; then
    fail "issue view fallback should not be needed with bulk issue list"
  fi
}

run_atomic_write_race_test() {
  local atomic_dir="$TMP_ROOT/atomic"
  mkdir -p "$atomic_dir"

  bash -s "$SCRIPT_DIR/cron-common.sh" "$atomic_dir" << 'EOF_BASH'
set -euo pipefail

cron_common="$1"
target_dir="$2"
source "$cron_common"

target_file="$target_dir/race-output.json"
for i in $(seq 1 64); do
  atomic_write_file "$target_file" "{\"seq\":$i}" &
done
wait

jq -e '.seq | numbers' "$target_file" >/dev/null

if find "$target_dir" -maxdepth 1 -type f -name '.tmp.race-output.json.*' | grep -q .; then
  echo "leftover temp files detected" >&2
  exit 1
fi
EOF_BASH
}

run_warning_vs_critical_test
run_zero_token_and_coverage_test
run_atomic_write_race_test

echo "PASS: cron optimization tests completed"
