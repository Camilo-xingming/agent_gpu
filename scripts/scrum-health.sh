#!/usr/bin/env bash
# Compute scrum automation health coverage from gather outputs.

set -uo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

SHARED_DIR="${SHARED_DIR:-$HOME/.openclaw/shared-memory/ralphgpu}"
RETRO_FILE="${RETRO_FILE:-$SHARED_DIR/retro-data.json}"
PLANNING_FILE="${PLANNING_FILE:-$SHARED_DIR/sprint-planning-data.json}"
RETRO_STATUS_FILE="${RETRO_STATUS_FILE:-$SHARED_DIR/status/retro-gather.status.json}"
PLANNING_STATUS_FILE="${PLANNING_STATUS_FILE:-$SHARED_DIR/status/sprint-planning-gather.status.json}"
STRICT='false'

for arg in "$@"; do
  case "$arg" in
    --strict)
      STRICT='true'
      ;;
  esac
done

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found" >&2
  exit 1
fi

results='[]'
checks_total=0
checks_pass=0

add_check() {
  local name="$1"
  local file="$2"
  local expr="$3"

  checks_total=$((checks_total + 1))

  local status='fail'
  local detail=''

  if [[ ! -s "$file" ]]; then
    detail="missing file: $file"
  elif jq -e "$expr" "$file" >/dev/null 2>&1; then
    status='pass'
    detail='ok'
    checks_pass=$((checks_pass + 1))
  else
    detail="failed jq check: $expr"
  fi

  results="$(jq -cn --argjson arr "$results" --arg name "$name" --arg file "$file" --arg status "$status" --arg detail "$detail" '$arr + [{name: $name, file: $file, status: $status, detail: $detail}]')"
}

# retro-data checks
add_check "retro_json_object" "$RETRO_FILE" 'type == "object"'
add_check "retro_status" "$RETRO_FILE" '.status | type == "string"'
add_check "retro_ci_runs" "$RETRO_FILE" '.ci_runs | type == "array"'
add_check "retro_action_items" "$RETRO_FILE" '.action_items.combined | type == "array"'
add_check "retro_scrum_health" "$RETRO_FILE" '.scrum_health.coverage_pct | numbers'
add_check "retro_errors" "$RETRO_FILE" '.errors | type == "array"'
add_check "retro_warnings" "$RETRO_FILE" '.warnings | type == "array"'

# planning-data checks
add_check "planning_json_object" "$PLANNING_FILE" 'type == "object"'
add_check "planning_status" "$PLANNING_FILE" '.status | type == "string"'
add_check "planning_backlog" "$PLANNING_FILE" '(.backlog.p0 | type == "array") and (.backlog.p1 | type == "array") and (.backlog.all_unassigned | type == "array")'
add_check "planning_ci_runs" "$PLANNING_FILE" '.ci_runs | type == "array"'
add_check "planning_open_prs" "$PLANNING_FILE" '.open_prs | type == "array"'
add_check "planning_action_items" "$PLANNING_FILE" '.retro_action_items.combined | type == "array"'
add_check "planning_scrum_health" "$PLANNING_FILE" '.scrum_health.coverage_pct | numbers'
add_check "planning_errors" "$PLANNING_FILE" '.errors | type == "array"'
add_check "planning_warnings" "$PLANNING_FILE" '.warnings | type == "array"'

# status file checks
add_check "retro_status_file" "$RETRO_STATUS_FILE" '.component == "retro-gather" and (.state | type == "string") and (.updated_at | type == "string")'
add_check "planning_status_file" "$PLANNING_STATUS_FILE" '.component == "sprint-planning-gather" and (.state | type == "string") and (.updated_at | type == "string")'

coverage_pct=0
if [[ "$checks_total" -gt 0 ]]; then
  coverage_pct=$((checks_pass * 100 / checks_total))
fi

output_json="$(jq -cn --argjson total "$checks_total" --argjson pass "$checks_pass" --argjson coverage "$coverage_pct" --argjson checks "$results" '{
  checks_total: $total,
  checks_pass: $pass,
  coverage_pct: $coverage,
  checks: $checks
}')"

printf '%s\n' "$output_json"
printf 'scrum-health=%s%% (%s/%s)\n' "$coverage_pct" "$checks_pass" "$checks_total" >&2

if [[ "$STRICT" == 'true' && "$coverage_pct" -lt 100 ]]; then
  exit 1
fi
