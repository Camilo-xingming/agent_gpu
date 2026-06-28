#!/usr/bin/env bash
# Periodic dev-system self-review: trigger every N sprints and escalate actionable findings.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || pwd)"
# shellcheck source=cron-common.sh
source "$SCRIPT_DIR/cron-common.sh"

REPO_ROOT="$(resolve_repo_root)"
GITHUB_REPO="${GITHUB_REPO:-ssql2014/RalphGPU}"

GH_BIN_INPUT="${GH_BIN:-}"
GH_BIN="$(resolve_command_path "$GH_BIN_INPUT" gh /opt/homebrew/bin/gh /usr/local/bin/gh 2>/dev/null || true)"
if [[ -z "$GH_BIN" ]]; then
  GH_BIN="${GH_BIN_INPUT:-gh}"
fi

OPENCLAW_BIN_INPUT="${OPENCLAW_BIN:-}"
OPENCLAW_BIN="$(resolve_command_path "$OPENCLAW_BIN_INPUT" openclaw /opt/homebrew/bin/openclaw /usr/local/bin/openclaw 2>/dev/null || true)"
if [[ -z "$OPENCLAW_BIN" ]]; then
  OPENCLAW_BIN="${OPENCLAW_BIN_INPUT:-openclaw}"
fi

configure_gh_proxy_env

SHARED_DIR="${SHARED_DIR:-$HOME/.openclaw/shared-memory/ralphgpu}"
OUTPUT_FILE="${OUTPUT_FILE:-$SHARED_DIR/dev-system-review.json}"
STATUS_FILE="${STATUS_FILE:-$SHARED_DIR/status/dev-system-review.status.json}"
STATE_FILE="${STATE_FILE:-$SHARED_DIR/status/dev-system-review.last-sprint}"
RETRO_MD_PATH="${RETRO_MD_PATH:-$REPO_ROOT/docs/RETRO.md}"
MEMORY_FILE="${MEMORY_FILE:-$REPO_ROOT/MEMORY.md}"

REVIEW_INTERVAL_SPRINTS="${REVIEW_INTERVAL_SPRINTS:-5}"
MEMORY_STALE_GAP="${MEMORY_STALE_GAP:-3}"
RETRO_REPEAT_CONSECUTIVE_MIN="${RETRO_REPEAT_CONSECUTIVE_MIN:-2}"
RETRO_ESCALATE_CONSECUTIVE_MIN="${RETRO_ESCALATE_CONSECUTIVE_MIN:-3}"

DISCORD_ACCOUNT="${DISCORD_ACCOUNT:-lily}"
DISCORD_DEV_TARGET="${DISCORD_DEV_TARGET:-channel:1475083010968649778}"
JERRY_MENTION="${JERRY_MENTION:-@ssql2014}"

NOW_TS="$(date '+%Y-%m-%dT%H:%M:%S')"
TODAY="$(date '+%Y-%m-%d')"

mkdir -p "$SHARED_DIR" "$(dirname -- "$STATUS_FILE")" "$(dirname -- "$STATE_FILE")"

errors_json='[]'
warnings_json='[]'
health_checks_json='[]'
status='ok'

latest_sprint=''
triggered='false'
trigger_reason=''
last_reviewed_sprint=''
memory_state_sprint=''
memory_gap=''
memory_stale='false'
process_recurrence_json='{"checked":false,"threshold_consecutive":2,"escalate_after_consecutive":3,"matches":[],"escalations":[]}'
findings_json='[]'
created_issue_number=''
created_issue_url=''
discord_notify_result='not_triggered'

if ! has_command jq; then
  echo "jq not found" >&2
  exit 1
fi

for numeric_var in REVIEW_INTERVAL_SPRINTS MEMORY_STALE_GAP RETRO_REPEAT_CONSECUTIVE_MIN RETRO_ESCALATE_CONSECUTIVE_MIN; do
  value="${!numeric_var:-}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    append_warning "Invalid ${numeric_var}=${value}; using fallback"
    case "$numeric_var" in
      REVIEW_INTERVAL_SPRINTS) REVIEW_INTERVAL_SPRINTS=5 ;;
      MEMORY_STALE_GAP) MEMORY_STALE_GAP=3 ;;
      RETRO_REPEAT_CONSECUTIVE_MIN) RETRO_REPEAT_CONSECUTIVE_MIN=2 ;;
      RETRO_ESCALATE_CONSECUTIVE_MIN) RETRO_ESCALATE_CONSECUTIVE_MIN=3 ;;
    esac
  fi
done

if [[ "$RETRO_ESCALATE_CONSECUTIVE_MIN" -lt "$RETRO_REPEAT_CONSECUTIVE_MIN" ]]; then
  RETRO_ESCALATE_CONSECUTIVE_MIN="$RETRO_REPEAT_CONSECUTIVE_MIN"
fi

add_health_check "jq_available" "pass" "jq command available" true

lock_path=''
status_file_finalized='false'

cleanup() {
  local rc=$?
  stop_heartbeat
  if [[ "$status_file_finalized" != 'true' && "$rc" -ne 0 ]]; then
    write_status_file "$STATUS_FILE" "dev-system-review" "error" "script exited with code $rc" || true
  fi
  if [[ -n "$lock_path" ]]; then
    release_script_lock "$lock_path"
  fi
}
trap cleanup EXIT

if ! lock_path="$(acquire_script_lock "$SHARED_DIR" "dev-system-review")"; then
  log_event "dev-system-review" "WARN" "lock busy, skipping duplicate run"
  exit 0
fi

if write_status_file "$STATUS_FILE" "dev-system-review" "running" "evaluating dev system checks"; then
  add_health_check "status_file_running" "pass" "running state written" true
else
  append_warning "Failed to write running status file: $STATUS_FILE"
  add_health_check "status_file_running" "fail" "running state write failed" true
fi

start_heartbeat "dev-system-review" "checking memory staleness and retro recurrence"

if has_command "$GH_BIN"; then
  add_health_check "github_cli_available" "pass" "gh command available" true
else
  append_error "gh command not found: $GH_BIN"
  status='error'
  add_health_check "github_cli_available" "fail" "gh command missing" true
fi

all_milestones_json='[]'
if [[ "$status" == 'ok' ]]; then
  if capture_with_retry all_milestones_json 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=all&per_page=100"; then
    add_health_check "milestones_query" "pass" "milestones queried" true
  else
    append_error "Failed to query milestones"
    status='error'
    add_health_check "milestones_query" "fail" "milestone query failed" true
  fi
fi

if [[ "$status" == 'ok' ]]; then
  latest_sprint="$(printf '%s\n' "$all_milestones_json" | jq -r '
    [ .[]
      | (.title // "")
      | (try (capture("Sprint[[:space:]]+(?<n>[0-9]+)").n | tonumber) catch empty)
    ]
    | sort
    | last // empty
  ' 2>/dev/null || true)"

  if [[ -z "$latest_sprint" ]]; then
    append_warning "No numeric sprint milestone found; skipping dev review trigger"
    add_health_check "latest_sprint_detected" "fail" "no numeric sprint milestone found" true
  else
    add_health_check "latest_sprint_detected" "pass" "latest sprint=$latest_sprint" true
  fi
fi

if [[ -f "$STATE_FILE" ]]; then
  last_reviewed_sprint="$(cat "$STATE_FILE" 2>/dev/null || true)"
fi

if [[ -n "$latest_sprint" ]]; then
  mod_value=$((latest_sprint % REVIEW_INTERVAL_SPRINTS))
  if [[ "$mod_value" -ne 0 ]]; then
    triggered='false'
    trigger_reason="latest sprint ${latest_sprint} is not aligned to interval ${REVIEW_INTERVAL_SPRINTS}"
  elif [[ -n "$last_reviewed_sprint" && "$last_reviewed_sprint" == "$latest_sprint" ]]; then
    triggered='false'
    trigger_reason="sprint ${latest_sprint} already reviewed"
  else
    triggered='true'
    trigger_reason="interval trigger matched at sprint ${latest_sprint}"
  fi
else
  triggered='false'
  trigger_reason='latest sprint unavailable'
fi

if [[ -f "$MEMORY_FILE" && -n "$latest_sprint" ]]; then
  memory_state_sprint="$(grep -Eo 'Current State \(Sprint[[:space:]]+[0-9]+\)' "$MEMORY_FILE" | head -n 1 | grep -Eo '[0-9]+' || true)"
  if [[ -n "$memory_state_sprint" ]]; then
    memory_gap=$((latest_sprint - memory_state_sprint))
    if [[ "$memory_gap" -lt 0 ]]; then
      memory_gap=0
    fi
    if [[ "$memory_gap" -gt "$MEMORY_STALE_GAP" ]]; then
      memory_stale='true'
      findings_json="$(jq -cn --argjson arr "$findings_json" --arg type "memory_staleness" --arg severity "warn" --arg detail "MEMORY.md current sprint ${memory_state_sprint} lags latest sprint ${latest_sprint} by ${memory_gap}" '$arr + [{type: $type, severity: $severity, detail: $detail}]')"
    fi
  else
    append_warning "Current State (Sprint N) not found in MEMORY.md"
  fi
else
  append_warning "MEMORY.md missing or latest sprint unknown"
fi

retro_refs_json='[]'
open_process_issues_json='[]'
retro_refs_json="$(build_retro_refs_json_from_file "$RETRO_MD_PATH")"

if [[ "$status" == 'ok' ]]; then
  if capture_with_retry open_process_issues_json 3 2 "$GH_BIN" issue list --repo "$GITHUB_REPO" --state open --label process --limit 200 --json number,title,url; then
    process_recurrence_json="$(build_process_recurrence_json "$retro_refs_json" "$open_process_issues_json" "$RETRO_REPEAT_CONSECUTIVE_MIN" "$RETRO_ESCALATE_CONSECUTIVE_MIN")"

    recurrence_count="$(printf '%s\n' "$process_recurrence_json" | jq '.matches | length' 2>/dev/null || echo 0)"
    escalation_count="$(printf '%s\n' "$process_recurrence_json" | jq '.escalations | length' 2>/dev/null || echo 0)"

    if [[ "$recurrence_count" -gt 0 ]]; then
      findings_json="$(jq -cn --argjson arr "$findings_json" --arg type "retro_recurrence" --arg severity "warn" --arg detail "${recurrence_count} open process issue(s) repeated in >=${RETRO_REPEAT_CONSECUTIVE_MIN} consecutive retros" '$arr + [{type: $type, severity: $severity, detail: $detail}]')"
    fi

    if [[ "$escalation_count" -gt 0 ]]; then
      escalation_preview="$(printf '%s\n' "$process_recurrence_json" | jq -r '.escalations | map("#" + (.number|tostring) + "(" + (.max_consecutive|tostring) + "x)") | join(", ")' 2>/dev/null || true)"
      findings_json="$(jq -cn --argjson arr "$findings_json" --arg type "retro_escalation" --arg severity "escalate" --arg detail "recurring >2 sprints without progress: ${escalation_preview}" '$arr + [{type: $type, severity: $severity, detail: $detail}]')"
    fi
  else
    append_warning "Failed to query open process issues for recurrence check"
  fi
fi

findings_count="$(printf '%s\n' "$findings_json" | jq 'length' 2>/dev/null || echo 0)"

if [[ "$triggered" == 'true' ]]; then
  summary_msg="📋 Dev system review (Sprint ${latest_sprint}) complete. Findings: ${findings_count}."
  if [[ "$findings_count" -gt 0 ]]; then
    details_preview="$(printf '%s\n' "$findings_json" | jq -r 'map("- [" + .severity + "] " + .detail) | join("\\n")' 2>/dev/null || true)"
    summary_msg+=$'\n'"${details_preview}"
  else
    summary_msg+=" No action needed."
  fi
  send_discord_message_best_effort "$summary_msg"
  if has_command "$OPENCLAW_BIN"; then
    discord_notify_result='sent_or_attempted'
  else
    discord_notify_result='openclaw_missing'
  fi

  if [[ "$findings_count" -gt 0 && "$status" == 'ok' ]]; then
    issue_title="process: dev-system-review Sprint ${latest_sprint} findings"
    issue_body="## Summary\nAutomated dev-system review triggered at Sprint ${latest_sprint} (interval=${REVIEW_INTERVAL_SPRINTS}).\n\n## Findings\n$(printf '%s\n' "$findings_json" | jq -r 'to_entries | map("- [ ] " + .value.detail) | join("\\n")')\n\n## Source\n- Script: scripts/dev-system-review.sh\n- Date: ${TODAY}\n"
    issue_url_raw=''
    issue_url_raw="$($GH_BIN issue create --repo "$GITHUB_REPO" --title "$issue_title" --label process --body "$issue_body" 2>/dev/null || true)"
    if [[ -n "$issue_url_raw" ]]; then
      created_issue_url="$issue_url_raw"
      created_issue_number="$(printf '%s\n' "$issue_url_raw" | grep -Eo '[0-9]+$' || true)"
    else
      append_warning "Failed to create process issue from dev-system-review findings"
    fi
  fi

  printf '%s\n' "$latest_sprint" > "$STATE_FILE"
fi

scrum_health_json="$(build_health_summary)"

latest_sprint_json='null'
if [[ -n "$latest_sprint" ]]; then
  latest_sprint_json="$latest_sprint"
fi

last_reviewed_json='null'
if [[ -n "$last_reviewed_sprint" ]]; then
  last_reviewed_json="$last_reviewed_sprint"
fi

memory_state_json='null'
if [[ -n "$memory_state_sprint" ]]; then
  memory_state_json="$memory_state_sprint"
fi

memory_gap_json='null'
if [[ -n "$memory_gap" ]]; then
  memory_gap_json="$memory_gap"
fi

output_json="$(jq -cn \
  --arg status "$status" \
  --arg timestamp "$NOW_TS" \
  --arg date "$TODAY" \
  --argjson latest_sprint "$latest_sprint_json" \
  --arg triggered "$triggered" \
  --arg trigger_reason "$trigger_reason" \
  --argjson review_interval "$REVIEW_INTERVAL_SPRINTS" \
  --argjson last_reviewed_sprint "$last_reviewed_json" \
  --arg memory_file "$MEMORY_FILE" \
  --argjson memory_stale_gap "$MEMORY_STALE_GAP" \
  --argjson memory_state_sprint "$memory_state_json" \
  --argjson memory_gap "$memory_gap_json" \
  --arg memory_stale "$memory_stale" \
  --argjson process_recurrence "$process_recurrence_json" \
  --argjson findings "$findings_json" \
  --arg created_issue_number "$created_issue_number" \
  --arg created_issue_url "$created_issue_url" \
  --arg discord_notify_result "$discord_notify_result" \
  --argjson scrum_health "$scrum_health_json" \
  --argjson errors "$errors_json" \
  --argjson warnings "$warnings_json" \
  '{
    status: $status,
    timestamp: $timestamp,
    date: $date,
    latest_sprint: $latest_sprint,
    trigger: {
      interval_sprints: $review_interval,
      triggered: ($triggered == "true"),
      reason: $trigger_reason,
      last_reviewed_sprint: $last_reviewed_sprint
    },
    memory_staleness: {
      memory_file: $memory_file,
      threshold: $memory_stale_gap,
      memory_sprint: $memory_state_sprint,
      gap: $memory_gap,
      stale: ($memory_stale == "true")
    },
    retro_recurrence: $process_recurrence,
    findings: $findings,
    created_issue: {
      number: (if ($created_issue_number | length) == 0 then null else ($created_issue_number | tonumber) end),
      url: (if ($created_issue_url | length) == 0 then null else $created_issue_url end)
    },
    discord_notify_result: $discord_notify_result,
    scrum_health: $scrum_health,
    errors: $errors,
    warnings: $warnings
  }')"

if ! atomic_write_file "$OUTPUT_FILE" "$output_json"; then
  append_error "Failed to write output file: $OUTPUT_FILE"
  status='error'
fi

status_state='success'
status_detail='dev system review complete'
if [[ "$status" != 'ok' ]]; then
  status_state='error'
  status_detail='dev system review failed'
fi

if write_status_file "$STATUS_FILE" "dev-system-review" "$status_state" "$status_detail"; then
  status_file_finalized='true'
else
  append_warning "Failed to write final status file: $STATUS_FILE"
fi

log_event "dev-system-review" "INFO" "triggered=${triggered} sprint=${latest_sprint:-none} findings=${findings_count}"
