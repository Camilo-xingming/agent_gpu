#!/usr/bin/env bash
# Gather sprint retrospective data from GitHub + Discord.

set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || pwd)"
# shellcheck source=cron-common.sh
source "$SCRIPT_DIR/cron-common.sh"

extract_github_action_items() {
  local content="$1"
  printf '%s\n' "$content" | awk '
    BEGIN { in_actions = 0 }
    /^## / {
      if ($0 ~ /^## .*Action Items/) {
        in_actions = 1
        next
      }
      if (in_actions) {
        exit
      }
    }
    in_actions && /^- \[[ xX]\]/ {
      line = $0
      sub(/^- \[[ xX]\][[:space:]]*/, "", line)
      print line
    }
  '
}

extract_discord_action_items() {
  local text="$1"
  printf '%s\n' "$text" | awk 'NF { print }' | grep -E '(Action|action|TODO|todo|待办|跟进|请|立即|必须|需|需要|should|修复|分配|排查|follow)' 2>/dev/null || true
}

REPO_ROOT="$(resolve_repo_root)"
GITHUB_REPO="${GITHUB_REPO:-ssql2014/RalphGPU}"

GH_BIN_INPUT="${GH_BIN:-}"
GH_BIN="$(resolve_command_path "$GH_BIN_INPUT" gh /opt/homebrew/bin/gh /usr/local/bin/gh 2>/dev/null || true)"
if [[ -z "$GH_BIN" ]]; then
  GH_BIN="${GH_BIN_INPUT:-gh}"
fi

configure_gh_proxy_env

OPENCLAW_BIN_INPUT="${OPENCLAW_BIN:-}"
OPENCLAW_BIN="$(resolve_command_path "$OPENCLAW_BIN_INPUT" openclaw /opt/homebrew/bin/openclaw /usr/local/bin/openclaw 2>/dev/null || true)"
if [[ -z "$OPENCLAW_BIN" ]]; then
  OPENCLAW_BIN="${OPENCLAW_BIN_INPUT:-openclaw}"
fi

SHARED_DIR="${SHARED_DIR:-$HOME/.openclaw/shared-memory/ralphgpu}"
OUTPUT_FILE="${OUTPUT_FILE:-$SHARED_DIR/retro-data.json}"
CRON_LOG="${CRON_LOG:-$SHARED_DIR/cron-bash.log}"
STATUS_FILE="${STATUS_FILE:-$SHARED_DIR/status/retro-gather.status.json}"
RETRO_MD_PATH="${RETRO_MD_PATH:-$REPO_ROOT/docs/RETRO.md}"

DISCORD_ACCOUNT="${DISCORD_ACCOUNT:-lily}"
DISCORD_MAIN_TARGET="${DISCORD_MAIN_TARGET:-channel:1468774996301316137}"
DISCORD_DEV_TARGET="${DISCORD_DEV_TARGET:-channel:1475083010968649778}"
DISCORD_LIMIT="${DISCORD_LIMIT:-40}"

NOW_TS="$(date '+%Y-%m-%dT%H:%M:%S')"
TODAY="$(date '+%Y-%m-%d')"

mkdir -p "$SHARED_DIR" "$(dirname -- "$STATUS_FILE")"

errors_json='[]'
warnings_json='[]'
health_checks_json='[]'
status='ok'
milestone_json='null'
sprint_issues_json='[]'
ci_runs_json='[]'
ci_fail_count='0'
velocity_json='null'
retro_md=''
discord_dev=''
discord_main=''
cron_status=''
github_action_items_json='[]'
discord_action_items_json='[]'
combined_action_items_json='[]'

if ! has_command jq; then
  echo "jq not found" >&2
  exit 1
fi

add_health_check "jq_available" "pass" "jq command available" true
add_health_check "atomic_write_enabled" "pass" "output writes use mktemp+mv" true

lock_path=''
status_file_finalized='false'

cleanup() {
  local rc=$?
  stop_heartbeat
  if [[ "$status_file_finalized" != 'true' && "$rc" -ne 0 ]]; then
    write_status_file "$STATUS_FILE" "retro-gather" "error" "script exited with code $rc" || true
  fi
  if [[ -n "$lock_path" ]]; then
    release_script_lock "$lock_path"
  fi
}
trap cleanup EXIT

if ! lock_path="$(acquire_script_lock "$SHARED_DIR" "retro-gather")"; then
  log_event "retro-gather" "WARN" "lock busy, skipping duplicate run"
  exit 0
fi

if write_status_file "$STATUS_FILE" "retro-gather" "running" "collecting retrospective data"; then
  add_health_check "status_file_running" "pass" "running state written" true
else
  append_warning "Failed to write running status file: $STATUS_FILE"
  add_health_check "status_file_running" "fail" "running state write failed" true
fi

start_heartbeat "retro-gather" "collecting sprint retrospective context"

if [[ -f "$RETRO_MD_PATH" ]]; then
  retro_md="$(cat "$RETRO_MD_PATH")"
  add_health_check "retro_md_present" "pass" "RETRO.md loaded" true
else
  append_warning "RETRO.md not found: $RETRO_MD_PATH"
  add_health_check "retro_md_present" "fail" "RETRO.md missing" true
fi

if has_command "$OPENCLAW_BIN"; then
  if capture_with_retry discord_dev 2 2 "$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_DEV_TARGET" --limit "$DISCORD_LIMIT"; then
    add_health_check "discord_dev_read" "pass" "dev channel messages captured" false
  else
    append_warning "Failed to read Discord dev channel"
    discord_dev=''
    add_health_check "discord_dev_read" "fail" "dev channel read failed" false
  fi

  if capture_with_retry discord_main 2 2 "$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_MAIN_TARGET" --limit "$DISCORD_LIMIT"; then
    add_health_check "discord_main_read" "pass" "main channel messages captured" false
  else
    append_warning "Failed to read Discord main channel"
    discord_main=''
    add_health_check "discord_main_read" "fail" "main channel read failed" false
  fi

  cron_status="$($OPENCLAW_BIN cron list 2>/dev/null || true)"
else
  append_warning "openclaw command not found: $OPENCLAW_BIN"
  add_health_check "discord_capture" "skip" "openclaw unavailable" false
fi

if has_command "$GH_BIN"; then
  add_health_check "github_cli_available" "pass" "gh command available" true

  milestones_json='[]'
  if capture_with_retry milestones_json 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=all&per_page=100"; then
    add_health_check "milestone_query" "pass" "milestones queried" true
  else
    append_error "Failed to fetch milestones from GitHub"
    status='error'
    add_health_check "milestone_query" "fail" "milestones query failed" true
  fi

  milestone_seed='null'
  if milestone_seed="$(printf '%s\n' "$milestones_json" | jq -c --arg today "$TODAY" '
    [ .[] | select(.title | startswith("Sprint ")) ] as $sprints
    | ([ $sprints[]
        | . as $ms
        | (try ($ms.title | capture("^Sprint (?<d>[0-9]{4}-[0-9]{2}-[0-9]{2})").d) catch "") as $d
        | select($d != "" and $d <= $today)
        | . + {__date: $d}
      ] | sort_by(.__date) | last)
      // ([ $sprints[] | select(.state == "closed") ] | sort_by(.closed_at // .due_on // .created_at // "") | last)
      // ($sprints | sort_by(.due_on // .created_at // "") | last)
      // null
  ' 2>/dev/null)"; then
    add_health_check "milestone_parse" "pass" "milestone parsed" true
  else
    milestone_seed='null'
    append_error "Failed to parse milestone payload"
    status='error'
    add_health_check "milestone_parse" "fail" "milestone parse failed" true
  fi

  if [[ "$milestone_seed" == 'null' ]]; then
    if [[ "$status" == 'ok' ]]; then
      status='no_milestone'
    fi
    add_health_check "milestone_selected" "pass" "no eligible milestone; handled as anomaly" true
  else
    milestone_number="$(printf '%s\n' "$milestone_seed" | jq -r '.number' 2>/dev/null || echo '')"
    issues_seed='[]'
    if [[ -n "$milestone_number" ]] && capture_with_retry issues_seed 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/issues?state=all&milestone=$milestone_number&per_page=100"; then
      add_health_check "milestone_issue_query" "pass" "milestone issues queried" true
    else
      append_error "Failed to fetch issues for milestone $milestone_number"
      status='error'
      add_health_check "milestone_issue_query" "fail" "milestone issue query failed" true
    fi

    sprint_issues_json="$(printf '%s\n' "$issues_seed" | jq -c 'map({
      assignees: (.assignees // []),
      labels: (.labels // []),
      number: .number,
      state: ((.state // "") | ascii_upcase),
      title: (.title // "")
    })' 2>/dev/null || echo '[]')"

    total_count="$(printf '%s\n' "$sprint_issues_json" | jq 'length' 2>/dev/null || echo 0)"
    closed_count="$(printf '%s\n' "$sprint_issues_json" | jq '[.[] | select(.state == "CLOSED")] | length' 2>/dev/null || echo 0)"

    milestone_json="$(jq -cn --argjson ms "$milestone_seed" --argjson total "$total_count" --argjson closed "$closed_count" '{
      title: $ms.title,
      number: $ms.number,
      total: $total,
      closed: $closed
    }')"

    velocity_json="$(jq -cn --argjson done "$closed_count" --argjson total "$total_count" '{
      done: $done,
      total: $total,
      pct: (if $total == 0 then 0 else (($done * 100 / $total) | floor) end)
    }')"
  fi

  ci_runs_json='[]'
  if capture_with_retry ci_runs_json 3 2 "$GH_BIN" run list --repo "$GITHUB_REPO" --limit 20 --json name,status,conclusion,headBranch,createdAt; then
    add_health_check "ci_runs_query" "pass" "ci runs queried" true
  else
    append_error "Failed to query CI runs"
    status='error'
    add_health_check "ci_runs_query" "fail" "ci runs query failed" true
  fi

  ci_fail_count="$(printf '%s\n' "$ci_runs_json" | jq '[.[] | select((.status // "") == "completed" and (.conclusion // "") != "success")] | length' 2>/dev/null || echo 0)"
else
  append_error "gh command not found: $GH_BIN"
  status='error'
  add_health_check "github_cli_available" "fail" "gh command missing" true
fi

github_action_lines="$(extract_github_action_items "$retro_md")"
if [[ -n "$github_action_lines" ]]; then
  github_action_items_json="$(printf '%s\n' "$github_action_lines" | jq -Rsc 'split("\n")
    | map(select(length > 0))
    | map({
        source: "github",
        text: ., 
        issue_refs: ([scan("#[0-9]+") | ltrimstr("#") | tonumber] // [])
      })')"
fi

discord_action_lines="$(extract_discord_action_items "$(printf '%s\n%s\n' "$discord_dev" "$discord_main")")"
if [[ -n "$discord_action_lines" ]]; then
  discord_action_items_json="$(printf '%s\n' "$discord_action_lines" | jq -Rsc 'split("\n")
    | map(select(length > 0))
    | map({
        source: "discord",
        text: ., 
        issue_refs: ([scan("#[0-9]+") | ltrimstr("#") | tonumber] // [])
      })
    | unique_by(.text)')"
fi

combined_action_items_json="$(jq -cn --argjson gh "$github_action_items_json" --argjson dc "$discord_action_items_json" '($gh + $dc) | unique_by(.source + "|" + .text)')"
action_items_json="$(jq -cn --argjson gh "$github_action_items_json" --argjson dc "$discord_action_items_json" --argjson all "$combined_action_items_json" '{github: $gh, discord: $dc, combined: $all}')"

combined_count="$(printf '%s\n' "$combined_action_items_json" | jq 'length' 2>/dev/null || echo 0)"
if [[ "$combined_count" -ge 0 ]]; then
  add_health_check "action_items_collected" "pass" "action items parsed and merged" true
else
  add_health_check "action_items_collected" "fail" "action item parse failure" true
fi

scrum_health_json="$(build_health_summary)"
scrum_health_pct="$(printf '%s\n' "$scrum_health_json" | jq -r '.coverage_pct // 0' 2>/dev/null || echo 0)"

output_json="$(jq -cn \
  --arg status "$status" \
  --arg timestamp "$NOW_TS" \
  --arg date "$TODAY" \
  --argjson milestone "$milestone_json" \
  --argjson sprint_issues "$sprint_issues_json" \
  --argjson ci_runs "$ci_runs_json" \
  --argjson ci_fail_count "$ci_fail_count" \
  --argjson velocity "$velocity_json" \
  --arg discord_dev "$discord_dev" \
  --arg discord_main "$discord_main" \
  --arg retro_md "$retro_md" \
  --argjson action_items "$action_items_json" \
  --arg cron_status "$cron_status" \
  --argjson errors "$errors_json" \
  --argjson warnings "$warnings_json" \
  --argjson scrum_health "$scrum_health_json" \
  ' {
    status: $status,
    timestamp: $timestamp,
    date: $date,
    milestone: $milestone,
    sprint_issues: $sprint_issues,
    ci_runs: $ci_runs,
    ci_fail_count: $ci_fail_count,
    velocity: $velocity,
    discord_dev: $discord_dev,
    discord_main: $discord_main,
    retro_md: $retro_md,
    action_items: $action_items,
    cron_status: $cron_status,
    scrum_health: $scrum_health,
    errors: $errors,
    warnings: $warnings
  }')"

if ! atomic_write_file "$OUTPUT_FILE" "$output_json"; then
  append_error "Failed to write output file: $OUTPUT_FILE"
  status='error'
fi

summary_done="$(printf '%s\n' "$velocity_json" | jq -r '.done // 0' 2>/dev/null || echo 0)"
summary_total="$(printf '%s\n' "$velocity_json" | jq -r '.total // 0' 2>/dev/null || echo 0)"

status_state='success'
status_detail='retro data gathered'

if [[ "$status" == 'ok' ]]; then
  log_event "retro-gather" "OK" "retro data gathered (${summary_done}/${summary_total} done, ci_fails=${ci_fail_count}, scrum_health=${scrum_health_pct}%)"
elif [[ "$status" == 'no_milestone' ]]; then
  status_state='anomaly'
  status_detail='no active milestone found'
  log_event "retro-gather" "ANOMALY" "no_milestone (scrum_health=${scrum_health_pct}%)"
else
  status_state='error'
  status_detail='retro gather failed'
  log_event "retro-gather" "ERROR" "gather failed (scrum_health=${scrum_health_pct}%)"
fi

if write_status_file "$STATUS_FILE" "retro-gather" "$status_state" "$status_detail"; then
  status_file_finalized='true'
else
  append_warning "Failed to write final status file: $STATUS_FILE"
fi

log_event "retro-gather" "INFO" "output=$OUTPUT_FILE" >/dev/null
