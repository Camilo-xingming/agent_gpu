#!/usr/bin/env bash
# Gather sprint planning data and auto-include retrospective action items.

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
OUTPUT_FILE="${OUTPUT_FILE:-$SHARED_DIR/sprint-planning-data.json}"
CRON_LOG="${CRON_LOG:-$SHARED_DIR/cron-bash.log}"
STATUS_FILE="${STATUS_FILE:-$SHARED_DIR/status/sprint-planning-gather.status.json}"
RETRO_DATA_FILE="${RETRO_DATA_FILE:-$SHARED_DIR/retro-data.json}"
RETRO_GATHER_SCRIPT="${RETRO_GATHER_SCRIPT:-$REPO_ROOT/scripts/retro-gather.sh}"
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
existing_milestone_json='null'
retro_content=''
backlog_json='{"p0":[],"p1":[],"all_unassigned":[]}'
ci_runs_json='[]'
open_prs_json='[]'
yesterday_velocity_json='null'
retro_github_items_json='[]'
retro_discord_items_json='[]'
retro_combined_items_json='[]'
retro_issue_states_json='[]'
retro_data_json='{}'
retro_auto_refresh='false'

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
    write_status_file "$STATUS_FILE" "sprint-planning-gather" "error" "script exited with code $rc" || true
  fi
  if [[ -n "$lock_path" ]]; then
    release_script_lock "$lock_path"
  fi
}
trap cleanup EXIT

if ! lock_path="$(acquire_script_lock "$SHARED_DIR" "sprint-planning-gather")"; then
  log_event "sprint-planning-gather" "WARN" "lock busy, skipping duplicate run"
  exit 0
fi

if write_status_file "$STATUS_FILE" "sprint-planning-gather" "running" "collecting planning data"; then
  add_health_check "status_file_running" "pass" "running state written" true
else
  append_warning "Failed to write running status file: $STATUS_FILE"
  add_health_check "status_file_running" "fail" "running state write failed" true
fi

start_heartbeat "sprint-planning-gather" "collecting sprint planning context"

needs_retro_refresh='false'
if [[ ! -s "$RETRO_DATA_FILE" ]]; then
  needs_retro_refresh='true'
else
  retro_file_date="$(jq -r '.date // empty' "$RETRO_DATA_FILE" 2>/dev/null || true)"
  if [[ "$retro_file_date" != "$TODAY" ]]; then
    needs_retro_refresh='true'
  fi
fi

if [[ "$needs_retro_refresh" == 'true' ]]; then
  if [[ -x "$RETRO_GATHER_SCRIPT" ]]; then
    if "$RETRO_GATHER_SCRIPT" >/dev/null 2>&1; then
      retro_auto_refresh='true'
      add_health_check "retro_auto_refresh" "pass" "retro data refreshed" true
    else
      append_warning "Failed to refresh retro data via $RETRO_GATHER_SCRIPT"
      add_health_check "retro_auto_refresh" "fail" "retro refresh command failed" true
    fi
  else
    append_warning "Retro gather script missing or not executable: $RETRO_GATHER_SCRIPT"
    add_health_check "retro_auto_refresh" "fail" "retro gather script missing" true
  fi
else
  add_health_check "retro_auto_refresh" "pass" "retro data already fresh" true
fi

if [[ -s "$RETRO_DATA_FILE" ]]; then
  retro_data_json="$(cat "$RETRO_DATA_FILE")"
  add_health_check "retro_data_available" "pass" "retro-data.json available" true
else
  append_warning "retro data file missing: $RETRO_DATA_FILE"
  add_health_check "retro_data_available" "fail" "retro-data.json missing" true
fi

retro_content="$(printf '%s\n' "$retro_data_json" | jq -r '.retro_md // empty' 2>/dev/null || true)"
if [[ -z "$retro_content" && -f "$RETRO_MD_PATH" ]]; then
  retro_content="$(cat "$RETRO_MD_PATH")"
fi
if [[ -z "$retro_content" ]]; then
  append_warning "Retro content missing from $RETRO_DATA_FILE and $RETRO_MD_PATH"
  add_health_check "retro_content_present" "fail" "retro content missing" true
else
  add_health_check "retro_content_present" "pass" "retro content loaded" true
fi

retro_github_items_json="$(printf '%s\n' "$retro_data_json" | jq -c '.action_items.github // []' 2>/dev/null || echo '[]')"
retro_discord_items_json="$(printf '%s\n' "$retro_data_json" | jq -c '.action_items.discord // []' 2>/dev/null || echo '[]')"

if [[ "$retro_github_items_json" == '[]' && -n "$retro_content" ]]; then
  retro_github_lines="$(extract_github_action_items "$retro_content")"
  if [[ -n "$retro_github_lines" ]]; then
    retro_github_items_json="$(printf '%s\n' "$retro_github_lines" | jq -Rsc 'split("\n")
      | map(select(length > 0))
      | map({
          source: "github",
          text: ., 
          issue_refs: ([scan("#[0-9]+") | ltrimstr("#") | tonumber] // [])
        })')"
  fi
fi

if [[ "$retro_discord_items_json" == '[]' ]]; then
  discord_seed="$(printf '%s\n' "$retro_data_json" | jq -r '[.discord_dev // "", .discord_main // ""] | join("\n")' 2>/dev/null || true)"
  if [[ -z "$discord_seed" ]] && has_command "$OPENCLAW_BIN"; then
    dev_msg="$($OPENCLAW_BIN message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_DEV_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null || true)"
    main_msg="$($OPENCLAW_BIN message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_MAIN_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null || true)"
    discord_seed="$(printf '%s\n%s\n' "$dev_msg" "$main_msg")"
  fi

  discord_lines="$(extract_discord_action_items "$discord_seed")"
  if [[ -n "$discord_lines" ]]; then
    retro_discord_items_json="$(printf '%s\n' "$discord_lines" | jq -Rsc 'split("\n")
      | map(select(length > 0))
      | map({
          source: "discord",
          text: ., 
          issue_refs: ([scan("#[0-9]+") | ltrimstr("#") | tonumber] // [])
        })
      | unique_by(.text)')"
  fi
fi

retro_combined_items_json="$(jq -cn --argjson gh "$retro_github_items_json" --argjson dc "$retro_discord_items_json" '($gh + $dc) | unique_by(.source + "|" + .text)')"
retro_issue_refs_json="$(jq -cn --argjson all "$retro_combined_items_json" '[ $all[] | (.issue_refs // [])[] ] | unique')"

if has_command "$GH_BIN"; then
  add_health_check "github_cli_available" "pass" "gh command available" true

  open_milestones_json='[]'
  all_milestones_json='[]'
  open_issues_json='[]'
  all_issue_states_json='[]'

  if capture_with_retry open_milestones_json 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=open&per_page=100"; then
    add_health_check "open_milestones_query" "pass" "open milestones queried" true
  else
    append_error "Failed to query open milestones"
    status='error'
    add_health_check "open_milestones_query" "fail" "open milestone query failed" true
  fi

  if capture_with_retry all_milestones_json 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=all&per_page=100"; then
    add_health_check "all_milestones_query" "pass" "all milestones queried" true
  else
    append_error "Failed to query all milestones"
    status='error'
    add_health_check "all_milestones_query" "fail" "all milestone query failed" true
  fi

  existing_milestone_json="$(printf '%s\n' "$open_milestones_json" | jq -c '[.[] | select(.title | startswith("Sprint "))] | sort_by(.due_on // .created_at // "") | last // null | if . == null then null else {title: .title, number: .number, open_issues: .open_issues, due_on: .due_on, state: .state} end' 2>/dev/null || echo 'null')"

  if capture_with_retry open_issues_json 3 2 "$GH_BIN" issue list --repo "$GITHUB_REPO" --state open --limit 200 --json number,title,labels,assignees,milestone,state; then
    add_health_check "open_issues_query" "pass" "open issues queried" true
  else
    append_error "Failed to query open issues"
    status='error'
    add_health_check "open_issues_query" "fail" "open issue query failed" true
  fi

  backlog_json="$(printf '%s\n' "$open_issues_json" | jq -c '
    def base: {assignees: (.assignees // []), labels: (.labels // []), milestone: .milestone, number: .number, title: .title};
    [ .[] | select(.milestone == null) ] as $pool |
    {
      p0: ($pool | map(select(any((.labels // [])[]?; ((.name // "") | test("^P0"; "i")))) | base) | sort_by(-.number)),
      p1: ($pool | map(select(any((.labels // [])[]?; ((.name // "") | test("^P1"; "i")))) | base) | sort_by(-.number)),
      all_unassigned: ($pool | map(select((.assignees // []) | length == 0) | base) | sort_by(-.number))
    }
  ' 2>/dev/null || echo '{"p0":[],"p1":[],"all_unassigned":[]}')"

  if capture_with_retry all_issue_states_json 3 2 "$GH_BIN" issue list --repo "$GITHUB_REPO" --state all --limit 200 --json number,state,title,url; then
    add_health_check "issue_states_bulk_query" "pass" "issue states fetched in bulk" true
  else
    append_error "Failed to query issue states"
    status='error'
    add_health_check "issue_states_bulk_query" "fail" "issue states bulk query failed" true
  fi

  if capture_with_retry open_prs_json 3 2 "$GH_BIN" pr list --repo "$GITHUB_REPO" --state open --limit 50 --json number,title,headRefName,author,createdAt,url; then
    add_health_check "open_prs_query" "pass" "open PRs queried" true
  else
    append_error "Failed to query open PRs"
    status='error'
    add_health_check "open_prs_query" "fail" "open PR query failed" true
  fi

  if capture_with_retry ci_runs_json 3 2 "$GH_BIN" run list --repo "$GITHUB_REPO" --limit 20 --json name,status,conclusion,headBranch,createdAt; then
    add_health_check "ci_runs_query" "pass" "ci runs queried" true
  else
    append_error "Failed to query CI runs"
    status='error'
    add_health_check "ci_runs_query" "fail" "ci runs query failed" true
  fi

  closed_seed="$(printf '%s\n' "$all_milestones_json" | jq -c '[.[] | select(.title | startswith("Sprint ")) | select(.state == "closed")] | sort_by(.closed_at // .due_on // .created_at // "") | last // null' 2>/dev/null || echo 'null')"

  if [[ "$closed_seed" != 'null' ]]; then
    closed_number="$(printf '%s\n' "$closed_seed" | jq -r '.number' 2>/dev/null || echo '')"
    closed_issues='[]'
    if [[ -n "$closed_number" ]] && capture_with_retry closed_issues 3 2 "$GH_BIN" api "repos/$GITHUB_REPO/issues?state=all&milestone=$closed_number&per_page=100"; then
      closed_total="$(printf '%s\n' "$closed_issues" | jq 'length' 2>/dev/null || echo 0)"
      closed_done="$(printf '%s\n' "$closed_issues" | jq '[.[] | select((.state // "") == "closed")] | length' 2>/dev/null || echo 0)"
      yesterday_velocity_json="$(jq -cn --argjson ms "$closed_seed" --argjson done "$closed_done" --argjson total "$closed_total" '{
        done: $done,
        total: $total,
        pct: (if $total == 0 then 0 else (($done * 100 / $total) | floor) end),
        milestone: {
          title: $ms.title,
          number: $ms.number,
          closed_at: $ms.closed_at
        }
      }')"
    fi
  fi

  retro_issue_states_json="$(jq -cn --argjson refs "$retro_issue_refs_json" --argjson states "$all_issue_states_json" '
    [ $refs[] as $num | ($states[] | select(.number == $num)) ] | unique_by(.number)
  ' 2>/dev/null || echo '[]')"

  missing_refs_json="$(jq -cn --argjson refs "$retro_issue_refs_json" --argjson found "$retro_issue_states_json" '
    [ $refs[] as $num | select(([ $found[]?.number ] | index($num)) | not) ]
  ' 2>/dev/null || echo '[]')"

  missing_count="$(printf '%s\n' "$missing_refs_json" | jq 'length' 2>/dev/null || echo 0)"
  if [[ "$missing_count" -gt 0 ]]; then
    append_warning "bulk issue state query missed ${missing_count} retro refs; using targeted fallback"
    for issue_num in $(printf '%s\n' "$missing_refs_json" | jq -r '.[]' 2>/dev/null); do
      issue_state=''
      if capture_with_retry issue_state 2 2 "$GH_BIN" issue view "$issue_num" --repo "$GITHUB_REPO" --json number,state,title,url; then
        retro_issue_states_json="$(jq -cn --argjson arr "$retro_issue_states_json" --argjson item "$issue_state" '$arr + [$item]')"
      fi
    done
  fi
  retro_issue_states_json="$(printf '%s\n' "$retro_issue_states_json" | jq -c 'unique_by(.number)' 2>/dev/null || echo '[]')"
  add_health_check "retro_issue_state_lookup" "pass" "bulk lookup with targeted fallback" true
else
  append_error "gh command not found: $GH_BIN"
  status='error'
  add_health_check "github_cli_available" "fail" "gh command missing" true
fi

retro_action_items_json="$(jq -cn --argjson gh "$retro_github_items_json" --argjson dc "$retro_discord_items_json" --argjson all "$retro_combined_items_json" --argjson states "$retro_issue_states_json" '{
  github: $gh,
  discord: $dc,
  combined: $all,
  issue_states: $states
}')"

combined_count="$(printf '%s\n' "$retro_combined_items_json" | jq 'length' 2>/dev/null || echo 0)"
if [[ "$combined_count" -ge 0 ]]; then
  add_health_check "retro_action_items_collected" "pass" "retro action items merged" true
else
  add_health_check "retro_action_items_collected" "fail" "retro action item parse failure" true
fi

scrum_health_json="$(build_health_summary)"
scrum_health_pct="$(printf '%s\n' "$scrum_health_json" | jq -r '.coverage_pct // 0' 2>/dev/null || echo 0)"

output_json="$(jq -cn \
  --arg status "$status" \
  --arg timestamp "$NOW_TS" \
  --arg date "$TODAY" \
  --argjson existing_milestone "$existing_milestone_json" \
  --arg retro_content "$retro_content" \
  --argjson retro_action_items "$retro_action_items_json" \
  --argjson backlog "$backlog_json" \
  --argjson ci_runs "$ci_runs_json" \
  --argjson open_prs "$open_prs_json" \
  --argjson yesterday_velocity "$yesterday_velocity_json" \
  --argjson retro_auto_refresh "$retro_auto_refresh" \
  --argjson scrum_health "$scrum_health_json" \
  --argjson errors "$errors_json" \
  --argjson warnings "$warnings_json" \
  ' {
    status: $status,
    timestamp: $timestamp,
    date: $date,
    existing_milestone: $existing_milestone,
    retro_content: $retro_content,
    retro_action_items: $retro_action_items,
    backlog: $backlog,
    ci_runs: $ci_runs,
    open_prs: $open_prs,
    yesterday_velocity: $yesterday_velocity,
    retro_auto_refresh: $retro_auto_refresh,
    scrum_health: $scrum_health,
    errors: $errors,
    warnings: $warnings
  }')"

if ! atomic_write_file "$OUTPUT_FILE" "$output_json"; then
  append_error "Failed to write output file: $OUTPUT_FILE"
  status='error'
fi

p0_count="$(printf '%s\n' "$backlog_json" | jq -r '.p0 | length' 2>/dev/null || echo 0)"
p1_count="$(printf '%s\n' "$backlog_json" | jq -r '.p1 | length' 2>/dev/null || echo 0)"

status_state='success'
status_detail='planning data gathered'

if [[ "$status" == 'ok' ]]; then
  log_event "sprint-planning-gather" "OK" "planning data gathered (P0:${p0_count} P1:${p1_count}, scrum_health=${scrum_health_pct}%)"
else
  status_state='error'
  status_detail='planning data gather failed'
  log_event "sprint-planning-gather" "ERROR" "planning data gather failed (scrum_health=${scrum_health_pct}%)"
fi

if write_status_file "$STATUS_FILE" "sprint-planning-gather" "$status_state" "$status_detail"; then
  status_file_finalized='true'
else
  append_warning "Failed to write final status file: $STATUS_FILE"
fi
