#!/usr/bin/env bash
# Gather sprint planning data and auto-include retrospective action items.

set -uo pipefail

resolve_repo_root() {
  if [[ -n "${REPO_ROOT:-}" ]]; then
    printf '%s\n' "$REPO_ROOT"
    return
  fi

  local default_root="$HOME/.openclaw/workspace/RalphGPU"
  if [[ -d "$default_root/.git" ]]; then
    printf '%s\n' "$default_root"
    return
  fi

  local script_root
  script_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." 2>/dev/null && pwd || true)"
  if [[ -n "$script_root" && -d "$script_root/.git" ]]; then
    printf '%s\n' "$script_root"
    return
  fi

  pwd
}

has_command() {
  local cmd="$1"
  if [[ "$cmd" == */* ]]; then
    [[ -x "$cmd" ]]
  else
    command -v "$cmd" >/dev/null 2>&1
  fi
}

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

append_error() {
  local msg="$1"
  errors_json="$(jq -cn --argjson arr "$errors_json" --arg m "$msg" '$arr + [$m]')"
}

REPO_ROOT="$(resolve_repo_root)"
GITHUB_REPO="${GITHUB_REPO:-ssql2014/RalphGPU}"

GH_BIN="${GH_BIN:-}"
if [[ -z "$GH_BIN" ]]; then
  if command -v gh >/dev/null 2>&1; then
    GH_BIN="$(command -v gh)"
  elif [[ -x /opt/homebrew/bin/gh ]]; then
    GH_BIN="/opt/homebrew/bin/gh"
  elif [[ -x /usr/local/bin/gh ]]; then
    GH_BIN="/usr/local/bin/gh"
  else
    GH_BIN="gh"
  fi
fi

OPENCLAW_BIN="${OPENCLAW_BIN:-}"
if [[ -z "$OPENCLAW_BIN" ]]; then
  if command -v openclaw >/dev/null 2>&1; then
    OPENCLAW_BIN="$(command -v openclaw)"
  elif [[ -x /opt/homebrew/bin/openclaw ]]; then
    OPENCLAW_BIN="/opt/homebrew/bin/openclaw"
  elif [[ -x /usr/local/bin/openclaw ]]; then
    OPENCLAW_BIN="/usr/local/bin/openclaw"
  else
    OPENCLAW_BIN="openclaw"
  fi
fi

SHARED_DIR="${SHARED_DIR:-$HOME/.openclaw/shared-memory/ralphgpu}"
OUTPUT_FILE="${OUTPUT_FILE:-$SHARED_DIR/sprint-planning-data.json}"
CRON_LOG="${CRON_LOG:-$SHARED_DIR/cron-bash.log}"
RETRO_DATA_FILE="${RETRO_DATA_FILE:-$SHARED_DIR/retro-data.json}"
RETRO_GATHER_SCRIPT="${RETRO_GATHER_SCRIPT:-$REPO_ROOT/scripts/retro-gather.sh}"
RETRO_MD_PATH="${RETRO_MD_PATH:-$REPO_ROOT/docs/RETRO.md}"

DISCORD_ACCOUNT="${DISCORD_ACCOUNT:-lily}"
DISCORD_MAIN_TARGET="${DISCORD_MAIN_TARGET:-channel:1468774996301316137}"
DISCORD_DEV_TARGET="${DISCORD_DEV_TARGET:-channel:1475083010968649778}"
DISCORD_LIMIT="${DISCORD_LIMIT:-40}"

NOW_TS="$(date '+%Y-%m-%dT%H:%M:%S')"
TODAY="$(date '+%Y-%m-%d')"

mkdir -p "$SHARED_DIR"

errors_json='[]'
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
    else
      append_error "Failed to refresh retro data via $RETRO_GATHER_SCRIPT"
    fi
  else
    append_error "Retro gather script missing or not executable: $RETRO_GATHER_SCRIPT"
  fi
fi

if [[ -s "$RETRO_DATA_FILE" ]]; then
  retro_data_json="$(cat "$RETRO_DATA_FILE")"
fi

retro_content="$(printf '%s\n' "$retro_data_json" | jq -r '.retro_md // empty')"
if [[ -z "$retro_content" && -f "$RETRO_MD_PATH" ]]; then
  retro_content="$(cat "$RETRO_MD_PATH")"
fi
if [[ -z "$retro_content" ]]; then
  append_error "Retro content missing from $RETRO_DATA_FILE and $RETRO_MD_PATH"
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
    dev_msg="$("$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_DEV_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null || true)"
    main_msg="$("$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_MAIN_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null || true)"
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
  open_milestones_json="$("$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=open&per_page=100" 2>/dev/null || echo '[]')"
  all_milestones_json="$("$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=all&per_page=100" 2>/dev/null || echo '[]')"

  existing_milestone_json="$(printf '%s\n' "$open_milestones_json" | jq -c '[.[] | select(.title | startswith("Sprint "))] | sort_by(.due_on // .created_at // "") | last // null | if . == null then null else {title: .title, number: .number, open_issues: .open_issues, due_on: .due_on, state: .state} end')"

  open_issues_json="$("$GH_BIN" issue list --repo "$GITHUB_REPO" --state open --limit 200 --json number,title,labels,assignees,milestone,state 2>/dev/null || echo '[]')"

  backlog_json="$(printf '%s\n' "$open_issues_json" | jq -c '
    def base: {assignees: (.assignees // []), labels: (.labels // []), milestone: .milestone, number: .number, title: .title};
    [ .[] | select(.milestone == null) ] as $pool |
    {
      p0: ($pool | map(select(any((.labels // [])[]?; ((.name // "") | test("^P0"; "i")))) | base) | sort_by(-.number)),
      p1: ($pool | map(select(any((.labels // [])[]?; ((.name // "") | test("^P1"; "i")))) | base) | sort_by(-.number)),
      all_unassigned: ($pool | map(select((.assignees // []) | length == 0) | base) | sort_by(-.number))
    }
  ')"

  open_prs_json="$("$GH_BIN" pr list --repo "$GITHUB_REPO" --state open --limit 50 --json number,title,headRefName,author,createdAt,url 2>/dev/null || echo '[]')"
  ci_runs_json="$("$GH_BIN" run list --repo "$GITHUB_REPO" --limit 20 --json name,status,conclusion,headBranch,createdAt 2>/dev/null || echo '[]')"

  closed_seed="$(printf '%s\n' "$all_milestones_json" | jq -c '[.[] | select(.title | startswith("Sprint ")) | select(.state == "closed")] | sort_by(.closed_at // .due_on // .created_at // "") | last // null')"

  if [[ "$closed_seed" != 'null' ]]; then
    closed_number="$(printf '%s\n' "$closed_seed" | jq -r '.number')"
    closed_issues="$("$GH_BIN" api "repos/$GITHUB_REPO/issues?state=all&milestone=$closed_number&per_page=100" 2>/dev/null || echo '[]')"
    closed_total="$(printf '%s\n' "$closed_issues" | jq 'length')"
    closed_done="$(printf '%s\n' "$closed_issues" | jq '[.[] | select((.state // "") == "closed")] | length')"
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

  for issue_num in $(printf '%s\n' "$retro_issue_refs_json" | jq -r '.[]'); do
    issue_state="$("$GH_BIN" issue view "$issue_num" --repo "$GITHUB_REPO" --json number,state,title,url 2>/dev/null || true)"
    if [[ -n "$issue_state" ]]; then
      retro_issue_states_json="$(jq -cn --argjson arr "$retro_issue_states_json" --argjson item "$issue_state" '$arr + [$item]')"
    fi
  done
  retro_issue_states_json="$(printf '%s\n' "$retro_issue_states_json" | jq -c 'unique_by(.number)')"
else
  append_error "gh command not found: $GH_BIN"
  status='error'
fi

retro_action_items_json="$(jq -cn --argjson gh "$retro_github_items_json" --argjson dc "$retro_discord_items_json" --argjson all "$retro_combined_items_json" --argjson states "$retro_issue_states_json" '{
  github: $gh,
  discord: $dc,
  combined: $all,
  issue_states: $states
}')"

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
  --argjson errors "$errors_json" \
  --argjson retro_auto_refresh "$retro_auto_refresh" \
  '{
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
    errors: $errors
  }')"

printf '%s\n' "$output_json" > "$OUTPUT_FILE"

p0_count="$(printf '%s\n' "$backlog_json" | jq -r '.p0 | length' 2>/dev/null || echo 0)"
p1_count="$(printf '%s\n' "$backlog_json" | jq -r '.p1 | length' 2>/dev/null || echo 0)"

if [[ "$status" == 'ok' ]]; then
  log_line="$(date '+%Y-%m-%d %H:%M') | sprint-planning-gather | OK | planning data gathered (P0:${p0_count} P1:${p1_count})"
else
  log_line="$(date '+%Y-%m-%d %H:%M') | sprint-planning-gather | ERROR | planning data gather failed"
fi

printf '%s\n' "$log_line" >> "$CRON_LOG"
printf '%s\n' "$log_line"
