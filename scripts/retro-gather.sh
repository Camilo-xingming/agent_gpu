#!/usr/bin/env bash
# Gather sprint retrospective data from GitHub + Discord.

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
OUTPUT_FILE="${OUTPUT_FILE:-$SHARED_DIR/retro-data.json}"
CRON_LOG="${CRON_LOG:-$SHARED_DIR/cron-bash.log}"
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

if [[ -f "$RETRO_MD_PATH" ]]; then
  retro_md="$(cat "$RETRO_MD_PATH")"
else
  append_error "RETRO.md not found: $RETRO_MD_PATH"
fi

if has_command "$OPENCLAW_BIN"; then
  if ! discord_dev="$("$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_DEV_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null)"; then
    append_error "Failed to read Discord dev channel"
    discord_dev=''
  fi
  if ! discord_main="$("$OPENCLAW_BIN" message read --account "$DISCORD_ACCOUNT" --channel discord --target "$DISCORD_MAIN_TARGET" --limit "$DISCORD_LIMIT" 2>/dev/null)"; then
    append_error "Failed to read Discord main channel"
    discord_main=''
  fi
  cron_status="$("$OPENCLAW_BIN" cron list 2>/dev/null || true)"
else
  append_error "openclaw command not found: $OPENCLAW_BIN"
fi

if has_command "$GH_BIN"; then
  milestones_json="$("$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=all&per_page=100" 2>/dev/null || echo '[]')"
  milestone_seed="$(printf '%s\n' "$milestones_json" | jq -c --arg today "$TODAY" '
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
  ')"

  if [[ "$milestone_seed" == 'null' ]]; then
    status='no_milestone'
  else
    milestone_number="$(printf '%s\n' "$milestone_seed" | jq -r '.number')"
    issues_seed="$("$GH_BIN" api "repos/$GITHUB_REPO/issues?state=all&milestone=$milestone_number&per_page=100" 2>/dev/null || echo '[]')"

    sprint_issues_json="$(printf '%s\n' "$issues_seed" | jq -c 'map({
      assignees: (.assignees // []),
      labels: (.labels // []),
      number: .number,
      state: ((.state // "") | ascii_upcase),
      title: (.title // "")
    })')"

    total_count="$(printf '%s\n' "$sprint_issues_json" | jq 'length')"
    closed_count="$(printf '%s\n' "$sprint_issues_json" | jq '[.[] | select(.state == "CLOSED")] | length')"

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

  ci_runs_json="$("$GH_BIN" run list --repo "$GITHUB_REPO" --limit 20 --json name,status,conclusion,headBranch,createdAt 2>/dev/null || echo '[]')"
  ci_fail_count="$(printf '%s\n' "$ci_runs_json" | jq '[.[] | select((.status // "") == "completed" and (.conclusion // "") != "success")] | length')"
else
  append_error "gh command not found: $GH_BIN"
  status='error'
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
  '{
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
    errors: $errors
  }')"

printf '%s\n' "$output_json" > "$OUTPUT_FILE"

summary_done="$(printf '%s\n' "$velocity_json" | jq -r '.done // 0' 2>/dev/null || echo 0)"
summary_total="$(printf '%s\n' "$velocity_json" | jq -r '.total // 0' 2>/dev/null || echo 0)"

if [[ "$status" == 'ok' ]]; then
  log_line="$(date '+%Y-%m-%d %H:%M') | retro-gather | OK | retro data gathered (${summary_done}/${summary_total} done, ci_fails=${ci_fail_count})"
elif [[ "$status" == 'no_milestone' ]]; then
  log_line="$(date '+%Y-%m-%d %H:%M') | retro-gather | ANOMALY | no_milestone"
else
  log_line="$(date '+%Y-%m-%d %H:%M') | retro-gather | ERROR | gather failed"
fi

printf '%s\n' "$log_line" >> "$CRON_LOG"
printf '%s\n' "$log_line"
