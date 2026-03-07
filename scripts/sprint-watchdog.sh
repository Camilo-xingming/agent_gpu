#!/usr/bin/env bash
# Sprint watchdog: enforce assignee and branch start SLA for current sprint issues.

set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/cron-common.sh" ]]; then
  # shellcheck source=scripts/cron-common.sh
  source "$SCRIPT_DIR/cron-common.sh"
  configure_gh_proxy_env || true
fi

GITHUB_REPO="${GITHUB_REPO:-ssql2014/RalphGPU}"
WIP_SLA_HOURS="${WIP_SLA_HOURS:-2}"
MILESTONE_TITLE="${MILESTONE_TITLE:-}"
GH_BIN="${GH_BIN:-gh}"
OUTPUT_JSON='false'

usage() {
  cat <<'USAGE'
Usage: sprint-watchdog.sh [options]

Options:
  --repo <owner/repo>       GitHub repo (default: ssql2014/RalphGPU)
  --milestone <title>       Sprint milestone title to inspect (default: latest open sprint)
  --threshold-hours <num>   Branch SLA threshold in hours (default: 2)
  --json                    Output JSON only
  --help                    Show this help message

Exit code:
  0: all OK
  1: WARN exists (assigned issue missing branch after threshold)
  2: FAIL exists (issue missing assignee)
  3: runtime/config error
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      GITHUB_REPO="$2"
      shift 2
      ;;
    --milestone)
      MILESTONE_TITLE="$2"
      shift 2
      ;;
    --threshold-hours)
      WIP_SLA_HOURS="$2"
      shift 2
      ;;
    --json)
      OUTPUT_JSON='true'
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 3
      ;;
  esac
done

if ! command -v "$GH_BIN" >/dev/null 2>&1; then
  echo "gh not found: $GH_BIN" >&2
  exit 3
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found" >&2
  exit 3
fi

if ! [[ "$WIP_SLA_HOURS" =~ ^[0-9]+$ ]]; then
  echo "--threshold-hours must be a non-negative integer" >&2
  exit 3
fi

milestones_json='[]'
if ! milestones_json="$("$GH_BIN" api "repos/$GITHUB_REPO/milestones?state=open&per_page=100" 2>/dev/null)"; then
  echo "failed to query milestones for $GITHUB_REPO" >&2
  exit 3
fi

milestone_json='null'
if [[ -n "$MILESTONE_TITLE" ]]; then
  milestone_json="$(printf '%s\n' "$milestones_json" | jq -c --arg title "$MILESTONE_TITLE" '[.[] | select((.title // "") == $title)] | first // null' 2>/dev/null || echo 'null')"
else
  milestone_json="$(printf '%s\n' "$milestones_json" | jq -c '
    ([.[] | select((.title // "") | test("^Sprint[[:space:]]+[0-9]+"; "i"))] | sort_by(.number // 0) | last)
    // ([.[]] | sort_by(.number // 0) | last)
    // null
  ' 2>/dev/null || echo 'null')"
fi

if [[ "$milestone_json" == 'null' ]]; then
  if [[ "$OUTPUT_JSON" == 'true' ]]; then
    jq -cn --arg repo "$GITHUB_REPO" --arg milestone "$MILESTONE_TITLE" '{repo: $repo, milestone: (if ($milestone | length) == 0 then null else $milestone end), status: "error", error: "no open milestone found", summary: {ok: 0, warn: 0, fail: 0, total: 0}, results: []}'
  else
    echo "Sprint Watchdog: no open milestone found in $GITHUB_REPO"
  fi
  exit 3
fi

milestone_title="$(printf '%s\n' "$milestone_json" | jq -r '.title // empty' 2>/dev/null || true)"
milestone_number="$(printf '%s\n' "$milestone_json" | jq -r '.number // empty' 2>/dev/null || true)"

if [[ -z "$milestone_title" || -z "$milestone_number" ]]; then
  echo "failed to resolve sprint milestone" >&2
  exit 3
fi

issues_json='[]'
if ! issues_json="$("$GH_BIN" issue list --repo "$GITHUB_REPO" --state open --milestone "$milestone_title" --limit 200 --json number,title,assignees,createdAt,url 2>/dev/null)"; then
  echo "failed to query open issues for milestone: $milestone_title" >&2
  exit 3
fi

now_epoch="$(date +%s)"
threshold_seconds=$((WIP_SLA_HOURS * 3600))
results_json='[]'
ok_count=0
warn_count=0
fail_count=0

while IFS= read -r issue_item; do
  [[ -n "$issue_item" ]] || continue

  issue_number="$(printf '%s\n' "$issue_item" | jq -r '.number' 2>/dev/null || echo '')"
  issue_title="$(printf '%s\n' "$issue_item" | jq -r '.title // ""' 2>/dev/null || echo '')"
  issue_url="$(printf '%s\n' "$issue_item" | jq -r '.url // ""' 2>/dev/null || echo '')"
  assignee_count="$(printf '%s\n' "$issue_item" | jq '(.assignees // []) | length' 2>/dev/null || echo 0)"
  assignees_json="$(printf '%s\n' "$issue_item" | jq -c '[.assignees[]?.login]' 2>/dev/null || echo '[]')"
  created_at="$(printf '%s\n' "$issue_item" | jq -r '.createdAt // empty' 2>/dev/null || echo '')"

  age_seconds=0
  if [[ -n "$created_at" ]]; then
    created_epoch="$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$created_at" +%s 2>/dev/null || echo 0)"
    if [[ "$created_epoch" -gt 0 ]]; then
      age_seconds=$((now_epoch - created_epoch))
      if [[ "$age_seconds" -lt 0 ]]; then
        age_seconds=0
      fi
    fi
  fi

  has_branch='false'
  if [[ "$assignee_count" -gt 0 && -n "$issue_number" ]]; then
    branch_refs_json='[]'
    if branch_refs_json="$("$GH_BIN" api "repos/$GITHUB_REPO/git/matching-refs/heads/issue-$issue_number/" 2>/dev/null)"; then
      branch_count="$(printf '%s\n' "$branch_refs_json" | jq 'if type == "array" then length else 0 end' 2>/dev/null || echo 0)"
      if [[ "$branch_count" -gt 0 ]]; then
        has_branch='true'
      fi
    fi
  fi

  status='OK'
  reason='all checks passed'

  if [[ "$assignee_count" -eq 0 ]]; then
    status='FAIL'
    reason='no assignee'
    fail_count=$((fail_count + 1))
  elif [[ "$age_seconds" -ge "$threshold_seconds" && "$has_branch" != 'true' ]]; then
    status='WARN'
    reason="no branch after ${WIP_SLA_HOURS}h"
    warn_count=$((warn_count + 1))
  else
    ok_count=$((ok_count + 1))
  fi

  result_item="$(jq -cn \
    --argjson num "$issue_number" \
    --arg title "$issue_title" \
    --arg url "$issue_url" \
    --arg status "$status" \
    --arg reason "$reason" \
    --argjson assignees "$assignees_json" \
    --argjson age_seconds "$age_seconds" \
    --arg has_branch "$has_branch" \
    '{
      number: $num,
      title: $title,
      url: $url,
      status: $status,
      reason: $reason,
      assignees: $assignees,
      age_seconds: $age_seconds,
      age_hours: (($age_seconds / 3600 * 100 | floor) / 100),
      has_branch: ($has_branch == "true")
    }')"

  results_json="$(jq -cn --argjson arr "$results_json" --argjson item "$result_item" '$arr + [$item]')"
done < <(printf '%s\n' "$issues_json" | jq -c '.[]' 2>/dev/null)

total_count=$((ok_count + warn_count + fail_count))
overall_status='OK'
exit_code=0
if [[ "$fail_count" -gt 0 ]]; then
  overall_status='FAIL'
  exit_code=2
elif [[ "$warn_count" -gt 0 ]]; then
  overall_status='WARN'
  exit_code=1
fi

output_json="$(jq -cn \
  --arg repo "$GITHUB_REPO" \
  --arg milestone_title "$milestone_title" \
  --argjson milestone_number "$milestone_number" \
  --arg status "$overall_status" \
  --argjson threshold_hours "$WIP_SLA_HOURS" \
  --argjson now "$now_epoch" \
  --argjson ok "$ok_count" \
  --argjson warn "$warn_count" \
  --argjson fail "$fail_count" \
  --argjson total "$total_count" \
  --argjson results "$results_json" \
  '{
    repo: $repo,
    milestone: {title: $milestone_title, number: $milestone_number},
    status: $status,
    threshold_hours: $threshold_hours,
    generated_at_epoch: $now,
    summary: {ok: $ok, warn: $warn, fail: $fail, total: $total},
    results: $results
  }')"

if [[ "$OUTPUT_JSON" == 'true' ]]; then
  printf '%s\n' "$output_json"
else
  echo "Sprint Watchdog | $GITHUB_REPO | $milestone_title (#$milestone_number)"
  echo "Threshold: ${WIP_SLA_HOURS}h"
  if [[ "$total_count" -eq 0 ]]; then
    echo "OK no open issues in milestone"
  else
    printf '%s\n' "$output_json" | jq -r '.results[] | "\(.status) #\(.number) \(.title) [\(.reason)]"'
  fi
  echo "Summary: OK=$ok_count WARN=$warn_count FAIL=$fail_count TOTAL=$total_count"
fi

exit "$exit_code"
