#!/usr/bin/env bash
# Sprint watchdog: enforce assignee/branch start SLA and cross-review SLA for sprint issues.

set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/cron-common.sh" ]]; then
  # shellcheck source=scripts/cron-common.sh
  source "$SCRIPT_DIR/cron-common.sh"
  configure_gh_proxy_env || true
fi

REPO_ROOT="${REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." 2>/dev/null && pwd || pwd)}"

GITHUB_REPO="${GITHUB_REPO:-ssql2014/RalphGPU}"
WIP_SLA_HOURS="${WIP_SLA_HOURS:-2}"
REVIEW_SLA_MINUTES="${REVIEW_SLA_MINUTES:-30}"
MILESTONE_TITLE="${MILESTONE_TITLE:-}"
GH_BIN="${GH_BIN:-gh}"
OPENCLAW_BIN="${OPENCLAW_BIN:-openclaw}"
DISCORD_ACCOUNT="${DISCORD_ACCOUNT:-codex}"
DISCORD_DEV_TARGET="${DISCORD_DEV_TARGET:-channel:1475083010968649778}"
DISCORD_NOTIFY="${DISCORD_NOTIFY:-true}"
MEMORY_FILE="${MEMORY_FILE:-$REPO_ROOT/MEMORY.md}"
MEMORY_STALE_GAP="${MEMORY_STALE_GAP:-3}"
OUTPUT_JSON='false'

usage() {
  cat <<'USAGE'
Usage: sprint-watchdog.sh [options]

Options:
  --repo <owner/repo>       GitHub repo (default: ssql2014/RalphGPU)
  --milestone <title>       Sprint milestone title to inspect (default: latest open sprint)
  --threshold-hours <num>   Branch SLA threshold in hours (default: 2)
  --review-sla-minutes <n>  Cross-review SLA threshold in minutes (default: 30)
  --no-discord-notify       Disable Discord escalation notification on SLA breach
  --json                    Output JSON only
  --help                    Show this help message

Exit code:
  0: all OK
  1: WARN exists (assigned issue missing branch after threshold)
  2: FAIL exists (issue missing assignee beyond threshold)
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
    --review-sla-minutes)
      REVIEW_SLA_MINUTES="$2"
      shift 2
      ;;
    --no-discord-notify)
      DISCORD_NOTIFY='false'
      shift
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

if ! [[ "$REVIEW_SLA_MINUTES" =~ ^[0-9]+$ ]]; then
  echo "--review-sla-minutes must be a non-negative integer" >&2
  exit 3
fi

if ! [[ "$MEMORY_STALE_GAP" =~ ^[0-9]+$ ]]; then
  echo "MEMORY_STALE_GAP must be a non-negative integer" >&2
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

active_sprint_number=''
if [[ "$milestone_title" =~ [Ss]print[[:space:]]+([0-9]+) ]]; then
  active_sprint_number="${BASH_REMATCH[1]}"
fi

memory_state_sprint=''
memory_gap=''
memory_reason='memory check skipped'
memory_stale_flag='false'
memory_notify_result='not_triggered'

if [[ -n "$active_sprint_number" ]]; then
  if [[ -f "$MEMORY_FILE" ]]; then
    memory_state_sprint="$(grep -Eo 'Current State \(Sprint[[:space:]]+[0-9]+\)' "$MEMORY_FILE" | head -n 1 | grep -Eo '[0-9]+' || true)"
    if [[ -n "$memory_state_sprint" ]]; then
      memory_gap=$((active_sprint_number - memory_state_sprint))
      if [[ "$memory_gap" -lt 0 ]]; then
        memory_gap=0
      fi
      if [[ "$memory_gap" -gt "$MEMORY_STALE_GAP" ]]; then
        memory_stale_flag='true'
        memory_reason="MEMORY.md current state is ${memory_gap} sprint(s) behind"
      else
        memory_reason='memory sprint state is fresh'
      fi
    else
      memory_reason='Current State (Sprint N) not found in MEMORY.md'
    fi
  else
    memory_reason='MEMORY.md file not found'
  fi
fi

issues_json='[]'
if ! issues_json="$("$GH_BIN" issue list --repo "$GITHUB_REPO" --state open --milestone "$milestone_title" --limit 200 --json number,title,assignees,createdAt,url 2>/dev/null)"; then
  echo "failed to query open issues for milestone: $milestone_title" >&2
  exit 3
fi

open_prs_json='[]'
if ! open_prs_json="$("$GH_BIN" pr list --repo "$GITHUB_REPO" --state open --limit 200 --json number,title,url,headRefName,comments 2>/dev/null)"; then
  echo "failed to query open PRs for $GITHUB_REPO" >&2
  exit 3
fi

now_epoch="$(date +%s)"
threshold_seconds=$((WIP_SLA_HOURS * 3600))
review_sla_seconds=$((REVIEW_SLA_MINUTES * 60))
results_json='[]'
review_sla_violations_json='[]'
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
    created_epoch="$(printf '%s\n' "$issue_item" | jq -r '(.createdAt // "" | fromdateiso8601? // 0)' 2>/dev/null || echo 0)"
    if [[ "$created_epoch" =~ ^[0-9]+$ && "$created_epoch" -gt 0 ]]; then
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

  issue_prs_json='[]'
  if [[ -n "$issue_number" ]]; then
    issue_prs_json="$(printf '%s\n' "$open_prs_json" | jq -c --arg n "$issue_number" '
      [ .[] | select((.headRefName // "") | test("^issue-" + $n + "(/|$)")) ]
    ' 2>/dev/null || echo '[]')"
  fi

  review_sla_checks_json='[]'
  review_sla_violation_count=0
  review_sla_pending_count=0
  while IFS= read -r pr_item; do
    [[ -n "$pr_item" ]] || continue
    pr_number="$(printf '%s\n' "$pr_item" | jq -r '.number // empty' 2>/dev/null || true)"
    pr_comments_json="$(printf '%s\n' "$pr_item" | jq -c '.comments // []' 2>/dev/null || echo '[]')"

    pr_checks_json="$(jq -cn \
      --argjson comments "$pr_comments_json" \
      --argjson now "$now_epoch" \
      --argjson threshold "$review_sla_seconds" '
        ($comments | sort_by(.createdAt // "")) as $ordered
        | [range(0; ($ordered | length)) as $idx
           | ($ordered[$idx]) as $req
           | select(($req.body // "") | test("\\bCROSS_REVIEW_REQUEST\\b"; "i"))
           | ($req.createdAt | fromdateiso8601?) as $req_epoch
           | select($req_epoch != null)
           | (
               [range(($idx + 1); ($ordered | length)) as $after
                | ($ordered[$after]) as $resp
                | select(($resp.body // "") | test("\\bCROSS_REVIEW_(PASS|FAIL)\\b"; "i"))
                | select(($resp.author.login // "") != ($req.author.login // ""))
                | ($resp.createdAt | fromdateiso8601?) as $resp_epoch
                | select($resp_epoch != null)
                | {
                    author: ($resp.author.login // "unknown"),
                    created_at: $resp.createdAt,
                    verdict: (if (($resp.body // "") | test("\\bCROSS_REVIEW_FAIL\\b"; "i")) then "FAIL" else "PASS" end),
                    latency_seconds: ($resp_epoch - $req_epoch)
                  }
               ] | first // null
             ) as $first_resp
           | (if $first_resp == null then ($now - $req_epoch) else $first_resp.latency_seconds end) as $elapsed
           | {
               request_author: ($req.author.login // "unknown"),
               request_created_at: $req.createdAt,
               requested_reviewer: (
                 ((($req.body // "") | match("@[A-Za-z0-9_-]+"; "i") | .string)? // "")
                 | ltrimstr("@")
               ),
               response_author: ($first_resp.author // null),
               response_created_at: ($first_resp.created_at // null),
               verdict: ($first_resp.verdict // null),
               latency_seconds: $elapsed,
               latency_minutes: (($elapsed / 60 * 100 | floor) / 100),
               pending: ($first_resp == null),
               breached: ($elapsed > $threshold)
             }
         ]')"

    pr_check_item="$(jq -cn \
      --argjson pr_num "$pr_number" \
      --argjson checks "$pr_checks_json" \
      '{
        pr_number: $pr_num,
        checks: $checks,
        request_count: ($checks | length),
        pending_count: ($checks | map(select(.pending)) | length),
        violation_count: ($checks | map(select(.breached)) | length)
      }')"

    review_sla_checks_json="$(jq -cn --argjson arr "$review_sla_checks_json" --argjson item "$pr_check_item" '$arr + [$item]')"

    pr_violation_count="$(printf '%s\n' "$pr_check_item" | jq '.violation_count' 2>/dev/null || echo 0)"
    pr_pending_count="$(printf '%s\n' "$pr_check_item" | jq '.pending_count' 2>/dev/null || echo 0)"
    review_sla_violation_count=$((review_sla_violation_count + pr_violation_count))
    review_sla_pending_count=$((review_sla_pending_count + pr_pending_count))

    if [[ "$pr_violation_count" -gt 0 ]]; then
      pr_violations_json="$(printf '%s\n' "$pr_check_item" | jq -c --argjson pr_num "$pr_number" '
        .checks
        | map(select(.breached))
        | map({
            pr_number: $pr_num,
            request_author,
            requested_reviewer,
            request_created_at,
            response_author,
            response_created_at,
            verdict,
            latency_minutes
          })
      ' 2>/dev/null || echo '[]')"
      review_sla_violations_json="$(jq -cn \
        --argjson arr "$review_sla_violations_json" \
        --argjson extra "$pr_violations_json" \
        '$arr + $extra')"
    fi
  done < <(printf '%s\n' "$issue_prs_json" | jq -c '.[]' 2>/dev/null)

  review_sla_status='not_requested'
  if [[ "$(printf '%s\n' "$review_sla_checks_json" | jq 'length' 2>/dev/null || echo 0)" -eq 0 ]]; then
    review_sla_status='no_open_pr'
  else
    request_total="$(printf '%s\n' "$review_sla_checks_json" | jq '[.[].request_count] | add // 0' 2>/dev/null || echo 0)"
    if [[ "$request_total" -eq 0 ]]; then
      review_sla_status='no_request'
    elif [[ "$review_sla_violation_count" -gt 0 ]]; then
      review_sla_status='violated'
    elif [[ "$review_sla_pending_count" -gt 0 ]]; then
      review_sla_status='pending'
    else
      review_sla_status='ok'
    fi
  fi

  status='OK'
  reason='all checks passed'

  if [[ "$assignee_count" -eq 0 ]]; then
    if [[ "$age_seconds" -ge "$threshold_seconds" ]]; then
      status='FAIL'
      reason="no assignee after ${WIP_SLA_HOURS}h"
    else
      status='OK'
      reason='no assignee (within SLA grace period)'
    fi
  elif [[ "$age_seconds" -ge "$threshold_seconds" && "$has_branch" != 'true' ]]; then
    status='WARN'
    reason="no branch after ${WIP_SLA_HOURS}h"
  fi

  if [[ "$review_sla_violation_count" -gt 0 ]]; then
    status='FAIL'
    reason="cross-review SLA violated (${REVIEW_SLA_MINUTES}m)"
  elif [[ "$status" == 'OK' && "$review_sla_status" == 'pending' ]]; then
    reason="cross-review pending (within ${REVIEW_SLA_MINUTES}m SLA)"
  fi

  case "$status" in
    FAIL)
      fail_count=$((fail_count + 1))
      ;;
    WARN)
      warn_count=$((warn_count + 1))
      ;;
    *)
      ok_count=$((ok_count + 1))
      ;;
  esac

  result_item="$(jq -cn \
    --argjson num "$issue_number" \
    --arg title "$issue_title" \
    --arg url "$issue_url" \
    --arg status "$status" \
    --arg reason "$reason" \
    --argjson assignees "$assignees_json" \
    --argjson age_seconds "$age_seconds" \
    --arg has_branch "$has_branch" \
    --arg review_sla_status "$review_sla_status" \
    --argjson review_sla_checks "$review_sla_checks_json" \
    --argjson review_sla_violation_count "$review_sla_violation_count" \
    --argjson review_sla_pending_count "$review_sla_pending_count" \
    --argjson review_sla_threshold "$REVIEW_SLA_MINUTES" \
    '{
      number: $num,
      title: $title,
      url: $url,
      status: $status,
      reason: $reason,
      assignees: $assignees,
      age_seconds: $age_seconds,
      age_hours: (($age_seconds / 3600 * 100 | floor) / 100),
      has_branch: ($has_branch == "true"),
      cross_review_sla: {
        threshold_minutes: $review_sla_threshold,
        status: $review_sla_status,
        violation_count: $review_sla_violation_count,
        pending_count: $review_sla_pending_count,
        checks: $review_sla_checks
      }
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

if [[ "$memory_stale_flag" == "true" && "$overall_status" == "OK" ]]; then
  overall_status='WARN'
  exit_code=1
fi

review_sla_violation_total="$(printf '%s\n' "$review_sla_violations_json" | jq 'length' 2>/dev/null || echo 0)"
notify_result='not_triggered'

if [[ "$review_sla_violation_total" -gt 0 && "$DISCORD_NOTIFY" == 'true' ]]; then
  if command -v "$OPENCLAW_BIN" >/dev/null 2>&1; then
    violation_preview="$(printf '%s\n' "$review_sla_violations_json" | jq -r '
      .[0:3] | map("PR #\(.pr_number) " + ((.latency_minutes | tostring)) + "m") | join(", ")
    ' 2>/dev/null || true)"
    if [[ -z "$violation_preview" ]]; then
      violation_preview='review SLA breached'
    fi
    notify_msg="**[Watchdog]** cross-review SLA>${REVIEW_SLA_MINUTES}m breached: ${violation_preview}. @Lily 请立即跟进并按需重新分配 reviewer。"
    if "$OPENCLAW_BIN" message send --channel discord --account "$DISCORD_ACCOUNT" --target "$DISCORD_DEV_TARGET" -m "$notify_msg" >/dev/null 2>&1; then
      notify_result='sent'
    else
      notify_result='send_failed'
    fi
  else
    notify_result='openclaw_missing'
  fi
fi

if [[ "$memory_stale_flag" == 'true' && "$DISCORD_NOTIFY" == 'true' ]]; then
  if command -v "$OPENCLAW_BIN" >/dev/null 2>&1; then
    stale_msg="**[Watchdog]** ⚠️ MEMORY.md stale: active Sprint ${active_sprint_number}, MEMORY current Sprint ${memory_state_sprint:-unknown}, gap ${memory_gap:-unknown} (> ${MEMORY_STALE_GAP})."
    if "$OPENCLAW_BIN" message send --channel discord --account "$DISCORD_ACCOUNT" --target "$DISCORD_DEV_TARGET" -m "$stale_msg" >/dev/null 2>&1; then
      memory_notify_result='sent'
    else
      memory_notify_result='send_failed'
    fi
  else
    memory_notify_result='openclaw_missing'
  fi
fi

output_json="$(jq -cn \
  --arg repo "$GITHUB_REPO" \
  --arg milestone_title "$milestone_title" \
  --argjson milestone_number "$milestone_number" \
  --arg status "$overall_status" \
  --argjson threshold_hours "$WIP_SLA_HOURS" \
  --argjson now "$now_epoch" \
  --arg memory_file "$MEMORY_FILE" \
  --arg active_sprint_number "$active_sprint_number" \
  --arg memory_state_sprint "$memory_state_sprint" \
  --arg memory_gap "$memory_gap" \
  --arg memory_reason "$memory_reason" \
  --arg memory_stale "$memory_stale_flag" \
  --arg memory_stale_gap "$MEMORY_STALE_GAP" \
  --argjson ok "$ok_count" \
  --argjson warn "$warn_count" \
  --argjson fail "$fail_count" \
  --argjson total "$total_count" \
  --argjson review_sla_threshold "$REVIEW_SLA_MINUTES" \
  --argjson review_sla_violations "$review_sla_violations_json" \
  --arg notify_result "$notify_result" \
  --arg memory_notify_result "$memory_notify_result" \
  --argjson results "$results_json" \
  '{
    repo: $repo,
    milestone: {title: $milestone_title, number: $milestone_number},
    status: $status,
    threshold_hours: $threshold_hours,
    generated_at_epoch: $now,
    cross_review_sla: {
      threshold_minutes: $review_sla_threshold,
      violation_count: ($review_sla_violations | length),
      notify_result: $notify_result,
      violations: $review_sla_violations
    },
    memory_staleness: {
      checked: (($active_sprint_number | length) > 0),
      memory_file: $memory_file,
      active_sprint: (if ($active_sprint_number | length) == 0 then null else ($active_sprint_number | tonumber) end),
      memory_sprint: (if ($memory_state_sprint | length) == 0 then null else ($memory_state_sprint | tonumber) end),
      gap: (if ($memory_gap | length) == 0 then null else ($memory_gap | tonumber) end),
      threshold: ($memory_stale_gap | tonumber),
      stale: ($memory_stale == "true"),
      reason: $memory_reason,
      notify_result: $memory_notify_result
    },
    summary: {ok: $ok, warn: $warn, fail: $fail, total: $total},
    results: $results
  }')"

if [[ "$OUTPUT_JSON" == 'true' ]]; then
  printf '%s\n' "$output_json"
else
  echo "Sprint Watchdog | $GITHUB_REPO | $milestone_title (#$milestone_number)"
  echo "Threshold: WIP=${WIP_SLA_HOURS}h review=${REVIEW_SLA_MINUTES}m"
  if [[ "$total_count" -eq 0 ]]; then
    echo "OK no open issues in milestone"
  else
    printf '%s\n' "$output_json" | jq -r '.results[] | "\(.status) #\(.number) \(.title) [\(.reason)]"'
  fi
  if [[ "$review_sla_violation_total" -gt 0 ]]; then
    printf '%s\n' "$output_json" | jq -r '.cross_review_sla.violations[] | "SLA BREACH PR #\(.pr_number) \(.latency_minutes)m request=\(.request_created_at)"'
  fi
  echo "Summary: OK=$ok_count WARN=$warn_count FAIL=$fail_count TOTAL=$total_count"
fi

exit "$exit_code"
