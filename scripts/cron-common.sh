#!/usr/bin/env bash
# Shared helpers for cron gather scripts.

set -uo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

configure_gh_proxy_env() {
  if [[ "${GH_PROXY_AUTO_CONFIG:-1}" == "0" ]]; then
    return
  fi

  local https_proxy_val="${GH_HTTPS_PROXY:-http://127.0.0.1:7897}"
  local http_proxy_val="${GH_HTTP_PROXY:-$https_proxy_val}"

  export HTTPS_PROXY="$https_proxy_val"
  export https_proxy="$https_proxy_val"
  export HTTP_PROXY="$http_proxy_val"
  export http_proxy="$http_proxy_val"
}

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

resolve_command_path() {
  local override="${1:-}"
  shift || true

  if [[ -n "$override" ]]; then
    if has_command "$override"; then
      if [[ "$override" == */* ]]; then
        printf '%s\n' "$override"
      else
        command -v "$override"
      fi
      return 0
    fi
    printf '%s\n' "$override"
    return 1
  fi

  local candidate
  for candidate in "$@"; do
    if has_command "$candidate"; then
      if [[ "$candidate" == */* ]]; then
        printf '%s\n' "$candidate"
      else
        command -v "$candidate"
      fi
      return 0
    fi
  done

  return 1
}

build_retro_refs_json_from_content() {
  local retro_content="${1:-}"
  local retro_refs_tsv=''

  if [[ -n "$retro_content" ]]; then
    retro_refs_tsv="$(printf '%s\n' "$retro_content" | awk '
      BEGIN { sprint = "" }
      /^#/ {
        if ($0 ~ /Sprint[[:space:]]+[0-9]+/) {
          sprint = $0
          sub(/.*Sprint[[:space:]]+/, "", sprint)
          sub(/[^0-9].*$/, "", sprint)
        }
      }
      /^- \[[[:space:]]\]/ {
        if (sprint == "") {
          next
        }
        line = $0
        sub(/^- \[[ xX]\][[:space:]]*/, "", line)
        rest = line
        while (match(rest, /#[0-9]+/)) {
          issue = substr(rest, RSTART + 1, RLENGTH - 1)
          printf "%s\t%s\t%s\n", sprint, issue, line
          rest = substr(rest, RSTART + RLENGTH)
        }
      }
    ' || true)"
  fi

  if [[ -z "$retro_refs_tsv" ]]; then
    printf '[]\n'
    return 0
  fi

  printf '%s\n' "$retro_refs_tsv" | jq -Rsc '
    split("\n")
    | map(select(length > 0))
    | map(split("\t"))
    | map(select(length >= 2))
    | map({sprint: (.[0] | tonumber), issue: (.[1] | tonumber), text: (.[2] // "")})
  ' 2>/dev/null || printf '[]\n'
}

build_retro_refs_json_from_file() {
  local retro_file_path="${1:-}"
  if [[ -z "$retro_file_path" || ! -f "$retro_file_path" ]]; then
    printf '[]\n'
    return 0
  fi
  build_retro_refs_json_from_content "$(cat "$retro_file_path")"
}

build_process_recurrence_json() {
  local retro_refs_json="${1:-[]}"
  local open_process_issues_json="${2:-[]}"
  local repeat_min="${3:-2}"
  local escalate_min="${4:-3}"

  jq -cn \
    --argjson refs "$retro_refs_json" \
    --argjson open "$open_process_issues_json" \
    --argjson repeat_min "$repeat_min" \
    --argjson escalate_min "$escalate_min" \
    '
    def max_consecutive($arr):
      reduce $arr[] as $s ({prev: null, cur: 0, max: 0};
        if .prev == null then
          {prev: $s, cur: 1, max: 1}
        elif $s == (.prev + 1) then
          {prev: $s, cur: (.cur + 1), max: (if (.cur + 1) > .max then (.cur + 1) else .max end)}
        else
          {prev: $s, cur: 1, max: (if .max > 1 then .max else 1 end)}
        end
      ) | .max;
    [ $open[] as $issue
      | ($refs | map(select(.issue == $issue.number)) | map(.sprint) | unique | sort) as $sprints
      | ($sprints | length) as $occ
      | (if $occ == 0 then 0 else max_consecutive($sprints) end) as $max_run
      | {
          number: $issue.number,
          title: ($issue.title // ""),
          url: ($issue.url // ""),
          sprints: $sprints,
          occurrences: $occ,
          max_consecutive: $max_run,
          repeated: ($max_run >= $repeat_min),
          escalate: ($max_run >= $escalate_min)
        }
    ] as $rows
    | {
        checked: true,
        threshold_consecutive: $repeat_min,
        escalate_after_consecutive: $escalate_min,
        matches: ($rows | map(select(.repeated and .occurrences > 0))),
        escalations: ($rows | map(select(.escalate and .occurrences > 0)))
      }
  ' 2>/dev/null || printf '{"checked":false,"threshold_consecutive":%s,"escalate_after_consecutive":%s,"matches":[],"escalations":[]}\n' "$repeat_min" "$escalate_min"
}

append_json_message() {
  local var_name="$1"
  local message="$2"
  local current="${!var_name:-[]}"
  printf -v "$var_name" '%s' "$(jq -cn --argjson arr "$current" --arg m "$message" '$arr + [$m]')"
}

append_error() {
  append_json_message errors_json "$1"
}

append_warning() {
  append_json_message warnings_json "$1"
}

add_health_check() {
  local name="$1"
  local status="$2"
  local detail="$3"
  local mandatory="${4:-true}"

  health_checks_json="${health_checks_json:-[]}"
  health_checks_json="$(jq -cn \
    --argjson arr "$health_checks_json" \
    --arg name "$name" \
    --arg status "$status" \
    --arg detail "$detail" \
    --arg mandatory "$mandatory" \
    '$arr + [{name: $name, status: $status, detail: $detail, mandatory: ($mandatory == "true")}]')"
}

build_health_summary() {
  local checks="${health_checks_json:-[]}"
  jq -cn --argjson checks "$checks" '
    ($checks | map(select(.mandatory == true))) as $mandatory
    | ($mandatory | length) as $total
    | ($mandatory | map(select(.status == "pass")) | length) as $pass
    | {
        checks: $checks,
        mandatory_total: $total,
        mandatory_pass: $pass,
        coverage_pct: (if $total == 0 then 100 else (($pass * 100 / $total) | floor) end)
      }
  '
}

log_event() {
  local component="$1"
  local level="$2"
  local message="$3"
  local line

  line="$(date '+%Y-%m-%d %H:%M:%S') | ${component} | ${level} | ${message}"

  if [[ -n "${CRON_LOG:-}" ]]; then
    mkdir -p "$(dirname -- "$CRON_LOG")"
    printf '%s\n' "$line" >> "$CRON_LOG"
  fi

  printf '%s\n' "$line"
}

atomic_write_file() {
  local target="$1"
  local content="$2"
  local target_dir
  local tmp_file

  target_dir="$(dirname -- "$target")"
  mkdir -p "$target_dir"

  tmp_file="$(mktemp "$target_dir/.tmp.$(basename -- "$target").XXXXXX")" || return 1

  if ! printf '%s\n' "$content" > "$tmp_file"; then
    rm -f "$tmp_file"
    return 1
  fi

  if ! mv "$tmp_file" "$target"; then
    rm -f "$tmp_file"
    return 1
  fi

  return 0
}

write_status_file() {
  local status_file="$1"
  local component="$2"
  local state="$3"
  local detail="${4:-}"
  local payload

  payload="$(jq -cn \
    --arg component "$component" \
    --arg state "$state" \
    --arg detail "$detail" \
    --arg host "$(hostname)" \
    --arg pid "$$" \
    --arg updated_at "$(date '+%Y-%m-%dT%H:%M:%S')" \
    '{
      component: $component,
      state: $state,
      detail: $detail,
      host: $host,
      pid: ($pid | tonumber),
      updated_at: $updated_at
    }')"

  atomic_write_file "$status_file" "$payload"
}

acquire_script_lock() {
  local shared_dir="$1"
  local name="$2"
  local lock_root="$shared_dir/locks"
  local lock_path="$lock_root/${name}.lock"

  mkdir -p "$lock_root"

  if mkdir "$lock_path" 2>/dev/null; then
    printf '%s\n' "$$" > "$lock_path/pid"
    printf '%s\n' "$lock_path"
    return 0
  fi

  local lock_pid=''
  if [[ -f "$lock_path/pid" ]]; then
    lock_pid="$(cat "$lock_path/pid" 2>/dev/null || true)"
  fi

  if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
    rm -rf "$lock_path"
    if mkdir "$lock_path" 2>/dev/null; then
      printf '%s\n' "$$" > "$lock_path/pid"
      printf '%s\n' "$lock_path"
      return 0
    fi
  fi

  return 1
}

release_script_lock() {
  local lock_path="$1"
  if [[ -n "$lock_path" ]]; then
    rm -rf "$lock_path"
  fi
}

send_discord_message_best_effort() {
  local text="$1"

  # Prevent raw tool output and system messages from leaking
  if echo "$text" | grep -qE 'startcall:|sessionId:|Stats:|Tool:|\[System Message\]'; then
    log_event "discord_msg" "ERROR" "Message intercepted: raw output leaked. Filtering..."
    text=$(echo "$text" | sed -E 's/Stats:.*//g')
    text=$(echo "$text" | sed -E 's/Tool:.*//g')
    text=$(echo "$text" | sed -E 's/\[System Message\] sessionId: [a-zA-Z0-9-]+//g')
    # Filter lines containing startcall: or sessionId:
    text=$(echo "$text" | grep -vE 'startcall:|sessionId:|Stats:|Tool:' || true)
    # If empty after filtering, exit without sending
    if [[ -z "$(echo "$text" | tr -d ' \n')" ]]; then
      return 0
    fi
  fi

  if [[ -z "${OPENCLAW_BIN:-}" ]] || ! has_command "$OPENCLAW_BIN"; then
    return 0
  fi

  if [[ -z "${DISCORD_DEV_TARGET:-}" ]]; then
    return 0
  fi

  local account="${DISCORD_ACCOUNT:-lily}"

  "$OPENCLAW_BIN" message send --account "$account" --channel discord --target "$DISCORD_DEV_TARGET" --body "$text" >/dev/null 2>&1 \
    || "$OPENCLAW_BIN" message send --account "$account" --channel discord --target "$DISCORD_DEV_TARGET" --text "$text" >/dev/null 2>&1 \
    || "$OPENCLAW_BIN" message create --account "$account" --channel discord --target "$DISCORD_DEV_TARGET" --body "$text" >/dev/null 2>&1 \
    || true
}

HEARTBEAT_PID=''

start_heartbeat() {
  local component="$1"
  local context="$2"
  local interval="${HEARTBEAT_INTERVAL_SEC:-60}"
  local threshold="${HEARTBEAT_START_AFTER_SEC:-180}"
  local start_epoch

  start_epoch="$(date +%s)"

  (
    while true; do
      sleep "$interval" || exit 0

      local now
      local elapsed
      now="$(date +%s)"
      elapsed=$((now - start_epoch))

      if (( elapsed < threshold )); then
        continue
      fi

      local msg="long-task heartbeat elapsed=${elapsed}s ${context}"
      log_event "$component" "HEARTBEAT" "$msg"
      send_discord_message_best_effort "${component}: ${msg}"
    done
  ) &

  HEARTBEAT_PID=$!
}

stop_heartbeat() {
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
    wait "$HEARTBEAT_PID" 2>/dev/null || true
    HEARTBEAT_PID=''
  fi
}

capture_with_retry() {
  local out_var="$1"
  local retries="$2"
  local delay_sec="$3"
  shift 3

  local attempt=1
  local output=''

  while true; do
    if output="$($@ 2>/dev/null)"; then
      printf -v "$out_var" '%s' "$output"
      return 0
    fi

    if (( attempt >= retries )); then
      printf -v "$out_var" '%s' ''
      return 1
    fi

    attempt=$((attempt + 1))
    sleep "$delay_sec"
  done
}

