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
