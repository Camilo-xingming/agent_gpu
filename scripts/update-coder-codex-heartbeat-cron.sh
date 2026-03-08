#!/usr/bin/env bash
# Apply the tracked Codex heartbeat prompt to the OpenClaw cron job.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
JOB_ID="${JOB_ID:-5c612c20-ae3f-47e0-a80b-f884c2ec50cd}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/coder-codex-heartbeat.prompt.txt}"
REQUIRED_GH_PROXY_PREFIX="${REQUIRED_GH_PROXY_PREFIX:-env HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 gh}"
REQUIRED_SSH_IP="${REQUIRED_SSH_IP:-100.81.212.41}"
REQUIRED_HEARTBEAT_KEYWORD="${REQUIRED_HEARTBEAT_KEYWORD:-长任务心跳}"
REQUIRED_HEARTBEAT_INTERVAL_KEYWORD="${REQUIRED_HEARTBEAT_INTERVAL_KEYWORD:-每 60 秒}"
REQUIRED_HEARTBEAT_THRESHOLD_KEYWORD="${REQUIRED_HEARTBEAT_THRESHOLD_KEYWORD:-3 分钟}"
REQUIRED_NO_AUTO_ASSIGN_KEYWORD="${REQUIRED_NO_AUTO_ASSIGN_KEYWORD:-不要自动认领无人认领 issue}"
REQUIRED_STANDBY_KEYWORD="${REQUIRED_STANDBY_KEYWORD:-默认输出 NO_REPLY 并保持 standby}"

require_prompt_policies() {
  local source_name="$1"
  local content="$2"

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_GH_PROXY_PREFIX"; then
    echo "$source_name missing required explicit gh proxy prefix: $REQUIRED_GH_PROXY_PREFIX" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_SSH_IP"; then
    echo "$source_name missing required SSH IP hint: $REQUIRED_SSH_IP" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_HEARTBEAT_KEYWORD"; then
    echo "$source_name missing required heartbeat policy marker: $REQUIRED_HEARTBEAT_KEYWORD" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_HEARTBEAT_INTERVAL_KEYWORD"; then
    echo "$source_name missing required heartbeat interval marker: $REQUIRED_HEARTBEAT_INTERVAL_KEYWORD" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_HEARTBEAT_THRESHOLD_KEYWORD"; then
    echo "$source_name missing required heartbeat threshold marker: $REQUIRED_HEARTBEAT_THRESHOLD_KEYWORD" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_NO_AUTO_ASSIGN_KEYWORD"; then
    echo "$source_name missing required no-auto-assign marker: $REQUIRED_NO_AUTO_ASSIGN_KEYWORD" >&2
    exit 1
  fi

  if ! printf '%s\n' "$content" | grep -Fq "$REQUIRED_STANDBY_KEYWORD"; then
    echo "$source_name missing required standby marker: $REQUIRED_STANDBY_KEYWORD" >&2
    exit 1
  fi
}

if ! command -v openclaw >/dev/null 2>&1; then
  echo "openclaw command not found" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq command not found" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi

message="$(cat "$PROMPT_FILE")"
require_prompt_policies "prompt file" "$message"

openclaw cron edit "$JOB_ID" --message "$message" >/dev/null

stored_message="$(openclaw cron list --all --json | jq -r --arg id "$JOB_ID" '.jobs[] | select(.id == $id) | .payload.message // empty')"
if [[ -z "$stored_message" ]]; then
  echo "cron job not found: $JOB_ID" >&2
  exit 1
fi

if [[ "$stored_message" != "$message" ]]; then
  echo "cron payload mismatch after update" >&2
  exit 1
fi

require_prompt_policies "stored cron payload" "$stored_message"

echo "updated coder-codex heartbeat cron: $JOB_ID (proxy + SSH-IP + heartbeat policy verified)"
