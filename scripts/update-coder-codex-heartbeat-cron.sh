#!/usr/bin/env bash
# Apply the tracked Codex heartbeat prompt to the OpenClaw cron job.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
JOB_ID="${JOB_ID:-5c612c20-ae3f-47e0-a80b-f884c2ec50cd}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/coder-codex-heartbeat.prompt.txt}"
REQUIRED_GH_PROXY_PREFIX="${REQUIRED_GH_PROXY_PREFIX:-env HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 gh}"
REQUIRED_SSH_IP="${REQUIRED_SSH_IP:-100.81.212.41}"

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

echo "updated coder-codex heartbeat cron: $JOB_ID (proxy + SSH-IP policy verified)"
