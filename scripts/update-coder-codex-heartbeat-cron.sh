#!/usr/bin/env bash
# Apply the tracked Codex heartbeat prompt to the OpenClaw cron job.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
JOB_ID="${JOB_ID:-5c612c20-ae3f-47e0-a80b-f884c2ec50cd}"
PROMPT_FILE="${PROMPT_FILE:-$SCRIPT_DIR/coder-codex-heartbeat.prompt.txt}"

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

echo "updated coder-codex heartbeat cron: $JOB_ID"
