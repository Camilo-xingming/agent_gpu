#!/usr/bin/env bash
# Install gather scripts to ~/.local/bin for OpenClaw cron jobs.

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TARGET_DIR="${TARGET_DIR:-$HOME/.local/bin}"

mkdir -p "$TARGET_DIR"
install -m 0755 "$SCRIPT_DIR/cron-common.sh" "$TARGET_DIR/cron-common.sh"
install -m 0755 "$SCRIPT_DIR/retro-gather.sh" "$TARGET_DIR/retro-gather.sh"
install -m 0755 "$SCRIPT_DIR/sprint-planning-gather.sh" "$TARGET_DIR/sprint-planning-gather.sh"
install -m 0755 "$SCRIPT_DIR/sprint-watchdog.sh" "$TARGET_DIR/sprint-watchdog.sh"
install -m 0755 "$SCRIPT_DIR/scrum-health.sh" "$TARGET_DIR/scrum-health.sh"

printf 'Installed to %s\n' "$TARGET_DIR"
printf ' - %s\n' "$TARGET_DIR/cron-common.sh"
printf ' - %s\n' "$TARGET_DIR/retro-gather.sh"
printf ' - %s\n' "$TARGET_DIR/sprint-planning-gather.sh"
printf ' - %s\n' "$TARGET_DIR/sprint-watchdog.sh"
printf ' - %s\n' "$TARGET_DIR/scrum-health.sh"
