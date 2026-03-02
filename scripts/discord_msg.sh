#!/bin/bash
# Wrapper to filter raw output leaks from agents before sending to Discord

MESSAGE=""
CHANNEL="discord"
ACCOUNT="gemini"
TARGET=""
LIMIT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--message)
      MESSAGE="$2"
      shift 2
      ;;
    --channel)
      CHANNEL="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    send)
      MODE="send"
      shift
      ;;
    read)
      MODE="read"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$MODE" == "read" ]]; then
  openclaw message read --channel "$CHANNEL" --account "$ACCOUNT" --target "$TARGET" --limit "$LIMIT"
  exit $?
fi

if echo "$MESSAGE" | grep -qE 'startcall:|sessionId:|Stats: |\[System Message\]'; then
  echo "ERROR: Message intercepted. Raw tool logs or system metrics detected."
  echo "Please rewrite your message to be human-readable and free of tool traces."
  exit 1
fi

# Clean up any trailing stats if somehow missed
CLEAN_MESSAGE=$(echo "$MESSAGE" | sed -E 's/Stats: runtime.*//g')

openclaw message send --channel "$CHANNEL" --account "$ACCOUNT" --target "$TARGET" -m "$CLEAN_MESSAGE"
