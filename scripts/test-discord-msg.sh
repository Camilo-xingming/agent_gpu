#!/bin/bash

# Test script for discord_msg.sh and cron-common.sh output filtering

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== Testing discord_msg.sh ==="

# Mock openclaw
mkdir -p "$DIR/mock_bin"
cat << 'EOF' > "$DIR/mock_bin/openclaw"
#!/bin/bash
if [[ "$1" == "message" && "$2" == "send" ]]; then
  shift 2
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-m" || "$1" == "--text" || "$1" == "--body" ]]; then
      echo "$2" > "$(dirname "$0")/last_message.txt"
      break
    fi
    shift
  done
fi
EOF
chmod +x "$DIR/mock_bin/openclaw"

export PATH="$DIR/mock_bin:$PATH"

# Test 1: discord_msg.sh should fail if message contains Tool:
echo "Test 1: Reject 'Tool: read_file'"
if bash "$DIR/discord_msg.sh" send --channel discord --account gemini --target test -m "Here is some text. Tool: read_file used."; then
  echo "FAIL: Expected discord_msg.sh to reject message with Tool:"
  exit 1
fi
echo "PASS: Rejected message with Tool:"

# Test 2: discord_msg.sh should fail if message contains Stats:
echo "Test 2: Reject 'Stats: runtime...'"
if bash "$DIR/discord_msg.sh" send --channel discord --account gemini --target test -m "Done. Stats: runtime 10s"; then
  echo "FAIL: Expected discord_msg.sh to reject message with Stats:"
  exit 1
fi
echo "PASS: Rejected message with Stats:"

# Test 3: discord_msg.sh should pass normal message
echo "Test 3: Accept normal message"
if ! bash "$DIR/discord_msg.sh" send --channel discord --account gemini --target test -m "Issue 318 complete."; then
  echo "FAIL: Expected discord_msg.sh to accept normal message"
  exit 1
fi
echo "PASS: Accepted normal message"

echo "=== Testing cron-common.sh send_discord_message_best_effort ==="

source "$DIR/cron-common.sh"
export OPENCLAW_BIN="openclaw"
export DISCORD_DEV_TARGET="test"

rm -f "$DIR/mock_bin/last_message.txt"

# Test 4: cron-common.sh should filter Tool: and Stats:
echo "Test 4: Filter out Tool: and Stats: lines"
MESSAGE="Hello team.
startcall: read_file
Tool: read_file
Some info here.
Stats: runtime 20s"

send_discord_message_best_effort "$MESSAGE"

if [[ -f "$DIR/mock_bin/last_message.txt" ]]; then
  LAST_MSG=$(cat "$DIR/mock_bin/last_message.txt")
  if echo "$LAST_MSG" | grep -qE "startcall:|Tool:|Stats:"; then
    echo "FAIL: Message was not filtered properly. Output:"
    echo "$LAST_MSG"
    exit 1
  fi
  echo "PASS: Message filtered correctly"
else
  echo "FAIL: No message was sent"
  exit 1
fi

rm -rf "$DIR/mock_bin"

echo "All tests passed!"
