#!/usr/bin/env bash
set -uo pipefail

list_file="${1:-tests/regression_list.txt}"

if [[ ! -f "$list_file" ]]; then
  echo "Regression list not found: $list_file" >&2
  exit 1
fi

total=0
fail=0
start_ts=$(date +%s)

while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"
  line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  if [[ -z "$line" ]]; then
    continue
  fi

  total=$((total + 1))
  echo "==> [$total] $line"
  if bash -lc "$line"; then
    echo "PASS: $line"
  else
    echo "FAIL: $line"
    fail=$((fail + 1))
  fi
done < "$list_file"

elapsed=$(( $(date +%s) - start_ts ))
echo "Regression complete: ${total} run, ${fail} failed, ${elapsed}s elapsed"

if [[ $fail -ne 0 ]]; then
  exit 1
fi
