#!/usr/bin/env bash
# check-manifest.sh — CI check: every file in a MANIFEST-tracked directory must be listed in MANIFEST.md
#
# Finds all directories containing MANIFEST.md, extracts listed filenames from
# markdown tables, and reports any files that exist on disk but are not listed.
#
# Exit code: 0 = all files covered, 1 = unlisted files found

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Check if a filename should be ignored
should_ignore() {
    local file="$1"
    case "$file" in
        .DS_Store|.gitkeep|.gitignore|MANIFEST.md) return 0 ;;
        *.pyc) return 0 ;;
        __pycache__) return 0 ;;
    esac
    return 1
}

# Extract filenames from the first column of markdown tables in a MANIFEST.md.
# Skips: header rows, separator rows, glob patterns, backtick-wrapped entries, directory entries.
# Output: one filename per line (empty if no real filenames found).
extract_manifest_files() {
    local manifest="$1"

    while IFS= read -r line; do
        # Only process table rows (lines containing |)
        case "$line" in
            *\|*) ;;
            *) continue ;;
        esac

        # Skip separator rows: |---|---|...|
        case "$line" in
            *\|*---*\|*) continue ;;
        esac

        # Extract first column content
        local col1
        col1=$(printf '%s' "$line" | sed 's/^[[:space:]]*|[[:space:]]*//' | sed 's/[[:space:]]*|.*//')

        # Skip known header keywords (Chinese and English)
        case "$col1" in
            文件|File|file|目录|前缀模式|Directory) continue ;;
        esac

        # Skip empty cells
        [ -z "$col1" ] && continue

        # Skip backtick-wrapped entries (pattern descriptions like `tb_{module}.v`)
        case "$col1" in
            *\`*) continue ;;
        esac

        # Skip glob patterns (contain { or *)
        case "$col1" in
            *\{*|*\**) continue ;;
        esac

        # Skip directory entries (end with /)
        case "$col1" in
            */) continue ;;
        esac

        printf '%s\n' "$col1"
    done < "$manifest"
}

found_issues=0
checked_dirs=0
skipped_dirs=0

echo "=== MANIFEST Coverage Check ==="
echo ""

# Collect MANIFEST paths into a temp file to avoid subshell variable scoping issues
manifest_paths=$(mktemp)
find "$REPO_ROOT" -name "MANIFEST.md" -type f | sort > "$manifest_paths"

while IFS= read -r manifest_path; do
    dir="$(dirname "$manifest_path")"
    rel_dir="${dir#"$REPO_ROOT"}"
    rel_dir="${rel_dir:-/}"
    if [ "$rel_dir" = "/" ]; then
        display_dir="(root)"
    else
        display_dir="$rel_dir"
    fi

    # Extract listed filenames from MANIFEST.md
    manifest_list=$(mktemp)
    extract_manifest_files "$manifest_path" > "$manifest_list"

    # If no filenames extracted, skip (index-only MANIFEST without per-file listing)
    if [ ! -s "$manifest_list" ]; then
        echo "SKIP $display_dir — MANIFEST.md has no per-file listing table"
        rm -f "$manifest_list"
        skipped_dirs=$((skipped_dirs + 1))
        continue
    fi

    checked_dirs=$((checked_dirs + 1))

    # Get actual files in the directory (files only, not subdirectories, not hidden)
    actual_list=$(mktemp)
    find "$dir" -maxdepth 1 -type f -not -name '.*' -exec basename {} \; | sort > "$actual_list"

    # Find files on disk but not in MANIFEST
    issue_count=0
    issue_files=""
    total_files=0
    while IFS= read -r actual; do
        total_files=$((total_files + 1))

        if should_ignore "$actual"; then
            continue
        fi

        # Check if file is listed in MANIFEST (exact match)
        if ! grep -qxF "$actual" "$manifest_list"; then
            issue_count=$((issue_count + 1))
            issue_files="${issue_files}  - ${actual}
"
        fi
    done < "$actual_list"

    if [ "$issue_count" -gt 0 ]; then
        echo "FAIL $display_dir — ${issue_count} file(s) not in MANIFEST.md:"
        printf '%s' "$issue_files"
        found_issues=$((found_issues + issue_count))
    else
        echo "  OK $display_dir — all ${total_files} files covered"
    fi

    rm -f "$manifest_list" "$actual_list"

done < "$manifest_paths"

rm -f "$manifest_paths"

echo ""
echo "--- Summary ---"
echo "Checked: $checked_dirs directories"
echo "Skipped: $skipped_dirs directories (no per-file listing)"
echo "Unlisted files: $found_issues"

if [ "$found_issues" -gt 0 ]; then
    echo ""
    echo "FAILED: $found_issues file(s) missing from MANIFEST.md."
    echo "Add them to the appropriate MANIFEST.md or add to IGNORE_PATTERNS if they should be excluded."
    exit 1
else
    echo ""
    echo "PASSED: All files are covered by MANIFEST.md."
    exit 0
fi
