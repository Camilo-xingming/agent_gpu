#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

API_ROOT = "https://api.github.com"
PROTECTED_BRANCHES = {
    "master",
    "main",
    "develop",
    "dev",
    "release",
    "nightly",
}
ALLOWED_PREFIXES = (
    "issue-",
    "fix/",
    "hotfix/",
    "chore/",
    "cleanup/",
    "refactor/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete stale merged branches from a GitHub repository."
    )
    parser.add_argument(
        "--repo",
        default=os.getenv("GITHUB_REPOSITORY", ""),
        help="GitHub repo in owner/name format (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN", ""),
        help="GitHub token (default: $GH_TOKEN or $GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=14,
        help="Delete branches whose merged PR is older than this many days (default: 14)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max closed PR pages to scan (100 PRs/page, default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidates without deleting branches",
    )
    args = parser.parse_args()

    if not args.repo:
        parser.error("--repo is required (or set GITHUB_REPOSITORY)")
    if not args.token:
        parser.error("--token is required (or set GH_TOKEN/GITHUB_TOKEN)")
    if args.stale_days < 1:
        parser.error("--stale-days must be >= 1")
    if args.max_pages < 1:
        parser.error("--max-pages must be >= 1")
    return args


def github_request(
    token: str,
    method: str,
    path: str,
    *,
    query: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[object], int]:
    url = API_ROOT + path
    if query:
        url += "?" + urllib.parse.urlencode(query)

    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ralphgpu-stale-branch-cleanup",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            if not body:
                return None, status
            return json.loads(body), status
    except urllib.error.HTTPError as err:
        if err.code in (404, 422):
            return None, err.code
        msg = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {err.code} {msg}") from err


def parse_utc(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def is_candidate_branch(branch: str) -> bool:
    if branch in PROTECTED_BRANCHES:
        return False
    return branch.startswith(ALLOWED_PREFIXES)


def collect_stale_branches(
    token: str,
    repo: str,
    cutoff: dt.datetime,
    max_pages: int,
) -> List[str]:
    branches: Set[str] = set()

    for page in range(1, max_pages + 1):
        data, status = github_request(
            token,
            "GET",
            f"/repos/{repo}/pulls",
            query={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": "100",
                "page": str(page),
            },
        )

        if status != 200:
            raise RuntimeError(f"Unexpected status {status} while listing closed PRs")
        prs = data if isinstance(data, list) else []
        if not prs:
            break

        for pr in prs:
            merged_at = pr.get("merged_at")
            if not merged_at:
                continue
            if parse_utc(merged_at) > cutoff:
                continue

            head = pr.get("head") or {}
            ref = head.get("ref")
            head_repo = (head.get("repo") or {}).get("full_name")
            if not ref or head_repo != repo:
                continue
            if not is_candidate_branch(ref):
                continue
            branches.add(ref)

    return sorted(branches)


def has_open_pr(token: str, repo: str, owner: str, branch: str) -> bool:
    data, status = github_request(
        token,
        "GET",
        f"/repos/{repo}/pulls",
        query={"state": "open", "head": f"{owner}:{branch}", "per_page": "1"},
    )
    if status != 200:
        raise RuntimeError(f"Unexpected status {status} while checking open PR for {branch}")
    prs = data if isinstance(data, list) else []
    return len(prs) > 0


def branch_exists(token: str, repo: str, branch: str) -> bool:
    encoded = urllib.parse.quote(branch, safe="")
    _, status = github_request(token, "GET", f"/repos/{repo}/branches/{encoded}")
    return status == 200


def delete_branch(token: str, repo: str, branch: str) -> bool:
    encoded = urllib.parse.quote(branch, safe="")
    _, status = github_request(token, "DELETE", f"/repos/{repo}/git/refs/heads/{encoded}")
    return status in (204, 200)


def main() -> int:
    args = parse_args()

    owner = args.repo.split("/", 1)[0]
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=args.stale_days)

    print(f"repo={args.repo}")
    print(f"stale_days={args.stale_days}, cutoff={cutoff.isoformat()}")
    print(f"dry_run={args.dry_run}")

    candidates = collect_stale_branches(args.token, args.repo, cutoff, args.max_pages)
    print(f"candidate_count={len(candidates)}")

    deleted = 0
    skipped_open_pr = 0
    skipped_missing = 0

    for branch in candidates:
        if has_open_pr(args.token, args.repo, owner, branch):
            skipped_open_pr += 1
            print(f"SKIP(open_pr): {branch}")
            continue

        if not branch_exists(args.token, args.repo, branch):
            skipped_missing += 1
            print(f"SKIP(missing): {branch}")
            continue

        if args.dry_run:
            print(f"DRY_RUN(delete): {branch}")
            continue

        if delete_branch(args.token, args.repo, branch):
            deleted += 1
            print(f"DELETED: {branch}")
        else:
            raise RuntimeError(f"Delete failed for branch {branch}")

    print(
        "summary "
        f"candidates={len(candidates)} "
        f"deleted={deleted} "
        f"skipped_open_pr={skipped_open_pr} "
        f"skipped_missing={skipped_missing}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
