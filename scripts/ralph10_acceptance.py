#!/usr/bin/env python3
"""RALPH-10 acceptance runner (10c-10e skeleton).

Runs a perf target, parses key counters, and compares against a 10a baseline:
- Writebacks
- Fetches
- FU stalls
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PerfMetrics:
    writebacks: int
    fetches: int
    fu_stalls: int
    cycles: Optional[int] = None
    issues: Optional[int] = None
    status: str = "UNKNOWN"


def parse_metrics(text: str) -> PerfMetrics:
    def find_int(pattern: str, default: int = 0) -> int:
        m = re.search(pattern, text, re.MULTILINE)
        return int(m.group(1)) if m else default

    wb = find_int(r"^Writebacks:\s*(\d+)")
    fetches = find_int(r"^Fetches:\s*(\d+)")
    cycles = find_int(r"^Cycles:\s*(\d+)", default=-1)
    issues = find_int(r"^Issues:\s*(\d+)", default=-1)

    m_stall = re.search(r"^Stalls:\s*.*\bfu=(\d+)\b", text, re.MULTILINE)
    fu_stalls = int(m_stall.group(1)) if m_stall else 0

    status = "UNKNOWN"
    if re.search(r"\bPASS:", text):
        status = "PASS"
    elif re.search(r"\bFAIL:", text):
        status = "FAIL"

    return PerfMetrics(
        writebacks=wb,
        fetches=fetches,
        fu_stalls=fu_stalls,
        cycles=None if cycles < 0 else cycles,
        issues=None if issues < 0 else issues,
        status=status,
    )


def run_make_target(repo: Path, target: str) -> str:
    cmd = ["make", target]
    proc = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"make {target} failed with exit code {proc.returncode}")
    return output


def load_baseline(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    return {
        "Writebacks": int(metrics.get("Writebacks", 0)),
        "Fetches": int(metrics.get("Fetches", 0)),
        "FuStalls": int(metrics.get("FuStalls", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RALPH-10 acceptance checker")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target", default="test_sm_v2_perf_gemm16_ptx")
    parser.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parent / "ralph10_baseline_10a.json")
    parser.add_argument("--log", type=Path, help="Parse existing log instead of running make")
    parser.add_argument("--max-fetch-ratio", type=float, default=1.30)
    parser.add_argument("--min-wb-ratio", type=float, default=0.95)
    parser.add_argument("--max-fu-stalls-add", type=int, default=128)
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)

    if args.log:
        text = args.log.read_text()
    else:
        text = run_make_target(args.repo, args.target)

    cur = parse_metrics(text)

    wb_ratio = (cur.writebacks / baseline["Writebacks"]) if baseline["Writebacks"] else 0.0
    fetch_ratio = (cur.fetches / baseline["Fetches"]) if baseline["Fetches"] else 0.0
    fu_stalls_delta = cur.fu_stalls - baseline["FuStalls"]

    checks = {
        "status_pass": (cur.status == "PASS"),
        "wb_ratio": (wb_ratio >= args.min_wb_ratio),
        "fetch_ratio": (fetch_ratio <= args.max_fetch_ratio),
        "fu_stalls_delta": (fu_stalls_delta <= args.max_fu_stalls_add),
    }

    print("=== RALPH-10 Acceptance (10c-10e skeleton) ===")
    print(f"Target: {args.target}")
    print(
        "Baseline: "
        f"WB={baseline['Writebacks']} Fetches={baseline['Fetches']} FU_stalls={baseline['FuStalls']}"
    )
    print(
        "Current:  "
        f"WB={cur.writebacks} Fetches={cur.fetches} FU_stalls={cur.fu_stalls} "
        f"Status={cur.status}"
    )
    if cur.cycles is not None:
        print(f"Cycles:   {cur.cycles}")
    if cur.issues is not None:
        print(f"Issues:   {cur.issues}")
    print(
        "Ratios:   "
        f"WB={wb_ratio:.3f} (>= {args.min_wb_ratio:.3f}), "
        f"Fetch={fetch_ratio:.3f} (<= {args.max_fetch_ratio:.3f}), "
        f"FU_delta={fu_stalls_delta} (<= {args.max_fu_stalls_add})"
    )

    ok = all(checks.values())
    print("Result:   PASS" if ok else "Result:   FAIL")

    if not ok:
        for key, passed in checks.items():
            if not passed:
                print(f" - failed check: {key}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
