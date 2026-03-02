#!/usr/bin/env python3
"""
RalphGPU Performance Dashboard
===============================
Unified dashboard for IPC / stall / utilization metrics.
Runs perf benchmarks, parses simulation output, exports CSV/JSON,
generates Markdown report, and detects regressions against baseline.

Usage:
    # Run all benchmarks and generate dashboard
    python tools/perf_dashboard.py --run

    # Parse existing log files
    python tools/perf_dashboard.py --parse build/bench_atomics.log --parse build/mlp_perf.log

    # Compare against baseline
    python tools/perf_dashboard.py --run --baseline perf_baselines/latest.json

    # Export formats
    python tools/perf_dashboard.py --run --csv perf_results.csv --json perf_results.json
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build"


# ============================================================================
# Workload Definitions
# ============================================================================

WORKLOADS = {
    "alu_heavy": {
        "name": "ALU-Heavy (FP32 FMA GEMM)",
        "category": "ALU",
        "make_target": "test_sm_v2_perf",
        "description": "FP32 FMA stream -- measures ALU throughput and scheduler efficiency",
    },
    "tensor": {
        "name": "Tensor Core (WMMA MMA)",
        "category": "Tensor",
        "make_target": "test_sm_v2_perf_tensor",
        "description": "WMMA m16n16k16 FP16 -- measures tensor core throughput",
    },
    "tensor_multiwarp": {
        "name": "Tensor Multi-Warp",
        "category": "Tensor",
        "make_target": "test_sm_v2_perf_tensor_multiwarp",
        "description": "Multi-warp tensor core -- measures backpressure handling",
    },
    "memory_heavy": {
        "name": "Memory-Heavy (Atomics)",
        "category": "Memory",
        "make_target": "bench_atomics",
        "description": "Atomic operations -- measures memory subsystem and contention",
    },
    "mixed_divergence": {
        "name": "Mixed (Branch Divergence)",
        "category": "Mixed",
        "make_target": "bench_divergence",
        "description": "Divergent branches -- measures control flow and reconvergence",
    },
}


def resolve_workload_ids(selected: Optional[List[str]]) -> List[str]:
    """Return workload ids in a stable order, optionally filtered by user input."""
    if not selected:
        return list(WORKLOADS.keys())

    seen = set()
    ordered = []
    for wid in selected:
        if wid not in WORKLOADS:
            raise ValueError(f"Unknown workload: {wid}")
        if wid in seen:
            continue
        ordered.append(wid)
        seen.add(wid)
    return ordered


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class PerfMetrics:
    """Performance metrics for a single workload run."""
    workload: str = ""
    category: str = ""
    status: str = "UNKNOWN"  # PASS / FAIL / TIMEOUT

    # Core
    cycles: int = 0
    instructions: int = 0
    ipc: float = 0.0
    occupancy_pct: float = 0.0

    # Stall breakdown (cycle counts)
    stall_raw: int = 0
    stall_fu: int = 0
    stall_mem: int = 0
    stall_sync: int = 0
    stall_ifetch: int = 0
    stall_atomic: int = 0
    stall_tensor: int = 0
    stall_wbq: int = 0

    # FU utilization (cycle counts)
    fu_alu: int = 0
    fu_fpu: int = 0
    fu_sfu: int = 0
    fu_ldst: int = 0
    fu_tensor: int = 0

    # Memory
    l1_hits: int = 0
    l1_misses: int = 0
    fetches: int = 0
    writebacks: int = 0

    # Extra (workload-specific)
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def total_stalls(self) -> int:
        return (self.stall_raw + self.stall_fu + self.stall_mem +
                self.stall_sync + self.stall_ifetch + self.stall_atomic +
                self.stall_tensor + self.stall_wbq)

    @property
    def stall_pct(self) -> float:
        return (self.total_stalls / self.cycles * 100) if self.cycles else 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return (self.l1_hits / total * 100) if total else 0.0

    def stall_breakdown(self) -> Dict[str, float]:
        """Return stall breakdown as percentages of total cycles."""
        if not self.cycles:
            return {}
        result = {}
        for name in ["raw", "fu", "mem", "sync", "ifetch", "atomic", "tensor", "wbq"]:
            val = getattr(self, f"stall_{name}")
            if val > 0:
                result[name] = val / self.cycles * 100
        return result

    def fu_utilization(self) -> Dict[str, float]:
        """Return FU utilization as percentages of total cycles."""
        if not self.cycles:
            return {}
        result = {}
        for name in ["alu", "fpu", "sfu", "ldst", "tensor"]:
            val = getattr(self, f"fu_{name}")
            if val > 0:
                result[name] = val / self.cycles * 100
        return result


# ============================================================================
# Parsers
# ============================================================================

def parse_sim_output(content: str, workload: str = "", category: str = "") -> PerfMetrics:
    """Parse simulation output from various testbench formats."""
    m = PerfMetrics(workload=workload, category=category)

    # Cycles
    if match := re.search(r'Cycles:\s*(\d+)', content):
        m.cycles = int(match.group(1))

    # Instructions / Issues
    if match := re.search(r'(?:Instructions|Issues):\s*(\d+)', content):
        m.instructions = int(match.group(1))

    # IPC (explicit)
    if match := re.search(r'IPC:\s*([\d.]+)', content):
        m.ipc = float(match.group(1))
    elif m.cycles and m.instructions:
        m.ipc = m.instructions / m.cycles

    # Occupancy percentage
    if match := re.search(r'Occupancy:\s*([\d.]+)%', content):
        m.occupancy_pct = float(match.group(1))

    # Stalls: raw=N fu=N mem=N atomic=N tensor=N wbq=N
    if match := re.search(r'Stalls?:\s*(.*)', content):
        stall_str = match.group(1)
        for k, attr in [("raw", "stall_raw"), ("fu", "stall_fu"), ("mem", "stall_mem"),
                        ("sync", "stall_sync"), ("ifetch", "stall_ifetch"),
                        ("atomic", "stall_atomic"), ("tensor", "stall_tensor"),
                        ("wbq", "stall_wbq")]:
            if sm := re.search(rf'{k}=(\d+)', stall_str):
                setattr(m, attr, int(sm.group(1)))

    # Stall: Scoreboard / I-Fetch / Memory (perf_report.py format)
    for label, attr in [("Scoreboard", "stall_raw"), ("I-Fetch", "stall_ifetch"),
                        ("Memory", "stall_mem")]:
        if match := re.search(rf'Stall:\s*{label}\s*=\s*(\d+)', content):
            setattr(m, attr, int(match.group(1)))

    # FU: ALU Active / FPU Active / LDST Active (perf_report.py format)
    for label, attr in [("ALU", "fu_alu"), ("FPU", "fu_fpu"), ("SFU", "fu_sfu"),
                        ("LDST", "fu_ldst"), ("Tensor", "fu_tensor")]:
        if match := re.search(rf'FU:\s*{label}\s*Active\s*=\s*(\d+)', content):
            setattr(m, attr, int(match.group(1)))

    # Fetches / Writebacks
    if match := re.search(r'Fetches:\s*(\d+)', content):
        m.fetches = int(match.group(1))
    if match := re.search(r'Writebacks?:\s*(\d+)', content):
        m.writebacks = int(match.group(1))

    # PASS/FAIL/TIMEOUT
    if re.search(r'PASS', content):
        m.status = "PASS"
    elif re.search(r'TIMEOUT|timeout', content):
        m.status = "TIMEOUT"
    elif re.search(r'FAIL', content):
        m.status = "FAIL"

    # bench_atomics / bench_divergence format
    if match := re.search(r'Ops/Cycle:\s*([\d.]+)', content):
        m.extra["ops_per_cycle"] = match.group(1)
    if match := re.search(r'Ops:\s*(\d+)', content):
        m.extra["ops"] = match.group(1)

    # WMMA-specific
    for label in ["WMMA issues", "SMEM traffic", "Tensor path"]:
        if match := re.search(rf'{label}:\s*(.*)', content):
            m.extra[label] = match.group(1).strip()

    return m


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(target: str) -> str:
    """Run a make target and return combined stdout+stderr."""
    try:
        result = subprocess.run(
            ["make", target],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=300
        )
        return result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT: benchmark exceeded 300s limit"
    except Exception as e:
        return f"ERROR: {e}"


def run_all_benchmarks(selected_workloads: Optional[List[str]] = None) -> List[PerfMetrics]:
    """Run workloads and collect metrics."""
    results = []
    workload_ids = resolve_workload_ids(selected_workloads)
    for wid in workload_ids:
        wdef = WORKLOADS[wid]
        print(f"Running {wdef['name']}...", file=sys.stderr)
        output = run_benchmark(wdef["make_target"])

        # Save raw log
        log_path = BUILD_DIR / f"dashboard_{wid}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output)

        metrics = parse_sim_output(output, workload=wid, category=wdef["category"])
        results.append(metrics)
        print(f"  -> {metrics.status}: {metrics.cycles} cycles, IPC={metrics.ipc:.3f}",
              file=sys.stderr)

    return results


# ============================================================================
# Export
# ============================================================================

def export_csv(results: List[PerfMetrics], path: Path):
    """Export metrics to CSV."""
    fields = ["workload", "category", "status", "cycles", "instructions", "ipc", "occupancy_pct",
              "stall_raw", "stall_fu", "stall_mem", "stall_sync", "stall_ifetch",
              "stall_atomic", "stall_tensor", "stall_wbq",
              "fu_alu", "fu_fpu", "fu_sfu", "fu_ldst", "fu_tensor",
              "l1_hits", "l1_misses", "fetches", "writebacks"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            writer.writerow(row)
    print(f"CSV exported: {path}", file=sys.stderr)


def export_json(results: List[PerfMetrics], path: Path):
    """Export metrics to JSON with metadata."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "workloads": {}
    }
    for r in results:
        d = asdict(r)
        d["stall_pct"] = r.stall_pct
        d["l1_hit_rate"] = r.l1_hit_rate
        d["stall_breakdown"] = r.stall_breakdown()
        d["fu_utilization"] = r.fu_utilization()
        data["workloads"][r.workload] = d
    path.write_text(json.dumps(data, indent=2))
    print(f"JSON exported: {path}", file=sys.stderr)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ============================================================================
# Regression Detection
# ============================================================================

def check_regression(current: List[PerfMetrics], baseline_path: Path,
                     threshold: float = 5.0) -> List[str]:
    """Compare current results against baseline JSON. Return list of alerts."""
    if not baseline_path.exists():
        return [f"Baseline not found: {baseline_path}"]

    baseline = json.loads(baseline_path.read_text())
    baseline_workloads = baseline.get("workloads", {})
    alerts = []

    for r in current:
        if r.workload not in baseline_workloads:
            continue
        b = baseline_workloads[r.workload]
        b_ipc = b.get("ipc", 0)
        if b_ipc > 0 and r.ipc > 0:
            delta_pct = (r.ipc - b_ipc) / b_ipc * 100
            if delta_pct < -threshold:
                alerts.append(
                    f"REGRESSION: {r.workload} IPC dropped {abs(delta_pct):.1f}% "
                    f"({b_ipc:.3f} -> {r.ipc:.3f})")
            elif delta_pct > threshold:
                alerts.append(
                    f"IMPROVEMENT: {r.workload} IPC improved {delta_pct:.1f}% "
                    f"({b_ipc:.3f} -> {r.ipc:.3f})")

        b_cycles = b.get("cycles", 0)
        if b_cycles > 0 and r.cycles > 0:
            delta_pct = (r.cycles - b_cycles) / b_cycles * 100
            if delta_pct > threshold:
                alerts.append(
                    f"REGRESSION: {r.workload} cycles increased {delta_pct:.1f}% "
                    f"({b_cycles} -> {r.cycles})")

    return alerts


# ============================================================================
# Markdown Report
# ============================================================================

def _bar(pct: float, width: int = 20) -> str:
    """ASCII bar chart."""
    filled = int(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def generate_markdown(results: List[PerfMetrics],
                      alerts: Optional[List[str]] = None,
                      repro_cmd: Optional[str] = None) -> str:
    """Generate Markdown dashboard report."""
    commit = _git_commit()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# RalphGPU Performance Dashboard",
        "",
        f"**Generated:** {now}  ",
        f"**Commit:** `{commit}`",
        "",
    ]

    if repro_cmd:
        lines.append("## Reproduce")
        lines.append("")
        lines.append(f"Run: `{repro_cmd}`")
        lines.append("")

    # Regression alerts
    if alerts:
        has_regression = any(a.startswith("REGRESSION") for a in alerts)
        lines.append("## Regression Alerts")
        lines.append("")
        for a in alerts:
            icon = "X" if a.startswith("REGRESSION") else "OK"
            lines.append(f"- [{icon}] {a}")
        lines.append("")
        if has_regression:
            lines.append("> **Action required:** IPC regression > 5% detected.")
            lines.append("")

    # Summary table
    lines.append("## IPC Summary")
    lines.append("")
    lines.append("| Workload | Category | Status | Cycles | Instr | IPC | Occupancy | Stall% |")
    lines.append("|----------|----------|--------|--------|-------|-----|-----------|--------|")
    for r in results:
        wname = WORKLOADS.get(r.workload, {}).get("name", r.workload)
        lines.append(
            f"| {wname} | {r.category} | {r.status} | "
            f"{r.cycles:,} | {r.instructions:,} | {r.ipc:.3f} | {r.occupancy_pct:.1f}% | {r.stall_pct:.1f}% |"
        )
    lines.append("")

    # KPI table required for reproducible baselines.
    lines.append("## Baseline KPI (Cycles/IPC/Stalls)")
    lines.append("")
    lines.append("| Workload | Cycles | IPC | Occupancy | Stall Mem | Stall Scoreboard | Stall Fetch |")
    lines.append("|----------|--------|-----|-----------|-----------|------------------|-------------|")
    for r in results:
        if not r.cycles:
            continue
        wname = WORKLOADS.get(r.workload, {}).get("name", r.workload)
        lines.append(
            f"| {wname} | {r.cycles:,} | {r.ipc:.3f} | {r.occupancy_pct:.1f}% | "
            f"{r.stall_mem} ({r.stall_mem / r.cycles * 100:.1f}%) | "
            f"{r.stall_raw} ({r.stall_raw / r.cycles * 100:.1f}%) | "
            f"{r.stall_ifetch} ({r.stall_ifetch / r.cycles * 100:.1f}%) |"
        )
    lines.append("")

    # Stall breakdown per workload
    lines.append("## Stall Breakdown")
    lines.append("")
    for r in results:
        if not r.cycles:
            continue
        wname = WORKLOADS.get(r.workload, {}).get("name", r.workload)
        breakdown = r.stall_breakdown()
        if not breakdown:
            continue
        lines.append(f"### {wname}")
        lines.append("")
        lines.append("```")
        for name, pct in sorted(breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:>8s} {_bar(pct)} {pct:5.1f}%")
        lines.append(f"  {'TOTAL':>8s} {' ' * 20} {r.stall_pct:5.1f}%")
        lines.append("```")
        lines.append("")

    # FU utilization
    lines.append("## FU Utilization")
    lines.append("")
    lines.append("| Workload | ALU | FPU | SFU | LDST | Tensor |")
    lines.append("|----------|-----|-----|-----|------|--------|")
    for r in results:
        if not r.cycles:
            continue
        wname = WORKLOADS.get(r.workload, {}).get("name", r.workload)
        util = r.fu_utilization()
        lines.append(
            f"| {wname} | "
            f"{util.get('alu', 0):.1f}% | "
            f"{util.get('fpu', 0):.1f}% | "
            f"{util.get('sfu', 0):.1f}% | "
            f"{util.get('ldst', 0):.1f}% | "
            f"{util.get('tensor', 0):.1f}% |"
        )
    lines.append("")

    # Workload descriptions
    lines.append("## Benchmark Workloads")
    lines.append("")
    lines.append("| ID | Name | Category | Description |")
    lines.append("|----|------|----------|-------------|")
    used_workloads = [r.workload for r in results if r.workload in WORKLOADS]
    for wid in resolve_workload_ids(used_workloads):
        wdef = WORKLOADS[wid]
        lines.append(f"| `{wid}` | {wdef['name']} | {wdef['category']} | {wdef['description']} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by `tools/perf_dashboard.py` at {now}*")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RalphGPU Performance Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", action="store_true",
                        help="Run all benchmark workloads")
    parser.add_argument("--parse", action="append", metavar="FILE",
                        help="Parse existing log file(s)")
    parser.add_argument("--workload", action="append", choices=sorted(WORKLOADS.keys()),
                        help="Workload id(s) to run/filter (repeatable)")
    parser.add_argument("--baseline", type=Path, metavar="FILE",
                        help="Baseline JSON for regression detection")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Regression threshold percentage (default: 5.0)")
    parser.add_argument("--csv", type=Path, metavar="FILE",
                        help="Export results to CSV")
    parser.add_argument("--json", type=Path, metavar="FILE",
                        help="Export results to JSON")
    parser.add_argument("--repro-cmd", type=str,
                        help="Command used to reproduce this report")
    parser.add_argument("-o", "--output", type=Path,
                        default=PROJECT_ROOT / "docs" / "PERF_DASHBOARD.md",
                        help="Markdown output path")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    if not args.run and not args.parse:
        parser.print_help()
        return 1

    results: List[PerfMetrics] = []

    selected_workloads = resolve_workload_ids(args.workload) if args.workload else None

    if args.run:
        results = run_all_benchmarks(selected_workloads)
    elif args.parse:
        for i, filepath in enumerate(args.parse):
            p = Path(filepath)
            if not p.exists():
                print(f"Warning: {filepath} not found", file=sys.stderr)
                continue
            content = p.read_text()
            # Derive workload name from filename
            wid = p.stem.replace("dashboard_", "").replace("bench_", "")
            cat = "Unknown"
            if wid in WORKLOADS:
                cat = WORKLOADS[wid]["category"]
            else:
                matches = [k for k in WORKLOADS if k in wid or wid in k]
                if matches:
                    wid = max(matches, key=len)
                    cat = WORKLOADS[wid]["category"]
            if selected_workloads and wid not in selected_workloads:
                continue
            results.append(parse_sim_output(content, workload=wid, category=cat))

    if not results:
        print("No results collected.", file=sys.stderr)
        return 1

    # Regression check
    alerts = []
    if args.baseline:
        alerts = check_regression(results, args.baseline, args.threshold)
        for a in alerts:
            print(a, file=sys.stderr)

    # Export
    if args.csv:
        export_csv(results, args.csv)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        export_json(results, args.json)

    # Markdown report
    report = generate_markdown(results, alerts, args.repro_cmd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    if not args.quiet:
        print(report)
        print(f"\nDashboard written to: {args.output}", file=sys.stderr)

    # Exit non-zero if regressions detected
    if any(a.startswith("REGRESSION") for a in alerts):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
