#!/usr/bin/env python3
"""
RalphGPU Performance Analysis
=============================

Parses benchmark output from tb_bench_atomics.v and tb_bench_divergence.v,
collects metrics (cycles, ops/cycle, timeout), and generates PERFORMANCE_REPORT.md.

Usage:
    python perf_analysis.py [--parse FILE] [--report] [--all]
    python perf_analysis.py --parse atomics.log --parse divergence.log --report
"""

import re
import sys
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class BenchmarkResult:
    """Parsed result from a single benchmark test"""
    name: str
    category: str  # 'atomic' or 'divergence'
    cycles: int = 0
    baseline_cycles: int = 0
    improvement_pct: float = 0.0
    ops: int = 0
    cycles_per_op: float = 0.0
    ops_per_cycle: float = 0.0
    status: str = "UNKNOWN"  # PASS, FAIL, TIMEOUT

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def timed_out(self) -> bool:
        return self.status == "TIMEOUT"


def parse_testbench_output(content: str) -> List[BenchmarkResult]:
    """
    Parse output from tb_bench_atomics.v or tb_bench_divergence.v.

    Expected format per test:
        Test Name
        |-- Baseline: X cycles
        |-- Optimized: Y cycles
        |-- Improvement: Z%
        |-- Ops: N
        |-- Cycles/Op: A
        |-- Ops/Cycle: B
        `-- Conclusion: PASS/FAIL/TIMEOUT
    """
    results = []

    # Split into test blocks - each starts with a name (no prefix) followed by |--
    pattern = r'^([A-Za-z][^\n]+)\n((?:\|--[^\n]+\n)+`--[^\n]+)'

    for match in re.finditer(pattern, content, re.MULTILINE):
        name = match.group(1).strip()
        block = match.group(2)

        # Determine category from name
        if 'Atomic' in name:
            category = 'atomic'
        elif 'Divergence' in name:
            category = 'divergence'
        else:
            category = 'other'

        result = BenchmarkResult(name=name, category=category)

        # Parse each metric
        if m := re.search(r'Baseline:\s*(\d+)\s*cycles', block):
            result.baseline_cycles = int(m.group(1))

        if m := re.search(r'Optimized:\s*(\d+)\s*cycles', block):
            result.cycles = int(m.group(1))

        if m := re.search(r'Improvement:\s*([-\d.]+)%', block):
            result.improvement_pct = float(m.group(1))

        if m := re.search(r'Ops:\s*(\d+)', block):
            result.ops = int(m.group(1))

        if m := re.search(r'Cycles/Op:\s*([\d.]+)', block):
            result.cycles_per_op = float(m.group(1))

        if m := re.search(r'Ops/Cycle:\s*([\d.]+)', block):
            result.ops_per_cycle = float(m.group(1))

        if m := re.search(r'Conclusion:\s*(\w+)', block):
            result.status = m.group(1).upper()

        results.append(result)

    return results


def parse_summary(content: str) -> dict:
    """Parse the summary line: 'Summary: X passed, Y failed'"""
    if m := re.search(r'Summary:\s*(\d+)\s*passed,\s*(\d+)\s*failed', content):
        return {'passed': int(m.group(1)), 'failed': int(m.group(2))}
    return {'passed': 0, 'failed': 0}


def generate_report(results: List[BenchmarkResult], output_path: Optional[Path] = None) -> str:
    """Generate PERFORMANCE_REPORT.md from parsed results."""

    atomic_results = [r for r in results if r.category == 'atomic']
    div_results = [r for r in results if r.category == 'divergence']

    atomic_passed = sum(1 for r in atomic_results if r.passed)
    div_passed = sum(1 for r in div_results if r.passed)
    total_passed = sum(1 for r in results if r.passed)
    total_timeout = sum(1 for r in results if r.timed_out)

    # Calculate averages
    def avg(lst, attr):
        vals = [getattr(r, attr) for r in lst if getattr(r, attr, 0) > 0]
        return sum(vals) / len(vals) if vals else 0

    report = f"""# RalphGPU Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {len(results)} |
| Passed | {total_passed} |
| Failed | {len(results) - total_passed - total_timeout} |
| Timeout | {total_timeout} |

---

## Atomic Operations (Track 1)

| Test | Cycles | Ops | Ops/Cycle | Status |
|------|--------|-----|-----------|--------|
"""

    for r in atomic_results:
        status_icon = "✓" if r.passed else ("⏱" if r.timed_out else "✗")
        ops_cycle = f"{r.ops_per_cycle:.4f}" if r.ops_per_cycle > 0 else "N/A"
        report += f"| {r.name} | {r.cycles} | {r.ops} | {ops_cycle} | {status_icon} {r.status} |\n"

    if not atomic_results:
        report += "| (no results) | - | - | - | - |\n"

    report += f"""
**Atomic Summary:** {atomic_passed}/{len(atomic_results)} passed

---

## Branch Divergence (Track 2)

| Test | Cycles | Ops | Ops/Cycle | Status |
|------|--------|-----|-----------|--------|
"""

    for r in div_results:
        status_icon = "✓" if r.passed else ("⏱" if r.timed_out else "✗")
        ops_cycle = f"{r.ops_per_cycle:.4f}" if r.ops_per_cycle > 0 else "N/A"
        report += f"| {r.name} | {r.cycles} | {r.ops} | {ops_cycle} | {status_icon} {r.status} |\n"

    if not div_results:
        report += "| (no results) | - | - | - | - |\n"

    report += f"""
**Divergence Summary:** {div_passed}/{len(div_results)} passed

---

## Performance Metrics

| Category | Avg Cycles | Avg Ops/Cycle |
|----------|------------|---------------|
| Atomic | {avg(atomic_results, 'cycles'):.0f} | {avg(atomic_results, 'ops_per_cycle'):.4f} |
| Divergence | {avg(div_results, 'cycles'):.0f} | {avg(div_results, 'ops_per_cycle'):.4f} |

---

## Status Legend

- ✓ PASS - Test completed with correct results
- ✗ FAIL - Test completed but results incorrect
- ⏱ TIMEOUT - Test exceeded maximum cycle count

---

*Generated by scripts/perf_analysis.py*
"""

    if output_path:
        output_path.write_text(report)
        print(f"Report written to: {output_path}")

    return report


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse benchmark output and generate performance report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse simulation logs and generate report
  python perf_analysis.py --parse atomics.log --parse divergence.log

  # Parse from stdin (piped simulation output)
  cat simulation.log | python perf_analysis.py --stdin

  # Specify output location
  python perf_analysis.py --parse results.log -o PERFORMANCE_REPORT.md
        """
    )
    parser.add_argument('--parse', action='append', metavar='FILE',
                        help='Log file(s) to parse (can specify multiple)')
    parser.add_argument('--stdin', action='store_true',
                        help='Read from stdin')
    parser.add_argument('-o', '--output', type=Path,
                        default=PROJECT_ROOT / 'PERFORMANCE_REPORT.md',
                        help='Output report path (default: PERFORMANCE_REPORT.md)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress console output')

    args = parser.parse_args()

    # Collect all input content
    content = ""

    if args.stdin:
        content = sys.stdin.read()
    elif args.parse:
        for filepath in args.parse:
            p = Path(filepath)
            if p.exists():
                content += p.read_text() + "\n"
            else:
                print(f"Warning: File not found: {filepath}", file=sys.stderr)
    else:
        # No input specified - show help
        parser.print_help()
        return 1

    if not content.strip():
        print("Error: No content to parse", file=sys.stderr)
        return 1

    # Parse the content
    results = parse_testbench_output(content)
    summary = parse_summary(content)

    if not args.quiet:
        print(f"Parsed {len(results)} benchmark results")
        for r in results:
            status = "PASS" if r.passed else ("TIMEOUT" if r.timed_out else "FAIL")
            ops_info = f", {r.ops_per_cycle:.4f} ops/cycle" if r.ops_per_cycle > 0 else ""
            print(f"  {r.name}: {r.cycles} cycles{ops_info} [{status}]")

    # Generate report
    report = generate_report(results, args.output)

    if not args.quiet:
        print(f"\nReport written to: {args.output}")

    # Return exit code based on results
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
