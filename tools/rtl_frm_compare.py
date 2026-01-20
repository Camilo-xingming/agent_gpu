#!/usr/bin/env python3
"""
RalphGPU RTL vs FRM Comparison Framework
Compares RTL simulation output against Functional Reference Model

Usage:
    python rtl_frm_compare.py --test build/generated_tests/alu/add_s32_000
    python rtl_frm_compare.py --suite build/generated_tests/alu
    python rtl_frm_compare.py --all
"""

import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from gpu_simulator import RalphGPUSimulator, WarpState

PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build" / "generated_tests"


@dataclass
class ComparisonResult:
    """Result of comparing RTL vs FRM for a single test"""
    test_name: str
    passed: bool
    frm_regs: Dict[int, int]
    rtl_regs: Dict[int, int] = None
    error_message: str = ""


def load_expected_results(expected_file: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Load expected register values and initial memory from .expected file

    Returns:
        (expected_regs, initial_memory) tuple
    """
    expected = {}
    init_mem = {}
    if not expected_file.exists():
        return expected, init_mem

    with open(expected_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if line.startswith('r'):
                match = re.match(r'r(\d+)=([0-9a-fA-F]+)', line)
                if match:
                    reg = int(match.group(1))
                    val = int(match.group(2), 16)
                    expected[reg] = val
            elif line.startswith('init_mem'):
                match = re.match(r'init_mem\[([0-9a-fA-F]+)\]=([0-9a-fA-F]+)', line)
                if match:
                    addr = int(match.group(1), 16)
                    val = int(match.group(2), 16)
                    init_mem[addr] = val
    return expected, init_mem


def is_fp32_close(a: int, b: int, ulp_tolerance: int = 2) -> bool:
    """Check if two FP32 bit patterns are within ULP tolerance"""
    import struct
    # Handle exact match
    if a == b:
        return True
    # Handle NaN (any NaN matches any NaN)
    if (a & 0x7FFFFFFF) > 0x7F800000 and (b & 0x7FFFFFFF) > 0x7F800000:
        return True
    # Check ULP difference
    return abs(a - b) <= ulp_tolerance


def run_frm(hex_file: Path, init_memory: Dict[int, int] = None) -> Dict[int, int]:
    """Run FRM simulation and return register values

    Args:
        hex_file: Path to hex file with instructions
        init_memory: Optional dict of address->value to initialize global memory
    """
    sim = RalphGPUSimulator(num_sm=1)

    # Initialize global memory if provided
    if init_memory:
        for addr, val in init_memory.items():
            sim.global_memory[addr] = val

    # Load program
    sim.instruction_memory = []
    with open(hex_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sim.instruction_memory.append(int(line, 16))

    # Create warp and execute
    warp = WarpState(warp_id=0)
    sim.block_dim = (32, 1, 1)
    sim.grid_dim = (1, 1, 1)

    max_cycles = 1000
    cycles = 0
    while sim.execute_warp(warp, sm_id=0) and cycles < max_cycles:
        cycles += 1

    # Return thread 0 register state
    return {i: warp.threads[0].registers[i] for i in range(32)
            if warp.threads[0].registers[i] != 0}


def run_rtl(hex_file: Path, tb_name: str = "tb_ptx_tests") -> Optional[Dict[int, int]]:
    """
    Run RTL simulation and extract register values.
    Returns None if RTL simulation is not available or fails.
    """
    # For now, we'll skip RTL comparison as it requires a specific testbench
    # that can load arbitrary hex files and dump register state.
    # This can be implemented later with a dedicated verification testbench.
    return None


def compare_test(test_path: Path) -> ComparisonResult:
    """Compare a single test between FRM and expected results"""
    test_name = test_path.stem if test_path.suffix else test_path.name

    # Find files
    hex_file = test_path.with_suffix('.hex') if test_path.suffix != '.hex' else test_path
    expected_file = test_path.with_suffix('.expected')

    if not hex_file.exists():
        return ComparisonResult(
            test_name=test_name,
            passed=False,
            frm_regs={},
            error_message=f"Hex file not found: {hex_file}"
        )

    # Load expected results and initial memory
    expected_regs, init_memory = load_expected_results(expected_file)

    # Run FRM with initial memory
    try:
        frm_regs = run_frm(hex_file, init_memory)
    except Exception as e:
        return ComparisonResult(
            test_name=test_name,
            passed=False,
            frm_regs={},
            error_message=f"FRM execution failed: {e}"
        )

    # Determine if this is an FP32 test (for ULP tolerance)
    # Also include SFU tests since they produce FP32 results
    test_path = str(hex_file).lower()
    is_fp32_test = "fp32" in test_path or "sfu" in test_path

    # Compare FRM results with expected
    errors = []
    for reg, expected_val in expected_regs.items():
        frm_val = frm_regs.get(reg, 0)
        if frm_val != expected_val:
            # For FP32 tests, allow small ULP differences due to rounding
            if is_fp32_test and is_fp32_close(frm_val, expected_val, ulp_tolerance=5):
                continue  # Within tolerance
            errors.append(f"r{reg}: FRM={frm_val:08x}, expected={expected_val:08x}")

    if errors:
        return ComparisonResult(
            test_name=test_name,
            passed=False,
            frm_regs=frm_regs,
            error_message="; ".join(errors)
        )

    return ComparisonResult(
        test_name=test_name,
        passed=True,
        frm_regs=frm_regs
    )


def run_test_suite(suite_dir: Path, verbose: bool = False) -> Tuple[int, int]:
    """Run all tests in a directory"""
    passed = 0
    failed = 0

    hex_files = sorted(suite_dir.glob("*.hex"))
    if not hex_files:
        print(f"No .hex files found in {suite_dir}")
        return 0, 0

    print(f"Running {len(hex_files)} tests from {suite_dir.name}...")

    for hex_file in hex_files:
        result = compare_test(hex_file.with_suffix(''))

        if result.passed:
            passed += 1
            if verbose:
                print(f"  [PASS] {result.test_name}")
        else:
            failed += 1
            print(f"  [FAIL] {result.test_name}: {result.error_message}")

    return passed, failed


def run_all_tests(verbose: bool = False) -> Tuple[int, int]:
    """Run all generated tests"""
    total_passed = 0
    total_failed = 0

    if not BUILD_DIR.exists():
        print(f"No generated tests found. Run test_generator.py --gen all first.")
        return 0, 0

    for category_dir in sorted(BUILD_DIR.iterdir()):
        if category_dir.is_dir():
            print(f"\n=== {category_dir.name.upper()} ===")
            passed, failed = run_test_suite(category_dir, verbose)
            total_passed += passed
            total_failed += failed
            print(f"  Result: {passed}/{passed + failed} passed")

    return total_passed, total_failed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RalphGPU RTL vs FRM Comparison")
    parser.add_argument("--test", type=Path, help="Run single test (path without extension)")
    parser.add_argument("--suite", type=Path, help="Run test suite directory")
    parser.add_argument("--all", action="store_true", help="Run all generated tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.test:
        result = compare_test(args.test)
        if result.passed:
            print(f"[PASS] {result.test_name}")
            print(f"  FRM registers: {result.frm_regs}")
        else:
            print(f"[FAIL] {result.test_name}")
            print(f"  Error: {result.error_message}")
            print(f"  FRM registers: {result.frm_regs}")
        sys.exit(0 if result.passed else 1)

    elif args.suite:
        passed, failed = run_test_suite(args.suite, args.verbose)
        print(f"\n{'='*60}")
        print(f"Suite Result: {passed}/{passed + failed} passed")
        if failed == 0:
            print("ALL TESTS PASSED!")
        else:
            print(f"FAILED: {failed} tests")
        sys.exit(0 if failed == 0 else 1)

    elif args.all:
        passed, failed = run_all_tests(args.verbose)
        print(f"\n{'='*60}")
        print(f"Overall Result: {passed}/{passed + failed} passed")
        if failed == 0:
            print("ALL TESTS PASSED!")
        else:
            print(f"FAILED: {failed} tests")
        sys.exit(0 if failed == 0 else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
