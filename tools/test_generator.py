#!/usr/bin/env python3
"""
RalphGPU Python Test Generator
Generates randomized PTX test cases and expected results from FRM

Usage:
    python test_generator.py --gen alu        # Generate ALU tests
    python test_generator.py --gen all        # Generate all tests
    python test_generator.py --list           # List generated tests
"""

import os
import sys
import random
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from ptx_assembler import PTXAssembler

# Output directory for generated tests
OUTPUT_DIR = Path(__file__).parent.parent / "build" / "generated_tests"
PTX_VERSION = "9.1"
PTX_TARGET = "sm_90"
PTX_ADDRESS_SIZE = 64


@dataclass
class TestCase:
    """A single test case with input, program, and expected output"""
    name: str
    category: str
    ptx_code: List[str]
    initial_regs: Dict[int, int]  # Register index -> initial value
    expected_regs: Dict[int, int]  # Register index -> expected value
    initial_memory: Dict[int, int] = None  # Address -> value
    expected_memory: Dict[int, int] = None  # Address -> value


class ALUTestGenerator:
    """Generate randomized ALU test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_add_s32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate add.s32 test cases"""
        tests = []
        for i in range(count):
            # Generate random operands
            a = random.randint(-2**31, 2**31 - 1) & 0xFFFFFFFF
            b = random.randint(-2**31, 2**31 - 1) & 0xFFFFFFFF
            expected = (a + b) & 0xFFFFFFFF

            tests.append(TestCase(
                name=f"add_s32_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "add.s32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))
        return tests

    def gen_sub_s32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate sub.s32 test cases"""
        tests = []
        for i in range(count):
            a = random.randint(-2**31, 2**31 - 1) & 0xFFFFFFFF
            b = random.randint(-2**31, 2**31 - 1) & 0xFFFFFFFF
            expected = (a - b) & 0xFFFFFFFF

            tests.append(TestCase(
                name=f"sub_s32_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "sub.s32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))
        return tests

    def gen_mul_lo_tests(self, count: int = 10) -> List[TestCase]:
        """Generate mul.lo.s32 test cases across full 32-bit range"""
        tests = []

        def rand_s32() -> int:
            return random.randint(-2**31, 2**31 - 1)

        edge_cases = [
            (0, 0),
            (1, -1),
            (-1, -1),
            (0x7FFFFFFF, 2),
            (-0x80000000, 1),
            (-0x80000000, -1),
        ]

        pairs = edge_cases + [(rand_s32(), rand_s32()) for _ in range(max(0, count - len(edge_cases)))]
        pairs = pairs[:count]

        for i, (a_signed, b_signed) in enumerate(pairs):
            a = a_signed & 0xFFFFFFFF
            b = b_signed & 0xFFFFFFFF
            expected = (a_signed * b_signed) & 0xFFFFFFFF

            tests.append(TestCase(
                name=f"mul_lo_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "mul.lo.s32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))
        return tests

    def gen_logic_tests(self, count: int = 10) -> List[TestCase]:
        """Generate bitwise logic test cases"""
        tests = []
        ops = [
            ("and.b32", lambda a, b: a & b),
            ("or.b32", lambda a, b: a | b),
            ("xor.b32", lambda a, b: a ^ b),
        ]

        for i in range(count):
            a = random.randint(0, 0xFFFFFFFF)
            b = random.randint(0, 0xFFFFFFFF)
            op_name, op_func = random.choice(ops)
            expected = op_func(a, b)

            tests.append(TestCase(
                name=f"logic_{op_name.replace('.', '_')}_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    f"{op_name} r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))
        return tests

    def gen_shift_tests(self, count: int = 10) -> List[TestCase]:
        """Generate shift test cases"""
        tests = []

        for i in range(count):
            a = random.randint(0, 0xFFFFFFFF)
            shift = random.randint(0, 31)

            # Test SHL
            expected_shl = (a << shift) & 0xFFFFFFFF
            tests.append(TestCase(
                name=f"shl_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {shift}",
                    "shl.b32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_shl}
            ))

            # Test SHR.U
            expected_shr = a >> shift
            tests.append(TestCase(
                name=f"shr_u_{i:03d}",
                category="ALU",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {shift}",
                    "shr.u32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_shr}
            ))
        return tests

    def gen_all_alu_tests(self) -> List[TestCase]:
        """Generate comprehensive ALU test suite"""
        tests = []
        tests.extend(self.gen_add_s32_tests(10))
        tests.extend(self.gen_sub_s32_tests(10))
        tests.extend(self.gen_mul_lo_tests(10))
        tests.extend(self.gen_logic_tests(10))
        tests.extend(self.gen_shift_tests(10))
        return tests


class FP32TestGenerator:
    """Generate randomized FP32 test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def float_to_uint(self, f: float) -> int:
        return struct.unpack('I', struct.pack('f', f))[0]

    def uint_to_float(self, i: int) -> float:
        return struct.unpack('f', struct.pack('I', i))[0]

    def gen_fp32_arith_tests(self, count: int = 10) -> List[TestCase]:
        """Generate FP32 arithmetic test cases"""
        tests = []
        ops = [
            ("add.f32", lambda a, b: a + b),
            ("sub.f32", lambda a, b: a - b),
            ("mul.f32", lambda a, b: a * b),
        ]

        for i in range(count):
            a = random.uniform(-100.0, 100.0)
            b = random.uniform(-100.0, 100.0)
            if abs(b) < 0.001:  # Avoid division issues
                b = 1.0

            op_name, op_func = random.choice(ops)
            expected_f = op_func(a, b)

            a_uint = self.float_to_uint(a)
            b_uint = self.float_to_uint(b)
            expected_uint = self.float_to_uint(expected_f)

            tests.append(TestCase(
                name=f"fp32_{op_name.replace('.', '_')}_{i:03d}",
                category="FP32",
                ptx_code=[
                    f"mov.u32 r1, {a_uint}",
                    f"mov.u32 r2, {b_uint}",
                    f"{op_name} r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_uint}
            ))
        return tests


class MemoryTestGenerator:
    """Generate memory operation test cases (LD/ST global and shared)"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_ld_st_global_tests(self, count: int = 10) -> List[TestCase]:
        """Generate ld.global and st.global test cases"""
        tests = []

        for i in range(count):
            # Generate test value and address offset
            value = random.randint(0, 0xFFFFFFFF)
            addr_offset = i * 4  # Each test uses different address

            # Test: store value to global memory, then load it back
            tests.append(TestCase(
                name=f"ld_st_global_{i:03d}",
                category="memory",
                ptx_code=[
                    f"mov.u32 r1, {value}",      # Value to store
                    f"mov.u32 r2, {addr_offset}", # Address
                    "st.global.u32 [r2], r1",    # Store to global
                    "ld.global.u32 r3, [r2]",    # Load from global
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: value},
                initial_memory={addr_offset: 0},
                expected_memory={addr_offset: value}
            ))
        return tests

    def gen_ld_st_shared_tests(self, count: int = 10) -> List[TestCase]:
        """Generate ld.shared and st.shared test cases"""
        tests = []

        for i in range(count):
            value = random.randint(0, 0xFFFFFFFF)
            addr_offset = i * 4

            # Test: store value to shared memory, then load it back
            tests.append(TestCase(
                name=f"ld_st_shared_{i:03d}",
                category="memory",
                ptx_code=[
                    f"mov.u32 r1, {value}",
                    f"mov.u32 r2, {addr_offset}",
                    "st.shared.u32 [r2], r1",
                    "ld.shared.u32 r3, [r2]",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: value}
            ))
        return tests

    def gen_all_memory_tests(self) -> List[TestCase]:
        """Generate all memory test cases"""
        tests = []
        tests.extend(self.gen_ld_st_global_tests(10))
        tests.extend(self.gen_ld_st_shared_tests(10))
        return tests


class DivTestGenerator:
    """Generate integer division and remainder test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_div_s32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate div.s32 (signed division) test cases"""
        tests = []

        # Edge cases first
        edge_cases = [
            (10, 3),      # Simple positive
            (-10, 3),     # Negative dividend
            (10, -3),     # Negative divisor
            (-10, -3),    # Both negative
            (0, 5),       # Zero dividend
            (100, 1),     # Division by 1
            (-100, 1),    # Negative divided by 1
            (0x7FFFFFFF, 2),  # Large positive
        ]

        for i, (a_s, b_s) in enumerate(edge_cases[:count]):
            a = a_s & 0xFFFFFFFF
            b = b_s & 0xFFFFFFFF

            # Python truncation toward zero (same as C/PTX)
            if b_s != 0:
                expected_s = int(a_s / b_s)  # Truncate toward zero
            else:
                expected_s = 0xFFFFFFFF if a_s >= 0 else 1

            expected = expected_s & 0xFFFFFFFF

            tests.append(TestCase(
                name=f"div_s32_{i:03d}",
                category="div",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "div.s32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))

        return tests

    def gen_div_u32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate div.u32 (unsigned division) test cases"""
        tests = []

        edge_cases = [
            (100, 3),      # Simple
            (0, 5),        # Zero dividend
            (100, 1),      # Division by 1
            (0xFFFFFFFF, 2),  # Large number
            (0x80000000, 2),  # Sign bit set
            (1000, 10),
            (255, 16),
            (1024, 32),
        ]

        for i, (a, b) in enumerate(edge_cases[:count]):
            if b != 0:
                expected = a // b
            else:
                expected = 0xFFFFFFFF

            tests.append(TestCase(
                name=f"div_u32_{i:03d}",
                category="div",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "div.u32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))

        return tests

    def gen_rem_s32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate rem.s32 (signed remainder) test cases"""
        tests = []

        edge_cases = [
            (10, 3),      # 10 % 3 = 1
            (-10, 3),     # -10 % 3 = -1 (C/PTX truncation)
            (10, -3),     # 10 % -3 = 1
            (-10, -3),    # -10 % -3 = -1
            (0, 5),       # 0 % 5 = 0
            (7, 7),       # 7 % 7 = 0
            (100, 10),    # 100 % 10 = 0
            (-17, 5),     # -17 % 5 = -2
        ]

        for i, (a_s, b_s) in enumerate(edge_cases[:count]):
            a = a_s & 0xFFFFFFFF
            b = b_s & 0xFFFFFFFF

            if b_s != 0:
                # Remainder has same sign as dividend (truncation toward zero)
                expected_s = a_s - int(a_s / b_s) * b_s
            else:
                expected_s = a_s

            expected = expected_s & 0xFFFFFFFF

            tests.append(TestCase(
                name=f"rem_s32_{i:03d}",
                category="div",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "rem.s32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))

        return tests

    def gen_rem_u32_tests(self, count: int = 10) -> List[TestCase]:
        """Generate rem.u32 (unsigned remainder) test cases"""
        tests = []

        edge_cases = [
            (10, 3),      # 10 % 3 = 1
            (100, 7),     # 100 % 7 = 2
            (0, 5),       # 0 % 5 = 0
            (255, 16),    # 255 % 16 = 15
            (1000, 100),  # 1000 % 100 = 0
            (0xFFFFFFFF, 10),  # Large % 10
            (123, 123),   # n % n = 0
            (50, 100),    # a < b: a % b = a
        ]

        for i, (a, b) in enumerate(edge_cases[:count]):
            if b != 0:
                expected = a % b
            else:
                expected = a

            tests.append(TestCase(
                name=f"rem_u32_{i:03d}",
                category="div",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    "rem.u32 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))

        return tests

    def gen_all_div_tests(self) -> List[TestCase]:
        """Generate all DIV/REM test cases"""
        tests = []
        tests.extend(self.gen_div_s32_tests(8))
        tests.extend(self.gen_div_u32_tests(8))
        tests.extend(self.gen_rem_s32_tests(8))
        tests.extend(self.gen_rem_u32_tests(8))
        return tests


class BranchTestGenerator:
    """Generate branch/control flow test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_unconditional_branch_tests(self, count: int = 5) -> List[TestCase]:
        """Generate unconditional branch tests"""
        tests = []

        # Test 1: Simple forward branch (skip one instruction)
        tests.append(TestCase(
            name="bra_forward_000",
            category="branch",
            ptx_code=[
                "mov.u32 r1, 100",        # r1 = 100
                "bra skip1",              # Jump over next instruction
                "mov.u32 r1, 999",        # Should be skipped
                "skip1:",
                "mov.u32 r2, 200",        # r2 = 200
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 100, 2: 200}  # r1 should be 100, not 999
        ))

        # Test 2: Branch with computation
        tests.append(TestCase(
            name="bra_forward_001",
            category="branch",
            ptx_code=[
                "mov.u32 r1, 10",
                "mov.u32 r2, 20",
                "add.s32 r3, r1, r2",     # r3 = 30
                "bra done",
                "mov.u32 r3, 0",          # Should be skipped
                "done:",
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 30}
        ))

        # Test 3: Multiple branches
        tests.append(TestCase(
            name="bra_chain_000",
            category="branch",
            ptx_code=[
                "mov.u32 r1, 1",
                "bra step2",
                "mov.u32 r1, 0",          # Skipped
                "step2:",
                "add.s32 r1, r1, r1",     # r1 = 2
                "bra step3",
                "mov.u32 r1, 0",          # Skipped
                "step3:",
                "add.s32 r1, r1, r1",     # r1 = 4
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 4}
        ))

        return tests

    def gen_conditional_setp_tests(self, count: int = 5) -> List[TestCase]:
        """Generate conditional tests using setp and predicated branch"""
        tests = []

        # Test: setp comparison followed by predicated branch (FRM supports @p bra)
        test_cases = [
            # (a, b, relation, expected_result if a rel b else alt)
            (10, 5, "gt", 100, 200),   # 10 > 5: true, expect 100
            (5, 10, "gt", 100, 200),   # 5 > 10: false, expect 200
            (5, 5, "eq", 100, 200),    # 5 == 5: true, expect 100
            (5, 10, "eq", 100, 200),   # 5 == 10: false, expect 200
            (3, 10, "lt", 100, 200),   # 3 < 10: true, expect 100
        ]

        for i, (a, b, rel, val_true, val_false) in enumerate(test_cases):
            expected = val_true if (
                (rel == "gt" and a > b) or
                (rel == "eq" and a == b) or
                (rel == "lt" and a < b)
            ) else val_false

            # Use predicated branch pattern instead of predicated move
            # (FRM supports @p bra, but not @p mov)
            tests.append(TestCase(
                name=f"setp_{rel}_{i:03d}",
                category="branch",
                ptx_code=[
                    f"mov.u32 r1, {a}",
                    f"mov.u32 r2, {b}",
                    f"setp.{rel}.s32 p0, r1, r2",  # Set predicate p0
                    f"mov.u32 r3, {val_false}",   # Default: false value
                    f"@p0 bra set_true",          # Branch if condition true
                    "bra done",                   # Skip to end
                    "set_true:",
                    f"mov.u32 r3, {val_true}",    # Set true value
                    "done:",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected}
            ))

        return tests

    def gen_all_branch_tests(self) -> List[TestCase]:
        """Generate all branch test cases"""
        tests = []
        tests.extend(self.gen_unconditional_branch_tests())
        tests.extend(self.gen_conditional_setp_tests())
        return tests


def write_test_case(test: TestCase, output_dir: Path):
    """Write a test case to PTX and expected results files"""
    output_dir.mkdir(parents=True, exist_ok=True)

    header = [
        f"// Auto-generated test: {test.name}",
        f"// Category: {test.category}",
        f".version {PTX_VERSION}",
        f".target {PTX_TARGET}",
        f".address_size {PTX_ADDRESS_SIZE}",
        "",
        f".entry {test.name}()",
        "{"
    ]

    body = [f"    {line}" for line in test.ptx_code]
    footer = ["}"]
    ptx_program = header + body + footer

    # Write PTX file
    ptx_file = output_dir / f"{test.name}.ptx"
    with open(ptx_file, 'w') as f:
        for line in ptx_program:
            f.write(line + "\n")

    # Assemble to HEX
    try:
        assembler = PTXAssembler()
        asm_lines = []
        for line in ptx_program:
            stripped = line.split('//')[0].strip()
            if not stripped or stripped.startswith('.') or stripped in ('{', '}'):
                continue
            asm_lines.append(stripped)
        assembler.first_pass(asm_lines)
        machine_code = assembler.second_pass(asm_lines)

        hex_file = output_dir / f"{test.name}.hex"
        with open(hex_file, 'w') as f:
            for code in machine_code:
                f.write(f"{code:08x}\n")
    except Exception as e:
        print(f"Warning: Failed to assemble {test.name}: {e}")
        return False

    # Write expected results
    expected_file = output_dir / f"{test.name}.expected"
    with open(expected_file, 'w') as f:
        f.write(f"# Expected results for {test.name}\n")
        for reg, val in test.expected_regs.items():
            f.write(f"r{reg}={val:08x}\n")
        if test.expected_memory:
            for addr, val in test.expected_memory.items():
                f.write(f"mem[{addr:08x}]={val:08x}\n")

    return True


def generate_all_tests():
    """Generate all test suites"""
    output_dir = OUTPUT_DIR

    print(f"Generating tests to {output_dir}")

    # ALU tests
    alu_gen = ALUTestGenerator(seed=42)
    alu_tests = alu_gen.gen_all_alu_tests()
    print(f"Generated {len(alu_tests)} ALU tests")

    success_count = 0
    for test in alu_tests:
        if write_test_case(test, output_dir / "alu"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(alu_tests)} ALU tests")

    # FP32 tests
    fp32_gen = FP32TestGenerator(seed=42)
    fp32_tests = fp32_gen.gen_fp32_arith_tests(10)
    print(f"Generated {len(fp32_tests)} FP32 tests")

    success_count = 0
    for test in fp32_tests:
        if write_test_case(test, output_dir / "fp32"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(fp32_tests)} FP32 tests")

    # Memory tests
    mem_gen = MemoryTestGenerator(seed=42)
    mem_tests = mem_gen.gen_all_memory_tests()
    print(f"Generated {len(mem_tests)} Memory tests")

    success_count = 0
    for test in mem_tests:
        if write_test_case(test, output_dir / "memory"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(mem_tests)} Memory tests")

    # Branch tests
    branch_gen = BranchTestGenerator(seed=42)
    branch_tests = branch_gen.gen_all_branch_tests()
    print(f"Generated {len(branch_tests)} Branch tests")

    success_count = 0
    for test in branch_tests:
        if write_test_case(test, output_dir / "branch"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(branch_tests)} Branch tests")

    # DIV/REM tests
    div_gen = DivTestGenerator(seed=42)
    div_tests = div_gen.gen_all_div_tests()
    print(f"Generated {len(div_tests)} DIV/REM tests")

    success_count = 0
    for test in div_tests:
        if write_test_case(test, output_dir / "div"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(div_tests)} DIV/REM tests")

    # Summary
    total_tests = len(alu_tests) + len(fp32_tests) + len(mem_tests) + len(branch_tests) + len(div_tests)
    print(f"\nTotal: {total_tests} tests generated")
    return total_tests


def list_tests():
    """List all generated tests"""
    if not OUTPUT_DIR.exists():
        print("No generated tests found. Run --gen first.")
        return

    for category_dir in sorted(OUTPUT_DIR.iterdir()):
        if category_dir.is_dir():
            ptx_files = list(category_dir.glob("*.ptx"))
            print(f"\n{category_dir.name}: {len(ptx_files)} tests")
            for ptx_file in sorted(ptx_files)[:5]:
                print(f"  - {ptx_file.stem}")
            if len(ptx_files) > 5:
                print(f"  ... and {len(ptx_files) - 5} more")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RalphGPU Test Generator")
    parser.add_argument("--gen", choices=["alu", "fp32", "memory", "branch", "div", "all"],
                       help="Generate test cases")
    parser.add_argument("--list", action="store_true",
                       help="List generated tests")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.list:
        list_tests()
    elif args.gen:
        if args.gen == "all":
            generate_all_tests()
        elif args.gen == "alu":
            gen = ALUTestGenerator(seed=args.seed)
            tests = gen.gen_all_alu_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "alu"))
            print(f"Generated {success}/{len(tests)} ALU tests")
        elif args.gen == "fp32":
            gen = FP32TestGenerator(seed=args.seed)
            tests = gen.gen_fp32_arith_tests(10)
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "fp32"))
            print(f"Generated {success}/{len(tests)} FP32 tests")
        elif args.gen == "memory":
            gen = MemoryTestGenerator(seed=args.seed)
            tests = gen.gen_all_memory_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "memory"))
            print(f"Generated {success}/{len(tests)} Memory tests")
        elif args.gen == "branch":
            gen = BranchTestGenerator(seed=args.seed)
            tests = gen.gen_all_branch_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "branch"))
            print(f"Generated {success}/{len(tests)} Branch tests")
        elif args.gen == "div":
            gen = DivTestGenerator(seed=args.seed)
            tests = gen.gen_all_div_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "div"))
            print(f"Generated {success}/{len(tests)} DIV/REM tests")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
