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
import math
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


class SFUTestGenerator:
    """Generate FP32 Special Function Unit (SFU) test cases

    Tests sin.f32, cos.f32, sqrt.f32, rcp.f32, rsqrt.f32, lg2.f32, ex2.f32.
    Uses tolerance-based comparison due to FP approximations.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def float_to_uint(self, f: float) -> int:
        return struct.unpack('I', struct.pack('f', f))[0]

    def uint_to_float(self, i: int) -> float:
        return struct.unpack('f', struct.pack('I', i))[0]

    def gen_sin_tests(self, count: int = 3) -> List[TestCase]:
        """Generate sin.f32 test cases"""
        tests = []
        # Test specific angles with known results
        test_values = [
            (0.0, 0.0),                           # sin(0) = 0
            (math.pi / 6, 0.5),                   # sin(30°) = 0.5
            (math.pi / 4, math.sqrt(2) / 2),      # sin(45°) ≈ 0.707
            (math.pi / 2, 1.0),                   # sin(90°) = 1
            (math.pi, 0.0),                       # sin(180°) = 0
        ]

        for i, (angle, expected) in enumerate(test_values[:count]):
            angle_uint = self.float_to_uint(angle)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_sin_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {angle_uint}",
                    "sin.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_cos_tests(self, count: int = 3) -> List[TestCase]:
        """Generate cos.f32 test cases"""
        tests = []
        test_values = [
            (0.0, 1.0),                           # cos(0) = 1
            (math.pi / 3, 0.5),                   # cos(60°) = 0.5
            (math.pi / 4, math.sqrt(2) / 2),      # cos(45°) ≈ 0.707
            (math.pi / 2, 0.0),                   # cos(90°) = 0
            (math.pi, -1.0),                      # cos(180°) = -1
        ]

        for i, (angle, expected) in enumerate(test_values[:count]):
            angle_uint = self.float_to_uint(angle)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_cos_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {angle_uint}",
                    "cos.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_sqrt_tests(self, count: int = 3) -> List[TestCase]:
        """Generate sqrt.f32 test cases"""
        tests = []
        test_values = [
            (1.0, 1.0),
            (4.0, 2.0),
            (9.0, 3.0),
            (16.0, 4.0),
            (2.0, math.sqrt(2)),
        ]

        for i, (val, expected) in enumerate(test_values[:count]):
            val_uint = self.float_to_uint(val)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_sqrt_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {val_uint}",
                    "sqrt.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_rcp_tests(self, count: int = 3) -> List[TestCase]:
        """Generate rcp.f32 (reciprocal) test cases"""
        tests = []
        test_values = [
            (1.0, 1.0),
            (2.0, 0.5),
            (4.0, 0.25),
            (0.5, 2.0),
            (10.0, 0.1),
        ]

        for i, (val, expected) in enumerate(test_values[:count]):
            val_uint = self.float_to_uint(val)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_rcp_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {val_uint}",
                    "rcp.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_rsqrt_tests(self, count: int = 3) -> List[TestCase]:
        """Generate rsqrt.f32 (reciprocal sqrt) test cases"""
        tests = []
        test_values = [
            (1.0, 1.0),
            (4.0, 0.5),
            (16.0, 0.25),
            (0.25, 2.0),
        ]

        for i, (val, expected) in enumerate(test_values[:count]):
            val_uint = self.float_to_uint(val)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_rsqrt_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {val_uint}",
                    "rsqrt.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_lg2_tests(self, count: int = 3) -> List[TestCase]:
        """Generate lg2.f32 (log base 2) test cases"""
        tests = []
        test_values = [
            (1.0, 0.0),
            (2.0, 1.0),
            (4.0, 2.0),
            (8.0, 3.0),
            (0.5, -1.0),
        ]

        for i, (val, expected) in enumerate(test_values[:count]):
            val_uint = self.float_to_uint(val)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_lg2_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {val_uint}",
                    "lg2.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_ex2_tests(self, count: int = 3) -> List[TestCase]:
        """Generate ex2.f32 (2^x) test cases"""
        tests = []
        test_values = [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
            (3.0, 8.0),
            (-1.0, 0.5),
        ]

        for i, (val, expected) in enumerate(test_values[:count]):
            val_uint = self.float_to_uint(val)
            expected_uint = self.float_to_uint(expected)

            tests.append(TestCase(
                name=f"sfu_ex2_{i:03d}",
                category="sfu",
                ptx_code=[
                    f"mov.u32 r1, {val_uint}",
                    "ex2.f32 r2, r1",
                    "exit"
                ],
                initial_regs={},
                expected_regs={2: expected_uint}
            ))
        return tests

    def gen_all_sfu_tests(self) -> List[TestCase]:
        """Generate all SFU test cases"""
        tests = []
        tests.extend(self.gen_sin_tests(3))
        tests.extend(self.gen_cos_tests(3))
        tests.extend(self.gen_sqrt_tests(3))
        tests.extend(self.gen_rcp_tests(3))
        tests.extend(self.gen_rsqrt_tests(3))
        tests.extend(self.gen_lg2_tests(3))
        tests.extend(self.gen_ex2_tests(3))
        return tests


class FP16TestGenerator:
    """Generate FP16 arithmetic test cases

    FP16 (half-precision) values are stored in the lower 16 bits of 32-bit registers.
    Uses IEEE 754 half-precision format.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def fp16_to_uint16(self, f: float) -> int:
        """Convert float to FP16 bit pattern"""
        return struct.unpack('H', struct.pack('e', f))[0]

    def gen_fp16_add_tests(self, count: int = 3) -> List[TestCase]:
        """Generate add.f16 test cases"""
        tests = []
        test_values = [
            (1.0, 2.0, 3.0),
            (0.5, 0.25, 0.75),
            (-1.0, 1.0, 0.0),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_uint = self.fp16_to_uint16(a)
            b_uint = self.fp16_to_uint16(b)
            expected_uint = self.fp16_to_uint16(expected)

            tests.append(TestCase(
                name=f"fp16_add_{i:03d}",
                category="fp16",
                ptx_code=[
                    f"mov.u32 r1, {a_uint}",
                    f"mov.u32 r2, {b_uint}",
                    "add.f16 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_uint}
            ))
        return tests

    def gen_fp16_sub_tests(self, count: int = 3) -> List[TestCase]:
        """Generate sub.f16 test cases"""
        tests = []
        test_values = [
            (3.0, 1.0, 2.0),
            (1.0, 0.5, 0.5),
            (0.0, 1.0, -1.0),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_uint = self.fp16_to_uint16(a)
            b_uint = self.fp16_to_uint16(b)
            expected_uint = self.fp16_to_uint16(expected)

            tests.append(TestCase(
                name=f"fp16_sub_{i:03d}",
                category="fp16",
                ptx_code=[
                    f"mov.u32 r1, {a_uint}",
                    f"mov.u32 r2, {b_uint}",
                    "sub.f16 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_uint}
            ))
        return tests

    def gen_fp16_mul_tests(self, count: int = 3) -> List[TestCase]:
        """Generate mul.f16 test cases"""
        tests = []
        test_values = [
            (2.0, 3.0, 6.0),
            (0.5, 4.0, 2.0),
            (-1.0, 2.0, -2.0),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_uint = self.fp16_to_uint16(a)
            b_uint = self.fp16_to_uint16(b)
            expected_uint = self.fp16_to_uint16(expected)

            tests.append(TestCase(
                name=f"fp16_mul_{i:03d}",
                category="fp16",
                ptx_code=[
                    f"mov.u32 r1, {a_uint}",
                    f"mov.u32 r2, {b_uint}",
                    "mul.f16 r3, r1, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={3: expected_uint}
            ))
        return tests

    def gen_all_fp16_tests(self) -> List[TestCase]:
        """Generate all FP16 test cases"""
        tests = []
        tests.extend(self.gen_fp16_add_tests(3))
        tests.extend(self.gen_fp16_sub_tests(3))
        tests.extend(self.gen_fp16_mul_tests(3))
        return tests


class FP64TestGenerator:
    """Generate FP64 (double-precision) test cases

    FP64 values use register pairs: rd:rd+1 where rd holds low 32 bits and rd+1 holds high 32 bits.
    Uses IEEE 754 double-precision format.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def fp64_to_uint64(self, f: float) -> tuple:
        """Convert float64 to (lo, hi) 32-bit values"""
        bits = struct.unpack('Q', struct.pack('d', f))[0]
        lo = bits & 0xFFFFFFFF
        hi = (bits >> 32) & 0xFFFFFFFF
        return (lo, hi)

    def gen_fp64_add_tests(self, count: int = 3) -> List[TestCase]:
        """Generate add.f64 test cases"""
        tests = []
        test_values = [
            (1.0, 2.0, 3.0),
            (0.5, 0.25, 0.75),
            (1e10, 2e10, 3e10),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_lo, a_hi = self.fp64_to_uint64(a)
            b_lo, b_hi = self.fp64_to_uint64(b)
            exp_lo, exp_hi = self.fp64_to_uint64(expected)

            # Use register pairs: r0:r1 for A, r2:r3 for B, r4:r5 for result
            tests.append(TestCase(
                name=f"fp64_add_{i:03d}",
                category="fp64",
                ptx_code=[
                    f"mov.u32 r0, {a_lo}",
                    f"mov.u32 r1, {a_hi}",
                    f"mov.u32 r2, {b_lo}",
                    f"mov.u32 r3, {b_hi}",
                    "add.f64 r4, r0, r2",  # r4:r5 = r0:r1 + r2:r3
                    "exit"
                ],
                initial_regs={},
                expected_regs={4: exp_lo, 5: exp_hi}
            ))
        return tests

    def gen_fp64_sub_tests(self, count: int = 3) -> List[TestCase]:
        """Generate sub.f64 test cases"""
        tests = []
        test_values = [
            (5.0, 2.0, 3.0),
            (1.0, 0.5, 0.5),
            (1e10, 3e9, 7e9),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_lo, a_hi = self.fp64_to_uint64(a)
            b_lo, b_hi = self.fp64_to_uint64(b)
            exp_lo, exp_hi = self.fp64_to_uint64(expected)

            tests.append(TestCase(
                name=f"fp64_sub_{i:03d}",
                category="fp64",
                ptx_code=[
                    f"mov.u32 r0, {a_lo}",
                    f"mov.u32 r1, {a_hi}",
                    f"mov.u32 r2, {b_lo}",
                    f"mov.u32 r3, {b_hi}",
                    "sub.f64 r4, r0, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={4: exp_lo, 5: exp_hi}
            ))
        return tests

    def gen_fp64_mul_tests(self, count: int = 3) -> List[TestCase]:
        """Generate mul.f64 test cases"""
        tests = []
        test_values = [
            (2.0, 3.0, 6.0),
            (0.5, 4.0, 2.0),
            (1e5, 1e5, 1e10),
        ]

        for i, (a, b, expected) in enumerate(test_values[:count]):
            a_lo, a_hi = self.fp64_to_uint64(a)
            b_lo, b_hi = self.fp64_to_uint64(b)
            exp_lo, exp_hi = self.fp64_to_uint64(expected)

            tests.append(TestCase(
                name=f"fp64_mul_{i:03d}",
                category="fp64",
                ptx_code=[
                    f"mov.u32 r0, {a_lo}",
                    f"mov.u32 r1, {a_hi}",
                    f"mov.u32 r2, {b_lo}",
                    f"mov.u32 r3, {b_hi}",
                    "mul.f64 r4, r0, r2",
                    "exit"
                ],
                initial_regs={},
                expected_regs={4: exp_lo, 5: exp_hi}
            ))
        return tests

    def gen_all_fp64_tests(self) -> List[TestCase]:
        """Generate all FP64 test cases"""
        tests = []
        tests.extend(self.gen_fp64_add_tests(3))
        tests.extend(self.gen_fp64_sub_tests(3))
        tests.extend(self.gen_fp64_mul_tests(3))
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


class SpecialRegTestGenerator:
    """Generate special register read test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_tid_tests(self) -> List[TestCase]:
        """Generate %tid.x read tests"""
        tests = []

        # Test: read tid.x and verify it equals thread ID
        # For thread 0, tid.x should be 0
        tests.append(TestCase(
            name="special_tid_x_000",
            category="special",
            ptx_code=[
                "mov.u32 r1, %tid.x",   # r1 = thread ID
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 0}  # Thread 0's tid.x = 0
        ))

        return tests

    def gen_ntid_tests(self) -> List[TestCase]:
        """Generate %ntid.x read tests"""
        tests = []

        # Test: read ntid.x (block dimension)
        # Default block_dim is (32, 1, 1) so ntid.x = 32
        tests.append(TestCase(
            name="special_ntid_x_000",
            category="special",
            ptx_code=[
                "mov.u32 r1, %ntid.x",  # r1 = block dimension x
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 32}  # Default block_dim[0] = 32
        ))

        return tests

    def gen_laneid_tests(self) -> List[TestCase]:
        """Generate %laneid read tests"""
        tests = []

        # Test: read laneid (thread ID within warp, 0-31)
        tests.append(TestCase(
            name="special_laneid_000",
            category="special",
            ptx_code=[
                "mov.u32 r1, %laneid",  # r1 = lane ID (tid % 32)
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 0}  # Thread 0's laneid = 0
        ))

        return tests

    def gen_ctaid_tests(self) -> List[TestCase]:
        """Generate %ctaid.x read tests"""
        tests = []

        # Test: read ctaid.x (block ID)
        tests.append(TestCase(
            name="special_ctaid_x_000",
            category="special",
            ptx_code=[
                "mov.u32 r1, %ctaid.x", # r1 = block ID x
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 0}  # First block's ctaid.x = 0
        ))

        return tests

    def gen_all_special_tests(self) -> List[TestCase]:
        """Generate all special register tests"""
        tests = []
        tests.extend(self.gen_tid_tests())
        tests.extend(self.gen_ntid_tests())
        tests.extend(self.gen_laneid_tests())
        tests.extend(self.gen_ctaid_tests())
        return tests


class AtomTestGenerator:
    """Generate atomic operation test cases

    Note on Thread Ordering:
    In the FRM (gpu_simulator.py), threads within a warp execute in order
    (thread 0, 1, 2, ..., 31) within a single instruction. This means thread 0
    is always the first to execute an atomic operation.

    On real hardware, thread execution order within a warp is not guaranteed.
    However, for FRM verification purposes, our deterministic ordering is
    acceptable and allows us to verify that:
    1. The atomic operation correctly returns the old value
    2. The atomic operation correctly updates memory

    These tests validate the FRM's atomic semantics, not real hardware timing.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_atom_add_tests(self) -> List[TestCase]:
        """Generate atom.add test cases

        Note: All 32 threads in a warp execute atomics, so we only verify
        that thread 0's returned value (old value) is correct.
        """
        tests = []

        # Test: atomic add - verify thread 0 gets the initial value as old
        tests.append(TestCase(
            name="atom_add_000",
            category="atom",
            ptx_code=[
                "mov.u32 r1, 0",         # Address 0
                "mov.u32 r2, 10",        # Value to add
                "atom.add.s32 r3, [r1], r2",  # r3 = old value (thread 0 is first)
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 100},  # Thread 0 sees initial value
            initial_memory={0: 100}
        ))

        # Test: atomic add with zero
        tests.append(TestCase(
            name="atom_add_001",
            category="atom",
            ptx_code=[
                "mov.u32 r1, 4",         # Address 4
                "mov.u32 r2, 0",         # Value to add (0)
                "atom.add.s32 r3, [r1], r2",
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 50},  # Returns old value
            initial_memory={4: 50}
        ))

        return tests

    def gen_atom_exch_tests(self) -> List[TestCase]:
        """Generate atom.exch test cases"""
        tests = []

        # Test: atomic exchange - verify thread 0 gets the initial value
        tests.append(TestCase(
            name="atom_exch_000",
            category="atom",
            ptx_code=[
                "mov.u32 r1, 0",         # Address 0
                "mov.u32 r2, 999",       # New value
                "atom.exch.b32 r3, [r1], r2",  # r3 = old value (thread 0 is first)
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 123},  # Thread 0 sees initial value
            initial_memory={0: 123}
        ))

        return tests

    def gen_atom_cas_tests(self) -> List[TestCase]:
        """Generate atom.cas test cases"""
        tests = []

        # Test: CAS success - thread 0 succeeds, sees initial value
        tests.append(TestCase(
            name="atom_cas_success_000",
            category="atom",
            ptx_code=[
                "mov.u32 r1, 0",         # Address
                "mov.u32 r2, 200",       # New value if match
                "mov.u32 r5, 100",       # Compare value (should match for thread 0)
                "atom.cas.b32 r3, [r1], r5, r2",  # CAS: if mem==r5, mem=r2
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 100},  # Thread 0 sees initial value (successful CAS)
            initial_memory={0: 100}
        ))

        # Test: CAS failure (compare doesn't match)
        tests.append(TestCase(
            name="atom_cas_fail_000",
            category="atom",
            ptx_code=[
                "mov.u32 r1, 0",
                "mov.u32 r2, 200",       # Would-be new value
                "mov.u32 r5, 999",       # Compare value (won't match)
                "atom.cas.b32 r3, [r1], r5, r2",
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 100},  # Thread 0 sees current value (failed CAS)
            initial_memory={0: 100}
        ))

        return tests

    def gen_all_atom_tests(self) -> List[TestCase]:
        """Generate all atomic operation tests"""
        tests = []
        tests.extend(self.gen_atom_add_tests())
        tests.extend(self.gen_atom_exch_tests())
        tests.extend(self.gen_atom_cas_tests())
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


class ParamConstTestGenerator:
    """Generate parameter and constant memory test cases

    Tests LD.PARAM and LD.CONST instructions which load from
    kernel parameter memory and constant memory respectively.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_ld_param_tests(self) -> List[TestCase]:
        """Generate ld.param test cases"""
        tests = []

        # Test: Load from parameter memory
        # Note: FRM uses param_memory dict which needs initialization
        tests.append(TestCase(
            name="ld_param_000",
            category="param",
            ptx_code=[
                "mov.u32 r1, 0",          # Address 0
                "ld.param.u32 r2, [r1]",  # Load from param[0]
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 0}  # Uninitialized param memory returns 0
        ))

        return tests

    def gen_ld_const_tests(self) -> List[TestCase]:
        """Generate ld.const test cases"""
        tests = []

        # Test: Load from constant memory
        tests.append(TestCase(
            name="ld_const_000",
            category="param",
            ptx_code=[
                "mov.u32 r1, 0",          # Address 0
                "ld.const.u32 r2, [r1]",  # Load from const[0]
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 0}  # Uninitialized const memory returns 0
        ))

        return tests

    def gen_all_param_const_tests(self) -> List[TestCase]:
        """Generate all parameter/constant memory tests"""
        tests = []
        tests.extend(self.gen_ld_param_tests())
        tests.extend(self.gen_ld_const_tests())
        return tests


class MembarTestGenerator:
    """Generate memory barrier test cases

    Tests MEMBAR instruction which enforces memory ordering.
    In the FRM, memory operations are instantaneous, so membar
    is essentially a no-op that doesn't cause errors.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_membar_tests(self) -> List[TestCase]:
        """Generate membar test cases"""
        tests = []

        # Test: membar.cta (CTA-level memory fence)
        tests.append(TestCase(
            name="membar_cta_000",
            category="membar",
            ptx_code=[
                "mov.u32 r1, 10",
                "membar.cta",             # CTA-level fence
                "add.s32 r2, r1, r1",     # r2 = 20
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 20}
        ))

        # Test: membar.gl (Global memory fence)
        tests.append(TestCase(
            name="membar_gl_000",
            category="membar",
            ptx_code=[
                "mov.u32 r1, 5",
                "membar.gl",              # Global fence
                "add.s32 r2, r1, r1",     # r2 = 10
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 10}
        ))

        # Test: membar.sys (System-level memory fence)
        tests.append(TestCase(
            name="membar_sys_000",
            category="membar",
            ptx_code=[
                "mov.u32 r1, 3",
                "membar.sys",             # System fence
                "add.s32 r2, r1, r1",     # r2 = 6
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 6}
        ))

        return tests

    def gen_all_membar_tests(self) -> List[TestCase]:
        """Generate all memory barrier tests"""
        return self.gen_membar_tests()


class BarSyncTestGenerator:
    """Generate barrier synchronization test cases

    Note on Single-Warp FRM:
    In a single-warp FRM with block_dim=(32,1,1), all 32 threads arrive at
    the barrier in the same instruction execution, so barriers release
    immediately. This tests that:
    1. BAR.SYNC instruction is correctly decoded and executed
    2. Computation before/after barrier is correct
    3. Different barrier IDs work

    True multi-warp synchronization testing requires RTL simulation.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def gen_bar_sync_basic_tests(self) -> List[TestCase]:
        """Generate basic bar.sync tests"""
        tests = []

        # Test: basic bar.sync with barrier 0
        # Compute before barrier, then compute after barrier
        tests.append(TestCase(
            name="bar_sync_000",
            category="sync",
            ptx_code=[
                "mov.u32 r1, 10",         # r1 = 10
                "mov.u32 r2, 20",         # r2 = 20
                "add.s32 r3, r1, r2",     # r3 = 30 (before barrier)
                "bar.sync 0",             # Barrier 0
                "add.s32 r4, r3, r1",     # r4 = 40 (after barrier)
                "exit"
            ],
            initial_regs={},
            expected_regs={3: 30, 4: 40}
        ))

        # Test: bar.sync with different barrier ID
        tests.append(TestCase(
            name="bar_sync_001",
            category="sync",
            ptx_code=[
                "mov.u32 r1, 5",
                "bar.sync 1",             # Barrier 1
                "add.s32 r2, r1, r1",     # r2 = 10
                "bar.sync 2",             # Barrier 2
                "add.s32 r3, r2, r2",     # r3 = 20
                "exit"
            ],
            initial_regs={},
            expected_regs={2: 10, 3: 20}
        ))

        # Test: multiple barriers in sequence
        tests.append(TestCase(
            name="bar_sync_002",
            category="sync",
            ptx_code=[
                "mov.u32 r1, 1",
                "bar.sync 0",
                "add.s32 r1, r1, r1",     # r1 = 2
                "bar.sync 0",
                "add.s32 r1, r1, r1",     # r1 = 4
                "bar.sync 0",
                "add.s32 r1, r1, r1",     # r1 = 8
                "exit"
            ],
            initial_regs={},
            expected_regs={1: 8}
        ))

        return tests

    def gen_all_bar_sync_tests(self) -> List[TestCase]:
        """Generate all barrier synchronization tests"""
        return self.gen_bar_sync_basic_tests()


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
        # Write initial memory (for FRM initialization)
        if test.initial_memory:
            for addr, val in test.initial_memory.items():
                f.write(f"init_mem[{addr:08x}]={val:08x}\n")
        # Write expected registers
        for reg, val in test.expected_regs.items():
            f.write(f"r{reg}={val:08x}\n")
        # Write expected memory
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

    # Special register tests
    special_gen = SpecialRegTestGenerator(seed=42)
    special_tests = special_gen.gen_all_special_tests()
    print(f"Generated {len(special_tests)} Special Register tests")

    success_count = 0
    for test in special_tests:
        if write_test_case(test, output_dir / "special"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(special_tests)} Special Register tests")

    # Atomic tests
    atom_gen = AtomTestGenerator(seed=42)
    atom_tests = atom_gen.gen_all_atom_tests()
    print(f"Generated {len(atom_tests)} Atomic tests")

    success_count = 0
    for test in atom_tests:
        if write_test_case(test, output_dir / "atom"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(atom_tests)} Atomic tests")

    # Sync (BAR.SYNC) tests
    sync_gen = BarSyncTestGenerator(seed=42)
    sync_tests = sync_gen.gen_all_bar_sync_tests()
    print(f"Generated {len(sync_tests)} Sync (BAR.SYNC) tests")

    success_count = 0
    for test in sync_tests:
        if write_test_case(test, output_dir / "sync"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(sync_tests)} Sync tests")

    # Param/Const memory tests
    param_gen = ParamConstTestGenerator(seed=42)
    param_tests = param_gen.gen_all_param_const_tests()
    print(f"Generated {len(param_tests)} Param/Const tests")

    success_count = 0
    for test in param_tests:
        if write_test_case(test, output_dir / "param"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(param_tests)} Param/Const tests")

    # Membar tests
    membar_gen = MembarTestGenerator(seed=42)
    membar_tests = membar_gen.gen_all_membar_tests()
    print(f"Generated {len(membar_tests)} Membar tests")

    success_count = 0
    for test in membar_tests:
        if write_test_case(test, output_dir / "membar"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(membar_tests)} Membar tests")

    # SFU (Special Function Unit) tests
    sfu_gen = SFUTestGenerator(seed=42)
    sfu_tests = sfu_gen.gen_all_sfu_tests()
    print(f"Generated {len(sfu_tests)} SFU tests")

    success_count = 0
    for test in sfu_tests:
        if write_test_case(test, output_dir / "sfu"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(sfu_tests)} SFU tests")

    # FP16 tests
    fp16_gen = FP16TestGenerator(seed=42)
    fp16_tests = fp16_gen.gen_all_fp16_tests()
    print(f"Generated {len(fp16_tests)} FP16 tests")

    success_count = 0
    for test in fp16_tests:
        if write_test_case(test, output_dir / "fp16"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(fp16_tests)} FP16 tests")

    # Generate FP64 tests
    fp64_gen = FP64TestGenerator(seed=42)
    fp64_tests = fp64_gen.gen_all_fp64_tests()
    print(f"Generated {len(fp64_tests)} FP64 tests")

    success_count = 0
    for test in fp64_tests:
        if write_test_case(test, output_dir / "fp64"):
            success_count += 1
    print(f"Successfully wrote {success_count}/{len(fp64_tests)} FP64 tests")

    # Summary
    total_tests = (len(alu_tests) + len(fp32_tests) + len(mem_tests) + len(branch_tests) +
                   len(div_tests) + len(special_tests) + len(atom_tests) + len(sync_tests) +
                   len(param_tests) + len(membar_tests) + len(sfu_tests) + len(fp16_tests) + len(fp64_tests))
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
    parser.add_argument("--gen", choices=["alu", "fp32", "memory", "branch", "div", "special", "atom", "sync", "param", "membar", "sfu", "fp16", "fp64", "all"],
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
        elif args.gen == "special":
            gen = SpecialRegTestGenerator(seed=args.seed)
            tests = gen.gen_all_special_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "special"))
            print(f"Generated {success}/{len(tests)} Special Register tests")
        elif args.gen == "atom":
            gen = AtomTestGenerator(seed=args.seed)
            tests = gen.gen_all_atom_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "atom"))
            print(f"Generated {success}/{len(tests)} Atomic tests")
        elif args.gen == "sync":
            gen = BarSyncTestGenerator(seed=args.seed)
            tests = gen.gen_all_bar_sync_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "sync"))
            print(f"Generated {success}/{len(tests)} Sync tests")
        elif args.gen == "param":
            gen = ParamConstTestGenerator(seed=args.seed)
            tests = gen.gen_all_param_const_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "param"))
            print(f"Generated {success}/{len(tests)} Param/Const tests")
        elif args.gen == "membar":
            gen = MembarTestGenerator(seed=args.seed)
            tests = gen.gen_all_membar_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "membar"))
            print(f"Generated {success}/{len(tests)} Membar tests")
        elif args.gen == "sfu":
            gen = SFUTestGenerator(seed=args.seed)
            tests = gen.gen_all_sfu_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "sfu"))
            print(f"Generated {success}/{len(tests)} SFU tests")
        elif args.gen == "fp16":
            gen = FP16TestGenerator(seed=args.seed)
            tests = gen.gen_all_fp16_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "fp16"))
            print(f"Generated {success}/{len(tests)} FP16 tests")
        elif args.gen == "fp64":
            gen = FP64TestGenerator(seed=args.seed)
            tests = gen.gen_all_fp64_tests()
            success = sum(1 for t in tests if write_test_case(t, OUTPUT_DIR / "fp64"))
            print(f"Generated {success}/{len(tests)} FP64 tests")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
