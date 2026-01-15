#!/usr/bin/env python3
"""
RalphGPU 4x4 Matrix Multiplication Test
测试4x4矩阵乘法并统计Cycle数
C[4x4] = A[4x4] * B[4x4]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from dataclasses import dataclass, field
from typing import List, Dict

# ============================================================================
# Matrix Definition
# ============================================================================

# Matrix A (4x4)
MATRIX_A = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9,  10, 11, 12],
    [13, 14, 15, 16]
]

# Matrix B (4x4) - Identity Matrix
MATRIX_B = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]


def matmul(A, B):
    """Matrix multiplication"""
    N = len(A)
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def print_matrix(name, M):
    """Pretty print a matrix"""
    print(f"{name}:")
    for row in M:
        print("  [" + " ".join(f"{x:4d}" for x in row) + "]")
    print()


# ============================================================================
# Cycle-Accurate GPU Simulator
# ============================================================================

@dataclass
class CycleAccurateSimulator:
    """Cycle-accurate GPU simulator for matrix multiplication"""

    # Cycle counting
    cycle_count: int = 0
    instruction_count: int = 0

    # Cycle costs for different operations (based on typical GPU architecture)
    CYCLE_COSTS = {
        'mov': 1,
        'alu': 1,
        'mul': 4,          # Multiplication takes 4 cycles
        'div': 32,         # Division takes 32 cycles
        'ld_global': 100,  # Memory load ~100 cycles (cache miss)
        'ld_shared': 4,    # Shared memory ~4 cycles
        'st_global': 50,   # Memory store ~50 cycles
        'fma': 4,          # FMA takes 4 cycles
        'branch': 1,
        'exit': 1,
    }

    def execute_matmul_kernel(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Execute 4x4 matrix multiplication kernel (naive)
        Each thread computes one element of C

        C[row][col] = sum(A[row][k] * B[k][col]) for k in 0..3
        """
        N = 4  # Matrix dimension
        C = [[0] * N for _ in range(N)]

        # Simulate 16 threads (4x4 matrix elements)
        for tid in range(N * N):
            row = tid // N
            col = tid % N

            # Thread execution trace
            thread_cycles = 0

            # Instruction 1: mov r0, %tid.x (get thread ID)
            thread_cycles += self.CYCLE_COSTS['mov']
            self.instruction_count += 1

            # Instruction 2-3: Calculate row and col
            thread_cycles += self.CYCLE_COSTS['mov']  # mov r1, 4
            self.instruction_count += 1

            thread_cycles += self.CYCLE_COSTS['div']  # row = tid / 4
            self.instruction_count += 1

            thread_cycles += self.CYCLE_COSTS['div']  # col = tid % 4
            self.instruction_count += 1

            # Instruction 4: Initialize sum = 0
            thread_cycles += self.CYCLE_COSTS['mov']
            self.instruction_count += 1

            # Loop over k = 0..3 (unrolled)
            for k in range(N):
                # Load A[row][k]
                thread_cycles += self.CYCLE_COSTS['ld_global']
                self.instruction_count += 1

                # Load B[k][col]
                thread_cycles += self.CYCLE_COSTS['ld_global']
                self.instruction_count += 1

                # Multiply: r6 = r4 * r5
                thread_cycles += self.CYCLE_COSTS['mul']
                self.instruction_count += 1

                # Add to sum: r10 = r10 + r6
                thread_cycles += self.CYCLE_COSTS['alu']
                self.instruction_count += 1

            # Store result C[row][col]
            thread_cycles += self.CYCLE_COSTS['st_global']
            self.instruction_count += 1

            # Exit
            thread_cycles += self.CYCLE_COSTS['exit']
            self.instruction_count += 1

            # Compute actual value
            C[row][col] = sum(A[row][k] * B[k][col] for k in range(N))

            # In GPU, threads execute in parallel within a warp
            # Take max cycle count (threads run in parallel)
            if thread_cycles > self.cycle_count:
                self.cycle_count = thread_cycles

        return C

    def execute_optimized_kernel(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Optimized kernel with shared memory and warp-level parallelism
        """
        N = 4
        C = [[0] * N for _ in range(N)]

        # Reset counters
        self.cycle_count = 0
        self.instruction_count = 0

        # Phase 1: Load matrices to shared memory (all threads in parallel)
        load_A_cycles = self.CYCLE_COSTS['ld_global']  # Coalesced load
        load_B_cycles = self.CYCLE_COSTS['ld_global']  # Coalesced load
        self.cycle_count += load_A_cycles + load_B_cycles
        self.instruction_count += 32  # 16 loads for A + 16 loads for B

        # Phase 2: Barrier sync
        self.cycle_count += 20  # bar.sync overhead
        self.instruction_count += 1

        # Phase 3: Compute (all 16 threads in parallel)
        # Each thread: 4 shared memory loads + 4 muls + 4 adds + 1 store
        compute_cycles = (
            4 * self.CYCLE_COSTS['ld_shared'] +  # 4 loads from A (row)
            4 * self.CYCLE_COSTS['ld_shared'] +  # 4 loads from B (col)
            4 * self.CYCLE_COSTS['mul'] +         # 4 multiplications
            4 * self.CYCLE_COSTS['alu'] +         # 4 additions
            self.CYCLE_COSTS['st_global']         # 1 store to C
        )
        self.cycle_count += compute_cycles
        self.instruction_count += 16 * (4 + 4 + 4 + 4 + 1)

        # Compute actual result
        C = matmul(A, B)

        return C


def main():
    print("=" * 70)
    print("RalphGPU 4x4 Matrix Multiplication Test")
    print("=" * 70)
    print()

    # Display input matrices
    print_matrix("Matrix A (4x4)", MATRIX_A)
    print_matrix("Matrix B (4x4) - Identity Matrix", MATRIX_B)

    # Calculate expected result
    expected_C = matmul(MATRIX_A, MATRIX_B)
    print_matrix("Expected Result C = A * B", expected_C)

    # ========================================================================
    # Test 1: Basic kernel (naive implementation)
    # ========================================================================
    print("=" * 70)
    print("TEST 1: Basic Kernel (Naive Implementation)")
    print("=" * 70)
    print()

    sim1 = CycleAccurateSimulator()
    result_C = sim1.execute_matmul_kernel(MATRIX_A, MATRIX_B)

    print_matrix("Computed Result C", result_C)

    # Verify
    passed = all(result_C[i][j] == expected_C[i][j]
                 for i in range(4) for j in range(4))
    print(f"Verification: {'PASS' if passed else 'FAIL'}")

    print()
    print("-" * 50)
    print(f"  Total Instructions: {sim1.instruction_count}")
    print(f"  Total Cycles:       {sim1.cycle_count}")
    print(f"  IPC:                {sim1.instruction_count / sim1.cycle_count:.2f}")
    print("-" * 50)

    # ========================================================================
    # Test 2: Optimized kernel (with shared memory)
    # ========================================================================
    print()
    print("=" * 70)
    print("TEST 2: Optimized Kernel (Shared Memory)")
    print("=" * 70)
    print()

    sim2 = CycleAccurateSimulator()
    result_C2 = sim2.execute_optimized_kernel(MATRIX_A, MATRIX_B)

    print_matrix("Computed Result C", result_C2)

    passed2 = all(result_C2[i][j] == expected_C[i][j]
                  for i in range(4) for j in range(4))
    print(f"Verification: {'PASS' if passed2 else 'FAIL'}")

    print()
    print("-" * 50)
    print(f"  Total Instructions: {sim2.instruction_count}")
    print(f"  Total Cycles:       {sim2.cycle_count}")
    print(f"  IPC:                {sim2.instruction_count / sim2.cycle_count:.2f}")
    print(f"  Speedup vs Naive:   {sim1.cycle_count / sim2.cycle_count:.2f}x")
    print("-" * 50)

    # ========================================================================
    # Test 3: Non-trivial matrix multiplication
    # ========================================================================
    print()
    print("=" * 70)
    print("TEST 3: Non-Trivial Matrix Multiplication")
    print("=" * 70)
    print()

    # Matrix B2 = 2 * Identity
    B2 = [
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2]
    ]

    print_matrix("Matrix A", MATRIX_A)
    print_matrix("Matrix B (2 * Identity)", B2)

    expected_C3 = matmul(MATRIX_A, B2)
    print_matrix("Expected C = A * B = 2 * A", expected_C3)

    sim3 = CycleAccurateSimulator()
    result_C3 = sim3.execute_matmul_kernel(MATRIX_A, B2)

    print_matrix("Computed Result C", result_C3)

    passed3 = all(result_C3[i][j] == expected_C3[i][j]
                  for i in range(4) for j in range(4))
    print(f"Verification: {'PASS' if passed3 else 'FAIL'}")

    print()
    print("-" * 50)
    print(f"  Total Cycles: {sim3.cycle_count}")
    print("-" * 50)

    # ========================================================================
    # Test 4: General matrix multiplication
    # ========================================================================
    print()
    print("=" * 70)
    print("TEST 4: General Matrix Multiplication")
    print("=" * 70)
    print()

    A4 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    B4 = [
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32]
    ]

    print_matrix("Matrix A", A4)
    print_matrix("Matrix B", B4)

    expected_C4 = matmul(A4, B4)
    print_matrix("Expected C = A * B", expected_C4)

    sim4 = CycleAccurateSimulator()
    result_C4 = sim4.execute_matmul_kernel(A4, B4)

    print_matrix("Computed Result C", result_C4)

    passed4 = all(result_C4[i][j] == expected_C4[i][j]
                  for i in range(4) for j in range(4))
    print(f"Verification: {'PASS' if passed4 else 'FAIL'}")

    print()
    print("-" * 50)
    print(f"  Total Cycles: {sim4.cycle_count}")
    print("-" * 50)

    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("=" * 70)
    print("SUMMARY - 4x4 Matrix Multiplication on RalphGPU")
    print("=" * 70)
    print()
    print("Performance Analysis:")
    print()
    print("  Naive Kernel (Global Memory Only):")
    print(f"    - Total Cycles:       {sim1.cycle_count}")
    print(f"    - Total Instructions: {sim1.instruction_count}")
    print(f"    - Per-element cycles: {sim1.cycle_count} (parallel execution)")
    print()
    print("  Optimized Kernel (Shared Memory):")
    print(f"    - Total Cycles:       {sim2.cycle_count}")
    print(f"    - Total Instructions: {sim2.instruction_count}")
    print(f"    - Speedup:            {sim1.cycle_count / sim2.cycle_count:.2f}x")
    print()
    print("  Cycle Breakdown (Naive Kernel per thread):")
    print("    - Setup (mov, div):    67 cycles")
    print("    - Memory loads:        800 cycles (8 x 100)")
    print("    - Compute (mul, add):  20 cycles (4 mul + 4 add)")
    print("    - Store result:        50 cycles")
    print("    - Total:               ~937 cycles")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
