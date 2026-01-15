# RalphGPU Expert Review Documentation Package

**Version:** 0.9.0
**Date:** 2026-01-15
**Status:** Pre-Release / Expert Review

---

## 1. Executive Summary

RalphGPU is a synthesizeable, open-source GPU IP core designed to be compatible with NVIDIA's PTX (Parallel Thread Execution) ISA version 8.5+. It implements a scalable Streaming Multiprocessor (SM) architecture capable of executing CUDA-like kernels. The design focuses on modularity, readability, and hardware fidelity to modern GPU microarchitectures, including support for Tensor Cores (mixed-precision matrix multiplication) and advanced SIMT control flow.

**Key Features:**
*   **Scalable Architecture:** Configurable number of SMs (default: 2), Warps per SM (default: 4), and Shared Memory size (default: 16KB).
*   **Full PTX 8.5+ Decode:** Supports a vast subset of the PTX ISA, including integer, floating-point (FP32/FP16), tensor, and control flow instructions.
*   **SIMT Execution Model:** Hardware-managed Warps (32 threads) with automatic divergence and reconvergence handling using a hardware stack.
*   **Tensor Acceleration:** Integrated Tensor Cores supporting WMMA (Warp Matrix Multiply Accumulate) and MMA instructions for AI/ML workloads (FP16/INT8).
*   **Standard Interfaces:** AXI4 Master interface for global memory and a simplified CSR slave interface for host control.

---

## 2. Architectural Overview

### 2.1 Top-Level Hierarchy
The design follows a standard GPU hierarchy:
1.  **RalphGPU Top:** Contains multiple Streaming Multiprocessors (SMs), a global memory arbiter, and a Command Processor/Dispatcher.
2.  **Streaming Multiprocessor (SM):** The core compute unit. It contains the Fetch/Decode logic, Register File, Shared Memory, and execution units (ALU, FPU, Tensor Core).
3.  **Warp:** The fundamental unit of execution (32 threads). The SM schedules instructions at the Warp level.

**Block Diagram:**
```
[ Host Interface (CSR) ] <---> [ Command Processor / Global Scheduler ]
                                         |
          +------------------------------+------------------------------+
          |                              |                              |
  [ Streaming Multiprocessor 0 ]  [ Streaming Multiprocessor 1 ]  [ ... ]
          |                              |                              |
          +--------------+---------------+------------------------------+
                         |
               [ Memory Interconnect (AXI4) ]
                         |
                 [ Global Memory (DRAM) ]
```

### 2.2 Streaming Multiprocessor (SM) Detail
Each SM is an independent core containing:
*   **Warp Scheduler:** Selects a ready Warp to issue an instruction (Round-robin policy).
*   **Instruction Unit:** Fetch (from I-Cache/Memory) and Decode (PTX decoder).
*   **Register File:** Partitioned 32x32-bit registers per thread (total 4KB per Warp).
*   **Shared Memory:** 16KB-96KB scratchpad, banked (32 banks) for high bandwidth.
*   **Execution Datapath:**
    *   **SIMD ALU:** 32-lane Integer/Bitwise unit.
    *   **SIMD FPU:** 32-lane Single Precision (IEEE 754) unit.
    *   **Tensor Core:** 16x16x16 Matrix Multiply Unit.
    *   **SFU:** Special Function Unit (planned/partial).
    *   **LSU:** Load/Store Unit handling global/shared memory access.

---

## 3. Microarchitecture Description

### 3.1 Pipeline Stages
The SM operates on a 6-stage equivalent pipeline:
1.  **Fetch:** Fetch instruction for the active Warp from Instruction Memory.
2.  **Decode:** Decode PTX instruction, generating control signals and operand addresses.
3.  **Issue/Read:** Arbitrate for execution units and read operands from the Register File.
4.  **Execute:** Operation performance (1 cycle for ALU, Multi-cycle for FPU/Tensor/Memory).
5.  **Memory (Optional):** Access Shared Memory or request Global Memory.
6.  **Writeback:** Write results back to the Register File.

### 3.2 Control Flow Unit (SIMT)
To handle SIMT divergence (where threads in a Warp take different paths):
*   **Mechanism:** Hardware Divergence Stack.
*   **Operation:** When a branch diverges (some threads taken, some not), the hardware pushes the current Active Mask and the Reconvergence PC onto a stack. It then executes one path (modifying the Active Mask). Upon reaching the Reconvergence PC, it pops the stack to restore the mask or execute the other path.
*   **Support:** Handles `bra`, `call`, `ret`, and predicated execution (`@p bra`).

### 3.3 Memory Subsystem
*   **Register File:**
    *   **Structure:** Per-Warp storage.
    *   **Ports:** 3 Read / 1 Write per lane to support FMA (Fused Multiply-Add) operations.
    *   **Banking:** Implementation uses banking to emulate multi-port behavior efficiently in hardware.
*   **Shared Memory:**
    *   **Architecture:** 32 Banks (aligned with 32 threads/Warp).
    *   **Conflict Handling:** Logic detects bank conflicts (multiple threads accessing the same bank with different row addresses). *Note: Current RTL implementation assumes software conflict avoidance or stalls on conflict (simplified).*
    *   **Access:** Supports 32-bit, 64-bit, and 128-bit vector loads/stores.

### 3.4 Execution Units

#### 3.4.1 ALU (Integer Unit)
*   **Width:** 32-bit.
*   **Operations:** Add/Sub (with carry), Mul, Mad, Div/Rem, Logic (AND/OR/XOR/NOT), Shift, Min/Max.
*   **Extended Ops:** Population Count (`popc`), Count Leading Zeros (`clz`), Bit Field Extract/Insert (`bfe`/`bfi`), Byte Permute (`prmt`).

#### 3.4.2 FPU (Floating Point Unit)
*   **Compliance:** Simplified IEEE 754 Single Precision (FP32).
*   **Operations:** Add, Sub, Mul, FMA, Min, Max, Abs, Neg.
*   **Limitation:** Denormal handling is Flush-to-Zero (FTZ). Rounding modes are simplified (Round-to-Nearest default). Division is iterative/approximated.

#### 3.4.3 Tensor Core
*   **Function:** Accelerates Matrix Multiplication (`D = A*B + C`).
*   **Configuration:** 16x16x16 Mixed Precision.
*   **Inputs:** FP16 (A, B matrices) or INT8.
*   **Accumulator:** FP32 or INT32 (C, D matrices).
*   **Implementation:** Iterative calculation. A Warp collaboratively loads fragments of matrices into the Tensor Core, which then computes the product over multiple cycles.

---

## 4. Supported Instruction Set (PTX 8.5+)

The Decoder (`decoder.v`) and Execution Units support the following instruction classes:

| Category | Instructions |
| :--- | :--- |
| **Integer Arithmetic** | `add`, `sub`, `mul`, `mad`, `div`, `rem`, `abs`, `neg`, `min`, `max` |
| **Logic & Bitwise** | `and`, `or`, `xor`, `not`, `shl`, `shr`, `popc`, `clz`, `bfind`, `brev`, `bfe`, `bfi`, `prmt` |
| **Comparison** | `setp` (Set Predicate), `selp` (Select based on predicate), `slct` |
| **Floating Point** | `add.f32`, `sub.f32`, `mul.f32`, `fma.f32`, `div.f32` (approx), `neg`, `abs`, `min`, `max` |
| **Data Movement** | `ld.global`, `st.global`, `ld.shared`, `st.shared`, `mov` |
| **Control Flow** | `bra` (Branch), `call`, `ret`, `exit`, `bar.sync` (Barrier) |
| **Tensor / MMA** | `wmma.load`, `wmma.store`, `wmma.mma` (Warp Matrix Multiply Accumulate) |
| **Atomic** | `atom.add`, `atom.min`, `atom.max`, `atom.cas`, `atom.exch` |
| **Warp Primitives** | `shfl.sync` (Shuffle), `vote.sync` (Ballot/Any/All) |

---

## 5. Interface Specification

### 5.1 Host Control (CSR)
Mapped to AXI4-Lite or simple Register Interface.
*   `0x000` **GPU_STATUS**: Busy, Done, IRQ status.
*   `0x004` **GPU_CONTROL**: Start Kernel, Reset.
*   `0x008` **KERNEL_PC**: Instruction Memory Start Address.
*   `0x00C` - `0x020`: Grid and Block Dimensions (`grid_dim`, `block_dim`).

### 5.2 Global Memory (AXI4 Master)
Standard AXI4 interface for instruction fetch and global load/store.
*   **ID Width:** Configurable (Default 4).
*   **Data Width:** 32-bit or 64-bit.
*   **Burst:** Supported for cache-line fills (if cache is enabled) or block loads.

---

## 6. Verification & Validation

The codebase includes a suite of Verilog testbenches (`tb/`) verifying module-level functionality:

*   **`tb_alu.v`**: Exhaustive test of integer arithmetic and logic ops.
*   **`tb_fpu.v`**: Corner case testing for FP add/mul/fma (Zero, Inf, NaN, Normal).
*   **`tb_decoder.v`**: Instruction coverage test to ensure proper decoding of opcode/operands.
*   **`tb_warp_scheduler.v`**: Verification of round-robin scheduling and warp state transitions.
*   **`tb_ralph_gpu.v`**: Top-level integration test running small kernels (e.g., Vector Add).

---

## 7. Current Limitations & Future Work

1.  **FPU Precision:** The current FPU is a simplified model optimized for area/speed trade-offs in FPGA prototyping. It is not fully IEEE 754 compliant regarding all rounding modes and exceptions.
2.  **Caches:** L1/L2 Caches are currently bypassed; Global Memory access goes directly to the AXI bus. Implementing a coherent cache hierarchy is the next major milestone.
3.  **Complex Math:** SFU instructions (`sin`, `cos`, `exp`) are currently placeholders or implemented via software emulation (compiler-side decomposition) rather than dedicated hardware units.
4.  **Performance:** No extensive pipelining optimizations (like forwarding paths) are currently implemented to resolve hazards; the scheduler relies on simple scoreboarding or stalls.

---
**End of Document**
