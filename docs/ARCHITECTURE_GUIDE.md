# RalphGPU Architecture Guide (Hopper/Blackwell-Class)

## 1. Overview
RalphGPU is a high-performance, CUDA/PTX-compatible GPGPU IP designed for parallel computing workloads. It features a scalable architecture centered around multiple Streaming Multiprocessors (SMs), an advanced memory subsystem with virtual memory support, and multi-generational Tensor Core acceleration.

The architecture targets compatibility with **PTX ISA 8.5+** and implements features inspired by NVIDIA's Hopper and Blackwell architectures.

## 2. Top-Level Architecture (ralph_gpu_top)

The top-level module integrates compute cores with a high-bandwidth memory subsystem, system interfaces, and control logic.

### 2.1 Key Components
*   **Streaming Multiprocessors (SMs)**: The primary compute engines. Scalable from 1 to 128 SMs.
*   **Global Interconnect**: A round-robin arbiter for instruction fetch and data memory access among SMs.
*   **Memory Controller (HBM/AXI4)**: Manages global memory access via a 32-bit or 64-bit AXI4 master interface.
*   **Virtual Memory Subsystem**:
    *   **L1 TLB**: Per-SM, 32-128 entries.
    *   **L2 TLB**: Shared among SMs, typically 512-1024 entries.
    *   **Hardware Page Table Walker (PTW)**: Automatically handles TLB misses by traversing 3-level page tables in global memory.
    *   **Address Space**: 48-bit Virtual, 40-bit Physical.
*   **WGMMA / TMEM Engine**: Shared asynchronous matrix-multiply-accumulate engine for large-scale tensor operations.
*   **CSR Unit**: Memory-mapped registers for host control and status monitoring.

### 2.2 System Interfaces
*   **CSR Interface (APB-like)**: 12-bit address, 32-bit data for configuration.
*   **Instruction Memory (IMEM)**: 64-bit interface for fetching instructions.
*   **Global Memory (AXI4)**: High-bandwidth master interface for data loads, stores, and atomics.

---

## 3. Streaming Multiprocessor V2 (SM)

The SM is the core processing unit, implementing a 5-stage dual-issue pipeline with advanced scheduling logic.

### 3.1 Pipeline Stages
1.  **Fetch**: Instruction fetch from I-Cache with round-robin warp arbitration.
2.  **Decode / Schedule**:
    *   **Pre-decoding**: Instruction type and dependency analysis.
    *   **Scoreboard**: Tracks register hazards (WAW, RAW).
    *   **Multi-Scheduler (Blackwell-style)**: Up to 4 parallel schedulers (configurable) using Greedy-Then-Oldest (GTO) policy.
3.  **Issue**: Multi-way issue logic (typically 2-way or 4-way) capable of issuing instructions to independent execution units.
4.  **Execute**: Parallel execution across specialized functional units.
5.  **Writeback**: Committing results to the Register File and clearing scoreboard entries.

### 3.2 Execution Units
*   **SIMD ALU (32-lane)**: Integer arithmetic, logic, and bitwise operations.
*   **SIMD FPU (32-lane)**: Supports IEEE 754 floating-point (FP32, FP64, BF16, FP16).
*   **SFU (Special Function Unit)**: Transcendental functions (sin, cos, rcp, sqrt, lg2, ex2) via table-lookup and interpolation.
*   **Tensor Core (MMA)**:
    *   **WMMA**: Warp-level Matrix Multiply-Accumulate (Hopper).
    *   **WGMMA**: Warp-Group Matrix Multiply-Accumulate (Hopper/Blackwell).
    *   **TCGEN05**: 5th-gen Tensor Core (Blackwell) with per-thread async MMA and TMEM (Tensor Memory) support.
*   **LSU (Load/Store Unit)**: Manages memory access, coalescing requests, and interfacing with L1D/SMEM.
*   **Branch Unit**: Handles control flow, branch reconvergence (via sync stack), and prediction.

### 3.3 Warp Scheduling & Execution Model
*   **Warps**: Threads are grouped into 32-thread Warps.
*   **Dual Issue**: Capable of issuing two independent instructions from the same warp or different warps depending on resource availability.
*   **SIMT Execution**: 32-lane SIMD execution for all thread-level instructions.

---

## 4. Memory Subsystem

### 4.1 L1 Data Cache
*   **Size**: 16KB - 128KB (Configurable).
*   **Organization**: 4-way set associative, 128-byte line size.
*   **Features**: Supports cache hints (ca, cg, cs, lu, cv) and write-back/write-through policies.
*   **Bypass Mode**: High-performance mode for direct memory access when latency is more critical than locality.

### 4.2 Shared Memory (SMEM)
*   **Size**: 16KB - 256KB per SM.
*   **Organization**: 32 banks, word-interleaved to avoid bank conflicts.
*   **Access**: Low-latency communication between threads within a block.

### 4.3 Tensor Memory (TMEM) - Blackwell Feature
*   **Size**: 256KB per SM.
*   **Function**: Dedicated high-speed accumulator memory for TCGEN05 tensor operations.

---

## 5. Control & Status Registers (CSR)

Base Address: Configurable (typically 0x000 in local space).

| Offset | Register Name | Access | Description |
|--------|---------------|--------|-------------|
| 0x000  | GPU_STATUS    | RO     | bit 0: Busy, bit 1: Error, bits 31:16: Version |
| 0x004  | GPU_CONTROL   | RW     | bit 0: START_KERNEL, bit 1: RESET, bit 2: INTERRUPT_EN |
| 0x008  | KERNEL_PC     | RW     | 32-bit entry point address for the kernel |
| 0x00C  | GRID_DIM_X    | RW     | Number of thread blocks in X dimension |
| 0x010  | GRID_DIM_Y    | RW     | Number of thread blocks in Y dimension |
| 0x014  | GRID_DIM_Z    | RW     | Number of thread blocks in Z dimension |
| 0x018  | BLOCK_DIM_X   | RW     | Threads per block in X dimension |
| 0x01C  | BLOCK_DIM_Y   | RW     | Threads per block in Y dimension |
| 0x020  | BLOCK_DIM_Z   | RW     | Threads per block in Z dimension |
| 0x024  | ERROR_STATUS  | RO     | Error code (0: None, 1: Illegal Inst, 2: Mem Fault, etc.) |
| 0x028  | ERR_WARP_MASK | RO     | Mask of warps that encountered the error |
| 0x02C  | ERROR_INFO    | RO     | Additional debug information for the error (e.g., faulting address) |

---

## 6. Power & Clock Management
*   **Gated Clocks**: Per-SM and per-FU clock gating for reduced idle power.
*   **DVFS**: Support for Dynamic Voltage and Frequency Scaling interfaces.
