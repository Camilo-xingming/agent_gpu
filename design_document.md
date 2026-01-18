# RalphGPU Design Document

## 1. Overview
RalphGPU is a CUDA/PTX-compatible General Purpose GPU (GPGPU) IP designed for high-performance parallel computing. It features a scalable architecture centered around Streaming Multiprocessors (SMs) with NVIDIA Hopper-class features, including asynchronous tensor operations, advanced memory management, and dual-issue scheduling.

## 2. Top-Level Architecture (`ralph_gpu_top`)

The top-level module integrates the compute cores with the memory subsystem and system interfaces.

### 2.1 Parameters
*   **NUM_SM**: Number of Streaming Multiprocessors (Default: 2)
*   **AXI_DATA_WIDTH**: 32-bit (Global Memory Data)
*   **AXI_ADDR_WIDTH**: 32-bit (Global Memory Address)

### 2.2 Interfaces
*   **Clock/Reset**: `clk`, `rst_n`
*   **CSR Interface**: 32-bit APB-like interface (`csr_wr_en`, `csr_addr`, `csr_wr_data`, `csr_rd_data`)
*   **Instruction Memory**: Read-only interface with 64-bit data width (fetching 8-byte cache lines).
*   **Global Memory**: AXI4 Master interface for data load/store.
*   **Interrupts**: `irq_kernel_done`

### 2.3 Control Status Registers (CSR)
The GPU is controlled via a memory-mapped register file:
*   `0x000` **GPU_STATUS**: Read-only (Bit 0: Busy, Bit 1: Ready)
*   `0x004` **GPU_CONTROL**: Bit 0: Kernel Start
*   `0x008` **KERNEL_PC**: Kernel Start Program Counter
*   `0x00C` - `0x014`: **GRID_DIM** (X, Y, Z)
*   `0x018` - `0x020`: **BLOCK_DIM** (X, Y, Z)

### 2.4 Sub-modules & Shared Resources
*   **Memory Controller (HBM)**: FR-FCFS scheduling with realistic DRAM timing parameters (tCL, tRCD, etc.).
*   **Memory QoS**: Per-SM bandwidth allocation and priority arbitration.
*   **TLB Enhanced**: Two-level TLB with hardware page walker.
*   **WGMMA Tile Engine**: Shared asynchronous tensor operation engine.
*   **Arbitration**: Round-robin arbitration for Instruction Memory and AXI bus access among SMs.

---

## 3. Streaming Multiprocessor V2 (`streaming_multiprocessor_v2`)

The core compute unit implementing a 5-stage pipeline with dual-issue capability.

### 3.1 Key Parameters
*   **NUM_WARPS**: 8 Warps per SM (configurable)
*   **NUM_LANES**: 32 Threads per Warp (SIMD width)
*   **DATA_WIDTH**: 32-bit

### 3.2 Pipeline Stages
1.  **Fetch**: 
    *   Round-robin arbitration among warps.
    *   4KB Instruction Cache (2-way set associative) with prefetch.
    *   Decoupled Instruction Fetch Queue (IFQ).
2.  **Decode / Schedule**:
    *   **Pre-decode**: Extracts register usage and instruction type.
    *   **Scheduler**: `advanced_warp_scheduler` (see Section 5).
    *   **Scoreboard**: Tracks register dependencies (RAW, WAW) per warp.
3.  **Issue**:
    *   Dual-issue logic (Slot 0 and Slot 1).
    *   Operand collection from Banked Register File.
4.  **Execute**:
    *   Parallel execution units (ALU, FPU, Tensor, SFU, LSU, Branch).
    *   Variable latency support (Scoreboard tracks completion).
5.  **Writeback**:
    *   Writes results back to Register File.
    *   Clears Scoreboard busy bits.

### 3.3 Execution Units
*   **ALU**: 32-lane SIMD Integer Arithmetic / Logic.
*   **FPU**: Supports FP32, FP64, and FP16 operations.
*   **Tensor Core**: Interface to WGMMA engine for matrix operations.
*   **SFU**: Special Function Unit (Transcendental functions, etc.).
*   **LSU**: Load/Store Unit interfacing with L1 Data Cache.
*   **Branch Unit**: Handles control flow with TAGE branch predictor.

### 3.4 Register File
*   **Banked Design**: 4 banks to reduce read port conflicts.
*   **Capacity**: Per-warp registers (32 registers x 32 lanes).

---

## 4. Memory Subsystem

### 4.1 L1 Data Cache (`l1_data_cache`)
*   **Size**: 16KB (Configurable)
*   **Organization**: 4-way Set Associative
*   **Line Size**: 128 Bytes
*   **Throughput**: Designed for 4-cycle hit latency.
*   **Interface**: Handles 32-lane SIMT requests with coalescing.

### 4.2 HBM Memory Controller (`memory_controller_hbm`)
*   **Scheduling**: First-Ready First-Come-First-Served (FR-FCFS).
*   **Channels**: 8 Channels.
*   **Banks**: 16 Banks per channel.
*   **Timing**: Models real delays (tCL=14, tRCD=14, tRP=14, tRAS=32, etc.).
*   **Queues**: Per-channel request queues with reordering capability.

### 4.3 WGMMA Tile Engine (`wgmma_tile_engine`)
*   **Function**: Offloads matrix multiplication (Hopper-style).
*   **Staging**: 16KB Asynchronous Shared Memory (SMEM) buffer.
*   **Flow**:
    *   Global Memory -> SMEM Staging (Async)
    *   SMEM -> MMA Core (Tensor Op)
    *   Accumulation -> Output

### 4.4 Virtual Memory (`tlb_enhanced`)
*   **Structure**: Two-level TLB (L1 per-SM, L2 shared).
*   **L1 TLB**: 32 entries.
*   **L2 TLB**: 512 entries.
*   **Page Walker**: Hardware-based page table walker.

---

## 5. Advanced Warp Scheduler (`advanced_warp_scheduler`)

A dual-issue, split-queue scheduler implementing a Greedy-Then-Oldest (GTO) policy.

### 5.1 Architecture
*   **Split Queues**:
    *   Compute (ALU/FPU)
    *   Tensor (Matrix Ops)
    *   Memory (Load/Store)
    *   Branch (Control Flow)
*   **Dual Issue**: Can issue two instructions per cycle if they target different pipelines and have no dependencies.

### 5.2 Priority Logic
1.  **Branch**: Highest priority to resolve control flow quickly.
2.  **Memory**: High priority to hide latency.
3.  **Tensor**: Medium priority.
4.  **Compute**: Lowest priority (fill in gaps).

### 5.3 Hazard Detection
*   **RAW**: Checks Scoreboard for pending writes to source registers.
*   **WAW**: Checks Scoreboard for pending writes to destination register.
*   **Conflict**: Prevents dual-issuing conflicting instructions (same warp, same register dependencies).
