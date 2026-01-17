# RalphGPU: Architectural & Implementation Guide

**Target Audience:** Hardware Engineers, GPU Architects, System Integrators
**Architecture Generation:** RalphGPU Generation 2 (Volta-Class equivalent)
**Status:** Integrated RTL (Superscalar / Multithreaded)

---

## 1. High-Level Architecture

RalphGPU follows a standard SIMT (Single Instruction, Multiple Threads) execution model. The core of the system is the **Streaming Multiprocessor (SM)**, which executes groups of 32 threads called **Warps**.

### 1.1 Pipeline Stages
The SM V2 uses a decoupled 5-stage pipeline designed for high-frequency operation and latency hiding:

1.  **Fetch (ICache):** Uses `icache.v` to fetch 32-bit instructions. It maintains per-warp PC tracking and supports prefetching.
2.  **Pre-Decode & Buffer:** Instructions are placed into per-warp **Instruction Buffers (IBuffers)**. A pre-decode logic extracts dependency and unit-affinity signals for the scheduler.
3.  **Advanced Issue (Dual-Issue GTO):** The `advanced_warp_scheduler` selects up to two instructions per cycle from different warps or independent slots. It uses a **Greedy-Then-Oldest (GTO)** policy.
4.  **Execute (Parallel FUs):** Multiple SIMD units operate in parallel:
    *   **ALU:** Single-cycle integer/bitwise ops.
    *   **FPU (FP32/64/16):** Multi-cycle IEEE 754 compliant units.
    *   **Tensor Core:** HMMA (Half-precision Matrix Multiply-Add) unit.
    *   **LSU:** Load/Store unit connecting to Shared and Global memory.
5.  **Writeback (Banked):** A round-robin arbiter manages results from FUs and writes them back to the `register_file_banked`.

---

## 2. Key Component Implementation Details

### 2.1 Advanced Warp Scheduler
*   **Dual-Issue:** The scheduler attempts to fill two issue slots per cycle.
*   **Scoreboarding:** A bit-mask per warp tracks "busy" registers. An instruction is only eligible if its source and destination registers are not marked in the scoreboard.
*   **GTO Policy:** Prioritizes the warp that issued most recently (greedy) to maintain cache locality and loop throughput, falling back to the oldest warp to prevent starvation.

### 2.2 Banked Register File
*   **Structure:** 32 lanes (threads) x 32 registers.
*   **Implementation:** Registers are partitioned into **4 banks**.
*   **Conflict Handling:** To support dual-issue (which may require up to 6 read ports), the scheduler must check for bank conflicts. If two instructions in the same slot read from the same bank, one is stalled.

### 2.3 Instruction Cache (ICache)
*   **Geometry:** 4KB, 2-way set-associative (configurable).
*   **Logic:** Decouples the SM from the high-latency AXI instruction memory. It uses a valid/ready handshake to stall the pipeline on misses.

---

## 3. Comparison with NVIDIA Hopper (Gap Analysis)

To achieve state-of-the-art performance parity with NVIDIA's H100 (Hopper), implementation of the following "Gen 5" features is required:

### 3.1 Register File vs. WGMMA
*   **Current (RalphGPU):** Tensor operations are **HMMA**. Data must move: `Memory -> Register -> Tensor Core`.
*   **Hopper (WGMMA):** Operands flow `Shared Memory -> Tensor Core`. 
*   **Implementation Note:** To implement this, the `wgmma_tile_engine.v` must be connected to the `shared_memory.v` read ports via a 128-byte wide bus, bypassing the Issue stage's register file reads.

### 3.2 Synchronous LSU vs. TMA
*   **Current (RalphGPU):** Data movement is handled by standard `ld`/`st` instructions which occupy pipeline slots.
*   **Hopper (TMA):** An **Asynchronous Copy Engine** moves data tiles independently.
*   **Implementation Note:** Integrate `async_copy_engine.v` so it can receive "Copy Descriptors" from the SM and perform AXI bursts to Shared Memory without involving the main execution pipeline.

---

## 4. Implementation Checklist for Engineers

If you are picking up this codebase to implement a physical GPU:

1.  **Clock Gating:** The `register_file_banked` and `tensor_core` are power-intensive. Implement fine-grained clock gating on the `valid_in` signals.
2.  **SRAM Integration:** Replace the Verilog `reg` arrays in `shared_memory.v` and `register_file_banked.v` with vendor-specific Single-Port or Dual-Port SRAM macros.
3.  **AXI Bursting:** The `memory_interface` currently issues single-beat requests. Wrap it with a **Coalescing Unit** that detects contiguous lane addresses and converts them into AXI `INCR` bursts ($len > 0$).
4.  **FP16 Optimization:** The current `fp16_unit.v` uses FP32 promotion. For silicon area efficiency, rewrite this to use a native 16-bit multiplier tree.

---
**Verdict:** RalphGPU SM V2 is a robust, silicon-ready baseline for a high-performance compute-oriented GPU.
