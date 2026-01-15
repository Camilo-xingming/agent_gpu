# RalphGPU Memory Architecture

## Overview

RalphGPU implements a comprehensive memory hierarchy designed for high-bandwidth, low-latency GPU workloads. The architecture supports four configuration profiles (Edge, Mobile, Desktop, Datacenter) with scalable parameters.

## Memory Hierarchy

```
┌────────────────────────────────────────────────────────────────┐
│                         SM 0                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Register │  │ L1 Inst  │  │ L1 Data  │  │ Shared   │       │
│  │   File   │  │  Cache   │  │  Cache   │  │  Memory  │       │
│  │ 32-128KB │  │ 8-32KB   │  │ 16-128KB │  │ 16-96KB  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │              │
│       └─────────────┴──────┬──────┴─────────────┘              │
│                            │ L1 TLB (32 entries)               │
└────────────────────────────┼───────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│     SM 1      │    │     SM 2      │    │    SM N-1     │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────┴────────┐
                    │ L2 TLB (512)    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  L2 Bank 0  │      │  L2 Bank 1  │      │  L2 Bank N  │
│  256KB-1MB  │      │  256KB-1MB  │      │  256KB-1MB  │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │    Memory Controller      │
              │  1-8 channels × 128-512b  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────┴─────────────┐
              │    DDR5 / LPDDR5 / HBM    │
              └───────────────────────────┘
```

## Configuration Profiles

### Profile Summary

| Parameter | Edge | Mobile | Desktop | Datacenter |
|-----------|------|--------|---------|------------|
| **L1 Data Cache** | 16 KB | 32 KB | 64 KB | 128 KB |
| **L1 Inst Cache** | 8 KB | 16 KB | 32 KB | 32 KB |
| **L1 Ways** | 4 | 4 | 8 | 8 |
| **L1 Line Size** | 32 B | 64 B | 128 B | 128 B |
| **L2 Total** | 256 KB | 512 KB | 2 MB | 4 MB |
| **L2 Banks** | 2 | 4 | 8 | 16 |
| **L2 Ways** | 8 | 8 | 16 | 16 |
| **Shared Memory** | 16 KB | 32 KB | 64 KB | 96 KB |
| **Register File** | 32 KB | 64 KB | 128 KB | 128 KB |
| **Mem Channels** | 1×128b | 2×256b | 4×256b | 8×512b |
| **Memory Type** | LPDDR5 | LPDDR5 | DDR5 | HBM3 |

### To Select Profile

In `memory_config.vh`:
```verilog
// Uncomment ONE profile:
// `define GPU_CONFIG_EDGE
// `define GPU_CONFIG_MOBILE
// `define GPU_CONFIG_DESKTOP
`define GPU_CONFIG_DATACENTER
```

## L1 Data Cache

### Architecture
- **Organization**: Set-associative, write-through
- **Non-blocking**: MSHR-based miss handling
- **Prefetching**: Hardware stride prefetcher
- **Write Combining**: Coalesces stores before L2

### Parameters

| Parameter | Edge | Mobile | Desktop | Datacenter |
|-----------|------|--------|---------|------------|
| Size | 16 KB | 32 KB | 64 KB | 128 KB |
| Ways | 4 | 4 | 8 | 8 |
| Line Size | 32 B | 64 B | 128 B | 128 B |
| Sets | 128 | 128 | 64 | 128 |
| MSHR Entries | 8 | 8 | 8 | 8 |
| WCB Entries | 8 | 8 | 8 | 8 |
| Hit Latency | 2 cycles | 2 cycles | 2 cycles | 2 cycles |

### Memory Array Dimensions

```
L1D Data Array:
- Width:  LINE_SIZE × 8 bits = 256-1024 bits
- Depth:  NUM_SETS × NUM_WAYS = 512-1024 entries
- Total:  SIZE_KB × 1024 × 8 bits

L1D Tag Array:
- Width:  TAG_BITS + valid + dirty = ~20 bits per way
- Depth:  NUM_SETS = 64-128 entries
- Total:  NUM_SETS × NUM_WAYS × 20 bits
```

## L2 Cache

### Architecture
- **Organization**: Multi-banked, set-associative
- **Policy**: Write-back, LRU replacement
- **ECC**: Optional SECDED protection
- **Arbitration**: Round-robin per bank

### Parameters

| Parameter | Edge | Mobile | Desktop | Datacenter |
|-----------|------|--------|---------|------------|
| Total Size | 256 KB | 512 KB | 2 MB | 4 MB |
| Banks | 2 | 4 | 8 | 16 |
| Size/Bank | 128 KB | 128 KB | 256 KB | 256 KB |
| Ways | 8 | 8 | 16 | 16 |
| Line Size | 128 B | 128 B | 128 B | 128 B |
| Sets/Bank | 128 | 128 | 128 | 128 |
| MSHR/Bank | 8 | 8 | 8 | 8 |
| Hit Latency | 20 cycles | 20 cycles | 20 cycles | 20 cycles |

### Memory Array Dimensions (Per Bank)

```
L2 Data Array:
- Width:  LINE_SIZE × 8 = 1024 bits
- Depth:  SETS × WAYS = 1024-2048 entries
- Total:  128-256 KB per bank

L2 Tag Array:
- Width:  TAG_BITS + valid + dirty + ECC = ~32 bits per way
- Depth:  SETS = 128 entries
- Total:  SETS × WAYS × 32 bits = 64-128 KB per bank
```

## Shared Memory

### Architecture
- **Organization**: Banked SRAM with conflict detection
- **Banks**: 16-32 banks for parallel access
- **Conflict Handling**: Serialize conflicting accesses

### Parameters

| Parameter | Edge | Mobile | Desktop | Datacenter |
|-----------|------|--------|---------|------------|
| Size | 16 KB | 32 KB | 64 KB | 96 KB |
| Banks | 16 | 32 | 32 | 32 |
| Bank Width | 32 bits | 32 bits | 32 bits | 32 bits |
| Depth/Bank | 256 | 256 | 512 | 768 |
| Latency | 4 cycles | 4 cycles | 4 cycles | 4 cycles |

### Memory Array Dimensions

```
Shared Memory Bank:
- Width:  32 bits (1 word)
- Depth:  SIZE / (NUM_BANKS × 4) = 256-768 words
- Total:  32 banks × depth × 32 bits
```

## Register File

### Architecture
- **Organization**: Multi-banked for parallel read/write
- **Ports**: 4-6 read, 2-4 write (profile dependent)
- **Allocation**: Dynamic per-thread allocation

### Parameters

| Parameter | Edge | Mobile | Desktop | Datacenter |
|-----------|------|--------|---------|------------|
| Size/SM | 32 KB | 64 KB | 128 KB | 128 KB |
| Banks | 4 | 8 | 16 | 16 |
| Read Ports | 4 | 4 | 6 | 6 |
| Write Ports | 2 | 2 | 4 | 4 |
| Regs/Thread | 32 | 32 | 32 | 32 |
| Max Regs | 255 | 255 | 255 | 255 |

### Memory Array Dimensions

```
Register File Bank:
- Width:  32 bits
- Depth:  SIZE / (NUM_BANKS × 4) = 2K-8K entries
- Ports:  Multi-port (read: 4-6, write: 2-4)
```

## TLB (Translation Lookaside Buffer)

### Two-Level Hierarchy

**L1 TLB (Per SM)**
- Entries: 32
- Ways: 4-way set associative
- Page Sizes: 4KB, 2MB
- Hit Latency: 1 cycle

**L2 TLB (Shared)**
- Entries: 512
- Ways: 8-way set associative
- Page Sizes: 4KB, 2MB, 1GB
- Hit Latency: 5 cycles

### TLB Entry Format
```
┌─────────┬───────────┬────────────┬──────┐
│ Valid   │ VPN Tag   │ PPN        │ Perm │
│ (1 bit) │ (36 bits) │ (28 bits)  │(4 b) │
└─────────┴───────────┴────────────┴──────┘
Total: 69 bits per entry
```

## Memory Controller

### Interface Support
- DDR4-3200
- DDR5-4800/5600
- LPDDR5-6400
- HBM2E / HBM3

### Address Mapping
```
┌──────────┬────────────┬──────┬─────────┬────────┬────────┐
│   Row    │ Bank Group │ Bank │ Channel │ Column │ Offset │
│ Variable │   2 bits   │ 2 b  │ 0-3 b   │ 10 b   │ 4-7 b  │
└──────────┴────────────┴──────┴─────────┴────────┴────────┘
```

### Scheduling
- FR-FCFS (First-Ready, First-Come-First-Served)
- Row buffer hit prioritization
- Per-bank request queues

## Bandwidth Analysis

### Peak Bandwidth by Profile

| Profile | Channels | Width | Freq | Peak BW |
|---------|----------|-------|------|---------|
| Edge | 1 | 128b | 3.2 GHz | 51.2 GB/s |
| Mobile | 2 | 256b | 3.2 GHz | 204.8 GB/s |
| Desktop | 4 | 256b | 4.8 GHz | 614.4 GB/s |
| Datacenter | 8 | 512b | 3.2 GHz | 1638.4 GB/s |

### Effective Bandwidth

Accounting for typical memory efficiency (~70%):

| Profile | Effective BW |
|---------|--------------|
| Edge | ~36 GB/s |
| Mobile | ~143 GB/s |
| Desktop | ~430 GB/s |
| Datacenter | ~1147 GB/s |

## SRAM Macro Requirements

### Summary Table

| Memory | Width | Depth | Instances | Total Bits |
|--------|-------|-------|-----------|------------|
| L1D Data | 1024b | 512-1K | per SM | 0.5-1 Mb/SM |
| L1D Tag | 160b | 64-128 | per SM | 10-20 Kb/SM |
| L1I Data | 512b | 256-512 | per SM | 128-256 Kb/SM |
| L2 Data | 1024b | 1K-2K | per bank | 1-2 Mb/bank |
| L2 Tag | 512b | 128 | per bank | 64 Kb/bank |
| Shared | 32b | 256-768 | 32/SM | 256-768 Kb/SM |
| RegFile | 32b | 2K-8K | 4-16/SM | 256-4096 Kb/SM |
| L1 TLB | 72b | 32 | per SM | 2.3 Kb/SM |
| L2 TLB | 72b | 512 | 1 | 36 Kb |

### Total SRAM per Profile

| Profile | SMs | SRAM/SM | L2 SRAM | Total SRAM |
|---------|-----|---------|---------|------------|
| Edge | 4 | ~0.5 Mb | ~0.5 Mb | ~2.5 Mb |
| Mobile | 8 | ~1 Mb | ~1 Mb | ~9 Mb |
| Desktop | 16 | ~2 Mb | ~4 Mb | ~36 Mb |
| Datacenter | 64 | ~2 Mb | ~8 Mb | ~136 Mb |

## Implementation Notes

### Without Memory Macros

The RTL uses behavioral Verilog arrays:
```verilog
reg [WIDTH-1:0] memory [0:DEPTH-1];
```

This allows:
- Functional simulation
- FPGA implementation (using block RAM)
- Synthesis to standard cells (area inefficient)

### With Memory Macros

For ASIC implementation, replace with:
```verilog
// Instantiate foundry SRAM macro
sram_sp_256x128 u_mem (
    .clk(clk),
    .addr(addr),
    .din(din),
    .dout(dout),
    .we(we)
);
```

Memory compiler inputs:
- Word width
- Number of words
- Number of ports
- Read/write configuration
- Power/performance targets

---

*Document Version: 1.0*
*Last Updated: 2026-01-16*
