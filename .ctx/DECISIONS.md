# RalphGPU Architectural Decisions

## Decision Log

### D001: Memory Interface Width (2026-01-17)
**Context**: SM V2 requires 64-bit instruction memory for 8-byte cache lines.
**Decision**: Changed imem_data from 32-bit to 64-bit at top level.
**Rationale**: Better cache line utilization and instruction fetch efficiency.
**Status**: Implemented

### D002: L1D Cache Bypass Mode (2026-01-20)
**Context**: Fast simulation vs. realistic memory hierarchy.
**Decision**: L1D_BYPASS parameter (default=1 for fast testing, 0 for full cache).
**Rationale**: Enables rapid iteration during development while preserving full implementation.
**Status**: Implemented

### D003: L2 Cache Optional (2026-01-20)
**Context**: L2 cache adds complexity for simple tests.
**Decision**: L2_ENABLE parameter (default=0, enable for HPC mode).
**Rationale**: Modular design allows testing at different memory hierarchy levels.
**Status**: Implemented

### D004: Scheduler Width Scaling (2026-01-20)
**Context**: B300 supports 4-way issue.
**Decision**: SCHED_LANES and NUM_SCHEDULERS configurable via GPU_PROFILE_HPC.
**Rationale**: Supports both simple (2-way) and HPC (4-way) configurations.
**Status**: Implemented

### D005: Register File ECC (2026-01-20)
**Context**: RAS requirements for HPC/datacenter use.
**Decision**: SEC-DED ECC with ECC_ENABLE parameter (default=1).
**Rationale**: Single-bit correction, double-bit detection critical for reliability.
**Status**: Implemented

### D006: PTX Assembler vs. Real ptxas (Initial)
**Context**: Bootstrap rule requires toolchain before RTL verification.
**Decision**: Implemented Python ptx_assembler.py as ptxas-stub.
**Rationale**: Enables PTX assembly without NVIDIA toolchain dependency.
**Status**: Implemented

### D007: Tensor Core Precision Support (2026-01-17)
**Context**: Which tensor precisions to support.
**Decision**: FP16, BF16, FP8, FP4 in tensor_core.v and wgmma.v.
**Rationale**: Matches B300 capabilities for ML workloads.
**Status**: Implemented

## Pending Decisions
None at this time.

## Conflict Resolutions
None requiring arbitration.
