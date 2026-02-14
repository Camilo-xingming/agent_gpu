# RalphGPU Dual-Issue Architecture

## Overview

RalphGPU implements a 2-wide issue pipeline with two scheduler slots (slot 0, slot 1).
Each slot is driven by an independent sub-scheduler within `blackwell_scheduler`.

## Scheduler Architecture

### Warp-to-Scheduler Mapping
```
NUM_SCHEDULERS = 2
Scheduler 0 → warps where (warp_id % 2 == 0): warps 0, 2
Scheduler 1 → warps where (warp_id % 2 == 1): warps 1, 3
```

Each sub-scheduler independently picks ONE warp per cycle from its subset using round-robin arbitration.

### Issue Pipeline
```
Scheduler 0 → sched_issue_valid_mask[0] → issue0_fire (gated by decode_stalled_any)
Scheduler 1 → sched_issue_valid_mask[1] → issue1_fire (gated by decode_stalled_slot1 + lane_unit_conflict)
```

### Current Behavior: Round-Robin Alternation (Not Parallel)

**Observation (RALPH-9 P1):** The scheduler alternates between slot 0 and slot 1 each cycle:
```
Cycle N:   sched_mask=01, buf_valid=0001  (scheduler 0 fires)
Cycle N+1: sched_mask=10, buf_valid=0010  (scheduler 1 fires)
Cycle N+2: sched_mask=01, buf_valid=0100  (scheduler 0 fires)
...
```

**Root cause:** Fetch bandwidth. Only ONE warp has a valid instruction buffer at any time.
The fetch pipeline fills one warp's buffer, it gets consumed, then the next warp's buffer is filled.
For true same-cycle dual-issue, TWO warps need valid buffers simultaneously.

### Exception: Tensor/WMMA Dual-Issue

Tensor operations achieve effective dual-issue through a different mechanism:
- `tensor_issue_push` + `tensor_push_lane1_raw` bypass the normal issue pipeline
- `tensor_conflict_mask` and reissue logic allow slot 1 tensor ops independently
- This explains why tensor tests show `issue1_fire` edges (8174 pre-PR60) while ALU-only tests don't

## Lane Unit Conflict

When both slots decode to the same FU type, `lane_unit_conflict` blocks slot 1:
```verilog
wire lane_unit_conflict = (lane0_alu && lane1_alu) ||
                          (lane0_mul && lane1_mul) ||
                          (lane0_fp32 && lane1_fp32) ||
                          // ... all single-instance FUs
                          (lane0_mem && lane1_mem) ||
                          lane0_control;  // slot 0 control blocks slot 1
```

**Affected FUs (single instance):** ALU, MUL, FPU32, FPU64, FP16, SFU, SHFL, VIDEO, MEM.
All have 1 execution unit — slot 1 is silently dropped if both slots target the same FU.

## Key Findings

### block_dim_x Determines Active Warps
```
block_dim_x = 32  → 1 warp  (32 threads / 32 threads_per_warp)
block_dim_x = 64  → 2 warps
block_dim_x = 128 → 4 warps (full utilization with NUM_WARPS=4)
```
**Gotcha:** Tests with `block_dim_x=32` will only activate 1 warp, making slot 1 impossible.

### Dual-Issue Verification Results (RALPH-9 P1)
```
Test: tb_multiwarp_mixed_fu.v
Workload: 4 warps × 64 ALU+MUL pairs (interleaved)
block_dim_x = 128

Results:
  Slot 0: 262 issues
  Slot 1: 263 issues (balanced!)
  Dual-issue (same cycle): 0
  Conflicts: 1
  Cycles: 541
  All 4 warps EXIT ✅
```

## Known Limitations

1. **No same-cycle dual-issue for ALU workloads** — fetch bandwidth bottleneck
2. **Single-instance FUs** — ALU, MUL, etc. have 1 unit; slot 1 blocked on same-FU conflict
3. **No per-register RAW scoreboard** — `lane0_stall_raw = 1'b0` hardcoded (RALPH-9 P2 target)
4. **Fetch extraction regression** — PR #60 introduced −127 WB in tensor test (2× over-fetching, timing mismatch between fetch module and PC advance)

## Future Work

- **Fetch bandwidth improvement:** Wider fetch (2 instructions per cycle) or dual-port instruction buffer
- **RAW scoreboard (RALPH-9 P2):** Enable memory-dependent workloads (LD → compute → ST)
- **FU duplication:** Add second ALU/MUL instance for true dual-issue on same-FU types
