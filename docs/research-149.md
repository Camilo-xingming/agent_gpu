# Issue #149 Research: Pipeline Replay / Scheduler Fairness (WB Precision)

## 1. Problem Statement

test_sm_v2_perf_tensor_multiwarp runs 4 warps × 1024 WMMA ops = 4096 expected writebacks.
Actual: WB=4100 (+4 extra). Per-warp imbalance: w0=-2, w1=+2, w2=-1, w3=+5.

Root cause per issue #5: "d1 stale re-issue in slot0 PC-advance scenario can bypass PC-based dedup."

Previous failed attempts: PC dedup tracker (timing window), credit/token mechanism (pipeline latency).

## 2. Architecture Overview

### Pipeline Stages
```
Scheduler → Decode (dec0/dec1) → Issue → Execute → WBQ FIFO → WB Arbiter → RF Write
           ↑                       |
           ├── warp_inst_consume ──┘ (advances inst buffer)
           └── scoreboard check
```

### Key Modules
- `advanced_warp_scheduler.v` (594 lines): Dual-issue, split compute/tensor/memory/branch.
  Round-robin per category. Scoreboard for RAW/WAW. Sets scoreboard on issue, clears on WB.
- `blackwell_scheduler.v` (649 lines): Same interface, adds tcgen05/async MMA tracking.
  Selected via `USE_BLACKWELL_SCHEDULER` define.
- `sm_writeback_arbiter.v`: Round-robin across 17 FU sources (10 queued + 7 latched).
  One writeback per cycle. Arbiter is FU-centric, not warp-aware.
- `wb_fifo.v`: Generic sync FIFO for WBQ. Drops with FATAL on push-while-full-without-pop.
- `sm_wbq_bank.v`: 9 WBQ instances (ALU/MUL/FPU32/FPU64/FP16/SFU/SHFL/VIDEO/Special).
- `streaming_multiprocessor_v2.v` (5813 lines): God module. Houses pipeline, scheduler
  instantiation, tensor issue queue, lockout, and all FU dispatch.

### Tensor Push Path (Critical)
```verilog
// sm_v2.v:3360-3365
wire tensor_push_lane0_raw = issue_valid && issue_tensor_op;
wire tensor_push_lane1_raw = issue1_valid && issue1_tensor_op && !tensor_push_lane0;
wire tensor_push_lane0 = tensor_push_lane0_raw;  // NO lockout on lane0!
wire tensor_push_lane1 = tensor_push_lane1_raw && !tensor_push_locked[issue1_warp_id];
```

Lane0 has NO lockout gating. The 2-cycle lockout only blocks lane1.
Comment: "slot0 must not be blocked" — blocking slot0 could suppress legitimate instructions.

### 2-Cycle Lockout Mechanism
```
tensor_push_lockout_0[warp] ← 1 on successful tensor_issue_push_fire
tensor_push_lockout_1       ← shift from lockout_0
tensor_push_locked = lockout_0 | lockout_1
```
Only used to gate `tensor_push_lane1`. Catches most stale replays but not slot0 replays.

### Issue Stage Timing
```verilog
// sm_v2.v:2391
issue_valid <= dec_valid && !decode_stalled && !branch_flush_dec0;
// sm_v2.v:2047-2048
dec0_valid <= decode_stalled ? 1'b1 : issue0_fire;  // Hold if stalled
```

When decode_stalled transitions 1→0:
- Cycle N: dec0_valid=1, decode_stalled=1 → issue_valid suppressed
- Cycle N+1: stall clears → issue_valid=1 (fires OLD instruction from dec0)
- Cycle N+1 also: dec0_valid <= issue0_fire (new from scheduler)
- Cycle N+2: issue_valid fires NEW instruction

This is correct for most FUs. But for tensor: if the same warp's tensor op was already
pushed on an earlier un-stalled cycle, and then a stall causes it to sit in dec0, when
the stall clears the op fires again as issue_valid — double push on lane0.

### WB Arbiter: Not Warp-Aware
```verilog
// sm_writeback_arbiter.v: Round-robin across FU indices 0-16
// No per-warp fairness — warp with more entries in WBQs gets more WB slots
```

## 3. Root Cause Analysis

The +4 extra writebacks come from tensor instructions being pushed to the tensor issue
queue more than once. The 2-cycle lockout catches lane1 replays but lane0 is unprotected.

**Scenario for stale lane0 tensor push:**
1. Scheduler issues warp W tensor op → dec0 gets it (cycle N)
2. Cycle N+1: issue_valid=1, tensor_push_lane0 fires, tensor_issue_push_fire=1
3. Lockout sets tensor_push_lockout_0[W]=1
4. But if decode stalls for some reason AFTER the push fire in step 2, dec0 HOLDS
   the same instruction (dec0_valid <= 1 on stall)
5. Stall could be from tensor_issue_full_next or some other stall source
6. When stall clears, issue_valid fires AGAIN for the same instruction
7. But lockout_0[W] may have already shifted to lockout_1[W] or cleared
8. tensor_push_lane0 is NOT gated by lockout → double push

The 2-cycle window is insufficient if stalls can last >2 cycles, or if the lockout
shift timing doesn't align with when the stale instruction re-fires.

## 4. Per-Warp WB Imbalance

The round-robin WB arbiter (sm_writeback_arbiter) rotates across FU indices, not warps.
When multiple warps' tensor results sit in the same tensor WBQ, they're dequeued in FIFO
order. The arbiter gives equal priority to each FU, not each warp. So a warp that happens
to have more entries at the front of the tensor WBQ gets more WB cycles.

This doesn't directly cause the +4 overcount, but it means the overcount distributes
unevenly across warps (w0=-2, w1=+2, w2=-1, w3=+5).

## 5. Solution Space

### A. Precise Stale-Issue Detection (Issue Epoch Tag)
**Concept:** Each instruction gets a unique tag when consumed from instruction buffer.
Tag flows through pipeline. At tensor push point, compare tag with expected — stale tags
are suppressed.

- Pro: Precise, no false suppression
- Con: Requires pipeline-width tag propagation, adds latency/area
- Complexity: Medium

### B. Lane0 Lockout Extension
**Concept:** Apply the existing 2-cycle lockout to lane0 as well, with an escape for
legitimate re-execution (loop back-edge).

- Pro: Minimal change, leverages existing mechanism
- Con: Risk of false suppression on legitimate loop re-execution. Need to distinguish
  "stale replay" from "loop iteration" — PC comparison could work since loop iterations
  advance the inst buffer pointer but stale replays don't.

### C. Pipeline Replay Buffer (Issue #149's proposed approach)
**Concept:** Instead of immediate re-issue of stalled instructions, capture them in a
replay buffer. Replay buffer drains when the pipeline is ready. This eliminates the
double-issue problem because the replayed instruction comes from the buffer, not from
the decoder re-firing.

- Pro: Architecturally clean, matches real GPU designs (NVIDIA has replay)
- Con: Significant new logic — replay buffer, replay selection, integration with scoreboard
- Complexity: High

### D. Credit-Based Tensor Issue Quota
**Concept:** Each warp gets exactly N_OPS_PER_WARP tensor credits. Decrement on
tensor_issue_push_fire. When credit=0, suppress further pushes for that warp.

- Pro: Guaranteed precise WB count
- Con: Requires knowing the expected count in advance (only works for fixed workloads).
  Not a general solution. Also, credit init is test-specific.

### E. Dedup at Tensor Issue Queue Entry
**Concept:** When pushing to tensor issue queue, check if the queue already contains
an entry with the same {warp, PC} — if so, suppress.

- Pro: Catches all duplicates regardless of timing
- Con: Requires CAM lookup on every push (area/timing). Also, same {warp, PC} is
  legitimate in loops — needs epoch or sequence disambiguation.

## 6. Recommended Approach

**Primary: Issue Epoch Tag (A) — lightweight variant**

Instead of full pipeline-width tags, use a per-warp "issue sequence number" (ISN):
- Scheduler increments ISN[warp] on warp_inst_consume
- ISN flows through dec0/issue pipeline alongside instruction
- At tensor push point: compare ISN with last-pushed ISN per warp
- If ISN matches last-pushed → stale, suppress
- If ISN is newer → legitimate, allow and update last-pushed

Width: 4-8 bits per warp (wrapping is fine since pipeline depth < 2^4).
Cost: ~4 bits * 4 warps = 16 bits of state + comparators.

**Secondary: WB Arbiter Warp-Aware Fairness**

Modify sm_writeback_arbiter to do inner round-robin per warp within each FU source.
When tensor WBQ is selected, alternate between warps (round-robin on wb_warp_id).
This doesn't fix the +4 overcount but equalizes distribution.

## 7. Test Infrastructure

- Testbench: `tb/tb_sm_v2_perf_tensor_multiwarp.v`
- Acceptance: `wb_count == TOTAL_OPS` (4096) with zero lane0 suppress
- Build tool: iverilog (NOT currently installed on ist-mac-s)
- Run: `make test_sm_v2_perf_tensor_multiwarp`
- Existing counters: wb_count, per-warp wb, tensor suppress counts, stall breakdowns

## 8. Files to Modify

1. `rtl/streaming_multiprocessor_v2.v` — ISN tracking + tensor push gating
2. `rtl/sm_writeback_arbiter.v` — (optional) warp-aware fairness
3. `rtl/advanced_scheduler.v` / `rtl/blackwell_scheduler.v` — ISN increment on consume
4. `tb/tb_sm_v2_perf_tensor_multiwarp.v` — verify WB=4096 exact
