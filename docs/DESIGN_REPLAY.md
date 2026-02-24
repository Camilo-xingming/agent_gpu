# RalphGPU Pipeline Replay Mechanism Design Spec

## 1. Overview
This document describes the design of a Pipeline Replay mechanism for RalphGPU. Replay allows instructions that cannot proceed due to dynamic hazards (e.g., cache misses, bank conflicts, structural hazards) to be dropped from the pipeline and re-issued later, rather than stalling the pipeline and blocking issue slots for other warps.

## 2. Motivation
- **Improve IPC**: By replaying instead of stalling, the Decode/Issue slots are freed for other warps to make progress on independent instructions.
- **Simplify Hazard Handling**: Eliminates "stale re-issue" bugs where instructions held in decode registers during long stalls can bypass PC-based deduplication or lockout mechanisms.
- **Support Non-blocking Caches**: Essential for implementing MSHRs and allowing the cache to serve other warps while a miss is being handled.

## 3. Architecture Changes

### 3.1 Instruction Tracking
To support replay, the PC of the instruction must be propagated through the pipeline stages until the point of no return (Commit/Writeback).
- **dec_pc**: PC of instruction in Decode stage.
- **issue_pc**: PC of instruction in Issue stage.
- **mem_pc / exe_pc**: PC of instruction in Memory/Execute stage.

### 3.2 Replay Trigger
A `replay` signal is introduced from execution units (L1 Cache, Shared Memory, etc.) to the SM controller.
- **Signal**: `pipe_replay_valid`, `pipe_replay_warp_id`, `pipe_replay_pc`.

### 3.3 Replay Handling (SM)
When a replay is triggered:
1. **Flush Pipeline**: The instruction in the reporting stage and any earlier stages for that warp are invalidated.
2. **PC Rollback**: The warp's PC is set back to the `pipe_replay_pc`.
   - `warp_pc[warp_id] <= pipe_replay_pc;`
3. **Scoreboard Rollback**: The busy bit in the scoreboard for the aborted instruction's destination register must be cleared.
   - This reuses the `fu_conflict_sb_clr` mechanism.
4. **Stall Clearing**: Any stall bits (e.g., `warp_stalled_mem`) associated with the aborted instruction are cleared.

### 3.4 Scheduler Interaction
The scheduler (Blackwell) remains largely unchanged, as the PC rollback and scoreboard clearing naturally make the warp eligible for re-issue.
- **Optimization**: To avoid livelocks, a warp that was replayed due to a cache miss may be temporarily deprioritized or masked until the cache refill is complete.

## 4. Implementation Details - L1 Cache Miss Replay

### Current Path (Stall):
1. Load issued -> Warp stalled in SM.
2. L1 Cache receives request.
3. If miss: L1 goes to `ST_MISS`. SM waits for `resp_valid`.
4. L1 busy for N cycles. SM warp stalled for N cycles.

### Proposed Path (Replay):
1. Load issued -> Warp marked as "in-flight" (scoreboard set).
2. L1 Cache receives request.
3. If miss:
   - L1 returns `resp_replay = 1` immediately.
   - L1 starts background refill (MSHR).
4. SM receives `resp_replay`:
   - `warp_pc[w] <= load_pc`
   - `scoreboard[w][rd] <= 0`
   - Warp becomes eligible for other instructions OR waits for "refill done".
5. L1 finishes refill:
   - Signals "refill done for PC/Addr".
   - Warp re-issues the load.

## 5. Summary of Benefits
- Dual-issue utilization: Freeing a slot from a stalled instruction allows the other slot to be used.
- Robustness: Decouples pipeline state from long-latency external events.
- Scalability: Foundation for more complex memory systems (MSHRs, bank conflict handling).
