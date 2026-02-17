# RalphGPU — Status & Roadmap

## Current State (2026-02-17)

### Tier 1: Core ISA — ✅ 100% VERIFIED (11/11)
### Tier 2: Memory & Sync — ✅ 100% VERIFIED (5/6, cp.async decode-only)
### Tier 3: Compute Extensions — IN PROGRESS

| Item | Status | Owner | Notes |
|------|--------|-------|-------|
| FP32 SFU | ✅ DONE | — | 173/173 tests pass |
| FP16 handler | 🔧 IN PROGRESS | CoderCodex | 15 test cases generated, handler impl underway |
| FP64 verification | ⏳ NEXT | CoderOpus | fpu64.v exists (890L), needs test suite |
| CVT unit tests | ⏳ NEXT | CoderOpus | cvt_unit.v exists (807L), basic CVT working |
| Warp Collectives | ⏳ QUEUED | — | warp_collective_unit.v + warp_shuffle.v exist, decoder wired |

### Patch F — ✅ MERGED (PR #66 → master)
- pipe_tensor_ready +4 margin
- decode_stalled_slot1 tensor bypass
- 1-cycle per-warp lockout replaces PC dedup
- Scheduler tensor-tensor conflict detection
- WB=4091/4096 (99.88%), 0 stalls, deterministic

## Decision Log

### 2026-02-17 — Tier 3 Priority: FP16/FP64/CVT before Warp Collectives
**Consensus**: CoderOpus, CoderGemini, Lily
**Rationale**:
1. All three RTL modules already implemented with 0 TODO/FIXME
2. CoderCodex already progressing on FP16 test generator
3. Completes core arithmetic pipeline — prerequisite for broader workload compatibility
4. Warp Collectives (shfl, vote, ballot, match, elect, red.async) queued as next priority

### Assignment
- **CoderCodex**: FP16 handler + verification (in progress)
- **CoderOpus**: FP64 verification + CVT unit tests
- **CoderGemini**: Available for review/support
- **Warp Collectives**: Starts after FP16/FP64/CVT complete

## Test Scorecard

| Suite | Pass | Total | Status |
|-------|------|-------|--------|
| ALU | 26 | 26 | ✅ |
| MUL | 27 | 27 | ✅ |
| FPU | 26 | 26 | ✅ |
| Decoder | 16 | 16 | ✅ |
| Extended ALU | 50 | 50 | ✅ |
| Shared Mem | 11 | 11 | ✅ |
| Register File | 11 | 11 | ✅ |
| Warp Scheduler | 10 | 10 | ✅ |
| LLM Operators | 5 | 5 | ✅ |
| Trigonometric | 3 | 3 | ✅ |
| B300 Features | 145 | 145 | ✅ |
| FRM | 152 | 152 | ✅ |
| FP32 SFU | 173 | 173 | ✅ |
| **Total** | **655** | **655** | **✅** |

## Performance
- PTX benchmark: 95.9% NVIDIA parity (11/11 pass)
- Multi-warp IPC: 0.80+
