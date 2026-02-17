# RalphGPU — Status & Roadmap

**Last updated: 2026-02-17**

## Tier Summary

| Tier | Scope | Status |
|------|-------|--------|
| 1 | Core ISA (ALU/MUL/DIV/Branch/Sync/Memory) | ✅ 100% VERIFIED |
| 2 | Memory & Sync (LD/ST/Atomic/cp.async/mbarrier) | ✅ 100% VERIFIED |
| 3 | Compute Extensions (FP16/FP64/CVT/SFU/Warp Collectives) | ✅ 100% VERIFIED |
| 4 | Advanced Features (TMA/Texture/Surface/FP4-8/LZ4) | ⏳ PENDING |

---

## Tier 3: Compute Extensions — ✅ COMPLETE

| Item | Tests | Status | PR |
|------|-------|--------|----|
| FP32 SFU (sin/cos/sqrt/rcp/rsqrt/lg2/ex2) | 173/173 | ✅ | — |
| FP16 unit (add/sub/mul/fma/neg/abs/min/max/tanh/ex2/cmp) | 46/46 RTL + 15/15 FRM | ✅ | #68 |
| FP64 unit (add/sub/mul/div/fma/neg/abs/min/max/sqrt/rcp/rsqrt/copysign/testp) | 62/62 RTL | ✅ | — |
| CVT unit (s32/u32/f32/f64/f16 + rounding + special values) | 50/50 RTL | ✅ | — |
| Warp Shuffle (shfl.idx/up/down/bfly) | 25/25 RTL | ✅ | #75 |
| Warp Vote (all/any/uni/ballot) | 25/25 RTL | ✅ | #75 |

### Bug Fixes During Tier 3
- **fpu64.v**: CLZ loop direction (low→high for correct MSB), adder normalization carry, subtraction sign, div quotient overflow
- **cvt_unit.v**: CLZ loop direction (same root cause as fpu64)

---

## Tier 4: Advanced Features — PENDING

### Priority 1: FRM Path Completion
RTL modules exist but lack FRM (software model) integration for full-stack testing.

| Item | RTL Module | RTL Lines | TB Exists | FRM Path | Priority |
|------|-----------|-----------|-----------|----------|----------|
| cp.async / st.async | async_copy_engine.v | 927 | ✅ tb_async_copy_unit.v | ❌ Need FRM handler | P1 |
| mbarrier | mbarrier_unit.v | 486 | ✅ tb_mbarrier_unit.v | ❌ Need FRM handler | P1 |
| shfl/vote FRM | warp_shuffle.v | 300 | ✅ tb_warp_collective_unit.v | ❌ Need FRM handler | P1 |
| Tensor MMA FRM | tensor_core.v | — | ✅ multiple TBs | ❌ Need FRM handler | P2 |
| FP16/FP64/CVT FRM | fp16_unit.v, fpu64.v, cvt_unit.v | — | ✅ | ✅ FP16 done, FP64/CVT partial | P2 |

### Priority 2: New Feature Verification

| Item | RTL Module | RTL Lines | TB Exists | TODOs | Notes |
|------|-----------|-----------|-----------|-------|-------|
| TMA (Tensor Memory Accelerator) | tma_unit.v | 383 | ✅ tb_tma_unit.v | 0 | Bulk copy, tiled addressing |
| Texture unit | texture_unit.v | 531 | ✅ tb_texture_unit.v | 1 | tex2D/3D, filtering, TODO in bilinear calc |
| Surface load/store | (in texture_unit) | — | — | — | suld/sust/sured opcodes in decoder |
| st.bulk | st_bulk_unit.v | 324 | ✅ tb_st_bulk_unit.v | 0 | Bulk zeroing/init |
| LZ4 decompressor | lz4_decompressor.v | 446 | — | 0 | HW decompression for FP4 bandwidth |
| CHI controller | chi_controller.v | 531 | — | 0 | Multi-chiplet coherency |
| DPX unit | dpx_unit.v | 385 | ✅ tb_dpx_unit.v | 0 | Dynamic programming extensions |
| Multimem unit | multimem_unit.v | 375 | ✅ tb_multimem_unit.v | 0 | Distributed shared memory |
| Grid dependency | griddep_unit.v | 258 | ✅ tb_griddep_unit.v | 0 | Cross-grid scheduling |

### Priority 3: Integration & Hardening

| Item | Notes |
|------|-------|
| Top-level perf counter wiring | 6 TODOs in ralph_gpu_top.v (sync/SFU/L1/warp/tensor stubs) |
| SM predicate register file | 2 TODOs in streaming_multiprocessor_v2.v (pred_in/carry_in) |
| Async copy → shared memory path | 1 TODO: ace_smem_rd_data not connected |
| Branch predictor misprediction | 1 TODO: compare with prediction |
| Pipeline replay (Issue #5) | Replace stall-based hazard with replay for tensor residual |
| Vector Add + MatMul validation (Issue #11) | End-to-end PTX workload verification |

---

## Test Scorecard

| Suite | Pass | Total | Status |
|-------|------|-------|--------|
| ALU | 26 | 26 | ✅ |
| MUL | 27 | 27 | ✅ |
| FPU (FP32) | 26 | 26 | ✅ |
| FP16 unit | 46 | 46 | ✅ |
| FPU64 (FP64) | 62 | 62 | ✅ |
| CVT unit | 50 | 50 | ✅ |
| Warp Shuffle | 25 | 25 | ✅ |
| Warp Vote | 25 | 25 | ✅ |
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
| FP16 FRM | 15 | 15 | ✅ |
| **Total** | **878** | **878** | **✅** |

## Performance
- PTX benchmark: 95.9% NVIDIA parity (11/11 pass)
- Multi-warp IPC: 0.80+
- Tensor WB: 4091/4096 (99.88%), 0 stalls

## Open GitHub Issues
- #1: Tensor Multiwarp test (d1+dedup+pfu fix) — fixed by Patch F, needs close
- #5: Pipeline replay mechanism for tensor writebacks — Tier 4 P3
- #11: Phase 2 vector add + matmul validation — Tier 4 P3

## Decision Log

### 2026-02-17 — Tier 3 Priority Order
**Consensus**: CoderOpus + CoderGemini + Lily
**Decision**: FP16/FP64/CVT → Warp Collectives
**Result**: All completed same day

### 2026-02-17 — Tier 4 Pending
**Next**: FRM path completion (cp.async, mbarrier, shfl/vote, tensor MMA, FP64/CVT)
**Then**: TMA/Texture/Surface verification, integration hardening
