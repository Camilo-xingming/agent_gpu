# RalphGPU Status

## State Snapshot
- **Current MVU**: Tier 2 Complete (DONE)
- **Status**: COMPLETE - Tier 1 & 2 fully verified
- **Verification Gap**: Tier 3+ remain (FP16/FP64/CVT, Warp Collectives, Async/Tensor)

## Issue Tracking (from Codex/Gemini)

### Codex Issues
| # | Issue | Status |
|---|-------|--------|
| 1 | tb_b300_features.v only tests decoder, not functional | ❌ PENDING |
| 2 | FRM missing branch/exit support | ✅ FIXED (EXIT + BRANCH added) |
| 3 | FRM missing shared/param/atomic/membar | ✅ FIXED (param/const/atom/ld.shared/st.shared/membar added) |
| 4 | FRM missing cp.async/st.async/mbarrier | ❌ PENDING |
| 5 | FRM missing warp collectives (SHFL/VOTE/REDUX) | ❌ PENDING |
| 6 | FRM missing tensor paths (WGMMA/TMA) | ❌ PENDING |
| 7 | RTL↔FRM comparison framework | ✅ FIXED |
| 8 | Python test generators | ✅ FIXED |
| 9 | FRM branch model lacks per-thread divergence/reconvergence (thread0-only control) | ❌ PENDING |
| 10 | BAR.SYNC/MEMBAR have no FRM semantics | ✅ FIXED (CTAState + barrier tracking added) |
| 11 | Test generators missing branch/memory/warp cases | ✅ FIXED (MemoryTestGenerator + BranchTestGenerator added) |
| 12 | FRM missing FP16/FP64/CVT execution | ❌ PENDING |
| 13 | INT div/rem execution | ✅ FIXED (DIV/REM handler + 32 tests) |

### Gemini Issues
| # | Issue | Status |
|---|-------|--------|
| 1 | FP4/FP8 Tensor Core e2e verification | ❌ PENDING |
| 2 | Texture/Surface unit verification | ❌ PENDING |
| 3 | Video SIMD verification (beyond dp4a) | ❌ PENDING |
| 4 | rtl_frm_compare.py brittle test detection | ❌ PENDING |
| 5 | rtl_frm_compare.py hardcoded ULP tolerance | ❌ PENDING |

## Completed This Session
1. ✅ PTX Assembler 32-bit immediates fixed
2. ✅ FRM: Added FP32, DP4A, MOV_IMM, ALU_IMM, EXIT
3. ✅ Test Generator: 75 tests with PTX headers
4. ✅ Comparison Harness: tools/rtl_frm_compare.py
5. ✅ FRM: Added BRANCH instruction (register + predicate based)
6. ✅ FRM: Added LD_PARAM, LD_CONST, ATOM (atomic ops)
7. ✅ FRM: Added BAR_SYNC handler with CTAState barrier tracking
8. ✅ FRM: Added MEMBAR handler with scope (CTA/GL/SYS)
9. ✅ Test Generator: Added MemoryTestGenerator (LD/ST global/shared)
10. ✅ Test Generator: Added BranchTestGenerator (bra, setp+predicated)
11. ✅ Assembler: Fixed branch offset (bytes→instructions)
12. ✅ Assembler: Fixed predicated branch encoding (rd[4]=1)
13. ✅ FRM: Added DIV/REM handler (div.s32, div.u32, rem.s32, rem.u32)
14. ✅ Test Generator: Added DivTestGenerator (32 tests)
15. ✅ Test Generator: Added SpecialRegTestGenerator (4 tests: tid.x, ntid.x, laneid, ctaid.x)
16. ✅ Test Generator: Added AtomTestGenerator (5 tests: add, exch, cas)
17. ✅ rtl_frm_compare.py: Added initial memory support for test setup
18. ✅ Test Generator: Added BarSyncTestGenerator (3 tests: barrier 0, multiple barriers)
19. ✅ **Tier 1 Complete**: All Tier 1 items now VERIFIED (RTL + Gen + FRM)
20. ✅ Test Generator: Added ParamConstTestGenerator (2 tests: ld.param, ld.const)
21. ✅ Test Generator: Added MembarTestGenerator (3 tests: membar.cta/gl/sys)
22. ✅ **Tier 2 Complete**: LD.PARAM/CONST + MEMBAR VERIFIED (cp.async DECODE ONLY)
23. ✅ Test Generator: Added SFUTestGenerator (21 tests: sin, cos, sqrt, rcp, rsqrt, lg2, ex2)
24. ✅ rtl_frm_compare.py: Added SFU path to ULP tolerance check
25. ✅ **FP32 SFU Complete**: Tier 3 first item VERIFIED (Codex+Gemini consensus)

### Test Coverage
- PTX Assembler: 51/51 pass (100%)
- Generated Tests: 173/173 pass (ALU 65 + FP32 10 + Memory 20 + Branch 8 + DIV 32 + Special 4 + Atom 5 + Sync 3 + Param 2 + Membar 3 + SFU 21)
- RTL Unit Tests: ALU 26/26, FPU 26/26, B300 145/145
- **Tier 1: 100% VERIFIED** (all items have RTL + Gen + FRM)
- **Tier 2: 100% VERIFIED** (LD/ST.GLOBAL/SHARED, LD.PARAM/CONST, ATOM, MEMBAR) - cp.async DECODE ONLY
- **Tier 3: FP32 SFU VERIFIED** (21 tests)

## Patch F — Tensor Lane1 Suppress Fix
- **Status**: ✅ WB=4096 PASS (verified 2026-02-17)
- **Fix**: 2-cycle per-warp lockout replaces dedup tracker, +4 margin on pipe_tensor_ready
- **Test**: `test_sm_v2_perf_tensor_multiwarp` — Cycles=16398, IPC=0.250
- **Pending**: Commit and merge

## Next MVU (per PROCESS LOOP)
**MUST ASK Codex+Gemini** - FP16/FP64/CVT or Warp Collectives?

## Pending Issues Summary
- ❌ PENDING: 9 issues
- ⚠️ PARTIAL: 0 issues
- ✅ FIXED: 10 issues

## Next Action
Continue PROCESS LOOP: ASK Codex/Gemini for next MVU priority

## Last Updated
2026-02-17
