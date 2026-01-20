# RalphGPU Status

## State Snapshot
- **Current MVU**: FRM Expansion for Memory/Branch (PENDING)
- **Status**: IN PROGRESS - Verification infrastructure done, FRM gaps remain
- **Verification Gap**: Reduced but significant gaps in FRM coverage

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
| 11 | Test generators missing branch/memory/warp cases (ALU/FP32 only) | ❌ PENDING |
| 12 | FRM missing FP16/FP64/CVT and INT div/rem execution | ❌ PENDING |

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

### Test Coverage
- PTX Assembler: 51/51 pass (100%)
- Generated Tests: 75/75 pass (ALU + FP32)
- Generated FP32 Tests: 10/10 pass (100%)
- RTL Unit Tests: ALU 26/26, FPU 26/26, B300 145/145

## Next MVU (per PROCESS LOOP)
**FRM Memory/Branch Fidelity** - Add per-thread branch divergence + reconvergence, BAR.SYNC/MEMBAR semantics, and LD/ST.shared/global coverage with matching generators/FRM

## Pending Issues Summary
- ❌ PENDING: 11 issues
- ⚠️ PARTIAL: 0 issues
- ✅ FIXED: 6 issues

## Next Action
Continue PROCESS LOOP: ASK Codex/Gemini for next MVU priority

## Last Updated
2026-01-20
