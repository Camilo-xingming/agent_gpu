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
| 3 | FRM missing shared/param/atomic/membar | ⚠️ PARTIAL (param/const/atom added) |
| 4 | FRM missing cp.async/st.async/mbarrier | ❌ PENDING |
| 5 | FRM missing warp collectives (SHFL/VOTE/REDUX) | ❌ PENDING |
| 6 | FRM missing tensor paths (WGMMA/TMA) | ❌ PENDING |
| 7 | RTL↔FRM comparison framework | ✅ FIXED |
| 8 | Python test generators | ✅ FIXED |

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

### Test Coverage
- PTX Assembler: 51/51 pass (100%)
- Generated Tests: 75/75 pass (ALU + FP32)
- Generated FP32 Tests: 10/10 pass (100%)
- RTL Unit Tests: ALU 26/26, FPU 26/26, B300 145/145

## Next MVU (per PROCESS LOOP)
**FRM Expansion: Memory + Branch** - Add LD/ST.shared, branch, atomics to FRM

## Pending Issues Summary
- ❌ PENDING: 9 issues
- ⚠️ PARTIAL: 0 issues
- ✅ FIXED: 5 issues

## Next Action
Continue PROCESS LOOP: ASK Codex/Gemini for next MVU priority

## Last Updated
2026-01-20
