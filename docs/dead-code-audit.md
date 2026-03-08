# RTL Dead Code Audit Report

**Date**: 2026-03-09
**Branch**: `issue-609/alex`
**Tool**: Verilator 5.044 with `-Wall`
**Top module**: `ralph_gpu_top`

## Summary

| Warning Category | Count |
|-----------------|-------|
| UNUSEDSIGNAL    | 510   |
| UNUSEDPARAM     | 79    |
| UNDRIVEN        | 0     |
| **Total**       | **589** |

No undriven signals detected — all declared signals are assigned somewhere.

## Top 10 Modules by UNUSEDSIGNAL Count

| Rank | Module | Unused Signals | Unused Params | Total |
|------|--------|---------------|---------------|-------|
| 1 | streaming_multiprocessor_v2.v | 223 | 2 | 225 |
| 2 | ralph_gpu_top.v | 44 | 8 | 52 |
| 3 | fpu64.v | 25 | 11 | 36 |
| 4 | fp16_unit.v | 19 | 8 | 27 |
| 5 | wgmma.v | 16 | 4 | 20 |
| 6 | video_unit.v | 15 | 0 | 15 |
| 7 | texture_unit.v | 15 | 6 | 21 |
| 8 | sfu.v | 13 | 2 | 15 |
| 9 | branch_predictor.v | 12 | 1 | 13 |
| 10 | tlb_enhanced.v | 11 | 4 | 15 |

## Full Module Breakdown — UNUSEDSIGNAL

| Module | Count |
|--------|-------|
| streaming_multiprocessor_v2.v | 223 |
| ralph_gpu_top.v | 44 |
| fpu64.v | 25 |
| fp16_unit.v | 19 |
| wgmma.v | 16 |
| video_unit.v | 15 |
| texture_unit.v | 15 |
| sfu.v | 13 |
| branch_predictor.v | 12 |
| tlb_enhanced.v | 11 |
| memory_interface_wide.v | 11 |
| shared_memory.v | 9 |
| memory_controller_hbm.v | 9 |
| l1_data_cache.v | 8 |
| fpu.v | 8 |
| command_processor.v | 8 |
| tensor_core.v | 7 |
| alu.v | 7 |
| wgmma_tile_engine.v | 6 |
| sm_fetch_pipeline.v | 5 |
| memory_interface.v | 5 |
| mbarrier_unit.v | 5 |
| icache.v | 5 |
| async_copy_engine.v | 5 |
| register_file_banked.v | 4 |
| performance_counters.v | 4 |
| blackwell_scheduler.v | 3 |
| atomic_unit.v | 3 |
| mul_unit.v | 2 |
| memory_qos.v | 2 |
| sm_gmem_arbiter.v | 1 |

## Full Module Breakdown — UNUSEDPARAM

| Module | Count |
|--------|-------|
| fpu64.v | 11 |
| ralph_gpu_top.v | 8 |
| fp16_unit.v | 8 |
| memory_controller_hbm.v | 7 |
| texture_unit.v | 6 |
| mbarrier_unit.v | 5 |
| wgmma.v | 4 |
| tlb_enhanced.v | 4 |
| register_file_banked.v | 4 |
| wgmma_tile_engine.v | 3 |
| fpu.v | 3 |
| async_copy_engine.v | 3 |
| streaming_multiprocessor_v2.v | 2 |
| sfu.v | 2 |
| l1_data_cache.v | 2 |
| blackwell_scheduler.v | 2 |
| tma_unit.v | 1 |
| sm_writeback_arbiter.v | 1 |
| memory_interface_wide.v | 1 |
| memory_interface.v | 1 |
| branch_predictor.v | 1 |

## Recommendations

### High Priority (safe to clean up)

1. **streaming_multiprocessor_v2.v (223 unused signals)**: This module dominates the audit. Many signals appear to be internal debug/scaffolding wires or outputs from sub-blocks that are instantiated but not fully connected. A focused cleanup pass here would eliminate ~42% of all warnings.

2. **ralph_gpu_top.v (44 unused signals, 8 unused params)**: Top-level integration often has pass-through signals declared for future use. Signals that are genuinely unconnected at top level can be removed or replaced with explicit `/* verilator lint_off */` annotations if intentionally reserved.

3. **fpu64.v (25 unused signals, 11 unused params)**: FP64 constants (bias, special values) are commonly pre-declared for completeness but only a subset is used. Safe to remove unused constant params.

### Medium Priority

4. **fp16_unit.v (19+8)**: Similar pattern to fpu64 — pre-declared FP16/BF16 constants not all referenced.

5. **wgmma.v (16+4)** and **wgmma_tile_engine.v (6+3)**: Matrix multiply units have internal pipeline signals that may be partially connected.

6. **video_unit.v (15)** and **texture_unit.v (15+6)**: These specialized units likely have stubbed-out features.

### Low Priority (likely intentional)

7. **branch_predictor.v**, **sfu.v**, **tlb_enhanced.v**: Small counts, likely reserved signals for future microarchitectural features.

### Approach

- Start with unused params in FP units (fpu64, fp16_unit, fpu) — these are constant definitions and lowest risk to remove.
- For streaming_multiprocessor_v2.v, audit in sections (fetch, decode, execute, writeback) rather than all at once.
- Do NOT remove signals that are part of module port interfaces without checking all instantiation sites.

## Raw Data

Full lint output saved in `dead_code_audit.txt` (593 lines).
