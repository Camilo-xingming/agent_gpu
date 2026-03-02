# Cache Baseline (Issue #325)

Date: 2026-03-02  
Branch: `issue-325/codex`

## Scope
- Collect L1D/L2 hit-miss baseline from representative workloads.
- Record miss-rate observations and tuning direction.

## Method
Commands used:
- `make test_vector_add`
- `make test_perf_saxpy TB_L1D_BYPASS=0`
- `make test_sm_v2_perf_tensor_multiwarp`

Each workload prints a standardized line:
- `CacheStats: L1_hits=... L1_misses=... L1_hit_rate=... L2_hits=... L2_misses=... L2_hit_rate=...`

## Baseline Data
| Workload | Status | Cycles | Instr/Issues | IPC | L1 Hits | L1 Misses | L1 Hit Rate | L2 Hits | L2 Misses | L2 Hit Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vector_add | PASS | 201 | 14 | 0.06 | 3 | 2 | 60.0% | 0 | 0 | N/A |
| saxpy | PASS | 198 | 14 | 0.071 | 3 | 2 | 60.0% | 0 | 0 | N/A |
| tensor_multiwarp | FAIL (known WB-count issue) | 12324 | 2052 | 0.250 | 0 | 0 | N/A | 0 | 0 | N/A |

Notes:
- `tensor_multiwarp` is compute-dominant in this TB; no L1/L2 data traffic was observed.
- L2 counters remain zero for these three workloads in current top-level flow.

## Interpretation
- For memory-bearing kernels (`vector_add`, `saxpy`), current L1D hit rate is above 50% baseline threshold.
- L2 hit/miss currently does not provide useful baseline signal for this workload set.

## Tuning / Follow-up Proposal
Because a measured cache hit rate below 50% would trigger tuning actions, the next actionable experiments are:
1. L1D pressure experiment: run larger-stride/working-set kernels and recheck whether L1 hit rate drops below 50%.
2. If L1 < 50% on those kernels, try `l1_data_cache` parameter sweep:
   - `CACHE_SIZE_KB`: 16 -> 32
   - `NUM_WAYS`: 4 -> 8
3. Enable meaningful L2 baseline first (workload + wiring path), then tune `L2_SIZE_KB` / `L2_WAYS` once non-zero L2 accesses are observed.
