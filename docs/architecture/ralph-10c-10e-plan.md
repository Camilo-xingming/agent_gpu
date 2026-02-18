# RALPH-10c~10e Patch Skeleton and Acceptance

## Scope
This document is the handoff skeleton to continue after RALPH-10b lands.
Focus is limited to fetch/scheduler timing cleanup and measurable perf recovery.

## 10c - Same-Cycle Fill/NIB Visibility to Scheduler

Goal:
- Eliminate the 1-cycle IFetch bubble caused by scheduler only seeing registered `buf_valid`.

Minimal patch boundary:
- `rtl/streaming_multiprocessor_v2.v`
- `rtl/blackwell_scheduler.v`

Patch skeleton:
1. Add combinational eligibility path for same-cycle fill visibility:
   - Introduce explicit "effective valid" signals (`buf_valid_effective` style) that combine registered buffer valid + same-cycle fill/NIB events.
2. Keep existing issue gating and hazard checks unchanged.
3. No scoreboard policy changes in 10c.

Guardrails:
- No new dual-write path to warp PC.
- No changes to FU arbitration behavior.

## 10d - Fetch/Issue Handshake Stabilization

Goal:
- Ensure fetch-side advance and scheduler consume do not reintroduce over-fetch or hidden bubbles under backpressure.

Minimal patch boundary:
- `rtl/streaming_multiprocessor_v2.v`

Patch skeleton:
1. Centralize fetch-consume handshake conditions into one derived signal block.
2. Make same-cycle event priority explicit in comments and code ordering.
3. Add lightweight debug counters/wires (if needed) behind existing debug macros only.

Guardrails:
- Preserve branch priority: `branch_flush > branch_taken > fetch/nib advance`.
- Keep testbench interface stable.

## 10e - Regression Hardening + Signoff

Goal:
- Lock in behavior with repeatable metrics and low-noise pass/fail criteria.

Deliverables:
1. Acceptance script: `scripts/ralph10_acceptance.py`
2. Baseline file: `scripts/ralph10_baseline_10a.json`
3. Runbook command examples for CI/local.

Signoff criteria (relative to 10a baseline):
- `Fetches` should stay near issue scale and not regress toward 2x behavior.
- `FU stalls` should be significantly below regression level and close to baseline band.
- `WB` should recover toward 10a band.

## Execution Order
1. Merge 10b minimal PC-advance fix.
2. Apply 10c visibility bypass.
3. Apply 10d handshake stabilization.
4. Run 10e acceptance script and attach metrics summary.

## Acceptance Commands
```bash
# Default run (uses baseline file and test_sm_v2_perf_gemm16_ptx)
python3 scripts/ralph10_acceptance.py

# Custom target/baseline
python3 scripts/ralph10_acceptance.py \
  --target test_sm_v2_perf_tensor \
  --baseline scripts/ralph10_baseline_10a.json
```
