# Plan: Issue #149 — Pipeline Replay / Scheduler Fairness (WB Precision)

**Goal:** Achieve exact WB=4096 in test_sm_v2_perf_tensor_multiwarp.
**Branch:** issue-149/opus

---

## Task 1: Per-Warp Issue Sequence Number (ISN) — Stale Tensor Push Elimination

### 1.1 Add ISN counter in scheduler (advanced_scheduler.v + blackwell_scheduler.v)

Add per-warp 4-bit counter incremented on `warp_inst_consume`:

```verilog
reg [3:0] issue_seq [0:NUM_WARPS-1];

// In scoreboard update block:
for (sb_w = 0; sb_w < NUM_ISSUE; sb_w = sb_w + 1) begin
    if (warp_consume_r[issue_warp_r[sb_w]])
        issue_seq[issue_warp_r[sb_w]] <= issue_seq[issue_warp_r[sb_w]] + 1;
end
```

Expose via output port:
```verilog
output wire [NUM_WARPS*4-1:0] issue_seq_out
```

### 1.2 Propagate ISN through decode/issue pipeline (streaming_multiprocessor_v2.v)

Add `dec0_isn`, `dec1_isn` pipeline regs (4 bits each).
Capture scheduler's ISN for the selected warp when instruction enters decode:

```verilog
reg [3:0] dec0_isn, dec1_isn;
reg [3:0] issue_isn, issue1_isn;

// In dec0 capture block:
if (issue0_fire && !decode_stalled_any) begin
    dec0_isn <= sched_issue_seq[sched_issue_warp_id[0]];
end
// In issue capture block:
if (dec_valid && !branch_flush_dec0) begin
    issue_isn <= dec0_isn;
end
```

### 1.3 Gate tensor push with ISN check (streaming_multiprocessor_v2.v)

Add per-warp "last pushed ISN" tracker:
```verilog
reg [3:0] tensor_last_pushed_isn [0:NUM_WARPS-1];

wire isn_mismatch_lane0 = (issue_isn != tensor_last_pushed_isn[issue_warp_id]);
wire isn_mismatch_lane1 = (issue1_isn != tensor_last_pushed_isn[issue1_warp_id]);

// Replace current tensor push logic:
wire tensor_push_lane0 = tensor_push_lane0_raw && isn_mismatch_lane0;
wire tensor_push_lane1 = tensor_push_lane1_raw && isn_mismatch_lane1;
// Remove old lockout (or keep as belt-and-suspenders)
```

Update last-pushed on successful push:
```verilog
if (tensor_issue_push_fire) begin
    if (tensor_push_lane0)
        tensor_last_pushed_isn[issue_warp_id] <= issue_isn;
    else if (tensor_push_lane1)
        tensor_last_pushed_isn[issue1_warp_id] <= issue1_isn;
end
```

### 1.4 Wiring: Connect scheduler ISN output to SM v2

In scheduler instantiation, add:
```verilog
.issue_seq_out(sched_issue_seq_flat)
```

Unpack into array:
```verilog
wire [3:0] sched_issue_seq [0:NUM_WARPS-1];
wire [NUM_WARPS*4-1:0] sched_issue_seq_flat;
genvar si;
for (si = 0; si < NUM_WARPS; si = si + 1) begin
    assign sched_issue_seq[si] = sched_issue_seq_flat[si*4 +: 4];
end
```

---

## Task 2: Remove Old Lockout (Optional — can keep for safety)

- [ ] Remove `tensor_push_lockout_0/1` and `tensor_push_locked`
- [ ] Or keep them as additional guard (belt-and-suspenders)

Decision: Keep lockout on lane1 only. ISN handles lane0 precisely.

---

## Task 3: Lint Check

- [ ] Run `make lint` to verify no regressions

---

## Task 4: Test Verification

- [ ] Run `make test_sm_v2_perf_tensor_multiwarp` → WB=4096 exact
- [ ] Run full regression `make test_all` or key tests to verify no regressions
- [ ] Note: iverilog not currently on ist-mac-s PATH — may need to locate or install

---

## Task 5: PR

- [ ] Commit changes to `issue-149/opus` branch
- [ ] Create PR with `Closes #149`
- [ ] Report to #ralphgpu-dev

---

## Risk Assessment

- **False suppression:** ISN comparison is precise — new loop iterations get a new ISN
  (because warp_inst_consume fires on each iteration), so no false suppression.
- **Wrapping:** 4-bit ISN wraps at 16. Pipeline depth is ~3-4 stages. Even with stalls,
  the ISN distance between stale and fresh is 0 (stale has SAME ISN as the already-pushed
  instruction). So wrapping is not a concern — we're checking equality, not ordering.
- **IPC regression:** Zero — the ISN check is purely additive gating on the push path.
  Legitimate instructions always have a new ISN and pass the check.
