#!/usr/bin/env python3
"""Fix tensor lockout v7: reduce to 1-cycle lockout instead of 2-cycle.

The 2-cycle lockout (lockout_0 | lockout_1) is too aggressive with fill bypass,
causing 11 suppress events and lost WBs. A 1-cycle lockout (lockout_0 only)
should prevent the double-push while being less aggressive."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Change lockout from 2-cycle to 1-cycle
sm = sm.replace(
    "    wire [NUM_WARPS-1:0] tensor_push_locked = {NUM_WARPS{1'b0}};  // DISABLED: rely on tensor_issue_conflict",
    "    wire [NUM_WARPS-1:0] tensor_push_locked = tensor_push_lockout_0;  // 1-cycle lockout only"
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Tensor lockout: 1-cycle only")
