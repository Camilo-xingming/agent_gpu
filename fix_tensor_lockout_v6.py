#!/usr/bin/env python3
"""Fix tensor lockout v6: remove lockout entirely, rely on tensor_issue_conflict.

The 2-cycle lockout was added to prevent 'pipeline echo double-push' but it
causes lost writebacks when combined with fill bypass. The tensor_issue_conflict
signal already prevents dual tensor issue in the same cycle.

Test hypothesis: is the lockout still needed, or does tensor_issue_conflict
+ deferred scoreboard already prevent double-push?"""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Disable the lockout by making tensor_push_locked always 0
old_locked = '    wire [NUM_WARPS-1:0] tensor_push_locked = tensor_push_lockout_0 | tensor_push_lockout_1;'
new_locked = '    wire [NUM_WARPS-1:0] tensor_push_locked = {NUM_WARPS{1\'b0}};  // DISABLED: rely on tensor_issue_conflict'
sm = sm.replace(old_locked, new_locked)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Tensor push lockout DISABLED")
