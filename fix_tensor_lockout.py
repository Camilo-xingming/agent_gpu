#!/usr/bin/env python3
"""Fix tensor lockout suppress: prevent consume when tensor push is locked out.

Root cause: when tensor_push_lane1 is suppressed by lockout, the scheduler
still consumes the instruction from the buffer, but the tensor op never gets
pushed to the tensor pipeline -> lost writeback.

Fix: wire tensor lockout suppress into the scheduler as an additional conflict."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# 1. Create a tensor_lockout_suppress_lane1 signal
# This fires when lane1 has a tensor op but it's locked out
# Add it near the tensor_push definitions
old_locked = "    wire [NUM_WARPS-1:0] tensor_push_locked = tensor_push_lockout_0 | tensor_push_lockout_1;"
new_locked = old_locked + """
    // Signal to scheduler: lane1 tensor is suppressed by lockout (don't consume)
    wire tensor_lockout_suppress_lane1 = tensor_push_lane1_raw && tensor_push_locked[issue1_warp_id];"""
sm = sm.replace(old_locked, new_locked)

# 2. Wire this into the scheduler's tensor_issue_conflict
# Currently: tensor_issue_conflict = both lanes tensor
# New: also include lockout suppress for lane1
old_conflict = """        .tensor_issue_conflict(sched_issue_valid_mask[0] && sched_issue_valid_mask[1] &&
                               sched_issue_pipe[0] == 3'd2 && sched_issue_pipe[1] == 3'd2),"""
new_conflict = """        .tensor_issue_conflict((sched_issue_valid_mask[0] && sched_issue_valid_mask[1] &&
                                sched_issue_pipe[0] == 3'd2 && sched_issue_pipe[1] == 3'd2) ||
                               tensor_lockout_suppress_lane1),"""
sm = sm.replace(old_conflict, new_conflict)

# Also suppress lane0 if it's locked out
old_locked2 = "    wire tensor_lockout_suppress_lane1 = tensor_push_lane1_raw && tensor_push_locked[issue1_warp_id];"
new_locked2 = old_locked2 + """
    wire tensor_lockout_suppress_lane0 = tensor_push_lane0_raw && tensor_push_locked[issue_warp_id];"""
sm = sm.replace(old_locked2, new_locked2)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Tensor lockout suppress fix applied")
