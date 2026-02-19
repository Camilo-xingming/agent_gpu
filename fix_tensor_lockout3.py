#!/usr/bin/env python3
"""Fix tensor lockout: feed lockout state into scheduler eligibility.

Instead of suppressing at the push stage (which loses instructions),
prevent the scheduler from selecting locked-out tensor warps entirely.
The warp stays in the buffer, and when lockout expires, it gets re-selected."""

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'r') as f:
    sched = f.read()

# 1. Add tensor_push_locked input port (after warp_is_tensor)
sched = sched.replace(
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,',
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,\n    input  wire [NUM_WARPS-1:0]     tensor_push_locked,'
)

# 2. Gate tensor selection by lockout in the scheduler
# Find the tensor pipe selection and add lockout check
sched = sched.replace(
    'end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready) begin',
    'end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready && !tensor_push_locked[warp_idx]) begin'
)

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'w') as f:
    f.write(sched)
print("Scheduler: tensor lockout eligibility check added")

# === Fix SM wiring ===
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# 1. Revert the decode stall changes from fix_tensor_lockout2.py
sm = sm.replace(
    '    wire lane1_tensor_can_push = dec1_valid && dec1_tensor_op && !tensor_issue_full_next && !tensor_push_locked[dec1_warp_id];',
    '    wire lane1_tensor_can_push = dec1_valid && dec1_tensor_op && !tensor_issue_full_next;'
)
sm = sm.replace(
    '    wire lane0_stall_tensor = dec0_valid && dec_tensor_op && (tensor_issue_full_next || tensor_push_locked[dec0_warp_id]);',
    '    wire lane0_stall_tensor = dec0_valid && dec_tensor_op && tensor_issue_full_next;'
)
sm = sm.replace(
    '    wire lane1_stall_tensor = dec1_valid && dec1_tensor_op && (tensor_issue_full_next || tensor_push_locked[dec1_warp_id]);',
    '    wire lane1_stall_tensor = dec1_valid && dec1_tensor_op && tensor_issue_full_next;'
)

# 2. Wire tensor_push_locked to the blackwell_scheduler instance
# Find the .warp_is_tensor connection in the first scheduler instance
sm = sm.replace(
    '        .warp_is_tensor(pd_is_tensor),\n        .warp_is_memory(pd_is_memory),',
    '        .warp_is_tensor(pd_is_tensor),\n        .tensor_push_locked(tensor_push_locked),\n        .warp_is_memory(pd_is_memory),'
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("SM: tensor_push_locked wired to scheduler")
