#!/usr/bin/env python3
"""Fix tensor lockout: gate lane1_tensor_can_push by lockout state.

Root cause: lane1_tensor_can_push only checks tensor_issue_full_next, not lockout.
When the warp is locked out, the tensor op passes through decode/issue, gets consumed,
but the tensor push is suppressed -> lost writeback.

Also gate lane0_stall_tensor similarly for lane0 lockout."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# 1. Revert the previous tensor_issue_conflict change (it was wrong approach)
sm = sm.replace(
    """        .tensor_issue_conflict((sched_issue_valid_mask[0] && sched_issue_valid_mask[1] &&
                                sched_issue_pipe[0] == 3'd2 && sched_issue_pipe[1] == 3'd2) ||
                               tensor_lockout_suppress_lane1),""",
    """        .tensor_issue_conflict(sched_issue_valid_mask[0] && sched_issue_valid_mask[1] &&
                               sched_issue_pipe[0] == 3'd2 && sched_issue_pipe[1] == 3'd2),"""
)

# 2. Fix lane1_tensor_can_push: add lockout check
# The lockout signal uses dec1_warp_id which is available at decode stage
sm = sm.replace(
    '    wire lane1_tensor_can_push = dec1_valid && dec1_tensor_op && !tensor_issue_full_next;',
    '    wire lane1_tensor_can_push = dec1_valid && dec1_tensor_op && !tensor_issue_full_next && !tensor_push_locked[dec1_warp_id];'
)

# 3. Also fix lane0_stall_tensor: add lockout check
sm = sm.replace(
    '    wire lane0_stall_tensor = dec0_valid && dec_tensor_op && tensor_issue_full_next;',
    '    wire lane0_stall_tensor = dec0_valid && dec_tensor_op && (tensor_issue_full_next || tensor_push_locked[dec0_warp_id]);'
)

# 4. Also fix lane1_stall_tensor: add lockout check
sm = sm.replace(
    '    wire lane1_stall_tensor = dec1_valid && dec1_tensor_op && tensor_issue_full_next;',
    '    wire lane1_stall_tensor = dec1_valid && dec1_tensor_op && (tensor_issue_full_next || tensor_push_locked[dec1_warp_id]);'
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("Tensor lockout decode stall fix applied")
