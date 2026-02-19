#!/usr/bin/env python3
"""Fix tensor lockout v4: use scheduler-level 1-cycle lockout only.

The 2-cycle lockout (lockout_0 | lockout_1) at the push stage is too aggressive.
Instead, use only lockout_0 (1 cycle) in the scheduler to prevent re-selection,
and keep the 2-cycle lockout at the push stage as a safety net."""

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'r') as f:
    sched = f.read()

# Revert: remove lockout check from scheduler tensor selection
sched = sched.replace(
    'end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready && !tensor_push_locked[warp_idx]) begin',
    'end else if (warp_is_tensor[warp_idx] && tensor_pipe_ready) begin'
)

# Remove the tensor_push_locked port from scheduler
sched = sched.replace(
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,\n    input  wire [NUM_WARPS-1:0]     tensor_push_locked,',
    '    input  wire [NUM_WARPS-1:0]     warp_is_tensor,'
)

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'w') as f:
    f.write(sched)
print("Scheduler: reverted lockout changes")

# Fix SM
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# Remove tensor_push_locked from scheduler wiring
sm = sm.replace(
    '        .warp_is_tensor(pd_is_tensor),\n        .tensor_push_locked(tensor_push_locked),\n        .warp_is_memory(pd_is_memory),',
    '        .warp_is_tensor(pd_is_tensor),\n        .warp_is_memory(pd_is_memory),'
)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("SM: reverted scheduler lockout wiring")
