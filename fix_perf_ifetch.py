#!/usr/bin/env python3
"""Fix perf_stall_ifetch metric: scheduler-centric instead of any-warp."""
import re

# === Fix blackwell_scheduler.v ===
with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'r') as f:
    sched = f.read()

# 1. Add new output port before scoreboard_out
old_port = '    output wire [31:0]              scoreboard_out [0:NUM_WARPS-1]'
new_port = """    //------------------------------------------------------------------------
    // Scheduler-centric IFetch stall (true when IFetch is the bottleneck)
    //------------------------------------------------------------------------
    output wire                     perf_sched_stall_ifetch,

    output wire [31:0]              scoreboard_out [0:NUM_WARPS-1]"""
sched = sched.replace(old_port, new_port)

# 2. Add the logic after warp_eligible definition
# Find the line after "wire [NUM_WARPS-1:0] warp_eligible = ..."
old_eligible = '    // Combined eligibility\n    wire [NUM_WARPS-1:0] warp_eligible = warp_eligible_base | warp_tcgen05_eligible;'
new_eligible = old_eligible + """

    // Scheduler-centric IFetch stall: no warp eligible, but at least one warp
    // WOULD be eligible if it had a valid instruction (IFetch is the bottleneck)
    wire [NUM_WARPS-1:0] warp_ifetch_blocked = warp_valid & warp_ready & ~warp_inst_valid
                                              & ~warp_has_hazard & ~warp_diverged & ~warp_at_barrier;
    assign perf_sched_stall_ifetch = (warp_eligible == {NUM_WARPS{1'b0}}) && (|warp_ifetch_blocked);"""
sched = sched.replace(old_eligible, new_eligible)

with open('/Users/jerry/RalphGPU/rtl/blackwell_scheduler.v', 'w') as f:
    f.write(sched)
print("blackwell_scheduler.v patched")

# === Fix streaming_multiprocessor_v2.v ===
with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    sm = f.read()

# 1. Add wire for scheduler output
# Find where sched_issue_valid_mask is declared
old_wire = '    wire [SCHED_LANES-1:0] sched_issue_valid_mask;'
new_wire = old_wire + '\n    wire sched_perf_stall_ifetch;'
sm = sm.replace(old_wire, new_wire)

# 2. Wire the new port in scheduler instantiation
# Find the .stat_stalls connection
old_inst = '        .stat_stalls(sched_stat_stalls),'
new_inst = old_inst + '\n        .perf_sched_stall_ifetch(sched_perf_stall_ifetch),'
sm = sm.replace(old_inst, new_inst)

# 3. Fix perf_stall_ifetch assignment
old_assign = "    assign perf_stall_ifetch       = |( warp_valid & ~warp_inst_buf_valid & ~warp_stalled_mem );"
new_assign = "    assign perf_stall_ifetch       = sched_perf_stall_ifetch;"
sm = sm.replace(old_assign, new_assign)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.write(sm)
print("streaming_multiprocessor_v2.v patched")
