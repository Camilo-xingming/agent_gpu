#!/usr/bin/env python3
"""Wire perf_sched_stall_ifetch in the blackwell_scheduler instantiation."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    lines = f.readlines()

# Find the FIRST .scoreboard_out(sched_scoreboard) and add perf port before it
found_first = False
out = []
for i, line in enumerate(lines):
    if not found_first and '.scoreboard_out(sched_scoreboard)' in line:
        out.append('        .perf_sched_stall_ifetch(sched_perf_stall_ifetch),\n')
        found_first = True
    out.append(line)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.writelines(out)
print("Added perf_sched_stall_ifetch to first blackwell_scheduler instance")
