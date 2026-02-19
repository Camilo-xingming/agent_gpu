#!/usr/bin/env python3
"""Remove tensor_push_locked from advanced_warp_scheduler instance (it doesn't have this port)."""

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'r') as f:
    lines = f.readlines()

# Find the second occurrence of tensor_push_locked in scheduler wiring
# It's after the `else (advanced_warp_scheduler)
found_first = False
out = []
for line in lines:
    if 'tensor_push_locked(tensor_push_locked)' in line:
        if found_first:
            # Skip this line (second instance, in advanced_warp_scheduler)
            continue
        found_first = True
    out.append(line)

with open('/Users/jerry/RalphGPU/rtl/streaming_multiprocessor_v2.v', 'w') as f:
    f.writelines(out)
print("Removed tensor_push_locked from advanced_warp_scheduler instance")
