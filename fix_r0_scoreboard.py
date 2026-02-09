#!/usr/bin/env python3
import sys

with open('rtl/blackwell_scheduler.v', 'r') as f:
    lines = f.readlines()

result = []
for i, line in enumerate(lines):
    # Fix line 384: add && warp_rd != 0 check
    if 'if (warp_writes_reg[issue_warp_r[sb_s]]) begin' in line:
        # Replace with version that checks for non-zero rd
        result.append('                    // Don\'t track R0 in scoreboard (R0 is hardwired to zero)\n')
        result.append('                    if (warp_writes_reg[issue_warp_r[sb_s]] && warp_rd[issue_warp_r[sb_s]] != 5\'b0) begin\n')
    else:
        result.append(line)

with open('rtl/blackwell_scheduler.v', 'w') as f:
    f.writelines(result)

print("Fixed R0 scoreboard tracking")
