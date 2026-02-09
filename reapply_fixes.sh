#!/bin/bash
cd ~/RalphGPU

# Fix 1: Disable branch stall at schedule time
sed -i.bak '5289,5294s/^/\/\/ DISABLED: /' rtl/streaming_multiprocessor_v2.v

# Fix 2: Skip R0 in scoreboard
python3 -c "
with open('rtl/blackwell_scheduler.v', 'r') as f:
    lines = f.readlines()

result = []
for line in lines:
    if 'if (warp_writes_reg[issue_warp_r[sb_s]]) begin' in line:
        result.append('                    // Don\\'t track R0 in scoreboard (R0 is hardwired to zero)\n')
        result.append('                    if (warp_writes_reg[issue_warp_r[sb_s]] && warp_rd[issue_warp_r[sb_s]] != 5\\'b0) begin\n')
    else:
        result.append(line)

with open('rtl/blackwell_scheduler.v', 'w') as f:
    f.writelines(result)
"

echo "Fixes reapplied"
