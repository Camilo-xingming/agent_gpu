#!/usr/bin/env python3
import sys

with open('rtl/atomic_unit.v', 'r') as f:
    lines = f.readlines()

result = []
for i, line in enumerate(lines):
    result.append(line)
    
    # Add debug after state machine transitions
    if 'IDLE: begin' in line and i > 0:
        result.append('                    `ifdef SIMULATION\n')
        result.append('                    if (req_valid)\n')
        result.append('                        $display("[%0t ATOMIC] REQ: func=%0d lanes=%b", $time, func, lane_mask);\n')
        result.append('                    `endif\n')
    
    # Add debug in READ_WAIT
    if 'READ_WAIT: begin' in line:
        result.append('                    `ifdef SIMULATION\n')
        result.append('                    if (!mem_ready)\n')
        result.append('                        $display("[%0t ATOMIC] Waiting for READ: lane=%0d addr=0x%08x", $time, current_lane, addr_lane);\n')
        result.append('                    `endif\n')

with open('rtl/atomic_unit.v', 'w') as f:
    f.writelines(result)

print("Added atomic unit debug")
