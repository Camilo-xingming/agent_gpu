#!/usr/bin/env python3
import sys

with open('rtl/atomic_unit.v', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    
    # IDLE debug
    if 'current_lane <= find_next_lane(6\'d0, lane_mask);' in line:
        new_lines.append('                        `ifdef SIMULATION\n')
        new_lines.append('                        $display("[%0t ATOMIC] START: func=%0d addr=0x%08x mask=%b", $time, func, addr[0+:32], lane_mask);\n')
        new_lines.append('                        `endif\n')

    # WRITE_WAIT debug (next lane logic)
    if 'if ((pending_mask & ~lane_onehot(current_lane)) != 0) begin' in line:
        new_lines.insert(len(new_lines)-1, '                        `ifdef SIMULATION\n')
        new_lines.insert(len(new_lines)-1, '                        $display("[%0t ATOMIC] Check next: pending=%b current=%d", $time, pending_mask, current_lane);\n')
        new_lines.insert(len(new_lines)-1, '                        `endif\n')

with open('rtl/atomic_unit.v', 'w') as f:
    f.writelines(new_lines)

print("Added safe atomic debug")
