#!/usr/bin/env python3
import sys

with open('rtl/atomic_unit.v', 'r') as f:
    lines = f.readlines()

result = []
for line in lines:
    result.append(line)
    
    # Debug pending mask init
    if 'pending_mask <= lane_mask;' in line:
        result.append('                    `ifdef SIMULATION\n')
        result.append('                    $display("[%0t ATOMIC] START: func=%0d addr=0x%08x mask=%b", $time, func, addr[0+:32], lane_mask);\n')
        result.append('                    `endif\n')

    # Debug loop next lane
    if 'current_lane <= find_next_lane' in line:
        result.append('                        `ifdef SIMULATION\n')
        result.append('                        $display("[%0t ATOMIC] Next lane selection. Current=%0d Pending=%b", $time, current_lane, pending_mask);\n')
        result.append('                        `endif\n')

    # Debug write done
    if 'WRITE_WAIT: begin' in line:
        # Find where write happens
        pass 

with open('rtl/atomic_unit.v', 'w') as f:
    f.writelines(result)

print("Added atomic serialization debug")
