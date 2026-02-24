#!/usr/bin/env python3
import sys

def add_debug(lines):
    """Add scheduler input debug"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add after the existing INPUTS debug
        if 'warp=0 INPUTS: valid=%b ready=%b eligible_base=%b eligible=%b' in line:
            # Find end of this display
            while i < len(lines) and ');' not in lines[i]:
                result.append(lines[i])
                i += 1
            if i < len(lines):
                result.append(lines[i])  # The ');' line
                i += 1
            # Now add new debug
            result.append('        if (warp_inst_valid[0] && !warp_eligible[0])\n')
            result.append('            $display("[%0t SCHED] warp=0 DETAILED: inst_valid=%b has_hazard=%b diverged=%b at_barrier=%b",\n')
            result.append('                     $time, warp_inst_valid[0], warp_has_hazard[0], warp_diverged[0], warp_at_barrier[0]);\n')
            continue
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added detailed scheduler debug to {sys.argv[1]}")
