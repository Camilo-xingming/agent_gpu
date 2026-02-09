#!/usr/bin/env python3
import sys

def add_debug(lines):
    """Add warp_ready debug to scheduler"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add debug after existing debug
        if '$display("[%0t SCHED] warp=0 NOT ELIGIBLE' in line:
            # Find the end of this display statement
            while i < len(lines) and ');' not in lines[i]:
                result.append(lines[i])
                i += 1
            # Add the closing
            if i < len(lines):
                result.append(lines[i])  # The ');' line
                i += 1
            # Now add new debug
            result.append('        if (warp_inst_valid[0] && !warp_eligible[0])\n')
            result.append('            $display("[%0t SCHED] warp=0 INPUTS: valid=%b ready=%b eligible_base=%b eligible=%b",\n')
            result.append('                     $time, warp_valid[0], warp_ready[0], warp_eligible_base[0], warp_eligible[0]);\n')
            continue
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added warp_ready debug to {sys.argv[1]}")
