#!/usr/bin/env python3
import sys

def add_debug(lines):
    """Add debug for branch stall changes"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add debug after setting warp_stalled_branch
        if 'warp_stalled_branch[sched_issue_warp_id[0]] <= 1\'b1;' in line:
            result.append('                `ifdef SIMULATION\n')
            result.append('                $display("[%0t SM%0d] SET branch stall for warp=%0d (issue0)",\n')
            result.append('                         $time, SM_ID, sched_issue_warp_id[0]);\n')
            result.append('                `endif\n')
        
        # Add debug after clearing warp_stalled_branch (non-divergent)
        if 'warp_stalled_branch[issue_warp_id] <= 1\'b0;' in line and 'non-divergent' in ''.join(lines[max(0,i-10):i]):
            result.append('                    `ifdef SIMULATION\n')
            result.append('                    $display("[%0t SM%0d] CLEAR branch stall for warp=%0d (non-divergent resolve)",\n')
            result.append('                             $time, SM_ID, issue_warp_id);\n')
            result.append('                    `endif\n')
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added branch stall debug to {sys.argv[1]}")
