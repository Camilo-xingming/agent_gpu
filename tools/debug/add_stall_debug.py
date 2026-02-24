#!/usr/bin/env python3
import sys

def add_stall_debug(lines):
    """Add debug output for warp stall conditions"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add debug after the existing sched debug display
        if 'issue0_fire=%b buf_valid=%b sched_mask=%b' in line and i+1 < len(lines):
            # Check if we already added this debug
            if i+1 < len(lines) and 'WARP_STALL' not in lines[i+1]:
                indent = '            '
                result.append(f'{indent}// Debug: Why isn\'t warp 0 ready?\n')
                result.append(f'{indent}if (warp_inst_buf_valid[0] && !sched_issue_valid_mask[0])\n')
                result.append(f'{indent}    $display("[%0t SM%0d WARP_STALL] warp=0 valid=%b ready=%b stall_mem=%b stall_fu=%b stall_sync=%b stall_branch=%b exit_pend=%b",\n')
                result.append(f'{indent}             $time, SM_ID, warp_valid[0], warp_ready[0], warp_stalled_mem[0], warp_stalled_fu[0], warp_stalled_sync[0], warp_stalled_branch[0], warp_exit_pending[0]);\n')
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_stall_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added stall debug to {sys.argv[1]}")
