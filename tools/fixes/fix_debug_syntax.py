#!/usr/bin/env python3
import sys

def fix_debug(lines):
    """Fix the corrupted debug display"""
    result = []
    i = 0
    skip_until_endif = False
    
    while i < len(lines):
        line = lines[i]
        
        # Find and replace the broken section
        if '`ifdef SIMULATION' in line and i > 0:
            # Check next few lines for corruption
            if i+1 < len(lines) and 'if (|warp_inst_buf_valid)' in lines[i+1]:
                # Found the broken section - replace it
                result.append(line)  # `ifdef SIMULATION
                result.append('        if (|warp_inst_buf_valid) begin\n')
                result.append('            $display("[%0t SM%0d SCHED] issue0_fire=%b buf_valid=%b sched_mask=%b",\n')
                result.append('                     $time, SM_ID, issue0_fire, warp_inst_buf_valid, sched_issue_valid_mask);\n')
                result.append('            // Debug: Why isn\'t warp 0 ready?\n')
                result.append('            if (warp_inst_buf_valid[0] && !sched_issue_valid_mask[0])\n')
                result.append('                $display("[%0t SM%0d WARP_STALL] warp=0 valid=%b ready=%b stall_mem=%b stall_fu=%b stall_sync=%b stall_branch=%b exit_pend=%b",\n')
                result.append('                         $time, SM_ID, warp_valid[0], warp_ready[0], warp_stalled_mem[0], warp_stalled_fu[0], warp_stalled_sync[0], warp_stalled_branch[0], warp_exit_pending[0]);\n')
                result.append('        end\n')
                result.append('        `endif\n')
                
                # Skip the broken lines
                while i < len(lines) and '`endif' not in lines[i]:
                    i += 1
                i += 1  # Skip the `endif
                continue
        
        result.append(line)
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = fix_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Fixed debug syntax in {sys.argv[1]}")
