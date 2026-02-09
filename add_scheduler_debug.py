#!/usr/bin/env python3
import sys

def add_debug(lines):
    """Add scheduler eligibility debug"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add debug after warp_eligible assignment
        if 'wire [NUM_WARPS-1:0] warp_eligible = warp_eligible_base | warp_tcgen05_eligible;' in line:
            result.append('\n')
            result.append('    // Debug: Why is warp 0 not eligible?\n')
            result.append('    always @(posedge clk) begin\n')
            result.append('        `ifdef SIMULATION\n')
            result.append('        if (warp_inst_valid[0] && !warp_eligible[0])\n')
            result.append('            $display("[%0t SCHED] warp=0 NOT ELIGIBLE: inst_valid=%b has_hazard=%b diverged=%b at_barrier=%b",\n')
            result.append('                     $time, warp_inst_valid[0], warp_has_hazard[0], warp_diverged[0], warp_at_barrier[0]);\n')
            result.append('        `endif\n')
            result.append('    end\n')
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_debug(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added scheduler debug to {sys.argv[1]}")
