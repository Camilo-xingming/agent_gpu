#!/usr/bin/env python3
import sys

def fix_branch_stall(lines):
    """Remove the premature branch stall setting"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Comment out the lines that set branch stall at schedule time
        if 'if (issue0_fire && pd_is_branch[sched_issue_warp_id[0]] && !branch_flush_dec0) begin' in line:
            # Comment out this block
            result.append('            // DISABLED: Branch stall no longer set at schedule time\n')
            result.append('            // Fix for pipeline timing issue - stall was set before branch reached issue stage\n')
            result.append('            // ' + line)
            i += 1
            # Comment out the next line (warp_stalled_branch assignment)
            while i < len(lines) and 'end' not in lines[i]:
                result.append('            // ' + lines[i])
                i += 1
            if i < len(lines):
                result.append('            // ' + lines[i])  # The 'end'
                i += 1
            continue
        
        # Same for issue1
        if 'if (issue1_fire && pd_is_branch[sched_issue_warp_id[1]] && !branch_flush_dec1) begin' in line:
            result.append('            // DISABLED: Branch stall no longer set at schedule time\n')
            result.append('            // ' + line)
            i += 1
            while i < len(lines) and 'end' not in lines[i]:
                result.append('            // ' + lines[i])
                i += 1
            if i < len(lines):
                result.append('            // ' + lines[i])
                i += 1
            continue
        
        result.append(line)
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = fix_branch_stall(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Fixed branch stall timing issue in {sys.argv[1]}")
