#!/usr/bin/env python3
import sys

def fix_dup(lines):
    """Remove duplicate display"""
    result = []
    prev_display = False
    
    for line in lines:
        # Skip duplicate display lines
        if '$display("[%0t SCHED] warp=0 NOT ELIGIBLE:' in line:
            if prev_display:
                # Skip this duplicate
                continue
            prev_display = True
        else:
            prev_display = False
        
        result.append(line)
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = fix_dup(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Fixed duplicate in {sys.argv[1]}")
