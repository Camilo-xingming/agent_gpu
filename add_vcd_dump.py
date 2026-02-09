#!/usr/bin/env python3
import sys

def add_vcd_dump(lines):
    """Add VCD dump to testbench for waveform analysis"""
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        result.append(line)
        
        # Add VCD dump after initial begin
        if 'initial begin' in line and i > 0:
            # Check if VCD already exists
            if i+1 < len(lines) and '$dumpfile' not in lines[i+1]:
                indent = '    '
                result.append(f'{indent}$dumpfile("atomic_contention.vcd");\n')
                result.append(f'{indent}$dumpvars(0, tb_atomic_contention_minimal);\n')
                result.append(f'{indent}// Waveform debugging enabled\n')
        
        i += 1
    
    return result

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    
    fixed = add_vcd_dump(lines)
    
    with open(sys.argv[1], 'w') as f:
        f.writelines(fixed)
    
    print(f"Added VCD dump to {sys.argv[1]}")
