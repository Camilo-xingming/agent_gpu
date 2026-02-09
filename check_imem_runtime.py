#!/usr/bin/env python3

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    lines = f.readlines()

result = []
for i, line in enumerate(lines):
    result.append(line)
    
    # Add runtime check in imem response
    if '$display("[%0t TB] IMEM_RESP:' in line:
        result.append('                $display("[%0t TB] imem[0]=0x%08x imem[1]=0x%08x (current values)", $time, imem[0], imem[1]);\n')

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.writelines(result)

print("Added runtime imem check")
