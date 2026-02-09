#!/usr/bin/env python3

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    lines = f.readlines()

# Find the last endmodule
endmodule_idx = -1
for i in range(len(lines) - 1, -1, -1):
    if 'endmodule' in lines[i]:
        endmodule_idx = i
        break

if endmodule_idx != -1:
    # Everything after endmodule is the new initial block
    extra_content = lines[endmodule_idx+1:]
    # Remove endmodule
    lines = lines[:endmodule_idx] + extra_content + ['\nendmodule\n']

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.writelines(lines)

print("Fixed testbench structure")
