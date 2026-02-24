#!/usr/bin/env python3
import re

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
initial_count = 0

for line in lines:
    # Remove the injected VCD dump blocks (except the main one if we want it, but better to have just one cleanly)
    if '$dumpfile("atomic_contention.vcd");' in line:
        continue
    if '$dumpvars(0, tb_atomic_contention_minimal);' in line:
        continue
    if '// Waveform debugging enabled' in line:
        continue
        
    new_lines.append(line)

# Now inject a single clean VCD dump at the top
final_lines = []
inserted_vcd = False
for line in new_lines:
    final_lines.append(line)
    if 'module tb_atomic_contention_minimal;' in line and not inserted_vcd:
        final_lines.append('\n')
        final_lines.append('    initial begin\n')
        final_lines.append('        $dumpfile("atomic_contention.vcd");\n')
        final_lines.append('        $dumpvars(0, tb_atomic_contention_minimal);\n')
        final_lines.append('    end\n')
        inserted_vcd = True

# Add a check event
final_lines.append('\n')
final_lines.append('    initial begin\n')
final_lines.append('        #1;\n')
final_lines.append('        $display("[TB] Time=1 imem[0]=0x%08x", imem[0]);\n')
final_lines.append('        #100;\n')
final_lines.append('        $display("[TB] Time=101 imem[0]=0x%08x", imem[0]);\n')
final_lines.append('    end\n')

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.writelines(final_lines)

print("Cleaned up testbench")
