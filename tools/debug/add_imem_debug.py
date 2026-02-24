#!/usr/bin/env python3

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    lines = f.readlines()

result = []
for i, line in enumerate(lines):
    result.append(line)
    
    # Add debug after imem_req check
    if 'if (imem_req) begin' in line and i > 0:
        result.append('                $display("[%0t TB] IMEM_REQ: addr=0x%08x", $time, imem_addr);\n')
    
    # Add debug after imem response
    if 'imem_data <= {imem[(imem_req_addr_d >> 2) + 1], imem[imem_req_addr_d >> 2]};' in line:
        result.append('                $display("[%0t TB] IMEM_RESP: addr=0x%08x data=0x%016x", $time, imem_req_addr_d, {imem[(imem_req_addr_d >> 2) + 1], imem[imem_req_addr_d >> 2]});\n')

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.writelines(result)

print("Added imem debug")
