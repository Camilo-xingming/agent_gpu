#!/usr/bin/env python3

with open('tb/tb_atomic_contention_minimal.v', 'r') as f:
    content = f.read()

# Fix the hex file path
content = content.replace('test_23_mem_consistency_atomicity.hex', 'atomic_divergent_test.hex')
content = content.replace('../hex/ptx_comprehensive_tests/', '../asm/')

# Add display after readmemh
content = content.replace(
    '$readmemh("../asm/atomic_divergent_test.hex", imem);',
    '''$readmemh("../asm/atomic_divergent_test.hex", imem);
        $display("[TB] Loaded imem[0]=0x%08x imem[1]=0x%08x imem[2]=0x%08x", imem[0], imem[1], imem[2]);'''
)

with open('tb/tb_atomic_contention_minimal.v', 'w') as f:
    f.write(content)

print("Fixed testbench")
