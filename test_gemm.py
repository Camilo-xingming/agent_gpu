import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tools'))
from gpu_simulator import RalphGPUSimulator

def main():
    print('Testing 2x2 GEMM (Matrix Multiply)...')
    sim = RalphGPUSimulator(num_sm=1)
    
    sim.load_program('llm_gemm_2x2.hex')
    sim.run_kernel(entry_pc=0, max_cycles=1000)
    
    # Expected: C = [[19, 22], [43, 50]]
    c00 = sim.global_memory.get(0x1000, 0)
    c01 = sim.global_memory.get(0x1004, 0)
    c10 = sim.global_memory.get(0x1008, 0)
    c11 = sim.global_memory.get(0x100C, 0)
    
    print(f'Result C matrix:')
    print(f'  [{c00:3d}, {c01:3d}]')
    print(f'  [{c10:3d}, {c11:3d}]')
    print(f'Expected:')
    print(f'  [ 19,  22]')
    print(f'  [ 43,  50]')
    
    if c00 == 19 and c01 == 22 and c10 == 43 and c11 == 50:
        print('✅ GEMM Test PASSED')
    else:
        print('❌ GEMM Test FAILED')

if __name__ == '__main__':
    main()
