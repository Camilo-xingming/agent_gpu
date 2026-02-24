#!/usr/bin/env python3
"""
RalphGPU LLM Test Suite
Tests individual LLM operators and a complete nano-LLM
"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tools'))
from gpu_simulator import RalphGPUSimulator

def test_attention_score():
    print('\n' + '='*60)
    print('Test 1: Attention Score (Q·K^T with DP4A)')
    print('='*60)
    
    sim = RalphGPUSimulator(num_sm=1)
    sim.load_program('programs/llm_attention_score.hex')
    sim.run_kernel(entry_pc=0, max_cycles=1000)
    
    result = sim.global_memory.get(0x1000, 0)
    expected = 15
    
    print(f'Result: {result}')
    print(f'Expected: {expected}')
    
    if result == expected:
        print('✅ PASSED')
        return True
    else:
        print('❌ FAILED')
        return False

def test_gemm():
    print('\n' + '='*60)
    print('Test 2: Matrix Multiply (2x2 GEMM)')
    print('='*60)
    
    sim = RalphGPUSimulator(num_sm=1)
    sim.load_program('programs/llm_gemm_2x2.hex')
    sim.run_kernel(entry_pc=0, max_cycles=1000)
    
    c00 = sim.global_memory.get(0x1000, 0)
    c01 = sim.global_memory.get(0x1004, 0)
    c10 = sim.global_memory.get(0x1008, 0)
    c11 = sim.global_memory.get(0x100C, 0)
    
    print(f'Result: [[{c00}, {c01}], [{c10}, {c11}]]')
    print(f'Expected: [[19, 22], [43, 50]]')
    
    if c00 == 19 and c01 == 22 and c10 == 43 and c11 == 50:
        print('✅ PASSED')
        return True
    else:
        print('❌ FAILED')
        return False

def test_nano_llm():
    print('\n' + '='*60)
    print('Test 3: Nano-LLM (Mini Transformer)')
    print('='*60)
    
    sim = RalphGPUSimulator(num_sm=1)
    sim.load_program('programs/nano_llm.hex')
    sim.run_kernel(entry_pc=0, max_cycles=1000)
    
    output = sim.global_memory.get(0x2000, 0)
    expected = 2
    
    print(f'Result: {output}')
    print(f'Expected: {expected}')
    
    if output == expected:
        print('✅ PASSED')
        return True
    else:
        print('❌ FAILED')
        return False

def main():
    print('\n' + '#'*60)
    print('# RalphGPU LLM Test Suite')
    print('#'*60)
    
    results = []
    results.append(('Attention Score', test_attention_score()))
    results.append(('2x2 GEMM', test_gemm()))
    results.append(('Nano-LLM', test_nano_llm()))
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    for name, passed in results:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f'{name:20s} {status}')
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f'\nTotal: {passed}/{total} passed')
    
    if passed == total:
        print('\n🎉 All LLM tests passed! RalphGPU is working correctly.')
        return 0
    else:
        print('\n⚠️  Some tests failed.')
        return 1

if __name__ == '__main__':
    sys.exit(main())
