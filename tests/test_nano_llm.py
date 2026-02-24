import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tools'))
from gpu_simulator import RalphGPUSimulator

def main():
    print('='*60)
    print('Testing Nano-LLM (Mini Transformer Layer)')
    print('='*60)
    
    sim = RalphGPUSimulator(num_sm=1)
    sim.load_program('programs/nano_llm.hex')
    sim.run_kernel(entry_pc=0, max_cycles=1000)
    
    # Read intermediate values
    x0 = sim.global_memory.get(0x1000, 0)
    x1 = sim.global_memory.get(0x1004, 0)
    query = sim.global_memory.get(0x1008, 0)
    key = sim.global_memory.get(0x100C, 0)
    score = sim.global_memory.get(0x1010, 0)
    output = sim.global_memory.get(0x2000, 0)
    
    print('\nForward Pass Trace:')
    print(f'  Input embedding:    x = [{x0}, {x1}]')
    print(f'  Query projection:   Q = {query}  (W_q·x)')
    print(f'  Key projection:     K = {key}  (W_k·x)')
    print(f'  Attention score:    S = {score}  (Q·K^T)')
    print(f'  After ReLU:         Y = {output}  (activation)')
    
    print('\nExpected values:')
    print('  Input:  [1, 2]')
    print('  Query:  1')
    print('  Key:    2') 
    print('  Score:  2')
    print('  Output: 2')
    
    if (x0 == 1 and x1 == 2 and query == 1 and 
        key == 2 and score == 2 and output == 2):
        print('\n✅ Nano-LLM Test PASSED')
        print('All transformer components working correctly!')
        return True
    else:
        print('\n❌ Nano-LLM Test FAILED')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
