#!/usr/bin/env python3
"""Parse RalphGPU performance counter output and generate a report."""
import sys
import re

def parse_perf(lines):
    """Extract perf counters from simulation output."""
    counters = {}
    for line in lines:
        m = re.match(r'\s+(\S.*?)\s+=\s+(\d+)', line)
        if m:
            counters[m.group(1).strip()] = int(m.group(2))
        m = re.match(r'\s+IPC\s+=\s+\d+\s*/\s*\d+\s*=\s*([\d.]+)', line)
        if m:
            counters['IPC'] = float(m.group(1))
    return counters

def format_report(name, counters, freq_mhz=100):
    """Format a perf report for one workload."""
    cycles = counters.get('Cycles', 0)
    instr = counters.get('Instructions', 0)
    ipc = counters.get('IPC', instr/cycles if cycles else 0)
    
    stall_sb = counters.get('Stall: Scoreboard', 0)
    stall_if = counters.get('Stall: I-Fetch', 0)
    stall_mem = counters.get('Stall: Memory', 0)
    
    alu = counters.get('FU: ALU Active', 0)
    fpu = counters.get('FU: FPU Active', 0)
    ldst = counters.get('FU: LDST Active', 0)
    
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Workload: {name}")
    lines.append(f"{'='*60}")
    lines.append(f"  Cycles:           {cycles:>8,}")
    lines.append(f"  Instructions:     {instr:>8,}")
    lines.append(f"  IPC:              {ipc:>8.3f}")
    lines.append(f"")
    lines.append(f"  Stall breakdown:")
    if cycles > 0:
        lines.append(f"    Scoreboard:     {stall_sb/cycles*100:>7.1f}%  ({stall_sb} cycles)")
        lines.append(f"    I-Fetch:        {stall_if/cycles*100:>7.1f}%  ({stall_if} cycles)")
        lines.append(f"    Memory:         {stall_mem/cycles*100:>7.1f}%  ({stall_mem} cycles)")
        lines.append(f"")
        lines.append(f"  FU Utilization:")
        lines.append(f"    ALU:            {alu/cycles*100:>7.1f}%  ({alu} cycles)")
        lines.append(f"    FPU:            {fpu/cycles*100:>7.1f}%  ({fpu} cycles)")
        lines.append(f"    LDST:           {ldst/cycles*100:>7.1f}%  ({ldst} cycles)")
    
    lines.append(f"")
    if cycles > 0:
        gops = instr / cycles * freq_mhz  # MOPS at freq_mhz
        lines.append(f"  Projected @ {freq_mhz}MHz: {gops:.1f} MOPS")
    lines.append(f"{'='*60}")
    return '\n'.join(lines)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: perf_report.py <log_file> [workload_name]")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    
    name = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    counters = parse_perf(lines)
    
    if not counters:
        print("No performance counters found in output.")
        sys.exit(1)
    
    print(format_report(name, counters))
