#!/usr/bin/env python3
import os
import subprocess
import re
import sys

def main():
    print("Running performance gate check (IPC >= 0.80)...")
    env = os.environ.copy()
    cmd = ["make", "test_sm_v2_perf"]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except Exception as e:
        print(f"Error running make test_sm_v2_perf: {e}")
        sys.exit(1)
        
    output = result.stdout + "\n" + result.stderr
    
    if result.returncode != 0:
        print(f"make test_sm_v2_perf failed with return code {result.returncode}.")
        print("Output:\n" + output)
        sys.exit(1)
    
    match = re.search(r"IPC:\s*([\d.]+)", output)
    ipc = 0.0
    if not match:
        match_c = re.search(r"(?:Cycles):\s*(\d+)", output)
        match_i = re.search(r"(?:Instructions|Issues):\s*(\d+)", output)
        if match_c and match_i:
            c = int(match_c.group(1))
            i = int(match_i.group(1))
            if c > 0:
                ipc = i / c
        else:
            print("Failed to find IPC or Cycles/Instructions in output. Run output:")
            print(output)
            sys.exit(1)
    else:
        ipc = float(match.group(1))
        
    print(f"Detected IPC: {ipc:.3f}")
    if ipc < 0.80:
        print(f"FAIL: IPC regression gate failed. {ipc:.3f} < 0.80")
        sys.exit(1)
    else:
        print(f"PASS: IPC regression gate passed. {ipc:.3f} >= 0.80")
        sys.exit(0)

if __name__ == '__main__':
    main()
