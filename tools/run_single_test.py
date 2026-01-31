
import subprocess
import sys
from pathlib import Path

def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent

def run_simulation(hex_file: str, test_name: str):
    """Run Verilog simulation with HEX file"""
    root = get_project_root()

    # Compile testbench
    tb_file = root / "tb" / "tb_top_level_unified.v"
    rtl_files = list((root / "rtl").glob("*.v"))
    
    vvp_file = Path(f"/tmp/tb_{test_name}.vvp")
    if vvp_file.exists():
        vvp_file.unlink()

    compile_cmd = [
        "iverilog", "-g2012",
        f"-I{root}/rtl",
        f"-DHEX_FILE=\"{hex_file}\"",
        "-DTIMEOUT_CYCLES=50000",
        "-DSUCCESS_ADDR=32'h2000",
        "-DSUCCESS_VALUE=32'h0",
        "-s", "tb_top_level_unified",
        "-o", str(vvp_file),
        str(tb_file)
    ] + [str(f) for f in rtl_files]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"Compile failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("Compile timeout")
        return False
    except Exception as e:
        print(f"Compile error: {e}")
        return False

    # Run simulation
    try:
        result = subprocess.run(
            ["vvp", str(vvp_file)],
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + result.stderr

        if "TEST PASSED" in output or "PASS" in output:
            return True
        elif "TEST FAILED" in output or "FAIL" in output:
            print(f"FAIL: Test assertion failed")
            return False
        elif "TIMEOUT" in output:
            print(f"FAIL: Simulation timeout")
            return False
        else:
            if result.returncode == 0:
                return True
            print(f"FAIL: Unknown status\n{output[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("FAIL: Process timeout")
        return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 run_single_test.py <hex_file>")
        sys.exit(1)

    hex_file = sys.argv[1]
    test_name = Path(hex_file).stem

    if run_simulation(hex_file, test_name):
        print("TEST PASSED")
        sys.exit(0)
    else:
        print("TEST FAILED")
        sys.exit(1)
