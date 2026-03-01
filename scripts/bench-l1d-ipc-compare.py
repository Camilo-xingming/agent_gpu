#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path


def run_mode(repo: Path, mode: int, label: str, env: dict[str, str]):
    log = repo / "build" / f"l1d_ipc_{label}.log"
    cmd = ["make", "--no-print-directory", "test_perf_counters", f"TB_L1D_BYPASS={mode}"]
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(errors="replace")

    passed = "*** INTEGRATION TEST PASSED ***" in text

    matches = list(re.finditer(r"IPC\s*=\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text))
    if not matches:
        tail = "\n".join(text.splitlines()[-80:])
        raise RuntimeError(f"IPC line not found for mode={mode}\n{tail}")

    m = matches[-1]
    instr = int(m.group(1))
    cycles = int(m.group(2))
    ipc = float(m.group(3))
    status = "PASS" if passed else "FAIL"
    print(f"[RESULT] {label}: status={status} instructions={instr} cycles={cycles} ipc={ipc:.6f}")
    print(f"[LOG] {log}")

    if proc.returncode != 0 and not matches:
        raise RuntimeError(f"make failed for mode={mode} and no IPC extracted")

    return passed, instr, cycles, ipc


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    (repo / "build").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    print("[RUN] L1D_BYPASS=1 (bypass)")
    passed_bypass, _, _, ipc_bypass = run_mode(repo, 1, "bypass", env)

    print("[RUN] L1D_BYPASS=0 (enable)")
    passed_enable, _, _, ipc_enable = run_mode(repo, 0, "enable", env)

    delta = ipc_enable - ipc_bypass
    gain_pct = 0.0 if ipc_bypass == 0.0 else (delta / ipc_bypass * 100.0)
    print(
        "[SUMMARY] IPC bypass={:.6f} enable={:.6f} delta={:+.6f} ({:+.2f}%)".format(
            ipc_bypass, ipc_enable, delta, gain_pct
        )
    )
    print(f"[SUMMARY] integration_status bypass={'PASS' if passed_bypass else 'FAIL'} enable={'PASS' if passed_enable else 'FAIL'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
