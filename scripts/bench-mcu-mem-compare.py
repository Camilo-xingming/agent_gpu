#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path


IPC_RE = re.compile(r"IPC\s*=\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9]+(?:\.[0-9]+)?)")
CYC_RE = re.compile(r"Cycles\s*=\s*(\d+)")
INS_RE = re.compile(r"Instructions\s*=\s*(\d+)")

BENCHES = [
    ("mem_stream8", "test_perf_mem_stream"),
    ("mem_gather", "test_perf_mem_gather"),
]


def run_one(repo: Path, target: str, mode: int, label: str, env: dict[str, str]):
    log = repo / "build" / f"{label}_l1d{mode}.log"
    cmd = ["make", "--no-print-directory", target, f"TB_L1D_BYPASS={mode}"]
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)

    text = log.read_text(errors="replace")

    ipc_m = list(IPC_RE.finditer(text))
    cyc_m = list(CYC_RE.finditer(text))
    ins_m = list(INS_RE.finditer(text))

    if not ipc_m or not cyc_m or not ins_m:
        tail = "\n".join(text.splitlines()[-120:])
        raise RuntimeError(f"Metrics not found for {target} mode={mode}\n{tail}")

    ipc = float(ipc_m[-1].group(3))
    cycles = int(cyc_m[-1].group(1))
    instr = int(ins_m[-1].group(1))
    passed = "*** MEMORY BENCHMARK PASSED ***" in text

    print(
        f"[RESULT] {label} mode={mode} status={'PASS' if passed else 'FAIL'} "
        f"instr={instr} cycles={cycles} ipc={ipc:.6f}"
    )
    print(f"[LOG] {log}")

    if proc.returncode != 0 and not passed:
        raise RuntimeError(f"make failed for {target} mode={mode}")

    return {
        "passed": passed,
        "instr": instr,
        "cycles": cycles,
        "ipc": ipc,
        "log": str(log),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    (repo / "build").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    rows: list[dict[str, object]] = []

    for bench_name, target in BENCHES:
        bypass = run_one(repo, target, 1, bench_name, env)
        enable = run_one(repo, target, 0, bench_name, env)

        ipc_delta = float(enable["ipc"]) - float(bypass["ipc"])
        ipc_gain = 0.0 if float(bypass["ipc"]) == 0.0 else ipc_delta / float(bypass["ipc"]) * 100.0

        cyc_delta = int(bypass["cycles"]) - int(enable["cycles"])
        cyc_gain = 0.0 if int(bypass["cycles"]) == 0 else cyc_delta / int(bypass["cycles"]) * 100.0

        rows.append(
            {
                "benchmark": bench_name,
                "ipc_bypass": float(bypass["ipc"]),
                "ipc_enable": float(enable["ipc"]),
                "ipc_delta": ipc_delta,
                "ipc_gain_pct": ipc_gain,
                "cycles_bypass": int(bypass["cycles"]),
                "cycles_enable": int(enable["cycles"]),
                "cycles_saved": cyc_delta,
                "cycles_saved_pct": cyc_gain,
                "status": "PASS" if (bypass["passed"] and enable["passed"]) else "FAIL",
            }
        )

    print("\n| Benchmark | IPC (bypass) | IPC (L1D+MCU) | IPC Delta | IPC Gain | Cycles (bypass) | Cycles (L1D+MCU) | Cycles Saved | Cycle Gain | Status |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print(
            "| {benchmark} | {ipc_bypass:.6f} | {ipc_enable:.6f} | {ipc_delta:+.6f} | {ipc_gain_pct:+.2f}% | "
            "{cycles_bypass} | {cycles_enable} | {cycles_saved:+d} | {cycles_saved_pct:+.2f}% | {status} |".format(**r)
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
