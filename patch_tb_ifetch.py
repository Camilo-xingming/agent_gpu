#!/usr/bin/env python3
"""Add perf_stall_ifetch counter to tensor_multiwarp and gemm16 testbenches."""

# === tensor_multiwarp ===
with open('/Users/jerry/RalphGPU/tb/tb_sm_v2_perf_tensor_multiwarp.v', 'r') as f:
    tb = f.read()

# 1. Add stall_ifetch counter declaration
tb = tb.replace(
    '    integer stall_wbq;',
    '    integer stall_wbq;\n    integer stall_ifetch;'
)

# 2. Add reset (in reset block — two occurrences)
tb = tb.replace(
    '            stall_wbq <= 0;\n        end else begin',
    '            stall_wbq <= 0;\n            stall_ifetch <= 0;\n        end else begin'
)
tb = tb.replace(
    '                stall_wbq <= 0;\n            end else if (running) begin',
    '                stall_wbq <= 0;\n                stall_ifetch <= 0;\n            end else if (running) begin'
)

# 3. Add counting logic — use the scheduler-centric signal
# Find where issue_count is incremented and add after it
tb = tb.replace(
    '                if (dut.issue_valid) begin\n                    issue_count <= issue_count + 1;\n                end',
    '                if (dut.issue_valid) begin\n                    issue_count <= issue_count + 1;\n                end\n                if (dut.perf_stall_ifetch) begin\n                    stall_ifetch <= stall_ifetch + 1;\n                end'
)

# 4. Add to results display
tb = tb.replace(
    "        $display(\"Stalls: raw=%0d fu=%0d mem=%0d atomic=%0d tensor=%0d wbq=%0d\",\n                 stall_raw, stall_fu, stall_mem, stall_atomic, stall_tensor, stall_wbq);",
    "        $display(\"Stalls: raw=%0d fu=%0d mem=%0d atomic=%0d tensor=%0d wbq=%0d ifetch=%0d (%0d%%)\",\n                 stall_raw, stall_fu, stall_mem, stall_atomic, stall_tensor, stall_wbq,\n                 stall_ifetch, (cycle_count > 0) ? (stall_ifetch * 100 / cycle_count) : 0);"
)

with open('/Users/jerry/RalphGPU/tb/tb_sm_v2_perf_tensor_multiwarp.v', 'w') as f:
    f.write(tb)
print("tensor_multiwarp TB patched")

# === gemm16 ===
with open('/Users/jerry/RalphGPU/tb/tb_sm_v2_perf_gemm16_ptx.v', 'r') as f:
    tb = f.read()

# 1. Add stall_ifetch counter declaration
tb = tb.replace(
    '    integer stall_wbq;',
    '    integer stall_wbq;\n    integer stall_ifetch;'
)

# 2. Add reset (two occurrences)
tb = tb.replace(
    '            stall_wbq <= 0;\n        end else begin',
    '            stall_wbq <= 0;\n            stall_ifetch <= 0;\n        end else begin'
)
tb = tb.replace(
    '                stall_wbq <= 0;\n            end else if (running) begin',
    '                stall_wbq <= 0;\n                stall_ifetch <= 0;\n            end else if (running) begin'
)

# 3. Add counting logic
tb = tb.replace(
    '                if (dut.issue_valid) begin\n                    issue_count <= issue_count + 1;\n                end',
    '                if (dut.issue_valid) begin\n                    issue_count <= issue_count + 1;\n                end\n                if (dut.perf_stall_ifetch) begin\n                    stall_ifetch <= stall_ifetch + 1;\n                end'
)

with open('/Users/jerry/RalphGPU/tb/tb_sm_v2_perf_gemm16_ptx.v', 'w') as f:
    f.write(tb)
print("gemm16 TB patched")
