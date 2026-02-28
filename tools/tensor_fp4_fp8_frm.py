#!/usr/bin/env python3
"""Generate FP4/FP8 Tensor Core FRM vectors and emit a self-checking RTL testbench."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MASK22 = (1 << 22) - 1
MASK23 = (1 << 23) - 1
MASK24 = (1 << 24) - 1
MASK25 = (1 << 25) - 1


@dataclass(frozen=True)
class TensorCase:
    name: str
    macro: str
    a: int
    b: int
    c: int
    expected: int


def s8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def fp4_to_fp16_bits(fp4: int, *, e3m0: bool) -> int:
    fp4 &= 0xF
    sign = (fp4 >> 3) & 0x1

    if e3m0:
        exp3 = fp4 & 0x7
        if exp3 == 0:
            return sign << 15
        if exp3 == 0x7:
            return (sign << 15) | (0x1F << 10)
        exp16 = (exp3 - 3) + 15
        return (sign << 15) | ((exp16 & 0x1F) << 10)

    exp2 = (fp4 >> 1) & 0x3
    man = fp4 & 0x1

    if exp2 == 0:
        if man == 0:
            return sign << 15
        # RTL maps FP4 E2M1 denormal (man=1, exp=0) to exact 0.5 in FP16.
        return (sign << 15) | (14 << 10)
    if exp2 == 0x3:
        man16 = 0x200 if man else 0
        return (sign << 15) | (0x1F << 10) | man16

    exp16 = (exp2 - 1) + 15
    man16 = man << 9
    return (sign << 15) | ((exp16 & 0x1F) << 10) | man16


def fp8_to_fp16_bits(fp8: int, *, e5m2: bool) -> int:
    fp8 &= 0xFF
    sign = (fp8 >> 7) & 0x1

    if e5m2:
        exp5 = (fp8 >> 2) & 0x1F
        man2 = fp8 & 0x3

        if exp5 == 0:
            if man2 == 0:
                return sign << 15
            return (sign << 15) | (man2 << 8)
        if exp5 == 0x1F:
            man16 = 0x200 if man2 != 0 else 0
            return (sign << 15) | (0x1F << 10) | man16

        exp16 = (exp5 - 15) + 15
        man16 = man2 << 8
        return (sign << 15) | ((exp16 & 0x1F) << 10) | man16

    exp4 = (fp8 >> 3) & 0xF
    man3 = fp8 & 0x7

    if exp4 == 0:
        if man3 == 0:
            return sign << 15

        if man3 & 0b100:
            exp16 = 8
            man16 = (man3 & 0b11) << 8
        elif man3 & 0b010:
            exp16 = 7
            man16 = (man3 & 0b1) << 9
        else:
            exp16 = 6
            man16 = 0
        return (sign << 15) | (exp16 << 10) | man16

    if exp4 == 0xF:
        man16 = 0x200 if man3 != 0 else 0
        return (sign << 15) | (0x1F << 10) | man16

    exp16 = exp4 + 8
    man16 = man3 << 7
    return (sign << 15) | ((exp16 & 0x1F) << 10) | man16


def fp16_mul_to_fp32_bits(a: int, b: int) -> int:
    a &= 0xFFFF
    b &= 0xFFFF

    sign_a = (a >> 15) & 1
    exp_a = (a >> 10) & 0x1F
    man_a = a & 0x3FF

    sign_b = (b >> 15) & 1
    exp_b = (b >> 10) & 0x1F
    man_b = b & 0x3FF

    result_sign = sign_a ^ sign_b

    a_zero = exp_a == 0 and man_a == 0
    b_zero = exp_b == 0 and man_b == 0
    a_inf = exp_a == 0x1F and man_a == 0
    b_inf = exp_b == 0x1F and man_b == 0
    a_nan = exp_a == 0x1F and man_a != 0
    b_nan = exp_b == 0x1F and man_b != 0

    if a_nan or b_nan:
        return 0x7FC00000
    if (a_inf and b_zero) or (b_inf and a_zero):
        return 0x7FC00000
    if a_inf or b_inf:
        return (result_sign << 31) | (0xFF << 23)
    if a_zero or b_zero:
        return result_sign << 31

    eff_exp_a = 1 if exp_a == 0 else exp_a
    eff_exp_b = 1 if exp_b == 0 else exp_b

    sig_a = man_a if exp_a == 0 else ((1 << 10) | man_a)
    sig_b = man_b if exp_b == 0 else ((1 << 10) | man_b)

    product = sig_a * sig_b

    exp_sum = s8((eff_exp_a - 15) + (eff_exp_b - 15) + 127)

    if product == 0:
        lead_pos = 31
    else:
        lead_pos = product.bit_length() - 1

    shift_amt = (21 - lead_pos) & 0x1F
    norm_product = (product << shift_amt) & MASK22
    norm_exp = s8(exp_sum + (lead_pos - 20))

    result_man = ((norm_product & 0x1FFFFF) << 2) & MASK23
    return (result_sign << 31) | ((norm_exp & 0xFF) << 23) | result_man


def fp32_add_bits(a: int, b: int) -> int:
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF

    a_sign = (a >> 31) & 1
    a_exp = (a >> 23) & 0xFF
    a_man = a & MASK23

    b_sign = (b >> 31) & 1
    b_exp = (b >> 23) & 0xFF
    b_man = b & MASK23

    a_zero = a_exp == 0 and a_man == 0
    b_zero = b_exp == 0 and b_man == 0
    a_inf = a_exp == 0xFF and a_man == 0
    b_inf = b_exp == 0xFF and b_man == 0
    a_nan = a_exp == 0xFF and a_man != 0
    b_nan = b_exp == 0xFF and b_man != 0

    if a_nan or b_nan:
        return 0x7FC00000
    if a_inf and b_inf and (a_sign != b_sign):
        return 0x7FC00000
    if a_inf:
        return a
    if b_inf:
        return b
    if a_zero:
        return b
    if b_zero:
        return a

    a_sig = a_man if a_exp == 0 else ((1 << 23) | a_man)
    b_sig = b_man if b_exp == 0 else ((1 << 23) | b_man)

    a_larger = (a_exp > b_exp) or (a_exp == b_exp and a_sig >= b_sig)
    exp_diff = (a_exp - b_exp) if a_larger else (b_exp - a_exp)
    common_exp = a_exp if a_larger else b_exp

    a_aligned = a_sig if a_larger else (a_sig >> exp_diff)
    b_aligned = (b_sig >> exp_diff) if a_larger else b_sig

    same_sign = a_sign == b_sign
    if same_sign:
        sum_val = (a_aligned + b_aligned) & MASK25
    else:
        if a_larger:
            sum_val = (a_aligned - b_aligned) & MASK25
        else:
            sum_val = (b_aligned - a_aligned) & MASK25

    if sum_val == 0:
        return 0

    result_sign = a_sign if same_sign else (a_sign if a_larger else b_sign)

    if sum_val & (1 << 24):
        result_exp = (common_exp + 1) & 0xFF
        result_man = (sum_val >> 1) & MASK23
    elif sum_val & (1 << 23):
        result_exp = common_exp & 0xFF
        result_man = sum_val & MASK23
    else:
        result_exp = (common_exp - 1) & 0xFF
        result_man = ((sum_val & 0x3FFFFF) << 1) & MASK23

    return (result_sign << 31) | (result_exp << 23) | result_man


def dot_fp8(a_word: int, b_word: int, c_word: int, *, e5m2: bool) -> int:
    prods = []
    for idx in range(4):
        a8 = (a_word >> (idx * 8)) & 0xFF
        b8 = (b_word >> (idx * 8)) & 0xFF
        a16 = fp8_to_fp16_bits(a8, e5m2=e5m2)
        b16 = fp8_to_fp16_bits(b8, e5m2=e5m2)
        prods.append(fp16_mul_to_fp32_bits(a16, b16))

    sum01 = fp32_add_bits(prods[0], prods[1])
    sum23 = fp32_add_bits(prods[2], prods[3])
    sum0123 = fp32_add_bits(sum01, sum23)
    return fp32_add_bits(sum0123, c_word)


def dot_fp4(a_word: int, b_word: int, c_word: int, *, e3m0: bool) -> int:
    prods = []
    for idx in range(8):
        a4 = (a_word >> (idx * 4)) & 0xF
        b4 = (b_word >> (idx * 4)) & 0xF
        a16 = fp4_to_fp16_bits(a4, e3m0=e3m0)
        b16 = fp4_to_fp16_bits(b4, e3m0=e3m0)
        prods.append(fp16_mul_to_fp32_bits(a16, b16))

    sum01 = fp32_add_bits(prods[0], prods[1])
    sum23 = fp32_add_bits(prods[2], prods[3])
    sum45 = fp32_add_bits(prods[4], prods[5])
    sum67 = fp32_add_bits(prods[6], prods[7])
    sum0123 = fp32_add_bits(sum01, sum23)
    sum4567 = fp32_add_bits(sum45, sum67)
    sum_all = fp32_add_bits(sum0123, sum4567)
    return fp32_add_bits(sum_all, c_word)


def compute_expected(macro: str, a_word: int, b_word: int, c_word: int) -> int:
    if macro == "`TC_DATA_FP8_E4M3":
        return dot_fp8(a_word, b_word, c_word, e5m2=False)
    if macro == "`TC_DATA_FP8_E5M2":
        return dot_fp8(a_word, b_word, c_word, e5m2=True)
    if macro == "`TC_DATA_FP4_E2M1":
        return dot_fp4(a_word, b_word, c_word, e3m0=False)
    if macro == "`TC_DATA_FP4_E3M0":
        return dot_fp4(a_word, b_word, c_word, e3m0=True)
    raise ValueError(f"Unsupported macro: {macro}")


def random_finite_fp32_bits(rng: random.Random) -> int:
    if rng.random() < 0.1:
        return 0
    sign = rng.getrandbits(1)
    exp = rng.randint(1, 254)
    man = rng.getrandbits(23)
    return (sign << 31) | (exp << 23) | man


def directed_cases() -> Iterable[tuple[str, str, int, int, int]]:
    return [
        ("fp8_e4m3_ones", "`TC_DATA_FP8_E4M3", 0x38383838, 0x38383838, 0x00000000),
        ("fp8_e4m3_small", "`TC_DATA_FP8_E4M3", 0x20283038, 0x38383838, 0x00000000),
        ("fp8_e5m2_ones", "`TC_DATA_FP8_E5M2", 0x3C3C3C3C, 0x3C3C3C3C, 0x00000000),
        ("fp8_e5m2_mixed", "`TC_DATA_FP8_E5M2", 0x3C3C403C, 0x3C3C3C40, 0x3F800000),
        ("fp4_e2m1_ones", "`TC_DATA_FP4_E2M1", 0x22222222, 0x22222222, 0x00000000),
        ("fp4_e2m1_denorm", "`TC_DATA_FP4_E2M1", 0x11111111, 0x22222222, 0x00000000),
        ("fp4_e3m0_ones", "`TC_DATA_FP4_E3M0", 0x33333333, 0x33333333, 0x00000000),
        ("fp4_e3m0_zero", "`TC_DATA_FP4_E3M0", 0x00000000, 0x33333333, 0x40400000),
    ]


def generate_cases(cases_per_dtype: int, seed: int) -> list[TensorCase]:
    rng = random.Random(seed)
    cases: list[TensorCase] = []

    dtype_to_prefix = {
        "`TC_DATA_FP8_E4M3": "fp8_e4m3",
        "`TC_DATA_FP8_E5M2": "fp8_e5m2",
        "`TC_DATA_FP4_E2M1": "fp4_e2m1",
        "`TC_DATA_FP4_E3M0": "fp4_e3m0",
    }

    per_dtype_count = {macro: 0 for macro in dtype_to_prefix}

    for name, macro, a_word, b_word, c_word in directed_cases():
        expected = compute_expected(macro, a_word, b_word, c_word)
        cases.append(TensorCase(name=name, macro=macro, a=a_word, b=b_word, c=c_word, expected=expected))
        per_dtype_count[macro] += 1

    for macro, prefix in dtype_to_prefix.items():
        needed = max(0, cases_per_dtype - per_dtype_count[macro])
        for idx in range(needed):
            a_word = rng.getrandbits(32)
            b_word = rng.getrandbits(32)
            c_word = random_finite_fp32_bits(rng)
            expected = compute_expected(macro, a_word, b_word, c_word)
            cases.append(
                TensorCase(
                    name=f"{prefix}_rand_{idx:03d}",
                    macro=macro,
                    a=a_word,
                    b=b_word,
                    c=c_word,
                    expected=expected,
                )
            )

    return cases


def emit_verilog(cases: list[TensorCase], out_path: Path, seed: int, cases_per_dtype: int) -> None:
    body_lines = []
    for case in cases:
        body_lines.append(
            f"        check_mma({case.macro}, 32'h{case.a:08X}, 32'h{case.b:08X}, 32'h{case.c:08X}, "
            f"32'h{case.expected:08X}, \"{case.name}\");"
        )

    tb = f"""// Auto-generated by tools/tensor_fp4_fp8_frm.py
// Seed={seed}, cases_per_dtype={cases_per_dtype}, total_cases={len(cases)}
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_fp4_fp8_frm_generated;
    localparam NUM_LANES = 1;
    localparam DATA_WIDTH = 32;
    localparam TC_LATENCY = 2;
    localparam TC_NUM_CORES = 1;

    reg  clk, rst_n;
    reg  op_valid;
    wire op_ready;
    reg  [3:0] op_type;
    reg  [NUM_LANES*DATA_WIDTH-1:0] frag_a, frag_b, frag_c;
    wire result_valid;
    reg  result_ready;
    wire [NUM_LANES*DATA_WIDTH-1:0] result_data;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    tensor_core #(
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TC_NUM_CORES(TC_NUM_CORES),
        .TC_LATENCY(TC_LATENCY),
        .TC_USE_OP_TYPE(1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .op_valid(op_valid), .op_ready(op_ready),
        .op_type(op_type),
        .frag_a(frag_a), .frag_b(frag_b), .frag_c(frag_c),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_data(result_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    task check_mma;
        input [3:0]   t_type;
        input [31:0]  t_a, t_b, t_c;
        input [31:0]  expected;
        input [255:0] name;
        integer timeout;
        begin
            test_num = test_num + 1;
            @(posedge clk);
            while (!op_ready) @(posedge clk);

            op_valid <= 1'b1;
            op_type  <= t_type;
            frag_a   <= t_a;
            frag_b   <= t_b;
            frag_c   <= t_c;
            @(posedge clk);
            op_valid <= 1'b0;

            timeout = 0;
            while (!result_valid && timeout < 80) begin
                @(posedge clk);
                #1;
                timeout = timeout + 1;
            end

            if (!result_valid) begin
                $display("FAIL #%0d %0s: TIMEOUT", test_num, name);
                fail_count = fail_count + 1;
            end else if (result_data[31:0] === expected) begin
                pass_count = pass_count + 1;
            end else begin
                $display(
                    "FAIL #%0d %0s: type=%0d a=%08h b=%08h c=%08h got=%08h expected=%08h",
                    test_num, name, t_type, t_a, t_b, t_c, result_data[31:0], expected
                );
                fail_count = fail_count + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        rst_n = 0;
        op_valid = 0;
        result_ready = 1;
        op_type = 0;
        frag_a = 0;
        frag_b = 0;
        frag_c = 0;

        #30;
        rst_n = 1;
        #10;

{chr(10).join(body_lines)}

        #50;
        $display("===== Tensor FP4/FP8 FRM Compare: %0d/%0d passed =====", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("FAILURES: %0d", fail_count);
        $finish;
    end

    initial begin
        #200000;
        $display("TIMEOUT");
        $finish;
    end
endmodule
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tb)


def emit_json(cases: list[TensorCase], out_path: Path) -> None:
    payload = [
        {
            "name": c.name,
            "macro": c.macro,
            "a": f"0x{c.a:08X}",
            "b": f"0x{c.b:08X}",
            "c": f"0x{c.c:08X}",
            "expected": f"0x{c.expected:08X}",
        }
        for c in cases
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FP4/FP8 Tensor Core FRM vectors")
    parser.add_argument(
        "--emit-tb",
        type=Path,
        default=Path("build/tb_tensor_fp4_fp8_frm_generated.v"),
        help="Output self-checking generated Verilog testbench path",
    )
    parser.add_argument(
        "--emit-json",
        type=Path,
        default=Path("build/tensor_fp4_fp8_frm_vectors.json"),
        help="Output JSON vector dump path",
    )
    parser.add_argument("--seed", type=int, default=239, help="Random seed")
    parser.add_argument("--cases-per-dtype", type=int, default=32, help="Number of test cases per dtype")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cases_per_dtype < 1:
        raise ValueError("--cases-per-dtype must be >= 1")

    cases = generate_cases(cases_per_dtype=args.cases_per_dtype, seed=args.seed)
    emit_verilog(cases, args.emit_tb, args.seed, args.cases_per_dtype)
    emit_json(cases, args.emit_json)

    print(
        f"Generated {len(cases)} cases (seed={args.seed}, cases_per_dtype={args.cases_per_dtype}) "
        f"-> {args.emit_tb}"
    )
    print(f"Wrote vectors -> {args.emit_json}")


if __name__ == "__main__":
    main()
