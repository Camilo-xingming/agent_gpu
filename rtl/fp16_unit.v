/* verilator lint_off BLKSEQ */
//============================================================================
// RalphGPU - FP16/BF16 Half-Precision Unit
// IEEE 754 FP16 and Brain Float 16 support for ML workloads
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module fp16_unit (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire [5:0]  func,
    input  wire        valid_in,
    input  wire        packed_mode,  // 1=FP16x2, 0=single FP16/BF16

    // Operands (32-bit: single FP16 in lower 16 bits, or packed FP16x2)
    input  wire [31:0] operand_a,
    input  wire [31:0] operand_b,
    input  wire [31:0] operand_c,    // For FMA

    // Result
    output reg  [31:0] result,
    output reg         valid_out,

    // Exception flags
    output reg         overflow,
    output reg         underflow,
    output reg         inexact,
    output reg         invalid
);

    //------------------------------------------------------------------------
    // FP16 Format: 1-5-10 (sign-exponent-mantissa)
    // BF16 Format: 1-8-7  (sign-exponent-mantissa)
    //------------------------------------------------------------------------

    // Pipeline registers
    reg [5:0]  func_r, func_r2;           // Two stages for func
    reg        packed_mode_r, packed_mode_r2;
    reg [31:0] op_a_r, op_b_r, op_c_r;
    reg [31:0] op_a_r2, op_b_r2, op_c_r2;  // Operands for stage 2
    reg        valid_r1, valid_r2;

    // FP16 extraction from STAGE 2 operands (for use in pipeline stage 2 computation)
    wire        fp16_a_sign = op_a_r2[15];
    wire [4:0]  fp16_a_exp  = op_a_r2[14:10];
    wire [9:0]  fp16_a_mant = op_a_r2[9:0];

    wire        fp16_b_sign = op_b_r2[15];
    wire [4:0]  fp16_b_exp  = op_b_r2[14:10];
    wire [9:0]  fp16_b_mant = op_b_r2[9:0];

    // FP16 upper (for packed mode) from STAGE 2 operands
    wire        fp16_a_hi_sign = op_a_r2[31];
    wire [4:0]  fp16_a_hi_exp  = op_a_r2[30:26];
    wire [9:0]  fp16_a_hi_mant = op_a_r2[25:16];

    wire        fp16_b_hi_sign = op_b_r2[31];
    wire [4:0]  fp16_b_hi_exp  = op_b_r2[30:26];
    wire [9:0]  fp16_b_hi_mant = op_b_r2[25:16];

    // BF16 extraction from STAGE 2 operands
    wire        bf16_a_sign = op_a_r2[15];
    wire [7:0]  bf16_a_exp  = op_a_r2[14:7];
    wire [6:0]  bf16_a_mant = op_a_r2[6:0];

    wire        bf16_b_sign = op_b_r2[15];
    wire [7:0]  bf16_b_exp  = op_b_r2[14:7];
    wire [6:0]  bf16_b_mant = op_b_r2[6:0];

    // Special value detection - FP16 (from stage 2 operands)
    wire fp16_a_is_zero = (fp16_a_exp == 5'b0) && (fp16_a_mant == 10'b0);
    wire fp16_b_is_zero = (fp16_b_exp == 5'b0) && (fp16_b_mant == 10'b0);
    wire fp16_a_is_inf  = (fp16_a_exp == 5'h1F) && (fp16_a_mant == 10'b0);
    wire fp16_b_is_inf  = (fp16_b_exp == 5'h1F) && (fp16_b_mant == 10'b0);
    wire fp16_a_is_nan  = (fp16_a_exp == 5'h1F) && (fp16_a_mant != 10'b0);
    wire fp16_b_is_nan  = (fp16_b_exp == 5'h1F) && (fp16_b_mant != 10'b0);

    // Special value detection - BF16 (from stage 2 operands)
    wire bf16_a_is_zero = (bf16_a_exp == 8'b0) && (bf16_a_mant == 7'b0);
    wire bf16_b_is_zero = (bf16_b_exp == 8'b0) && (bf16_b_mant == 7'b0);
    wire bf16_a_is_inf  = (bf16_a_exp == 8'hFF) && (bf16_a_mant == 7'b0);
    wire bf16_b_is_inf  = (bf16_b_exp == 8'hFF) && (bf16_b_mant == 7'b0);
    wire bf16_a_is_nan  = (bf16_a_exp == 8'hFF) && (bf16_a_mant != 7'b0);
    wire bf16_b_is_nan  = (bf16_b_exp == 8'hFF) && (bf16_b_mant != 7'b0);

    // Special value detection - FP32 for operand B (used in mixed FP16-FP32 compares)
    wire [7:0]  fp32_b_exp  = op_b_r2[30:23];
    wire [22:0] fp32_b_mant = op_b_r2[22:0];
    wire fp32_b_is_nan  = (fp32_b_exp == 8'hFF) && (fp32_b_mant != 23'b0);
    wire fp32_b_is_inf  = (fp32_b_exp == 8'hFF) && (fp32_b_mant == 23'b0);
    wire fp32_b_is_zero = (fp32_b_exp == 8'b0) && (fp32_b_mant == 23'b0);

    //------------------------------------------------------------------------
    // FP16 Constants
    //------------------------------------------------------------------------
    localparam [15:0] FP16_ZERO     = 16'h0000;
    localparam [15:0] FP16_NEG_ZERO = 16'h8000;
    localparam [15:0] FP16_ONE      = 16'h3C00;  // 1.0
    localparam [15:0] FP16_INF      = 16'h7C00;
    localparam [15:0] FP16_NEG_INF  = 16'hFC00;
    localparam [15:0] FP16_NAN      = 16'h7E00;  // Quiet NaN

    localparam [15:0] BF16_ZERO     = 16'h0000;
    localparam [15:0] BF16_ONE      = 16'h3F80;  // 1.0
    localparam [15:0] BF16_INF      = 16'h7F80;
    localparam [15:0] BF16_NAN      = 16'h7FC0;

    //------------------------------------------------------------------------
    // FP16 to FP32 conversion (for internal calculations)
    //------------------------------------------------------------------------
    function [31:0] fp16_to_fp32;
        input [15:0] fp16;
        reg         sign;
        reg [4:0]   exp16;
        reg [9:0]   mant16;
        reg [7:0]   exp32;
        reg [22:0]  mant32;
        begin
            sign   = fp16[15];
            exp16  = fp16[14:10];
            mant16 = fp16[9:0];

            if (exp16 == 5'b0) begin
                // Zero or denormal
                exp32  = 8'b0;
                mant32 = {mant16, 13'b0};
            end else if (exp16 == 5'h1F) begin
                // Inf or NaN
                exp32  = 8'hFF;
                mant32 = {mant16, 13'b0};
            end else begin
                // Normal number
                exp32  = {3'b0, exp16} + 8'd112;  // bias adjust: 127-15=112
                mant32 = {mant16, 13'b0};
            end

            fp16_to_fp32 = {sign, exp32, mant32};
        end
    endfunction

    //------------------------------------------------------------------------
    // FP32 to FP16 conversion (with rounding)
    //------------------------------------------------------------------------
    function [15:0] fp32_to_fp16;
        input [31:0] fp32;
        reg         sign;
        reg [7:0]   exp32;
        reg [22:0]  mant32;
        reg [4:0]   exp16;
        reg [9:0]   mant16;
        reg         round_bit;
        begin
            sign   = fp32[31];
            exp32  = fp32[30:23];
            mant32 = fp32[22:0];

            if (exp32 == 8'b0) begin
                // Zero or denormal -> zero
                exp16  = 5'b0;
                mant16 = 10'b0;
            end else if (exp32 == 8'hFF) begin
                // Inf or NaN
                exp16  = 5'h1F;
                mant16 = mant32[22:13];
            end else if (exp32 < 8'd103) begin
                // Too small -> zero
                exp16  = 5'b0;
                mant16 = 10'b0;
            end else if (exp32 > 8'd142) begin
                // Too large -> infinity
                exp16  = 5'h1F;
                mant16 = 10'b0;
            end else begin
                // Normal range
                begin : fp32_to_fp16_normal
                    reg [7:0] exp_diff;
                    exp_diff = exp32 - 8'd112;
                    exp16 = exp_diff[4:0];
                end
                mant16 = mant32[22:13];
                // Round to nearest
                round_bit = mant32[12];
                if (round_bit) begin
                    {exp16, mant16} = {exp16, mant16} + 1;
                end
            end

            fp32_to_fp16 = {sign, exp16, mant16};
        end
    endfunction

    //------------------------------------------------------------------------
    // BF16 to FP32 conversion (simple - same exponent range)
    //------------------------------------------------------------------------
    function [31:0] bf16_to_fp32;
        input [15:0] bf16;
        begin
            // BF16 is just truncated FP32, so extend with zeros
            bf16_to_fp32 = {bf16, 16'b0};
        end
    endfunction

    //------------------------------------------------------------------------
    // FP32 to BF16 conversion (truncate with rounding)
    //------------------------------------------------------------------------
    function [15:0] fp32_to_bf16;
        input [31:0] fp32;
        reg round_bit;
        begin
            round_bit = fp32[15];  // Round to nearest
            if (round_bit) begin
                fp32_to_bf16 = fp32[31:16] + 1;
            end else begin
                fp32_to_bf16 = fp32[31:16];
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Simple FP16 arithmetic using FP32 intermediate
    // Use STAGE 2 operands (op_a_r2, op_b_r2, op_c_r2) for pipeline correctness
    //------------------------------------------------------------------------
    wire [31:0] fp32_a = fp16_to_fp32(op_a_r2[15:0]);
    wire [31:0] fp32_b = fp16_to_fp32(op_b_r2[15:0]);
    wire [31:0] fp32_c = fp16_to_fp32(op_c_r2[15:0]);

    wire [31:0] fp32_a_hi = fp16_to_fp32(op_a_r2[31:16]);
    wire [31:0] fp32_b_hi = fp16_to_fp32(op_b_r2[31:16]);
    wire [31:0] fp32_c_hi = fp16_to_fp32(op_c_r2[31:16]);

    // Pre-computed FP32 intermediates (Yosys cannot synthesize nested function calls)
    wire [31:0] fp32_add_ab    = fp32_add(fp32_a, fp32_b);
    wire [31:0] fp32_add_ab_neg = fp32_add(fp32_a, {~fp32_b[31], fp32_b[30:0]});
    wire [31:0] fp32_mul_ab    = fp32_mul(fp32_a, fp32_b);
    wire [31:0] fp32_fma_ab_c  = fp32_add(fp32_mul_ab, fp32_c);
    wire [31:0] fp32_add_ab_hi    = fp32_add(fp32_a_hi, fp32_b_hi);
    wire [31:0] fp32_add_ab_neg_hi = fp32_add(fp32_a_hi, {~fp32_b_hi[31], fp32_b_hi[30:0]});
    wire [31:0] fp32_mul_ab_hi    = fp32_mul(fp32_a_hi, fp32_b_hi);
    wire [31:0] fp32_fma_ab_c_hi  = fp32_add(fp32_mul_ab_hi, fp32_c_hi);

    // BF16 to FP32 (from stage 2 operands)
    wire [31:0] bf32_a = bf16_to_fp32(op_a_r2[15:0]);
    wire [31:0] bf32_b = bf16_to_fp32(op_b_r2[15:0]);
    wire [31:0] bf32_c = bf16_to_fp32(op_c_r2[15:0]);

    // Pre-computed BF32 intermediates (Yosys synthesis)
    wire [31:0] bf32_add_ab    = fp32_add(bf32_a, bf32_b);
    wire [31:0] bf32_add_ab_neg = fp32_add(bf32_a, {~bf32_b[31], bf32_b[30:0]});
    wire [31:0] bf32_mul_ab    = fp32_mul(bf32_a, bf32_b);
    wire [31:0] bf32_fma_ab_c  = fp32_add(bf32_mul_ab, bf32_c);

    //------------------------------------------------------------------------
    // FP32 arithmetic results (using simple operations)
    //------------------------------------------------------------------------
    // For real implementation, use full FP32 adder/multiplier
    // This is simplified for synthesis feasibility

    reg [31:0] fp32_result_lo;
    reg [31:0] fp32_result_hi;
    reg [15:0] fp16_result_lo;
    reg [15:0] fp16_result_hi;

    //------------------------------------------------------------------------
    // Main computation pipeline
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_r1 <= 1'b0;
            valid_r2 <= 1'b0;
            valid_out <= 1'b0;
            result <= 32'b0;
            overflow <= 1'b0;
            underflow <= 1'b0;
            inexact <= 1'b0;
            invalid <= 1'b0;
        end else begin
            // Pipeline stage 1: Register inputs
            valid_r1 <= valid_in;
            func_r <= func;
            packed_mode_r <= packed_mode;
            op_a_r <= operand_a;
            op_b_r <= operand_b;
            op_c_r <= operand_c;

            // Pipeline stage 2: Compute - advance stage 1 to stage 2
            valid_r2 <= valid_r1;
            func_r2 <= func_r;
            packed_mode_r2 <= packed_mode_r;
            op_a_r2 <= op_a_r;
            op_b_r2 <= op_b_r;
            op_c_r2 <= op_c_r;

            `ifdef SIMULATION
            if (valid_r1)
                `ifdef SIMULATION
                $display("[%0t FP16_STAGE1] func_r=%0d op_a_r=0x%08x op_b_r=0x%08x",
                         $time, func_r, op_a_r, op_b_r);
                `endif
            `endif

            // Pipeline stage 3: Output
            valid_out <= valid_r2;
            overflow <= 1'b0;
            underflow <= 1'b0;
            inexact <= 1'b0;
            invalid <= 1'b0;

            `ifdef SIMULATION
            if (valid_r2)
                `ifdef SIMULATION
                $display("[%0t FP16_STAGE2] func_r2=%0d op_a_r2=0x%08x op_b_r2=0x%08x",
                         $time, func_r2, op_a_r2, op_b_r2);
                `endif
            `endif

            if (valid_r2) begin
                /* verilator lint_off CASEOVERLAP */
                case (func_r2)
                    //----------------------------------------------------
                    // FP16 Operations
                    //----------------------------------------------------
                    `FP16_ADD: begin
                        // Simple add using sign comparison
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else if (fp16_a_is_inf && fp16_b_is_inf &&
                                    (fp16_a_sign != fp16_b_sign)) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else if (fp16_a_is_inf) begin
                            fp16_result_lo = {fp16_a_sign, 5'h1F, 10'b0};
                        end else if (fp16_b_is_inf) begin
                            fp16_result_lo = {fp16_b_sign, 5'h1F, 10'b0};
                        end else if (fp16_a_is_zero) begin
                            fp16_result_lo = op_b_r2[15:0];
                        end else if (fp16_b_is_zero) begin
                            fp16_result_lo = op_a_r2[15:0];
                        end else begin
                            // Use FP32 intermediate
                            fp16_result_lo = fp32_to_fp16(fp32_add_ab);
                        end
                        result <= {16'b0, fp16_result_lo};
                    end

                    `FP16_SUB: begin
                        // Negate b and add
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else begin
                            fp16_result_lo = fp32_to_fp16(fp32_add_ab_neg);
                        end
                        result <= {16'b0, fp16_result_lo};
                    end

                    `FP16_MUL: begin
                        `ifdef SIMULATION
                        $display("[%0t FP16_MUL] op_a_r2=0x%08x op_b_r2=0x%08x fp32_a=0x%08x fp32_b=0x%08x",
                                 $time, op_a_r2, op_b_r2, fp32_a, fp32_b);
                        `endif
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else if ((fp16_a_is_inf && fp16_b_is_zero) ||
                                    (fp16_a_is_zero && fp16_b_is_inf)) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else if (fp16_a_is_zero || fp16_b_is_zero) begin
                            fp16_result_lo = {fp16_a_sign ^ fp16_b_sign, 15'b0};
                        end else if (fp16_a_is_inf || fp16_b_is_inf) begin
                            fp16_result_lo = {fp16_a_sign ^ fp16_b_sign, 5'h1F, 10'b0};
                        end else begin
                            fp16_result_lo = fp32_to_fp16(fp32_mul_ab);
                            `ifdef SIMULATION
                            $display("[%0t FP16_MUL] fp32_product=0x%08x fp16_result=0x%04x",
                                     $time, fp32_mul_ab, fp16_result_lo);
                            `endif
                        end
                        result <= {16'b0, fp16_result_lo};
                    end

                    `FP16_MUL_F32: begin
                        // Mixed precision: FP16 inputs, FP32 output
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h7FC00000; // FP32 NaN
                        end else if ((fp16_a_is_inf && fp16_b_is_zero) ||
                                    (fp16_a_is_zero && fp16_b_is_inf)) begin
                            result <= 32'h7FC00000; // FP32 NaN
                        end else if (fp16_a_is_zero || fp16_b_is_zero) begin
                            result <= {fp16_a_sign ^ fp16_b_sign, 31'b0};
                        end else if (fp16_a_is_inf || fp16_b_is_inf) begin
                            result <= {fp16_a_sign ^ fp16_b_sign, 8'hFF, 23'b0};
                        end else begin
                            result <= fp32_mul(fp32_a, fp32_b); // Return FP32 product directly
                        end
                    end

                    `FP16_FMA: begin
                        // a * b + c
                        if (fp16_a_is_nan || fp16_b_is_nan ||
                            (op_c_r2[14:10] == 5'h1F && op_c_r2[9:0] != 10'b0)) begin
                            fp16_result_lo = FP16_NAN;
                            invalid <= 1'b1;
                        end else begin
                            fp16_result_lo = fp32_to_fp16(fp32_fma_ab_c);
                        end
                        result <= {16'b0, fp16_result_lo};
                    end

                    `FP16_NEG: begin
                        result <= {16'b0, ~op_a_r2[15], op_a_r2[14:0]};
                    end

                    `FP16_ABS: begin
                        result <= {16'b0, 1'b0, op_a_r2[14:0]};
                    end

                    `FP16_MIN: begin
                        if (fp16_a_is_nan) begin
                            result <= {16'b0, op_b_r2[15:0]};
                        end else if (fp16_b_is_nan) begin
                            result <= {16'b0, op_a_r2[15:0]};
                        end else begin
                            // Compare as signed integers for proper FP comparison
                            result <= fp16_less_than(op_a_r2[15:0], op_b_r2[15:0]) ?
                                     {16'b0, op_a_r2[15:0]} : {16'b0, op_b_r2[15:0]};
                        end
                    end

                    `FP16_MAX: begin
                        if (fp16_a_is_nan) begin
                            result <= {16'b0, op_b_r2[15:0]};
                        end else if (fp16_b_is_nan) begin
                            result <= {16'b0, op_a_r2[15:0]};
                        end else begin
                            result <= fp16_less_than(op_a_r2[15:0], op_b_r2[15:0]) ?
                                     {16'b0, op_b_r2[15:0]} : {16'b0, op_a_r2[15:0]};
                        end
                    end
                    //----------------------------------------------------
                    // FP16 Transcendental Operations (ML)
                    //----------------------------------------------------
                    `FP16_TANH: begin
                        // tanh(x) approximation
                        // For |x| >= 3.0, saturate to +/-1.0 (error < 0.005)
                        // For |x| < 3.0, use rational approx:
                        //   tanh(x) ~ x * (27 + x^2) / (27 + 9*x^2)
                        //   (Pade [1/1] based, max error ~2% over [-3,3])
                        if (fp16_a_is_nan) begin
                            result <= {16'b0, FP16_NAN};
                            invalid <= 1'b1;
                        end else if (fp16_a_is_zero) begin
                            result <= {16'b0, op_a_r2[15:0]};  // tanh(0) = 0, preserve sign
                        end else begin
                            begin : tanh_compute
                                reg [31:0] abs_x, x_sq, x_sq_9, numer, denom, ratio, tanh_approx;
                                abs_x = {1'b0, fp32_a[30:0]};
                                // |x| >= 3.0 (FP16 exp field >= 16 with mant >= 0x200, or exp >= 17)
                                // 3.0 in FP16 = 0x4200 (exp=16, mant=0x200)
                                // Simpler: compare raw magnitude bits
                                if ({1'b0, op_a_r2[14:0]} >= 16'h4200) begin
                                    result <= {16'b0, op_a_r2[15], 5'b01111, 10'b0};  // +/-1.0
                                end else begin
                                    // Pade approx: tanh(x) ~ x * (27 + x^2) / (27 + 9*x^2)
                                    // 27.0 in FP32 = 0x41D80000
                                    // 9.0 in FP32  = 0x41100000
                                    x_sq = fp32_mul(abs_x, abs_x);
                                    // numerator = 27 + x^2
                                    numer = fp32_add(32'h41D80000, x_sq);
                                    // 9 * x^2
                                    x_sq_9 = fp32_mul(32'h41100000, x_sq);
                                    // denominator = 27 + 9*x^2
                                    denom = fp32_add(32'h41D80000, x_sq_9);
                                    // ratio = numer / denom (using multiply by reciprocal)
                                    // 1/denom approx: use Newton-Raphson or direct
                                    // For simplicity and accuracy, use iterative reciprocal:
                                    // r0 = 1/27 ~ 0.037 = 0x3D170A3D as initial guess
                                    // But safer: just divide exponents and mantissas
                                    // Actually for Verilog sim, we can use a division function
                                    ratio = fp32_div(numer, denom);
                                    // tanh(x) = sign(x) * |x| * ratio
                                    tanh_approx = fp32_mul(fp32_a, ratio);
                                    if ({1'b0, tanh_approx[30:0]} > 32'h3F800000) begin
                                        result <= {16'b0, tanh_approx[31], 5'b01111, 10'b0};
                                    end else begin
                                        result <= {16'b0, fp32_to_fp16(tanh_approx)};
                                    end
                                end
                            end
                            inexact <= 1'b1;
                        end
                    end

                    `FP16_EX2: begin
                        // 2^x using range reduction + polynomial
                        // Split x = n + f where n = floor(x), f = x - n (fractional)
                        // Then 2^x = 2^n * 2^f
                        // 2^f ~ 1 + f*ln2 + (f*ln2)^2/2 + (f*ln2)^3/6 + (f*ln2)^4/24
                        // Since |f| < 1, 4-term Taylor converges well
                        if (fp16_a_is_nan) begin
                            result <= {16'b0, FP16_NAN};
                            invalid <= 1'b1;
                        end else if (fp16_a_is_zero) begin
                            result <= {16'b0, FP16_ONE};  // 2^0 = 1.0
                        end else if (fp16_a_is_inf && !op_a_r2[15]) begin
                            result <= {16'b0, FP16_INF};  // 2^(+inf) = +inf
                        end else if (fp16_a_is_inf && op_a_r2[15]) begin
                            result <= {16'b0, FP16_ZERO}; // 2^(-inf) = 0
                        end else if (!op_a_r2[15] && op_a_r2[14:10] >= 5'd19) begin
                            // x >= 16.0 -> overflow
                            result <= {16'b0, FP16_INF};
                            overflow <= 1'b1;
                        end else if (op_a_r2[15] && op_a_r2[14:10] >= 5'd19) begin
                            // x <= -16.0 -> underflow
                            result <= {16'b0, FP16_ZERO};
                            underflow <= 1'b1;
                        end else begin
                            // Range reduction: 2^x = 2^n * 2^f
                            // where n = floor(x), f = x - n
                            begin : ex2_compute
                                reg [31:0] x_fp32, n_fp32, f_fp32;
                                reg [31:0] fln2, fln2_sq, fln2_cu, fln2_4th;
                                reg [31:0] term2, term3, term4, exp_f;
                                reg signed [7:0] n_int;
                                reg [7:0] result_exp;
                                reg [15:0] fp16_exp_f;

                                x_fp32 = fp32_a;

                                // Extract integer part n = floor(x)
                                // For FP16 range [-15, 15], n fits in small integer
                                // Use FP32 floor: if exp >= 127, integer part exists
                                if (x_fp32[30:23] < 8'd127) begin
                                    // |x| < 1.0, so n=0 (positive) or n=-1 (negative)
                                    if (x_fp32[31]) begin
                                        n_int = -1;
                                        n_fp32 = 32'hBF800000; // -1.0
                                    end else begin
                                        n_int = 0;
                                        n_fp32 = 32'h00000000; // 0.0
                                    end
                                end else begin
                                    begin : extract_int
                                        reg [7:0] shift;
                                        reg [22:0] mant_full;
                                        reg [7:0] abs_n;
                                        shift = x_fp32[30:23] - 8'd127;
                                        mant_full = x_fp32[22:0];
                                        // Integer = (1.mantissa) >> (23 - shift)
                                        if (shift >= 8'd8)
                                            abs_n = 8'd15; // clamp
                                        else if (shift == 0)
                                            abs_n = 8'd1;
                                        else if (shift == 1)
                                            abs_n = {6'b0, 1'b1, mant_full[22]};
                                        else if (shift == 2)
                                            abs_n = {5'b0, 1'b1, mant_full[22:21]};
                                        else if (shift == 3)
                                            abs_n = {4'b0, 1'b1, mant_full[22:20]};
                                        else
                                            abs_n = {3'b0, 1'b1, mant_full[22:19]};

                                        if (x_fp32[31]) begin
                                            // For negative: floor(-2.3) = -3, need ceiling of magnitude
                                            // Check if there's a fractional part
                                            n_int = -abs_n;
                                            // Reconstruct n as FP32
                                            n_fp32 = {1'b1, x_fp32[30:23], 23'b0};
                                            // Mask out fractional bits
                                            if (shift < 23)
                                                n_fp32[22:0] = x_fp32[22:0] & ({23{1'b1}} << (23 - shift));
                                            // For negative floor: need to subtract 1 if there's a fraction
                                            f_fp32 = fp32_add(x_fp32, {~n_fp32[31], n_fp32[30:0]});
                                            if (f_fp32[31] && f_fp32[30:0] != 0) begin
                                                // x - trunc(x) < 0, so floor = trunc - 1
                                                n_int = n_int - 1;
                                                n_fp32 = fp32_add(n_fp32, 32'hBF800000);
                                            end
                                        end else begin
                                            n_int = abs_n;
                                            // Reconstruct n as FP32 (truncate fractional bits)
                                            n_fp32 = {1'b0, x_fp32[30:23], 23'b0};
                                            if (shift < 23)
                                                n_fp32[22:0] = x_fp32[22:0] & ({23{1'b1}} << (23 - shift));
                                        end
                                    end
                                end

                                // f = x - n (fractional part, 0 <= f < 1)
                                f_fp32 = fp32_add(x_fp32, {~n_fp32[31], n_fp32[30:0]});

                                // 2^f using Taylor: e^(f*ln2)
                                // ln(2) = 0x3F317218
                                fln2 = fp32_mul(f_fp32, 32'h3F317218);
                                fln2_sq = fp32_mul(fln2, fln2);
                                fln2_cu = fp32_mul(fln2_sq, fln2);
                                fln2_4th = fp32_mul(fln2_cu, fln2);

                                // term2 = fln2^2 / 2
                                if (fln2_sq[30:23] > 0)
                                    term2 = {fln2_sq[31], fln2_sq[30:23] - 8'd1, fln2_sq[22:0]};
                                else
                                    term2 = 32'b0;
                                // term3 = fln2^3 / 6 = fln2^3 * 0.16667
                                term3 = fp32_mul(fln2_cu, 32'h3E2AAAAB);
                                // term4 = fln2^4 / 24 = fln2^4 * 0.04167
                                term4 = fp32_mul(fln2_4th, 32'h3D2AAAAB);

                                // 2^f = 1 + fln2 + term2 + term3 + term4
                                exp_f = fp32_add(32'h3F800000, fln2);
                                exp_f = fp32_add(exp_f, term2);
                                exp_f = fp32_add(exp_f, term3);
                                exp_f = fp32_add(exp_f, term4);

                                // 2^x = 2^n * 2^f
                                // 2^n: adjust FP32 exponent by n
                                if (exp_f[30:23] == 0 || (exp_f[31] && n_int < 0)) begin
                                    result <= {16'b0, FP16_ZERO};
                                end else begin
                                    begin : scale_result
                                        reg signed [9:0] new_exp;
                                        new_exp = $signed({2'b0, exp_f[30:23]}) + $signed({{2{n_int[7]}}, n_int});
                                        if (new_exp >= 10'sd255) begin
                                            result <= {16'b0, FP16_INF};
                                            overflow <= 1'b1;
                                        end else if (new_exp <= 0) begin
                                            result <= {16'b0, FP16_ZERO};
                                            underflow <= 1'b1;
                                        end else begin
                                            result <= {16'b0, fp32_to_fp16({exp_f[31], new_exp[7:0], exp_f[22:0]})};
                                        end
                                    end
                                end
                            end
                            inexact <= 1'b1;
                        end
                    end


                    //----------------------------------------------------
                    // BF16 Operations
                    //----------------------------------------------------
                    `BF16_ADD: begin
                        if (bf16_a_is_nan || bf16_b_is_nan) begin
                            result <= {16'b0, BF16_NAN};
                            invalid <= 1'b1;
                        end else begin
                            result <= {16'b0, fp32_to_bf16(bf32_add_ab)};
                        end
                    end

                    `BF16_SUB: begin
                        result <= {16'b0, fp32_to_bf16(fp32_add(bf32_a,
                            {~bf32_b[31], bf32_b[30:0]}))};
                    end

                    `BF16_MUL: begin
                        if ((bf16_a_is_inf && bf16_b_is_zero) ||
                            (bf16_a_is_zero && bf16_b_is_inf)) begin
                            result <= {16'b0, BF16_NAN};
                            invalid <= 1'b1;
                        end else begin
                            result <= {16'b0, fp32_to_bf16(bf32_mul_ab)};
                        end
                    end

                    `BF16_FMA: begin
                        result <= {16'b0, fp32_to_bf16(
                            bf32_fma_ab_c)};
                    end

                    //----------------------------------------------------
                    // Packed FP16x2 Operations (SIMD)
                    //----------------------------------------------------
                    `FP16X2_ADD: begin
                        fp16_result_lo = fp32_to_fp16(fp32_add_ab);
                        fp16_result_hi = fp32_to_fp16(fp32_add_ab_hi);
                        result <= {fp16_result_hi, fp16_result_lo};
                    end

                    `FP16X2_SUB: begin
                        fp16_result_lo = fp32_to_fp16(fp32_add_ab_neg);
                        fp16_result_hi = fp32_to_fp16(fp32_add_ab_neg_hi);
                        result <= {fp16_result_hi, fp16_result_lo};
                    end

                    `FP16X2_MUL: begin
                        fp16_result_lo = fp32_to_fp16(fp32_mul_ab);
                        fp16_result_hi = fp32_to_fp16(fp32_mul_ab_hi);
                        result <= {fp16_result_hi, fp16_result_lo};
                    end

                    `FP16X2_FMA: begin
                        fp16_result_lo = fp32_to_fp16(fp32_fma_ab_c);
                        fp16_result_hi = fp32_to_fp16(fp32_fma_ab_c_hi);
                        result <= {fp16_result_hi, fp16_result_lo};
                    end

                    //----------------------------------------------------
                    // FP16 Comparison Operations (setp.f16)
                    // Returns 0xFFFF_FFFF (true) or 0x0000_0000 (false)
                    //----------------------------------------------------
                    `FP16_CMP_EQ: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h0;  // NaN comparisons return false
                        end else begin
                            result <= (op_a_r2[15:0] == op_b_r2[15:0]) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_NE: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'hFFFF_FFFF;  // NaN != anything is true
                        end else begin
                            result <= (op_a_r2[15:0] != op_b_r2[15:0]) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_LT: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h0;  // NaN comparisons return false
                        end else begin
                            result <= fp16_less_than(op_a_r2[15:0], op_b_r2[15:0]) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_LE: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h0;  // NaN comparisons return false
                        end else begin
                            result <= (fp16_less_than(op_a_r2[15:0], op_b_r2[15:0]) ||
                                      (op_a_r2[15:0] == op_b_r2[15:0])) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_GT: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h0;  // NaN comparisons return false
                        end else begin
                            result <= fp16_less_than(op_b_r2[15:0], op_a_r2[15:0]) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_GE: begin
                        if (fp16_a_is_nan || fp16_b_is_nan) begin
                            result <= 32'h0;  // NaN comparisons return false
                        end else begin
                            result <= (fp16_less_than(op_b_r2[15:0], op_a_r2[15:0]) ||
                                      (op_a_r2[15:0] == op_b_r2[15:0])) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_NUM: begin
                        // NUM (ordered): true if both are NOT NaN
                        result <= (fp16_a_is_nan || fp16_b_is_nan) ? 32'h0 : 32'hFFFF_FFFF;
                    end

                    `FP16_CMP_NAN: begin
                        // NAN (unordered): true if either is NaN
                        result <= (fp16_a_is_nan || fp16_b_is_nan) ? 32'hFFFF_FFFF : 32'h0;
                    end

                    //----------------------------------------------------
                    // Mixed FP16-FP32 Comparison Operations
                    // Operand A is FP16 (lower 16 bits), Operand B is FP32
                    // Convert FP16 to FP32, then compare as FP32
                    //----------------------------------------------------
                    `FP16_CMP_EQ_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'h0;
                        end else begin
                            result <= (fp32_a == op_b_r2) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_NE_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'hFFFF_FFFF;
                        end else begin
                            result <= (fp32_a != op_b_r2) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_LT_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'h0;
                        end else begin
                            result <= fp32_less_than(fp32_a, op_b_r2) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_LE_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'h0;
                        end else begin
                            result <= (fp32_less_than(fp32_a, op_b_r2) ||
                                      (fp32_a == op_b_r2)) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_GT_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'h0;
                        end else begin
                            result <= fp32_less_than(op_b_r2, fp32_a) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    `FP16_CMP_GE_F32: begin
                        if (fp16_a_is_nan || fp32_b_is_nan) begin
                            result <= 32'h0;
                        end else begin
                            result <= (fp32_less_than(op_b_r2, fp32_a) ||
                                      (fp32_a == op_b_r2)) ? 32'hFFFF_FFFF : 32'h0;
                        end
                    end

                    default: begin
                        result <= 32'b0;
                    end
                                /* verilator lint_on CASEOVERLAP */
endcase
            end
        end
    end

    //------------------------------------------------------------------------
    // FP16 comparison helper
    //------------------------------------------------------------------------
    function fp16_less_than;
        input [15:0] a, b;
        reg a_neg, b_neg;
        begin
            a_neg = a[15];
            b_neg = b[15];

            if (a_neg && !b_neg) begin
                fp16_less_than = 1'b1;  // negative < positive
            end else if (!a_neg && b_neg) begin
                fp16_less_than = 1'b0;  // positive > negative
            end else if (a_neg) begin
                // Both negative - larger magnitude is smaller
                fp16_less_than = (a[14:0] > b[14:0]);
            end else begin
                // Both positive - smaller magnitude is smaller
                fp16_less_than = (a[14:0] < b[14:0]);
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // FP32 comparison helper (for mixed FP16-FP32 comparisons)
    //------------------------------------------------------------------------
    function fp32_less_than;
        input [31:0] a, b;
        reg a_neg, b_neg;
        reg [7:0] a_exp, b_exp;
        reg [22:0] a_mant, b_mant;
        begin
            a_neg = a[31];
            b_neg = b[31];
            a_exp = a[30:23];
            b_exp = b[30:23];
            a_mant = a[22:0];
            b_mant = b[22:0];

            // Handle special cases for zeros (positive and negative zero are equal)
            if ((a_exp == 8'b0 && a_mant == 23'b0) && (b_exp == 8'b0 && b_mant == 23'b0)) begin
                fp32_less_than = 1'b0;  // +0 == -0
            end else if (a_neg && !b_neg) begin
                fp32_less_than = 1'b1;  // negative < positive
            end else if (!a_neg && b_neg) begin
                fp32_less_than = 1'b0;  // positive > negative
            end else if (a_neg) begin
                // Both negative - compare magnitudes (larger magnitude is smaller)
                if (a_exp > b_exp) begin
                    fp32_less_than = 1'b1;
                end else if (a_exp < b_exp) begin
                    fp32_less_than = 1'b0;
                end else begin
                    fp32_less_than = (a_mant > b_mant);
                end
            end else begin
                // Both positive - compare magnitudes (smaller magnitude is smaller)
                if (a_exp < b_exp) begin
                    fp32_less_than = 1'b1;
                end else if (a_exp > b_exp) begin
                    fp32_less_than = 1'b0;
                end else begin
                    fp32_less_than = (a_mant < b_mant);
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Simplified FP32 adder (for internal use)
    //------------------------------------------------------------------------
    function [31:0] fp32_add;
        input [31:0] a, b;
        reg         a_sign, b_sign, r_sign;
        reg [7:0]   a_exp, b_exp, r_exp;
        reg [23:0]  a_mant, b_mant;  // with implicit 1
        reg [24:0]  r_mant;
        reg [7:0]   exp_diff;
        reg         a_larger;
        begin
            a_sign = a[31];
            b_sign = b[31];
            a_exp  = a[30:23];
            b_exp  = b[30:23];
            a_mant = {1'b1, a[22:0]};
            b_mant = {1'b1, b[22:0]};

            // Handle zeros
            if (a_exp == 8'b0) begin
                a_mant = 24'b0;
            end
            if (b_exp == 8'b0) begin
                b_mant = 24'b0;
            end

            // Align exponents
            if (a_exp >= b_exp) begin
                exp_diff = a_exp - b_exp;
                b_mant = b_mant >> exp_diff;
                r_exp = a_exp;
                a_larger = 1'b1;
            end else begin
                exp_diff = b_exp - a_exp;
                a_mant = a_mant >> exp_diff;
                r_exp = b_exp;
                a_larger = 1'b0;
            end

            // Add or subtract
            if (a_sign == b_sign) begin
                r_mant = {1'b0, a_mant} + {1'b0, b_mant};
                r_sign = a_sign;
            end else begin
                // Different signs: subtract mantissas, sign follows larger magnitude
                if (a_mant >= b_mant) begin
                    r_mant = {1'b0, a_mant} - {1'b0, b_mant};
                    r_sign = a_sign;  // A has larger aligned magnitude
                end else begin
                    r_mant = {1'b0, b_mant} - {1'b0, a_mant};
                    r_sign = b_sign;  // B has larger aligned magnitude
                end
            end

            // Normalize
            if (r_mant[24]) begin
                r_mant = r_mant >> 1;
                r_exp = r_exp + 1;
            end else if (r_mant != 25'b0) begin
                // Fixed-iteration normalization (synthesizable, replaces while loop)
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
                if (!r_mant[23] && r_exp > 0) begin r_mant = r_mant << 1; r_exp = r_exp - 1; end
            end

            // Check for zero result
            if (r_mant == 25'b0) begin
                fp32_add = 32'b0;
            end else begin
                fp32_add = {r_sign, r_exp, r_mant[22:0]};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Simplified FP32 multiplier (for internal use)
    //------------------------------------------------------------------------
    function [31:0] fp32_mul;
        input [31:0] a, b;
        reg         r_sign;
        reg [8:0]   r_exp;
        reg [23:0]  a_mant, b_mant;
        reg [47:0]  r_mant_full;
        reg [22:0]  r_mant;
        begin
            r_sign = a[31] ^ b[31];

            // Handle zeros
            if (a[30:23] == 8'b0 || b[30:23] == 8'b0) begin
                fp32_mul = {r_sign, 31'b0};
            end else begin
                a_mant = {1'b1, a[22:0]};
                b_mant = {1'b1, b[22:0]};

                r_mant_full = a_mant * b_mant;
                r_exp = a[30:23] + b[30:23] - 8'd127;

                // Normalize
                if (r_mant_full[47]) begin
                    r_mant = r_mant_full[46:24];
                    r_exp = r_exp + 1;
                end else begin
                    r_mant = r_mant_full[45:23];
                end

                // Overflow/underflow check
                if (r_exp >= 9'd255) begin
                    fp32_mul = {r_sign, 8'hFF, 23'b0};  // Infinity
                end else if (r_exp[8]) begin
                    fp32_mul = {r_sign, 31'b0};  // Underflow to zero
                end else begin
                    fp32_mul = {r_sign, r_exp[7:0], r_mant};
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Simplified FP32 divider (for internal use)
    //------------------------------------------------------------------------
    function [31:0] fp32_div;
        input [31:0] a, b;
        reg         r_sign;
        reg [7:0]   a_exp, b_exp;
        reg [23:0]  a_mant, b_mant;
        reg [47:0]  a_ext;
        reg [24:0]  q;  // 25 bits to handle quotient >= 2^24
        reg [8:0]   q_exp;
        begin
            r_sign = a[31] ^ b[31];
            a_exp = a[30:23];
            b_exp = b[30:23];

            if (b_exp == 8'b0) begin
                fp32_div = {r_sign, 8'hFF, 23'b0};
            end else if (a_exp == 8'b0) begin
                fp32_div = {r_sign, 31'b0};
            end else begin
                a_mant = {1'b1, a[22:0]};
                b_mant = {1'b1, b[22:0]};

                // Compute mantissa quotient: (a_mant << 24) / b_mant
                a_ext = {a_mant, 24'b0};
                q = a_ext / {1'b0, b_mant};  // 48-bit / 25-bit -> 25-bit quotient

                // Base exponent: a/b = q * 2^(a_exp - b_exp - 24)
                // IEEE: (q/2^23) * 2^(q_exp - 127), so q_exp = a_exp - b_exp + 126
                q_exp = {1'b0, a_exp} - {1'b0, b_exp} + 9'd126;

                // When a_mant >= b_mant: q in [2^24, 2^25), shift right & inc exp
                // When a_mant < b_mant: q in [2^23, 2^24), already normalized
                if (q[24]) begin
                    q = q >> 1;
                    q_exp = q_exp + 1;
                end

                if (q_exp >= 9'd255) begin
                    fp32_div = {r_sign, 8'hFF, 23'b0};
                end else if (q_exp == 0 || q_exp[8]) begin
                    fp32_div = {r_sign, 31'b0};
                end else begin
                    fp32_div = {r_sign, q_exp[7:0], q[22:0]};
                end
            end
        end
    endfunction

endmodule

//============================================================================
// SIMD FP16 Unit - 32 lanes for warp-wide operations
//============================================================================
module fp16_simd_unit #(
    parameter NUM_LANES = 32
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [5:0]              func,
    input  wire                    valid_in,
    input  wire                    packed_mode,
    input  wire [NUM_LANES*32-1:0] operand_a,
    input  wire [NUM_LANES*32-1:0] operand_b,
    input  wire [NUM_LANES*32-1:0] operand_c,
    input  wire [NUM_LANES-1:0]    lane_mask,
    output wire [NUM_LANES*32-1:0] result,
    output wire                    valid_out
);

    wire [NUM_LANES-1:0] lane_valid_out;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : fp16_lanes
            fp16_unit u_fp16 (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .valid_in   (valid_in & lane_mask[i]),
                .packed_mode(packed_mode),
                .operand_a  (operand_a[i*32 +: 32]),
                .operand_b  (operand_b[i*32 +: 32]),
                .operand_c  (operand_c[i*32 +: 32]),
                .result     (result[i*32 +: 32]),
                .valid_out  (lane_valid_out[i]),
                .overflow   (),
                .underflow  (),
                .inexact    (),
                .invalid    ()
            );
        end
    endgenerate

    assign valid_out = |lane_valid_out;

endmodule
