//============================================================================
// RalphGPU - FPU (Floating-Point Unit)
// IEEE 754 单精度浮点运算单元 (简化版)
// 支持: add, sub, mul, fma, neg, abs, min, max
// 可配置精度和面积权衡
//============================================================================

`include "gpu_defines.vh"

module fpu (
    input  wire        clk,
    input  wire        rst_n,

    // 操作控制
    input  wire [5:0]  func,        // 功能码
    input  wire [1:0]  rnd_mode,    // 舍入模式: 00=RN, 01=RZ, 10=RM, 11=RP
    input  wire        ftz,         // Flush to zero (denormals)

    // 操作数 (IEEE 754 single precision)
    input  wire [31:0] operand_a,   // 操作数A
    input  wire [31:0] operand_b,   // 操作数B
    input  wire [31:0] operand_c,   // 操作数C (用于FMA: a*b+c)
    input  wire        valid_in,

    // 结果
    output reg  [31:0] result,
    output reg         valid_out,

    // 异常标志
    output reg         overflow,
    output reg         underflow,
    output reg         inexact,
    output reg         invalid,
    output reg         div_by_zero
);

    //------------------------------------------------------------------------
    // IEEE 754 单精度格式
    // [31]    = 符号位 (S)
    // [30:23] = 指数 (E), bias = 127
    // [22:0]  = 尾数 (M), 隐含1.M
    //------------------------------------------------------------------------
    localparam [7:0] EXP_BIAS = 8'd127;

    //------------------------------------------------------------------------
    // 操作数解析
    //------------------------------------------------------------------------
    wire        sign_a = operand_a[31];
    wire [7:0]  exp_a  = operand_a[30:23];
    wire [22:0] man_a  = operand_a[22:0];

    wire        sign_b = operand_b[31];
    wire [7:0]  exp_b  = operand_b[30:23];
    wire [22:0] man_b  = operand_b[22:0];

    // 特殊值检测
    wire a_is_zero   = (exp_a == 8'h00) && (man_a == 23'h0);
    wire a_is_inf    = (exp_a == 8'hFF) && (man_a == 23'h0);
    wire a_is_nan    = (exp_a == 8'hFF) && (man_a != 23'h0);

    wire b_is_zero   = (exp_b == 8'h00) && (man_b == 23'h0);
    wire b_is_inf    = (exp_b == 8'hFF) && (man_b == 23'h0);
    wire b_is_nan    = (exp_b == 8'hFF) && (man_b != 23'h0);

    //------------------------------------------------------------------------
    // 简单操作结果
    //------------------------------------------------------------------------
    wire [31:0] neg_result = {~sign_a, operand_a[30:0]};
    wire [31:0] abs_result = {1'b0, operand_a[30:0]};

    // MIN/MAX
    wire a_lt_b_sign = sign_a && !sign_b;
    wire a_gt_b_sign = !sign_a && sign_b;
    wire both_pos = !sign_a && !sign_b;
    wire both_neg = sign_a && sign_b;
    wire a_exp_lt = exp_a < exp_b;
    wire a_exp_gt = exp_a > exp_b;
    wire a_exp_eq = exp_a == exp_b;
    wire a_man_lt = man_a < man_b;

    wire a_lt_b = a_is_nan ? 1'b0 :
                  b_is_nan ? 1'b1 :
                  a_lt_b_sign ? 1'b1 :
                  a_gt_b_sign ? 1'b0 :
                  both_pos ? (a_exp_lt || (a_exp_eq && a_man_lt)) :
                  both_neg ? (a_exp_gt || (a_exp_eq && !a_man_lt)) : 1'b0;

    wire [31:0] min_result = a_is_nan ? operand_b :
                             b_is_nan ? operand_a :
                             a_lt_b ? operand_a : operand_b;

    wire [31:0] max_result = a_is_nan ? operand_b :
                             b_is_nan ? operand_a :
                             a_lt_b ? operand_b : operand_a;

    //------------------------------------------------------------------------
    // 浮点加法 (简化实现)
    //------------------------------------------------------------------------
    wire [31:0] add_result;
    wire add_invalid;

    fp_add_simple u_add (
        .a(operand_a),
        .b(func == `FP_SUB ? {~operand_b[31], operand_b[30:0]} : operand_b),
        .result(add_result),
        .invalid(add_invalid)
    );

    //------------------------------------------------------------------------
    // 浮点乘法 (简化实现)
    //------------------------------------------------------------------------
    wire [31:0] mul_result;
    wire mul_invalid;

    fp_mul_simple u_mul (
        .a(operand_a),
        .b(operand_b),
        .result(mul_result),
        .invalid(mul_invalid)
    );

    //------------------------------------------------------------------------
    // FMA (简化: mul then add)
    //------------------------------------------------------------------------
    wire [31:0] fma_mul_result;
    wire [31:0] fma_result;
    wire fma_invalid;

    fp_mul_simple u_fma_mul (
        .a(operand_a),
        .b(operand_b),
        .result(fma_mul_result),
        .invalid()
    );

    fp_add_simple u_fma_add (
        .a(fma_mul_result),
        .b(operand_c),
        .result(fma_result),
        .invalid(fma_invalid)
    );

    //------------------------------------------------------------------------
    // 结果选择
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result      <= 32'b0;
            valid_out   <= 1'b0;
            overflow    <= 1'b0;
            underflow   <= 1'b0;
            inexact     <= 1'b0;
            invalid     <= 1'b0;
            div_by_zero <= 1'b0;
        end else if (valid_in) begin
            valid_out   <= 1'b1;
            overflow    <= 1'b0;
            underflow   <= 1'b0;
            inexact     <= 1'b0;
            invalid     <= 1'b0;
            div_by_zero <= 1'b0;

            case (func)
                `FP_ADD, `FP_SUB: begin
                    result  <= add_result;
                    invalid <= add_invalid;
                end

                `FP_MUL: begin
                    result  <= mul_result;
                    invalid <= mul_invalid;
                end

                `FP_FMA: begin
                    result  <= fma_result;
                    invalid <= fma_invalid;
                end

                `FP_NEG: begin
                    result  <= neg_result;
                    invalid <= a_is_nan;
                end

                `FP_ABS: begin
                    result  <= abs_result;
                    invalid <= a_is_nan;
                end

                `FP_MIN: begin
                    result  <= min_result;
                    invalid <= a_is_nan && b_is_nan;
                end

                `FP_MAX: begin
                    result  <= max_result;
                    invalid <= a_is_nan && b_is_nan;
                end

                `FP_DIV: begin
                    // 简化除法: 特殊情况处理
                    if (a_is_nan || b_is_nan) begin
                        result  <= 32'h7FC00000;
                        invalid <= 1'b1;
                    end else if (a_is_inf && b_is_inf) begin
                        result  <= 32'h7FC00000;
                        invalid <= 1'b1;
                    end else if (a_is_zero && b_is_zero) begin
                        result  <= 32'h7FC00000;
                        invalid <= 1'b1;
                    end else if (b_is_zero) begin
                        result  <= {sign_a ^ sign_b, 8'hFF, 23'h0};
                        div_by_zero <= 1'b1;
                    end else if (a_is_zero) begin
                        result <= {sign_a ^ sign_b, 31'h0};
                    end else if (a_is_inf) begin
                        result <= {sign_a ^ sign_b, 8'hFF, 23'h0};
                    end else if (b_is_inf) begin
                        result <= {sign_a ^ sign_b, 31'h0};
                    end else begin
                        // 实际除法需要迭代实现，这里简化为近似
                        result <= {sign_a ^ sign_b, 8'd127, 23'h0};
                        inexact <= 1'b1;
                    end
                end

                default: begin
                    result    <= 32'b0;
                    valid_out <= 1'b0;
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule


//============================================================================
// 简化版浮点加法器
//============================================================================
module fp_add_simple (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] result,
    output reg         invalid
);

    localparam [7:0] EXP_BIAS = 8'd127;

    // 解析
    wire sign_a = a[31], sign_b = b[31];
    wire [7:0] exp_a = a[30:23], exp_b = b[30:23];
    wire [22:0] man_a = a[22:0], man_b = b[22:0];

    // 特殊值
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == 255) && (man_a == 0);
    wire b_inf  = (exp_b == 255) && (man_b == 0);
    wire a_nan  = (exp_a == 255) && (man_a != 0);
    wire b_nan  = (exp_b == 255) && (man_b != 0);

    // 有效尾数
    wire [23:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [23:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 指数差
    wire [7:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire a_larger = (exp_a > exp_b) || (exp_a == exp_b && man_a >= man_b);

    // 对齐
    wire [24:0] aligned_a = a_larger ? {1'b0, sig_a} : ({1'b0, sig_a} >> exp_diff);
    wire [24:0] aligned_b = a_larger ? ({1'b0, sig_b} >> exp_diff) : {1'b0, sig_b};
    wire [7:0] result_exp = a_larger ? exp_a : exp_b;

    // 有效操作
    wire eff_sub = sign_a ^ sign_b;
    wire [25:0] sum;
    wire result_sign;

    assign sum = eff_sub ?
                 (aligned_a >= aligned_b ? {1'b0, aligned_a} - {1'b0, aligned_b} :
                                           {1'b0, aligned_b} - {1'b0, aligned_a}) :
                 {1'b0, aligned_a} + {1'b0, aligned_b};

    assign result_sign = eff_sub ?
                         (aligned_a >= aligned_b ? (a_larger ? sign_a : sign_b) :
                                                   (a_larger ? sign_b : sign_a)) :
                         sign_a;

    // 规范化 (简化)
    // When sum[25]=1: overflow by 2 positions, exp+2
    // When sum[24]=1: overflow by 1 position, exp+1
    // When sum[23]=1: no overflow, exp unchanged
    // Otherwise: underflow, exp-1
    wire [7:0] final_exp;
    wire [22:0] final_man;

    assign final_exp = sum[25] ? result_exp + 8'd2 :
                       sum[24] ? result_exp + 8'd1 :
                       sum[23] ? result_exp :
                       result_exp - 8'd1;

    assign final_man = sum[25] ? sum[24:2] :
                       sum[24] ? sum[23:1] :
                       sum[23] ? sum[22:0] :
                       {sum[21:0], 1'b0};

    always @(*) begin
        invalid = 1'b0;

        if (a_nan || b_nan) begin
            result = 32'h7FC00000;
            invalid = 1'b1;
        end else if (a_inf && b_inf && eff_sub) begin
            result = 32'h7FC00000;
            invalid = 1'b1;
        end else if (a_inf) begin
            result = a;
        end else if (b_inf) begin
            result = b;
        end else if (a_zero && b_zero) begin
            result = 32'h00000000;
        end else if (a_zero) begin
            result = b;
        end else if (b_zero) begin
            result = a;
        end else if (sum == 0) begin
            result = 32'h00000000;
        end else begin
            result = {result_sign, final_exp, final_man};
        end
    end

endmodule


//============================================================================
// 简化版浮点乘法器
//============================================================================
module fp_mul_simple (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] result,
    output reg         invalid
);

    localparam [7:0] EXP_BIAS = 8'd127;

    // 解析
    wire sign_a = a[31], sign_b = b[31];
    wire [7:0] exp_a = a[30:23], exp_b = b[30:23];
    wire [22:0] man_a = a[22:0], man_b = b[22:0];

    // 结果符号
    wire result_sign = sign_a ^ sign_b;

    // 特殊值
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == 255) && (man_a == 0);
    wire b_inf  = (exp_b == 255) && (man_b == 0);
    wire a_nan  = (exp_a == 255) && (man_a != 0);
    wire b_nan  = (exp_b == 255) && (man_b != 0);

    // 有效尾数
    wire [23:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [23:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 乘法
    wire [47:0] product = sig_a * sig_b;

    // 指数
    wire [8:0] exp_sum = {1'b0, exp_a} + {1'b0, exp_b};
    wire [8:0] result_exp_raw = exp_sum - 9'd127;

    // 规范化
    wire norm_shift = product[47];
    wire [8:0] result_exp = norm_shift ? result_exp_raw + 9'd1 : result_exp_raw;
    wire [22:0] result_man = norm_shift ? product[46:24] : product[45:23];

    always @(*) begin
        invalid = 1'b0;

        if (a_nan || b_nan) begin
            result = 32'h7FC00000;
            invalid = 1'b1;
        end else if ((a_inf && b_zero) || (b_inf && a_zero)) begin
            result = 32'h7FC00000;
            invalid = 1'b1;
        end else if (a_inf || b_inf) begin
            result = {result_sign, 8'hFF, 23'h0};
        end else if (a_zero || b_zero) begin
            result = {result_sign, 31'h0};
        end else if (result_exp[8] || result_exp >= 9'd255) begin
            // 溢出
            result = {result_sign, 8'hFF, 23'h0};
        end else if (result_exp == 0) begin
            // 下溢
            result = {result_sign, 31'h0};
        end else begin
            result = {result_sign, result_exp[7:0], result_man};
        end
    end

endmodule


//============================================================================
// SIMD FPU - 32个并行FPU用于Warp执行
// Note: Excluded when SM_V2 is defined (V2 has its own wrapper)
//============================================================================
`ifndef SM_V2
module simd_fpu #(
    parameter LANES = 32
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [5:0]           func,
    input  wire [1:0]           rnd_mode,
    input  wire                 ftz,
    input  wire [LANES*32-1:0]  operand_a,
    input  wire [LANES*32-1:0]  operand_b,
    input  wire [LANES*32-1:0]  operand_c,
    input  wire                 valid_in,
    input  wire [LANES-1:0]     lane_mask,
    output wire [LANES*32-1:0]  result,
    output wire                 valid_out,
    output wire [LANES-1:0]     overflow_flags,
    output wire [LANES-1:0]     invalid_flags
);

    wire [LANES-1:0] lane_valid;
    wire [LANES-1:0] lane_ovf;
    wire [LANES-1:0] lane_inv;

    assign valid_out = |lane_valid;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : fpu_lane
            wire [31:0] lane_a = operand_a[i*32 +: 32];
            wire [31:0] lane_b = operand_b[i*32 +: 32];
            wire [31:0] lane_c = operand_c[i*32 +: 32];
            wire [31:0] lane_result;
            wire l_valid, l_ovf, l_udf, l_inx, l_inv, l_dbz;

            fpu u_fpu (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .rnd_mode   (rnd_mode),
                .ftz        (ftz),
                .operand_a  (lane_a),
                .operand_b  (lane_b),
                .operand_c  (lane_c),
                .valid_in   (valid_in && lane_mask[i]),
                .result     (lane_result),
                .valid_out  (l_valid),
                .overflow   (l_ovf),
                .underflow  (l_udf),
                .inexact    (l_inx),
                .invalid    (l_inv),
                .div_by_zero(l_dbz)
            );

            assign result[i*32 +: 32] = lane_mask[i] ? lane_result : 32'b0;
            assign lane_valid[i] = lane_mask[i] ? l_valid : 1'b1;
            assign lane_ovf[i] = lane_mask[i] & l_ovf;
            assign lane_inv[i] = lane_mask[i] & l_inv;
        end
    endgenerate

    assign overflow_flags = lane_ovf;
    assign invalid_flags = lane_inv;

endmodule
`endif  // SM_V2
