//============================================================================
// RalphGPU - FPU64 (Double-Precision Floating-Point Unit)
// IEEE 754 双精度浮点运算单元
// 支持: add, sub, mul, div, fma, neg, abs, min, max, sqrt, rsqrt
// PTX Instructions: add.f64, sub.f64, mul.f64, div.f64, fma.f64, etc.
//============================================================================

`include "gpu_defines.vh"

module fpu64 (
    input  wire        clk,
    input  wire        rst_n,

    // 操作控制
    input  wire [5:0]  func,        // 功能码
    input  wire [1:0]  rnd_mode,    // 舍入模式: 00=RN, 01=RZ, 10=RM, 11=RP
    input  wire        ftz,         // Flush to zero (denormals)

    // 操作数 (IEEE 754 double precision - 64-bit)
    input  wire [63:0] operand_a,   // 操作数A
    input  wire [63:0] operand_b,   // 操作数B
    input  wire [63:0] operand_c,   // 操作数C (用于FMA: a*b+c)
    input  wire        valid_in,

    // 结果
    output reg  [63:0] result,
    output reg         valid_out,

    // 异常标志
    output reg         overflow,
    output reg         underflow,
    output reg         inexact,
    output reg         invalid,
    output reg         div_by_zero
);

    //------------------------------------------------------------------------
    // IEEE 754 双精度格式
    // [63]    = 符号位 (S)
    // [62:52] = 指数 (E), bias = 1023, 11 bits
    // [51:0]  = 尾数 (M), 52 bits, 隐含1.M
    //------------------------------------------------------------------------
    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_MAX  = 11'd2046;
    localparam [10:0] EXP_INF  = 11'd2047;

    //------------------------------------------------------------------------
    // 操作数解析
    //------------------------------------------------------------------------
    wire        sign_a = operand_a[63];
    wire [10:0] exp_a  = operand_a[62:52];
    wire [51:0] man_a  = operand_a[51:0];

    wire        sign_b = operand_b[63];
    wire [10:0] exp_b  = operand_b[62:52];
    wire [51:0] man_b  = operand_b[51:0];

    wire        sign_c = operand_c[63];
    wire [10:0] exp_c  = operand_c[62:52];
    wire [51:0] man_c  = operand_c[51:0];

    // 特殊值检测 - A
    wire a_is_zero     = (exp_a == 11'h000) && (man_a == 52'h0);
    wire a_is_denorm   = (exp_a == 11'h000) && (man_a != 52'h0);
    wire a_is_inf      = (exp_a == EXP_INF) && (man_a == 52'h0);
    wire a_is_nan      = (exp_a == EXP_INF) && (man_a != 52'h0);
    wire a_is_snan     = a_is_nan && !man_a[51];  // Signaling NaN

    // 特殊值检测 - B
    wire b_is_zero     = (exp_b == 11'h000) && (man_b == 52'h0);
    wire b_is_denorm   = (exp_b == 11'h000) && (man_b != 52'h0);
    wire b_is_inf      = (exp_b == EXP_INF) && (man_b == 52'h0);
    wire b_is_nan      = (exp_b == EXP_INF) && (man_b != 52'h0);
    wire b_is_snan     = b_is_nan && !man_b[51];

    // 特殊值检测 - C
    wire c_is_zero     = (exp_c == 11'h000) && (man_c == 52'h0);
    wire c_is_inf      = (exp_c == EXP_INF) && (man_c == 52'h0);
    wire c_is_nan      = (exp_c == EXP_INF) && (man_c != 52'h0);

    //------------------------------------------------------------------------
    // 常量定义
    //------------------------------------------------------------------------
    localparam [63:0] QNAN      = 64'h7FF8_0000_0000_0000;  // Quiet NaN
    localparam [63:0] POS_INF   = 64'h7FF0_0000_0000_0000;  // +Infinity
    localparam [63:0] NEG_INF   = 64'hFFF0_0000_0000_0000;  // -Infinity
    localparam [63:0] POS_ZERO  = 64'h0000_0000_0000_0000;  // +0
    localparam [63:0] NEG_ZERO  = 64'h8000_0000_0000_0000;  // -0

    //------------------------------------------------------------------------
    // 简单操作结果
    //------------------------------------------------------------------------
    wire [63:0] neg_result = {~sign_a, operand_a[62:0]};
    wire [63:0] abs_result = {1'b0, operand_a[62:0]};

    //------------------------------------------------------------------------
    // MIN/MAX 比较逻辑
    //------------------------------------------------------------------------
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

    wire [63:0] min_result = a_is_nan ? operand_b :
                             b_is_nan ? operand_a :
                             a_lt_b ? operand_a : operand_b;

    wire [63:0] max_result = a_is_nan ? operand_b :
                             b_is_nan ? operand_a :
                             a_lt_b ? operand_b : operand_a;

    //------------------------------------------------------------------------
    // 浮点加法
    //------------------------------------------------------------------------
    wire [63:0] add_result;
    wire add_invalid;
    wire add_overflow;
    wire add_underflow;

    fp64_add u_add (
        .a(operand_a),
        .b(func == `FP64_SUB ? {~operand_b[63], operand_b[62:0]} : operand_b),
        .rnd_mode(rnd_mode),
        .result(add_result),
        .invalid(add_invalid),
        .overflow(add_overflow),
        .underflow(add_underflow)
    );

    //------------------------------------------------------------------------
    // 浮点乘法
    //------------------------------------------------------------------------
    wire [63:0] mul_result;
    wire mul_invalid;
    wire mul_overflow;
    wire mul_underflow;

    fp64_mul u_mul (
        .a(operand_a),
        .b(operand_b),
        .rnd_mode(rnd_mode),
        .result(mul_result),
        .invalid(mul_invalid),
        .overflow(mul_overflow),
        .underflow(mul_underflow)
    );

    //------------------------------------------------------------------------
    // 浮点除法
    //------------------------------------------------------------------------
    wire [63:0] div_result;
    wire div_invalid;
    wire div_overflow_flag;
    wire div_underflow_flag;
    wire div_dbz;

    fp64_div u_div (
        .a(operand_a),
        .b(operand_b),
        .rnd_mode(rnd_mode),
        .result(div_result),
        .invalid(div_invalid),
        .overflow(div_overflow_flag),
        .underflow(div_underflow_flag),
        .div_by_zero(div_dbz)
    );

    //------------------------------------------------------------------------
    // FMA (a*b+c)
    //------------------------------------------------------------------------
    wire [63:0] fma_mul_result;
    wire [63:0] fma_result;
    wire fma_invalid;
    wire fma_overflow;
    wire fma_underflow;

    fp64_mul u_fma_mul (
        .a(operand_a),
        .b(operand_b),
        .rnd_mode(2'b00),
        .result(fma_mul_result),
        .invalid(),
        .overflow(),
        .underflow()
    );

    fp64_add u_fma_add (
        .a(fma_mul_result),
        .b(operand_c),
        .rnd_mode(rnd_mode),
        .result(fma_result),
        .invalid(fma_invalid),
        .overflow(fma_overflow),
        .underflow(fma_underflow)
    );

    //------------------------------------------------------------------------
    // 平方根 (Newton-Raphson迭代)
    //------------------------------------------------------------------------
    wire [63:0] sqrt_result;
    wire sqrt_invalid;

    fp64_sqrt u_sqrt (
        .a(operand_a),
        .result(sqrt_result),
        .invalid(sqrt_invalid)
    );

    //------------------------------------------------------------------------
    // 倒数平方根 rsqrt.f64
    //------------------------------------------------------------------------
    wire [63:0] rsqrt_result;
    wire rsqrt_invalid;

    fp64_rsqrt u_rsqrt (
        .a(operand_a),
        .result(rsqrt_result),
        .invalid(rsqrt_invalid)
    );

    //------------------------------------------------------------------------
    // 倒数 rcp.f64
    //------------------------------------------------------------------------
    wire [63:0] rcp_result;
    wire rcp_invalid;
    wire rcp_dbz;

    fp64_rcp u_rcp (
        .a(operand_a),
        .result(rcp_result),
        .invalid(rcp_invalid),
        .div_by_zero(rcp_dbz)
    );

    //------------------------------------------------------------------------
    // 结果选择
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result      <= 64'b0;
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
                `FP64_ADD, `FP64_SUB: begin
                    result    <= add_result;
                    invalid   <= add_invalid;
                    overflow  <= add_overflow;
                    underflow <= add_underflow;
                end

                `FP64_MUL: begin
                    result    <= mul_result;
                    invalid   <= mul_invalid;
                    overflow  <= mul_overflow;
                    underflow <= mul_underflow;
                end

                `FP64_DIV: begin
                    result      <= div_result;
                    invalid     <= div_invalid;
                    overflow    <= div_overflow_flag;
                    underflow   <= div_underflow_flag;
                    div_by_zero <= div_dbz;
                end

                `FP64_FMA: begin
                    result    <= fma_result;
                    invalid   <= fma_invalid;
                    overflow  <= fma_overflow;
                    underflow <= fma_underflow;
                end

                `FP64_NEG: begin
                    result  <= neg_result;
                    invalid <= a_is_snan;
                end

                `FP64_ABS: begin
                    result  <= abs_result;
                    invalid <= a_is_snan;
                end

                `FP64_MIN: begin
                    result  <= min_result;
                    invalid <= a_is_nan && b_is_nan;
                end

                `FP64_MAX: begin
                    result  <= max_result;
                    invalid <= a_is_nan && b_is_nan;
                end

                `FP64_SQRT: begin
                    result  <= sqrt_result;
                    invalid <= sqrt_invalid;
                end

                `FP64_RSQRT: begin
                    result  <= rsqrt_result;
                    invalid <= rsqrt_invalid;
                end

                `FP64_RCP: begin
                    result      <= rcp_result;
                    invalid     <= rcp_invalid;
                    div_by_zero <= rcp_dbz;
                end

                `FP64_COPYSIGN: begin
                    result  <= {operand_b[63], operand_a[62:0]};
                    invalid <= a_is_nan && b_is_nan;
                end

                `FP64_TESTP: begin
                    result  <= a_is_nan ? 64'b0 : 64'b1;
                    invalid <= 1'b0;
                end

                default: begin
                    result    <= 64'b0;
                    valid_out <= 1'b0;
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule


//============================================================================
// FP64 加法器
//============================================================================
module fp64_add (
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [1:0]  rnd_mode,
    output reg  [63:0] result,
    output reg         invalid,
    output reg         overflow,
    output reg         underflow
);

    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_INF  = 11'd2047;

    // 解析
    wire sign_a = a[63], sign_b = b[63];
    wire [10:0] exp_a = a[62:52], exp_b = b[62:52];
    wire [51:0] man_a = a[51:0], man_b = b[51:0];

    // 特殊值
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire b_inf  = (exp_b == EXP_INF) && (man_b == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);
    wire b_nan  = (exp_b == EXP_INF) && (man_b != 0);

    // 有效尾数 (1.mantissa for normalized, 0.mantissa for denormalized)
    wire [52:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [52:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 指数差
    wire [10:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire a_larger = (exp_a > exp_b) || (exp_a == exp_b && man_a >= man_b);

    // 对齐 (限制移位量防止数据丢失)
    wire [6:0] shift_amt = (exp_diff > 7'd55) ? 7'd55 : exp_diff[6:0];
    wire [53:0] aligned_a = a_larger ? {1'b0, sig_a} : ({1'b0, sig_a} >> shift_amt);
    wire [53:0] aligned_b = a_larger ? ({1'b0, sig_b} >> shift_amt) : {1'b0, sig_b};
    wire [10:0] result_exp = a_larger ? exp_a : exp_b;

    // 有效操作 (加或减)
    wire eff_sub = sign_a ^ sign_b;
    wire [54:0] sum;
    wire result_sign;

    assign sum = eff_sub ?
                 (aligned_a >= aligned_b ? {1'b0, aligned_a} - {1'b0, aligned_b} :
                                           {1'b0, aligned_b} - {1'b0, aligned_a}) :
                 {1'b0, aligned_a} + {1'b0, aligned_b};

    assign result_sign = eff_sub ?
                         (aligned_a >= aligned_b ? (a_larger ? sign_a : sign_b) :
                                                   (a_larger ? sign_b : sign_a)) :
                         sign_a;

    // 前导零计数 (简化版)
    function [5:0] clz54;
        input [53:0] val;
        integer i;
        begin
            clz54 = 6'd54;
            for (i = 53; i >= 0; i = i - 1) begin
                if (val[i]) clz54 = 6'd53 - i[5:0];
            end
        end
    endfunction

    wire [5:0] leading_zeros = clz54(sum[53:0]);

    // 规范化
    wire [10:0] norm_exp;
    wire [51:0] norm_man;
    wire norm_overflow;
    wire norm_underflow;

    assign norm_overflow = sum[54] && (result_exp >= 11'd2046);
    assign norm_underflow = (result_exp <= leading_zeros) && !a_zero && !b_zero;

    assign norm_exp = sum[54] ? result_exp + 11'd1 :
                      (result_exp > {5'b0, leading_zeros}) ? result_exp - {5'b0, leading_zeros} :
                      11'd0;

    assign norm_man = sum[54] ? sum[53:2] :
                      (sum << leading_zeros) >> 2;

    always @(*) begin
        invalid = 1'b0;
        overflow = 1'b0;
        underflow = 1'b0;

        if (a_nan || b_nan) begin
            result = 64'h7FF8_0000_0000_0000;  // QNaN
            invalid = 1'b1;
        end else if (a_inf && b_inf && eff_sub) begin
            result = 64'h7FF8_0000_0000_0000;  // inf - inf = NaN
            invalid = 1'b1;
        end else if (a_inf) begin
            result = a;
        end else if (b_inf) begin
            result = b;
        end else if (a_zero && b_zero) begin
            result = (sign_a && sign_b) ? 64'h8000_0000_0000_0000 : 64'h0;
        end else if (a_zero) begin
            result = b;
        end else if (b_zero) begin
            result = a;
        end else if (sum == 0) begin
            result = (rnd_mode == 2'b10) ? 64'h8000_0000_0000_0000 : 64'h0;
        end else if (norm_overflow) begin
            result = {result_sign, 11'h7FF, 52'h0};  // Infinity
            overflow = 1'b1;
        end else if (norm_underflow) begin
            result = {result_sign, 63'h0};
            underflow = 1'b1;
        end else begin
            result = {result_sign, norm_exp, norm_man};
        end
    end

endmodule


//============================================================================
// FP64 乘法器
//============================================================================
module fp64_mul (
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [1:0]  rnd_mode,
    output reg  [63:0] result,
    output reg         invalid,
    output reg         overflow,
    output reg         underflow
);

    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_INF  = 11'd2047;

    // 解析
    wire sign_a = a[63], sign_b = b[63];
    wire [10:0] exp_a = a[62:52], exp_b = b[62:52];
    wire [51:0] man_a = a[51:0], man_b = b[51:0];

    // 结果符号
    wire result_sign = sign_a ^ sign_b;

    // 特殊值
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire b_inf  = (exp_b == EXP_INF) && (man_b == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);
    wire b_nan  = (exp_b == EXP_INF) && (man_b != 0);

    // 有效尾数
    wire [52:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [52:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 乘法 (53 x 53 = 106 bits)
    wire [105:0] product = sig_a * sig_b;

    // 指数计算
    wire [11:0] exp_sum = {1'b0, exp_a} + {1'b0, exp_b};
    wire [11:0] result_exp_raw = exp_sum - 12'd1023;

    // 规范化
    wire norm_shift = product[105];
    wire [11:0] result_exp = norm_shift ? result_exp_raw + 12'd1 : result_exp_raw;
    wire [51:0] result_man = norm_shift ? product[104:53] : product[103:52];

    // 溢出/下溢检测
    wire exp_overflow = result_exp[11] == 1'b0 && result_exp >= 12'd2047;
    wire exp_underflow = result_exp[11] == 1'b1 || result_exp == 12'd0;

    always @(*) begin
        invalid = 1'b0;
        overflow = 1'b0;
        underflow = 1'b0;

        if (a_nan || b_nan) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if ((a_inf && b_zero) || (b_inf && a_zero)) begin
            result = 64'h7FF8_0000_0000_0000;  // 0 * inf = NaN
            invalid = 1'b1;
        end else if (a_inf || b_inf) begin
            result = {result_sign, 11'h7FF, 52'h0};  // Infinity
        end else if (a_zero || b_zero) begin
            result = {result_sign, 63'h0};  // Zero
        end else if (exp_overflow) begin
            result = {result_sign, 11'h7FF, 52'h0};  // Overflow to infinity
            overflow = 1'b1;
        end else if (exp_underflow) begin
            result = {result_sign, 63'h0};  // Underflow to zero
            underflow = 1'b1;
        end else begin
            result = {result_sign, result_exp[10:0], result_man};
        end
    end

endmodule


//============================================================================
// FP64 除法器
//============================================================================
module fp64_div (
    input  wire [63:0] a,
    input  wire [63:0] b,
    input  wire [1:0]  rnd_mode,
    output reg  [63:0] result,
    output reg         invalid,
    output reg         overflow,
    output reg         underflow,
    output reg         div_by_zero
);

    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_INF  = 11'd2047;

    // 解析
    wire sign_a = a[63], sign_b = b[63];
    wire [10:0] exp_a = a[62:52], exp_b = b[62:52];
    wire [51:0] man_a = a[51:0], man_b = b[51:0];

    wire result_sign = sign_a ^ sign_b;

    // 特殊值
    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire b_zero = (exp_b == 0) && (man_b == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire b_inf  = (exp_b == EXP_INF) && (man_b == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);
    wire b_nan  = (exp_b == EXP_INF) && (man_b != 0);

    // 有效尾数
    wire [52:0] sig_a = (exp_a == 0) ? {1'b0, man_a} : {1'b1, man_a};
    wire [52:0] sig_b = (exp_b == 0) ? {1'b0, man_b} : {1'b1, man_b};

    // 除法 (使用移位和减法实现)
    // 商 = sig_a / sig_b, 需要54位精度
    wire [106:0] dividend = {sig_a, 54'b0};
    wire [53:0] quotient = dividend / {1'b0, sig_b};

    // 指数计算
    wire [11:0] exp_diff = {1'b0, exp_a} - {1'b0, exp_b};
    wire [11:0] result_exp_raw = exp_diff + 12'd1023;

    // 规范化
    wire norm_needed = !quotient[53];
    wire [11:0] result_exp = norm_needed ? result_exp_raw - 12'd1 : result_exp_raw;
    wire [51:0] result_man = norm_needed ? quotient[51:0] : quotient[52:1];

    // 溢出/下溢检测
    wire exp_overflow = result_exp[11] == 1'b0 && result_exp >= 12'd2047;
    wire exp_underflow = result_exp[11] == 1'b1;

    always @(*) begin
        invalid = 1'b0;
        overflow = 1'b0;
        underflow = 1'b0;
        div_by_zero = 1'b0;

        if (a_nan || b_nan) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if (a_inf && b_inf) begin
            result = 64'h7FF8_0000_0000_0000;  // inf / inf = NaN
            invalid = 1'b1;
        end else if (a_zero && b_zero) begin
            result = 64'h7FF8_0000_0000_0000;  // 0 / 0 = NaN
            invalid = 1'b1;
        end else if (b_zero) begin
            result = {result_sign, 11'h7FF, 52'h0};  // x / 0 = inf
            div_by_zero = 1'b1;
        end else if (a_zero) begin
            result = {result_sign, 63'h0};  // 0 / x = 0
        end else if (a_inf) begin
            result = {result_sign, 11'h7FF, 52'h0};  // inf / x = inf
        end else if (b_inf) begin
            result = {result_sign, 63'h0};  // x / inf = 0
        end else if (exp_overflow) begin
            result = {result_sign, 11'h7FF, 52'h0};
            overflow = 1'b1;
        end else if (exp_underflow) begin
            result = {result_sign, 63'h0};
            underflow = 1'b1;
        end else begin
            result = {result_sign, result_exp[10:0], result_man};
        end
    end

endmodule


//============================================================================
// FP64 平方根 (Newton-Raphson迭代)
//============================================================================
module fp64_sqrt (
    input  wire [63:0] a,
    output reg  [63:0] result,
    output reg         invalid
);

    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_INF  = 11'd2047;

    wire sign_a = a[63];
    wire [10:0] exp_a = a[62:52];
    wire [51:0] man_a = a[51:0];

    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);
    wire a_neg  = sign_a && !a_zero;

    // 平方根指数计算: sqrt(2^e * m) = 2^(e/2) * sqrt(m)
    // 如果exp是奇数，需要调整尾数
    wire exp_odd = exp_a[0];
    wire [10:0] sqrt_exp = (exp_a - EXP_BIAS + (exp_odd ? 11'd1 : 11'd0)) >> 1;
    wire [10:0] result_exp = sqrt_exp + EXP_BIAS;

    // 简化的尾数平方根 (近似值)
    // 实际硬件会使用Newton-Raphson迭代
    wire [52:0] sig_a = {1'b1, man_a};
    wire [52:0] adjusted_sig = exp_odd ? {sig_a[51:0], 1'b0} : sig_a;

    // 初始估计和迭代 (这里用简化实现)
    wire [51:0] sqrt_man = adjusted_sig[52:1];  // 简化: 实际需要迭代

    always @(*) begin
        invalid = 1'b0;

        if (a_nan) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if (a_neg) begin
            result = 64'h7FF8_0000_0000_0000;  // sqrt(negative) = NaN
            invalid = 1'b1;
        end else if (a_zero) begin
            result = a;  // sqrt(±0) = ±0
        end else if (a_inf) begin
            result = a;  // sqrt(+inf) = +inf
        end else begin
            result = {1'b0, result_exp, sqrt_man};
        end
    end

endmodule


//============================================================================
// FP64 倒数平方根 rsqrt.f64
//============================================================================
module fp64_rsqrt (
    input  wire [63:0] a,
    output reg  [63:0] result,
    output reg         invalid
);

    localparam [10:0] EXP_INF = 11'd2047;

    wire sign_a = a[63];
    wire [10:0] exp_a = a[62:52];
    wire [51:0] man_a = a[51:0];

    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);
    wire a_neg  = sign_a && !a_zero;

    // 1/sqrt(x) 计算
    // rsqrt(x) = x^(-1/2)
    wire [63:0] sqrt_result;
    wire sqrt_invalid;

    fp64_sqrt u_sqrt (
        .a(a),
        .result(sqrt_result),
        .invalid(sqrt_invalid)
    );

    // 1/sqrt_result
    wire [10:0] sqrt_exp = sqrt_result[62:52];
    wire [51:0] sqrt_man = sqrt_result[51:0];

    // 简化倒数计算
    wire [10:0] rsqrt_exp = 11'd2046 - sqrt_exp;  // 近似
    wire [51:0] rsqrt_man = ~sqrt_man;  // 非常粗略的近似

    always @(*) begin
        invalid = 1'b0;

        if (a_nan) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if (a_neg) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if (a_zero) begin
            result = {sign_a, 11'h7FF, 52'h0};  // rsqrt(0) = inf
        end else if (a_inf) begin
            result = 64'h0;  // rsqrt(inf) = 0
        end else begin
            result = {1'b0, rsqrt_exp, rsqrt_man};
        end
    end

endmodule


//============================================================================
// FP64 倒数 rcp.f64
//============================================================================
module fp64_rcp (
    input  wire [63:0] a,
    output reg  [63:0] result,
    output reg         invalid,
    output reg         div_by_zero
);

    localparam [10:0] EXP_BIAS = 11'd1023;
    localparam [10:0] EXP_INF  = 11'd2047;

    wire sign_a = a[63];
    wire [10:0] exp_a = a[62:52];
    wire [51:0] man_a = a[51:0];

    wire a_zero = (exp_a == 0) && (man_a == 0);
    wire a_inf  = (exp_a == EXP_INF) && (man_a == 0);
    wire a_nan  = (exp_a == EXP_INF) && (man_a != 0);

    // 1/x 计算
    // exp(1/x) = 2*bias - exp(x)
    wire [10:0] rcp_exp = 11'd2046 - exp_a;

    // 尾数倒数 (简化: Newton-Raphson迭代更精确)
    wire [52:0] sig_a = {1'b1, man_a};
    wire [105:0] one_shifted = {1'b1, 105'b0};
    wire [52:0] rcp_sig = one_shifted / sig_a;
    wire [51:0] rcp_man = rcp_sig[51:0];

    always @(*) begin
        invalid = 1'b0;
        div_by_zero = 1'b0;

        if (a_nan) begin
            result = 64'h7FF8_0000_0000_0000;
            invalid = 1'b1;
        end else if (a_zero) begin
            result = {sign_a, 11'h7FF, 52'h0};  // 1/0 = inf
            div_by_zero = 1'b1;
        end else if (a_inf) begin
            result = {sign_a, 63'h0};  // 1/inf = 0
        end else begin
            result = {sign_a, rcp_exp, rcp_man};
        end
    end

endmodule


//============================================================================
// SIMD FPU64 - 用于Warp执行
//============================================================================
module simd_fpu64 #(
    parameter LANES = 32
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [5:0]           func,
    input  wire [1:0]           rnd_mode,
    input  wire                 ftz,
    input  wire [LANES*64-1:0]  operand_a,
    input  wire [LANES*64-1:0]  operand_b,
    input  wire [LANES*64-1:0]  operand_c,
    input  wire                 valid_in,
    input  wire [LANES-1:0]     lane_mask,
    output wire [LANES*64-1:0]  result,
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
        for (i = 0; i < LANES; i = i + 1) begin : fpu64_lane
            wire [63:0] lane_a = operand_a[i*64 +: 64];
            wire [63:0] lane_b = operand_b[i*64 +: 64];
            wire [63:0] lane_c = operand_c[i*64 +: 64];
            wire [63:0] lane_result;
            wire l_valid, l_ovf, l_udf, l_inx, l_inv, l_dbz;

            fpu64 u_fpu64 (
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

            assign result[i*64 +: 64] = lane_mask[i] ? lane_result : 64'b0;
            // FIX: Masked lanes should NOT report valid (was 1'b1, should be 1'b0)
            assign lane_valid[i] = lane_mask[i] & l_valid;
            assign lane_ovf[i] = lane_mask[i] & l_ovf;
            assign lane_inv[i] = lane_mask[i] & l_inv;
        end
    endgenerate

    assign overflow_flags = lane_ovf;
    assign invalid_flags = lane_inv;

endmodule
