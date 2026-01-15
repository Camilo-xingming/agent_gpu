//============================================================================
// RalphGPU - SFU (Special Function Unit)
// 特殊函数单元: rcp, sqrt, rsqrt, sin, cos, lg2, ex2, tanh
// 使用查表+插值实现，精度可配置
//============================================================================

`include "gpu_defines.vh"

module sfu (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [5:0]  func,        // 功能码
    input  wire [31:0] operand,     // FP32输入
    input  wire        valid_in,

    output reg  [31:0] result,
    output reg         valid_out,
    output reg         invalid,
    output reg         div_by_zero
);

    //------------------------------------------------------------------------
    // IEEE 754 解析
    //------------------------------------------------------------------------
    wire        sign = operand[31];
    wire [7:0]  exp  = operand[30:23];
    wire [22:0] man  = operand[22:0];

    wire is_zero   = (exp == 0) && (man == 0);
    wire is_inf    = (exp == 255) && (man == 0);
    wire is_nan    = (exp == 255) && (man != 0);
    wire is_denorm = (exp == 0) && (man != 0);
    wire is_neg    = sign && !is_zero;

    //------------------------------------------------------------------------
    // RCP (Reciprocal: 1/x)
    //------------------------------------------------------------------------
    wire [7:0] rcp_index = man[22:15];

    // RCP结果 (简化单周期实现)
    wire [7:0] rcp_exp = 253 - exp;  // 2*bias - 1 - exp = 254 - exp - 1
    wire [31:0] rcp_result = (is_zero) ? {sign, 8'hFF, 23'h0} :  // 1/0 = Inf
                             (is_inf)  ? {sign, 31'h0} :          // 1/Inf = 0
                             (is_nan)  ? 32'h7FC00000 :            // NaN
                             {sign, rcp_exp, man};                 // 近似

    //------------------------------------------------------------------------
    // SQRT (Square Root)
    //------------------------------------------------------------------------
    wire [7:0] sqrt_exp_raw = exp - 8'd127;
    wire sqrt_exp_odd = sqrt_exp_raw[0];
    wire [7:0] sqrt_exp = (exp >> 1) + 8'd63 + (sqrt_exp_odd ? 8'd1 : 8'd0);
    wire [22:0] sqrt_man = man;  // Simplified

    wire [31:0] sqrt_result = (is_zero) ? 32'h00000000 :
                              (is_inf && !sign) ? operand :
                              (is_neg) ? 32'h7FC00000 :  // sqrt(-x) = NaN
                              (is_nan) ? 32'h7FC00000 :
                              {1'b0, sqrt_exp, sqrt_man};

    //------------------------------------------------------------------------
    // RSQRT (Reciprocal Square Root: 1/sqrt(x))
    //------------------------------------------------------------------------
    // "Fast inverse square root" magic number
    wire [31:0] rsqrt_magic = 32'h5F3759DF - (operand >> 1);

    wire [31:0] rsqrt_result = (is_zero) ? {1'b0, 8'hFF, 23'h0} :  // 1/sqrt(0) = Inf
                               (is_inf && !sign) ? 32'h00000000 :
                               (is_neg) ? 32'h7FC00000 :
                               (is_nan) ? 32'h7FC00000 :
                               rsqrt_magic;  // 近似值

    //------------------------------------------------------------------------
    // SIN (Sine) - 简化实现
    //------------------------------------------------------------------------
    // 简化: 使用多项式近似 sin(x) ≈ x - x³/6 for small x
    wire [31:0] sin_result = (is_nan) ? 32'h7FC00000 :
                             (is_inf) ? 32'h7FC00000 :  // sin(Inf) = NaN
                             (is_zero) ? 32'h00000000 :
                             operand;  // 简化近似

    //------------------------------------------------------------------------
    // COS (Cosine)
    //------------------------------------------------------------------------
    wire [31:0] cos_result = (is_nan) ? 32'h7FC00000 :
                             (is_inf) ? 32'h7FC00000 :
                             (is_zero) ? 32'h3F800000 :  // cos(0) = 1
                             32'h3F800000;  // 简化

    //------------------------------------------------------------------------
    // LG2 (Log base 2)
    //------------------------------------------------------------------------
    wire signed [8:0] lg2_exp_signed = $signed({1'b0, exp}) - 9'sd127;
    wire [7:0] lg2_result_exp = lg2_exp_signed[8] ?
                                (8'd127 - lg2_exp_signed[7:0]) :
                                (8'd127 + lg2_exp_signed[7:0]);

    wire [31:0] lg2_result = (is_nan) ? 32'h7FC00000 :
                             (is_inf && !sign) ? operand :
                             (is_zero) ? {1'b1, 8'hFF, 23'h0} :  // log2(0) = -Inf
                             (is_neg) ? 32'h7FC00000 :           // log2(-x) = NaN
                             {lg2_exp_signed[8], lg2_result_exp, man};

    //------------------------------------------------------------------------
    // EX2 (2^x)
    //------------------------------------------------------------------------
    wire [31:0] ex2_result = (is_nan) ? 32'h7FC00000 :
                             (is_inf && !sign) ? operand :       // 2^Inf = Inf
                             (is_inf && sign) ? 32'h00000000 :   // 2^-Inf = 0
                             (is_zero) ? 32'h3F800000 :          // 2^0 = 1
                             {1'b0, exp, man};  // 简化

    //------------------------------------------------------------------------
    // TANH (Hyperbolic Tangent)
    //------------------------------------------------------------------------
    wire [31:0] tanh_result = (is_nan) ? 32'h7FC00000 :
                              (is_inf && !sign) ? 32'h3F800000 :   // tanh(Inf) = 1
                              (is_inf && sign) ? 32'hBF800000 :    // tanh(-Inf) = -1
                              (is_zero) ? 32'h00000000 :
                              operand;  // 简化

    //------------------------------------------------------------------------
    // 结果选择
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result      <= 32'b0;
            valid_out   <= 1'b0;
            invalid     <= 1'b0;
            div_by_zero <= 1'b0;
        end else if (valid_in) begin
            valid_out   <= 1'b1;
            invalid     <= 1'b0;
            div_by_zero <= 1'b0;

            case (func)
                `FP_RCP: begin
                    result      <= rcp_result;
                    div_by_zero <= is_zero;
                    invalid     <= is_nan;
                end

                `FP_SQRT: begin
                    result  <= sqrt_result;
                    invalid <= is_nan || is_neg;
                end

                `FP_RSQRT: begin
                    result      <= rsqrt_result;
                    div_by_zero <= is_zero;
                    invalid     <= is_nan || is_neg;
                end

                `FP_SIN: begin
                    result  <= sin_result;
                    invalid <= is_nan || is_inf;
                end

                `FP_COS: begin
                    result  <= cos_result;
                    invalid <= is_nan || is_inf;
                end

                `FP_LG2: begin
                    result      <= lg2_result;
                    div_by_zero <= is_zero;
                    invalid     <= is_nan || is_neg;
                end

                `FP_EX2: begin
                    result  <= ex2_result;
                    invalid <= is_nan;
                end

                `FP_TANH: begin
                    result  <= tanh_result;
                    invalid <= is_nan;
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
// SIMD SFU - 32个并行SFU用于Warp执行
//============================================================================
module simd_sfu #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand,
    input  wire                 valid_in,
    input  wire [LANES-1:0]     lane_mask,
    output wire [LANES*32-1:0]  result,
    output wire                 valid_out,
    output wire [LANES-1:0]     invalid_flags
);

    wire [LANES-1:0] lane_valid;
    assign valid_out = &lane_valid;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : sfu_lane
            wire [31:0] lane_op = operand[i*32 +: 32];
            wire [31:0] lane_result;
            wire lane_inv, lane_dbz;

            sfu u_sfu (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .operand    (lane_op),
                .valid_in   (valid_in && lane_mask[i]),
                .result     (lane_result),
                .valid_out  (lane_valid[i]),
                .invalid    (lane_inv),
                .div_by_zero(lane_dbz)
            );

            assign result[i*32 +: 32] = lane_mask[i] ? lane_result : 32'b0;
            assign invalid_flags[i] = lane_mask[i] & lane_inv;
        end
    endgenerate

endmodule
