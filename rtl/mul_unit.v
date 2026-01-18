//============================================================================
// RalphGPU - Multiply Unit
// 支持 mul.lo, mul.hi, mad (multiply-add)
//============================================================================

`include "gpu_defines.vh"

module mul_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire        is_div,      // 1: DIV/REM path, 0: MUL/MAD path
    input  wire [5:0]  func,
    input  wire [31:0] operand_a,
    input  wire [31:0] operand_b,
    input  wire [31:0] operand_c,   // 用于MAD
    output reg         valid_out,
    output reg  [31:0] result
);

    //------------------------------------------------------------------------
    // 乘法结果 (64位)
    //------------------------------------------------------------------------
    wire signed [31:0] signed_a = operand_a;
    wire signed [31:0] signed_b = operand_b;
    wire signed [63:0] mul_result_signed = signed_a * signed_b;
    wire [63:0] mul_result = mul_result_signed;

    // 24-bit multiply
    wire signed [23:0] signed_a24 = operand_a[23:0];
    wire signed [23:0] signed_b24 = operand_b[23:0];
    wire signed [47:0] mul24_result = signed_a24 * signed_b24;
    wire [31:0] mul24_lo = mul24_result[31:0];

    //------------------------------------------------------------------------
    // 除法/取余结果 (组合计算，除0返回0以避免X)
    //------------------------------------------------------------------------
    wire div_by_zero = (operand_b == 0);
    wire [31:0] div_result_u = div_by_zero ? 32'b0 : (operand_a / operand_b);
    wire [31:0] rem_result_u = div_by_zero ? 32'b0 : (operand_a % operand_b);
    wire signed [31:0] div_result_s = div_by_zero ? 32'sd0 : ($signed(operand_a) / $signed(operand_b));
    wire signed [31:0] rem_result_s = div_by_zero ? 32'sd0 : ($signed(operand_a) % $signed(operand_b));

    //------------------------------------------------------------------------
    // 两级流水线实现 (可选，用于提高频率)
    //------------------------------------------------------------------------
    reg [63:0] mul_reg;
    reg [31:0] operand_c_reg;
    reg [31:0] div_reg_s, div_reg_u;
    reg [31:0] rem_reg_s, rem_reg_u;
    reg [5:0]  func_reg;
    reg        is_div_reg;
    reg        valid_reg;

    reg carry_out_dummy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_reg <= 64'b0;
            operand_c_reg <= 32'b0;
            div_reg_s <= 32'b0;
            div_reg_u <= 32'b0;
            rem_reg_s <= 32'b0;
            rem_reg_u <= 32'b0;
            func_reg <= 6'b0;
            is_div_reg <= 1'b0;
            valid_reg <= 1'b0;
        end else begin
            mul_reg <= mul_result;
            operand_c_reg <= operand_c;
            div_reg_s <= div_result_s;
            div_reg_u <= div_result_u;
            rem_reg_s <= rem_result_s;
            rem_reg_u <= rem_result_u;
            func_reg <= func;
            is_div_reg <= valid_in ? is_div : 1'b0;
            valid_reg <= valid_in;
        end
    end

    //------------------------------------------------------------------------
    // 结果选择
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 32'b0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_reg;
            if (is_div_reg) begin
                case (func_reg)
                    `DIV_FUNC_DIV_S: result <= div_reg_s;
                    `DIV_FUNC_DIV_U: result <= div_reg_u;
                    `DIV_FUNC_REM_S: result <= rem_reg_s;
                    `DIV_FUNC_REM_U: result <= rem_reg_u;
                    default:         result <= div_reg_s;
                endcase
            end else begin
                case (func_reg)
                    `FUNC_MUL_LO: result <= mul_reg[31:0];
                    `FUNC_MUL_HI: result <= mul_reg[63:32];
                    `FUNC_MAD_LO: result <= mul_reg[31:0] + operand_c_reg;
                    `FUNC_MAD_HI: result <= mul_reg[63:32] + operand_c_reg;
                    `FUNC_MUL24:  result <= mul24_lo;
                    `FUNC_MAD24:  result <= mul24_lo + operand_c_reg;
                    `FUNC_MAD_LO_CC: begin
                        {carry_out_dummy, result} <= mul_reg[31:0] + operand_c_reg;
                    end
                    `FUNC_MADC_LO: begin
                        {carry_out_dummy, result} <= mul_reg[31:0] + operand_c_reg + 1'b1;
                    end
                    default:      result <= mul_reg[31:0];
                endcase
            end
        end
    end

endmodule


//============================================================================
// SIMD Multiply Unit - 32并行乘法器
//============================================================================
module simd_mul_unit #(
    parameter LANES = `THREADS_PER_WARP
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    input  wire                 is_div,
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand_a,
    input  wire [LANES*32-1:0]  operand_b,
    input  wire [LANES*32-1:0]  operand_c,
    input  wire [LANES-1:0]     lane_mask,
    output wire                 valid_out,
    output wire [LANES*32-1:0]  result
    );

    wire [LANES-1:0] lane_valid;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : mul_lane
            mul_unit u_mul (
                .clk       (clk),
                .rst_n     (rst_n),
                .valid_in  (valid_in & lane_mask[i]),
                .is_div    (is_div),
                .func      (func),
                .operand_a (operand_a[i*32 +: 32]),
                .operand_b (operand_b[i*32 +: 32]),
                .operand_c (operand_c[i*32 +: 32]),
                .valid_out (lane_valid[i]),
                .result    (result[i*32 +: 32])
            );
        end
    endgenerate

    assign valid_out = |lane_valid;

endmodule
