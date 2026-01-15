//============================================================================
// RalphGPU - Multiply Unit
// 支持 mul.lo, mul.hi, mad (multiply-add)
//============================================================================

`include "gpu_defines.vh"

module mul_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
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

    //------------------------------------------------------------------------
    // 两级流水线实现 (可选，用于提高频率)
    //------------------------------------------------------------------------
    reg [63:0] mul_reg;
    reg [31:0] operand_c_reg;
    reg [5:0]  func_reg;
    reg        valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_reg <= 64'b0;
            operand_c_reg <= 32'b0;
            func_reg <= 6'b0;
            valid_reg <= 1'b0;
        end else begin
            mul_reg <= mul_result;
            operand_c_reg <= operand_c;
            func_reg <= func;
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
            case (func_reg)
                `FUNC_MUL_LO: result <= mul_reg[31:0];
                `FUNC_MUL_HI: result <= mul_reg[63:32];
                `FUNC_MAD_LO: result <= mul_reg[31:0] + operand_c_reg;
                default:      result <= mul_reg[31:0];
            endcase
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
