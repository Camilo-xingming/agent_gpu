//============================================================================
// RalphGPU - ALU (Arithmetic Logic Unit)
// 32位算术逻辑单元，支持PTX基本运算
//============================================================================

`include "gpu_defines.vh"

module alu (
    input  wire [5:0]  func,        // 功能码
    input  wire [31:0] operand_a,   // 操作数A
    input  wire [31:0] operand_b,   // 操作数B
    output reg  [31:0] result,      // 结果
    output wire        zero,        // 零标志
    output wire        negative,    // 负数标志
    output wire        overflow     // 溢出标志
);

    //------------------------------------------------------------------------
    // 内部信号
    //------------------------------------------------------------------------
    wire [32:0] add_result;
    wire [32:0] sub_result;
    wire signed [31:0] signed_a;
    wire signed [31:0] signed_b;

    assign signed_a = operand_a;
    assign signed_b = operand_b;
    assign add_result = {1'b0, operand_a} + {1'b0, operand_b};
    assign sub_result = {1'b0, operand_a} - {1'b0, operand_b};

    //------------------------------------------------------------------------
    // ALU 操作选择
    //------------------------------------------------------------------------
    always @(*) begin
        case (func)
            `FUNC_ADD:   result = add_result[31:0];
            `FUNC_SUB:   result = sub_result[31:0];
            `FUNC_AND:   result = operand_a & operand_b;
            `FUNC_OR:    result = operand_a | operand_b;
            `FUNC_XOR:   result = operand_a ^ operand_b;
            `FUNC_NOT:   result = ~operand_a;
            `FUNC_SHL:   result = operand_a << operand_b[4:0];
            `FUNC_SHR_U: result = operand_a >> operand_b[4:0];
            `FUNC_SHR_S: result = signed_a >>> operand_b[4:0];
            default:     result = 32'b0;
        endcase
    end

    //------------------------------------------------------------------------
    // 标志位生成
    //------------------------------------------------------------------------
    assign zero = (result == 32'b0);
    assign negative = result[31];

    // 溢出检测：加法时两个正数得负数，或两个负数得正数
    wire add_overflow = (~operand_a[31] & ~operand_b[31] & result[31]) |
                        (operand_a[31] & operand_b[31] & ~result[31]);
    wire sub_overflow = (~operand_a[31] & operand_b[31] & result[31]) |
                        (operand_a[31] & ~operand_b[31] & ~result[31]);

    assign overflow = (func == `FUNC_ADD) ? add_overflow :
                      (func == `FUNC_SUB) ? sub_overflow : 1'b0;

endmodule


//============================================================================
// SIMD ALU - 32个并行ALU用于Warp执行
// 每个Warp的32个线程同时执行
//============================================================================
module simd_alu #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand_a,  // 32个操作数A
    input  wire [LANES*32-1:0]  operand_b,  // 32个操作数B
    input  wire [LANES-1:0]     lane_mask,  // 活跃线程掩码
    output wire [LANES*32-1:0]  result,     // 32个结果
    output wire [LANES-1:0]     zero_flags,
    output wire [LANES-1:0]     neg_flags
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : alu_lane
            wire [31:0] lane_a = operand_a[i*32 +: 32];
            wire [31:0] lane_b = operand_b[i*32 +: 32];
            wire [31:0] lane_result;
            wire lane_zero, lane_neg, lane_ovf;

            alu u_alu (
                .func      (func),
                .operand_a (lane_a),
                .operand_b (lane_b),
                .result    (lane_result),
                .zero      (lane_zero),
                .negative  (lane_neg),
                .overflow  (lane_ovf)
            );

            // 只有活跃线程的结果有效
            assign result[i*32 +: 32] = lane_mask[i] ? lane_result : 32'b0;
            assign zero_flags[i] = lane_mask[i] & lane_zero;
            assign neg_flags[i] = lane_mask[i] & lane_neg;
        end
    endgenerate

endmodule
