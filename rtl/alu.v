//============================================================================
// RalphGPU - ALU (Arithmetic Logic Unit)
// 32位算术逻辑单元，支持完整PTX整数运算指令集
// 支持: 基础算术、位操作、位域操作、选择操作
//============================================================================

`include "gpu_defines.vh"

module alu (
    input  wire [5:0]  func,        // 功能码
    input  wire [31:0] operand_a,   // 操作数A
    input  wire [31:0] operand_b,   // 操作数B
    input  wire [31:0] operand_c,   // 操作数C (用于BFI, PRMT, SAD, SELP)
    input  wire        pred_in,     // 谓词输入 (用于SELP)
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
    // POPC (Population Count) - 计算1的个数
    //------------------------------------------------------------------------
    function [5:0] popc32;
        input [31:0] val;
        integer i;
        begin
            popc32 = 0;
            for (i = 0; i < 32; i = i + 1) begin
                popc32 = popc32 + val[i];
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // CLZ (Count Leading Zeros)
    //------------------------------------------------------------------------
    function [5:0] clz32;
        input [31:0] val;
        integer i;
        reg found;
        begin
            clz32 = 32;
            found = 0;
            for (i = 31; i >= 0 && !found; i = i - 1) begin
                if (val[i]) begin
                    clz32 = 31 - i;
                    found = 1;
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BFIND (Find Most Significant Bit) - 返回MSB位置
    //------------------------------------------------------------------------
    function [31:0] bfind32;
        input [31:0] val;
        input        is_signed;
        reg [31:0] search_val;
        integer i;
        reg found;
        begin
            // 对于有符号数，如果是负数，先取反
            search_val = (is_signed && val[31]) ? ~val : val;
            bfind32 = 32'hFFFFFFFF;  // -1 表示未找到
            found = 0;
            for (i = 31; i >= 0 && !found; i = i - 1) begin
                if (search_val[i]) begin
                    bfind32 = i;
                    found = 1;
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BREV (Bit Reverse)
    //------------------------------------------------------------------------
    function [31:0] brev32;
        input [31:0] val;
        integer i;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                brev32[31-i] = val[i];
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BFE (Bit Field Extract)
    // 从operand_a中提取从位置pos开始的len位
    // operand_b[7:0] = pos, operand_b[15:8] = len
    //------------------------------------------------------------------------
    wire [4:0] bfe_pos = operand_b[4:0];
    wire [4:0] bfe_len = operand_b[12:8];
    wire [31:0] bfe_mask = (bfe_len == 0) ? 32'b0 : ((32'hFFFFFFFF >> (32 - bfe_len)));
    wire [31:0] bfe_shifted = operand_a >> bfe_pos;
    wire [31:0] bfe_result_u = bfe_shifted & bfe_mask;
    // 有符号扩展
    wire bfe_sign_bit = (bfe_len > 0) ? bfe_shifted[bfe_len-1] : 1'b0;
    wire [31:0] bfe_sign_extend = (bfe_sign_bit && bfe_len > 0) ?
                                  (~bfe_mask) : 32'b0;
    wire [31:0] bfe_result_s = bfe_result_u | bfe_sign_extend;

    //------------------------------------------------------------------------
    // BFI (Bit Field Insert)
    // 将operand_a的低len位插入operand_b的pos位置
    // operand_c[7:0] = pos, operand_c[15:8] = len
    //------------------------------------------------------------------------
    wire [4:0] bfi_pos = operand_c[4:0];
    wire [4:0] bfi_len = operand_c[12:8];
    wire [31:0] bfi_mask = (bfi_len == 0) ? 32'b0 :
                           ((32'hFFFFFFFF >> (32 - bfi_len)) << bfi_pos);
    wire [31:0] bfi_insert = (operand_a << bfi_pos) & bfi_mask;
    wire [31:0] bfi_result = (operand_b & ~bfi_mask) | bfi_insert;

    //------------------------------------------------------------------------
    // PRMT (Permute Bytes)
    // 根据operand_c选择operand_a和operand_b的字节
    //------------------------------------------------------------------------
    wire [63:0] prmt_src = {operand_b, operand_a};  // 8个源字节
    wire [31:0] prmt_result;
    wire [2:0] prmt_sel0 = operand_c[2:0];
    wire [2:0] prmt_sel1 = operand_c[6:4];
    wire [2:0] prmt_sel2 = operand_c[10:8];
    wire [2:0] prmt_sel3 = operand_c[14:12];

    assign prmt_result[7:0]   = prmt_src[prmt_sel0*8 +: 8];
    assign prmt_result[15:8]  = prmt_src[prmt_sel1*8 +: 8];
    assign prmt_result[23:16] = prmt_src[prmt_sel2*8 +: 8];
    assign prmt_result[31:24] = prmt_src[prmt_sel3*8 +: 8];

    //------------------------------------------------------------------------
    // SAD (Sum of Absolute Differences)
    // result = |a - b| + c
    //------------------------------------------------------------------------
    wire signed [31:0] sad_diff = signed_a - signed_b;
    wire [31:0] sad_abs = sad_diff[31] ? (-sad_diff) : sad_diff;
    wire [31:0] sad_result = sad_abs + operand_c;

    //------------------------------------------------------------------------
    // ALU 操作选择
    //------------------------------------------------------------------------
    always @(*) begin
        case (func)
            // 基础运算
            `FUNC_ADD:   result = add_result[31:0];
            `FUNC_SUB:   result = sub_result[31:0];
            `FUNC_AND:   result = operand_a & operand_b;
            `FUNC_OR:    result = operand_a | operand_b;
            `FUNC_XOR:   result = operand_a ^ operand_b;
            `FUNC_NOT:   result = ~operand_a;
            `FUNC_SHL:   result = operand_a << operand_b[4:0];
            `FUNC_SHR_U: result = operand_a >> operand_b[4:0];
            `FUNC_SHR_S: result = signed_a >>> operand_b[4:0];

            // PTX扩展整数运算
            `FUNC_ABS:   result = signed_a[31] ? (-signed_a) : signed_a;
            `FUNC_NEG:   result = -signed_a;
            `FUNC_MIN_S: result = (signed_a < signed_b) ? operand_a : operand_b;
            `FUNC_MIN_U: result = (operand_a < operand_b) ? operand_a : operand_b;
            `FUNC_MAX_S: result = (signed_a > signed_b) ? operand_a : operand_b;
            `FUNC_MAX_U: result = (operand_a > operand_b) ? operand_a : operand_b;

            // 位操作指令
            `FUNC_POPC:  result = {26'b0, popc32(operand_a)};
            `FUNC_CLZ:   result = {26'b0, clz32(operand_a)};
            `FUNC_BFIND: result = bfind32(operand_a, 1'b1);  // 有符号版本
            `FUNC_BREV:  result = brev32(operand_a);

            // 位域操作
            `FUNC_BFE_S: result = bfe_result_s;
            `FUNC_BFE_U: result = bfe_result_u;
            `FUNC_BFI:   result = bfi_result;
            `FUNC_PRMT:  result = prmt_result;

            // 特殊运算
            `FUNC_SAD:   result = sad_result;

            // 选择操作
            `FUNC_SELP:  result = pred_in ? operand_a : operand_b;
            `FUNC_SLCT:  result = signed_b[31] ? operand_a : operand_b;  // 根据c的符号选择

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
// 支持完整PTX整数指令集
//============================================================================
module simd_alu #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand_a,  // 32个操作数A
    input  wire [LANES*32-1:0]  operand_b,  // 32个操作数B
    input  wire [LANES*32-1:0]  operand_c,  // 32个操作数C (BFI, PRMT, SAD, SELP)
    input  wire [LANES-1:0]     pred_in,    // 32个谓词输入 (SELP)
    input  wire [LANES-1:0]     lane_mask,  // 活跃线程掩码
    output wire [LANES*32-1:0]  result,     // 32个结果
    output wire [LANES-1:0]     zero_flags,
    output wire [LANES-1:0]     neg_flags,
    output wire [LANES-1:0]     ovf_flags
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : alu_lane
            wire [31:0] lane_a = operand_a[i*32 +: 32];
            wire [31:0] lane_b = operand_b[i*32 +: 32];
            wire [31:0] lane_c = operand_c[i*32 +: 32];
            wire lane_pred = pred_in[i];
            wire [31:0] lane_result;
            wire lane_zero, lane_neg, lane_ovf;

            alu u_alu (
                .func      (func),
                .operand_a (lane_a),
                .operand_b (lane_b),
                .operand_c (lane_c),
                .pred_in   (lane_pred),
                .result    (lane_result),
                .zero      (lane_zero),
                .negative  (lane_neg),
                .overflow  (lane_ovf)
            );

            // 只有活跃线程的结果有效
            assign result[i*32 +: 32] = lane_mask[i] ? lane_result : 32'b0;
            assign zero_flags[i] = lane_mask[i] & lane_zero;
            assign neg_flags[i] = lane_mask[i] & lane_neg;
            assign ovf_flags[i] = lane_mask[i] & lane_ovf;
        end
    endgenerate

endmodule
