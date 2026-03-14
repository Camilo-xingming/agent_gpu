//============================================================================
// RalphGPU - Warp Shuffle Unit
// Warp级数据交换单元
// 支持: shfl.idx, shfl.up, shfl.down, shfl.bfly
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module warp_shuffle #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire [5:0]           func,          // Shuffle类型
    input  wire [LANES*32-1:0]  src_data,      // 源数据 (所有lane)
    input  wire [LANES*5-1:0]   src_lane,      // 源lane索引 (用于idx)
    input  wire [LANES*5-1:0]   offset,        // 偏移量 (用于up/down/bfly)
    input  wire [LANES-1:0]     lane_mask,     // 活跃lane掩码
    input  wire [4:0]           width,         // Shuffle宽度 (通常32)
    input  wire [LANES-1:0]     membermask,    // 参与shuffle的线程掩码

    output wire [LANES*32-1:0]  result,        // 结果
    output wire [LANES-1:0]     valid_out      // 源lane是否有效
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : shuffle_lane
            wire [4:0] my_lane = i;
            wire [4:0] lane_offset = offset[i*5 +: 5];
            wire [4:0] idx_lane = src_lane[i*5 +: 5];

            // 计算源lane
            reg [4:0] src_lane_id;
            reg lane_valid;

            always @(*) begin
                case (func)
                    // shfl.idx: 直接使用指定的lane
                    `SHFL_IDX: begin
                        src_lane_id = idx_lane;
                        // 检查源lane是否在有效范围内
                        lane_valid = (idx_lane < width) && membermask[idx_lane];
                    end

                    // shfl.up: 从低位lane获取数据
                    // 当前lane i 从 lane (i - offset) 获取
                    `SHFL_UP: begin
                        /* verilator lint_off CMPCONST */
                        if (my_lane >= lane_offset) begin
                        /* verilator lint_on CMPCONST */
                            src_lane_id = my_lane - lane_offset;
                            lane_valid = membermask[src_lane_id];
                        end else begin
                            src_lane_id = my_lane;  // 返回自己的值
                            lane_valid = 1'b0;      // 标记为无效
                        end
                    end

                    // shfl.down: 从高位lane获取数据
                    // 当前lane i 从 lane (i + offset) 获取
                    `SHFL_DOWN: begin
                        if ((my_lane + lane_offset) < width) begin
                            src_lane_id = my_lane + lane_offset;
                            lane_valid = membermask[src_lane_id];
                        end else begin
                            src_lane_id = my_lane;
                            lane_valid = 1'b0;
                        end
                    end

                    // shfl.bfly: 蝴蝶交换 (XOR)
                    // 当前lane i 从 lane (i ^ offset) 获取
                    `SHFL_BFLY: begin
                        src_lane_id = my_lane ^ lane_offset;
                        lane_valid = (src_lane_id < width) && membermask[src_lane_id];
                    end

                    default: begin
                        src_lane_id = my_lane;
                        lane_valid = 1'b0;
                    end
                endcase
            end

            // 获取源数据
            wire [31:0] src_value = src_data[src_lane_id*32 +: 32];

            // 输出
            assign result[i*32 +: 32] = lane_mask[i] ? src_value : 32'b0;
            assign valid_out[i] = lane_mask[i] & lane_valid;
        end
    endgenerate

endmodule


//============================================================================
// Warp Vote Unit
// Warp级投票单元
// 支持: vote.all, vote.any, vote.uni, vote.ballot
//============================================================================
module warp_vote #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire [5:0]           func,          // Vote类型
    input  wire [LANES-1:0]     pred_in,       // 每个lane的谓词输入
    input  wire [LANES-1:0]     lane_mask,     // 活跃lane掩码
    input  wire [LANES-1:0]     membermask,    // 参与vote的线程掩码

    output reg  [31:0]          result,        // 结果
    output reg                  pred_out       // 谓词结果 (all/any/uni)
);

    // 有效谓词 = 活跃lane AND 参与vote的线程
    wire [LANES-1:0] active_preds = pred_in & lane_mask & membermask;
    wire [LANES-1:0] active_mask  = lane_mask & membermask;

    // vote.all: 所有活跃线程的谓词都为真
    wire vote_all = (active_preds == active_mask) && (active_mask != 0);

    // vote.any: 至少一个活跃线程的谓词为真
    wire vote_any = |active_preds;

    // vote.uni: 所有活跃线程的谓词值相同 (全为0或全为1)
    wire vote_uni = (active_preds == active_mask) || (active_preds == 0);

    // vote.ballot: 返回所有线程的谓词位图
    wire [31:0] vote_ballot = {{(32-LANES){1'b0}}, active_preds};

    always @(*) begin
        case (func)
            `VOTE_ALL: begin
                result   = {31'b0, vote_all};
                pred_out = vote_all;
            end

            `VOTE_ANY: begin
                result   = {31'b0, vote_any};
                pred_out = vote_any;
            end

            `VOTE_UNI: begin
                result   = {31'b0, vote_uni};
                pred_out = vote_uni;
            end

            `VOTE_BALLOT: begin
                result   = vote_ballot;
                pred_out = vote_any;  // ballot也设置谓词
            end

            default: begin
                result   = 32'b0;
                pred_out = 1'b0;
            end
        endcase
    end

endmodule


//============================================================================
// Warp Reduction Unit
// Warp级归约单元 (redux.sync)
// 支持: add, min, max, and, or, xor
//============================================================================
module warp_reduction #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire [5:0]           func,          // 归约类型
    input  wire [LANES*32-1:0]  src_data,      // 源数据
    input  wire [LANES-1:0]     lane_mask,     // 活跃lane掩码
    input  wire [LANES-1:0]     membermask,    // 参与归约的线程掩码

    output reg  [31:0]          result         // 归约结果 (所有lane相同)
);
    `ifndef SYNTHESIS


    // 提取有效数据
    wire [31:0] data [0:LANES-1];
    wire [LANES-1:0] valid = lane_mask & membermask;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : extract_data
            assign data[i] = valid[i] ? src_data[i*32 +: 32] : 32'b0;
        end
    endgenerate

    // 树形归约 - 使用generate实现并行归约
    // Level 0: 32 -> 16
    wire [31:0] l0 [0:15];
    generate
        for (i = 0; i < 16; i = i + 1) begin : level0
            wire [31:0] a = data[i*2];
            wire [31:0] b = data[i*2+1];
            wire v_a = valid[i*2];
            wire v_b = valid[i*2+1];

            reduce_op u_reduce (
                .func(func),
                .a(a), .b(b),
                .valid_a(v_a), .valid_b(v_b),
                .result(l0[i])
            );
        end
    endgenerate

    // Level 1: 16 -> 8
    wire [31:0] l1 [0:7];
    generate
        for (i = 0; i < 8; i = i + 1) begin : level1
            reduce_op u_reduce (
                .func(func),
                .a(l0[i*2]), .b(l0[i*2+1]),
                .valid_a(1'b1), .valid_b(1'b1),
                .result(l1[i])
            );
        end
    endgenerate

    // Level 2: 8 -> 4
    wire [31:0] l2 [0:3];
    generate
        for (i = 0; i < 4; i = i + 1) begin : level2
            reduce_op u_reduce (
                .func(func),
                .a(l1[i*2]), .b(l1[i*2+1]),
                .valid_a(1'b1), .valid_b(1'b1),
                .result(l2[i])
            );
        end
    endgenerate

    // Level 3: 4 -> 2
    wire [31:0] l3 [0:1];
    generate
        for (i = 0; i < 2; i = i + 1) begin : level3
            reduce_op u_reduce (
                .func(func),
                .a(l2[i*2]), .b(l2[i*2+1]),
                .valid_a(1'b1), .valid_b(1'b1),
                .result(l3[i])
            );
        end
    endgenerate

    // Level 4: 2 -> 1
    wire [31:0] l4;
    reduce_op u_final (
        .func(func),
        .a(l3[0]), .b(l3[1]),
        .valid_a(1'b1), .valid_b(1'b1),
        .result(l4)
    );

    always @(*) begin
        result = l4;
    end

`else
    always @(*) result = 32'b0;
`endif
endmodule


//============================================================================
// Reduce Operation Helper
//============================================================================
module reduce_op (
    input  wire [5:0]  func,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire        valid_a,
    input  wire        valid_b,
    output reg  [31:0] result
);

    wire signed [31:0] signed_a = a;
    wire signed [31:0] signed_b = b;

    always @(*) begin
        if (!valid_a && !valid_b) begin
            result = 32'b0;
        end else if (!valid_a) begin
            result = b;
        end else if (!valid_b) begin
            result = a;
        end else begin
            case (func)
                `ATOM_ADD:   result = a + b;
                `ATOM_MIN_S: result = (signed_a < signed_b) ? a : b;
                `ATOM_MIN_U: result = (a < b) ? a : b;
                `ATOM_MAX_S: result = (signed_a > signed_b) ? a : b;
                `ATOM_MAX_U: result = (a > b) ? a : b;
                `ATOM_AND:   result = a & b;
                `ATOM_OR:    result = a | b;
                `ATOM_XOR:   result = a ^ b;
                default:     result = a;
            endcase
        end
    end

endmodule
