//============================================================================
// RalphGPU - Dual Issue Warp Scheduler
// 双发射调度器：每周期发射2条独立指令
// NVIDIA SM可以每周期发射多条指令到不同的执行单元
//============================================================================

`timescale 1ns / 1ps

module dual_issue_scheduler #(
    parameter NUM_WARPS    = 4,
    parameter THREADS      = 32,
    parameter INST_WIDTH   = 32,
    parameter DATA_WIDTH   = 32,
    parameter WARP_ID_W    = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1
)(
    input  wire                 clk,
    input  wire                 rst_n,

    //------------------------------------------------------------------------
    // 指令输入 (来自每个warp的指令缓冲)
    //------------------------------------------------------------------------
    input  wire [INST_WIDTH-1:0] warp_inst [0:NUM_WARPS-1],
    input  wire [NUM_WARPS-1:0]  warp_valid,
    input  wire [NUM_WARPS-1:0]  warp_ready,  // warp可以执行

    //------------------------------------------------------------------------
    // 依赖检查输入
    //------------------------------------------------------------------------
    input  wire [4:0]           warp_rd [0:NUM_WARPS-1],     // 目标寄存器
    input  wire [4:0]           warp_rs1 [0:NUM_WARPS-1],    // 源寄存器1
    input  wire [4:0]           warp_rs2 [0:NUM_WARPS-1],    // 源寄存器2
    input  wire [NUM_WARPS-1:0] warp_writes_reg,             // 写寄存器
    input  wire [NUM_WARPS-1:0] warp_reads_mem,              // 读内存
    input  wire [NUM_WARPS-1:0] warp_writes_mem,             // 写内存

    //------------------------------------------------------------------------
    // 执行单元可用性
    //------------------------------------------------------------------------
    input  wire                 alu_ready,
    input  wire                 fma_ready,
    input  wire                 mem_ready,
    input  wire                 branch_ready,

    //------------------------------------------------------------------------
    // 双发射输出 (Slot 0 和 Slot 1)
    //------------------------------------------------------------------------
    output reg                  issue0_valid,
    output reg  [WARP_ID_W-1:0] issue0_warp_id,
    output reg  [INST_WIDTH-1:0] issue0_inst,
    output reg  [2:0]           issue0_unit,     // 0=ALU, 1=FMA, 2=MEM, 3=BRANCH

    output reg                  issue1_valid,
    output reg  [WARP_ID_W-1:0] issue1_warp_id,
    output reg  [INST_WIDTH-1:0] issue1_inst,
    output reg  [2:0]           issue1_unit,

    //------------------------------------------------------------------------
    // Warp消费确认
    //------------------------------------------------------------------------
    output reg  [NUM_WARPS-1:0] warp_consumed,

    //------------------------------------------------------------------------
    // 统计
    //------------------------------------------------------------------------
    output reg  [31:0]          stat_single_issue,
    output reg  [31:0]          stat_dual_issue,
    output reg  [31:0]          stat_stall_cycles
);

    //------------------------------------------------------------------------
    // 操作码解码 (简化)
    //------------------------------------------------------------------------
    localparam OP_ALU    = 3'd0;
    localparam OP_FMA    = 3'd1;
    localparam OP_LOAD   = 3'd2;
    localparam OP_STORE  = 3'd3;
    localparam OP_BRANCH = 3'd4;

    // 从指令解码执行单元
    function [2:0] decode_unit;
        input [INST_WIDTH-1:0] inst;
        reg [6:0] opcode;
        begin
            opcode = inst[6:0];
            case (opcode)
                7'b0110011: decode_unit = OP_ALU;    // R-type ALU
                7'b0010011: decode_unit = OP_ALU;    // I-type ALU
                7'b0000011: decode_unit = OP_LOAD;   // Load
                7'b0100011: decode_unit = OP_STORE;  // Store
                7'b1100011: decode_unit = OP_BRANCH; // Branch
                7'b1101111: decode_unit = OP_BRANCH; // JAL
                7'b1100111: decode_unit = OP_BRANCH; // JALR
                7'b0110111: decode_unit = OP_ALU;    // LUI
                7'b0010111: decode_unit = OP_ALU;    // AUIPC
                default:    decode_unit = OP_ALU;
            endcase
            // FMA特殊检测 (funct7)
            if (opcode == 7'b0110011 && inst[31:25] == 7'b0000001) begin
                decode_unit = OP_FMA;  // MUL/DIV -> FMA单元
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // 依赖检查
    //------------------------------------------------------------------------
    function check_dependency;
        input [WARP_ID_W-1:0] warp_a;
        input [WARP_ID_W-1:0] warp_b;
        begin
            // RAW: warp_a写的寄存器被warp_b读
            // WAW: 两个warp写同一寄存器
            // WAR: warp_a读的寄存器被warp_b写

            check_dependency = 0;

            // RAW检查
            if (warp_writes_reg[warp_a]) begin
                if (warp_rd[warp_a] == warp_rs1[warp_b] ||
                    warp_rd[warp_a] == warp_rs2[warp_b]) begin
                    check_dependency = 1;
                end
            end

            // WAW检查
            if (warp_writes_reg[warp_a] && warp_writes_reg[warp_b]) begin
                if (warp_rd[warp_a] == warp_rd[warp_b]) begin
                    check_dependency = 1;
                end
            end

            // 内存依赖 (保守: 任何内存操作都冲突)
            if ((warp_reads_mem[warp_a] || warp_writes_mem[warp_a]) &&
                (warp_reads_mem[warp_b] || warp_writes_mem[warp_b])) begin
                check_dependency = 1;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // 执行单元冲突检查
    //------------------------------------------------------------------------
    function check_unit_conflict;
        input [2:0] unit_a;
        input [2:0] unit_b;
        begin
            // 相同执行单元冲突
            check_unit_conflict = (unit_a == unit_b);
        end
    endfunction

    //------------------------------------------------------------------------
    // 调度逻辑
    //------------------------------------------------------------------------
    reg [WARP_ID_W-1:0] selected_warp0;
    reg [WARP_ID_W-1:0] selected_warp1;
    reg       found_warp0;
    reg       found_warp1;
    reg [2:0] unit0;
    reg [2:0] unit1;

    integer i, j;

    always @(*) begin
        selected_warp0 = 0;
        selected_warp1 = 0;
        found_warp0 = 0;
        found_warp1 = 0;
        unit0 = 0;
        unit1 = 0;

        // 第一遍: 找第一个可调度的warp
        for (i = 0; i < NUM_WARPS; i = i + 1) begin
            if (warp_valid[i] && warp_ready[i] && !found_warp0) begin
                unit0 = decode_unit(warp_inst[i]);

                // 检查执行单元是否可用
                case (unit0)
                    OP_ALU:    if (alu_ready)    begin found_warp0 = 1; selected_warp0 = i[WARP_ID_W-1:0]; end
                    OP_FMA:    if (fma_ready)    begin found_warp0 = 1; selected_warp0 = i[WARP_ID_W-1:0]; end
                    OP_LOAD:   if (mem_ready)    begin found_warp0 = 1; selected_warp0 = i[WARP_ID_W-1:0]; end
                    OP_STORE:  if (mem_ready)    begin found_warp0 = 1; selected_warp0 = i[WARP_ID_W-1:0]; end
                    OP_BRANCH: if (branch_ready) begin found_warp0 = 1; selected_warp0 = i[WARP_ID_W-1:0]; end
                    default: ; // lint: CASEINCOMPLETE
                endcase
            end
        end

        // 第二遍: 找第二个可调度的warp (双发射)
        if (found_warp0) begin
            for (j = 0; j < NUM_WARPS; j = j + 1) begin
                if (warp_valid[j] && warp_ready[j] && !found_warp1 && j[WARP_ID_W-1:0] != selected_warp0) begin
                    unit1 = decode_unit(warp_inst[j]);

                    // 检查执行单元冲突
                    if (!check_unit_conflict(unit0, unit1)) begin
                        // 检查依赖
                        if (!check_dependency(selected_warp0, j[WARP_ID_W-1:0]) &&
                            !check_dependency(j[WARP_ID_W-1:0], selected_warp0)) begin

                            // 检查执行单元可用
                            case (unit1)
                                OP_ALU:    if (alu_ready)    begin found_warp1 = 1; selected_warp1 = j[WARP_ID_W-1:0]; end
                                OP_FMA:    if (fma_ready)    begin found_warp1 = 1; selected_warp1 = j[WARP_ID_W-1:0]; end
                                OP_LOAD:   if (mem_ready)    begin found_warp1 = 1; selected_warp1 = j[WARP_ID_W-1:0]; end
                                OP_STORE:  if (mem_ready)    begin found_warp1 = 1; selected_warp1 = j[WARP_ID_W-1:0]; end
                                OP_BRANCH: if (branch_ready) begin found_warp1 = 1; selected_warp1 = j[WARP_ID_W-1:0]; end
                                default: ; // lint: CASEINCOMPLETE
                            endcase
                        end
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // 输出寄存
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue0_valid   <= 0;
            issue0_warp_id <= 0;
            issue0_inst    <= 0;
            issue0_unit    <= 0;

            issue1_valid   <= 0;
            issue1_warp_id <= 0;
            issue1_inst    <= 0;
            issue1_unit    <= 0;

            warp_consumed  <= 0;

            stat_single_issue <= 0;
            stat_dual_issue   <= 0;
            stat_stall_cycles <= 0;

        end else begin
            warp_consumed <= 0;

            // Slot 0
            if (found_warp0) begin
                issue0_valid   <= 1;
                issue0_warp_id <= selected_warp0;
                issue0_inst    <= warp_inst[selected_warp0];
                issue0_unit    <= unit0;
                warp_consumed[selected_warp0] <= 1;
            end else begin
                issue0_valid <= 0;
                stat_stall_cycles <= stat_stall_cycles + 1;
            end

            // Slot 1 (双发射)
            if (found_warp1) begin
                issue1_valid   <= 1;
                issue1_warp_id <= selected_warp1;
                issue1_inst    <= warp_inst[selected_warp1];
                issue1_unit    <= unit1;
                warp_consumed[selected_warp1] <= 1;
            end else begin
                issue1_valid <= 0;
            end

            // 统计
            if (found_warp0 && found_warp1) begin
                stat_dual_issue <= stat_dual_issue + 1;
            end else if (found_warp0) begin
                stat_single_issue <= stat_single_issue + 1;
            end
        end
    end

endmodule


//============================================================================
// Instruction Level Parallelism Analyzer
// 分析指令流中的ILP机会
//============================================================================
module ilp_analyzer #(
    parameter WINDOW_SIZE = 8,
    parameter INST_WIDTH  = 32
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // 指令窗口输入
    input  wire [INST_WIDTH-1:0] inst_window [0:WINDOW_SIZE-1],
    input  wire [WINDOW_SIZE-1:0] inst_valid,

    // ILP分析输出
    output reg  [2:0]           available_ilp,      // 可并行指令数
    output reg  [WINDOW_SIZE-1:0] independent_mask, // 独立指令掩码
    output reg  [31:0]          stat_avg_ilp        // 平均ILP (x100)
);

    // 简化依赖图
    reg [WINDOW_SIZE-1:0] depends_on [0:WINDOW_SIZE-1];

    // 解码寄存器
    wire [4:0] rd  [0:WINDOW_SIZE-1];
    wire [4:0] rs1 [0:WINDOW_SIZE-1];
    wire [4:0] rs2 [0:WINDOW_SIZE-1];

    generate
        genvar g;
        for (g = 0; g < WINDOW_SIZE; g = g + 1) begin : gen_decode
            assign rd[g]  = inst_window[g][11:7];
            assign rs1[g] = inst_window[g][19:15];
            assign rs2[g] = inst_window[g][24:20];
        end
    endgenerate

    // 依赖分析
    integer i, j;
    reg [31:0] total_samples;
    reg [31:0] total_ilp;

    always @(*) begin
        // 初始化依赖图
        for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
            depends_on[i] = 0;
        end

        // 构建依赖图
        for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
            if (inst_valid[i]) begin
                for (j = 0; j < i; j = j + 1) begin
                    if (inst_valid[j]) begin
                        // RAW依赖
                        if (rd[j] != 0 && (rd[j] == rs1[i] || rd[j] == rs2[i])) begin
                            depends_on[i][j] = 1;
                        end
                        // WAW依赖
                        if (rd[j] != 0 && rd[i] != 0 && rd[j] == rd[i]) begin
                            depends_on[i][j] = 1;
                        end
                    end
                end
            end
        end

        // 找独立指令 (没有前向依赖的)
        independent_mask = 0;
        available_ilp = 0;

        for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
            if (inst_valid[i] && depends_on[i] == 0) begin
                independent_mask[i] = 1;
                if (available_ilp < 7) begin
                    available_ilp = available_ilp + 1;
                end
            end
        end
    end

    // ILP统计 (时序)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_samples <= 0;
            total_ilp <= 0;
            stat_avg_ilp <= 0;
        end else begin
            if (|inst_valid) begin
                total_samples <= total_samples + 1;
                total_ilp <= total_ilp + available_ilp;
                if (total_samples > 0) begin
                    stat_avg_ilp <= (total_ilp * 100) / total_samples;
                end
            end
        end
    end

endmodule
