//============================================================================
// RalphGPU - Warp Scheduler
// 负责调度SM内的多个Warp执行
// 实现简单的轮询调度策略
//============================================================================

`include "gpu_defines.vh"

module warp_scheduler #(
    parameter NUM_WARPS    = `WARPS_PER_SM,    // 4
    parameter WARP_ID_W    = `WARP_ID_WIDTH    // 2
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Warp状态输入
    input  wire [NUM_WARPS-1:0]     warp_valid,     // Warp是否有效（已分配任务）
    input  wire [NUM_WARPS-1:0]     warp_ready,     // Warp是否准备就绪（无阻塞）
    input  wire [NUM_WARPS-1:0]     warp_waiting,   // Warp等待内存/同步

    // 调度输出
    output reg  [WARP_ID_W-1:0]     active_warp_id, // 当前执行的Warp ID
    output reg                      warp_selected,  // 是否有Warp被选中
    output reg  [NUM_WARPS-1:0]     warp_active_oh  // 独热码表示
);

    //------------------------------------------------------------------------
    // 调度状态
    //------------------------------------------------------------------------
    reg [WARP_ID_W-1:0] last_warp;  // 上次调度的Warp
    reg [WARP_ID_W-1:0] next_warp;  // 下一个要检查的Warp

    //------------------------------------------------------------------------
    // 轮询调度逻辑
    // 从上次调度的Warp开始，找下一个准备好的Warp
    //------------------------------------------------------------------------
    wire [NUM_WARPS-1:0] schedulable = warp_valid & warp_ready & ~warp_waiting;

    // 优先级编码器 - 找到第一个可调度的Warp
    reg [WARP_ID_W-1:0] found_warp;
    reg found_valid;

    integer i;
    always @(*) begin
        found_valid = 1'b0;
        found_warp = {WARP_ID_W{1'b0}};

        // 从next_warp开始搜索
        for (i = 0; i < NUM_WARPS; i = i + 1) begin
            if (!found_valid) begin
                // 计算实际检查的Warp索引 (轮询)
                // 简化：直接从0开始轮询
                if (schedulable[(next_warp + i) % NUM_WARPS]) begin
                    found_valid = 1'b1;
                    found_warp = (next_warp + i) % NUM_WARPS;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // 调度决策 (时序逻辑)
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_warp_id <= {WARP_ID_W{1'b0}};
            warp_selected  <= 1'b0;
            warp_active_oh <= {NUM_WARPS{1'b0}};
            last_warp      <= {WARP_ID_W{1'b0}};
            next_warp      <= {WARP_ID_W{1'b0}};
        end else begin
            if (found_valid) begin
                active_warp_id <= found_warp;
                warp_selected  <= 1'b1;
                warp_active_oh <= (1 << found_warp);
                last_warp      <= found_warp;
                next_warp      <= (found_warp + 1) % NUM_WARPS;
            end else begin
                warp_selected  <= 1'b0;
                warp_active_oh <= {NUM_WARPS{1'b0}};
            end
        end
    end

endmodule


//============================================================================
// Warp状态管理器
// 跟踪每个Warp的PC和执行状态
//============================================================================
module warp_state #(
    parameter NUM_WARPS = `WARPS_PER_SM,
    parameter WARP_ID_W = `WARP_ID_WIDTH,
    parameter PC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Warp分配
    input  wire                     alloc_en,
    input  wire [WARP_ID_W-1:0]     alloc_warp_id,
    input  wire [PC_WIDTH-1:0]      alloc_pc,       // 起始PC

    // Warp释放
    input  wire                     dealloc_en,
    input  wire [WARP_ID_W-1:0]     dealloc_warp_id,

    // PC更新
    input  wire                     pc_update_en,
    input  wire [WARP_ID_W-1:0]     pc_update_warp,
    input  wire [PC_WIDTH-1:0]      pc_update_value,
    input  wire                     pc_is_branch,

    // 同步控制
    input  wire                     sync_start,
    input  wire [WARP_ID_W-1:0]     sync_warp_id,
    input  wire                     sync_complete,

    // 状态输出
    output reg  [NUM_WARPS-1:0]     warp_valid,
    output reg  [NUM_WARPS-1:0]     warp_ready,
    output reg  [NUM_WARPS-1:0]     warp_waiting,
    output wire [PC_WIDTH-1:0]      warp_pc [0:NUM_WARPS-1]
);

    //------------------------------------------------------------------------
    // 每个Warp的状态
    //------------------------------------------------------------------------
    reg [PC_WIDTH-1:0] pc_regs [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] sync_pending;

    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : pc_out
            assign warp_pc[w] = pc_regs[w];
        end
    endgenerate

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_valid   <= {NUM_WARPS{1'b0}};
            warp_ready   <= {NUM_WARPS{1'b0}};
            warp_waiting <= {NUM_WARPS{1'b0}};
            sync_pending <= {NUM_WARPS{1'b0}};
            for (i = 0; i < NUM_WARPS; i = i + 1) begin
                pc_regs[i] <= {PC_WIDTH{1'b0}};
            end
        end else begin
            // 分配新Warp
            if (alloc_en) begin
                warp_valid[alloc_warp_id] <= 1'b1;
                warp_ready[alloc_warp_id] <= 1'b1;
                pc_regs[alloc_warp_id]    <= alloc_pc;
            end

            // 释放Warp
            if (dealloc_en) begin
                warp_valid[dealloc_warp_id]   <= 1'b0;
                warp_ready[dealloc_warp_id]   <= 1'b0;
                warp_waiting[dealloc_warp_id] <= 1'b0;
            end

            // 更新PC
            if (pc_update_en) begin
                pc_regs[pc_update_warp] <= pc_update_value;
            end

            // 同步开始
            if (sync_start) begin
                warp_waiting[sync_warp_id] <= 1'b1;
                sync_pending[sync_warp_id] <= 1'b1;
            end

            // 同步完成（所有Warp到达屏障）
            if (sync_complete) begin
                warp_waiting <= {NUM_WARPS{1'b0}};
                sync_pending <= {NUM_WARPS{1'b0}};
            end
        end
    end

    // 准备好 = 有效 且 不在等待
    always @(*) begin
        warp_ready = warp_valid & ~warp_waiting;
    end

endmodule
