//============================================================================
// RalphGPU - Memory Coalescing Unit
// 内存合并单元：将32线程的内存访问合并为最少的内存事务
// 这是NVIDIA GPU高性能的关键：减少内存带宽需求10-32倍
//============================================================================

`timescale 1ns / 1ps

module memory_coalescing_unit #(
    parameter THREADS         = 32,
    parameter DATA_WIDTH      = 32,
    parameter ADDR_WIDTH      = 32,
    parameter CACHE_LINE_SIZE = 128,    // 128 bytes per cache line
    parameter MAX_COALESCED   = 4       // 最多4个独立事务
)(
    input  wire                 clk,
    input  wire                 rst_n,

    //------------------------------------------------------------------------
    // 来自Warp的请求
    //------------------------------------------------------------------------
    input  wire                 req_valid,
    input  wire                 req_write,
    input  wire [ADDR_WIDTH-1:0] req_addr [0:THREADS-1],
    input  wire [DATA_WIDTH-1:0] req_wdata [0:THREADS-1],
    input  wire [THREADS-1:0]   req_mask,

    //------------------------------------------------------------------------
    // 到内存子系统的合并请求
    //------------------------------------------------------------------------
    output reg                  mem_req_valid,
    output reg                  mem_req_write,
    output reg  [ADDR_WIDTH-1:0] mem_req_addr,
    output reg  [CACHE_LINE_SIZE*8-1:0] mem_req_wdata,
    output reg  [CACHE_LINE_SIZE-1:0]   mem_req_wmask,  // 字节写掩码
    input  wire [CACHE_LINE_SIZE*8-1:0] mem_resp_rdata,
    input  wire                 mem_resp_valid,

    //------------------------------------------------------------------------
    // 返回给Warp的响应
    //------------------------------------------------------------------------
    output reg  [DATA_WIDTH-1:0] resp_rdata [0:THREADS-1],
    output reg                  resp_valid,
    output reg                  ready,

    //------------------------------------------------------------------------
    // 统计
    //------------------------------------------------------------------------
    output reg  [31:0]          stat_requests,      // 原始请求数
    output reg  [31:0]          stat_transactions,  // 实际内存事务数
    output reg  [31:0]          stat_coalesce_ratio // 合并比率 (x100)
);

    //------------------------------------------------------------------------
    // 地址分析
    //------------------------------------------------------------------------
    localparam OFFSET_BITS = $clog2(CACHE_LINE_SIZE);  // 7 bits for 128B

    // 计算每个线程的cache line地址 (使用锁存请求)
    wire [ADDR_WIDTH-OFFSET_BITS-1:0] line_addr [0:THREADS-1];
    wire [OFFSET_BITS-1:0]            line_offset [0:THREADS-1];

    generate
        genvar t;
        for (t = 0; t < THREADS; t = t + 1) begin : gen_addr
            assign line_addr[t]   = saved_addr[t][ADDR_WIDTH-1:OFFSET_BITS];
            assign line_offset[t] = saved_addr[t][OFFSET_BITS-1:0];
        end
    endgenerate

    //------------------------------------------------------------------------
    // 合并分析 - 找出不同的cache line
    //------------------------------------------------------------------------
    localparam LINE_IDX_W = (MAX_COALESCED > 1) ? $clog2(MAX_COALESCED) : 1;
    localparam UNIQUE_COUNT_W = $clog2(MAX_COALESCED + 1);

    reg [ADDR_WIDTH-OFFSET_BITS-1:0] unique_lines [0:MAX_COALESCED-1];
    reg [MAX_COALESCED-1:0]          line_valid;
    reg [UNIQUE_COUNT_W-1:0]         num_unique_lines;
    reg [LINE_IDX_W-1:0]             thread_to_line [0:THREADS-1];

    integer i, j;
    reg found;

    always @(*) begin
        // 初始化
        for (i = 0; i < MAX_COALESCED; i = i + 1) begin
            unique_lines[i] = 0;
            line_valid[i] = 0;
        end
        num_unique_lines = 0;

        for (i = 0; i < THREADS; i = i + 1) begin
            thread_to_line[i] = 0;
        end

        // 扫描所有活跃线程，找出不同的cache line
        for (i = 0; i < THREADS; i = i + 1) begin
            if (saved_mask[i]) begin
                found = 0;

                // 检查是否已经在列表中
                for (j = 0; j < MAX_COALESCED; j = j + 1) begin
                    if (line_valid[j] && unique_lines[j] == line_addr[i]) begin
                        found = 1;
                        thread_to_line[i] = j[LINE_IDX_W-1:0];
                    end
                end

                // 如果是新的cache line
                if (!found && num_unique_lines < MAX_COALESCED) begin
                    unique_lines[num_unique_lines] = line_addr[i];
                    line_valid[num_unique_lines] = 1;
                    thread_to_line[i] = num_unique_lines;
                    num_unique_lines = num_unique_lines + 1;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    localparam ST_IDLE      = 3'd0;
    localparam ST_ANALYZE   = 3'd1;
    localparam ST_REQUEST   = 3'd2;
    localparam ST_WAIT      = 3'd3;
    localparam ST_COLLECT   = 3'd4;
    localparam ST_DONE      = 3'd5;

    reg [2:0] state;
    reg [LINE_IDX_W-1:0] current_line;

    // 保存的请求
    reg                 saved_write;
    reg [ADDR_WIDTH-1:0] saved_addr [0:THREADS-1];
    reg [DATA_WIDTH-1:0] saved_wdata [0:THREADS-1];
    reg [THREADS-1:0]   saved_mask;

    // 响应缓存
    reg [CACHE_LINE_SIZE*8-1:0] line_data [0:MAX_COALESCED-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            current_line <= 0;
            mem_req_valid <= 0;
            mem_req_write <= 0;
            mem_req_addr <= 0;
            mem_req_wdata <= 0;
            mem_req_wmask <= 0;
            resp_valid <= 0;
            ready <= 1;
            stat_requests <= 0;
            stat_transactions <= 0;
            stat_coalesce_ratio <= 0;
            saved_write <= 0;
            saved_mask <= 0;

            for (i = 0; i < THREADS; i = i + 1) begin
                saved_addr[i] <= 0;
                saved_wdata[i] <= 0;
                resp_rdata[i] <= 0;
            end
            for (i = 0; i < MAX_COALESCED; i = i + 1) begin
                line_data[i] <= 0;
            end

        end else begin
            mem_req_valid <= 0;
            resp_valid <= 0;

            case (state)
                ST_IDLE: begin
                    ready <= 1;
                    if (req_valid) begin
                        // 保存请求
                        saved_write <= req_write;
                        saved_mask <= req_mask;
                        for (i = 0; i < THREADS; i = i + 1) begin
                            saved_addr[i] <= req_addr[i];
                            saved_wdata[i] <= req_wdata[i];
                        end
                        state <= ST_ANALYZE;
                        ready <= 0;
                        stat_requests <= stat_requests + 1;
                    end
                end

                ST_ANALYZE: begin
                    // 分析完成，开始发送请求
                    current_line <= 0;
                    state <= ST_REQUEST;
                end

                ST_REQUEST: begin
                    if (current_line < num_unique_lines) begin
                        // 发送当前cache line的请求
                        mem_req_valid <= 1;
                        mem_req_write <= saved_write;
                        mem_req_addr <= {unique_lines[current_line], {OFFSET_BITS{1'b0}}};
                        mem_req_wdata <= 0;
                        mem_req_wmask <= 0;

                        // 构建写数据和掩码
                        if (saved_write) begin
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i] && thread_to_line[i] == current_line) begin
                                    // 设置对应位置的数据和掩码
                                    mem_req_wdata[saved_addr[i][OFFSET_BITS-1:0]*8 +: DATA_WIDTH] <= saved_wdata[i];
                                    mem_req_wmask[saved_addr[i][OFFSET_BITS-1:0] +: 4] <= 4'hF;
                                end
                            end
                        end

                        state <= ST_WAIT;
                        stat_transactions <= stat_transactions + 1;
                    end else begin
                        state <= ST_COLLECT;
                    end
                end

                ST_WAIT: begin
                    if (mem_resp_valid) begin
                        // 保存响应数据
                        line_data[current_line] <= mem_resp_rdata;
                        current_line <= current_line + 1;
                        state <= ST_REQUEST;
                    end
                end

                ST_COLLECT: begin
                    // 分发数据到各个线程
                    for (i = 0; i < THREADS; i = i + 1) begin
                        if (saved_mask[i]) begin
                            resp_rdata[i] <= line_data[thread_to_line[i]][saved_addr[i][OFFSET_BITS-1:0]*8 +: DATA_WIDTH];
                        end else begin
                            resp_rdata[i] <= 0;
                        end
                    end

                    // 更新合并比率
                    if (stat_transactions > 0) begin
                        stat_coalesce_ratio <= (stat_requests * 100) / stat_transactions;
                    end

                    resp_valid <= 1;
                    state <= ST_DONE;
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end
                default: ; // lint: CASEINCOMPLETE
            endcase
        end
    end

endmodule


//============================================================================
// Warp Level Memory Access Unit
// 支持32线程的并行内存访问，集成合并逻辑
//============================================================================
module warp_memory_unit #(
    parameter THREADS = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // Warp请求
    input  wire                 req_valid,
    input  wire                 req_write,
    input  wire [31:0]          req_addr [0:THREADS-1],
    input  wire [31:0]          req_wdata [0:THREADS-1],
    input  wire [THREADS-1:0]   req_mask,

    // L1 Cache接口
    output reg                  l1_req_valid,
    output reg                  l1_req_write,
    output reg  [31:0]          l1_req_addr,
    output reg  [1023:0]        l1_req_wdata,
    output reg  [127:0]         l1_req_wmask,
    input  wire [1023:0]        l1_resp_rdata,
    input  wire                 l1_resp_valid,

    // Warp响应
    output reg  [31:0]          resp_rdata [0:THREADS-1],
    output reg                  resp_valid,
    output reg                  ready
);

    // 检测完美合并 (所有线程访问连续地址)
    wire perfect_coalesce;
    wire [31:0] base_addr;

    assign base_addr = req_addr[0] & ~32'h7F;  // 128B对齐

    // 检查所有活跃线程是否在同一cache line
    reg all_same_line;
    integer i;

    always @(*) begin
        all_same_line = 1;
        for (i = 0; i < THREADS; i = i + 1) begin
            if (req_mask[i]) begin
                if ((req_addr[i] & ~32'h7F) != base_addr) begin
                    all_same_line = 0;
                end
            end
        end
    end

    assign perfect_coalesce = all_same_line;

    // 简化状态机 (假设完美合并)
    localparam ST_IDLE = 2'd0;
    localparam ST_REQ  = 2'd1;
    localparam ST_WAIT = 2'd2;
    localparam ST_DONE = 2'd3;

    reg [1:0] state;
    reg [31:0] saved_addr [0:THREADS-1];
    reg [THREADS-1:0] saved_mask;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            l1_req_valid <= 0;
            resp_valid <= 0;
            ready <= 1;
            for (i = 0; i < THREADS; i = i + 1) begin
                resp_rdata[i] <= 0;
                saved_addr[i] <= 0;
            end
            saved_mask <= 0;
        end else begin
            l1_req_valid <= 0;
            resp_valid <= 0;

            case (state)
                ST_IDLE: begin
                    ready <= 1;
                    if (req_valid) begin
                        for (i = 0; i < THREADS; i = i + 1) begin
                            saved_addr[i] <= req_addr[i];
                        end
                        saved_mask <= req_mask;
                        state <= ST_REQ;
                        ready <= 0;
                    end
                end

                ST_REQ: begin
                    l1_req_valid <= 1;
                    l1_req_addr <= base_addr;
                    state <= ST_WAIT;
                end

                ST_WAIT: begin
                    if (l1_resp_valid) begin
                        // 分发数据
                        for (i = 0; i < THREADS; i = i + 1) begin
                            if (saved_mask[i]) begin
                                resp_rdata[i] <= l1_resp_rdata[(saved_addr[i][6:2]) * 32 +: 32];
                            end
                        end
                        resp_valid <= 1;
                        state <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
