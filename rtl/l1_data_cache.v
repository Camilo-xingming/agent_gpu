//============================================================================
// RalphGPU - L1 Data Cache
// 直接映射L1数据缓存，支持32线程SIMT并行访问
// 设计目标: 将全局内存延迟从100 cycles降低到4 cycles (hit)
//============================================================================

`timescale 1ns / 1ps

module l1_data_cache #(
    parameter CACHE_SIZE_KB   = 16,        // 16KB cache
    parameter LINE_SIZE_BYTES = 128,       // 128 bytes per line (32 words)
    parameter NUM_WAYS        = 4,         // 4-way set associative
    parameter HIT_LATENCY     = 4,         // 4 cycles for hit
    parameter THREADS         = 32,        // SIMT threads per warp
    parameter DATA_WIDTH      = 32         // 32-bit data
)(
    input  wire                 clk,
    input  wire                 rst_n,

    //------------------------------------------------------------------------
    // Warp请求接口 (来自SM)
    //------------------------------------------------------------------------
    input  wire                 req_valid,
    input  wire                 req_write,          // 0=read, 1=write
    input  wire [31:0]          req_addr [0:THREADS-1],   // 32个地址
    input  wire [31:0]          req_wdata [0:THREADS-1],  // 32个写数据
    input  wire [THREADS-1:0]   req_mask,           // 活跃线程掩码
    output reg  [31:0]          resp_rdata [0:THREADS-1], // 32个读数据
    output reg                  resp_valid,
    output reg                  resp_hit,           // 全部命中

    //------------------------------------------------------------------------
    // 到全局内存接口 (AXI-like)
    //------------------------------------------------------------------------
    output reg                  mem_req,
    output reg                  mem_write,
    output reg  [31:0]          mem_addr,
    output reg  [LINE_SIZE_BYTES*8-1:0] mem_wdata,  // 整行写
    input  wire [LINE_SIZE_BYTES*8-1:0] mem_rdata,  // 整行读
    input  wire                 mem_valid,
    input  wire                 mem_ready,

    //------------------------------------------------------------------------
    // 统计接口
    //------------------------------------------------------------------------
    output reg  [31:0]          stat_hits,
    output reg  [31:0]          stat_misses
);

    //------------------------------------------------------------------------
    // 参数计算
    //------------------------------------------------------------------------
    localparam CACHE_SIZE_BYTES = CACHE_SIZE_KB * 1024;
    localparam NUM_LINES        = CACHE_SIZE_BYTES / LINE_SIZE_BYTES;
    localparam NUM_SETS         = NUM_LINES / NUM_WAYS;
    localparam WORDS_PER_LINE   = LINE_SIZE_BYTES / 4;

    // 地址字段
    localparam OFFSET_BITS      = $clog2(LINE_SIZE_BYTES);  // 7 bits for 128B
    localparam INDEX_BITS       = $clog2(NUM_SETS);         // depends on size
    localparam TAG_BITS         = 32 - OFFSET_BITS - INDEX_BITS;

    //------------------------------------------------------------------------
    // Cache存储结构
    //------------------------------------------------------------------------
    // Tag数组: [way][set]
    reg [TAG_BITS-1:0]   tag_array   [0:NUM_WAYS-1][0:NUM_SETS-1];
    reg                  valid_array [0:NUM_WAYS-1][0:NUM_SETS-1];
    reg                  dirty_array [0:NUM_WAYS-1][0:NUM_SETS-1];

    // 数据数组: [way][set][word]
    reg [31:0]           data_array  [0:NUM_WAYS-1][0:NUM_SETS-1][0:WORDS_PER_LINE-1];

    // LRU状态 (简化: 2-bit per set for 4-way)
    reg [1:0]            lru_array   [0:NUM_SETS-1];

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    localparam ST_IDLE       = 3'd0;
    localparam ST_TAG_CHECK  = 3'd1;
    localparam ST_HIT        = 3'd2;
    localparam ST_MISS       = 3'd3;
    localparam ST_WRITEBACK  = 3'd4;
    localparam ST_FILL       = 3'd5;
    localparam ST_FILL_WAIT  = 3'd6;
    localparam ST_DONE       = 3'd7;

    reg [2:0] state, next_state;

    //------------------------------------------------------------------------
    // 请求寄存
    //------------------------------------------------------------------------
    reg                 saved_write;
    reg [31:0]          saved_addr [0:THREADS-1];
    reg [31:0]          saved_wdata [0:THREADS-1];
    reg [THREADS-1:0]   saved_mask;

    // 主地址 (用第一个活跃线程的地址作为代表)
    wire [31:0]         primary_addr;
    wire [TAG_BITS-1:0] primary_tag;
    wire [INDEX_BITS-1:0] primary_index;
    wire [OFFSET_BITS-1:0] primary_offset;

    // 找第一个活跃线程
    integer first_active;
    reg found;
    always @(*) begin
        first_active = 0;
        found = 1'b0;
        for (integer i = 0; i < THREADS; i = i + 1) begin
            if (!found && saved_mask[i]) begin
                first_active = i;
                found = 1'b1;
            end
        end
    end

    assign primary_addr   = saved_addr[first_active];
    assign primary_tag    = primary_addr[31:32-TAG_BITS];
    assign primary_index  = primary_addr[OFFSET_BITS +: INDEX_BITS];
    assign primary_offset = primary_addr[OFFSET_BITS-1:0];

    //------------------------------------------------------------------------
    // Tag比较和命中检测
    //------------------------------------------------------------------------
    reg [NUM_WAYS-1:0] way_hit;
    reg [1:0]          hit_way;
    reg                cache_hit;

    always @(*) begin
        way_hit = 0;
        hit_way = 0;
        cache_hit = 0;

        for (integer w = 0; w < NUM_WAYS; w = w + 1) begin
            if (valid_array[w][primary_index] &&
                tag_array[w][primary_index] == primary_tag) begin
                way_hit[w] = 1;
                hit_way = w[1:0];
                cache_hit = 1;
            end
        end
    end

    //------------------------------------------------------------------------
    // LRU替换选择
    //------------------------------------------------------------------------
    reg [1:0] replace_way;

    always @(*) begin
        // 简单LRU: 选择最近最少使用的way
        replace_way = lru_array[primary_index];

        // 如果有无效行，优先使用
        for (integer w = 0; w < NUM_WAYS; w = w + 1) begin
            if (!valid_array[w][primary_index]) begin
                replace_way = w[1:0];
            end
        end
    end

    //------------------------------------------------------------------------
    // 延迟计数器 (模拟hit延迟)
    //------------------------------------------------------------------------
    reg [2:0] latency_counter;

    //------------------------------------------------------------------------
    // 状态机逻辑
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        next_state = state;

        case (state)
            ST_IDLE: begin
                if (req_valid) begin
                    next_state = ST_TAG_CHECK;
                end
            end

            ST_TAG_CHECK: begin
                if (cache_hit) begin
                    next_state = ST_HIT;
                end else begin
                    // Miss: 检查是否需要写回
                    if (dirty_array[replace_way][primary_index] &&
                        valid_array[replace_way][primary_index]) begin
                        next_state = ST_WRITEBACK;
                    end else begin
                        next_state = ST_FILL;
                    end
                end
            end

            ST_HIT: begin
                if (latency_counter == 0) begin
                    next_state = ST_DONE;
                end
            end

            ST_WRITEBACK: begin
                if (mem_valid) begin
                    next_state = ST_FILL;
                end
            end

            ST_FILL: begin
                if (mem_ready) begin
                    next_state = ST_FILL_WAIT;
                end
            end

            ST_FILL_WAIT: begin
                if (mem_valid) begin
                    next_state = ST_DONE;
                end
            end

            ST_DONE: begin
                next_state = ST_IDLE;
            end

            default: next_state = ST_IDLE;
        endcase
    end

    //------------------------------------------------------------------------
    // 数据路径逻辑
    //------------------------------------------------------------------------
    integer i, w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid <= 0;
            resp_hit   <= 0;
            mem_req    <= 0;
            mem_write  <= 0;
            mem_addr   <= 0;
            mem_wdata  <= 0;
            latency_counter <= 0;
            stat_hits  <= 0;
            stat_misses <= 0;

            // 初始化cache
            for (w = 0; w < NUM_WAYS; w = w + 1) begin
                for (i = 0; i < NUM_SETS; i = i + 1) begin
                    valid_array[w][i] <= 0;
                    dirty_array[w][i] <= 0;
                    tag_array[w][i]   <= 0;
                end
            end

            for (i = 0; i < NUM_SETS; i = i + 1) begin
                lru_array[i] <= 0;
            end

            for (i = 0; i < THREADS; i = i + 1) begin
                resp_rdata[i] <= 0;
                saved_addr[i] <= 0;
                saved_wdata[i] <= 0;
            end
            saved_mask <= 0;
            saved_write <= 0;

        end else begin
            // 默认值
            resp_valid <= 0;
            mem_req    <= 0;

            case (state)
                ST_IDLE: begin
                    if (req_valid) begin
                        // 保存请求
                        saved_write <= req_write;
                        saved_mask  <= req_mask;
                        for (i = 0; i < THREADS; i = i + 1) begin
                            saved_addr[i]  <= req_addr[i];
                            saved_wdata[i] <= req_wdata[i];
                        end
                        latency_counter <= HIT_LATENCY - 1;
                    end
                end

                ST_TAG_CHECK: begin
                    // Tag检查在组合逻辑中完成
                end

                ST_HIT: begin
                    if (latency_counter > 0) begin
                        latency_counter <= latency_counter - 1;
                    end

                    if (latency_counter == 0) begin
                        // Hit完成
                        stat_hits <= stat_hits + 1;
                        resp_hit  <= 1;
                        resp_valid <= 1;

                        if (saved_write) begin
                            // 写操作
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i]) begin
                                    data_array[hit_way][primary_index][saved_addr[i][OFFSET_BITS-1:2]] <= saved_wdata[i];
                                end
                            end
                            dirty_array[hit_way][primary_index] <= 1;
                        end else begin
                            // 读操作
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i]) begin
                                    resp_rdata[i] <= data_array[hit_way][primary_index][saved_addr[i][OFFSET_BITS-1:2]];
                                end
                            end
                        end

                        // 更新LRU
                        lru_array[primary_index] <= (hit_way == 0) ? 2'd1 :
                                                    (hit_way == 1) ? 2'd2 :
                                                    (hit_way == 2) ? 2'd3 : 2'd0;
                    end
                end

                ST_WRITEBACK: begin
                    // 写回脏行
                    mem_req   <= 1;
                    mem_write <= 1;
                    mem_addr  <= {tag_array[replace_way][primary_index], primary_index, {OFFSET_BITS{1'b0}}};

                    // 打包整行数据
                    for (i = 0; i < WORDS_PER_LINE; i = i + 1) begin
                        mem_wdata[i*32 +: 32] <= data_array[replace_way][primary_index][i];
                    end
                end

                ST_FILL: begin
                    // 请求新行
                    mem_req   <= 1;
                    mem_write <= 0;
                    mem_addr  <= {primary_tag, primary_index, {OFFSET_BITS{1'b0}}};
                end

                ST_FILL_WAIT: begin
                    if (mem_valid) begin
                        stat_misses <= stat_misses + 1;
                        resp_hit    <= 0;
                        resp_valid  <= 1;

                        // 填充cache行
                        tag_array[replace_way][primary_index]   <= primary_tag;
                        valid_array[replace_way][primary_index] <= 1;
                        dirty_array[replace_way][primary_index] <= saved_write;

                        for (i = 0; i < WORDS_PER_LINE; i = i + 1) begin
                            data_array[replace_way][primary_index][i] <= mem_rdata[i*32 +: 32];
                        end

                        // 处理原始请求
                        if (saved_write) begin
                            // 写入新数据
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i]) begin
                                    data_array[replace_way][primary_index][saved_addr[i][OFFSET_BITS-1:2]] <= saved_wdata[i];
                                end
                            end
                        end else begin
                            // 返回读数据
                            for (i = 0; i < THREADS; i = i + 1) begin
                                if (saved_mask[i]) begin
                                    resp_rdata[i] <= mem_rdata[saved_addr[i][OFFSET_BITS-1:2]*32 +: 32];
                                end
                            end
                        end

                        // 更新LRU
                        lru_array[primary_index] <= (replace_way == 0) ? 2'd1 :
                                                    (replace_way == 1) ? 2'd2 :
                                                    (replace_way == 2) ? 2'd3 : 2'd0;
                    end
                end

                ST_DONE: begin
                    // 完成状态，返回IDLE
                end
            endcase
        end
    end

endmodule
