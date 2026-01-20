//============================================================================
// RalphGPU - Async Copy Engine
// 支持 cp.async 和 st.async 指令用于异步内存拷贝
// PTX Instructions: cp.async.ca, cp.async.cg, cp.async.commit_group,
//                   cp.async.wait_group, cp.async.wait_all, cp.async.bulk,
//                   st.async.global, st.async.shared
//============================================================================

`include "gpu_defines.vh"

module async_copy_engine #(
    parameter MAX_GROUPS = 8,           // 最大并发拷贝组
    parameter MAX_PENDING = 16,         // 每组最大挂起拷贝数
    parameter SHARED_MEM_ADDR_W = 14,   // 共享内存地址宽度
    parameter GLOBAL_ADDR_W = 32        // 全局内存地址宽度
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // 控制接口
    input  wire [5:0]                   opcode,         // 操作码 (OP_CPASYNC or OP_ST_ASYNC)
    input  wire [5:0]                   func,           // 功能码
    input  wire                         valid_in,
    input  wire [31:0]                  src_addr,       // 源地址 (全局内存 for load, 共享内存 for store)
    input  wire [SHARED_MEM_ADDR_W-1:0] dst_addr,       // 目标地址 (共享内存 for load, global for store[13:0])
    input  wire [3:0]                   size,           // 拷贝大小: 4, 8, 16 bytes
    input  wire [2:0]                   cache_hint,     // 缓存提示
    input  wire [3:0]                   wait_count,     // wait_group的等待数量

    // TMA Interface (for cp.async.bulk.tensor)
    input  wire [63:0]                  tensor_desc,    // Tensor descriptor
    input  wire [31:0]                  tensor_coord_x, // X coordinate (byte offset)
    input  wire [31:0]                  tensor_coord_y, // Y coordinate (row offset)

    // st.async Interface (for st.async.global/st.async.shared)
    input  wire                         is_store,       // 1=store operation, 0=load operation
    input  wire [31:0]                  store_gmem_addr, // Global memory address for store
    input  wire [127:0]                 store_data,     // Data to store

    // 状态输出
    output reg                          ready,
    output reg                          done,
    output reg  [3:0]                   pending_count,  // 当前组挂起数量
    output wire                         tma_busy,       // TMA operation in progress

    // 全局内存读接口
    output reg                          gmem_req_valid,
    output reg  [GLOBAL_ADDR_W-1:0]     gmem_req_addr,
    output reg  [4:0]                   gmem_req_size,  // bytes
    output reg  [2:0]                   gmem_req_cache,
    input  wire                         gmem_resp_valid,
    input  wire [127:0]                 gmem_resp_data,

    // 全局内存写接口 (for st.async.global)
    output reg                          gmem_wr_valid,
    output reg  [GLOBAL_ADDR_W-1:0]     gmem_wr_addr,
    output reg  [127:0]                 gmem_wr_data,
    output reg  [4:0]                   gmem_wr_size,
    input  wire                         gmem_wr_done,

    // 共享内存写接口 (for cp.async loads)
    output reg                          smem_wr_en,
    output reg  [SHARED_MEM_ADDR_W-1:0] smem_wr_addr,
    output reg  [127:0]                 smem_wr_data,
    output reg  [4:0]                   smem_wr_size,  // 5 bits to hold values up to 16

    // 共享内存读接口 (for st.async.global)
    output reg                          smem_rd_en,
    output reg  [SHARED_MEM_ADDR_W-1:0] smem_rd_addr,
    input  wire [127:0]                 smem_rd_data,
    input  wire                         smem_rd_valid
);

    //------------------------------------------------------------------------
    // 拷贝请求队列
    //------------------------------------------------------------------------
    localparam QUEUE_DEPTH = MAX_GROUPS * MAX_PENDING;
    localparam QUEUE_ADDR_W = $clog2(QUEUE_DEPTH);

    reg [GLOBAL_ADDR_W-1:0]     req_src_addr  [0:QUEUE_DEPTH-1];  // For load: gmem addr; For store: smem addr
    reg [SHARED_MEM_ADDR_W-1:0] req_dst_addr  [0:QUEUE_DEPTH-1];  // For load: smem addr; For store: gmem addr[13:0]
    reg [GLOBAL_ADDR_W-1:0]     req_gmem_addr [0:QUEUE_DEPTH-1];  // For store: full gmem addr
    reg [127:0]                 req_data      [0:QUEUE_DEPTH-1];  // For store: data to write
    reg [3:0]                   req_size      [0:QUEUE_DEPTH-1];
    reg [2:0]                   req_cache     [0:QUEUE_DEPTH-1];
    reg [2:0]                   req_group     [0:QUEUE_DEPTH-1];
    reg [QUEUE_DEPTH-1:0]       req_valid;
    reg [QUEUE_DEPTH-1:0]       req_complete;
    reg [QUEUE_DEPTH-1:0]       req_is_store;  // 1=store, 0=load

    reg [QUEUE_ADDR_W-1:0]      req_head;       // 下一个写入位置
    reg [QUEUE_ADDR_W-1:0]      req_tail;       // 下一个处理位置
    reg [QUEUE_ADDR_W-1:0]      req_count;

    //------------------------------------------------------------------------
    // 组管理
    //------------------------------------------------------------------------
    reg [2:0]                   current_group;  // 当前活动组
    reg [4:0]                   group_pending [0:MAX_GROUPS-1];  // 每组挂起数
    reg [MAX_GROUPS-1:0]        group_committed;

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    localparam ST_IDLE       = 4'd0;
    localparam ST_ISSUE      = 4'd1;
    localparam ST_WAIT_RESP  = 4'd2;
    localparam ST_WRITE_SM   = 4'd3;
    localparam ST_COMMIT     = 4'd4;
    localparam ST_WAIT_GRP   = 4'd5;
    // New states for st.async
    localparam ST_READ_SM    = 4'd6;  // Read from shared memory for store
    localparam ST_WRITE_GM   = 4'd7;  // Write to global memory
    localparam ST_WAIT_WR    = 4'd8;  // Wait for global memory write completion

    reg [3:0] state;
    reg [QUEUE_ADDR_W-1:0] current_req;
    reg [127:0] data_buffer;
    reg [3:0] bytes_remaining;

    //------------------------------------------------------------------------
    // TMA Unit Integration
    //------------------------------------------------------------------------
    reg                         tma_start;
    wire                        tma_req_valid;
    wire [GLOBAL_ADDR_W-1:0]    tma_req_src_addr;
    wire [SHARED_MEM_ADDR_W-1:0] tma_req_dst_addr;
    wire [4:0]                  tma_req_size;
    wire                        tma_done;
    wire                        tma_busy_int;
    wire [15:0]                 tma_bytes_copied;

    // TMA request acceptance - accept when queue not full and not processing
    wire tma_req_ready = (req_count < QUEUE_DEPTH) && (state == ST_IDLE || state == ST_ISSUE);

    assign tma_busy = tma_busy_int;

    tma_unit #(
        .ADDR_W(GLOBAL_ADDR_W),
        .SMEM_ADDR_W(SHARED_MEM_ADDR_W),
        .TRANSFER_SIZE(16)
    ) u_tma (
        .clk(clk),
        .rst_n(rst_n),
        .start(tma_start),
        .tensor_desc(tensor_desc),
        .coord_x(tensor_coord_x),
        .coord_y(tensor_coord_y),
        .dst_base(dst_addr),
        .req_valid(tma_req_valid),
        .req_src_addr(tma_req_src_addr),
        .req_dst_addr(tma_req_dst_addr),
        .req_size(tma_req_size),
        .req_ready(tma_req_ready),
        .done(tma_done),
        .busy(tma_busy_int),
        .bytes_copied(tma_bytes_copied)
    );

    // TMA mode tracking
    reg tma_active;

    //------------------------------------------------------------------------
    // 主状态机
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            ready <= 1'b1;
            done <= 1'b0;
            pending_count <= 4'd0;
            req_head <= 0;
            req_tail <= 0;
            req_count <= 0;
            req_valid <= 0;
            req_complete <= 0;
            req_is_store <= 0;
            current_group <= 0;
            group_committed <= 0;
            gmem_req_valid <= 1'b0;
            gmem_wr_valid <= 1'b0;
            smem_wr_en <= 1'b0;
            smem_rd_en <= 1'b0;

            for (integer i = 0; i < MAX_GROUPS; i = i + 1) begin
                group_pending[i] <= 5'd0;
            end
            tma_start <= 1'b0;
            tma_active <= 1'b0;
        end else begin
            tma_start <= 1'b0;  // Default: deassert TMA start
            done <= 1'b0;
            gmem_req_valid <= 1'b0;
            gmem_wr_valid <= 1'b0;
            smem_wr_en <= 1'b0;
            smem_rd_en <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ready <= 1'b1;

                    if (valid_in) begin
                        // Route based on opcode first, then func
                        if (opcode == `OP_ST_ASYNC) begin
                            // st.async operations
                            case (func)
                                `ST_ASYNC_GLOBAL: begin
                                    // st.async.global: Async store to global memory
                                    if (req_count < QUEUE_DEPTH) begin
                                        req_src_addr[req_head] <= {18'b0, dst_addr};  // SMEM addr
                                        req_gmem_addr[req_head] <= store_gmem_addr;   // Global memory address
                                        req_data[req_head] <= store_data;             // Data to store
                                        req_size[req_head] <= size;
                                        req_cache[req_head] <= cache_hint;
                                        req_group[req_head] <= current_group;
                                        req_valid[req_head] <= 1'b1;
                                        req_complete[req_head] <= 1'b0;
                                        req_is_store[req_head] <= 1'b1;

                                        req_head <= req_head + 1;
                                        req_count <= req_count + 1;
                                        group_pending[current_group] <= group_pending[current_group] + 1;
                                        pending_count <= pending_count + 1;

                                        `ifdef SIMULATION
                                        $display("[ACE] st.async.global: gmem=0x%08x data=0x%08x size=%0d",
                                                 store_gmem_addr, store_data[31:0], size);
                                        `endif

                                        state <= ST_ISSUE;
                                    end
                                    done <= 1'b1;
                                end

                                `ST_ASYNC_SHARED: begin
                                    // st.async.shared: Async store to local shared memory
                                    smem_wr_en <= 1'b1;
                                    smem_wr_addr <= dst_addr;
                                    smem_wr_data <= store_data;
                                    smem_wr_size <= {1'b0, size};
                                    done <= 1'b1;

                                    `ifdef SIMULATION
                                    $display("[ACE] st.async.shared: addr=0x%04x data=0x%08x size=%0d",
                                             dst_addr, store_data[31:0], size);
                                    `endif
                                end

                                `ST_ASYNC_COMMIT: begin
                                    // Commit async store group
                                    group_committed[current_group] <= 1'b1;
                                    current_group <= (current_group + 1) % MAX_GROUPS;
                                    done <= 1'b1;
                                end

                                `ST_ASYNC_WAIT: begin
                                    // Wait for async store group
                                    state <= ST_WAIT_GRP;
                                    ready <= 1'b0;
                                end

                                default: begin
                                    done <= 1'b1;
                                end
                            endcase
                        end else begin
                            // cp.async operations (default)
                            case (func)
                                `CPASYNC_CA, `CPASYNC_CG: begin
                                    // 添加新的异步拷贝请求
                                    if (req_count < QUEUE_DEPTH) begin
                                        req_src_addr[req_head] <= src_addr;
                                        req_dst_addr[req_head] <= dst_addr;
                                        req_size[req_head] <= size;
                                        req_cache[req_head] <= (func == `CPASYNC_CA) ? `CACHE_CA : `CACHE_CG;
                                        req_group[req_head] <= current_group;
                                        req_valid[req_head] <= 1'b1;
                                        req_complete[req_head] <= 1'b0;
                                        req_is_store[req_head] <= 1'b0;

                                        req_head <= req_head + 1;
                                        req_count <= req_count + 1;
                                        group_pending[current_group] <= group_pending[current_group] + 1;
                                        pending_count <= pending_count + 1;

                                        // 开始处理
                                        state <= ST_ISSUE;
                                    end
                                    done <= 1'b1;
                                end

                                `CPASYNC_COMMIT: begin
                                // 提交当前组
                                group_committed[current_group] <= 1'b1;
                                current_group <= (current_group + 1) % MAX_GROUPS;
                                done <= 1'b1;
                            end

                            `CPASYNC_WAIT: begin
                                // 等待指定数量的组完成
                                state <= ST_WAIT_GRP;
                                ready <= 1'b0;
                            end

                            `CPASYNC_WAIT_ALL: begin
                                // 等待所有挂起拷贝完成
                                if (pending_count == 0) begin
                                    done <= 1'b1;
                                end else begin
                                    state <= ST_WAIT_GRP;
                                    ready <= 1'b0;
                                end
                            end

                            `CPASYNC_BULK: begin
                                // 批量拷贝 (简化实现: 同单次拷贝)
                                if (req_count < QUEUE_DEPTH) begin
                                    req_src_addr[req_head] <= src_addr;
                                    req_dst_addr[req_head] <= dst_addr;
                                    req_size[req_head] <= size;
                                    req_cache[req_head] <= cache_hint;
                                    req_group[req_head] <= current_group;
                                    req_valid[req_head] <= 1'b1;

                                    req_head <= req_head + 1;
                                    req_count <= req_count + 1;
                                    group_pending[current_group] <= group_pending[current_group] + 1;
                                    pending_count <= pending_count + 1;

                                    state <= ST_ISSUE;
                                end
                                done <= 1'b1;
                            end

                            `CPASYNC_BULK_TENSOR: begin
                                // TMA: Tensor Memory Accelerator bulk copy
                                // Start TMA unit which will generate requests
                                if (!tma_busy_int) begin
                                    tma_start <= 1'b1;
                                    tma_active <= 1'b1;
                                    ready <= 1'b0;
                                    `ifdef SIMULATION
                                    $display("[ACE] TMA start: desc=0x%016x coord=(%0d,%0d) dst=0x%04x",
                                             tensor_desc, tensor_coord_x, tensor_coord_y, dst_addr);
                                    `endif
                                end
                                done <= 1'b1;
                            end

                            default: begin
                                done <= 1'b1;
                            end
                        endcase
                        end  // end of cp.async (else branch)
                    end  // end of if (valid_in)

                    // Handle TMA request injection into the queue
                    if (tma_active && tma_req_valid && req_count < QUEUE_DEPTH) begin
                        req_src_addr[req_head] <= tma_req_src_addr;
                        req_dst_addr[req_head] <= tma_req_dst_addr;
                        req_size[req_head] <= tma_req_size[3:0];
                        req_cache[req_head] <= `CACHE_CG;  // TMA uses global caching
                        req_group[req_head] <= current_group;
                        req_valid[req_head] <= 1'b1;
                        req_complete[req_head] <= 1'b0;

                        req_head <= req_head + 1;
                        req_count <= req_count + 1;
                        group_pending[current_group] <= group_pending[current_group] + 1;
                        pending_count <= pending_count + 1;

                        if (state == ST_IDLE) begin
                            state <= ST_ISSUE;
                        end
                    end

                    // Check if TMA completed
                    if (tma_active && tma_done) begin
                        tma_active <= 1'b0;
                        `ifdef SIMULATION
                        $display("[ACE] TMA done: bytes_copied=%0d", tma_bytes_copied);
                        `endif
                    end

                    if (!valid_in && req_count > 0 && !req_complete[req_tail]) begin
                        // 有挂起请求，继续处理
                        state <= ST_ISSUE;
                    end
                end

                ST_ISSUE: begin
                    ready <= 1'b0;
                    if (req_valid[req_tail] && !req_complete[req_tail]) begin
                        current_req <= req_tail;

                        if (req_is_store[req_tail]) begin
                            // Store operation: write data to global memory
                            // Data is already in req_data, just issue the write
                            gmem_wr_valid <= 1'b1;
                            gmem_wr_addr <= req_gmem_addr[req_tail];
                            gmem_wr_data <= req_data[req_tail];
                            gmem_wr_size <= {1'b0, req_size[req_tail]};
                            state <= ST_WAIT_WR;

                            `ifdef SIMULATION
                            $display("[ACE] Store issue: gmem=0x%08x size=%0d",
                                     req_gmem_addr[req_tail], req_size[req_tail]);
                            `endif
                        end else begin
                            // Load operation: 发起全局内存读请求
                            gmem_req_valid <= 1'b1;
                            gmem_req_addr <= req_src_addr[req_tail];
                            gmem_req_size <= {1'b0, req_size[req_tail]};
                            gmem_req_cache <= req_cache[req_tail];
                            state <= ST_WAIT_RESP;
                        end
                    end else begin
                        // 移动到下一个请求
                        req_tail <= req_tail + 1;
                        if (req_count > 1) begin
                            state <= ST_ISSUE;
                        end else begin
                            state <= ST_IDLE;
                        end
                    end
                end

                ST_WAIT_RESP: begin
                    gmem_req_valid <= 1'b0;
                    if (gmem_resp_valid) begin
                        data_buffer <= gmem_resp_data;
                        state <= ST_WRITE_SM;
                    end
                end

                ST_WRITE_SM: begin
                    // 写入共享内存
                    smem_wr_en <= 1'b1;
                    smem_wr_addr <= req_dst_addr[current_req];
                    smem_wr_data <= data_buffer;
                    smem_wr_size <= req_size[current_req];

                    // 标记完成
                    req_complete[current_req] <= 1'b1;
                    req_valid[current_req] <= 1'b0;
                    req_count <= req_count - 1;

                    // 更新组计数
                    if (group_pending[req_group[current_req]] > 0) begin
                        group_pending[req_group[current_req]] <=
                            group_pending[req_group[current_req]] - 1;
                    end
                    if (pending_count > 0) begin
                        pending_count <= pending_count - 1;
                    end

                    req_tail <= req_tail + 1;

                    // 继续处理或返回空闲
                    if (req_count > 1) begin
                        state <= ST_ISSUE;
                    end else begin
                        state <= ST_IDLE;
                    end
                end

                ST_WAIT_WR: begin
                    // Wait for global memory write completion (st.async.global)
                    gmem_wr_valid <= 1'b0;
                    if (gmem_wr_done) begin
                        // 标记完成
                        req_complete[current_req] <= 1'b1;
                        req_valid[current_req] <= 1'b0;
                        req_is_store[current_req] <= 1'b0;
                        req_count <= req_count - 1;

                        // 更新组计数
                        if (group_pending[req_group[current_req]] > 0) begin
                            group_pending[req_group[current_req]] <=
                                group_pending[req_group[current_req]] - 1;
                        end
                        if (pending_count > 0) begin
                            pending_count <= pending_count - 1;
                        end

                        req_tail <= req_tail + 1;

                        `ifdef SIMULATION
                        $display("[ACE] Store complete: gmem=0x%08x", req_gmem_addr[current_req]);
                        `endif

                        // 继续处理或返回空闲
                        if (req_count > 1) begin
                            state <= ST_ISSUE;
                        end else begin
                            state <= ST_IDLE;
                        end
                    end
                end

                ST_WAIT_GRP: begin
                    // 等待组完成
                    if (func == `CPASYNC_WAIT_ALL || func == `ST_ASYNC_WAIT) begin
                        if (pending_count == 0) begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else if (req_count > 0) begin
                            // 继续处理挂起请求
                            state <= ST_ISSUE;
                        end
                    end else begin
                        // wait_group: 等待指定数量的组
                        // 简化: 检查已提交组的挂起数
                        reg [3:0] completed_groups;
                        completed_groups = 0;
                        for (integer g = 0; g < MAX_GROUPS; g = g + 1) begin
                            if (group_committed[g] && group_pending[g] == 0) begin
                                completed_groups = completed_groups + 1;
                            end
                        end

                        if (completed_groups >= wait_count) begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else if (req_count > 0) begin
                            state <= ST_ISSUE;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// 带缓存控制的加载单元
// 支持 ld.ca, ld.cg, ld.cs, ld.lu, ld.cv
//============================================================================
module load_unit_cached #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // 请求接口
    input  wire                     req_valid,
    input  wire [ADDR_WIDTH-1:0]    req_addr,
    input  wire [2:0]               cache_hint,     // 缓存提示
    input  wire [1:0]               size,           // 0=byte, 1=half, 2=word, 3=dword
    output reg                      req_ready,

    // 响应接口
    output reg                      resp_valid,
    output reg  [DATA_WIDTH-1:0]    resp_data,

    // L1 缓存接口
    output reg                      l1_req_valid,
    output reg  [ADDR_WIDTH-1:0]    l1_req_addr,
    output reg  [2:0]               l1_req_hint,
    input  wire                     l1_resp_valid,
    input  wire [DATA_WIDTH-1:0]    l1_resp_data,
    input  wire                     l1_hit,

    // L2/全局内存接口 (绕过L1)
    output reg                      gmem_req_valid,
    output reg  [ADDR_WIDTH-1:0]    gmem_req_addr,
    output reg  [2:0]               gmem_req_hint,
    input  wire                     gmem_resp_valid,
    input  wire [DATA_WIDTH-1:0]    gmem_resp_data
);

    localparam ST_IDLE      = 2'd0;
    localparam ST_L1_CHECK  = 2'd1;
    localparam ST_GMEM_REQ  = 2'd2;
    localparam ST_WAIT_RESP = 2'd3;

    reg [1:0] state;
    reg [2:0] saved_hint;
    reg [ADDR_WIDTH-1:0] saved_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            req_ready <= 1'b1;
            resp_valid <= 1'b0;
            l1_req_valid <= 1'b0;
            gmem_req_valid <= 1'b0;
        end else begin
            resp_valid <= 1'b0;
            l1_req_valid <= 1'b0;
            gmem_req_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    req_ready <= 1'b1;
                    if (req_valid) begin
                        saved_hint <= cache_hint;
                        saved_addr <= req_addr;

                        case (cache_hint)
                            `CACHE_CV: begin
                                // Volatile: 绕过所有缓存
                                gmem_req_valid <= 1'b1;
                                gmem_req_addr <= req_addr;
                                gmem_req_hint <= cache_hint;
                                state <= ST_WAIT_RESP;
                                req_ready <= 1'b0;
                            end

                            `CACHE_LU: begin
                                // Last use: 读取后使缓存行无效
                                l1_req_valid <= 1'b1;
                                l1_req_addr <= req_addr;
                                l1_req_hint <= cache_hint;
                                state <= ST_L1_CHECK;
                                req_ready <= 1'b0;
                            end

                            default: begin
                                // CA, CG, CS, DEFAULT: 通过L1缓存
                                l1_req_valid <= 1'b1;
                                l1_req_addr <= req_addr;
                                l1_req_hint <= cache_hint;
                                state <= ST_L1_CHECK;
                                req_ready <= 1'b0;
                            end
                        endcase
                    end
                end

                ST_L1_CHECK: begin
                    if (l1_resp_valid) begin
                        if (l1_hit) begin
                            resp_valid <= 1'b1;
                            resp_data <= l1_resp_data;
                            state <= ST_IDLE;
                        end else begin
                            // L1 miss, 请求全局内存
                            gmem_req_valid <= 1'b1;
                            gmem_req_addr <= saved_addr;
                            gmem_req_hint <= saved_hint;
                            state <= ST_WAIT_RESP;
                        end
                    end
                end

                ST_WAIT_RESP: begin
                    if (gmem_resp_valid) begin
                        resp_valid <= 1'b1;
                        resp_data <= gmem_resp_data;
                        state <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// 带缓存控制的存储单元
// 支持 st.wb, st.wt, st.cg
//============================================================================
module store_unit_cached #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // 请求接口
    input  wire                     req_valid,
    input  wire [ADDR_WIDTH-1:0]    req_addr,
    input  wire [DATA_WIDTH-1:0]    req_data,
    input  wire [2:0]               cache_hint,
    input  wire [1:0]               size,
    output reg                      req_ready,
    output reg                      done,

    // L1 缓存写接口
    output reg                      l1_wr_valid,
    output reg  [ADDR_WIDTH-1:0]    l1_wr_addr,
    output reg  [DATA_WIDTH-1:0]    l1_wr_data,
    output reg  [2:0]               l1_wr_hint,
    input  wire                     l1_wr_done,

    // 全局内存写接口
    output reg                      gmem_wr_valid,
    output reg  [ADDR_WIDTH-1:0]    gmem_wr_addr,
    output reg  [DATA_WIDTH-1:0]    gmem_wr_data,
    input  wire                     gmem_wr_done
);

    localparam ST_IDLE      = 2'd0;
    localparam ST_L1_WRITE  = 2'd1;
    localparam ST_GMEM_WRITE = 2'd2;
    localparam ST_WAIT_DONE = 2'd3;

    reg [1:0] state;
    reg write_through;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            req_ready <= 1'b1;
            done <= 1'b0;
            l1_wr_valid <= 1'b0;
            gmem_wr_valid <= 1'b0;
        end else begin
            done <= 1'b0;
            l1_wr_valid <= 1'b0;
            gmem_wr_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    req_ready <= 1'b1;
                    if (req_valid) begin
                        req_ready <= 1'b0;

                        case (cache_hint)
                            `CACHE_WT: begin
                                // Write-through: 同时写L1和全局
                                l1_wr_valid <= 1'b1;
                                l1_wr_addr <= req_addr;
                                l1_wr_data <= req_data;
                                l1_wr_hint <= cache_hint;

                                gmem_wr_valid <= 1'b1;
                                gmem_wr_addr <= req_addr;
                                gmem_wr_data <= req_data;

                                write_through <= 1'b1;
                                state <= ST_WAIT_DONE;
                            end

                            `CACHE_WB: begin
                                // Write-back: 只写L1
                                l1_wr_valid <= 1'b1;
                                l1_wr_addr <= req_addr;
                                l1_wr_data <= req_data;
                                l1_wr_hint <= cache_hint;
                                write_through <= 1'b0;
                                state <= ST_L1_WRITE;
                            end

                            `CACHE_CG: begin
                                // Cache at global: 绕过L1写全局
                                gmem_wr_valid <= 1'b1;
                                gmem_wr_addr <= req_addr;
                                gmem_wr_data <= req_data;
                                write_through <= 1'b0;
                                state <= ST_GMEM_WRITE;
                            end

                            default: begin
                                // 默认: write-back
                                l1_wr_valid <= 1'b1;
                                l1_wr_addr <= req_addr;
                                l1_wr_data <= req_data;
                                l1_wr_hint <= cache_hint;
                                write_through <= 1'b0;
                                state <= ST_L1_WRITE;
                            end
                        endcase
                    end
                end

                ST_L1_WRITE: begin
                    if (l1_wr_done) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                ST_GMEM_WRITE: begin
                    if (gmem_wr_done) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                ST_WAIT_DONE: begin
                    // Write-through: 等待两者完成
                    if (write_through) begin
                        if (l1_wr_done && gmem_wr_done) begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    end else begin
                        if (gmem_wr_done) begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// 预取单元
// 支持 prefetch.L1, prefetch.L2, prefetchu.L1
//============================================================================
module prefetch_unit #(
    parameter ADDR_WIDTH = 32,
    parameter QUEUE_DEPTH = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // 请求接口
    input  wire                     req_valid,
    input  wire [ADDR_WIDTH-1:0]    req_addr,
    input  wire [5:0]               func,           // prefetch类型
    output reg                      req_ready,

    // L1缓存预取接口
    output reg                      l1_prefetch_valid,
    output reg  [ADDR_WIDTH-1:0]    l1_prefetch_addr,
    input  wire                     l1_prefetch_done,

    // L2缓存预取接口
    output reg                      l2_prefetch_valid,
    output reg  [ADDR_WIDTH-1:0]    l2_prefetch_addr,
    input  wire                     l2_prefetch_done
);

    // 预取队列
    reg [ADDR_WIDTH-1:0] pf_queue [0:QUEUE_DEPTH-1];
    reg [1:0]            pf_level [0:QUEUE_DEPTH-1];  // 0=L1, 1=L2, 2=uniform
    reg [QUEUE_DEPTH-1:0] pf_valid;
    reg [$clog2(QUEUE_DEPTH)-1:0] pf_head, pf_tail;
    reg [$clog2(QUEUE_DEPTH):0] pf_count;

    localparam ST_IDLE    = 2'd0;
    localparam ST_ISSUE   = 2'd1;
    localparam ST_WAIT    = 2'd2;

    reg [1:0] state;
    reg [1:0] current_level;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            req_ready <= 1'b1;
            l1_prefetch_valid <= 1'b0;
            l2_prefetch_valid <= 1'b0;
            pf_valid <= 0;
            pf_head <= 0;
            pf_tail <= 0;
            pf_count <= 0;
        end else begin
            l1_prefetch_valid <= 1'b0;
            l2_prefetch_valid <= 1'b0;

            // 入队新请求
            if (req_valid && pf_count < QUEUE_DEPTH) begin
                pf_queue[pf_head] <= req_addr;
                case (func)
                    `PREFETCH_L1:  pf_level[pf_head] <= 2'd0;
                    `PREFETCH_L2:  pf_level[pf_head] <= 2'd1;
                    `PREFETCHU_L1: pf_level[pf_head] <= 2'd2;
                    default:       pf_level[pf_head] <= 2'd0;
                endcase
                pf_valid[pf_head] <= 1'b1;
                pf_head <= pf_head + 1;
                pf_count <= pf_count + 1;
            end

            req_ready <= (pf_count < QUEUE_DEPTH);

            case (state)
                ST_IDLE: begin
                    if (pf_count > 0 && pf_valid[pf_tail]) begin
                        state <= ST_ISSUE;
                    end
                end

                ST_ISSUE: begin
                    current_level <= pf_level[pf_tail];
                    case (pf_level[pf_tail])
                        2'd0, 2'd2: begin  // L1 or uniform L1
                            l1_prefetch_valid <= 1'b1;
                            l1_prefetch_addr <= pf_queue[pf_tail];
                        end
                        2'd1: begin  // L2
                            l2_prefetch_valid <= 1'b1;
                            l2_prefetch_addr <= pf_queue[pf_tail];
                        end
                    endcase
                    state <= ST_WAIT;
                end

                ST_WAIT: begin
                    if ((current_level != 2'd1 && l1_prefetch_done) ||
                        (current_level == 2'd1 && l2_prefetch_done)) begin
                        pf_valid[pf_tail] <= 1'b0;
                        pf_tail <= pf_tail + 1;
                        pf_count <= pf_count - 1;
                        state <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
