//============================================================================
// RalphGPU - Async Copy Engine
// 支持 cp.async 指令用于异步全局到共享内存拷贝
// PTX Instructions: cp.async.ca, cp.async.cg, cp.async.commit_group,
//                   cp.async.wait_group, cp.async.wait_all, cp.async.bulk
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
    input  wire [5:0]                   func,           // 功能码
    input  wire                         valid_in,
    input  wire [31:0]                  src_addr,       // 源地址 (全局内存)
    input  wire [SHARED_MEM_ADDR_W-1:0] dst_addr,       // 目标地址 (共享内存)
    input  wire [3:0]                   size,           // 拷贝大小: 4, 8, 16 bytes
    input  wire [2:0]                   cache_hint,     // 缓存提示
    input  wire [3:0]                   wait_count,     // wait_group的等待数量

    // 状态输出
    output reg                          ready,
    output reg                          done,
    output reg  [3:0]                   pending_count,  // 当前组挂起数量

    // 全局内存接口
    output reg                          gmem_req_valid,
    output reg  [GLOBAL_ADDR_W-1:0]     gmem_req_addr,
    output reg  [4:0]                   gmem_req_size,  // bytes
    output reg  [2:0]                   gmem_req_cache,
    input  wire                         gmem_resp_valid,
    input  wire [127:0]                 gmem_resp_data,

    // 共享内存写接口
    output reg                          smem_wr_en,
    output reg  [SHARED_MEM_ADDR_W-1:0] smem_wr_addr,
    output reg  [127:0]                 smem_wr_data,
    output reg  [4:0]                   smem_wr_size  // 5 bits to hold values up to 16
);

    //------------------------------------------------------------------------
    // 拷贝请求队列
    //------------------------------------------------------------------------
    localparam QUEUE_DEPTH = MAX_GROUPS * MAX_PENDING;
    localparam QUEUE_ADDR_W = $clog2(QUEUE_DEPTH);

    reg [GLOBAL_ADDR_W-1:0]     req_src_addr  [0:QUEUE_DEPTH-1];
    reg [SHARED_MEM_ADDR_W-1:0] req_dst_addr  [0:QUEUE_DEPTH-1];
    reg [3:0]                   req_size      [0:QUEUE_DEPTH-1];
    reg [2:0]                   req_cache     [0:QUEUE_DEPTH-1];
    reg [2:0]                   req_group     [0:QUEUE_DEPTH-1];
    reg [QUEUE_DEPTH-1:0]       req_valid;
    reg [QUEUE_DEPTH-1:0]       req_complete;

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
    localparam ST_IDLE      = 3'd0;
    localparam ST_ISSUE     = 3'd1;
    localparam ST_WAIT_RESP = 3'd2;
    localparam ST_WRITE_SM  = 3'd3;
    localparam ST_COMMIT    = 3'd4;
    localparam ST_WAIT_GRP  = 3'd5;

    reg [2:0] state;
    reg [QUEUE_ADDR_W-1:0] current_req;
    reg [127:0] data_buffer;
    reg [3:0] bytes_remaining;

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
            current_group <= 0;
            group_committed <= 0;
            gmem_req_valid <= 1'b0;
            smem_wr_en <= 1'b0;

            for (integer i = 0; i < MAX_GROUPS; i = i + 1) begin
                group_pending[i] <= 5'd0;
            end
        end else begin
            done <= 1'b0;
            gmem_req_valid <= 1'b0;
            smem_wr_en <= 1'b0;

            case (state)
                ST_IDLE: begin
                    ready <= 1'b1;

                    if (valid_in) begin
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

                            default: begin
                                done <= 1'b1;
                            end
                        endcase
                    end else if (req_count > 0 && !req_complete[req_tail]) begin
                        // 有挂起请求，继续处理
                        state <= ST_ISSUE;
                    end
                end

                ST_ISSUE: begin
                    ready <= 1'b0;
                    // 发起全局内存读请求
                    if (req_valid[req_tail] && !req_complete[req_tail]) begin
                        gmem_req_valid <= 1'b1;
                        gmem_req_addr <= req_src_addr[req_tail];
                        gmem_req_size <= {1'b0, req_size[req_tail]};
                        gmem_req_cache <= req_cache[req_tail];
                        current_req <= req_tail;
                        state <= ST_WAIT_RESP;
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

                ST_WAIT_GRP: begin
                    // 等待组完成
                    if (func == `CPASYNC_WAIT_ALL) begin
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
