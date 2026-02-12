//============================================================================
// RalphGPU - Global Memory Interface
// AXI4风格的全局内存接口
// 支持合并访问 (coalescing) 以提高带宽利用率
//============================================================================

`include "gpu_defines.vh"

module memory_interface #(
    parameter ADDR_WIDTH = `GLOBAL_ADDR_WIDTH,  // 32
    parameter DATA_WIDTH = `GLOBAL_DATA_WIDTH,  // 32
    parameter NUM_LANES  = `THREADS_PER_WARP,   // 32
    parameter AXI_ID_W   = 4
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // 来自执行单元的请求 (32个并行请求)
    //------------------------------------------------------------------------
    input  wire                     req_valid,
    input  wire                     req_write,
    input  wire [NUM_LANES*ADDR_WIDTH-1:0] req_addr,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] req_wdata,
    input  wire [NUM_LANES-1:0]     req_mask,
    output wire                     req_ready,

    output wire                     resp_valid,
    output wire [NUM_LANES*DATA_WIDTH-1:0] resp_rdata,

    //------------------------------------------------------------------------
    // AXI4 主接口
    //------------------------------------------------------------------------
    // 写地址通道
    output reg  [AXI_ID_W-1:0]      m_axi_awid,
    output reg  [ADDR_WIDTH-1:0]    m_axi_awaddr,
    output reg  [7:0]               m_axi_awlen,     // 突发长度
    output reg  [2:0]               m_axi_awsize,    // 2^size bytes
    output reg  [1:0]               m_axi_awburst,   // INCR
    output reg                      m_axi_awvalid,
    input  wire                     m_axi_awready,

    // 写数据通道
    output reg  [DATA_WIDTH-1:0]    m_axi_wdata,
    output reg  [DATA_WIDTH/8-1:0]  m_axi_wstrb,
    output reg                      m_axi_wlast,
    output reg                      m_axi_wvalid,
    input  wire                     m_axi_wready,

    // 写响应通道
    input  wire [AXI_ID_W-1:0]      m_axi_bid,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output reg                      m_axi_bready,

    // 读地址通道
    output reg  [AXI_ID_W-1:0]      m_axi_arid,
    output reg  [ADDR_WIDTH-1:0]    m_axi_araddr,
    output reg  [7:0]               m_axi_arlen,
    output reg  [2:0]               m_axi_arsize,
    output reg  [1:0]               m_axi_arburst,
    output reg                      m_axi_arvalid,
    input  wire                     m_axi_arready,

    // 读数据通道
    input  wire [AXI_ID_W-1:0]      m_axi_rid,
    input  wire [DATA_WIDTH-1:0]    m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output reg                      m_axi_rready
);

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    localparam IDLE       = 3'd0;
    localparam COALESCE   = 3'd1;
    localparam READ_ADDR  = 3'd2;
    localparam READ_DATA  = 3'd3;
    localparam WRITE_ADDR = 3'd4;
    localparam WRITE_DATA = 3'd5;
    localparam WRITE_RESP = 3'd6;

    reg [2:0] state, next_state;

    //------------------------------------------------------------------------
    // 请求缓存
    //------------------------------------------------------------------------
    reg [NUM_LANES*ADDR_WIDTH-1:0] addr_buf;
    reg [NUM_LANES*DATA_WIDTH-1:0] wdata_buf;
    reg [NUM_LANES-1:0] mask_buf;
    reg is_write_buf;

    // 当前处理的lane
    reg [5:0] current_lane;
    reg [5:0] lane_count;
    reg [5:0] processed_count;

    // 响应数据缓存
    reg [NUM_LANES*DATA_WIDTH-1:0] rdata_buf;
    reg resp_valid_reg;

    //------------------------------------------------------------------------
    // 地址合并分析 (简化版)
    // 检测连续地址以进行突发传输
    //------------------------------------------------------------------------
    wire [ADDR_WIDTH-1:0] lane0_addr = addr_buf[0 +: ADDR_WIDTH];

    //------------------------------------------------------------------------
    // 状态机逻辑
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (req_valid && |req_mask) begin
                    next_state = req_write ? WRITE_ADDR : READ_ADDR;
                end
            end

            READ_ADDR: begin
                if (m_axi_arready && m_axi_arvalid) begin
                    next_state = READ_DATA;
                end
            end

            READ_DATA: begin
                if (m_axi_rvalid && m_axi_rready) begin
                    if ((processed_count + 1'b1) >= lane_count) begin
                        next_state = IDLE;
                    end else begin
                        next_state = READ_ADDR;
                    end
                end
            end

            WRITE_ADDR: begin
                if (m_axi_awready && m_axi_awvalid) begin
                    next_state = WRITE_DATA;
                end
            end

            WRITE_DATA: begin
                if (m_axi_wready && m_axi_wvalid) begin
                    next_state = WRITE_RESP;
                end
            end

            WRITE_RESP: begin
                if (m_axi_bvalid && m_axi_bready) begin
                    if ((processed_count + 1'b1) >= lane_count) begin
                        next_state = IDLE;
                    end else begin
                        next_state = WRITE_ADDR;
                    end
                end
            end

            default: next_state = IDLE;
        endcase
    end

    //------------------------------------------------------------------------
    // 数据路径
    //------------------------------------------------------------------------
    integer i;

    // 计算活跃lane数量
    reg [5:0] active_count_req;
    always @(*) begin
        active_count_req = 0;
        for (i = 0; i < NUM_LANES; i = i + 1) begin
            if (req_mask[i]) active_count_req = active_count_req + 1;
        end
    end

    // 找到下一个活跃lane
    function [5:0] find_next_lane;
        input [5:0] start;
        input [NUM_LANES-1:0] mask;
        integer j;
        reg found;
        begin
            find_next_lane = start;
            found = 0;
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                if (!found && mask[(start + j) % NUM_LANES]) begin
                    find_next_lane = (start + j) % NUM_LANES;
                    found = 1;
                end
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_buf      <= 0;
            wdata_buf     <= 0;
            mask_buf      <= 0;
            is_write_buf  <= 0;
            current_lane  <= 0;
            lane_count    <= 0;
            processed_count <= 0;
            rdata_buf     <= 0;
            resp_valid_reg <= 0;

            m_axi_awid    <= 0;
            m_axi_awaddr  <= 0;
            m_axi_awlen   <= 0;
            m_axi_awsize  <= 3'b010;  // 4 bytes
            m_axi_awburst <= 2'b01;   // INCR
            m_axi_awvalid <= 0;
            m_axi_wdata   <= 0;
            m_axi_wstrb   <= 4'hF;
            m_axi_wlast   <= 1;
            m_axi_wvalid  <= 0;
            m_axi_bready  <= 0;
            m_axi_arid    <= 0;
            m_axi_araddr  <= 0;
            m_axi_arlen   <= 0;
            m_axi_arsize  <= 3'b010;
            m_axi_arburst <= 2'b01;
            m_axi_arvalid <= 0;
            m_axi_rready  <= 0;

        end else begin
            resp_valid_reg <= 1'b0;
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_bready  <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready  <= 1'b0;

            case (state)
                IDLE: begin
                    if (req_valid && |req_mask) begin
                        addr_buf     <= req_addr;
                        wdata_buf    <= req_wdata;
                        mask_buf     <= req_mask;
                        is_write_buf <= req_write;
                        current_lane <= find_next_lane(0, req_mask);
                        lane_count   <= active_count_req;
                        processed_count <= 0;
                    end
                end

                READ_ADDR: begin
                    m_axi_arid    <= current_lane[3:0];
                    m_axi_araddr  <= addr_buf[current_lane*ADDR_WIDTH +: ADDR_WIDTH];
                    m_axi_arlen   <= 8'd0;  // 单次传输
                    m_axi_arvalid <= 1'b1;
                end

                READ_DATA: begin
                    m_axi_rready <= 1'b1;
                    // Must check both rvalid AND rready for proper AXI handshake
                    if (m_axi_rvalid && m_axi_rready) begin
                        rdata_buf[current_lane*DATA_WIDTH +: DATA_WIDTH] <= m_axi_rdata;
                        processed_count <= processed_count + 1'b1;
                        if ((processed_count + 1'b1) >= lane_count) begin
                            resp_valid_reg <= 1'b1;
                        end else begin
                            current_lane <= find_next_lane(current_lane + 1, mask_buf);
                        end
                    end
                end

                WRITE_ADDR: begin
                    m_axi_awid    <= current_lane[3:0];
                    m_axi_awaddr  <= addr_buf[current_lane*ADDR_WIDTH +: ADDR_WIDTH];
                    m_axi_awlen   <= 8'd0;
                    m_axi_awvalid <= 1'b1;
                end

                WRITE_DATA: begin
                    m_axi_wdata  <= wdata_buf[current_lane*DATA_WIDTH +: DATA_WIDTH];
                    m_axi_wvalid <= 1'b1;
                    m_axi_wlast  <= 1'b1;
                end

                WRITE_RESP: begin
                    m_axi_bready <= 1'b1;
                    // Must check both bvalid AND bready for proper AXI handshake
                    if (m_axi_bvalid && m_axi_bready) begin
                        processed_count <= processed_count + 1'b1;
                        if ((processed_count + 1'b1) >= lane_count) begin
                            resp_valid_reg <= 1'b1;  // Signal write completion
                        end else begin
                            current_lane <= find_next_lane(current_lane + 1, mask_buf);
                        end
                    end
                end
                default: ; // lint: CASEINCOMPLETE
            endcase
        end
    end

    assign req_ready = (state == IDLE);
    assign resp_valid = resp_valid_reg;
    assign resp_rdata = rdata_buf;

`ifdef SIMULATION
    // Debug: trace AXI READ handshake only
    reg [31:0] mem_if_debug_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            mem_if_debug_cnt <= 0;
        else if (mem_if_debug_cnt < 30) begin
            if (state == READ_ADDR || state == READ_DATA) begin
                `ifdef SIMULATION
                $display("[%0t MEM_IF_RD] state=%0d arvalid=%b arready=%b rvalid=%b rready=%b addr=0x%08x rdata=0x%08x lane=%0d processed=%0d/%0d resp_valid=%b",
                         $time, state, m_axi_arvalid, m_axi_arready, m_axi_rvalid, m_axi_rready,
                         addr_buf[current_lane*ADDR_WIDTH +: ADDR_WIDTH], m_axi_rdata, current_lane, processed_count, lane_count, resp_valid_reg);
                `endif
                mem_if_debug_cnt <= mem_if_debug_cnt + 1;
            end
        end
    end
`endif

endmodule
