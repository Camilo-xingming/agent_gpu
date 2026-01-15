//============================================================================
// RalphGPU - Top Level Module
// CUDA/PTX兼容GPU IP
//
// 特点:
// - 可配置SM数量 (默认2个)
// - 每SM 4个Warp，每Warp 32线程
// - PTX指令集子集支持
// - AXI4内存接口
// - 易于扩展的模块化设计
//============================================================================

`include "gpu_defines.vh"

module ralph_gpu_top #(
    parameter NUM_SM         = `NUM_SM,           // 2
    parameter AXI_DATA_WIDTH = 32,
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_ID_WIDTH   = 4
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // 控制/状态接口 (CSR)
    //------------------------------------------------------------------------
    input  wire                         csr_wr_en,
    input  wire [11:0]                  csr_addr,
    input  wire [31:0]                  csr_wr_data,
    output reg  [31:0]                  csr_rd_data,

    // 中断
    output wire                         irq_kernel_done,

    //------------------------------------------------------------------------
    // 指令内存接口 (只读)
    //------------------------------------------------------------------------
    output wire                         imem_req,
    output wire [31:0]                  imem_addr,
    input  wire [31:0]                  imem_data,
    input  wire                         imem_valid,

    //------------------------------------------------------------------------
    // 全局内存 AXI4 主接口
    //------------------------------------------------------------------------
    // 写地址通道
    output wire [AXI_ID_WIDTH-1:0]      m_axi_awid,
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,
    output wire [7:0]                   m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,

    // 写数据通道
    output wire [AXI_DATA_WIDTH-1:0]    m_axi_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0]  m_axi_wstrb,
    output wire                         m_axi_wlast,
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,

    // 写响应通道
    input  wire [AXI_ID_WIDTH-1:0]      m_axi_bid,
    input  wire [1:0]                   m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready,

    // 读地址通道
    output wire [AXI_ID_WIDTH-1:0]      m_axi_arid,
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,

    // 读数据通道
    input  wire [AXI_ID_WIDTH-1:0]      m_axi_rid,
    input  wire [AXI_DATA_WIDTH-1:0]    m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready
);

    //========================================================================
    // CSR 寄存器定义
    //========================================================================
    // 地址映射:
    // 0x000 - GPU_STATUS     : 状态寄存器 (RO)
    // 0x004 - GPU_CONTROL    : 控制寄存器 (RW)
    // 0x008 - KERNEL_PC      : Kernel起始PC (RW)
    // 0x00C - GRID_DIM_X     : Grid X维度 (RW)
    // 0x010 - GRID_DIM_Y     : Grid Y维度 (RW)
    // 0x014 - GRID_DIM_Z     : Grid Z维度 (RW)
    // 0x018 - BLOCK_DIM_X    : Block X维度 (RW)
    // 0x01C - BLOCK_DIM_Y    : Block Y维度 (RW)
    // 0x020 - BLOCK_DIM_Z    : Block Z维度 (RW)
    // 0x100 - SM0_STATUS     : SM0状态 (RO)
    // ...

    localparam CSR_GPU_STATUS   = 12'h000;
    localparam CSR_GPU_CONTROL  = 12'h004;
    localparam CSR_KERNEL_PC    = 12'h008;
    localparam CSR_GRID_DIM_X   = 12'h00C;
    localparam CSR_GRID_DIM_Y   = 12'h010;
    localparam CSR_GRID_DIM_Z   = 12'h014;
    localparam CSR_BLOCK_DIM_X  = 12'h018;
    localparam CSR_BLOCK_DIM_Y  = 12'h01C;
    localparam CSR_BLOCK_DIM_Z  = 12'h020;

    //------------------------------------------------------------------------
    // CSR存储
    //------------------------------------------------------------------------
    reg         gpu_busy;
    reg         kernel_start_reg;
    reg [31:0]  kernel_pc_reg;
    reg [31:0]  grid_dim_x, grid_dim_y, grid_dim_z;
    reg [31:0]  block_dim_x, block_dim_y, block_dim_z;

    //------------------------------------------------------------------------
    // Block分配器状态
    //------------------------------------------------------------------------
    reg [31:0]  current_block_id;
    reg [NUM_SM-1:0] sm_busy;
    wire [NUM_SM-1:0] sm_done;

    //------------------------------------------------------------------------
    // SM实例化
    //------------------------------------------------------------------------
    // SM接口信号
    wire [NUM_SM-1:0] sm_kernel_start;
    wire [NUM_SM-1:0] sm_imem_req;
    wire [31:0] sm_imem_addr [0:NUM_SM-1];

    // AXI仲裁 (简化：轮询)
    wire [3:0]  sm_axi_awid    [0:NUM_SM-1];
    wire [31:0] sm_axi_awaddr  [0:NUM_SM-1];
    wire [7:0]  sm_axi_awlen   [0:NUM_SM-1];
    wire [2:0]  sm_axi_awsize  [0:NUM_SM-1];
    wire [1:0]  sm_axi_awburst [0:NUM_SM-1];
    wire        sm_axi_awvalid [0:NUM_SM-1];
    wire [31:0] sm_axi_wdata   [0:NUM_SM-1];
    wire [3:0]  sm_axi_wstrb   [0:NUM_SM-1];
    wire        sm_axi_wlast   [0:NUM_SM-1];
    wire        sm_axi_wvalid  [0:NUM_SM-1];
    wire        sm_axi_bready  [0:NUM_SM-1];
    wire [3:0]  sm_axi_arid    [0:NUM_SM-1];
    wire [31:0] sm_axi_araddr  [0:NUM_SM-1];
    wire [7:0]  sm_axi_arlen   [0:NUM_SM-1];
    wire [2:0]  sm_axi_arsize  [0:NUM_SM-1];
    wire [1:0]  sm_axi_arburst [0:NUM_SM-1];
    wire        sm_axi_arvalid [0:NUM_SM-1];
    wire        sm_axi_rready  [0:NUM_SM-1];

    genvar sm;
    generate
        for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : sm_gen
            streaming_multiprocessor #(
                .SM_ID (sm)
            ) u_sm (
                .clk           (clk),
                .rst_n         (rst_n),

                .kernel_start  (sm_kernel_start[sm]),
                .kernel_pc     (kernel_pc_reg),
                .block_id_x    (current_block_id + sm),
                .block_dim_x   (block_dim_x),
                .grid_dim_x    (grid_dim_x),
                .kernel_done   (sm_done[sm]),

                .imem_req      (sm_imem_req[sm]),
                .imem_addr     (sm_imem_addr[sm]),
                .imem_data     (imem_data),
                .imem_valid    (imem_valid),

                // AXI接口
                .m_axi_awid    (sm_axi_awid[sm]),
                .m_axi_awaddr  (sm_axi_awaddr[sm]),
                .m_axi_awlen   (sm_axi_awlen[sm]),
                .m_axi_awsize  (sm_axi_awsize[sm]),
                .m_axi_awburst (sm_axi_awburst[sm]),
                .m_axi_awvalid (sm_axi_awvalid[sm]),
                .m_axi_awready (m_axi_awready),
                .m_axi_wdata   (sm_axi_wdata[sm]),
                .m_axi_wstrb   (sm_axi_wstrb[sm]),
                .m_axi_wlast   (sm_axi_wlast[sm]),
                .m_axi_wvalid  (sm_axi_wvalid[sm]),
                .m_axi_wready  (m_axi_wready),
                .m_axi_bid     (m_axi_bid),
                .m_axi_bresp   (m_axi_bresp),
                .m_axi_bvalid  (m_axi_bvalid),
                .m_axi_bready  (sm_axi_bready[sm]),
                .m_axi_arid    (sm_axi_arid[sm]),
                .m_axi_araddr  (sm_axi_araddr[sm]),
                .m_axi_arlen   (sm_axi_arlen[sm]),
                .m_axi_arsize  (sm_axi_arsize[sm]),
                .m_axi_arburst (sm_axi_arburst[sm]),
                .m_axi_arvalid (sm_axi_arvalid[sm]),
                .m_axi_arready (m_axi_arready),
                .m_axi_rid     (m_axi_rid),
                .m_axi_rdata   (m_axi_rdata),
                .m_axi_rresp   (m_axi_rresp),
                .m_axi_rlast   (m_axi_rlast),
                .m_axi_rvalid  (m_axi_rvalid),
                .m_axi_rready  (sm_axi_rready[sm])
            );
        end
    endgenerate

    //------------------------------------------------------------------------
    // 指令内存仲裁 (轮询)
    //------------------------------------------------------------------------
    reg [$clog2(NUM_SM)-1:0] imem_arb_sel;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_arb_sel <= 0;
        end else begin
            if (|sm_imem_req) begin
                imem_arb_sel <= (imem_arb_sel + 1) % NUM_SM;
            end
        end
    end

    assign imem_req  = sm_imem_req[imem_arb_sel];
    assign imem_addr = sm_imem_addr[imem_arb_sel];

    //------------------------------------------------------------------------
    // AXI仲裁 (简化：选择第一个活跃SM)
    //------------------------------------------------------------------------
    reg [$clog2(NUM_SM)-1:0] axi_arb_sel;

    integer i;
    always @(*) begin
        axi_arb_sel = 0;
        for (i = 0; i < NUM_SM; i = i + 1) begin
            if (sm_axi_awvalid[i] || sm_axi_arvalid[i]) begin
                axi_arb_sel = i;
            end
        end
    end

    assign m_axi_awid    = sm_axi_awid[axi_arb_sel];
    assign m_axi_awaddr  = sm_axi_awaddr[axi_arb_sel];
    assign m_axi_awlen   = sm_axi_awlen[axi_arb_sel];
    assign m_axi_awsize  = sm_axi_awsize[axi_arb_sel];
    assign m_axi_awburst = sm_axi_awburst[axi_arb_sel];
    assign m_axi_awvalid = sm_axi_awvalid[axi_arb_sel];
    assign m_axi_wdata   = sm_axi_wdata[axi_arb_sel];
    assign m_axi_wstrb   = sm_axi_wstrb[axi_arb_sel];
    assign m_axi_wlast   = sm_axi_wlast[axi_arb_sel];
    assign m_axi_wvalid  = sm_axi_wvalid[axi_arb_sel];
    assign m_axi_bready  = sm_axi_bready[axi_arb_sel];
    assign m_axi_arid    = sm_axi_arid[axi_arb_sel];
    assign m_axi_araddr  = sm_axi_araddr[axi_arb_sel];
    assign m_axi_arlen   = sm_axi_arlen[axi_arb_sel];
    assign m_axi_arsize  = sm_axi_arsize[axi_arb_sel];
    assign m_axi_arburst = sm_axi_arburst[axi_arb_sel];
    assign m_axi_arvalid = sm_axi_arvalid[axi_arb_sel];
    assign m_axi_rready  = sm_axi_rready[axi_arb_sel];

    //------------------------------------------------------------------------
    // Kernel调度状态机
    //------------------------------------------------------------------------
    localparam SCHED_IDLE     = 2'd0;
    localparam SCHED_DISPATCH = 2'd1;
    localparam SCHED_WAIT     = 2'd2;
    localparam SCHED_DONE     = 2'd3;

    reg [1:0] sched_state;
    reg [31:0] total_blocks;
    reg [31:0] dispatched_blocks;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sched_state       <= SCHED_IDLE;
            current_block_id  <= 0;
            dispatched_blocks <= 0;
            total_blocks      <= 0;
            gpu_busy          <= 0;
            sm_busy           <= 0;
        end else begin
            case (sched_state)
                SCHED_IDLE: begin
                    if (kernel_start_reg) begin
                        sched_state       <= SCHED_DISPATCH;
                        total_blocks      <= grid_dim_x * grid_dim_y * grid_dim_z;
                        current_block_id  <= 0;
                        dispatched_blocks <= 0;
                        gpu_busy          <= 1;
                    end
                end

                SCHED_DISPATCH: begin
                    // 为空闲SM分配Block
                    for (i = 0; i < NUM_SM; i = i + 1) begin
                        if (!sm_busy[i] && dispatched_blocks < total_blocks) begin
                            sm_busy[i] <= 1;
                            dispatched_blocks <= dispatched_blocks + 1;
                        end
                    end

                    if (dispatched_blocks >= total_blocks) begin
                        sched_state <= SCHED_WAIT;
                    end
                end

                SCHED_WAIT: begin
                    // 更新SM完成状态
                    for (i = 0; i < NUM_SM; i = i + 1) begin
                        if (sm_done[i]) begin
                            sm_busy[i] <= 0;
                            current_block_id <= current_block_id + 1;
                        end
                    end

                    // 如果还有未分配的Block，继续分配
                    if (dispatched_blocks < total_blocks) begin
                        sched_state <= SCHED_DISPATCH;
                    end else if (sm_busy == 0) begin
                        sched_state <= SCHED_DONE;
                    end
                end

                SCHED_DONE: begin
                    gpu_busy    <= 0;
                    sched_state <= SCHED_IDLE;
                end
            endcase
        end
    end

    // SM启动信号
    generate
        for (sm = 0; sm < NUM_SM; sm = sm + 1) begin : sm_start
            assign sm_kernel_start[sm] = (sched_state == SCHED_DISPATCH) &&
                                          !sm_busy[sm] &&
                                          (dispatched_blocks < total_blocks);
        end
    endgenerate

    //------------------------------------------------------------------------
    // CSR读写
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_start_reg <= 0;
            kernel_pc_reg    <= 0;
            grid_dim_x       <= 1;
            grid_dim_y       <= 1;
            grid_dim_z       <= 1;
            block_dim_x      <= 32;
            block_dim_y      <= 1;
            block_dim_z      <= 1;
        end else begin
            // 自动清除启动标志
            if (kernel_start_reg && sched_state != SCHED_IDLE) begin
                kernel_start_reg <= 0;
            end

            if (csr_wr_en) begin
                case (csr_addr)
                    CSR_GPU_CONTROL: kernel_start_reg <= csr_wr_data[0];
                    CSR_KERNEL_PC:   kernel_pc_reg    <= csr_wr_data;
                    CSR_GRID_DIM_X:  grid_dim_x       <= csr_wr_data;
                    CSR_GRID_DIM_Y:  grid_dim_y       <= csr_wr_data;
                    CSR_GRID_DIM_Z:  grid_dim_z       <= csr_wr_data;
                    CSR_BLOCK_DIM_X: block_dim_x      <= csr_wr_data;
                    CSR_BLOCK_DIM_Y: block_dim_y      <= csr_wr_data;
                    CSR_BLOCK_DIM_Z: block_dim_z      <= csr_wr_data;
                endcase
            end
        end
    end

    // CSR读取
    always @(*) begin
        case (csr_addr)
            CSR_GPU_STATUS:  csr_rd_data = {30'b0, gpu_busy, 1'b1};  // bit0=ready
            CSR_GPU_CONTROL: csr_rd_data = {31'b0, kernel_start_reg};
            CSR_KERNEL_PC:   csr_rd_data = kernel_pc_reg;
            CSR_GRID_DIM_X:  csr_rd_data = grid_dim_x;
            CSR_GRID_DIM_Y:  csr_rd_data = grid_dim_y;
            CSR_GRID_DIM_Z:  csr_rd_data = grid_dim_z;
            CSR_BLOCK_DIM_X: csr_rd_data = block_dim_x;
            CSR_BLOCK_DIM_Y: csr_rd_data = block_dim_y;
            CSR_BLOCK_DIM_Z: csr_rd_data = block_dim_z;
            default:         csr_rd_data = 32'b0;
        endcase
    end

    //------------------------------------------------------------------------
    // 中断
    //------------------------------------------------------------------------
    reg kernel_done_latch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_done_latch <= 0;
        end else begin
            if (sched_state == SCHED_DONE) begin
                kernel_done_latch <= 1;
            end else if (csr_wr_en && csr_addr == CSR_GPU_STATUS) begin
                kernel_done_latch <= 0;  // 写状态寄存器清除中断
            end
        end
    end

    assign irq_kernel_done = kernel_done_latch;

endmodule
