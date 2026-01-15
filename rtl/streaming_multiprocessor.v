//============================================================================
// RalphGPU - Streaming Multiprocessor (SM)
// 完整的SM单元，集成所有子模块
//============================================================================

`include "gpu_defines.vh"

module streaming_multiprocessor #(
    parameter SM_ID      = 0,
    parameter NUM_WARPS  = `WARPS_PER_SM,
    parameter NUM_LANES  = `THREADS_PER_WARP,
    parameter DATA_WIDTH = `DATA_WIDTH
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Kernel启动接口
    input  wire                     kernel_start,
    input  wire [31:0]              kernel_pc,        // Kernel起始地址
    input  wire [31:0]              block_id_x,       // 分配的Block ID
    input  wire [31:0]              block_dim_x,      // Block维度
    input  wire [31:0]              grid_dim_x,       // Grid维度
    output wire                     kernel_done,      // Kernel执行完成

    // 指令内存接口
    output wire                     imem_req,
    output wire [31:0]              imem_addr,
    input  wire [31:0]              imem_data,
    input  wire                     imem_valid,

    // 全局内存接口 (AXI)
    output wire [3:0]               m_axi_awid,
    output wire [31:0]              m_axi_awaddr,
    output wire [7:0]               m_axi_awlen,
    output wire [2:0]               m_axi_awsize,
    output wire [1:0]               m_axi_awburst,
    output wire                     m_axi_awvalid,
    input  wire                     m_axi_awready,
    output wire [31:0]              m_axi_wdata,
    output wire [3:0]               m_axi_wstrb,
    output wire                     m_axi_wlast,
    output wire                     m_axi_wvalid,
    input  wire                     m_axi_wready,
    input  wire [3:0]               m_axi_bid,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output wire                     m_axi_bready,
    output wire [3:0]               m_axi_arid,
    output wire [31:0]              m_axi_araddr,
    output wire [7:0]               m_axi_arlen,
    output wire [2:0]               m_axi_arsize,
    output wire [1:0]               m_axi_arburst,
    output wire                     m_axi_arvalid,
    input  wire                     m_axi_arready,
    input  wire [3:0]               m_axi_rid,
    input  wire [31:0]              m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output wire                     m_axi_rready
);

    //------------------------------------------------------------------------
    // 内部信号
    //------------------------------------------------------------------------
    // Warp调度器信号
    wire [NUM_WARPS-1:0] warp_valid;
    wire [NUM_WARPS-1:0] warp_ready;
    wire [NUM_WARPS-1:0] warp_waiting;
    wire [1:0] active_warp_id;
    wire warp_selected;
    wire [NUM_WARPS-1:0] warp_active_oh;

    // 指令解码信号
    wire [31:0] current_instruction;
    wire decode_valid;
    wire [5:0] dec_opcode;
    wire [4:0] dec_rd, dec_ra, dec_rb, dec_rc;
    wire [5:0] dec_func;
    wire dec_alu_op, dec_mul_op, dec_mem_read, dec_mem_write;
    wire dec_mem_shared, dec_branch_op, dec_sync_op, dec_special_reg;
    wire dec_exit_op;  // EXIT/RET指令
    wire dec_reg_write, dec_pred_write;

    // 寄存器文件信号
    wire [NUM_LANES*DATA_WIDTH-1:0] rf_rd_data_a;
    wire [NUM_LANES*DATA_WIDTH-1:0] rf_rd_data_b;
    wire [NUM_LANES*DATA_WIDTH-1:0] rf_rd_data_c;
    wire rf_wr_en;
    wire [NUM_LANES*DATA_WIDTH-1:0] rf_wr_data;
    wire [NUM_LANES-1:0] rf_wr_mask;

    // ALU信号
    wire [NUM_LANES*DATA_WIDTH-1:0] alu_result;
    wire [NUM_LANES-1:0] alu_zero, alu_neg;

    // 乘法器信号
    wire mul_valid_out;
    wire [NUM_LANES*DATA_WIDTH-1:0] mul_result;

    // 共享内存信号
    wire smem_req_valid;
    wire smem_req_write;
    wire [NUM_LANES*14-1:0] smem_req_addr;
    wire [NUM_LANES*DATA_WIDTH-1:0] smem_req_wdata;
    wire smem_resp_valid;
    wire [NUM_LANES*DATA_WIDTH-1:0] smem_resp_rdata;

    // 全局内存信号
    wire gmem_req_valid;
    wire gmem_req_write;
    wire [NUM_LANES*32-1:0] gmem_req_addr;
    wire [NUM_LANES*DATA_WIDTH-1:0] gmem_req_wdata;
    wire gmem_req_ready;
    wire gmem_resp_valid;
    wire [NUM_LANES*DATA_WIDTH-1:0] gmem_resp_rdata;

    // 线程掩码 (活跃线程)
    reg [NUM_LANES-1:0] active_mask;

    // 特殊寄存器
    reg [31:0] block_id_reg;
    reg [31:0] block_dim_reg;
    reg [31:0] grid_dim_reg;

    //------------------------------------------------------------------------
    // 流水线状态
    //------------------------------------------------------------------------
    localparam PIPE_IDLE   = 3'd0;
    localparam PIPE_FETCH  = 3'd1;
    localparam PIPE_DECODE = 3'd2;
    localparam PIPE_EXEC   = 3'd3;
    localparam PIPE_MEM    = 3'd4;
    localparam PIPE_WB     = 3'd5;

    reg [2:0] pipe_state;
    reg [31:0] pc_reg;
    reg [31:0] instruction_reg;

    //------------------------------------------------------------------------
    // Warp状态
    //------------------------------------------------------------------------
    wire [NUM_WARPS*32-1:0] warp_pc_flat;

    // Warp退出信号
    wire warp_exit_en = dec_exit_op && (pipe_state == PIPE_EXEC);

    warp_state #(
        .NUM_WARPS (NUM_WARPS)
    ) u_warp_state (
        .clk              (clk),
        .rst_n            (rst_n),
        .alloc_en         (kernel_start),
        .alloc_warp_id    (2'd0),
        .alloc_pc         (kernel_pc),
        .dealloc_en       (warp_exit_en),           // EXIT时释放Warp
        .dealloc_warp_id  (active_warp_id),         // 释放当前Warp
        .pc_update_en     (pipe_state == PIPE_WB && !dec_exit_op),
        .pc_update_warp   (active_warp_id),
        .pc_update_value  (pc_reg + 4),
        .pc_is_branch     (dec_branch_op),
        .sync_start       (dec_sync_op && pipe_state == PIPE_EXEC),
        .sync_warp_id     (active_warp_id),
        .sync_complete    (1'b0),  // 简化：暂不实现完整同步
        .warp_valid       (warp_valid),
        .warp_ready       (warp_ready),
        .warp_waiting     (warp_waiting),
        .warp_pc_flat     (warp_pc_flat)
    );

    //------------------------------------------------------------------------
    // Warp调度器
    //------------------------------------------------------------------------
    warp_scheduler #(
        .NUM_WARPS (NUM_WARPS)
    ) u_warp_scheduler (
        .clk           (clk),
        .rst_n         (rst_n),
        .warp_valid    (warp_valid),
        .warp_ready    (warp_ready),
        .warp_waiting  (warp_waiting),
        .active_warp_id(active_warp_id),
        .warp_selected (warp_selected),
        .warp_active_oh(warp_active_oh)
    );

    //------------------------------------------------------------------------
    // 指令解码器
    //------------------------------------------------------------------------
    decoder u_decoder (
        .clk         (clk),
        .rst_n       (rst_n),
        .instruction (instruction_reg),
        .valid_in    (pipe_state == PIPE_DECODE),
        .valid_out   (decode_valid),
        .opcode      (dec_opcode),
        .rd          (dec_rd),
        .ra          (dec_ra),
        .rb          (dec_rb),
        .rc          (dec_rc),
        .func        (dec_func),
        .imm16       (),
        .imm21       (),
        .use_imm     (),
        .alu_op      (dec_alu_op),
        .mul_op      (dec_mul_op),
        .div_op      (),
        .mem_read    (dec_mem_read),
        .mem_write   (dec_mem_write),
        .mem_shared  (dec_mem_shared),
        .branch_op   (dec_branch_op),
        .sync_op     (dec_sync_op),
        .special_reg (dec_special_reg),
        .exit_op     (dec_exit_op),
        .reg_write   (dec_reg_write),
        .pred_write  (dec_pred_write),
        .pred_addr   ()
    );

    //------------------------------------------------------------------------
    // 寄存器文件 (每个Warp一个)
    //------------------------------------------------------------------------
    // 简化：只实例化一个，实际应该有NUM_WARPS个
    register_file u_regfile (
        .clk       (clk),
        .rst_n     (rst_n),
        .rd_addr_a (dec_ra),
        .rd_data_a (rf_rd_data_a),
        .rd_addr_b (dec_rb),
        .rd_data_b (rf_rd_data_b),
        .rd_addr_c (dec_rc),
        .rd_data_c (rf_rd_data_c),
        .wr_en     (rf_wr_en),
        .wr_addr   (dec_rd),
        .wr_data   (rf_wr_data),
        .wr_mask   (rf_wr_mask)
    );

    //------------------------------------------------------------------------
    // SIMD ALU
    //------------------------------------------------------------------------
    simd_alu u_simd_alu (
        .func       (dec_func),
        .operand_a  (rf_rd_data_a),
        .operand_b  (rf_rd_data_b),
        .lane_mask  (active_mask),
        .result     (alu_result),
        .zero_flags (alu_zero),
        .neg_flags  (alu_neg)
    );

    //------------------------------------------------------------------------
    // SIMD 乘法器
    //------------------------------------------------------------------------
    simd_mul_unit u_simd_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (dec_mul_op && pipe_state == PIPE_EXEC),
        .func      (dec_func),
        .operand_a (rf_rd_data_a),
        .operand_b (rf_rd_data_b),
        .operand_c (rf_rd_data_c),
        .lane_mask (active_mask),
        .valid_out (mul_valid_out),
        .result    (mul_result)
    );

    //------------------------------------------------------------------------
    // 共享内存
    //------------------------------------------------------------------------
    shared_memory u_shared_mem (
        .clk         (clk),
        .rst_n       (rst_n),
        .req_valid   (smem_req_valid),
        .req_write   (smem_req_write),
        .req_addr    (smem_req_addr),
        .req_wdata   (smem_req_wdata),
        .req_mask    (active_mask),
        .resp_valid  (smem_resp_valid),
        .resp_rdata  (smem_resp_rdata),
        .bank_conflict()
    );

    //------------------------------------------------------------------------
    // 全局内存接口
    //------------------------------------------------------------------------
    memory_interface u_mem_if (
        .clk         (clk),
        .rst_n       (rst_n),
        .req_valid   (gmem_req_valid),
        .req_write   (gmem_req_write),
        .req_addr    (gmem_req_addr),
        .req_wdata   (gmem_req_wdata),
        .req_mask    (active_mask),
        .req_ready   (gmem_req_ready),
        .resp_valid  (gmem_resp_valid),
        .resp_rdata  (gmem_resp_rdata),
        // AXI signals
        .m_axi_awid  (m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen (m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata (m_axi_wdata),
        .m_axi_wstrb (m_axi_wstrb),
        .m_axi_wlast (m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid   (m_axi_bid),
        .m_axi_bresp (m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid  (m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen (m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid   (m_axi_rid),
        .m_axi_rdata (m_axi_rdata),
        .m_axi_rresp (m_axi_rresp),
        .m_axi_rlast (m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    //------------------------------------------------------------------------
    // 流水线控制
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_state      <= PIPE_IDLE;
            pc_reg          <= 32'b0;
            instruction_reg <= 32'b0;
            active_mask     <= {NUM_LANES{1'b1}};
            block_id_reg    <= 32'b0;
            block_dim_reg   <= 32'b0;
            grid_dim_reg    <= 32'b0;
        end else begin
            case (pipe_state)
                PIPE_IDLE: begin
                    if (kernel_start) begin
                        pc_reg       <= kernel_pc;
                        block_id_reg <= block_id_x;
                        block_dim_reg<= block_dim_x;
                        grid_dim_reg <= grid_dim_x;
                        active_mask  <= {NUM_LANES{1'b1}};
                        pipe_state   <= PIPE_FETCH;
                    end
                end

                PIPE_FETCH: begin
                    if (warp_selected) begin
                        pc_reg <= warp_pc_flat[active_warp_id*32 +: 32];
                        if (imem_valid) begin
                            instruction_reg <= imem_data;
                            pipe_state <= PIPE_DECODE;
                        end
                    end
                end

                PIPE_DECODE: begin
                    if (decode_valid) begin
                        pipe_state <= PIPE_EXEC;
                    end
                end

                PIPE_EXEC: begin
                    if (dec_exit_op) begin
                        // EXIT指令：跳过MEM/WB，直接回到FETCH（或IDLE）
                        if (warp_valid == (1 << active_warp_id)) begin
                            // 这是最后一个有效的Warp，将退出后进入IDLE
                            pipe_state <= PIPE_IDLE;
                        end else begin
                            pipe_state <= PIPE_FETCH;
                        end
                    end else if (dec_mem_read || dec_mem_write) begin
                        pipe_state <= PIPE_MEM;
                    end else begin
                        pipe_state <= PIPE_WB;
                    end
                end

                PIPE_MEM: begin
                    if (dec_mem_shared) begin
                        if (smem_resp_valid || !dec_mem_read) begin
                            pipe_state <= PIPE_WB;
                        end
                    end else begin
                        if (gmem_resp_valid || !dec_mem_read) begin
                            pipe_state <= PIPE_WB;
                        end
                    end
                end

                PIPE_WB: begin
                    // 检查是否还有有效的Warp
                    if (warp_valid == 0) begin
                        pipe_state <= PIPE_IDLE;
                    end else begin
                        pipe_state <= PIPE_FETCH;
                    end
                end
            endcase
        end
    end

    //------------------------------------------------------------------------
    // 指令内存请求
    //------------------------------------------------------------------------
    assign imem_req  = (pipe_state == PIPE_FETCH);
    assign imem_addr = pc_reg;

    //------------------------------------------------------------------------
    // 内存请求生成
    //------------------------------------------------------------------------
    assign smem_req_valid = dec_mem_shared && (dec_mem_read || dec_mem_write) &&
                            (pipe_state == PIPE_MEM);
    assign smem_req_write = dec_mem_write;
    assign smem_req_addr  = rf_rd_data_a[NUM_LANES*14-1:0];  // 地址在RA
    assign smem_req_wdata = rf_rd_data_b;                     // 数据在RB

    assign gmem_req_valid = !dec_mem_shared && (dec_mem_read || dec_mem_write) &&
                            (pipe_state == PIPE_MEM);
    assign gmem_req_write = dec_mem_write;
    assign gmem_req_addr  = rf_rd_data_a;
    assign gmem_req_wdata = rf_rd_data_b;

    //------------------------------------------------------------------------
    // 写回选择
    //------------------------------------------------------------------------
    assign rf_wr_en = dec_reg_write && (pipe_state == PIPE_WB);
    assign rf_wr_mask = active_mask;

    // 结果多路选择
    assign rf_wr_data = dec_mul_op     ? mul_result :
                        dec_mem_read   ? (dec_mem_shared ? smem_resp_rdata : gmem_resp_rdata) :
                        dec_special_reg? generate_special_reg(1'b0) :
                                         alu_result;

    // 特殊寄存器值生成
    function [NUM_LANES*DATA_WIDTH-1:0] generate_special_reg;
        input dummy;  // Verilog-2001要求函数至少有一个输入
        integer t;
        reg [DATA_WIDTH-1:0] val;
        begin
            for (t = 0; t < NUM_LANES; t = t + 1) begin
                case (dec_ra)  // 特殊寄存器ID在RA字段
                    `SREG_TID_X:    val = t;  // 线程ID
                    `SREG_CTAID_X:  val = block_id_reg;
                    `SREG_NTID_X:   val = block_dim_reg;
                    `SREG_NCTAID_X: val = grid_dim_reg;
                    default:        val = 32'b0;
                endcase
                generate_special_reg[t*DATA_WIDTH +: DATA_WIDTH] = val;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Kernel完成检测
    //------------------------------------------------------------------------
    assign kernel_done = (pipe_state == PIPE_IDLE) && !kernel_start && (warp_valid == 0);

endmodule
