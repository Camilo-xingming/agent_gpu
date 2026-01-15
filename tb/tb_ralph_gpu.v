//============================================================================
// RalphGPU Testbench
// 测试基本的GPU功能
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_ralph_gpu;

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
    end

    initial begin
        rst_n = 0;
        #100 rst_n = 1;
    end

    //------------------------------------------------------------------------
    // DUT接口
    //------------------------------------------------------------------------
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [31:0] imem_data;
    reg         imem_valid;

    // AXI4 接口
    wire [3:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready;
    reg  [3:0]  m_axi_bid;
    reg  [1:0]  m_axi_bresp;
    reg         m_axi_bvalid;
    wire        m_axi_bready;
    wire [3:0]  m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready;
    reg  [3:0]  m_axi_rid;
    reg  [31:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rlast;
    reg         m_axi_rvalid;
    wire        m_axi_rready;

    //------------------------------------------------------------------------
    // DUT实例化
    //------------------------------------------------------------------------
    ralph_gpu_top dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .csr_wr_en       (csr_wr_en),
        .csr_addr        (csr_addr),
        .csr_wr_data     (csr_wr_data),
        .csr_rd_data     (csr_rd_data),
        .irq_kernel_done (irq_kernel_done),
        .imem_req        (imem_req),
        .imem_addr       (imem_addr),
        .imem_data       (imem_data),
        .imem_valid      (imem_valid),
        .m_axi_awid      (m_axi_awid),
        .m_axi_awaddr    (m_axi_awaddr),
        .m_axi_awlen     (m_axi_awlen),
        .m_axi_awsize    (m_axi_awsize),
        .m_axi_awburst   (m_axi_awburst),
        .m_axi_awvalid   (m_axi_awvalid),
        .m_axi_awready   (m_axi_awready),
        .m_axi_wdata     (m_axi_wdata),
        .m_axi_wstrb     (m_axi_wstrb),
        .m_axi_wlast     (m_axi_wlast),
        .m_axi_wvalid    (m_axi_wvalid),
        .m_axi_wready    (m_axi_wready),
        .m_axi_bid       (m_axi_bid),
        .m_axi_bresp     (m_axi_bresp),
        .m_axi_bvalid    (m_axi_bvalid),
        .m_axi_bready    (m_axi_bready),
        .m_axi_arid      (m_axi_arid),
        .m_axi_araddr    (m_axi_araddr),
        .m_axi_arlen     (m_axi_arlen),
        .m_axi_arsize    (m_axi_arsize),
        .m_axi_arburst   (m_axi_arburst),
        .m_axi_arvalid   (m_axi_arvalid),
        .m_axi_arready   (m_axi_arready),
        .m_axi_rid       (m_axi_rid),
        .m_axi_rdata     (m_axi_rdata),
        .m_axi_rresp     (m_axi_rresp),
        .m_axi_rlast     (m_axi_rlast),
        .m_axi_rvalid    (m_axi_rvalid),
        .m_axi_rready    (m_axi_rready)
    );

    //------------------------------------------------------------------------
    // 指令内存模型
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:1023];

    always @(posedge clk) begin
        imem_valid <= imem_req;
        if (imem_req) begin
            imem_data <= instruction_mem[imem_addr[11:2]];
        end
    end

    //------------------------------------------------------------------------
    // 简单AXI从机模型 (内存)
    //------------------------------------------------------------------------
    reg [31:0] data_memory [0:4095];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1;
            m_axi_wready  <= 1;
            m_axi_bresp   <= 0;
            m_axi_bvalid  <= 0;
            m_axi_arready <= 1;
            m_axi_rvalid  <= 0;
            m_axi_rresp   <= 0;
            m_axi_rlast   <= 0;
        end else begin
            // 写响应
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1;
                m_axi_bid    <= m_axi_awid;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 0;
            end

            // 读响应
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rvalid <= 1;
                m_axi_rid    <= m_axi_arid;
                m_axi_rdata  <= data_memory[m_axi_araddr[13:2]];
                m_axi_rlast  <= 1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
            end

            // 写数据
            if (m_axi_wvalid && m_axi_wready) begin
                data_memory[m_axi_awaddr[13:2]] <= m_axi_wdata;
            end
        end
    end

    //------------------------------------------------------------------------
    // CSR写入任务
    //------------------------------------------------------------------------
    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr    <= addr;
            csr_wr_data <= data;
            csr_wr_en   <= 1;
            @(posedge clk);
            csr_wr_en   <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer i;

    //------------------------------------------------------------------------
    // 测试程序
    //------------------------------------------------------------------------
    initial begin
        // 初始化
        csr_wr_en   = 0;
        csr_addr    = 0;
        csr_wr_data = 0;

        // 加载测试Kernel
        // 简单的向量加法: C[i] = A[i] + B[i]
        // PTX伪代码:
        //   mov.u32 r0, %tid.x       // r0 = thread_id
        //   shl.b32 r1, r0, 2        // r1 = r0 * 4 (字节偏移)
        //   ld.global r2, [r1+0]     // r2 = A[i] (假设A在地址0)
        //   ld.global r3, [r1+4096]  // r3 = B[i] (假设B在地址4096)
        //   add.s32 r4, r2, r3       // r4 = r2 + r3
        //   st.global [r1+8192], r4  // C[i] = r4 (假设C在地址8192)

        // 指令编码:
        // OP_MOV_SPECIAL: opcode=001001, rd=0, ra=SREG_TID_X(0), rb=0, rc=0, func=0
        instruction_mem[0] = {6'b001001, 5'd0, 5'd0, 5'd0, 5'd0, 6'd0};

        // OP_ALU (SHL): opcode=000000, rd=1, ra=0, rb=2(立即数位置), func=FUNC_SHL
        instruction_mem[1] = {6'b000000, 5'd1, 5'd0, 5'd2, 5'd0, `FUNC_SHL};

        // 简化：使用NOP占位 (实际需要更完整的指令)
        instruction_mem[2] = {6'b111111, 26'd0};  // NOP
        instruction_mem[3] = {6'b111111, 26'd0};  // NOP
        // EXIT指令结束kernel
        instruction_mem[4] = {`OP_EXIT, 26'd0};   // EXIT

        // 初始化数据内存
        for (i = 0; i < 32; i = i + 1) begin
            data_memory[i]        = i;        // A[i] = i
            data_memory[1024 + i] = i * 2;    // B[i] = i * 2
        end

        // 等待复位
        wait(rst_n);
        #200;

        $display("========================================");
        $display("RalphGPU Testbench Start");
        $display("========================================");

        // 配置Kernel
        $display("Configuring Kernel...");
        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1
        csr_write(12'h018, 32'h0000_0020);   // BLOCK_DIM_X = 32

        // 启动Kernel
        $display("Starting Kernel...");
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        // 等待完成
        $display("Waiting for Kernel completion...");
        wait(irq_kernel_done);

        $display("========================================");
        $display("Kernel Done!");
        $display("========================================");

        // 检查结果
        $display("Checking results...");
        for (i = 0; i < 8; i = i + 1) begin
            $display("C[%0d] = %0d (expected: %0d)",
                     i, data_memory[2048 + i], i + i*2);
        end

        #1000;
        $display("Test completed.");
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_ralph_gpu.vcd");
        $dumpvars(0, tb_ralph_gpu);
    end

    // 超时保护
    initial begin
        #100000;
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule
