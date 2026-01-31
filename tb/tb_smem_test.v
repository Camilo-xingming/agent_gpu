//============================================================================
// RalphGPU - Shared Memory Test
// 测试共享内存的基本功能: ST.SHARED 和 LD.SHARED
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_smem_test;

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
    reg  [63:0] imem_data;
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
            // Return 2 words for 8-byte cache line (64-bit response)
            imem_data <= {instruction_mem[imem_addr[11:2] + 1], instruction_mem[imem_addr[11:2]]};
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
                data_memory[m_axi_awaddr[13:2]] <= m_axi_wdata;
            end
            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 0;
            end
            // 读响应
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= data_memory[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1;
                m_axi_rlast <= 1;
            end
            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
                m_axi_rlast <= 0;
            end
        end
    end

    //------------------------------------------------------------------------
    // CSR写任务
    //------------------------------------------------------------------------
    task csr_write(input [11:0] addr, input [31:0] data);
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
    integer timeout_counter;
    reg [31:0] result;

    //------------------------------------------------------------------------
    // 测试程序
    //------------------------------------------------------------------------
    initial begin
        // 初始化
        csr_wr_en   = 0;
        csr_addr    = 0;
        csr_wr_data = 0;

        // 清空指令内存和数据内存
        for (i = 0; i < 1024; i = i + 1) begin
            instruction_mem[i] = 32'h00000000;
        end
        for (i = 0; i < 4096; i = i + 1) begin
            data_memory[i] = 32'h00000000;
        end

        // ============================================
        // 简单 Shared Memory 测试程序
        // ============================================
        // 测试: store 0x1234 到 shared memory, 然后 load 回来
        //       将 load 的结果存到 global memory 地址 0x2000
        
        // [0] MOV_IMM R1, 0x1234 (测试数据)
        instruction_mem[0] = {`OP_MOV_IMM, 5'd1, 21'h00001234};
        
        // [1] MOV_IMM R10, 0 (shared mem addr = 0)
        instruction_mem[1] = {`OP_MOV_IMM, 5'd10, 21'h00000000};
        
        // [2] ST.SHARED [R10], R1 (store 0x1234 to smem[0])
        instruction_mem[2] = {`OP_ST_SHARED, 5'd0, 5'd10, 5'd1, 11'b0};
        
        // [3-6] NOP (等待store完成)
        instruction_mem[3] = 32'hFC000000;  // NOP
        instruction_mem[4] = 32'hFC000000;  // NOP
        instruction_mem[5] = 32'hFC000000;  // NOP
        instruction_mem[6] = 32'hFC000000;  // NOP
        
        // [7] LD.SHARED R3, [R10] (load from smem[0])
        instruction_mem[7] = {`OP_LD_SHARED, 5'd3, 5'd10, 5'd0, 11'b0};
        
        // [8-11] NOP (等待load完成)
        instruction_mem[8] = 32'hFC000000;  // NOP
        instruction_mem[9] = 32'hFC000000;  // NOP
        instruction_mem[10] = 32'hFC000000;  // NOP
        instruction_mem[11] = 32'hFC000000;  // NOP
        
        // [12] MOV_IMM R20, 0x2000 (global addr for result)
        instruction_mem[12] = {`OP_MOV_IMM, 5'd20, 21'h00002000};
        
        // [13] ST.GLOBAL [R20], R3 (将从shared memory读到的值存到global)
        instruction_mem[13] = {`OP_ST_GLOBAL, 5'd0, 5'd20, 5'd3, 11'b0};
        
        // [14] NOP (等待store完成)
        instruction_mem[14] = 32'hFC000000;  // NOP
        
        // [15] EXIT
        instruction_mem[15] = {`OP_EXIT, 26'd0};

        // 等待复位
        wait(rst_n);
        #200;

        $display("========================================");
        $display("Shared Memory Test Start");
        $display("========================================");

        // 打印加载的指令
        $display("Loaded instructions:");
        for (i = 0; i < 16; i = i + 1) begin
            $display("  [%0d] 0x%08x", i, instruction_mem[i]);
        end

        // 配置Kernel
        $display("Configuring Kernel...");
        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1
        csr_write(12'h018, 32'h0000_0020);   // BLOCK_DIM_X = 32 (1 warp)

        // 启动Kernel
        $display("Starting Kernel...");
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        // 等待完成 (带超时)
        $display("Waiting for Kernel completion...");
        timeout_counter = 0;
        while (!irq_kernel_done && timeout_counter < 5000) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
            // 每100周期打印一次状态
            if (timeout_counter % 100 == 0) begin
                $display("[%0t] cycle=%0d imem_req=%b imem_addr=0x%08x imem_valid=%b kernel_done=%b",
                         $time, timeout_counter, imem_req, imem_addr, imem_valid, irq_kernel_done);
            end
        end

        if (irq_kernel_done) begin
            $display("Kernel completed in %0d cycles", timeout_counter);
            
            // 检查结果
            result = data_memory[32'h2000 >> 2];
            $display("Result at 0x2000: 0x%08x", result);
            
            if (result == 32'h00001234) begin
                $display("[PASS] Shared memory test passed!");
            end else begin
                $display("[FAIL] Expected 0x1234, got 0x%08x", result);
            end
        end else begin
            $display("[TIMEOUT] Kernel did not complete in %0d cycles", timeout_counter);
        end

        #1000;
        $display("Test completed.");
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_smem_test.vcd");
        $dumpvars(0, tb_smem_test);
    end

    // 超时保护
    initial begin
        #500000;
        $display("TIMEOUT - Simulation exceeded time limit");
        $finish;
    end

    //------------------------------------------------------------------------
    // 调试监控：观察内部状态
    //------------------------------------------------------------------------
    // 监控 SM 内部状态 (每个周期打印)
    reg [31:0] dbg_cycle;
    initial dbg_cycle = 0;
    always @(posedge clk) begin
        if (rst_n) begin
            dbg_cycle <= dbg_cycle + 1;
            // 打印前500个周期的详细信息
            if (dbg_cycle < 500 && dbg_cycle > 20) begin
                $display("[CYCLE %0d] imem_req=%b addr=0x%04x valid=%b kernel_done=%b",
                         dbg_cycle, imem_req, imem_addr, imem_valid, irq_kernel_done);
            end
        end
    end

    // 监控内部 kernel_start 信号
    wire sm0_kernel_start = dut.sm_kernel_start[0];
    wire [3:0] sm0_warp_valid = dut.sm_gen[0].u_sm.warp_valid;
    always @(posedge clk) begin
        if (sm0_kernel_start) begin
            $display("[TB DEBUG] SM0 kernel_start pulse detected!");
        end
        if (dbg_cycle > 20 && dbg_cycle < 100) begin
            $display("[TB DEBUG cycle %0d] sm0_kernel_start=%b sm0_warp_valid=0x%01x",
                     dbg_cycle, sm0_kernel_start, sm0_warp_valid);
        end
    end

endmodule
