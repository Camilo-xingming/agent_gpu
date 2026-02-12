//============================================================================
// RalphGPU - Vector Addition Integration Test
// 完整的向量加法kernel执行测试
// 验证: 特殊寄存器读取、ALU运算、全局内存读写
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_perf_counters;

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
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
    // 指令内存模型 - load from hex file
    //------------------------------------------------------------------------
    reg [31:0] imem [0:255];
    integer instr_count;

    initial begin
        // Initialize with NOPs/EXIT
        for (integer j = 0; j < 256; j = j + 1) begin
            imem[j] = 32'hFC000000;  // NOP opcode
        end
        // Load program from hex file
        $readmemh("vector_add.hex", imem);
        // Count instructions
        instr_count = 0;
        for (integer j = 0; j < 256; j = j + 1) begin
            if (imem[j] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from vector_add.hex", instr_count);
    end

    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // 数据内存模型 (简化AXI从机)
    //------------------------------------------------------------------------
    reg [31:0] data_memory [0:8191];  // 32KB

    // 内存地址映射:
    // 0x0000 - 0x007F: 向量A (32个元素)
    // 0x1000 - 0x107F: 向量B (32个元素)
    // 0x2000 - 0x207F: 向量C (32个元素) - 输出

    reg [31:0] pending_write_addr;

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
            // 保存写地址
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            // 写响应
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1;
                m_axi_bid    <= m_axi_awid;
                // 写入数据
                data_memory[pending_write_addr[14:2]] <= m_axi_wdata;
                $display("[MEM] Write: addr=0x%08X, data=0x%08X",
                         pending_write_addr, m_axi_wdata);
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 0;
            end

            // 读响应
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rvalid <= 1;
                m_axi_rid    <= m_axi_arid;
                m_axi_rdata  <= data_memory[m_axi_araddr[14:2]];
                m_axi_rlast  <= 1;
                $display("[MEM] Read: addr=0x%08X, data=0x%08X",
                         m_axi_araddr, data_memory[m_axi_araddr[14:2]]);
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
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
    integer passed = 0;
    integer failed = 0;

    //------------------------------------------------------------------------
    // 测试程序
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Vector Addition Integration Test");
        $display("============================================================");
        $display("Kernel: C[i] = A[i] + B[i]");
        $display("============================================================");

        // 初始化
        rst_n       = 0;
        csr_wr_en   = 0;
        csr_addr    = 0;
        csr_wr_data = 0;

        //--------------------------------------------------------------------
        // 初始化数据内存
        // Memory map (matching vector_add.ptx):
        // - A array: 0x0000 (word index 0-31)
        // - B array: 0x1000 (word index 1024-1055)
        // - C array: 0x2000 (word index 2048-2079)
        //--------------------------------------------------------------------
        $display("\nInitializing data memory...");
        for (i = 0; i < 32; i = i + 1) begin
            data_memory[i]          = i * 10;        // A[i] = i * 10
            data_memory[1024 + i]   = i * 5;         // B[i] = i * 5
            data_memory[2048 + i]   = 32'hDEADBEEF;  // C[i] = 初始化为无效值
        end
        $display("  A[0..3] = %0d, %0d, %0d, %0d",
                 data_memory[0], data_memory[1], data_memory[2], data_memory[3]);
        $display("  B[0..3] = %0d, %0d, %0d, %0d",
                 data_memory[1024], data_memory[1025], data_memory[1026], data_memory[1027]);

        //--------------------------------------------------------------------
        // 复位释放
        //--------------------------------------------------------------------
        #100;
        rst_n = 1;
        #200;

        //--------------------------------------------------------------------
        // 配置Kernel
        //--------------------------------------------------------------------
        $display("\n--- Configuring Kernel ---");

        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1
        csr_write(12'h010, 32'h0000_0001);   // GRID_DIM_Y = 1
        csr_write(12'h014, 32'h0000_0001);   // GRID_DIM_Z = 1
        csr_write(12'h018, 32'h0000_0020);   // BLOCK_DIM_X = 32

        $display("  KERNEL_PC   = 0x%08X", 32'h0);
        $display("  GRID_DIM    = (1, 1, 1)");
        $display("  BLOCK_DIM   = (32, 1, 1)");

        //--------------------------------------------------------------------
        // 启动Kernel
        //--------------------------------------------------------------------
        $display("\n--- Starting Kernel ---");
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        //--------------------------------------------------------------------
        // 等待完成或超时
        //--------------------------------------------------------------------
        $display("Waiting for kernel completion...");

        // Verilog-2001 兼容的超时等待
        fork: wait_kernel
            begin
                wait(irq_kernel_done);
                $display("\n--- Kernel Completed ---");
                disable wait_kernel;
            end
            begin
                #100000;
                $display("\n--- Timeout waiting for kernel ---");
                disable wait_kernel;
            end
        join

        // Wait for memory stores to drain (kernel completes before all stores finish)
        $display("Waiting for stores to complete...");
        #5000;

        //--------------------------------------------------------------------
        // 验证结果
        //--------------------------------------------------------------------
        $display("\n--- Verifying Results ---");
        $display("Expected: C[i] = A[i] + B[i] = i*10 + i*5 = i*15\n");

        // Check all 32 thread results
        for (i = 0; i < 32; i = i + 1) begin
            if (data_memory[2048 + i] == data_memory[i] + data_memory[1024 + i]) begin
                if (i < 8) $display("  C[%0d] = %0d (expected %0d) [PASS]",
                         i, data_memory[2048 + i],
                         data_memory[i] + data_memory[1024 + i]);
                passed = passed + 1;
            end else begin
                $display("  C[%0d] = %0d (expected %0d) [FAIL]",
                         i, data_memory[2048 + i],
                         data_memory[i] + data_memory[1024 + i]);
                failed = failed + 1;
            end
        end
        $display("  Vector add: %0d/32 correct", passed);

        //--------------------------------------------------------------------
        // CSR读取测试
        //--------------------------------------------------------------------
        $display("\n--- CSR Read Test ---");

        @(posedge clk);
        csr_addr <= 12'h000;  // GPU_STATUS
        @(posedge clk);
        @(posedge clk);
        $display("  GPU_STATUS = 0x%08X (busy=%b, ready=%b)",
                 csr_rd_data, csr_rd_data[1], csr_rd_data[0]);

        if (csr_rd_data[0] === 1'b1) begin
            $display("[PASS] GPU ready bit is set");
            passed = passed + 1;
        end else begin
            $display("[FAIL] GPU ready bit not set");
            failed = failed + 1;
        end

        //--------------------------------------------------------------------
        // 中断状态测试
        //--------------------------------------------------------------------
        $display("\n--- Interrupt Test ---");

        if (irq_kernel_done === 1'b1) begin
            $display("[PASS] Kernel done interrupt asserted");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Kernel done interrupt not asserted");
            failed = failed + 1;
        end

        //--------------------------------------------------------------------
        // 测试总结
        //--------------------------------------------------------------------
        #100;
        $display("\n============================================================");
        $display("Vector Addition Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** INTEGRATION TEST PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end


        // ============================================================
        // RALPH-7 P2: Read Performance Counters via CSR
        // ============================================================
        $display("\n============================================================");
        $display("Performance Counters (via CSR 0x100-0x13F):");
        $display("============================================================");
        
        // Read counter function: set csr_addr, read csr_rd_data
        // Counter select = csr_addr[5:0], base = 0x100
        
        // CTR_CYCLES (0)
        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        $display("  Cycles                   = %0d", csr_rd_data);
        
        // CTR_INSTRUCTIONS (1)
        @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
        $display("  Instructions             = %0d", csr_rd_data);
        
        // CTR_DUAL_ISSUED (2)
        @(posedge clk); csr_addr <= 12'h102; @(posedge clk); @(posedge clk);
        $display("  Dual Issued              = %0d", csr_rd_data);
        
        // CTR_STALL_SCOREBOARD (3)
        @(posedge clk); csr_addr <= 12'h103; @(posedge clk); @(posedge clk);
        $display("  Stall: Scoreboard        = %0d", csr_rd_data);
        
        // CTR_STALL_IFETCH (4)
        @(posedge clk); csr_addr <= 12'h104; @(posedge clk); @(posedge clk);
        $display("  Stall: I-Fetch           = %0d", csr_rd_data);
        
        // CTR_STALL_MEM (5)
        @(posedge clk); csr_addr <= 12'h105; @(posedge clk); @(posedge clk);
        $display("  Stall: Memory            = %0d", csr_rd_data);
        
        // CTR_ALU_CYCLES (8)
        @(posedge clk); csr_addr <= 12'h108; @(posedge clk); @(posedge clk);
        $display("  FU: ALU Active           = %0d", csr_rd_data);
        
        // CTR_FPU_CYCLES (9)
        @(posedge clk); csr_addr <= 12'h109; @(posedge clk); @(posedge clk);
        $display("  FU: FPU Active           = %0d", csr_rd_data);
        
        // CTR_LDST_CYCLES (12)
        @(posedge clk); csr_addr <= 12'h10C; @(posedge clk); @(posedge clk);
        $display("  FU: LDST Active          = %0d", csr_rd_data);
        
        // CTR_BRANCH_TAKEN (24)
        @(posedge clk); csr_addr <= 12'h118; @(posedge clk); @(posedge clk);
        $display("  Branch Taken             = %0d", csr_rd_data);
        
        // CTR_BRANCH_DIVERGENT (25)
        @(posedge clk); csr_addr <= 12'h119; @(posedge clk); @(posedge clk);
        $display("  Branch Divergent         = %0d", csr_rd_data);
        
        // Sanity: Cycles and Instructions should be > 0
        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        if (csr_rd_data > 0)
            $display("\n[PASS] Cycles > 0 (%0d)", csr_rd_data);
        else
            $display("\n[FAIL] Cycles = 0 (counters not working!)");
            
        @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
        if (csr_rd_data > 0)
            $display("[PASS] Instructions > 0 (%0d)", csr_rd_data);
        else
            $display("[FAIL] Instructions = 0");
        
        // IPC
        begin
            reg [31:0] c_val, i_val;
            @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
            c_val = csr_rd_data;
            @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
            i_val = csr_rd_data;
            if (c_val > 0)
                $display("\nIPC = %0d / %0d = %f", i_val, c_val,
                         $itor(i_val) / $itor(c_val));
        end
        
        $display("============================================================");

        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_perf_counters.vcd");
        $dumpvars(0, tb_perf_counters);
    end

    //------------------------------------------------------------------------
    // 调试监控
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        // Print on response cycle (when imem_valid is high), not request cycle
        // Non-blocking assignments update after all blocks execute
        if (imem_valid) begin
            $display("[FETCH] Inst=0x%016X (lo=0x%08X, hi=0x%08X)",
                     imem_data, imem_data[31:0], imem_data[63:32]);
        end
    end

    // 超时保护
    initial begin
        #200000;
        $display("ERROR: Global Timeout!");
        $finish;
    end

endmodule
