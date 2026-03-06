//============================================================================
// RalphGPU - Vector Addition Integration Test
// 完整的向量加法kernel执行测试
// 验证: 特殊寄存器读取、ALU运算、全局内存读写
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_vector_add;

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
    ralph_gpu_top #(
        .L1D_BYPASS(0)
    ) dut (
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
    reg [31:0] pending_read_addr;
    reg [7:0]  pending_read_beats;
    reg [3:0]  pending_read_id;
    reg        read_active;

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
            m_axi_rid     <= 0;
            pending_read_addr  <= 0;
            pending_read_beats <= 0;
            pending_read_id    <= 0;
            read_active        <= 0;
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

            // 读响应（支持 burst）
            if (!read_active && m_axi_arvalid && m_axi_arready) begin
                read_active        <= 1'b1;
                m_axi_arready      <= 1'b0;
                pending_read_addr  <= m_axi_araddr;
                pending_read_beats <= m_axi_arlen + 1'b1;
                pending_read_id    <= m_axi_arid;

                m_axi_rvalid <= 1'b1;
                m_axi_rid    <= m_axi_arid;
                m_axi_rdata  <= data_memory[m_axi_araddr[14:2]];
                m_axi_rlast  <= (m_axi_arlen == 0);
                $display("[MEM] Read start: addr=0x%08X len=%0d data=0x%08X",
                         m_axi_araddr, m_axi_arlen, data_memory[m_axi_araddr[14:2]]);
            end else if (read_active && m_axi_rvalid && m_axi_rready) begin
                if (pending_read_beats <= 8'd1) begin
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast  <= 1'b0;
                    read_active  <= 1'b0;
                    m_axi_arready <= 1'b1;
                end else begin
                    pending_read_addr  <= pending_read_addr + 32'd4;
                    pending_read_beats <= pending_read_beats - 1'b1;

                    m_axi_rvalid <= 1'b1;
                    m_axi_rid    <= pending_read_id;
                    m_axi_rdata  <= data_memory[(pending_read_addr + 32'd4) >> 2];
                    m_axi_rlast  <= (pending_read_beats == 8'd2);
                end
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
    integer cycle_counter = 0;
    integer kernel_start_cycle = 0;
    integer kernel_end_cycle = 0;
    integer perf_cycles = 0;
    integer perf_instructions = 0;
    integer perf_ipc_x100 = 0;
    integer perf_occ_pct = 0;
    integer perf_active_warp_sum = 0;
    integer perf_active_warp_samples = 0;
    integer perf_avg_occ_pct = 0;
    integer perf_l1_hits = 0;
    integer perf_l1_misses = 0;
    integer perf_l2_hits = 0;
    integer perf_l2_misses = 0;
    real perf_l1_hit_rate;
    real perf_l2_hit_rate;

    //------------------------------------------------------------------------
    // Occupancy sampler (average active warps while kernel is running)
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_active_warp_sum <= 0;
            perf_active_warp_samples <= 0;
        end else if (dut.gpu_busy) begin
            perf_active_warp_sum <= perf_active_warp_sum + dut.sm_gen[0].u_sm.perf_active_warp_count;
            perf_active_warp_samples <= perf_active_warp_samples + 1;
        end
    end

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
        kernel_start_cycle = cycle_counter;
        csr_write(12'h004, 32'h0000_0001);   // GPU_CONTROL.start = 1

        //--------------------------------------------------------------------
        // 等待完成或超时
        //--------------------------------------------------------------------
        $display("Waiting for kernel completion...");

        // Verilog-2001 兼容的超时等待
        fork: wait_kernel
            begin
                wait(irq_kernel_done);
                kernel_end_cycle = cycle_counter;
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
        $display("Kernel cycles (start->done): %0d", kernel_end_cycle - kernel_start_cycle);
        $display("SM0 L1D stats: hits=%0d misses=%0d last_hit_latency=%0d last_miss_latency=%0d",
                 dut.sm_gen[0].u_sm.l1_stat_hits,
                 dut.sm_gen[0].u_sm.l1_stat_misses,
                 dut.sm_gen[0].u_sm.l1_last_hit_latency,
                 dut.sm_gen[0].u_sm.l1_last_miss_latency);

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
        // 性能可观测性断言 (#324)
        //--------------------------------------------------------------------
        $display("\n--- Performance Metrics ---");

        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        perf_cycles = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
        perf_instructions = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h110; @(posedge clk); @(posedge clk);
        perf_l1_hits = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h111; @(posedge clk); @(posedge clk);
        perf_l1_misses = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h112; @(posedge clk); @(posedge clk);
        perf_l2_hits = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h113; @(posedge clk); @(posedge clk);
        perf_l2_misses = csr_rd_data;

        if ((perf_l1_hits + perf_l1_misses) == 0) begin
            perf_l1_hits = dut.sm_gen[0].u_sm.l1_stat_hits;
            perf_l1_misses = dut.sm_gen[0].u_sm.l1_stat_misses;
        end
        perf_l1_hit_rate = (perf_l1_hits + perf_l1_misses > 0) ?
                           (($itor(perf_l1_hits) * 100.0) / $itor(perf_l1_hits + perf_l1_misses)) : 0.0;
        perf_l2_hit_rate = (perf_l2_hits + perf_l2_misses > 0) ?
                           (($itor(perf_l2_hits) * 100.0) / $itor(perf_l2_hits + perf_l2_misses)) : 0.0;

        perf_ipc_x100 = dut.perf_achieved_ipc_x100;
        perf_occ_pct = dut.perf_sm_occupancy_pct[7:0];
        if (perf_active_warp_samples > 0)
            perf_avg_occ_pct = (perf_active_warp_sum * 100) / (perf_active_warp_samples * 4);
        else
            perf_avg_occ_pct = 0;

        $display("  Cycles: %0d", perf_cycles);
        $display("  Instructions: %0d", perf_instructions);
        $display("  IPC: %0d.%02d", perf_ipc_x100 / 100, perf_ipc_x100 % 100);
        $display("  Occupancy(SM0) instant: %0d%%", perf_occ_pct);
        $display("  Occupancy(SM0) average: %0d%%", perf_avg_occ_pct);
        $display("  CacheStats: L1_hits=%0d L1_misses=%0d L1_hit_rate=%0.2f%% L2_hits=%0d L2_misses=%0d L2_hit_rate=%0.2f%%",
                 perf_l1_hits, perf_l1_misses, perf_l1_hit_rate,
                 perf_l2_hits, perf_l2_misses, perf_l2_hit_rate);

        if (perf_ipc_x100 > 0) begin
            $display("[PASS] IPC metric is non-zero");
            passed = passed + 1;
        end else begin
            $display("[FAIL] IPC metric is zero");
            failed = failed + 1;
        end
        if (perf_avg_occ_pct > 0) begin
            $display("[PASS] Occupancy metric is non-zero");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Occupancy metric is zero");
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

        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_vector_add.vcd");
        $dumpvars(0, tb_vector_add);
    end

    //------------------------------------------------------------------------
    // 调试监控
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n) cycle_counter <= cycle_counter + 1;
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
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
