//============================================================================
// RalphGPU - Multi-SM Parallel Execution Test
// 验证多个SM的并行调度和AXI仲裁
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_multi_sm;

    //------------------------------------------------------------------------
    // 参数
    //------------------------------------------------------------------------
    localparam NUM_SM = `NUM_SM;  // 2

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
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
        .NUM_SM (NUM_SM),
        .L1D_BYPASS(0)    // Enable full L1D cache mode
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
    // 指令内存模型
    //------------------------------------------------------------------------
    reg [31:0] instruction_mem [0:255];

    always @(posedge clk) begin
        imem_valid <= imem_req;
        if (imem_req) begin
            // Return 2 words for 8-byte cache line (64-bit response)
            imem_data <= {instruction_mem[imem_addr[9:2] + 1], instruction_mem[imem_addr[9:2]]};
        end
    end

    //------------------------------------------------------------------------
    // 数据内存模型
    //------------------------------------------------------------------------
    reg [31:0] data_memory [0:16383];
    reg [31:0] pending_write_addr;

    // 统计
    integer read_count;
    integer write_count;
    integer sm0_accesses;
    integer sm1_accesses;

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
            read_count    <= 0;
            write_count   <= 0;
            sm0_accesses  <= 0;
            sm1_accesses  <= 0;
        end else begin
            // 保存写地址
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            // 写响应
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1;
                m_axi_bid    <= m_axi_awid;
                data_memory[pending_write_addr[15:2]] <= m_axi_wdata;
                write_count <= write_count + 1;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 0;
            end

            // 读响应
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rvalid <= 1;
                m_axi_rid    <= m_axi_arid;
                m_axi_rdata  <= data_memory[m_axi_araddr[15:2]];
                m_axi_rlast  <= 1;
                read_count <= read_count + 1;

                // 统计SM访问 (根据AXI ID)
                if (m_axi_arid[3] == 1'b0) sm0_accesses <= sm0_accesses + 1;
                if (m_axi_arid[3] == 1'b1) sm1_accesses <= sm1_accesses + 1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
            end
        end
    end

    //------------------------------------------------------------------------
    // 辅助函数
    //------------------------------------------------------------------------
    function [31:0] make_inst;
        input [5:0] op;
        input [4:0] rd, ra, rb, rc;
        input [5:0] fn;
        begin
            make_inst = {op, rd, ra, rb, rc, fn};
        end
    endfunction

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer i;
    integer k;          // 用于连续kernel测试
    integer passed = 0;
    integer failed = 0;
    reg     all_ok;     // 用于连续kernel测试

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

    task wait_gpu_idle;
        integer wait_i;
        begin : wait_idle
            for (wait_i = 0; wait_i < 1000; wait_i = wait_i + 1) begin
                @(posedge clk);
                if (!dut.gpu_busy) begin
                    disable wait_idle;
                end
            end
            $display("[FAIL] GPU idle wait timeout");
            failed = failed + 1;
        end
    endtask

    task prepare_kernel;
        begin
            csr_write(12'h000, 32'h0000_0000);
            wait_gpu_idle();
        end
    endtask

    //------------------------------------------------------------------------
    // 测试程序
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Multi-SM Parallel Execution Test");
        $display("Number of SMs: %0d", NUM_SM);
        $display("============================================================");

        // 初始化
        rst_n       = 0;
        csr_wr_en   = 0;
        csr_addr    = 0;
        csr_wr_data = 0;

        // 加载简单kernel (以EXIT结束)
        for (i = 0; i < 15; i = i + 1) begin
            instruction_mem[i] = make_inst(`OP_NOP, 5'd0, 5'd0, 5'd0, 5'd0, 6'd0);
        end
        // 最后一条是EXIT指令
        instruction_mem[15] = make_inst(`OP_EXIT, 5'd0, 5'd0, 5'd0, 5'd0, 6'd0);

        // 初始化数据
        for (i = 0; i < 256; i = i + 1) begin
            data_memory[i] = i;
        end

        #100;
        rst_n = 1;
        #200;

        //====================================================================
        // 测试1: 多Block分配 (2个Block给2个SM)
        //====================================================================
        $display("\n--- Multi-Block Dispatch Test ---");

        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC = 0
        csr_write(12'h00C, 32'h0000_0004);   // GRID_DIM_X = 4 (4个Block)
        csr_write(12'h010, 32'h0000_0001);   // GRID_DIM_Y = 1
        csr_write(12'h014, 32'h0000_0001);   // GRID_DIM_Z = 1
        csr_write(12'h018, 32'h0000_0020);   // BLOCK_DIM_X = 32

        $display("  Launching kernel with 4 blocks on %0d SMs", NUM_SM);
        $display("  Expected: Each SM handles ~%0d blocks", 4/NUM_SM);

        prepare_kernel();
        csr_write(12'h004, 32'h0000_0001);   // Start

        // 等待完成
        fork: wait_multiblock
            begin
                wait(irq_kernel_done);
                disable wait_multiblock;
            end
            begin
                #100000;
                disable wait_multiblock;
            end
        join

        if (irq_kernel_done) begin
            $display("[PASS] Multi-block kernel completed");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Multi-block kernel timeout");
            failed = failed + 1;
        end

        //====================================================================
        // 测试2: 调度状态机验证
        //====================================================================
        $display("\n--- Scheduler State Test ---");

        @(posedge clk);
        csr_addr <= 12'h000;  // GPU_STATUS
        @(posedge clk);
        @(posedge clk);

        if (!csr_rd_data[1]) begin  // busy = 0
            $display("[PASS] GPU not busy after completion");
            passed = passed + 1;
        end else begin
            $display("[FAIL] GPU still busy after completion");
            failed = failed + 1;
        end

        //====================================================================
        // 测试3: 指令获取仲裁
        //====================================================================
        $display("\n--- Instruction Fetch Arbitration Test ---");

        // 重新启动测试
        prepare_kernel();
        csr_write(12'h004, 32'h0000_0001);

        // 监控指令获取
        begin
            integer fetch_count;
            fetch_count = 0;

            repeat(100) begin
                @(posedge clk);
                if (imem_req) fetch_count = fetch_count + 1;
            end

            $display("  Instruction fetches in 100 cycles: %0d", fetch_count);

            if (fetch_count > 0) begin
                $display("[PASS] Instruction fetch working");
                passed = passed + 1;
            end else begin
                $display("[FAIL] No instruction fetches");
                failed = failed + 1;
            end
        end

        // 等待完成
        #50000;

        //====================================================================
        // 测试4: 单Block执行 (只用一个SM)
        //====================================================================
        $display("\n--- Single Block Test ---");

        prepare_kernel();
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X = 1 (1个Block)
        csr_write(12'h004, 32'h0000_0001);   // Start

        fork: wait_single
            begin
                wait(irq_kernel_done);
                disable wait_single;
            end
            begin
                #50000;
                disable wait_single;
            end
        join

        if (irq_kernel_done) begin
            $display("[PASS] Single block execution completed");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Single block execution timeout");
            failed = failed + 1;
        end

        //====================================================================
        // 测试5: 大Grid测试
        //====================================================================
        $display("\n--- Large Grid Test ---");

        prepare_kernel();
        csr_write(12'h00C, 32'h0000_0008);   // GRID_DIM_X = 8
        csr_write(12'h004, 32'h0000_0001);   // Start

        fork: wait_large
            begin
                wait(irq_kernel_done);
                disable wait_large;
            end
            begin
                #200000;
                disable wait_large;
            end
        join

        if (irq_kernel_done) begin
            $display("[PASS] Large grid (8 blocks) completed");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Large grid timeout");
            failed = failed + 1;
        end

        //====================================================================
        // 测试6: 连续Kernel启动
        //====================================================================
        $display("\n--- Back-to-Back Kernel Test ---");

        begin
            all_ok = 1;

            csr_write(12'h00C, 32'h0000_0002);   // GRID_DIM_X = 2

            for (k = 0; k < 3; k = k + 1) begin
                prepare_kernel();
                csr_write(12'h004, 32'h0000_0001);   // Start

                fork: wait_back2back
                    begin
                        wait(irq_kernel_done);
                        disable wait_back2back;
                    end
                    begin
                        #30000;
                        all_ok = 0;
                        disable wait_back2back;
                    end
                join

                if (!irq_kernel_done) all_ok = 0;

                // 清除中断
                csr_write(12'h000, 32'h0000_0000);
                #100;
            end

            if (all_ok) begin
                $display("[PASS] 3 consecutive kernels completed");
                passed = passed + 1;
            end else begin
                $display("[FAIL] Consecutive kernel execution failed");
                failed = failed + 1;
            end
        end

        //====================================================================
        // Test 7: 3D Grid Block ID Test (2x2x1)
        //====================================================================
        $display("\n--- 3D Grid Block ID Test ---");

        prepare_kernel();
        csr_write(12'h00C, 32'h0000_0002);   // GRID_DIM_X = 2
        csr_write(12'h010, 32'h0000_0002);   // GRID_DIM_Y = 2
        csr_write(12'h014, 32'h0000_0001);   // GRID_DIM_Z = 1
        csr_write(12'h004, 32'h0000_0001);   // Start

        // Check block IDs after dispatch
        #200;
        $display("  SM0: block_id=(%0d,%0d,%0d)", dut.sm_block_id_x[0],
                 dut.sm_block_id_y[0], dut.sm_block_id_z[0]);
        $display("  SM1: block_id=(%0d,%0d,%0d)", dut.sm_block_id_x[1],
                 dut.sm_block_id_y[1], dut.sm_block_id_z[1]);

        // Verify 3D decomposition: for grid 2x2x1, block_id_y should be non-zero
        // for blocks 2,3. The NOP kernel is fast so we may see any dispatch round.
        // Key check: x values differ, y values match within a dispatch round.
        if (dut.sm_block_id_x[0] != dut.sm_block_id_x[1] &&
            dut.sm_block_id_y[0] == dut.sm_block_id_y[1]) begin
            $display("[PASS] 3D grid block IDs correct (x differs, y matches in round)");
            passed = passed + 1;
        end else begin
            $display("[FAIL] 3D grid block IDs unexpected: SM0=(%0d,%0d,%0d) SM1=(%0d,%0d,%0d)",
                     dut.sm_block_id_x[0], dut.sm_block_id_y[0], dut.sm_block_id_z[0],
                     dut.sm_block_id_x[1], dut.sm_block_id_y[1], dut.sm_block_id_z[1]);
            failed = failed + 1;
        end

        fork: wait_3d
            begin
                wait(irq_kernel_done);
                disable wait_3d;
            end
            begin
                #100000;
                disable wait_3d;
            end
        join

        if (irq_kernel_done) begin
            $display("[PASS] 3D grid kernel completed");
            passed = passed + 1;
        end else begin
            $display("[FAIL] 3D grid kernel timeout");
            failed = failed + 1;
        end

        // Reset grid dims
        csr_write(12'h010, 32'h0000_0001);   // GRID_DIM_Y = 1
        csr_write(12'h014, 32'h0000_0001);   // GRID_DIM_Z = 1


        //====================================================================
        $display("\n--- Memory Access Statistics ---");
        $display("  Total reads:  %0d", read_count);
        $display("  Total writes: %0d", write_count);
        $display("  SM0 accesses: %0d", sm0_accesses);
        $display("  SM1 accesses: %0d", sm1_accesses);

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Multi-SM Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL MULTI-SM TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_multi_sm.vcd");
        $dumpvars(0, tb_multi_sm);
    end

`ifdef DEBUG_MULTI_SM
    reg [1:0] dbg_state;
    reg [NUM_SM-1:0] dbg_sm_start;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_state <= 0;
            dbg_sm_start <= 0;
        end else begin
            if (dut.sched_state != dbg_state) begin
                $display("[DBG] t=%0t state=%0d->%0d busy=%b disp=%0d total=%0d",
                         $time, dbg_state, dut.sched_state, dut.sm_busy,
                         dut.dispatched_blocks, dut.total_blocks);
                dbg_state <= dut.sched_state;
            end
            if (dut.sm_kernel_start != 0 && dbg_sm_start != dut.sm_kernel_start) begin
                $display("[DBG] t=%0t sm_kernel_start=%b block_id0=%0d block_id1=%0d",
                         $time, dut.sm_kernel_start, dut.sm_block_id_x[0],
                         dut.sm_block_id_x[1]);
            end
            if (irq_kernel_done) begin
                $display("[DBG] t=%0t irq_kernel_done=1", $time);
            end
            dbg_sm_start <= dut.sm_kernel_start;
        end
    end
`endif

    // 超时保护
    initial begin
        #500000;
        $display("ERROR: Global Timeout!");
        $finish;
    end

endmodule
