//============================================================================
// RalphGPU - Memory benchmark TB for MCU/L1D validation (Issue #306)
// Compile-time switches:
//   -DTB_MEM_BENCH_GATHER : run gather benchmark
//   -DTB_MEM_BENCH_SAXPY  : run SAXPY benchmark
//   -DTB_L1D_BYPASS=0/1   : top-level L1D bypass mode
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

`ifndef TB_L1D_BYPASS
`define TB_L1D_BYPASS 0
`endif

module tb_perf_mem_bench;

    localparam integer N_ELEMS = 32;

    // Gather benchmark map
    localparam integer G_A_BASE_IDX   = 0;      // 0x0000 / 4
    localparam integer G_B_BASE_IDX   = 1024;   // 0x1000 / 4
    localparam integer G_C_BASE_IDX   = 2048;   // 0x2000 / 4
    localparam integer G_IDX_BASE_IDX = 3072;   // 0x3000 / 4

    // Stream benchmark map (0x800 aligned segments)
    localparam integer S_A0_BASE_IDX = 0;       // 0x0000 / 4
    localparam integer S_A1_BASE_IDX = 512;     // 0x0800 / 4
    localparam integer S_A2_BASE_IDX = 1024;    // 0x1000 / 4
    localparam integer S_A3_BASE_IDX = 1536;    // 0x1800 / 4
    localparam integer S_B0_BASE_IDX = 2048;    // 0x2000 / 4
    localparam integer S_B1_BASE_IDX = 2560;    // 0x2800 / 4
    localparam integer S_B2_BASE_IDX = 3072;    // 0x3000 / 4
    localparam integer S_B3_BASE_IDX = 3584;    // 0x3800 / 4
    localparam integer S_C_BASE_IDX  = 4096;    // 0x4000 / 4

    // SAXPY benchmark map
    localparam integer SX_X_BASE_IDX = 0;       // 0x0000 / 4
    localparam integer SX_Y_BASE_IDX = 1024;    // 0x1000 / 4

    reg clk;
    reg rst_n;

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

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

    localparam integer TB_L1D_BYPASS_CFG = `TB_L1D_BYPASS;

    ralph_gpu_top #(
        .L1D_BYPASS(TB_L1D_BYPASS_CFG)
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

    reg [31:0] imem [0:255];
    reg [31:0] data_memory [0:8191];
    reg [31:0] pending_write_addr;
    reg [31:0] pending_read_addr;
    reg [7:0]  pending_read_beats;
    reg [3:0]  pending_read_id;
    reg        read_active;
    integer ii;
    integer jj;
    integer instr_count;

    initial begin
        for (jj = 0; jj < 256; jj = jj + 1) begin
            imem[jj] = 32'hFC000000;
        end
`ifdef TB_MEM_BENCH_GATHER
        $readmemh("mem_gather.hex", imem);
`elsif TB_MEM_BENCH_SAXPY
        $readmemh("saxpy.hex", imem);
`else
        $readmemh("mem_stream8.hex", imem);
`endif
        instr_count = 0;
        for (jj = 0; jj < 256; jj = jj + 1) begin
            if (imem[jj] != 32'hFC000000) instr_count = instr_count + 1;
        end
`ifdef TB_MEM_BENCH_GATHER
        $display("Loaded %0d instructions from mem_gather.hex", instr_count);
`elsif TB_MEM_BENCH_SAXPY
        $display("Loaded %0d instructions from saxpy.hex", instr_count);
`else
        $display("Loaded %0d instructions from mem_stream8.hex", instr_count);
`endif
    end

    always @(posedge clk) begin
        if (imem_req) begin
            imem_data  <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready  <= 1'b1;
            m_axi_bresp   <= 2'b00;
            m_axi_bvalid  <= 1'b0;
            m_axi_arready <= 1'b1;
            m_axi_rvalid  <= 1'b0;
            m_axi_rresp   <= 2'b00;
            m_axi_rlast   <= 1'b0;
            m_axi_rid     <= 4'b0;
            pending_read_addr  <= 32'b0;
            pending_read_beats <= 8'b0;
            pending_read_id    <= 4'b0;
            read_active        <= 1'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                data_memory[pending_write_addr[14:2]] <= m_axi_wdata;
                m_axi_bid    <= m_axi_awid;
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end

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

    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr    <= addr;
            csr_wr_data <= data;
            csr_wr_en   <= 1'b1;
            @(posedge clk);
            csr_wr_en   <= 1'b0;
        end
    endtask

    integer passed;
    integer failed;
    integer idx;
    integer expect_val;
    integer got_val;
    reg [31:0] c_val;
    reg [31:0] i_val;
    reg [31:0] l1_hits;
    reg [31:0] l1_misses;
    reg [31:0] l2_hits;
    reg [31:0] l2_misses;
    real l1_hit_rate;
    real l2_hit_rate;

    initial begin
        passed = 0;
        failed = 0;

        rst_n = 1'b0;
        csr_wr_en = 1'b0;
        csr_addr = 12'b0;
        csr_wr_data = 32'b0;

        for (ii = 0; ii < 8192; ii = ii + 1) begin
            data_memory[ii] = 32'h0;
        end

        for (ii = 0; ii < N_ELEMS; ii = ii + 1) begin
`ifdef TB_MEM_BENCH_GATHER
            data_memory[G_A_BASE_IDX + ii] = (ii * 10);
            data_memory[G_B_BASE_IDX + ii] = (ii * 5);
            data_memory[G_IDX_BASE_IDX + ii] = (ii * 5) & 32'h1F;
            data_memory[G_C_BASE_IDX + ii] = 32'hDEADBEEF;
`elsif TB_MEM_BENCH_SAXPY
            data_memory[SX_X_BASE_IDX + ii] = (ii + 1);
            data_memory[SX_Y_BASE_IDX + ii] = (100 + ii);
`else
            data_memory[S_A0_BASE_IDX + ii]  = (ii * 10);
            data_memory[S_A1_BASE_IDX + ii]  = (ii * 11);
            data_memory[S_A2_BASE_IDX + ii]  = (ii * 12);
            data_memory[S_A3_BASE_IDX + ii]  = (ii * 13);
            data_memory[S_B0_BASE_IDX + ii]  = (ii * 5);
            data_memory[S_B1_BASE_IDX + ii]  = (ii * 6);
            data_memory[S_B2_BASE_IDX + ii]  = (ii * 7);
            data_memory[S_B3_BASE_IDX + ii]  = (ii * 8);
            data_memory[S_C_BASE_IDX + ii] = 32'hDEADBEEF;
`endif
        end

`ifdef TB_MEM_BENCH_GATHER
        $display("=== Memory Benchmark: mem_gather ===");
        $display("TB_L1D_BYPASS=%0d", TB_L1D_BYPASS_CFG);
        $display("Init A[0..3]=%0d,%0d,%0d,%0d", data_memory[G_A_BASE_IDX], data_memory[G_A_BASE_IDX+1], data_memory[G_A_BASE_IDX+2], data_memory[G_A_BASE_IDX+3]);
        $display("Init B[0..3]=%0d,%0d,%0d,%0d", data_memory[G_B_BASE_IDX], data_memory[G_B_BASE_IDX+1], data_memory[G_B_BASE_IDX+2], data_memory[G_B_BASE_IDX+3]);
        $display("Init C[0..3]=0x%08h,0x%08h,0x%08h,0x%08h", data_memory[G_C_BASE_IDX], data_memory[G_C_BASE_IDX+1], data_memory[G_C_BASE_IDX+2], data_memory[G_C_BASE_IDX+3]);
`elsif TB_MEM_BENCH_SAXPY
        $display("=== Memory Benchmark: saxpy ===");
        $display("TB_L1D_BYPASS=%0d", TB_L1D_BYPASS_CFG);
        $display("Init X[0..3]=%0d,%0d,%0d,%0d", data_memory[SX_X_BASE_IDX], data_memory[SX_X_BASE_IDX+1], data_memory[SX_X_BASE_IDX+2], data_memory[SX_X_BASE_IDX+3]);
        $display("Init Y[0..3]=%0d,%0d,%0d,%0d", data_memory[SX_Y_BASE_IDX], data_memory[SX_Y_BASE_IDX+1], data_memory[SX_Y_BASE_IDX+2], data_memory[SX_Y_BASE_IDX+3]);
`else
        $display("=== Memory Benchmark: mem_stream8 ===");
        $display("TB_L1D_BYPASS=%0d", TB_L1D_BYPASS_CFG);
        $display("Init A0[0..3]=%0d,%0d,%0d,%0d", data_memory[S_A0_BASE_IDX], data_memory[S_A0_BASE_IDX+1], data_memory[S_A0_BASE_IDX+2], data_memory[S_A0_BASE_IDX+3]);
        $display("Init B0[0..3]=%0d,%0d,%0d,%0d", data_memory[S_B0_BASE_IDX], data_memory[S_B0_BASE_IDX+1], data_memory[S_B0_BASE_IDX+2], data_memory[S_B0_BASE_IDX+3]);
        $display("Init C[0..3]=0x%08h,0x%08h,0x%08h,0x%08h", data_memory[S_C_BASE_IDX], data_memory[S_C_BASE_IDX+1], data_memory[S_C_BASE_IDX+2], data_memory[S_C_BASE_IDX+3]);
`endif

        #100;
        rst_n = 1'b1;
        #200;

        csr_write(12'h008, 32'h0000_0000);   // KERNEL_PC
        csr_write(12'h00C, 32'h0000_0001);   // GRID_DIM_X
        csr_write(12'h010, 32'h0000_0001);   // GRID_DIM_Y
        csr_write(12'h014, 32'h0000_0001);   // GRID_DIM_Z
        csr_write(12'h018, 32'h0000_0020);   // BLOCK_DIM_X
        csr_write(12'h004, 32'h0000_0001);   // START

        fork : wait_kernel
            begin
                wait (irq_kernel_done);
                disable wait_kernel;
            end
            begin
                #120000;
                $display("ERROR: timeout waiting for kernel completion");
                $fatal(1);
            end
        join

        #200000;

        for (ii = 0; ii < N_ELEMS; ii = ii + 1) begin
`ifdef TB_MEM_BENCH_GATHER
            idx = data_memory[G_IDX_BASE_IDX + ii];
            expect_val = data_memory[G_A_BASE_IDX + idx] + data_memory[G_B_BASE_IDX + idx];
            got_val = data_memory[G_C_BASE_IDX + ii];
`elsif TB_MEM_BENCH_SAXPY
            idx = ii;
            expect_val = ((ii + 1) * 3) + (100 + ii);
            got_val = data_memory[SX_Y_BASE_IDX + ii];
`else
            expect_val = (data_memory[S_A0_BASE_IDX + ii] + data_memory[S_B0_BASE_IDX + ii]) * 4;
            got_val = data_memory[S_C_BASE_IDX + ii];
            idx = ii;
`endif
            if (got_val == expect_val) begin
                passed = passed + 1;
            end else begin
                failed = failed + 1;
                if (failed <= 8)
                    $display("FAIL i=%0d idx=%0d got=%0d expect=%0d", ii, idx, got_val, expect_val);
            end
        end
        @(posedge clk); csr_addr <= 12'h100; @(posedge clk); @(posedge clk);
        c_val = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h101; @(posedge clk); @(posedge clk);
        i_val = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h110; @(posedge clk); @(posedge clk);
        l1_hits = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h111; @(posedge clk); @(posedge clk);
        l1_misses = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h112; @(posedge clk); @(posedge clk);
        l2_hits = csr_rd_data;
        @(posedge clk); csr_addr <= 12'h113; @(posedge clk); @(posedge clk);
        l2_misses = csr_rd_data;

        if ((l1_hits + l1_misses) == 0) begin
            l1_hits = dut.sm_gen[0].u_sm.l1_stat_hits;
            l1_misses = dut.sm_gen[0].u_sm.l1_stat_misses;
        end
        l1_hit_rate = (l1_hits + l1_misses > 0) ? (($itor(l1_hits) * 100.0) / $itor(l1_hits + l1_misses)) : 0.0;
        l2_hit_rate = (l2_hits + l2_misses > 0) ? (($itor(l2_hits) * 100.0) / $itor(l2_hits + l2_misses)) : 0.0;

        $display("Cycles = %0d", c_val);
        $display("Instructions = %0d", i_val);
        if (c_val > 0)
            $display("IPC = %0d / %0d = %f", i_val, c_val, $itor(i_val) / $itor(c_val));
        else
            $display("IPC = 0 / 0 = 0.000000");

        $display("CacheStats: L1_hits=%0d L1_misses=%0d L1_hit_rate=%0.2f%% L2_hits=%0d L2_misses=%0d L2_hit_rate=%0.2f%%",
                 l1_hits, l1_misses, l1_hit_rate, l2_hits, l2_misses, l2_hit_rate);

        $display("Result = %0d pass, %0d fail", passed, failed);
        if (failed == 0) begin
            $display("*** MEMORY BENCHMARK PASSED ***");
            $finish;
        end else begin
            $display("*** MEMORY BENCHMARK FAILED ***");
            $fatal(1);
        end
    end

endmodule
