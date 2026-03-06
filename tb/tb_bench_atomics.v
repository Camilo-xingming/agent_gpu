//============================================================================
// RalphGPU - Atomic Microbenchmark Testbench
// Runs tb/bench_atomics.ptx (compile to bench_atomics.hex)
//============================================================================

`timescale 1ns / 1ps

module tb_bench_atomics;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    localparam ITERATIONS = 64;
    localparam MAX_CYCLES_DEFAULT = 200000;
    localparam MODE_ADDR = 32'h0000_1000;
    localparam SINGLE_TARGET = 32'h0000_1100;
    localparam MULTI_TARGET  = 32'h0000_1104;
    localparam GLOBAL_TARGET = 32'h0000_1108;
    localparam SINGLE_RESULT = 32'h0000_1110;
    localparam MULTI_RESULT  = 32'h0000_1114;
    localparam SHARED_RESULT = 32'h0000_1120;
    localparam GLOBAL_RESULT = 32'h0000_1124;
    localparam PERLANE_BASE  = 32'h0000_1200;
    localparam STRIPED_BASE  = 32'h0000_1300;
    localparam STRIPES       = 8;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Signals
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

    // AXI signals
    wire [AXI_ID_WIDTH-1:0]   m_axi_awid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_awaddr;
    wire [7:0]                m_axi_awlen;
    wire [2:0]                m_axi_awsize;
    wire [1:0]                m_axi_awburst;
    wire                      m_axi_awvalid;
    reg                       m_axi_awready;

    wire [AXI_DATA_WIDTH-1:0]   m_axi_wdata;
    wire [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb;
    wire                        m_axi_wlast;
    wire                        m_axi_wvalid;
    reg                         m_axi_wready;

    reg  [AXI_ID_WIDTH-1:0] m_axi_bid;
    reg  [1:0]              m_axi_bresp;
    reg                     m_axi_bvalid;
    wire                    m_axi_bready;

    wire [AXI_ID_WIDTH-1:0]   m_axi_arid;
    wire [AXI_ADDR_WIDTH-1:0] m_axi_araddr;
    wire [7:0]                m_axi_arlen;
    wire [2:0]                m_axi_arsize;
    wire [1:0]                m_axi_arburst;
    wire                      m_axi_arvalid;
    reg                       m_axi_arready;

    reg  [AXI_ID_WIDTH-1:0]   m_axi_rid;
    reg  [AXI_DATA_WIDTH-1:0] m_axi_rdata;
    reg  [1:0]                m_axi_rresp;
    reg                       m_axi_rlast;
    reg                       m_axi_rvalid;
    wire                      m_axi_rready;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    ralph_gpu_top #(
        .NUM_SM(1),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_ID_WIDTH(AXI_ID_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .csr_wr_en(csr_wr_en),
        .csr_addr(csr_addr),
        .csr_wr_data(csr_wr_data),
        .csr_rd_data(csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req(imem_req),
        .imem_addr(imem_addr),
        .imem_data(imem_data),
        .imem_valid(imem_valid),
        .m_axi_awid(m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    //------------------------------------------------------------------------
    // Instruction Memory - Load from hex file
    //------------------------------------------------------------------------
    reg [31:0] imem [0:255];
    integer instr_count;

    initial begin
        for (integer i = 0; i < 256; i = i + 1) begin
            imem[i] = 32'hFC000000;
        end
        $readmemh("bench_atomics.hex", imem);
        instr_count = 0;
        for (integer i = 0; i < 256; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from bench_atomics.hex", instr_count);
    end

    // Instruction memory response
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Data Memory (AXI)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_write_addr;

    initial begin
        for (integer i = 0; i < 4096; i = i + 1) begin
            gmem[i] = 32'h0;
        end
    end

    // AXI Write handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            pending_write_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_write_addr <= m_axi_awaddr;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                gmem[pending_write_addr[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1;
            end else if (m_axi_bready && m_axi_bvalid) begin
                m_axi_bvalid <= 1'b0;
            end
        end
    end

    // AXI Read handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1;
                m_axi_rlast <= 1'b1;
                m_axi_rid <= m_axi_arid;
            end else if (m_axi_rready && m_axi_rvalid) begin
                m_axi_rvalid <= 1'b0;
            end
        end
    end

    assign m_axi_bresp = 2'b00;
    assign m_axi_rresp = 2'b00;

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task write_csr;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr = addr;
            csr_wr_data = data;
            csr_wr_en = 1'b1;
            @(posedge clk);
            csr_wr_en = 1'b0;
        end
    endtask

    task launch_kernel;
        input [31:0] block_dim_x;
        begin
            write_csr(12'h018, block_dim_x);
            write_csr(12'h01C, 1);
            write_csr(12'h020, 1);
            write_csr(12'h008, 0);
            write_csr(12'h004, 1);
        end
    endtask

    task wait_kernel_done;
        output integer cycles;
        output reg timeout;
        begin
            cycles = 0;
            timeout = 1'b0;
            while (!irq_kernel_done && cycles < max_cycles) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            if (!irq_kernel_done) begin
                timeout = 1'b1;
            end
        end
    endtask

    function [31:0] read_mem;
        input [31:0] addr;
        begin
            read_mem = gmem[addr[13:2]];
        end
    endfunction

    function integer expected_stripe_count;
        input integer block_dim_x;
        input integer stripe_idx;
        integer t;
        integer count;
        begin
            count = 0;
            t = stripe_idx;
            while (t < block_dim_x) begin
                count = count + 1;
                t = t + STRIPES;
            end
            expected_stripe_count = count * ITERATIONS;
        end
    endfunction

    //------------------------------------------------------------------------
    // Baseline cycle inputs (optional)
    //------------------------------------------------------------------------
    integer baseline_single;
    integer baseline_multi;
    integer baseline_perlane;
    integer baseline_shared;
    integer baseline_global;
    integer baseline_striped;
    integer max_cycles;

    initial begin
        baseline_single = 0;
        baseline_multi = 0;
        baseline_perlane = 0;
        baseline_shared = 0;
        baseline_global = 0;
        baseline_striped = 0;
        max_cycles = MAX_CYCLES_DEFAULT;
        $value$plusargs("baseline_single=%d", baseline_single);
        $value$plusargs("baseline_multi=%d", baseline_multi);
        $value$plusargs("baseline_perlane=%d", baseline_perlane);
        $value$plusargs("baseline_shared=%d", baseline_shared);
        $value$plusargs("baseline_global=%d", baseline_global);
        $value$plusargs("baseline_striped=%d", baseline_striped);
        $value$plusargs("max_cycles=%d", max_cycles);
    end

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    integer cycles;
    integer i;
    integer lane_pass_count;
    integer fail_count;
    integer expected;
    integer ops_count;
    integer total_pass;
    integer total_fail;
    integer first_fail_lane;
    reg [31:0] first_fail_val;
    reg timeout;
    real improvement;

    task report_result;
        input [255:0] name;
        input integer baseline_cycles;
        input integer current_cycles;
        input reg pass;
        input reg timed_out;
        input integer operations;
        begin
            if (baseline_cycles > 0) begin
                improvement = 100.0 * (baseline_cycles - current_cycles) / baseline_cycles;
            end else begin
                improvement = 0.0;
            end
            $display("%0s", name);
            $display("|-- Baseline: %0d cycles", baseline_cycles);
            $display("|-- Optimized: %0d cycles", current_cycles);
            $display("|-- Improvement: %0.2f%%", improvement);
            if (operations > 0) begin
                $display("|-- Ops: %0d", operations);
                if (!timed_out) begin
                    $display("|-- Cycles/Op: %0.4f", current_cycles * 1.0 / operations);
                    $display("|-- Ops/Cycle: %0.6f", operations * 1.0 / current_cycles);
                end
            end
            if (timed_out) begin
                $display("`-- Conclusion: TIMEOUT");
                total_fail = total_fail + 1;
            end else if (pass) begin
                $display("`-- Conclusion: PASS");
                total_pass = total_pass + 1;
            end else begin
                $display("`-- Conclusion: FAIL");
                total_fail = total_fail + 1;
            end
            $display("");
        end
    endtask

    initial begin
        $display("\n============================================================");
        $display("RalphGPU Atomic Microbenchmarks");
        $display("Compile: python tools/ptx_assembler.py tb/bench_atomics.ptx -o bench_atomics.hex");
        $display("============================================================\n");

        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        total_pass = 0;
        total_fail = 0;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(10) @(posedge clk);

        // Test 0: Single-warp atomic throughput
        gmem[MODE_ADDR[13:2]] = 0;
        gmem[SINGLE_TARGET[13:2]] = 0;
        gmem[SINGLE_RESULT[13:2]] = 0;
        launch_kernel(32);
        wait_kernel_done(cycles, timeout);
        expected = 32 * ITERATIONS;
        ops_count = expected;
        if (!timeout && read_mem(SINGLE_RESULT) == expected) begin
            report_result("Atomic Single Warp", baseline_single, cycles, 1'b1, timeout, ops_count);
        end else begin
            if (!timeout) begin
                $display("Mismatch: expected %0d got %0d", expected, read_mem(SINGLE_RESULT));
            end
            report_result("Atomic Single Warp", baseline_single, cycles, 1'b0, timeout, ops_count);
        end

        // Test 1: Multi-warp atomic contention
        gmem[MODE_ADDR[13:2]] = 1;
        gmem[MULTI_TARGET[13:2]] = 0;
        gmem[MULTI_RESULT[13:2]] = 0;
        launch_kernel(128);
        wait_kernel_done(cycles, timeout);
        expected = 128 * ITERATIONS;
        ops_count = expected;
        if (!timeout && read_mem(MULTI_RESULT) == expected) begin
            report_result("Atomic Multi Warp", baseline_multi, cycles, 1'b1, timeout, ops_count);
        end else begin
            if (!timeout) begin
                $display("Mismatch: expected %0d got %0d", expected, read_mem(MULTI_RESULT));
            end
            report_result("Atomic Multi Warp", baseline_multi, cycles, 1'b0, timeout, ops_count);
        end

        // Test 2: Striped global atomics (reduced contention)
        gmem[MODE_ADDR[13:2]] = 5;
        for (i = 0; i < STRIPES; i = i + 1) begin
            gmem[(STRIPED_BASE[13:2]) + i] = 0;
        end
        launch_kernel(128);
        wait_kernel_done(cycles, timeout);
        fail_count = 0;
        first_fail_lane = -1;
        first_fail_val = 0;
        for (i = 0; i < STRIPES; i = i + 1) begin
            expected = expected_stripe_count(128, i);
            if (gmem[(STRIPED_BASE[13:2]) + i] != expected) begin
                fail_count = fail_count + 1;
                if (first_fail_lane < 0) begin
                    first_fail_lane = i;
                    first_fail_val = gmem[(STRIPED_BASE[13:2]) + i];
                end
            end
        end
        ops_count = 128 * ITERATIONS;
        if (!timeout && fail_count != 0) begin
            $display("Mismatch: stripe %0d expected %0d got %0d",
                     first_fail_lane, expected_stripe_count(128, first_fail_lane), first_fail_val);
        end
        report_result("Atomic Striped", baseline_striped, cycles, (fail_count == 0), timeout, ops_count);

        // Test 3: Per-lane atomics
        gmem[MODE_ADDR[13:2]] = 2;
        for (i = 0; i < 32; i = i + 1) begin
            gmem[(PERLANE_BASE[13:2]) + i] = 0;
        end
        launch_kernel(32);
        wait_kernel_done(cycles, timeout);
        lane_pass_count = 0;
        fail_count = 0;
        first_fail_lane = -1;
        first_fail_val = 0;
        for (i = 0; i < 32; i = i + 1) begin
            if (gmem[(PERLANE_BASE[13:2]) + i] == ITERATIONS) begin
                lane_pass_count = lane_pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                if (first_fail_lane < 0) begin
                    first_fail_lane = i;
                    first_fail_val = gmem[(PERLANE_BASE[13:2]) + i];
                end
            end
        end
        ops_count = 32 * ITERATIONS;
        if (!timeout && fail_count != 0) begin
            $display("Mismatch: lane %0d expected %0d got %0d", first_fail_lane, ITERATIONS, first_fail_val);
        end
        report_result("Atomic Per-Lane", baseline_perlane, cycles, (fail_count == 0), timeout, ops_count);

        // Test 4: Shared-memory atomics
        gmem[MODE_ADDR[13:2]] = 3;
        gmem[SHARED_RESULT[13:2]] = 0;
        launch_kernel(32);
        wait_kernel_done(cycles, timeout);
        expected = 32 * ITERATIONS;
        ops_count = expected;
        if (!timeout && read_mem(SHARED_RESULT) == expected) begin
            report_result("Atomic Shared", baseline_shared, cycles, 1'b1, timeout, ops_count);
        end else begin
            if (!timeout) begin
                $display("Mismatch: expected %0d got %0d", expected, read_mem(SHARED_RESULT));
            end
            report_result("Atomic Shared", baseline_shared, cycles, 1'b0, timeout, ops_count);
        end

        // Test 5: Global-memory atomics (shared-vs-global baseline)
        gmem[MODE_ADDR[13:2]] = 4;
        gmem[GLOBAL_TARGET[13:2]] = 0;
        gmem[GLOBAL_RESULT[13:2]] = 0;
        launch_kernel(32);
        wait_kernel_done(cycles, timeout);
        expected = 32 * ITERATIONS;
        ops_count = expected;
        if (!timeout && read_mem(GLOBAL_RESULT) == expected) begin
            report_result("Atomic Global", baseline_global, cycles, 1'b1, timeout, ops_count);
        end else begin
            if (!timeout) begin
                $display("Mismatch: expected %0d got %0d", expected, read_mem(GLOBAL_RESULT));
            end
            report_result("Atomic Global", baseline_global, cycles, 1'b0, timeout, ops_count);
        end

        $display("============================================================");
        $display("Atomic Microbenchmarks Complete");
        $display("Summary: %0d passed, %0d failed", total_pass, total_fail);
        $display("============================================================\n");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
