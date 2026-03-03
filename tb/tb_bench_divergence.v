//============================================================================
// RalphGPU - Divergence Microbenchmark Testbench
// Runs tb/bench_divergence.ptx (compile to bench_divergence.hex)
//============================================================================

`timescale 1ns / 1ps

module tb_bench_divergence;

    `include "../rtl/gpu_defines.vh"

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;
    parameter AXI_DATA_WIDTH = 32;
    parameter AXI_ADDR_WIDTH = 32;
    parameter AXI_ID_WIDTH = 4;

    localparam MODE_ADDR = 32'h0000_1000;
    localparam LOOP_BASE = 32'h0000_1100;
    localparam NESTED_BASE = 32'h0000_1200;
    localparam STACK_BASE = 32'h0000_1300;

    localparam LOOP_ITERS = 32;
    localparam NESTED_ITERS = 8;
    localparam NUM_THREADS = 64;
    localparam MAX_CYCLES_DEFAULT = 200000;

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
        $readmemh("bench_divergence.hex", imem);
        instr_count = 0;
        for (integer i = 0; i < 256; i = i + 1) begin
            if (imem[i] != 32'hFC000000) instr_count = instr_count + 1;
        end
        $display("Loaded %0d instructions from bench_divergence.hex", instr_count);
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
                $display("[TB-DBG] timeout cycles=%0d irq_done=%b kernel_done=%b", cycles, irq_kernel_done, u_dut.sm_gen[0].u_sm.kernel_done);
                $display("[TB-DBG] warp_valid=%b exit_pending=%b stalled_branch=%b stalled_mem=%b stalled_sync=%b stalled_async=%b",
                         u_dut.sm_gen[0].u_sm.warp_valid, u_dut.sm_gen[0].u_sm.warp_exit_pending,
                         u_dut.sm_gen[0].u_sm.warp_stalled_branch, u_dut.sm_gen[0].u_sm.warp_stalled_mem,
                         u_dut.sm_gen[0].u_sm.warp_stalled_sync, u_dut.sm_gen[0].u_sm.warp_stalled_async);
                $display("[TB-DBG] warp0 pc=%h fetch_pc=%h mask=%h sb=%h",
                         u_dut.sm_gen[0].u_sm.warp_pc[0], u_dut.sm_gen[0].u_sm.warp_fetch_pc[0],
                         u_dut.sm_gen[0].u_sm.warp_mask[0], u_dut.sm_gen[0].u_sm.u_scheduler.scoreboard[0]);
                $display("[TB-DBG] warp1 pc=%h fetch_pc=%h mask=%h sb=%h",
                         u_dut.sm_gen[0].u_sm.warp_pc[1], u_dut.sm_gen[0].u_sm.warp_fetch_pc[1],
                         u_dut.sm_gen[0].u_sm.warp_mask[1], u_dut.sm_gen[0].u_sm.u_scheduler.scoreboard[1]);
                $display("[TB-DBG] fetch_pending=%b inst_buf_valid=%b",
                         u_dut.sm_gen[0].u_sm.warp_fetch_pending,
                         u_dut.sm_gen[0].u_sm.warp_inst_buf_valid);
                $display("[TB-DBG] pending store=%b mem=%b smem=%b atomic=%b gmem_lat=%b smem_lat=%b",
                         u_dut.sm_gen[0].u_sm.store_pending_valid, u_dut.sm_gen[0].u_sm.mem_pending_valid,
                         u_dut.sm_gen[0].u_sm.smem_pending_valid, u_dut.sm_gen[0].u_sm.atomic_pending_valid,
                         u_dut.sm_gen[0].u_sm.gmem_resp_latched, u_dut.sm_gen[0].u_sm.smem_resp_latched);
                $display("[TB-DBG] inst_fast_valid=%b inst_fast_w0=%h inst_fast_w1=%h",
                         u_dut.sm_gen[0].u_sm.warp_inst_valid_fast,
                         u_dut.sm_gen[0].u_sm.warp_inst_buf_fast[0],
                         u_dut.sm_gen[0].u_sm.warp_inst_buf_fast[1]);
                $display("[TB-DBG] warp_ready=%b elig=%b has_haz=%b",
                         u_dut.sm_gen[0].u_sm.warp_ready,
                         u_dut.sm_gen[0].u_sm.u_scheduler.warp_eligible,
                         u_dut.sm_gen[0].u_sm.u_scheduler.warp_has_hazard);
                $display("[TB-DBG] fire0=%b fire1=%b dec0_v=%b dec1_v=%b stall_any=%b stall0=%b stall1=%b lane_conf=%b consume=%b",
                         u_dut.sm_gen[0].u_sm.issue0_fire, u_dut.sm_gen[0].u_sm.issue1_fire,
                         u_dut.sm_gen[0].u_sm.dec0_valid, u_dut.sm_gen[0].u_sm.dec1_valid,
                         u_dut.sm_gen[0].u_sm.decode_stalled_any, u_dut.sm_gen[0].u_sm.decode_stalled_slot0,
                         u_dut.sm_gen[0].u_sm.decode_stalled_slot1, u_dut.sm_gen[0].u_sm.lane_unit_conflict,
                         u_dut.sm_gen[0].u_sm.warp_inst_consume);
                $display("[TB-DBG] sched_valid=%b w0=%0d p0=%0d w1=%0d p1=%0d dec0_inst=%h dec1_inst=%h",
                         u_dut.sm_gen[0].u_sm.sched_issue_valid_mask,
                         u_dut.sm_gen[0].u_sm.sched_issue_warp_id[0], u_dut.sm_gen[0].u_sm.sched_issue_pipe[0],
                         u_dut.sm_gen[0].u_sm.sched_issue_warp_id[1], u_dut.sm_gen[0].u_sm.sched_issue_pipe[1],
                         u_dut.sm_gen[0].u_sm.dec0_instruction, u_dut.sm_gen[0].u_sm.dec1_instruction);
            end
        end
    endtask

    function [31:0] expected_nested;
        input integer tid;
        reg [31:0] val;
        begin
            val = (tid & 1) ? (NESTED_ITERS * 3) : (NESTED_ITERS * 1);
            val = val + ((tid & 2) ? 5 : 7);
            expected_nested = val;
        end
    endfunction

    function [31:0] expected_stack;
        input integer tid;
        reg [31:0] val;
        begin
            val = 0;
            if (tid & 1) begin
                val = val + 1;
                if (tid & 2) begin
                    val = val + 10;
                    if (tid & 4) begin
                        val = val + 100;
                        if (tid & 8) begin
                            val = val + 1000;
                            if (tid & 16) begin
                                val = val + 10000;
                                if (tid & 32) begin
                                    val = val + 100000;
                                end else begin
                                    val = val + 200000;
                                end
                            end else begin
                                val = val + 20000;
                            end
                        end else begin
                            val = val + 2000;
                        end
                    end else begin
                        val = val + 200;
                    end
                end else begin
                    val = val + 20;
                end
            end else begin
                val = val + 2;
            end
            expected_stack = val;
        end
    endfunction

    //------------------------------------------------------------------------
    // Baseline cycle inputs (optional)
    //------------------------------------------------------------------------
    integer baseline_loop;
    integer baseline_nested;
    integer baseline_stack;
    integer max_cycles;

    initial begin
        baseline_loop = 0;
        baseline_nested = 0;
        baseline_stack = 0;
        max_cycles = MAX_CYCLES_DEFAULT;
        $value$plusargs("baseline_loop=%d", baseline_loop);
        $value$plusargs("baseline_nested=%d", baseline_nested);
        $value$plusargs("baseline_stack=%d", baseline_stack);
        $value$plusargs("max_cycles=%d", max_cycles);
    end

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    integer cycles;
    integer i;
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
        $display("RalphGPU Divergence Microbenchmarks");
        $display("Compile: python tools/ptx_assembler.py tb/bench_divergence.ptx -o bench_divergence.hex");
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

        // Test 0: Simple loop (backward branch)
        gmem[MODE_ADDR[13:2]] = 0;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            gmem[(LOOP_BASE[13:2]) + i] = 0;
        end
        launch_kernel(NUM_THREADS);
        wait_kernel_done(cycles, timeout);
        expected = LOOP_ITERS * 2;
        fail_count = 0;
        first_fail_lane = -1;
        first_fail_val = 0;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            if (gmem[(LOOP_BASE[13:2]) + i] != expected) begin
                fail_count = fail_count + 1;
                if (first_fail_lane < 0) begin
                    first_fail_lane = i;
                    first_fail_val = gmem[(LOOP_BASE[13:2]) + i];
                end
            end
        end
        ops_count = NUM_THREADS * LOOP_ITERS;
        if (!timeout && fail_count != 0) begin
            $display("Mismatch: lane %0d expected %0d got %0d", first_fail_lane, expected, first_fail_val);
        end
        report_result("Divergence Loop", baseline_loop, cycles, (fail_count == 0), timeout, ops_count);

        // Test 1: Nested branches (if + loop)
        gmem[MODE_ADDR[13:2]] = 1;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            gmem[(NESTED_BASE[13:2]) + i] = 0;
        end
        launch_kernel(NUM_THREADS);
        wait_kernel_done(cycles, timeout);
        fail_count = 0;
        first_fail_lane = -1;
        first_fail_val = 0;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            if (gmem[(NESTED_BASE[13:2]) + i] != expected_nested(i)) begin
                fail_count = fail_count + 1;
                if (first_fail_lane < 0) begin
                    first_fail_lane = i;
                    first_fail_val = gmem[(NESTED_BASE[13:2]) + i];
                end
            end
        end
        ops_count = NUM_THREADS * NESTED_ITERS;
        if (!timeout && fail_count != 0) begin
            $display("Mismatch: lane %0d expected %0d got %0d",
                     first_fail_lane, expected_nested(first_fail_lane), first_fail_val);
        end
        report_result("Divergence Nested", baseline_nested, cycles, (fail_count == 0), timeout, ops_count);

        // Test 2: Deep nested branches (stack stress)
        gmem[MODE_ADDR[13:2]] = 2;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            gmem[(STACK_BASE[13:2]) + i] = 0;
        end
        launch_kernel(NUM_THREADS);
        wait_kernel_done(cycles, timeout);
        fail_count = 0;
        first_fail_lane = -1;
        first_fail_val = 0;
        for (i = 0; i < NUM_THREADS; i = i + 1) begin
            if (gmem[(STACK_BASE[13:2]) + i] != expected_stack(i)) begin
                fail_count = fail_count + 1;
                if (first_fail_lane < 0) begin
                    first_fail_lane = i;
                    first_fail_val = gmem[(STACK_BASE[13:2]) + i];
                end
            end
        end
        ops_count = NUM_THREADS;
        if (!timeout && fail_count != 0) begin
            $display("Mismatch: lane %0d expected %0d got %0d",
                     first_fail_lane, expected_stack(first_fail_lane), first_fail_val);
        end
        report_result("Divergence Stack", baseline_stack, cycles, (fail_count == 0), timeout, ops_count);

        $display("============================================================");
        $display("Divergence Microbenchmarks Complete");
        $display("Summary: %0d passed, %0d failed", total_pass, total_fail);
        $display("============================================================\n");
        $finish;
    end

endmodule
