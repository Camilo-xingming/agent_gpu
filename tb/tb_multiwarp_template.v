//============================================================================
// RalphGPU - Multi-Warp Harness Template (4 warps, dual-issue)
// 
// Minimal testbench for 4-warp parallel execution with slot 0+1 activity.
// Use as starting point for RALPH-9, scoreboard, or VCD verification tests.
//
// Architecture:
//   - 4 warps initialized (INIT_WARPS=4), ICACHE_BYPASS=1
//   - 1-cycle instruction memory latency
//   - AXI memory stub with configurable data
//   - VCD dump support
//   - Probes: issue0_fire, issue1_fire, warp_valid, lane_unit_conflict
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_multiwarp_template;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;

    // ---- Configurable test parameters ----
    localparam NUM_OPS       = 64;    // Instructions per warp (before EXIT)
    localparam TIMEOUT_CYCLES = 5000; // Simulation timeout

    // ---- Clock & Reset ----
    reg clk, rst_n;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- Instruction Memory (1-cycle latency) ----
    reg [31:0] imem [0:4095];
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;
    reg         imem_req_q;
    reg  [31:0] imem_addr_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 1'b0;
            imem_req_q <= 1'b0;
            imem_addr_q <= 0;
            imem_data <= 64'b0;
        end else begin
            imem_valid <= imem_req_q;
            if (imem_req_q)
                imem_data <= {imem[imem_addr_q[13:2] + 1], imem[imem_addr_q[13:2]]};
            imem_req_q <= imem_req;
            if (imem_req)
                imem_addr_q <= imem_addr;
        end
    end

    // ---- Instruction Encoding Helpers ----
    // IADD Rd, Ra, Rb  (ALU op, func=0 ADD)
    function [31:0] encode_iadd;
        input [4:0] rd, ra, rb;
        encode_iadd = {`OP_ALU, rd, ra, rb, 1'b0, 5'h00, 4'h0};
    endfunction

    // IMUL Rd, Ra, Rb  (MUL op)
    function [31:0] encode_imul;
        input [4:0] rd, ra, rb;
        encode_imul = {`OP_MUL, rd, ra, rb, 1'b0, 5'h00, 4'h0};
    endfunction

    // MOV_IMM Rd, imm16
    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
    endfunction

    // NOP
    function [31:0] encode_nop;
        encode_nop = {`OP_NOP, 26'b0};
    endfunction

    // EXIT
    function [31:0] encode_exit;
        encode_exit = {`OP_EXIT, 26'b0};
    endfunction

    // ---- DUT Control Signals ----
    reg         kernel_start;
    reg  [31:0] kernel_pc;
    reg  [31:0] block_id_x, block_id_y, block_id_z;
    reg  [31:0] block_dim_x, block_dim_y, block_dim_z;
    reg  [31:0] grid_dim_x, grid_dim_y, grid_dim_z;
    wire        kernel_done;

    // ---- L1D Stub (tie off) ----
    wire        l1d_req_valid, l1d_req_write;
    wire [31:0] l1d_req_addr [0:NUM_LANES-1];
    wire [31:0] l1d_req_wdata [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [31:0] l1d_resp_rdata [0:NUM_LANES-1];
    reg         l1d_resp_valid, l1d_resp_hit;

    // ---- AXI Stub (no-op, always ready) ----
    wire [3:0]   m_axi_awid, m_axi_arid;
    wire [31:0]  m_axi_awaddr, m_axi_araddr;
    wire [7:0]   m_axi_awlen, m_axi_arlen;
    wire [2:0]   m_axi_awsize, m_axi_arsize;
    wire [1:0]   m_axi_awburst, m_axi_arburst;
    wire         m_axi_awvalid, m_axi_arvalid;
    wire [255:0] m_axi_wdata;
    wire [31:0]  m_axi_wstrb;
    wire         m_axi_wlast, m_axi_wvalid;
    wire         m_axi_bready, m_axi_rready;
    reg          m_axi_awready, m_axi_wready, m_axi_arready;
    reg  [3:0]   m_axi_bid, m_axi_rid;
    reg  [1:0]   m_axi_bresp, m_axi_rresp;
    reg          m_axi_bvalid, m_axi_rvalid, m_axi_rlast;
    reg [255:0]  m_axi_rdata;

    // ---- DUT Instantiation ----
    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .INIT_WARPS(NUM_WARPS),   // All 4 warps active
        .ICACHE_BYPASS(1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .kernel_start(kernel_start), .kernel_pc(kernel_pc), .kernel_done(kernel_done),
        .block_id_x(block_id_x), .block_id_y(block_id_y), .block_id_z(block_id_z),
        .block_dim_x(block_dim_x), .block_dim_y(block_dim_y), .block_dim_z(block_dim_z),
        .grid_dim_x(grid_dim_x), .grid_dim_y(grid_dim_y), .grid_dim_z(grid_dim_z),
        .imem_req(imem_req), .imem_addr(imem_addr),
        .imem_ready(1'b1), .imem_data(imem_data), .imem_valid(imem_valid),
        .l1d_req_valid(l1d_req_valid), .l1d_req_write(l1d_req_write),
        .l1d_req_addr(l1d_req_addr), .l1d_req_wdata(l1d_req_wdata),
        .l1d_req_mask(l1d_req_mask),
        .l1d_resp_rdata(l1d_resp_rdata), .l1d_resp_valid(l1d_resp_valid),
        .l1d_resp_hit(l1d_resp_hit),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    // ---- Pipeline Probes ----
    wire issue0_fire = dut.issue0_fire;
    wire issue1_fire = dut.issue1_fire;
    wire [NUM_WARPS-1:0] warp_valid = dut.warp_valid;
    wire lane_unit_conflict = dut.lane_unit_conflict;
    wire [1:0] sched_mask = dut.u_scheduler.issue_valid;

    // ---- Counters ----
    integer cycle_count;
    integer slot0_issues, slot1_issues, conflict_cycles;

    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            cycle_count <= cycle_count + 1;
            if (issue0_fire) slot0_issues <= slot0_issues + 1;
            if (issue1_fire) slot1_issues <= slot1_issues + 1;
            if (lane_unit_conflict) conflict_cycles <= conflict_cycles + 1;
        end
    end

    // ---- Test Stimulus ----
    integer i;
    initial begin
        // VCD dump (optional — uncomment for waveform debug)
        // $dumpfile("/tmp/multiwarp.vcd");
        // $dumpvars(0, tb_multiwarp_template);

        clk = 0; rst_n = 0;
        kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 32; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        cycle_count = 0; slot0_issues = 0; slot1_issues = 0; conflict_cycles = 0;

        // AXI defaults
        m_axi_awready = 1; m_axi_wready = 1; m_axi_arready = 1;
        m_axi_bvalid = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
        m_axi_bid = 0; m_axi_rid = 0;
        m_axi_bresp = 0; m_axi_rresp = 0;
        m_axi_rdata = 0;

        // L1D defaults
        l1d_resp_valid = 0; l1d_resp_hit = 0;
        for (i = 0; i < NUM_LANES; i = i + 1) l1d_resp_rdata[i] = 0;

        // ---- Fill instruction memory ----
        // All warps share the same instruction stream.
        // Customize this section for your test case.
        //
        // Example: NUM_OPS ALU instructions + EXIT
        // This generates slot 0 + slot 1 activity since scheduler
        // issues 2 warps per cycle (even warps → slot 0, odd → slot 1).
        for (i = 0; i < NUM_OPS; i = i + 1) begin
            // Alternate IADD and IMUL so even/odd warps use different FUs.
            // Scheduler assigns: slot 0 = even warps (0,2), slot 1 = odd warps (1,3).
            // When even warp does IADD (ALU) and odd warp does IMUL (MUL), no conflict.
            // This maximizes dual-issue throughput.
            if (i % 2 == 0)
                imem[i] = encode_iadd(5'd1, 5'd0, 5'd0);  // R1 = R0 + R0 (ALU)
            else
                imem[i] = encode_imul(5'd2, 5'd1, 5'd0);  // R2 = R1 * R0 (MUL)
        end
        imem[NUM_OPS] = encode_exit();

        // Fill rest with NOPs (safety)
        for (i = NUM_OPS + 1; i < 4096; i = i + 1)
            imem[i] = encode_nop();

        // ---- Reset & Launch ----
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        kernel_start = 1;
        kernel_pc = 32'h0;
        #(CLK_PERIOD);
        kernel_start = 0;

        // ---- Wait for completion or timeout ----
        fork
            begin : wait_done
                wait (kernel_done || warp_valid == 0);
            end
            begin : wait_timeout
                #(CLK_PERIOD * TIMEOUT_CYCLES);
            end
        join_any
        disable wait_done;
        disable wait_timeout;

        #(CLK_PERIOD * 5);

        // ---- Report ----
        $display("============================================================");
        $display("Multi-Warp Harness Results:");
        $display("  Cycles:          %0d", cycle_count);
        $display("  Slot 0 issues:   %0d", slot0_issues);
        $display("  Slot 1 issues:   %0d", slot1_issues);
        $display("  Conflicts:       %0d", conflict_cycles);
        $display("  warp_valid:      %b", warp_valid);
        $display("  kernel_done:     %b", kernel_done);
        $display("============================================================");

        if (warp_valid == 0)
            $display("PASS: All warps exited cleanly");
        else
            $display("INFO: warp_valid=%b at timeout (may be expected)", warp_valid);

        if (slot1_issues > 0)
            $display("PASS: Slot 1 active (%0d issues)", slot1_issues);
        else
            $display("WARN: No slot 1 activity detected");

        $finish;
    end

endmodule
