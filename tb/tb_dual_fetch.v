//============================================================================
// RalphGPU - Dual-Fetch Verification Test (#148 True Dual-Issue)
//
// Tests icache port B dual-fetch: 4 warps, interleaved ALU+MUL instructions.
// Uses ICACHE_BYPASS=0 to exercise the real icache with dual read ports.
// All warps share the same code → trailing warps hit in cache → port B fires.
//
// Goal: prove same-cycle dual-issue with dual-fetch enabled.
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_dual_fetch;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam NUM_PAIRS  = 64;    // 64 ALU+MUL pairs = 128 instructions per warp
    localparam TIMEOUT    = 10000; // cycles (higher for icache warmup)

    reg clk, rst_n;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- Instruction Memory (1-cycle latency, backs icache miss path) ----
    reg [31:0] imem [0:4095];
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;
    reg         imem_req_q;
    reg  [31:0] imem_addr_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 0; imem_req_q <= 0; imem_addr_q <= 0; imem_data <= 0;
        end else begin
            imem_valid <= imem_req_q;
            if (imem_req_q)
                imem_data <= {imem[imem_addr_q[13:2] + 1], imem[imem_addr_q[13:2]]};
            imem_req_q <= imem_req;
            if (imem_req) imem_addr_q <= imem_addr;
        end
    end

    // ---- Encoding helpers ----
    function [31:0] encode_iadd;
        input [4:0] rd, ra, rb;
        encode_iadd = {`OP_ALU, rd, ra, rb, 1'b0, 5'h00, 4'h0};
    endfunction

    function [31:0] encode_imul;
        input [4:0] rd, ra, rb;
        encode_imul = {`OP_MUL, rd, ra, rb, 1'b0, 5'h00, 4'h0};
    endfunction

    function [31:0] encode_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
    endfunction

    function [31:0] encode_nop;
        encode_nop = {`OP_NOP, 26'b0};
    endfunction

    function [31:0] encode_exit;
        encode_exit = {`OP_EXIT, 26'b0};
    endfunction

    // ---- DUT ----
    reg         kernel_start;
    reg  [31:0] kernel_pc;
    reg  [31:0] block_id_x, block_id_y, block_id_z;
    reg  [31:0] block_dim_x, block_dim_y, block_dim_z;
    reg  [31:0] grid_dim_x, grid_dim_y, grid_dim_z;
    wire        kernel_done;

    wire        l1d_req_valid, l1d_req_write;
    wire [NUM_LANES*32-1:0] l1d_req_addr;
    wire [NUM_LANES*32-1:0] l1d_req_wdata;
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [NUM_LANES*32-1:0] l1d_resp_rdata;
    reg         l1d_resp_valid, l1d_resp_hit;

    wire [3:0]   m_axi_awid, m_axi_arid;
    wire [31:0]  m_axi_awaddr, m_axi_araddr;
    wire [7:0]   m_axi_awlen, m_axi_arlen;
    wire [2:0]   m_axi_awsize, m_axi_arsize;
    wire [1:0]   m_axi_awburst, m_axi_arburst;
    wire         m_axi_awvalid, m_axi_arvalid;
    wire [31:0]  m_axi_wdata;
    wire [3:0]   m_axi_wstrb;
    wire         m_axi_wlast, m_axi_wvalid;
    wire         m_axi_bready, m_axi_rready;
    reg          m_axi_awready, m_axi_wready, m_axi_arready;
    reg  [3:0]   m_axi_bid, m_axi_rid;
    reg  [1:0]   m_axi_bresp, m_axi_rresp;
    reg          m_axi_bvalid, m_axi_rvalid, m_axi_rlast;
    reg  [31:0]  m_axi_rdata;
    wire        perf_stall_ifetch;

    streaming_multiprocessor_v2 #(
        .SM_ID(0), .NUM_WARPS(NUM_WARPS), .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH), .INIT_WARPS(NUM_WARPS),
        .ICACHE_BYPASS(0)   // *** Use real icache with port B ***
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
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .perf_stall_ifetch(perf_stall_ifetch)
    );

    // ---- Probes ----
    wire issue0_fire = dut.issue0_fire;
    wire issue1_fire = dut.issue1_fire;
    wire [NUM_WARPS-1:0] warp_valid = dut.warp_valid;
    wire lane_conflict = dut.lane_unit_conflict;

    // Per-slot FU type detection
    wire s0_alu = dut.lane0_alu;
    wire s0_mul = dut.lane0_mul;
    wire s1_alu = dut.lane1_alu;
    wire s1_mul = dut.lane1_mul;

    // Fetch pipeline probes
    wire [NUM_WARPS-1:0] buf_valid_dbg = dut.warp_inst_buf_valid;
    wire fetch_fire_b_dbg = dut.u_fetch_pipeline.fetch_fire_b;

    // ---- Counters ----
    integer cycles, s0_issues, s1_issues, dual_issues, conflicts;
    integer s0_alu_cnt, s0_mul_cnt, s1_alu_cnt, s1_mul_cnt;
    integer alu_mul_dual;
    integer port_b_fires;
    integer ifetch_stall_cycles;

    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            cycles <= cycles + 1;
            if (issue0_fire) begin
                s0_issues <= s0_issues + 1;
                if (s0_alu) s0_alu_cnt <= s0_alu_cnt + 1;
                if (s0_mul) s0_mul_cnt <= s0_mul_cnt + 1;
            end
            if (issue1_fire) begin
                s1_issues <= s1_issues + 1;
                if (s1_alu) s1_alu_cnt <= s1_alu_cnt + 1;
                if (s1_mul) s1_mul_cnt <= s1_mul_cnt + 1;
            end
            if (issue0_fire && issue1_fire) begin
                dual_issues <= dual_issues + 1;
                if ((s0_alu && s1_mul) || (s0_mul && s1_alu))
                    alu_mul_dual <= alu_mul_dual + 1;
            end
            if (lane_conflict) conflicts <= conflicts + 1;
            if (fetch_fire_b_dbg) port_b_fires <= port_b_fires + 1;
            if (perf_stall_ifetch) ifetch_stall_cycles <= ifetch_stall_cycles + 1;
        end
    end

    // ---- Test ----
    integer i;
    initial begin
        clk = 0; rst_n = 0; kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 128; block_dim_y = 1; block_dim_z = 1;  // 128 threads = 4 warps x 32
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        cycles = 0; s0_issues = 0; s1_issues = 0; dual_issues = 0; conflicts = 0;
        s0_alu_cnt = 0; s0_mul_cnt = 0; s1_alu_cnt = 0; s1_mul_cnt = 0;
        alu_mul_dual = 0; port_b_fires = 0; ifetch_stall_cycles = 0;

        m_axi_awready = 1; m_axi_wready = 1; m_axi_arready = 1;
        m_axi_bvalid = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
        m_axi_bid = 0; m_axi_rid = 0; m_axi_bresp = 0; m_axi_rresp = 0; m_axi_rdata = 0;
        l1d_resp_valid = 0; l1d_resp_hit = 0; l1d_resp_rdata = 0;

        // ---- Kernel: Interleaved ALU/MUL with different destination regs ----
        imem[0] = encode_mov_imm(5'd1, 16'd1);   // R1 = 1
        imem[1] = encode_mov_imm(5'd2, 16'd2);   // R2 = 2

        for (i = 0; i < NUM_PAIRS; i = i + 1) begin
            imem[2 + i*2]     = encode_iadd(5'd3 + (i % 14) * 2, 5'd1, 5'd2);
            imem[2 + i*2 + 1] = encode_imul(5'd4 + (i % 14) * 2, 5'd1, 5'd2);
        end
        imem[2 + NUM_PAIRS*2] = encode_exit();

        for (i = 2 + NUM_PAIRS*2 + 1; i < 4096; i = i + 1) imem[i] = encode_nop();

        // ---- Launch ----
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        kernel_start = 1; kernel_pc = 0;
        #(CLK_PERIOD);
        kernel_start = 0;

        // ---- Wait ----
        fork
            begin : done_wait
                wait (kernel_done || warp_valid == 0);
            end
            begin : timeout_wait
                #(CLK_PERIOD * TIMEOUT);
            end
        join_any
        disable done_wait;
        disable timeout_wait;
        #(CLK_PERIOD * 5);

        // ---- Results ----
        $display("============================================================");
        $display("Dual-Fetch Test (#148 True Dual-Issue)");
        $display("============================================================");
        $display("  ICACHE_BYPASS:   0 (real icache with port B)");
        $display("  Warps:           %0d", NUM_WARPS);
        $display("  Pairs/warp:      %0d (= %0d ALU + %0d MUL)", NUM_PAIRS, NUM_PAIRS, NUM_PAIRS);
        $display("  Cycles:          %0d", cycles);
        $display("  Slot 0:          %0d issues (ALU=%0d, MUL=%0d)", s0_issues, s0_alu_cnt, s0_mul_cnt);
        $display("  Slot 1:          %0d issues (ALU=%0d, MUL=%0d)", s1_issues, s1_alu_cnt, s1_mul_cnt);
        $display("  Dual-issue:      %0d total, %0d ALU+MUL cross-FU", dual_issues, alu_mul_dual);
        $display("  Port B fires:    %0d", port_b_fires);
        $display("  I$ stall cycles: %0d (perf_stall_ifetch)", ifetch_stall_cycles);
        $display("  Conflicts:       %0d", conflicts);
        $display("  Total issues:    %0d", s0_issues + s1_issues);
        $display("  IPC:             %0d.%02d", (s0_issues + s1_issues) / cycles,
                 ((s0_issues + s1_issues) * 100 / (cycles > 0 ? cycles : 1)) % 100);
        $display("  warp_valid:      %b", warp_valid);
        $display("  kernel_done:     %b", kernel_done);
        $display("============================================================");
        $display("[METRIC] issue0_fire=%0d issue1_fire=%0d dual_issue=%0d perf_sched_stall_ifetch=%0d cycles=%0d port_b_fire=%0d",
                 s0_issues, s1_issues, dual_issues, ifetch_stall_cycles, cycles, port_b_fires);

        // Checks
        if (warp_valid == 0)
            $display("[PASS] All warps exited");
        else
            $fatal(1, "[FAIL] warp_valid=%b (expected 0)", warp_valid);

        if (port_b_fires > 0)
            $display("[PASS] Port B fired %0d times — dual-fetch WORKING", port_b_fires);
        else
            $fatal(1, "[FAIL] Port B never fired — dual-fetch NOT working");

        if (dual_issues > 0)
            $display("[PASS] Same-cycle dual-issue count = %0d", dual_issues);
        else
            $fatal(1, "[FAIL] No same-cycle dual-issue observed");

        // IPC check: with 4 warps x 130 instructions, baseline ~520 issues
        // Dual-issue should reduce cycles, improving IPC above 1.0
        if (s0_issues + s1_issues > 0 && cycles > 0) begin
            if ((s0_issues + s1_issues) * 100 / cycles > 130)
                $display("[PASS] IPC > 1.30 — meets >30%% improvement target");
            else if ((s0_issues + s1_issues) * 100 / cycles > 100)
                $display("[INFO] IPC > 1.0 — some dual-issue benefit");
            else
                $display("[INFO] IPC <= 1.0 — limited dual-issue benefit");
        end

        $finish;
    end

endmodule
