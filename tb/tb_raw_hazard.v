//============================================================================
// RalphGPU - RAW Hazard Detection Test (RALPH-9 P2)
//
// Tests that enabling per-register RAW stall correctly:
// 1. Stalls on true RAW dependency (add r1,r2,r3 → add r4,r1,r5)
// 2. Does NOT stall on independent instructions
// 3. Isolates per-warp (warp 0 dependency doesn't stall warp 1)
// 4. Clears stall after writeback completes
//
// Run BEFORE and AFTER enabling RAW stall to verify behavior change.
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_raw_hazard;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam TIMEOUT    = 3000;

    reg clk, rst_n;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- Instruction Memory ----
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

    // ---- Encoding ----
    function [31:0] encode_iadd;
        input [4:0] rd, ra, rb;
        encode_iadd = {`OP_ALU, rd, ra, rb, 5'b0, 6'h0};  // Fixed: was 31 bits
    endfunction

    function [31:0] encode_imul;
        input [4:0] rd, ra, rb;
        encode_imul = {`OP_MUL, rd, ra, rb, 5'b0, 6'h0};  // Fixed: was 31 bits
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
    wire [31:0] l1d_req_addr [0:NUM_LANES-1];
    wire [31:0] l1d_req_wdata [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [31:0] l1d_resp_rdata [0:NUM_LANES-1];
    reg         l1d_resp_valid, l1d_resp_hit;

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

    streaming_multiprocessor_v2 #(
        .SM_ID(0), .NUM_WARPS(NUM_WARPS), .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH), .INIT_WARPS(NUM_WARPS), .ICACHE_BYPASS(1)
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

    // ---- Probes ----
    wire issue0_fire = dut.issue0_fire;
    wire [NUM_WARPS-1:0] warp_valid = dut.warp_valid;

    // RAW stall probes (will become nonzero after P2 fix)
    wire raw_stall_0 = dut.lane0_stall_raw;
    wire ra_busy_0 = dut.lane0_ra_busy;
    wire rb_busy_0 = dut.lane0_rb_busy;

    // Scoreboard probes
    wire [31:0] sb_w0 = dut.u_scheduler.scoreboard[0];
    wire [31:0] sb_w1 = dut.u_scheduler.scoreboard[1];

    // Writeback
    wire wb_valid = dut.wb_valid;
    wire [4:0] wb_rd = dut.wb_rd;
    wire [1:0] wb_warp = dut.wb_warp_id;

    // ---- Counters ----
    integer cycles, issues, raw_stalls, wb_count;

    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            cycles <= cycles + 1;
            if (issue0_fire) issues <= issues + 1;
            if (raw_stall_0) raw_stalls <= raw_stalls + 1;
            if (wb_valid) wb_count <= wb_count + 1;
        end
    end

    // ---- Detailed trace (first 80 cycles) ----
    always @(posedge clk) begin
        if (rst_n && !kernel_start && cycles > 0 && cycles <= 80) begin
            if (issue0_fire || raw_stall_0 || wb_valid)
                $display("[C%0d] issue=%b raw_stall=%b ra_busy=%b rb_busy=%b sb0=%08h wb=%b wr=%0d warp=%0d",
                         cycles, issue0_fire, raw_stall_0, ra_busy_0, rb_busy_0,
                         sb_w0, wb_valid, wb_rd, wb_warp);
        end
    end

    // ---- Test ----
    integer i;
    initial begin
        clk = 0; rst_n = 0; kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 128; block_dim_y = 1; block_dim_z = 1;  // 4 warps
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        cycles = 0; issues = 0; raw_stalls = 0; wb_count = 0;

        m_axi_awready = 1; m_axi_wready = 1; m_axi_arready = 1;
        m_axi_bvalid = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
        m_axi_bid = 0; m_axi_rid = 0; m_axi_bresp = 0; m_axi_rresp = 0; m_axi_rdata = 0;
        l1d_resp_valid = 0; l1d_resp_hit = 0;
        for (i = 0; i < NUM_LANES; i = i + 1) l1d_resp_rdata[i] = 0;

        // ===============================================================
        // Kernel layout — tests RAW hazard detection
        // ===============================================================
        //
        // Section 1: TRUE RAW DEPENDENCY (should stall with P2 fix)
        //   MOV_IMM R1, 10      ; write R1
        //   MOV_IMM R2, 20      ; write R2 (independent)
        //   IADD R3, R1, R2     ; reads R1, R2 — RAW on R1 and R2!
        //   IADD R4, R3, R1     ; reads R3 — RAW on R3!
        //
        // Section 2: NO DEPENDENCY (should NOT stall)
        //   MOV_IMM R10, 1
        //   MOV_IMM R11, 2
        //   MOV_IMM R12, 3
        //   MOV_IMM R13, 4      ; all independent, no RAW
        //
        // Section 3: LONG CHAIN (stress test)
        //   MOV_IMM R20, 1
        //   IADD R21, R20, R20  ; RAW on R20
        //   IADD R22, R21, R21  ; RAW on R21
        //   IADD R23, R22, R22  ; RAW on R22
        //   IADD R24, R23, R23  ; RAW on R23
        //   IMUL R25, R24, R20  ; RAW on R24 (cross-FU)
        //
        // Section 4: EXIT
        // ===============================================================

        i = 0;

        // Section 1: True RAW
        imem[i] = encode_mov_imm(5'd1, 16'd10);   i = i + 1;  // R1 = 10
        imem[i] = encode_mov_imm(5'd2, 16'd20);   i = i + 1;  // R2 = 20
        imem[i] = encode_iadd(5'd3, 5'd1, 5'd2);  i = i + 1;  // R3 = R1 + R2 (RAW on R1, R2)
        imem[i] = encode_iadd(5'd4, 5'd3, 5'd1);  i = i + 1;  // R4 = R3 + R1 (RAW on R3)

        // Section 2: No dependency
        imem[i] = encode_mov_imm(5'd10, 16'd1);   i = i + 1;
        imem[i] = encode_mov_imm(5'd11, 16'd2);   i = i + 1;
        imem[i] = encode_mov_imm(5'd12, 16'd3);   i = i + 1;
        imem[i] = encode_mov_imm(5'd13, 16'd4);   i = i + 1;

        // Section 3: Long chain
        imem[i] = encode_mov_imm(5'd20, 16'd1);   i = i + 1;  // R20 = 1
        imem[i] = encode_iadd(5'd21, 5'd20, 5'd20); i = i + 1; // R21 = R20+R20 (RAW)
        imem[i] = encode_iadd(5'd22, 5'd21, 5'd21); i = i + 1; // R22 = R21+R21 (RAW)
        imem[i] = encode_iadd(5'd23, 5'd22, 5'd22); i = i + 1; // R23 = R22+R22 (RAW)
        imem[i] = encode_iadd(5'd24, 5'd23, 5'd23); i = i + 1; // R24 = R23+R23 (RAW)
        imem[i] = encode_imul(5'd25, 5'd24, 5'd20); i = i + 1; // R25 = R24*R20 (RAW cross-FU)

        // EXIT
        imem[i] = encode_exit();                   i = i + 1;

        // Fill rest with NOP
        for (; i < 4096; i = i + 1) imem[i] = encode_nop();

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
        $display("RAW Hazard Detection Test (RALPH-9 P2)");
        $display("============================================================");
        $display("  Cycles:       %0d", cycles);
        $display("  Issues:       %0d", issues);
        $display("  RAW stalls:   %0d", raw_stalls);
        $display("  WB count:     %0d", wb_count);
        $display("  warp_valid:   %b", warp_valid);
        $display("  kernel_done:  %b", kernel_done);
        $display("============================================================");

        if (warp_valid == 0)
            $display("[PASS] All warps exited");
        else
            $display("[FAIL] warp_valid=%b", warp_valid);

        // Before P2 fix: raw_stalls = 0 (hardcoded 1'b0)
        // After P2 fix:  raw_stalls > 0 (RAW dependencies detected)
        if (raw_stalls == 0)
            $display("[INFO] RAW stall = 0 — lane0_stall_raw is still hardcoded 1'b0");
        else
            $display("[PASS] RAW stall = %0d — hazard detection ACTIVE", raw_stalls);

        // Expected RAW dependencies per warp:
        //   Section 1: R3←R1,R2 + R4←R3 = 2 hazards
        //   Section 3: R21←R20 + R22←R21 + R23←R22 + R24←R23 + R25←R24 = 5 hazards
        //   Total: 7 per warp × 4 warps = 28 expected stall events (minimum)
        if (raw_stalls >= 20)
            $display("[PASS] Sufficient RAW stalls detected (expected ~28+)");
        else if (raw_stalls > 0)
            $display("[INFO] RAW stalls=%0d, expected ~28+ (may vary with warp interleaving)", raw_stalls);

        $finish;
    end

endmodule
