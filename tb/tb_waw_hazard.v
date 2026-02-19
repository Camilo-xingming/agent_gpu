//============================================================================
// RalphGPU - WAW Hazard Detection Test (RALPH-9 P2.1)
//
// Tests two HIGH-risk WAW scenarios:
// 1. Same warp, same dest reg, different FU → premature scoreboard clear
// 2. Same-cycle SET + CLEAR NBA race → new instruction's bit cleared by old WB
//
// Run BEFORE and AFTER WAW fix to verify behavior change.
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_waw_hazard;

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

    // Scoreboard probes
    wire [31:0] sb_w0 = dut.u_scheduler.scoreboard[0];

    // Writeback
    wire wb_valid = dut.wb_valid;
    wire [4:0] wb_rd = dut.wb_rd;
    wire [1:0] wb_warp = dut.wb_warp_id;

    // WAW detection: scoreboard bit already set when new write issues to same reg
    // (This is what Option C would stall on)
    wire waw_detected = issue0_fire && dut.dec_reg_write &&
                        sb_w0[dut.dec_rd] && (dut.dec0_warp_id == 0);

    // Premature clear: WB clears a reg that still has in-flight write
    // Track by watching scoreboard[0][R1] transitions
    wire sb_r1 = sb_w0[1];

    // ---- Counters ----
    integer cycles, issues, wb_count, waw_events;
    integer sb_r1_set_count, sb_r1_clear_count;

    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            cycles <= cycles + 1;
            if (issue0_fire) issues <= issues + 1;
            if (wb_valid) wb_count <= wb_count + 1;
            if (waw_detected) waw_events <= waw_events + 1;
        end
    end

    // Track R1 scoreboard transitions (for premature clear detection)
    reg sb_r1_prev;
    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            sb_r1_prev <= sb_r1;
            if (!sb_r1_prev && sb_r1) sb_r1_set_count <= sb_r1_set_count + 1;
            if (sb_r1_prev && !sb_r1) sb_r1_clear_count <= sb_r1_clear_count + 1;
        end
    end

    // ---- Detailed trace ----
    always @(posedge clk) begin
        if (rst_n && !kernel_start && cycles > 0 && cycles <= 120) begin
            if (issue0_fire || wb_valid || (sb_r1_prev != sb_r1))
                $display("[C%0d] issue=%b wb=%b(r%0d,w%0d) sb0=%08h sb_r1=%b waw=%b",
                         cycles, issue0_fire, wb_valid, wb_rd, wb_warp,
                         sb_w0, sb_r1, waw_detected);
        end
    end

    // ---- Test ----
    integer i;
    initial begin
        clk = 0; rst_n = 0; kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 128; block_dim_y = 1; block_dim_z = 1;  // 4 warps
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        cycles = 0; issues = 0; wb_count = 0; waw_events = 0;
        sb_r1_set_count = 0; sb_r1_clear_count = 0; sb_r1_prev = 0;

        m_axi_awready = 1; m_axi_wready = 1; m_axi_arready = 1;
        m_axi_bvalid = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
        m_axi_bid = 0; m_axi_rid = 0; m_axi_bresp = 0; m_axi_rresp = 0; m_axi_rdata = 0;
        l1d_resp_valid = 0; l1d_resp_hit = 0;
        for (i = 0; i < NUM_LANES; i = i + 1) l1d_resp_rdata[i] = 0;

        // ===============================================================
        // Kernel: WAW hazard test cases
        // ===============================================================
        //
        // Scenario 1: PREMATURE SCOREBOARD CLEAR
        //   MOV_IMM R10, 1      ; init source regs
        //   MOV_IMM R11, 2
        //   IADD R1, R10, R11   ; ALU writes R1 → scoreboard[w][R1] = 1
        //   IMUL R1, R10, R11   ; MUL writes R1 → scoreboard[w][R1] = 1 (still)
        //   IADD R2, R1, R10    ; reads R1 — should wait for BOTH writes to complete
        //                       ; BUG: if first WB clears sb[R1], this issues too early
        //
        // Scenario 2: REPEATED SAME-REG WRITES (stress test)
        //   IADD R5, R10, R11   ; write R5 via ALU
        //   IMUL R5, R10, R11   ; write R5 via MUL (WAW)
        //   IADD R5, R10, R10   ; write R5 via ALU again (triple WAW)
        //   IADD R6, R5, R10    ; read R5 — must wait for ALL three WBs
        //
        // Scenario 3: SAFE (no WAW, control)
        //   IADD R20, R10, R11
        //   IADD R21, R10, R11
        //   IADD R22, R20, R21  ; RAW only, no WAW
        //
        // EXIT
        // ===============================================================

        i = 0;

        // Init source regs (no hazards)
        imem[i] = encode_mov_imm(5'd10, 16'd1);   i = i + 1;  // R10 = 1
        imem[i] = encode_mov_imm(5'd11, 16'd2);   i = i + 1;  // R11 = 2

        // Scenario 1: ALU then MUL write same R1
        imem[i] = encode_iadd(5'd1, 5'd10, 5'd11);  i = i + 1;  // R1 = R10+R11 (ALU)
        imem[i] = encode_imul(5'd1, 5'd10, 5'd11);  i = i + 1;  // R1 = R10*R11 (MUL) — WAW!
        imem[i] = encode_iadd(5'd2, 5'd1, 5'd10);   i = i + 1;  // R2 = R1+R10 — reads R1

        // Scenario 2: Triple WAW on R5
        imem[i] = encode_iadd(5'd5, 5'd10, 5'd11);  i = i + 1;  // R5 = ALU
        imem[i] = encode_imul(5'd5, 5'd10, 5'd11);  i = i + 1;  // R5 = MUL (WAW)
        imem[i] = encode_iadd(5'd5, 5'd10, 5'd10);  i = i + 1;  // R5 = ALU (triple WAW)
        imem[i] = encode_iadd(5'd6, 5'd5, 5'd10);   i = i + 1;  // R6 = R5+R10 — reads R5

        // Scenario 3: Control (no WAW)
        imem[i] = encode_iadd(5'd20, 5'd10, 5'd11);  i = i + 1;
        imem[i] = encode_iadd(5'd21, 5'd10, 5'd11);  i = i + 1;
        imem[i] = encode_iadd(5'd22, 5'd20, 5'd21);  i = i + 1;  // RAW only

        // EXIT
        imem[i] = encode_exit();  i = i + 1;

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
        $display("WAW Hazard Detection Test (RALPH-9 P2.1)");
        $display("============================================================");
        $display("  Cycles:         %0d", cycles);
        $display("  Issues:         %0d", issues);
        $display("  WB count:       %0d", wb_count);
        $display("  WAW events:     %0d (issue while sb[rd] already set)", waw_events);
        $display("  sb[R1] sets:    %0d", sb_r1_set_count);
        $display("  sb[R1] clears:  %0d", sb_r1_clear_count);
        $display("  warp_valid:     %b", warp_valid);
        $display("  kernel_done:    %b", kernel_done);
        $display("============================================================");

        if (warp_valid == 0)
            $display("[PASS] All warps exited");
        else
            $display("[FAIL] warp_valid=%b (warps stuck)", warp_valid);

        // Before WAW fix: waw_events > 0 (instructions issue despite sb bit set)
        // After WAW fix:  waw_events = 0 (stalled on WAW)
        if (waw_events > 0)
            $display("[DETECT] WAW hazard: %0d instructions issued while dest reg busy", waw_events);
        else
            $display("[CLEAN] No WAW hazards detected");

        // Premature clear check: if R1 has 2 SETs but only sees 1 CLEAR between them,
        // the second SET was masked. With fix, should see WAW stall instead.
        if (sb_r1_set_count > sb_r1_clear_count)
            $display("[INFO] R1 scoreboard: %0d sets, %0d clears (sets > clears = idempotent SET)",
                     sb_r1_set_count, sb_r1_clear_count);
        else
            $display("[INFO] R1 scoreboard: %0d sets, %0d clears",
                     sb_r1_set_count, sb_r1_clear_count);

        $finish;
    end

endmodule
