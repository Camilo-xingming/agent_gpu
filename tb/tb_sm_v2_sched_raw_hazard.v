`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

// Verifies RAW hazard handling is enforced by scheduler scoreboard,
// without relying on decode-stage RAW recheck.
module tb_sm_v2_sched_raw_hazard;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam TIMEOUT    = 4000;

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
            imem_valid <= 0;
            imem_req_q <= 0;
            imem_addr_q <= 0;
            imem_data <= 0;
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
        encode_iadd = {`OP_ALU, rd, ra, rb, 5'b0, 6'h0};
    endfunction

    function [31:0] encode_imul;
        input [4:0] rd, ra, rb;
        encode_imul = {`OP_MUL, rd, ra, rb, 5'b0, 6'h0};
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
    reg [31:0]   m_axi_rdata;

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
    wire warp0_has_hazard = dut.u_scheduler.warp_has_hazard[0];
    wire warp0_inst_valid = dut.warp_inst_valid_fast[0];
    wire warp0_valid = dut.warp_valid[0];
    wire [31:0] sb_w0 = dut.u_scheduler.scoreboard[0];

    wire issue_slot0_w0 = dut.sched_issue_valid_mask[0] && (dut.sched_issue_warp_id[0] == 0);
    wire issue_slot1_w0 = dut.sched_issue_valid_mask[1] && (dut.sched_issue_warp_id[1] == 0);
    wire issue_warp0 = issue_slot0_w0 || issue_slot1_w0;

    wire decode_raw_stall = dut.lane0_stall_raw || dut.lane1_stall_raw;

    integer cycles;
    integer hazard_cycles;
    integer blocked_cycles;
    integer warp0_issue_cycles;
    integer decode_raw_stall_cycles;

    always @(posedge clk) begin
        if (rst_n && !kernel_start) begin
            cycles <= cycles + 1;
            if (warp0_has_hazard && warp0_inst_valid && warp0_valid) begin
                hazard_cycles <= hazard_cycles + 1;
                if (!issue_warp0) blocked_cycles <= blocked_cycles + 1;
            end
            if (issue_warp0) warp0_issue_cycles <= warp0_issue_cycles + 1;
            if (decode_raw_stall) decode_raw_stall_cycles <= decode_raw_stall_cycles + 1;
        end
    end

    integer i;
    initial begin
        clk = 0; rst_n = 0; kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 32; block_dim_y = 1; block_dim_z = 1;  // single active warp
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;
        cycles = 0;
        hazard_cycles = 0;
        blocked_cycles = 0;
        warp0_issue_cycles = 0;
        decode_raw_stall_cycles = 0;

        m_axi_awready = 1; m_axi_wready = 1; m_axi_arready = 1;
        m_axi_bvalid = 0; m_axi_rvalid = 0; m_axi_rlast = 0;
        m_axi_bid = 0; m_axi_rid = 0; m_axi_bresp = 0; m_axi_rresp = 0; m_axi_rdata = 0;
        l1d_resp_valid = 0; l1d_resp_hit = 0;
        l1d_resp_rdata = {(NUM_LANES*32){1'b0}};

        // Tight RAW chain:
        //   mov r1
        //   add r2,r1,r1
        //   add r3,r2,r2
        //   add r4,r3,r3
        //   mul r5,r4,r2
        //   exit
        i = 0;
        imem[i] = encode_mov_imm(5'd1, 16'd1);      i = i + 1;
        imem[i] = encode_iadd(5'd2, 5'd1, 5'd1);    i = i + 1;
        imem[i] = encode_iadd(5'd3, 5'd2, 5'd2);    i = i + 1;
        imem[i] = encode_iadd(5'd4, 5'd3, 5'd3);    i = i + 1;
        imem[i] = encode_imul(5'd5, 5'd4, 5'd2);    i = i + 1;
        imem[i] = encode_exit();                    i = i + 1;
        for (; i < 4096; i = i + 1) imem[i] = encode_nop();

        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);
        kernel_start = 1; kernel_pc = 0;
        #(CLK_PERIOD);
        kernel_start = 0;

        fork
            begin : done_wait
                wait (kernel_done || (dut.warp_valid == 0));
            end
            begin : timeout_wait
                #(CLK_PERIOD * TIMEOUT);
            end
        join_any
        disable done_wait;
        disable timeout_wait;
        #(CLK_PERIOD * 5);

        $display("============================================================");
        $display("SM V2 Scheduler RAW Hazard Test");
        $display("============================================================");
        $display("cycles=%0d", cycles);
        $display("hazard_cycles=%0d blocked_cycles=%0d issue_w0=%0d", hazard_cycles, blocked_cycles, warp0_issue_cycles);
        $display("decode_raw_stall_cycles=%0d", decode_raw_stall_cycles);
        $display("warp_valid=%b kernel_done=%b sb_w0=%08h", dut.warp_valid, kernel_done, sb_w0);
        $display("============================================================");

        if (!(kernel_done || (dut.warp_valid == 0))) begin
            $display("[FAIL] timeout before kernel completion");
            $fatal(1);
        end
        if (hazard_cycles == 0) begin
            $display("[FAIL] scheduler never reported warp0 hazard");
            $fatal(1);
        end
        if (blocked_cycles == 0) begin
            $display("[FAIL] warp0 was never blocked while hazard was active");
            $fatal(1);
        end
        if (decode_raw_stall_cycles != 0) begin
            $display("[FAIL] decode-stage RAW stall fired (%0d cycles)", decode_raw_stall_cycles);
            $fatal(1);
        end

        $display("[PASS] scheduler RAW hazard check blocks issue as expected");
        $finish;
    end

endmodule
