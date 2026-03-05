`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_sm_gmem_arbiter;
    localparam NUM_LANES  = 8;
    localparam SIMD_WIDTH = NUM_LANES * 32;

    reg                     clk;
    reg                     rst_n;

    reg                     normal_req_valid;
    reg                     normal_req_write;
    reg [NUM_LANES*32-1:0]  normal_req_addr;
    reg [SIMD_WIDTH-1:0]    normal_req_wdata;
    reg [NUM_LANES-1:0]     normal_req_mask;

    reg                     atomic_req;
    reg                     atomic_write;
    reg [5:0]               atomic_lane;
    reg [31:0]              atomic_addr;
    reg [31:0]              atomic_wdata;
    reg                     atomic_shared_pending;
    reg                     smem_atomic_resp_valid;
    reg [31:0]              smem_atomic_resp_rdata;

    wire                    atomic_ready;
    wire                    atomic_resp_read_valid;
    wire                    atomic_resp_write_valid;
    wire [SIMD_WIDTH-1:0]   atomic_rdata;
    wire                    atomic_pending;

    reg                     ace_req_valid;
    reg [31:0]              ace_req_addr;
    reg [4:0]               ace_req_size;

    wire                    ace_resp_valid;
    wire [127:0]            ace_resp_data;
    wire                    ace_pending;

    reg                     tex_req;
    reg [31:0]              tex_addr;
    reg                     tex_write;
    reg [127:0]             tex_wdata;

    wire                    tex_ready;
    wire                    tex_resp_valid;
    wire [127:0]            tex_resp_data;
    wire                    tex_pending;

    wire                    gmem_req_valid;
    wire                    gmem_req_write;
    wire [NUM_LANES*32-1:0] gmem_req_addr;
    wire [SIMD_WIDTH-1:0]   gmem_req_wdata;
    wire [NUM_LANES-1:0]    gmem_req_mask;

    reg                     gmem_req_ready;
    reg                     gmem_resp_valid;
    reg [SIMD_WIDTH-1:0]    gmem_resp_rdata;

    wire                    normal_resp_valid;
    wire [SIMD_WIDTH-1:0]   normal_resp_rdata;

    integer pass_count;
    integer fail_count;
    integer i;

    sm_gmem_arbiter #(
        .NUM_LANES(NUM_LANES),
        .SIMD_WIDTH(SIMD_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),

        .normal_req_valid(normal_req_valid),
        .normal_req_write(normal_req_write),
        .normal_req_addr(normal_req_addr),
        .normal_req_wdata(normal_req_wdata),
        .normal_req_mask(normal_req_mask),

        .atomic_req(atomic_req),
        .atomic_write(atomic_write),
        .atomic_lane(atomic_lane),
        .atomic_addr(atomic_addr),
        .atomic_wdata(atomic_wdata),
        .atomic_shared_pending(atomic_shared_pending),
        .smem_atomic_resp_valid(smem_atomic_resp_valid),
        .smem_atomic_resp_rdata(smem_atomic_resp_rdata),
        .atomic_ready(atomic_ready),
        .atomic_resp_read_valid(atomic_resp_read_valid),
        .atomic_resp_write_valid(atomic_resp_write_valid),
        .atomic_rdata(atomic_rdata),
        .atomic_pending(atomic_pending),

        .ace_req_valid(ace_req_valid),
        .ace_req_addr(ace_req_addr),
        .ace_req_size(ace_req_size),
        .ace_resp_valid(ace_resp_valid),
        .ace_resp_data(ace_resp_data),
        .ace_pending(ace_pending),

        .tex_req(tex_req),
        .tex_addr(tex_addr),
        .tex_write(tex_write),
        .tex_wdata(tex_wdata),
        .tex_ready(tex_ready),
        .tex_resp_valid(tex_resp_valid),
        .tex_resp_data(tex_resp_data),
        .tex_pending(tex_pending),

        .gmem_req_valid(gmem_req_valid),
        .gmem_req_write(gmem_req_write),
        .gmem_req_addr(gmem_req_addr),
        .gmem_req_wdata(gmem_req_wdata),
        .gmem_req_mask(gmem_req_mask),

        .gmem_req_ready(gmem_req_ready),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_rdata(gmem_resp_rdata),

        .normal_resp_valid(normal_resp_valid),
        .normal_resp_rdata(normal_resp_rdata)
    );

    always #5 clk = ~clk;

    function [NUM_LANES*32-1:0] rep_addr;
        input [31:0] addr;
        begin
            rep_addr = {NUM_LANES{addr}};
        end
    endfunction

    function [SIMD_WIDTH-1:0] rep_data;
        input [31:0] data;
        begin
            rep_data = {NUM_LANES{data}};
        end
    endfunction

    function [NUM_LANES-1:0] onehot_lane;
        input [5:0] lane;
        begin
            onehot_lane = {NUM_LANES{1'b0}};
            onehot_lane[lane] = 1'b1;
        end
    endfunction

    function [NUM_LANES*32-1:0] lane_addr_vec;
        input [5:0] lane;
        input [31:0] addr;
        integer j;
        begin
            lane_addr_vec = {NUM_LANES*32{1'b0}};
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                if (j[5:0] == lane)
                    lane_addr_vec[j*32 +: 32] = addr;
            end
        end
    endfunction

    function [SIMD_WIDTH-1:0] lane_data_vec;
        input [5:0] lane;
        input [31:0] data;
        integer j;
        begin
            lane_data_vec = {SIMD_WIDTH{1'b0}};
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                if (j[5:0] == lane)
                    lane_data_vec[j*32 +: 32] = data;
            end
        end
    endfunction

    task check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", msg);
            end
        end
    endtask

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task clear_inputs;
        begin
            normal_req_valid = 1'b0;
            normal_req_write = 1'b0;
            normal_req_addr  = {NUM_LANES*32{1'b0}};
            normal_req_wdata = {SIMD_WIDTH{1'b0}};
            normal_req_mask  = {NUM_LANES{1'b0}};

            atomic_req = 1'b0;
            atomic_write = 1'b0;
            atomic_lane = 6'd0;
            atomic_addr = 32'd0;
            atomic_wdata = 32'd0;
            atomic_shared_pending = 1'b0;
            smem_atomic_resp_valid = 1'b0;
            smem_atomic_resp_rdata = 32'd0;

            ace_req_valid = 1'b0;
            ace_req_addr  = 32'd0;
            ace_req_size  = 5'd0;

            tex_req = 1'b0;
            tex_addr = 32'd0;
            tex_write = 1'b0;
            tex_wdata = 128'd0;

            gmem_req_ready = 1'b1;
            gmem_resp_valid = 1'b0;
            gmem_resp_rdata = {SIMD_WIDTH{1'b0}};
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        pass_count = 0;
        fail_count = 0;

        clear_inputs();

        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        tick();

        // Case 1: reset/idle
        check(!gmem_req_valid, "case1: idle gmem_req_valid");
        check(!atomic_pending, "case1: idle atomic_pending");
        check(!ace_pending, "case1: idle ace_pending");
        check(!tex_pending, "case1: idle tex_pending");

        // Case 2: priority normal > atomic > ace > tex
        clear_inputs();
        normal_req_valid = 1'b1;
        normal_req_write = 1'b1;
        normal_req_addr  = rep_addr(32'h1000_0040);
        normal_req_wdata = rep_data(32'hCAFE_BABE);
        normal_req_mask  = 8'b1111_0000;

        atomic_req = 1'b1;
        atomic_write = 1'b1;
        atomic_lane = 6'd3;
        atomic_addr = 32'hDEAD_0003;
        atomic_wdata = 32'h1111_2222;

        ace_req_valid = 1'b1;
        ace_req_addr  = 32'h2000_0000;

        tex_req = 1'b1;
        tex_addr = 32'h3000_0000;
        tex_write = 1'b1;
        tex_wdata = 128'hABCD_0004_ABCD_0003_ABCD_0002_ABCD_0001;

        #1;
        check(gmem_req_valid, "case2: gmem_req_valid asserted");
        check(gmem_req_write, "case2: normal write selected");
        check(gmem_req_addr == rep_addr(32'h1000_0040), "case2: normal addr selected");
        check(gmem_req_wdata == rep_data(32'hCAFE_BABE), "case2: normal wdata selected");
        check(gmem_req_mask == 8'b1111_0000, "case2: normal mask selected");

        // Case 3: atomic lane-vector expansion and pending/read response
        clear_inputs();
        atomic_req = 1'b1;
        atomic_write = 1'b0;
        atomic_lane = 6'd3;
        atomic_addr = 32'h4000_00A0;
        atomic_wdata = 32'h5A5A_C3C3;
        #1;
        check(gmem_req_valid, "case3: atomic request selected");
        check(!gmem_req_write, "case3: atomic read selected");
        check(gmem_req_mask == onehot_lane(6'd3), "case3: atomic onehot mask");
        check(gmem_req_addr == lane_addr_vec(6'd3, 32'h4000_00A0), "case3: atomic addr vector");
        check(gmem_req_wdata == lane_data_vec(6'd3, 32'h5A5A_C3C3), "case3: atomic wdata vector");
        tick();
        check(atomic_pending, "case3: atomic pending set after handshake");

        gmem_resp_valid = 1'b1;
        gmem_resp_rdata = rep_data(32'hAAAA_BBBB);
        tick();
        check(!atomic_pending, "case3: atomic pending cleared by response");
        check(atomic_ready, "case3: atomic_ready pulsed");
        check(atomic_resp_read_valid, "case3: atomic read-valid pulsed");
        check(!atomic_resp_write_valid, "case3: atomic write-valid not set for read");
        check(atomic_rdata == rep_data(32'hAAAA_BBBB), "case3: atomic rdata routed from gmem");

        // Case 4: atomic write response type
        clear_inputs();
        atomic_req = 1'b1;
        atomic_write = 1'b1;
        atomic_lane = 6'd1;
        atomic_addr = 32'h4000_00B0;
        atomic_wdata = 32'h1234_5678;
        tick();
        check(atomic_pending, "case4: atomic write pending set");

        gmem_resp_valid = 1'b1;
        gmem_resp_rdata = rep_data(32'h0BAD_F00D);
        tick();
        check(atomic_ready, "case4: atomic write ready pulse");
        check(!atomic_resp_read_valid, "case4: read-valid low for write");
        check(atomic_resp_write_valid, "case4: write-valid high for write");

        // Case 5: shared-atomic bypass (no gmem arbitration)
        clear_inputs();
        atomic_req = 1'b1;
        atomic_write = 1'b0;
        atomic_shared_pending = 1'b1;
        smem_atomic_resp_valid = 1'b1;
        smem_atomic_resp_rdata = 32'h55AA_00FF;
        #1;
        check(!gmem_req_valid, "case5: shared atomic bypass suppresses gmem req");
        check(atomic_ready, "case5: atomic_ready from smem response");
        check(atomic_resp_read_valid, "case5: read-valid from smem response");
        check(!atomic_resp_write_valid, "case5: write-valid low for shared read");
        check(atomic_rdata == rep_data(32'h55AA_00FF), "case5: shared atomic data replicated");

        // Case 6: ACE request/response path
        clear_inputs();
        ace_req_valid = 1'b1;
        ace_req_addr  = 32'h5000_0080;
        normal_req_mask = 8'hFF;
        #1;
        check(gmem_req_valid, "case6: ace request selected");
        check(!gmem_req_write, "case6: ace is read-only");
        check(gmem_req_addr == rep_addr(32'h5000_0080), "case6: ace addr replicated");
        tick();
        check(ace_pending, "case6: ace pending set");

        gmem_resp_valid = 1'b1;
        gmem_resp_rdata = {128'h0000_0000_0000_0000_0000_0000_0000_0000,
                           128'hFACE_0004_FACE_0003_FACE_0002_FACE_0001};
        #1;
        check(ace_resp_valid, "case6: ace response valid");
        check(ace_resp_data == 128'hFACE_0004_FACE_0003_FACE_0002_FACE_0001, "case6: ace response data low128");
        tick();
        check(!ace_pending, "case6: ace pending cleared");

        // Case 7: texture request/response + tex_ready gating
        clear_inputs();
        tex_req = 1'b1;
        tex_addr = 32'h6000_0040;
        tex_write = 1'b1;
        tex_wdata = 128'h1234_0004_1234_0003_1234_0002_1234_0001;
        normal_req_mask = 8'hFF;
        #1;
        check(tex_ready, "case7: tex_ready high when no pending and gmem ready");
        check(gmem_req_valid, "case7: texture request selected");
        check(gmem_req_write, "case7: texture write propagated");
        check(gmem_req_addr == rep_addr(32'h6000_0040), "case7: texture addr replicated");
        check(gmem_req_wdata[127:0] == 128'h1234_0004_1234_0003_1234_0002_1234_0001, "case7: texture payload in low128");
        check(gmem_req_wdata[SIMD_WIDTH-1:128] == {(SIMD_WIDTH-128){1'b0}}, "case7: texture payload upper lanes zero");

        tick();
        check(tex_pending, "case7: texture pending set");
        check(!tex_ready, "case7: tex_ready low while pending");

        gmem_resp_valid = 1'b1;
        gmem_resp_rdata = {128'h0, 128'hABAB_0004_ABAB_0003_ABAB_0002_ABAB_0001};
        #1;
        check(tex_resp_valid, "case7: texture response valid");
        check(tex_resp_data == 128'hABAB_0004_ABAB_0003_ABAB_0002_ABAB_0001, "case7: texture response data low128");
        tick();
        check(!tex_pending, "case7: texture pending cleared");

        // Case 8: atomic priority beats ACE/TEX when normal absent
        clear_inputs();
        atomic_req = 1'b1;
        atomic_write = 1'b0;
        atomic_lane = 6'd2;
        atomic_addr = 32'h7000_0008;
        atomic_wdata = 32'hDEAD_BEEF;
        ace_req_valid = 1'b1;
        ace_req_addr = 32'h7000_1000;
        tex_req = 1'b1;
        tex_addr = 32'h7000_2000;
        #1;
        check(gmem_req_addr == lane_addr_vec(6'd2, 32'h7000_0008), "case8: atomic has priority over ace/tex");

        // Case 9: ace priority over tex when no normal/atomic
        clear_inputs();
        ace_req_valid = 1'b1;
        ace_req_addr = 32'h7100_1000;
        tex_req = 1'b1;
        tex_addr = 32'h7100_2000;
        #1;
        check(gmem_req_addr == rep_addr(32'h7100_1000), "case9: ace has priority over texture");

        // Case 10: normal response routing when side units idle
        clear_inputs();
        gmem_resp_valid = 1'b1;
        gmem_resp_rdata = rep_data(32'hC001_D00D);
        #1;
        check(normal_resp_valid, "case10: normal response valid when side pending low");
        check(normal_resp_rdata == rep_data(32'hC001_D00D), "case10: normal response data forwarded");

        // Case 11: tex_ready follows gmem_req_ready when not pending
        clear_inputs();
        tex_req = 1'b1;
        gmem_req_ready = 1'b0;
        #1;
        check(!tex_ready, "case11: tex_ready low when gmem not ready");

        if (fail_count == 0) begin
            $display("========================================");
            $display("[PASS] tb_sm_gmem_arbiter: %0d checks", pass_count);
            $display("========================================");
            $finish;
        end else begin
            $display("========================================");
            $display("[FAIL] tb_sm_gmem_arbiter: %0d passed, %0d failed", pass_count, fail_count);
            $display("========================================");
            $fatal(1, "tb_sm_gmem_arbiter failed");
        end
    end
endmodule
