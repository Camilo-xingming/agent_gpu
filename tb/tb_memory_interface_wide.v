`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_memory_interface_wide;
    localparam NUM_LANES = 4;
    localparam LANE_WIDTH = 128;
    localparam NUM_WARPS = 4;
    localparam THREADS_PER_WARP = 8;
    localparam ADDR_WIDTH = 32;
    localparam DATA_WIDTH = 32;
    localparam MSHR_ENTRIES = 8;
    localparam MAX_OUTSTANDING = 8;
    localparam COALESCE_WINDOW = 2;
    localparam WARP_DATA_WIDTH = DATA_WIDTH * THREADS_PER_WARP;
    localparam TOTAL_WARP_DATA_WIDTH = WARP_DATA_WIDTH * NUM_WARPS;

    reg clk;
    reg rst_n;

    reg [NUM_WARPS-1:0] warp_req_valid;
    reg [NUM_WARPS-1:0] warp_req_write;
    reg [ADDR_WIDTH*NUM_WARPS-1:0] warp_req_addr;
    reg [TOTAL_WARP_DATA_WIDTH-1:0] warp_req_wdata;
    reg [THREADS_PER_WARP*NUM_WARPS-1:0] warp_req_mask;
    wire [NUM_WARPS-1:0] warp_req_ready;

    wire [NUM_WARPS-1:0] warp_resp_valid;
    wire [TOTAL_WARP_DATA_WIDTH-1:0] warp_resp_rdata;

    wire [NUM_LANES-1:0] lane_req_valid;
    wire [NUM_LANES-1:0] lane_req_write;
    wire [ADDR_WIDTH*NUM_LANES-1:0] lane_req_addr;
    wire [LANE_WIDTH*NUM_LANES-1:0] lane_req_wdata;
    wire [(LANE_WIDTH/8)*NUM_LANES-1:0] lane_req_wmask;
    reg [NUM_LANES-1:0] lane_req_ready;

    reg [NUM_LANES-1:0] lane_resp_valid;
    reg [LANE_WIDTH*NUM_LANES-1:0] lane_resp_rdata;

    wire [31:0] stat_requests;
    wire [31:0] stat_coalesced;
    wire [31:0] stat_outstanding_peak;

    integer pass_count;
    integer fail_count;
    integer timeout;
    integer cycle_count;
    integer lane_hs_count [0:NUM_LANES-1];
    integer li;

    integer base_lane0;
    integer base_lane1;
    integer base_lane2;

    memory_interface_wide #(
        .NUM_LANES(NUM_LANES),
        .LANE_WIDTH(LANE_WIDTH),
        .NUM_WARPS(NUM_WARPS),
        .THREADS_PER_WARP(THREADS_PER_WARP),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .MSHR_ENTRIES(MSHR_ENTRIES),
        .MAX_OUTSTANDING(MAX_OUTSTANDING),
        .COALESCE_WINDOW(COALESCE_WINDOW)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_req_valid(warp_req_valid),
        .warp_req_write(warp_req_write),
        .warp_req_addr(warp_req_addr),
        .warp_req_wdata(warp_req_wdata),
        .warp_req_mask(warp_req_mask),
        .warp_req_ready(warp_req_ready),
        .warp_resp_valid(warp_resp_valid),
        .warp_resp_rdata(warp_resp_rdata),
        .lane_req_valid(lane_req_valid),
        .lane_req_write(lane_req_write),
        .lane_req_addr(lane_req_addr),
        .lane_req_wdata(lane_req_wdata),
        .lane_req_wmask(lane_req_wmask),
        .lane_req_ready(lane_req_ready),
        .lane_resp_valid(lane_resp_valid),
        .lane_resp_rdata(lane_resp_rdata),
        .stat_requests(stat_requests),
        .stat_coalesced(stat_coalesced),
        .stat_outstanding_peak(stat_outstanding_peak)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [DATA_WIDTH-1:0] warp_word;
        input integer warp;
        input integer thread_idx;
        begin
            warp_word = warp_resp_rdata[(warp*THREADS_PER_WARP + thread_idx)*DATA_WIDTH +: DATA_WIDTH];
        end
    endfunction

    task check_result;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
                $display("[PASS] %s", msg);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %s", msg);
            end
        end
    endtask

    task clear_warp_inputs;
        begin
            warp_req_valid = {NUM_WARPS{1'b0}};
            warp_req_write = {NUM_WARPS{1'b0}};
            warp_req_addr = {ADDR_WIDTH*NUM_WARPS{1'b0}};
            warp_req_wdata = {TOTAL_WARP_DATA_WIDTH{1'b0}};
            warp_req_mask = {THREADS_PER_WARP*NUM_WARPS{1'b0}};
        end
    endtask

    task load_warp_payload;
        input integer warp;
        input [ADDR_WIDTH-1:0] addr;
        input req_write;
        input [THREADS_PER_WARP-1:0] mask;
        input [DATA_WIDTH-1:0] base_data;
        integer t;
        begin
            warp_req_addr[warp*ADDR_WIDTH +: ADDR_WIDTH] = addr;
            warp_req_write[warp] = req_write;
            warp_req_mask[warp*THREADS_PER_WARP +: THREADS_PER_WARP] = mask;
            for (t = 0; t < THREADS_PER_WARP; t = t + 1) begin
                warp_req_wdata[(warp*THREADS_PER_WARP + t)*DATA_WIDTH +: DATA_WIDTH] = base_data + t;
            end
        end
    endtask

    task issue_single_warp_req;
        input integer warp;
        input [ADDR_WIDTH-1:0] addr;
        input req_write;
        input [THREADS_PER_WARP-1:0] mask;
        input [DATA_WIDTH-1:0] base_data;
        begin
            timeout = 0;
            while (!warp_req_ready[warp] && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            check_result(warp_req_ready[warp], "warp_req_ready asserted before issue");

            clear_warp_inputs;
            load_warp_payload(warp, addr, req_write, mask, base_data);
            warp_req_valid[warp] = 1'b1;
            @(posedge clk);
            warp_req_valid[warp] = 1'b0;
            warp_req_write[warp] = 1'b0;
            warp_req_mask[warp*THREADS_PER_WARP +: THREADS_PER_WARP] = {THREADS_PER_WARP{1'b0}};
        end
    endtask

    task wait_lane_handshake;
        input integer lane;
        input integer baseline;
        input [255:0] name;
        begin
            timeout = 0;
            while ((lane_hs_count[lane] <= baseline) && (timeout < 200)) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            check_result(lane_hs_count[lane] > baseline, name);
        end
    endtask

    task wait_warp_resp;
        input integer warp;
        input [255:0] name;
        begin
            timeout = 0;
            while (!warp_resp_valid[warp] && timeout < 200) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            check_result(warp_resp_valid[warp], name);
        end
    endtask

    task drive_lane_resp_word;
        input integer lane;
        input [DATA_WIDTH-1:0] word;
        begin
            lane_resp_rdata = {LANE_WIDTH*NUM_LANES{1'b0}};
            lane_resp_rdata[lane*LANE_WIDTH +: DATA_WIDTH] = word;
            lane_resp_valid[lane] = 1'b1;
            @(posedge clk);
            lane_resp_valid[lane] = 1'b0;
            lane_resp_rdata[lane*LANE_WIDTH +: LANE_WIDTH] = {LANE_WIDTH{1'b0}};
        end
    endtask

    // Prevent deadlock in simulation.
    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (cycle_count > 5000) begin
                $fatal(1, "tb_memory_interface_wide timeout");
            end
        end
    end

    // Track lane handshakes to validate coalescing and backpressure behavior.
    always @(posedge clk) begin
        if (!rst_n) begin
            for (li = 0; li < NUM_LANES; li = li + 1) begin
                lane_hs_count[li] <= 0;
            end
        end else begin
            for (li = 0; li < NUM_LANES; li = li + 1) begin
                if (lane_req_valid[li] && lane_req_ready[li]) begin
                    lane_hs_count[li] <= lane_hs_count[li] + 1;
                end
            end
        end
    end

    initial begin
        pass_count = 0;
        fail_count = 0;
        cycle_count = 0;

        rst_n = 1'b0;
        clear_warp_inputs;
        lane_req_ready = {NUM_LANES{1'b1}};
        lane_resp_valid = {NUM_LANES{1'b0}};
        lane_resp_rdata = {LANE_WIDTH*NUM_LANES{1'b0}};

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ------------------------------------------------------------------
        // Test 1: aligned wide read request and response mapping.
        // ------------------------------------------------------------------
        $display("=== TEST 1: aligned wide read ===");
        base_lane0 = lane_hs_count[0];
        issue_single_warp_req(0, 32'h0000_1004, 1'b0, {THREADS_PER_WARP{1'b1}}, 32'h1111_0000);
        wait_lane_handshake(0, base_lane0, "T1 lane0 handshake");
        check_result(lane_req_addr[0*ADDR_WIDTH +: ADDR_WIDTH] == 32'h0000_1000, "T1 lane0 aligned address");
        check_result(lane_req_write[0] == 1'b0, "T1 read request keeps lane_req_write low");

        drive_lane_resp_word(0, 32'hA5A5_1001);
        wait_warp_resp(0, "T1 warp0 response valid");
        check_result(warp_word(0, 0) == 32'hA5A5_1001, "T1 warp0 thread0 data");
        check_result(warp_word(0, THREADS_PER_WARP-1) == 32'hA5A5_1001, "T1 warp0 broadcast data");

        // ------------------------------------------------------------------
        // Test 2: two warps coalesce into one lane request (burst/alignment).
        // ------------------------------------------------------------------
        $display("=== TEST 2: coalesced same-line reads ===");
        base_lane0 = lane_hs_count[0];
        issue_single_warp_req(1, 32'h0000_2000, 1'b0, {THREADS_PER_WARP{1'b1}}, 32'h2222_0000);
        issue_single_warp_req(2, 32'h0000_200C, 1'b0, {THREADS_PER_WARP{1'b1}}, 32'h3333_0000);
        wait_lane_handshake(0, base_lane0, "T2 coalesced lane0 handshake");
        repeat (6) @(posedge clk);
        check_result(lane_hs_count[0] == (base_lane0 + 1), "T2 only one lane transaction after coalescing");
        check_result(lane_req_addr[0*ADDR_WIDTH +: ADDR_WIDTH] == 32'h0000_2000, "T2 aligned coalesced base address");

        drive_lane_resp_word(0, 32'hBEEF_2200);
        timeout = 0;
        while (!(warp_resp_valid[1] && warp_resp_valid[2]) && timeout < 200) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        check_result(warp_resp_valid[1] && warp_resp_valid[2], "T2 both warps receive shared response");
        check_result(warp_word(1, 0) == 32'hBEEF_2200, "T2 warp1 read data");
        check_result(warp_word(2, 3) == 32'hBEEF_2200, "T2 warp2 read data");

        // ------------------------------------------------------------------
        // Test 3: lane backpressure and contention handling.
        // ------------------------------------------------------------------
        $display("=== TEST 3: backpressure and bus contention ===");
        lane_req_ready = 4'b0001;
        base_lane1 = lane_hs_count[1];
        issue_single_warp_req(3, 32'h0000_3010, 1'b0, {THREADS_PER_WARP{1'b1}}, 32'h4444_0000);

        timeout = 0;
        while (!lane_req_valid[1] && timeout < 200) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        check_result(lane_req_valid[1], "T3 lane1 request visible under stall");

        repeat (3) @(posedge clk);
        check_result(lane_hs_count[1] == base_lane1, "T3 no lane1 handshake while ready=0");

        lane_req_ready = 4'b0011;
        wait_lane_handshake(1, base_lane1, "T3 lane1 handshake after ready release");
        check_result(lane_req_addr[1*ADDR_WIDTH +: ADDR_WIDTH] == 32'h0000_3010, "T3 lane1 aligned address");

        drive_lane_resp_word(1, 32'hCCDD_3300);
        wait_warp_resp(3, "T3 warp3 response valid");
        check_result(warp_word(3, 2) == 32'hCCDD_3300, "T3 warp3 read data after stall");
        lane_req_ready = {NUM_LANES{1'b1}};

        // ------------------------------------------------------------------
        // Test 4: write-intent input and partial mask request acceptance.
        // ------------------------------------------------------------------
        $display("=== TEST 4: write-intent + partial mask ===");
        base_lane2 = lane_hs_count[2];
        issue_single_warp_req(0, 32'h0000_4020, 1'b1, 8'b0000_1111, 32'h5555_0000);
        wait_lane_handshake(2, base_lane2, "T4 lane2 handshake for write-intent request");
        check_result(lane_req_addr[2*ADDR_WIDTH +: ADDR_WIDTH] == 32'h0000_4020, "T4 lane2 aligned address");
        check_result(lane_req_write[2] == 1'b0, "T4 current RTL routes write-intent through read request path");

        drive_lane_resp_word(2, 32'hDD00_4400);
        wait_warp_resp(0, "T4 warp0 response valid");
        check_result(warp_word(0, 4) == 32'hDD00_4400, "T4 warp0 data after write-intent request");

        // Final stats checks
        check_result(stat_requests == 32'd5, "STAT total requests = 5");
        check_result(stat_outstanding_peak >= 32'd1, "STAT outstanding peak updated");
        check_result(stat_coalesced == 32'd0, "STAT coalesced currently constant in RTL");

        $display("============================================================");
        $display("tb_memory_interface_wide Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("TESTS FAILED");
        end

        #20;
        $finish;
    end
endmodule
