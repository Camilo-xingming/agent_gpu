`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_memory_controller_hbm;
    parameter DATA_WIDTH   = 256;
    parameter BURST_LENGTH = 4;
    parameter ADDR_WIDTH   = 32;

    localparam BURST_BITS = DATA_WIDTH * BURST_LENGTH;
    localparam MASK_BITS  = BURST_BITS / 8;

    reg clk;
    reg mem_clk;
    reg rst_n;

    reg                         l2_req_valid;
    reg                         l2_req_write;
    reg  [ADDR_WIDTH-1:0]       l2_req_addr;
    reg  [BURST_BITS-1:0]       l2_req_wdata;
    reg  [MASK_BITS-1:0]        l2_req_wmask;
    reg  [7:0]                  l2_req_id;
    wire                        l2_req_ready;

    wire                        l2_resp_valid;
    wire [BURST_BITS-1:0]       l2_resp_rdata;
    wire [7:0]                  l2_resp_id;

    wire [31:0] stat_read_count;
    wire [31:0] stat_write_count;
    wire [31:0] stat_row_hits;
    wire [31:0] stat_row_misses;
    wire [31:0] stat_row_conflicts;
    wire [31:0] stat_avg_latency;

    integer pass_count;
    integer fail_count;
    integer i;

    reg [255:0] resp_seen;
    reg [BURST_BITS-1:0] resp_data [0:255];

    memory_controller_hbm #(
        .DATA_WIDTH(DATA_WIDTH),
        .BURST_LENGTH(BURST_LENGTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .mem_clk(mem_clk),
        .rst_n(rst_n),
        .l2_req_valid(l2_req_valid),
        .l2_req_write(l2_req_write),
        .l2_req_addr(l2_req_addr),
        .l2_req_wdata(l2_req_wdata),
        .l2_req_wmask(l2_req_wmask),
        .l2_req_id(l2_req_id),
        .l2_req_ready(l2_req_ready),
        .l2_resp_valid(l2_resp_valid),
        .l2_resp_rdata(l2_resp_rdata),
        .l2_resp_id(l2_resp_id),
        .stat_read_count(stat_read_count),
        .stat_write_count(stat_write_count),
        .stat_row_hits(stat_row_hits),
        .stat_row_misses(stat_row_misses),
        .stat_row_conflicts(stat_row_conflicts),
        .stat_avg_latency(stat_avg_latency)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        mem_clk = 1'b0;
        forever #5 mem_clk = ~mem_clk;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            resp_seen <= 256'd0;
        end else if (l2_resp_valid) begin
            resp_seen[l2_resp_id] <= 1'b1;
            resp_data[l2_resp_id] <= l2_resp_rdata;
        end
    end

    task expect_true;
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

    task expect_eq32;
        input [31:0] got;
        input [31:0] exp;
        input [255:0] msg;
        begin
            if (got === exp) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s exp=%0d got=%0d", msg, exp, got);
            end
        end
    endtask

    task expect_eq_data;
        input [BURST_BITS-1:0] got;
        input [BURST_BITS-1:0] exp;
        input [255:0] msg;
        begin
            if (got === exp) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s exp_lo=0x%08h got_lo=0x%08h", msg, exp[31:0], got[31:0]);
            end
        end
    endtask

    task clear_resp_tracking;
        integer idx;
        begin
            resp_seen = 256'd0;
            for (idx = 0; idx < 256; idx = idx + 1)
                resp_data[idx] = {BURST_BITS{1'b0}};
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            l2_req_valid = 1'b0;
            l2_req_write = 1'b0;
            l2_req_addr  = {ADDR_WIDTH{1'b0}};
            l2_req_wdata = {BURST_BITS{1'b0}};
            l2_req_wmask = {MASK_BITS{1'b1}};
            l2_req_id    = 8'h00;
            clear_resp_tracking();
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task issue_req;
        input                      wr;
        input [ADDR_WIDTH-1:0]     addr;
        input [BURST_BITS-1:0]     data;
        input [MASK_BITS-1:0]      mask;
        input [7:0]                rid;
        input [255:0]              name;
        integer guard;
        integer accepted;
        begin
            l2_req_valid = 1'b1;
            l2_req_write = wr;
            l2_req_addr  = addr;
            l2_req_wdata = data;
            l2_req_wmask = mask;
            l2_req_id    = rid;

            guard = 0;
            accepted = 0;
            while ((guard < 2000) && (accepted == 0)) begin
                @(posedge clk);
                if (l2_req_ready)
                    accepted = 1;
                guard = guard + 1;
            end

            l2_req_valid = 1'b0;
            if (!accepted) begin
                fail_count = fail_count + 1;
                $display("[FAIL] issue_req timeout: %0s", name);
            end
        end
    endtask

    task wait_resp;
        input [7:0] rid;
        input integer timeout_cycles;
        input [255:0] name;
        integer c;
        begin
            c = 0;
            while (!resp_seen[rid] && (c < timeout_cycles)) begin
                @(posedge clk);
                c = c + 1;
            end
            expect_true(resp_seen[rid], name);
        end
    endtask

    initial begin
        reg [BURST_BITS-1:0] data_a;
        reg [BURST_BITS-1:0] data_b;
        reg [BURST_BITS-1:0] data_c;
        reg [31:0] read_before;
        reg [31:0] write_before;
        reg [31:0] seed;
        integer k;
        integer accepted_cnt;
        integer saw_backpressure;

        pass_count = 0;
        fail_count = 0;

        data_a = {32{32'hDEAD_BEEF}};
        data_b = {32{32'hCAFE_BABE}};
        data_c = {32{32'h1111_2222}};

        //============================================================
        // Test 1: basic write/read transaction
        //============================================================
        reset_dut();
        issue_req(1'b1, 32'h0000_1000, data_a, {MASK_BITS{1'b1}}, 8'h01, "t1 write");
        issue_req(1'b0, 32'h0000_1000, {BURST_BITS{1'b0}}, {MASK_BITS{1'b1}}, 8'h02, "t1 read");
        wait_resp(8'h02, 3000, "t1 read response");
        expect_eq_data(resp_data[8'h02], data_a, "t1 readback data");
        expect_true(stat_write_count >= 1, "t1 write_count increment");
        expect_true(stat_read_count  >= 1, "t1 read_count increment");

        //============================================================
        // Test 2: burst payload + channel interleaving
        //============================================================
        clear_resp_tracking();
        issue_req(1'b1, 32'h0000_2000, data_b, {MASK_BITS{1'b1}}, 8'h10, "t2 write ch0");
        issue_req(1'b1, 32'h0002_0000, data_c, {MASK_BITS{1'b1}}, 8'h11, "t2 write ch1");
        repeat (400) @(posedge clk);
        issue_req(1'b0, 32'h0002_0000, {BURST_BITS{1'b0}}, {MASK_BITS{1'b1}}, 8'h12, "t2 read ch1");
        issue_req(1'b0, 32'h0000_2000, {BURST_BITS{1'b0}}, {MASK_BITS{1'b1}}, 8'h13, "t2 read ch0");

        wait_resp(8'h12, 4000, "t2 ch1 response");
        wait_resp(8'h13, 4000, "t2 ch0 response");
        expect_eq_data(resp_data[8'h12], data_c, "t2 ch1 data");
        expect_eq_data(resp_data[8'h13], data_b, "t2 ch0 data");

        //============================================================
        // Test 3: request queuing/arbitration (same channel burst of requests)
        //============================================================
        clear_resp_tracking();
        read_before  = stat_read_count;
        write_before = stat_write_count;

        for (k = 0; k < 3; k = k + 1) begin
            seed = 32'hA500_0000 + k;
            issue_req(1'b1,
                      32'h0000_3000 + (k * 32'h0000_0100),
                      {32{seed}},
                      {MASK_BITS{1'b1}},
                      (8'h20 + k[7:0]),
                      "t3 write queue");
        end

        for (k = 0; k < 3; k = k + 1) begin
            issue_req(1'b0,
                      32'h0000_3000 + (k * 32'h0000_0100),
                      {BURST_BITS{1'b0}},
                      {MASK_BITS{1'b1}},
                      (8'h30 + k[7:0]),
                      "t3 read queue");
        end

        seed = 32'hA500_0000;
        wait_resp(8'h30, 12000, "t3 first read response");
        expect_eq_data(resp_data[8'h30], {32{seed}}, "t3 first read data");
        expect_true(resp_seen[8'h30] || resp_seen[8'h31] || resp_seen[8'h32], "t3 queued read response observed");

        expect_true(stat_write_count >= (write_before + 3), "t3 write_count delta");
        expect_true(stat_read_count  >= (read_before  + 3), "t3 read_count delta");

        //============================================================
        // Test 4: back-to-back flood + full-queue/backpressure behavior
        //============================================================
        clear_resp_tracking();
        l2_req_valid = 1'b1;
        l2_req_write = 1'b1;
        l2_req_wmask = {MASK_BITS{1'b1}};
        l2_req_addr  = 32'h0000_4000;
        l2_req_id    = 8'h80;
        seed         = 32'hC000_0000;
        l2_req_wdata = {32{seed}};

        accepted_cnt = 0;
        saw_backpressure = 0;
        for (k = 0; k < 300; k = k + 1) begin
            @(posedge clk);
            if (l2_req_ready) begin
                accepted_cnt = accepted_cnt + 1;
                l2_req_id = l2_req_id + 1'b1;
                l2_req_addr = l2_req_addr + 32'h0000_0100;
                seed = seed + 32'h0000_0001;
                l2_req_wdata = {32{seed}};
            end else begin
                saw_backpressure = 1;
                k = 300;
            end
        end
        l2_req_valid = 1'b0;

        expect_true(accepted_cnt > 0, "t4 accepted back-to-back requests");
        expect_true(saw_backpressure, "t4 should observe l2_req_ready deassert (queue pressure)");

        // allow scheduler to drain and ready recover
        repeat (2000) @(posedge clk);
        expect_true(l2_req_ready, "t4 ready recovers after drain");

        //============================================================
        // Final summary
        //============================================================
        $display("============================================================");
        $display("tb_memory_controller_hbm Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $display("TESTS FAILED with %0d errors", fail_count);
            $fatal(1, "tb_memory_controller_hbm failed");
        end
    end
endmodule
