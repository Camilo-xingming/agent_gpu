`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_memory_controller_hbm;
    // Parameters
    parameter DATA_WIDTH        = 256;
    parameter BURST_LENGTH      = 4;
    parameter ADDR_WIDTH        = 32;

    reg clk;
    reg mem_clk;
    reg rst_n;

    // L2 Cache Interface
    reg                         l2_req_valid;
    reg                         l2_req_write;
    reg  [ADDR_WIDTH-1:0]       l2_req_addr;
    reg  [DATA_WIDTH*BURST_LENGTH-1:0] l2_req_wdata;
    reg  [DATA_WIDTH*BURST_LENGTH/8-1:0] l2_req_wmask;
    reg  [7:0]                  l2_req_id;
    wire                        l2_req_ready;

    wire                        l2_resp_valid;
    wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_resp_rdata;
    wire [7:0]                  l2_resp_id;

    // Stats
    wire [31:0] stat_read_count;
    wire [31:0] stat_write_count;
    wire [31:0] stat_row_hits;
    wire [31:0] stat_row_misses;
    wire [31:0] stat_row_conflicts;
    wire [31:0] stat_avg_latency;

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
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        mem_clk = 0;
        forever #5 mem_clk = ~mem_clk;
    end

    integer fail_count;

    initial begin
        fail_count = 0;
        // Initialize
        rst_n = 0;
        l2_req_valid = 0;
        l2_req_write = 0;
        l2_req_addr = 0;
        l2_req_wdata = 0;
        l2_req_wmask = { (DATA_WIDTH*BURST_LENGTH/8){1'b1} };
        l2_req_id = 0;

        #20;
        rst_n = 1;
        #20;

        // Test 1: Write request
        @(posedge clk);
        l2_req_valid = 1;
        l2_req_write = 1;
        l2_req_addr = 32'h0000_1000;
        l2_req_wdata = 1024'hDEADBEEF;
        l2_req_id = 8'h01;
        
        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);
        l2_req_valid = 0;

        // Wait for write to be processed
        #500;

        // Test 2: Read request
        @(posedge clk);
        l2_req_valid = 1;
        l2_req_write = 0;
        l2_req_addr = 32'h0000_1000;
        l2_req_id = 8'h02;

        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);
        l2_req_valid = 0;

        // Wait for read response
        wait(l2_resp_valid);
        if (l2_resp_id === 8'h02 && l2_resp_rdata === 1024'hDEADBEEF) begin
            $display("PASS: Read response correct");
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL: Read response incorrect. ID: %h, Data: %h", l2_resp_id, l2_resp_rdata);
        end

        @(posedge clk);

        // Test 3: Multiple channel requests
        // Channel 0
        l2_req_valid = 1;
        l2_req_write = 1;
        l2_req_addr = 32'h0000_2000; // CH0
        l2_req_wdata = 1024'hCAFEBABE;
        l2_req_id = 8'h03;
        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);

        // Channel 1
        l2_req_addr = 32'h0002_0000; // CH1
        l2_req_wdata = 1024'h11112222;
        l2_req_id = 8'h04;
        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);

        // Read them back
        l2_req_valid = 1;
        l2_req_write = 0;
        l2_req_addr = 32'h0000_2000; // CH0
        l2_req_id = 8'h05;
        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);

        l2_req_addr = 32'h0002_0000; // CH1
        l2_req_id = 8'h06;
        @(posedge clk);
        while (!l2_req_ready) @(posedge clk);

        l2_req_valid = 0;

        // Wait for both reads
        wait(l2_resp_valid && l2_resp_id == 8'h05);
        if (l2_resp_rdata === 1024'hCAFEBABE) begin
            $display("PASS: CH0 read");
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL: CH0 read. Expected %h, got %h", 1024'hCAFEBABE, l2_resp_rdata);
        end

        @(posedge clk);
        wait(l2_resp_valid && l2_resp_id == 8'h06);
        if (l2_resp_rdata === 1024'h11112222) begin
            $display("PASS: CH1 read");
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL: CH1 read. Expected %h, got %h", 1024'h11112222, l2_resp_rdata);
        end

        #1000;
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("TESTS FAILED with %0d errors", fail_count);
        end
        $finish;
    end
endmodule
