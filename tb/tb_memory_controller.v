`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_memory_controller;
    localparam DATA_WIDTH   = `MEM_DATA_WIDTH;
    localparam NUM_CHANNELS = `MEM_NUM_CHANNELS;
    localparam BURST_LENGTH = `MEM_BURST_LENGTH;
    localparam ADDR_WIDTH   = 32;
    localparam BURST_BITS   = DATA_WIDTH * BURST_LENGTH;
    localparam BURST_BYTES  = BURST_BITS / 8;

    reg clk;
    reg mem_clk;
    reg rst_n;

    reg l2_req_valid;
    reg l2_req_write;
    reg [ADDR_WIDTH-1:0] l2_req_addr;
    reg [BURST_BITS-1:0] l2_req_wdata;
    reg [BURST_BYTES-1:0] l2_req_wmask;
    wire l2_req_ready;
    wire l2_resp_valid;
    wire [BURST_BITS-1:0] l2_resp_rdata;

    wire [NUM_CHANNELS-1:0] mem_cs_n;
    wire [NUM_CHANNELS-1:0] mem_ras_n;
    wire [NUM_CHANNELS-1:0] mem_cas_n;
    wire [NUM_CHANNELS-1:0] mem_we_n;
    wire [NUM_CHANNELS*17-1:0] mem_addr;
    wire [NUM_CHANNELS*3-1:0] mem_ba;
    wire [NUM_CHANNELS*2-1:0] mem_bg;
    wire [NUM_CHANNELS*DATA_WIDTH-1:0] mem_dq_out;
    reg  [NUM_CHANNELS*DATA_WIDTH-1:0] mem_dq_in;
    wire [NUM_CHANNELS-1:0] mem_dq_oe;
    wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_out;
    reg  [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_in;
    wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dm;

    wire [31:0] stat_read_count;
    wire [31:0] stat_write_count;
    wire [31:0] stat_row_hits;
    wire [31:0] stat_row_misses;

    localparam [BURST_BYTES-1:0] FULL_MASK = {BURST_BYTES{1'b1}};

    reg [BURST_BITS-1:0] expected_rdata [0:1];
    reg [BURST_BITS-1:0] observed_rdata [0:1];
    integer expected_count;
    integer observed_count;
    integer timeout;

    function [BURST_BITS-1:0] make_pattern;
        input [7:0] seed;
        integer idx;
        begin
            for (idx = 0; idx < BURST_BYTES; idx = idx + 1)
                make_pattern[idx*8 +: 8] = seed + idx[7:0];
        end
    endfunction

    task automatic send_req;
        input req_write;
        input [ADDR_WIDTH-1:0] req_addr;
        input [BURST_BITS-1:0] req_wdata;
        input [BURST_BYTES-1:0] req_wmask;
        begin
            while (!l2_req_ready)
                @(posedge clk);

            l2_req_valid = 1'b1;
            l2_req_write = req_write;
            l2_req_addr  = req_addr;
            l2_req_wdata = req_wdata;
            l2_req_wmask = req_wmask;

            @(posedge clk);

            l2_req_valid = 1'b0;
            l2_req_write = 1'b0;
            l2_req_addr  = {ADDR_WIDTH{1'b0}};
            l2_req_wdata = {BURST_BITS{1'b0}};
            l2_req_wmask = {BURST_BYTES{1'b0}};
        end
    endtask

    memory_controller dut (
        .clk(clk),
        .mem_clk(mem_clk),
        .rst_n(rst_n),
        .l2_req_valid(l2_req_valid),
        .l2_req_write(l2_req_write),
        .l2_req_addr(l2_req_addr),
        .l2_req_wdata(l2_req_wdata),
        .l2_req_wmask(l2_req_wmask),
        .l2_req_ready(l2_req_ready),
        .l2_resp_valid(l2_resp_valid),
        .l2_resp_rdata(l2_resp_rdata),
        .mem_cs_n(mem_cs_n),
        .mem_ras_n(mem_ras_n),
        .mem_cas_n(mem_cas_n),
        .mem_we_n(mem_we_n),
        .mem_addr(mem_addr),
        .mem_ba(mem_ba),
        .mem_bg(mem_bg),
        .mem_dq_out(mem_dq_out),
        .mem_dq_in(mem_dq_in),
        .mem_dq_oe(mem_dq_oe),
        .mem_dqs_out(mem_dqs_out),
        .mem_dqs_in(mem_dqs_in),
        .mem_dm(mem_dm),
        .stat_read_count(stat_read_count),
        .stat_write_count(stat_write_count),
        .stat_row_hits(stat_row_hits),
        .stat_row_misses(stat_row_misses)
    );

    always #5 clk = ~clk;
    always #3 mem_clk = ~mem_clk;

    always @(posedge clk) begin
        if (!rst_n) begin
            observed_count <= 0;
        end else if (l2_resp_valid && observed_count < 2) begin
            observed_rdata[observed_count] <= l2_resp_rdata;
            observed_count <= observed_count + 1;
        end
    end

    initial begin
        clk = 1'b0;
        mem_clk = 1'b0;
        rst_n = 1'b0;
        l2_req_valid = 1'b0;
        l2_req_write = 1'b0;
        l2_req_addr  = {ADDR_WIDTH{1'b0}};
        l2_req_wdata = {BURST_BITS{1'b0}};
        l2_req_wmask = {BURST_BYTES{1'b0}};
        mem_dq_in = {(NUM_CHANNELS*DATA_WIDTH){1'b0}};
        mem_dqs_in = {(NUM_CHANNELS*DATA_WIDTH/8){1'b0}};
        expected_count = 0;
        observed_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        expected_rdata[0] = make_pattern(8'h11);
        expected_rdata[1] = make_pattern(8'hA0);
        expected_count = 2;

        send_req(1'b1, 32'h0000_0100, expected_rdata[0], FULL_MASK);
        send_req(1'b0, 32'h0000_0100, {BURST_BITS{1'b0}}, {BURST_BYTES{1'b0}});
        send_req(1'b1, 32'h0000_0200, expected_rdata[1], FULL_MASK);
        send_req(1'b0, 32'h0000_0200, {BURST_BITS{1'b0}}, {BURST_BYTES{1'b0}});

        timeout = 0;
        while ((observed_count < expected_count) && (timeout < 500)) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (observed_count != expected_count) begin
            $display("FAIL: Expected %0d responses, got %0d", expected_count, observed_count);
            $finish_and_return(1);
        end

        if (observed_rdata[0] !== expected_rdata[0]) begin
            $display("FAIL: Response[0] mismatch");
            $finish_and_return(1);
        end

        if (observed_rdata[1] !== expected_rdata[1]) begin
            $display("FAIL: Response[1] mismatch");
            $finish_and_return(1);
        end

        if (stat_read_count != 2 || stat_write_count != 2) begin
            $display("FAIL: Stats mismatch read=%0d write=%0d", stat_read_count, stat_write_count);
            $finish_and_return(1);
        end

        $display("PASS: memory_controller read response path + dequeue behavior verified");
        $finish;
    end
endmodule
