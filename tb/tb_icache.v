`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_icache;

    localparam CLK_PERIOD   = 10;
    localparam ADDR_WIDTH   = 32;
    localparam DATA_WIDTH   = 32;
    localparam LINE_SIZE    = 16;
    localparam LINE_BITS    = LINE_SIZE * 8;
    localparam WORDS_PER_LINE = LINE_SIZE / (DATA_WIDTH/8);

    reg clk;
    reg rst_n;

    reg fetch_req;
    reg [ADDR_WIDTH-1:0] fetch_addr;
    wire fetch_ready;
    wire [DATA_WIDTH-1:0] fetch_data;
    wire [LINE_BITS-1:0] fetch_line_data;
    wire fetch_valid;

    wire fetch_hit_bypass;
    wire [DATA_WIDTH-1:0] fetch_hit_bypass_data;
    wire [LINE_BITS-1:0] fetch_hit_bypass_line_data;

    reg invalidate_req;
    reg [ADDR_WIDTH-1:0] invalidate_addr;
    reg invalidate_all;
    wire invalidate_done;

    wire mem_req_valid;
    wire [ADDR_WIDTH-1:0] mem_req_addr;
    reg mem_req_ready;
    reg [LINE_BITS-1:0] mem_resp_data;
    reg mem_resp_valid;

    wire [31:0] stat_hits;
    wire [31:0] stat_misses;
    wire [31:0] stat_prefetch_hits;

    integer pass_count;
    integer fail_count;
    integer mem_req_count;

    reg pending_mem_resp;
    reg [1:0] mem_resp_delay;
    reg [ADDR_WIDTH-1:0] pending_line_addr;

    reg seen_fetch;
    reg [DATA_WIDTH-1:0] seen_fetch_data;
    reg seen_invalidate_done;

    icache #(
        .SIZE_KB(1),
        .LINE_SIZE(LINE_SIZE),
        .NUM_WAYS(2),
        .PREFETCH_DEPTH(4),
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .fetch_req(fetch_req),
        .fetch_addr(fetch_addr),
        .fetch_ready(fetch_ready),
        .fetch_data(fetch_data),
        .fetch_line_data(fetch_line_data),
        .fetch_valid(fetch_valid),
        .fetch_hit_bypass(fetch_hit_bypass),
        .fetch_hit_bypass_data(fetch_hit_bypass_data),
        .fetch_hit_bypass_line_data(fetch_hit_bypass_line_data),
        .invalidate_req(invalidate_req),
        .invalidate_addr(invalidate_addr),
        .invalidate_all(invalidate_all),
        .invalidate_done(invalidate_done),
        .mem_req_valid(mem_req_valid),
        .mem_req_addr(mem_req_addr),
        .mem_req_ready(mem_req_ready),
        .mem_resp_data(mem_resp_data),
        .mem_resp_valid(mem_resp_valid),
        .stat_hits(stat_hits),
        .stat_misses(stat_misses),
        .stat_prefetch_hits(stat_prefetch_hits)
    );

    function [LINE_BITS-1:0] make_line;
        input [ADDR_WIDTH-1:0] base_addr;
        integer i;
        reg [LINE_BITS-1:0] line;
        begin
            line = {LINE_BITS{1'b0}};
            for (i = 0; i < WORDS_PER_LINE; i = i + 1) begin
                line[i*DATA_WIDTH +: DATA_WIDTH] = base_addr + (i * 4);
            end
            make_line = line;
        end
    endfunction

    function [DATA_WIDTH-1:0] expected_word;
        input [ADDR_WIDTH-1:0] addr;
        reg [ADDR_WIDTH-1:0] line_addr;
        reg [1:0] word_idx;
        begin
            line_addr = {addr[ADDR_WIDTH-1:4], 4'b0000};
            word_idx = addr[3:2];
            expected_word = line_addr + (word_idx * 4);
        end
    endfunction

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always @(posedge clk) begin
        mem_resp_valid <= 1'b0;

        if (fetch_valid) begin
            seen_fetch <= 1'b1;
            seen_fetch_data <= fetch_data;
        end

        if (fetch_hit_bypass) begin
            seen_fetch <= 1'b1;
            seen_fetch_data <= fetch_hit_bypass_data;
        end

        if (invalidate_done) begin
            seen_invalidate_done <= 1'b1;
        end

        if (mem_req_valid && mem_req_ready) begin
            pending_mem_resp <= 1'b1;
            mem_resp_delay <= 2'd2;
            pending_line_addr <= mem_req_addr;
            mem_req_count <= mem_req_count + 1;
        end

        if (pending_mem_resp) begin
            if (mem_resp_delay == 0) begin
                mem_resp_valid <= 1'b1;
                mem_resp_data <= make_line(pending_line_addr);
                pending_mem_resp <= 1'b0;
            end else begin
                mem_resp_delay <= mem_resp_delay - 1'b1;
            end
        end
    end

    task automatic clear_fetch_capture;
        begin
            seen_fetch = 1'b0;
            seen_fetch_data = {DATA_WIDTH{1'b0}};
        end
    endtask

    task automatic clear_invalidate_capture;
        begin
            seen_invalidate_done = 1'b0;
        end
    endtask

    task automatic check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                $display("  PASS: %s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task automatic issue_fetch;
        input [ADDR_WIDTH-1:0] addr;
        begin
            @(negedge clk);
            fetch_addr <= addr;
            fetch_req <= 1'b1;
            @(negedge clk);
            fetch_req <= 1'b0;
        end
    endtask

    task automatic wait_seen_fetch;
        input integer timeout_cycles;
        output reg got;
        integer i;
        begin
            got = 1'b0;
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (seen_fetch) begin
                    got = 1'b1;
                    i = timeout_cycles;
                end
            end
        end
    endtask

    task automatic wait_seen_invalidate;
        input integer timeout_cycles;
        output reg got;
        integer i;
        begin
            got = 1'b0;
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (seen_invalidate_done) begin
                    got = 1'b1;
                    i = timeout_cycles;
                end
            end
        end
    endtask

    task automatic wait_for_mem_req_delta;
        input integer base;
        input integer delta;
        input integer timeout_cycles;
        output reg got;
        integer i;
        begin
            got = 1'b0;
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (mem_req_count >= (base + delta)) begin
                    got = 1'b1;
                    i = timeout_cycles;
                end
            end
        end
    endtask

    task automatic wait_bus_quiescent;
        input integer timeout_cycles;
        output reg got;
        integer i;
        begin
            got = 1'b0;
            for (i = 0; i < timeout_cycles; i = i + 1) begin
                @(posedge clk);
                if (!pending_mem_resp && !mem_req_valid) begin
                    got = 1'b1;
                    i = timeout_cycles;
                end
            end
        end
    endtask

    initial begin
        reg got;
        reg [ADDR_WIDTH-1:0] addr0;
        reg [ADDR_WIDTH-1:0] addr0_next;
        reg [DATA_WIDTH-1:0] exp0;
        reg [DATA_WIDTH-1:0] exp1;
        integer req_before;

        pass_count = 0;
        fail_count = 0;
        mem_req_count = 0;

        fetch_req = 1'b0;
        fetch_addr = 0;
        invalidate_req = 1'b0;
        invalidate_addr = 0;
        invalidate_all = 1'b0;
        mem_req_ready = 1'b1;
        mem_resp_data = 0;
        mem_resp_valid = 1'b0;

        pending_mem_resp = 1'b0;
        mem_resp_delay = 0;
        pending_line_addr = 0;

        clear_fetch_capture();
        clear_invalidate_capture();

        rst_n = 1'b0;
        #(CLK_PERIOD * 4);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        addr0 = 32'h0000_0048;
        addr0_next = 32'h0000_0050;
        exp0 = expected_word(addr0);
        exp1 = expected_word(addr0_next);

        $display("============================================");
        $display("ICache Testbench");
        $display("============================================");

        $display("\n[TEST 1] Demand miss fill");
        clear_fetch_capture();
        req_before = mem_req_count;
        issue_fetch(addr0);
        wait_seen_fetch(40, got);
        check(got, "fetch response observed after miss");
        check(seen_fetch_data == exp0, "fetch_data matches memory line content");
        check(mem_req_count >= req_before + 1, "memory request issued for demand miss");

        $display("\n[TEST 2] Re-fetch same address");
        clear_fetch_capture();
        req_before = mem_req_count;
        issue_fetch(addr0);
        wait_seen_fetch(20, got);
        check(got, "second fetch completed via hit/bypass");
        check(seen_fetch_data == exp0, "second fetch returns expected word");
        check(mem_req_count <= req_before + 1, "second fetch does not trigger extra demand miss");

        $display("\n[TEST 3] Prefetch hit on next line");
        wait_for_mem_req_delta(req_before, 1, 40, got);
        check(got, "deferred prefetch request observed");
        repeat (10) @(posedge clk);

        clear_fetch_capture();
        req_before = mem_req_count;
        issue_fetch(addr0_next);
        wait_seen_fetch(20, got);
        check(got, "next-line fetch completed");
        check(seen_fetch_data == exp1, "next-line fetch returns expected word");
        check(stat_prefetch_hits > 0, "prefetch hit counter increments");
        check(mem_req_count == req_before, "prefetched line served without new memory request");

        $display("\n[TEST 4] Invalidate all");
        wait_bus_quiescent(40, got);
        check(got, "memory side is quiescent before invalidate");

        clear_invalidate_capture();
        @(negedge clk);
        invalidate_req <= 1'b1;
        invalidate_all <= 1'b1;
        @(negedge clk);
        @(negedge clk);
        invalidate_req <= 1'b0;
        invalidate_all <= 1'b0;

        wait_seen_invalidate(20, got);
        check(got, "invalidate_done observed");

        clear_fetch_capture();
        req_before = mem_req_count;
        issue_fetch(addr0);
        wait_seen_fetch(40, got);
        check(got, "fetch response observed after invalidate_all");
        check(mem_req_count >= req_before + 1, "memory request re-issued after flush");

        $display("\n============================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("============================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_icache failed with %0d checks", fail_count);
        end
    end

endmodule
