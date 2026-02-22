`timescale 1ns / 1ps

module tb_memory_coalescing_unit;

    localparam THREADS = 4;
    localparam DATA_WIDTH = 32;
    localparam ADDR_WIDTH = 32;
    localparam CACHE_LINE_SIZE = 16;
    localparam MAX_COALESCED = 4;
    localparam LINE_BITS = CACHE_LINE_SIZE * 8;

    reg clk;
    reg rst_n;

    reg req_valid;
    reg req_write;
    reg [THREADS*ADDR_WIDTH-1:0] req_addr;
    reg [THREADS*DATA_WIDTH-1:0] req_wdata;
    reg [THREADS-1:0] req_mask;

    wire mem_req_valid;
    wire mem_req_write;
    wire [ADDR_WIDTH-1:0] mem_req_addr;
    wire [LINE_BITS-1:0] mem_req_wdata;
    wire [CACHE_LINE_SIZE-1:0] mem_req_wmask;
    reg  [LINE_BITS-1:0] mem_resp_rdata;
    reg mem_resp_valid;

    wire [THREADS*DATA_WIDTH-1:0] resp_rdata;
    wire resp_valid;
    wire ready;

    wire [31:0] stat_requests;
    wire [31:0] stat_transactions;
    wire [31:0] stat_coalesce_ratio;

    reg pending_resp;
    reg [ADDR_WIDTH-1:0] pending_addr;

    integer pass_count;
    integer fail_count;
    integer test_num;
    integer tx_count;
    integer tx_count_prev;
    integer i;

    reg last_req_write;
    reg [ADDR_WIDTH-1:0] last_req_addr;
    reg [CACHE_LINE_SIZE-1:0] last_req_wmask;

    memory_coalescing_unit #(
        .THREADS(THREADS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH),
        .CACHE_LINE_SIZE(CACHE_LINE_SIZE),
        .MAX_COALESCED(MAX_COALESCED)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_wdata(req_wdata),
        .req_mask(req_mask),
        .mem_req_valid(mem_req_valid),
        .mem_req_write(mem_req_write),
        .mem_req_addr(mem_req_addr),
        .mem_req_wdata(mem_req_wdata),
        .mem_req_wmask(mem_req_wmask),
        .mem_resp_rdata(mem_resp_rdata),
        .mem_resp_valid(mem_resp_valid),
        .resp_rdata(resp_rdata),
        .resp_valid(resp_valid),
        .ready(ready),
        .stat_requests(stat_requests),
        .stat_transactions(stat_transactions),
        .stat_coalesce_ratio(stat_coalesce_ratio)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [LINE_BITS-1:0] make_line_data;
        input [ADDR_WIDTH-1:0] line_base;
        integer word_idx;
        begin
            make_line_data = {LINE_BITS{1'b0}};
            for (word_idx = 0; word_idx < (CACHE_LINE_SIZE/4); word_idx = word_idx + 1) begin
                make_line_data[word_idx*32 +: 32] = 32'h1000_0000 + line_base + (word_idx * 4);
            end
        end
    endfunction

    task check_result;
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

    task set_thread_req;
        input integer tid;
        input [31:0] addr;
        input [31:0] data;
        begin
            req_addr[tid*ADDR_WIDTH +: ADDR_WIDTH] = addr;
            req_wdata[tid*DATA_WIDTH +: DATA_WIDTH] = data;
        end
    endtask

    task issue_req_and_wait_resp;
        begin
            wait (ready === 1'b1);
            @(negedge clk);
            req_valid <= 1'b1;
            @(negedge clk);
            req_valid <= 1'b0;
            wait (resp_valid === 1'b1);
            #1;
        end
    endtask

    // Simple memory model: one-cycle response latency per request
    always @(posedge clk) begin
        mem_resp_valid <= 1'b0;

        if (mem_req_valid) begin
            tx_count <= tx_count + 1;
            last_req_write <= mem_req_write;
            last_req_addr <= mem_req_addr;
            last_req_wmask <= mem_req_wmask;
            pending_resp <= 1'b1;
            pending_addr <= mem_req_addr;
        end

        if (pending_resp) begin
            mem_resp_valid <= 1'b1;
            mem_resp_rdata <= make_line_data(pending_addr);
            pending_resp <= 1'b0;
        end
    end

    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;
        tx_count = 0;
        tx_count_prev = 0;

        rst_n = 1'b0;
        req_valid = 1'b0;
        req_write = 1'b0;
        req_addr = 0;
        req_wdata = 0;
        req_mask = 0;
        mem_resp_rdata = 0;
        mem_resp_valid = 1'b0;
        pending_resp = 1'b0;
        pending_addr = 0;
        last_req_write = 1'b0;
        last_req_addr = 0;
        last_req_wmask = 0;

        #40;
        rst_n = 1'b1;
        #20;

        $display("====================================================");
        $display("Memory Coalescing Unit Testbench");
        $display("====================================================");

        // Test 1: Four threads in same cache line => one transaction
        test_num = test_num + 1;
        $display("\n[TEST %0d] Single-line read coalescing", test_num);
        req_write = 1'b0;
        req_mask = 4'b1111;
        set_thread_req(0, 32'h0000_1000, 32'hAAAA_0000);
        set_thread_req(1, 32'h0000_1004, 32'hAAAA_0001);
        set_thread_req(2, 32'h0000_1008, 32'hAAAA_0002);
        set_thread_req(3, 32'h0000_100C, 32'hAAAA_0003);
        tx_count_prev = tx_count;
        issue_req_and_wait_resp();

        check_result((tx_count - tx_count_prev) == 1, "coalesced to one memory transaction");
        check_result(resp_rdata[0*32 +: 32] == (32'h1000_0000 + 32'h0000_1000), "thread0 read data correct");
        check_result(resp_rdata[1*32 +: 32] == (32'h1000_0000 + 32'h0000_1004), "thread1 read data correct");
        check_result(resp_rdata[2*32 +: 32] == (32'h1000_0000 + 32'h0000_1008), "thread2 read data correct");
        check_result(resp_rdata[3*32 +: 32] == (32'h1000_0000 + 32'h0000_100C), "thread3 read data correct");

        // Test 2: Accesses to two cache lines => two transactions
        test_num = test_num + 1;
        $display("\n[TEST %0d] Two-line read coalescing", test_num);
        req_write = 1'b0;
        req_mask = 4'b1111;
        set_thread_req(0, 32'h0000_2000, 32'hBBBB_0000);
        set_thread_req(1, 32'h0000_2004, 32'hBBBB_0001);
        set_thread_req(2, 32'h0000_2080, 32'hBBBB_0002);
        set_thread_req(3, 32'h0000_2084, 32'hBBBB_0003);
        tx_count_prev = tx_count;
        issue_req_and_wait_resp();

        check_result((tx_count - tx_count_prev) == 2, "split into two memory transactions");
        check_result(resp_rdata[0*32 +: 32] == (32'h1000_0000 + 32'h0000_2000), "thread0 line0 read data correct");
        check_result(resp_rdata[2*32 +: 32] == (32'h1000_0000 + 32'h0000_2080), "thread2 line1 read data correct");

        // Test 3: Write coalescing and byte mask generation
        test_num = test_num + 1;
        $display("\n[TEST %0d] Write coalescing and write mask", test_num);
        req_write = 1'b1;
        req_mask = 4'b0101;
        set_thread_req(0, 32'h0000_3000, 32'hCAFE_0000);
        set_thread_req(1, 32'h0000_3004, 32'hCAFE_0001);
        set_thread_req(2, 32'h0000_3008, 32'hCAFE_0002);
        set_thread_req(3, 32'h0000_300C, 32'hCAFE_0003);
        tx_count_prev = tx_count;
        issue_req_and_wait_resp();

        check_result((tx_count - tx_count_prev) == 1, "write request coalesced into one transaction");
        check_result(last_req_write == 1'b1, "memory request marked as write");
        check_result(last_req_addr == 32'h0000_3000, "write request uses cache-line base address");
        check_result(last_req_wmask[3:0] == 4'hF, "thread0 bytes enabled in write mask");
        check_result(last_req_wmask[11:8] == 4'hF, "thread2 bytes enabled in write mask");

        // Basic statistics sanity
        check_result(stat_requests == 32'd3, "request counter tracks three warp requests");
        check_result(stat_transactions == tx_count, "transaction counter matches observed requests");

        $display("\n====================================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("====================================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_memory_coalescing_unit failed with %0d checks", fail_count);
        end
    end

endmodule
