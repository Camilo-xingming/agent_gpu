`timescale 1ns / 1ps

module tb_l1_data_cache_optimized;

    localparam CLK_PERIOD = 10;
    localparam THREADS = 4;
    localparam LINE_SIZE_BYTES = 128;
    localparam WORDS_PER_LINE = LINE_SIZE_BYTES / 4;
    localparam MAX_OUTSTANDING = 4;

    reg clk;
    reg rst_n;

    // Request interface
    reg                    req_valid;
    reg                    req_write;
    reg [THREADS*32-1:0]   req_addr;
    reg [THREADS*32-1:0]   req_wdata;
    reg [THREADS-1:0]      req_mask;
    wire [THREADS*32-1:0]  resp_rdata;
    wire                   resp_valid;
    wire                   resp_hit;

    // Memory interface
    wire                          mem_req;
    wire                          mem_write;
    wire [31:0]                   mem_addr;
    wire [LINE_SIZE_BYTES*8-1:0]  mem_wdata;
    reg  [LINE_SIZE_BYTES*8-1:0]  mem_rdata;
    reg                           mem_valid;
    reg                           mem_ready;
    reg  [3:0]                    mem_id;

    // Prefetch interface
    reg                           prefetch_enable;
    wire [31:0]                   prefetch_addr;
    wire                          prefetch_active;

    // Stats
    wire [31:0] stat_hits;
    wire [31:0] stat_misses;
    wire [31:0] stat_prefetch_hits;
    wire [31:0] stat_wcb_coalesces;

    integer pass_count;
    integer fail_count;
    integer timeout;
    integer i;
    integer m;
    integer miss_after_overflow;

    reg pending_read;
    reg pending_writeback;
    reg [31:0] pending_addr;
    reg [3:0]  pending_id;
    reg [3:0]  matched_id;

    reg [31:0] a0;
    reg [31:0] a1;
    reg [31:0] a2;
    reg [31:0] a3;
    reg [31:0] a4;

    l1_data_cache_optimized #(
        .CACHE_SIZE_KB   (1),
        .LINE_SIZE_BYTES (LINE_SIZE_BYTES),
        .NUM_WAYS        (4),
        .HIT_LATENCY     (2),
        .THREADS         (THREADS),
        .DATA_WIDTH      (32),
        .MAX_OUTSTANDING (MAX_OUTSTANDING),
        .PREFETCH_DEPTH  (2),
        .WCB_ENTRIES     (4)
    ) dut (
        .clk               (clk),
        .rst_n             (rst_n),
        .req_valid         (req_valid),
        .req_write         (req_write),
        .req_addr          (req_addr),
        .req_wdata         (req_wdata),
        .req_mask          (req_mask),
        .resp_rdata        (resp_rdata),
        .resp_valid        (resp_valid),
        .resp_hit          (resp_hit),
        .mem_req           (mem_req),
        .mem_write         (mem_write),
        .mem_addr          (mem_addr),
        .mem_wdata         (mem_wdata),
        .mem_rdata         (mem_rdata),
        .mem_valid         (mem_valid),
        .mem_ready         (mem_ready),
        .mem_id            (mem_id),
        .prefetch_enable   (prefetch_enable),
        .prefetch_addr     (prefetch_addr),
        .prefetch_active   (prefetch_active),
        .stat_hits         (stat_hits),
        .stat_misses       (stat_misses),
        .stat_prefetch_hits(stat_prefetch_hits),
        .stat_wcb_coalesces(stat_wcb_coalesces)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    function [31:0] expected_word;
        input [31:0] addr;
        reg [31:0] line_base;
        begin
            line_base = {addr[31:7], 7'b0};
            expected_word = line_base + {25'b0, addr[6:2], 2'b00};
        end
    endfunction

    task check_true;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s", msg);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", msg);
            end
        end
    endtask

    task check_eq32;
        input [31:0] got;
        input [31:0] exp;
        input [255:0] msg;
        begin
            if (got === exp) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%08x", msg, got);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got=0x%08x exp=0x%08x", msg, got, exp);
            end
        end
    endtask

    task reset_dut;
        begin
            rst_n <= 1'b0;
            req_valid <= 1'b0;
            req_write <= 1'b0;
            req_addr <= {(THREADS*32){1'b0}};
            req_wdata <= {(THREADS*32){1'b0}};
            req_mask <= {THREADS{1'b0}};
            mem_ready <= 1'b1;
            prefetch_enable <= 1'b0;
            mem_rdata <= {(LINE_SIZE_BYTES*8){1'b0}};
            mem_valid <= 1'b0;
            mem_id <= 4'd0;
            pending_read <= 1'b0;
            pending_writeback <= 1'b0;
            pending_addr <= 32'd0;
            pending_id <= 4'd0;
            #(CLK_PERIOD*5);
            rst_n <= 1'b1;
            #(CLK_PERIOD*2);
        end
    endtask

    task issue_read;
        input [THREADS-1:0] mask;
        input [31:0] addr0;
        input [31:0] addr1;
        input [31:0] addr2;
        input [31:0] addr3;
        begin
            @(posedge clk);
            req_valid <= 1'b1;
            req_write <= 1'b0;
            req_mask  <= mask;
            req_addr  <= {(THREADS*32){1'b0}};
            req_wdata <= {(THREADS*32){1'b0}};
            req_addr[0*32 +: 32] <= addr0;
            req_addr[1*32 +: 32] <= addr1;
            req_addr[2*32 +: 32] <= addr2;
            req_addr[3*32 +: 32] <= addr3;
            @(posedge clk);
            req_valid <= 1'b0;
            req_mask  <= {THREADS{1'b0}};
        end
    endtask

    task issue_write;
        input [THREADS-1:0] mask;
        input [31:0] addr0;
        input [31:0] addr1;
        input [31:0] addr2;
        input [31:0] addr3;
        input [31:0] w0;
        input [31:0] w1;
        input [31:0] w2;
        input [31:0] w3;
        begin
            @(posedge clk);
            req_valid <= 1'b1;
            req_write <= 1'b1;
            req_mask  <= mask;
            req_addr  <= {(THREADS*32){1'b0}};
            req_wdata <= {(THREADS*32){1'b0}};
            req_addr[0*32 +: 32] <= addr0;
            req_addr[1*32 +: 32] <= addr1;
            req_addr[2*32 +: 32] <= addr2;
            req_addr[3*32 +: 32] <= addr3;
            req_wdata[0*32 +: 32] <= w0;
            req_wdata[1*32 +: 32] <= w1;
            req_wdata[2*32 +: 32] <= w2;
            req_wdata[3*32 +: 32] <= w3;
            @(posedge clk);
            req_valid <= 1'b0;
            req_write <= 1'b0;
            req_mask  <= {THREADS{1'b0}};
        end
    endtask

    task wait_resp;
        input exp_hit;
        input [255:0] label;
        begin
            timeout = 0;
            while (!resp_valid && timeout < 300) begin
                timeout = timeout + 1;
                @(posedge clk);
            end
            check_true(resp_valid, {label, " resp_valid"});
            if (resp_valid) begin
                check_true(resp_hit == exp_hit, {label, " resp_hit"});
            end
        end
    endtask


    task wait_resp_any;
        input [255:0] label;
        begin
            timeout = 0;
            while (!resp_valid && timeout < 300) begin
                timeout = timeout + 1;
                @(posedge clk);
            end
            check_true(resp_valid, {label, " resp_valid"});
        end
    endtask
    task expect_no_resp_cycles;
        input integer cycles;
        input [255:0] label;
        integer c;
        begin
            for (c = 0; c < cycles; c = c + 1) begin
                @(posedge clk);
                if (resp_valid) begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] %0s: unexpected resp_valid", label);
                    c = cycles;
                end
            end
            if (!resp_valid) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s", label);
            end
        end
    endtask

    // Simple memory responder: one-cycle delayed handshake completion.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_valid <= 1'b0;
            mem_rdata <= {(LINE_SIZE_BYTES*8){1'b0}};
            mem_id <= 4'd0;
            pending_read <= 1'b0;
            pending_writeback <= 1'b0;
            pending_addr <= 32'd0;
            pending_id <= 4'd0;
        end else begin
            mem_valid <= 1'b0;

            if (pending_writeback) begin
                mem_valid <= 1'b1;
                mem_id <= pending_id;
                pending_writeback <= 1'b0;
            end else if (pending_read) begin
                mem_valid <= 1'b1;
                mem_id <= pending_id;
                for (i = 0; i < WORDS_PER_LINE; i = i + 1) begin
                    mem_rdata[i*32 +: 32] <= ({pending_addr[31:7], 7'b0} + (i << 2));
                end
                pending_read <= 1'b0;
            end

            if (mem_req && mem_ready) begin
                if (mem_write) begin
                    pending_writeback <= 1'b1;
                    pending_id <= 4'd0;
                end else begin
                    matched_id = 4'd0;
                    for (m = 0; m < MAX_OUTSTANDING; m = m + 1) begin
                        if (dut.mshr_valid[m] && (dut.mshr_addr[m] == mem_addr)) begin
                            matched_id = dut.mshr_id[m];
                        end
                    end
                    pending_read <= 1'b1;
                    pending_addr <= mem_addr;
                    pending_id <= matched_id;
                end
            end
        end
    end

    initial begin
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        // ------------------------------------------------------------------
        // Test 1: Basic read miss/hit path
        // ------------------------------------------------------------------
        a0 = 32'h0000_1000;
        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b0, "T1 first access miss");
        check_eq32(resp_rdata[31:0], expected_word(a0), "T1 lane0 miss data");

        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b1, "T1 second access hit");
        check_eq32(resp_rdata[31:0], expected_word(a0), "T1 lane0 hit data");
        check_eq32(stat_hits, 32'd1, "T1 stat_hits");
        check_eq32(stat_misses, 32'd1, "T1 stat_misses");

        // ------------------------------------------------------------------
        // Test 2: Write miss + readback hit
        // ------------------------------------------------------------------
        a1 = 32'h0000_1204;
        issue_write(4'b0001, a1, 32'd0, 32'd0, 32'd0,
                    32'hDEAD_BEEF, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b0, "T2 write miss response");

        issue_read(4'b0001, a1, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b1, "T2 readback hit");
        check_eq32(resp_rdata[31:0], 32'hDEAD_BEEF, "T2 readback data");

        // ------------------------------------------------------------------
        // Test 3: Coalesced/bank-pattern multi-lane access
        // ------------------------------------------------------------------
        a0 = 32'h0000_2000;
        a1 = 32'h0000_2010;
        a2 = 32'h0000_2020;
        a3 = 32'h0000_2030;

        issue_read(4'b1111, a0, a1, a2, a3);
        wait_resp(1'b0, "T3 multi-lane miss");
        check_eq32(resp_rdata[0*32 +: 32], expected_word(a0), "T3 lane0");
        check_eq32(resp_rdata[1*32 +: 32], expected_word(a1), "T3 lane1");
        check_eq32(resp_rdata[2*32 +: 32], expected_word(a2), "T3 lane2");
        check_eq32(resp_rdata[3*32 +: 32], expected_word(a3), "T3 lane3");

        issue_read(4'b1111, a0, a1, a2, a3);
        wait_resp(1'b1, "T3 multi-lane hit");

        // ------------------------------------------------------------------
        // Test 4: Eviction/replacement under capacity overflow
        // Same set with CACHE_SIZE=1KB, LINE=128B, WAYS=4 => set stride 0x100.
        // ------------------------------------------------------------------
        a0 = 32'h0000_0000;
        a1 = 32'h0000_0100;
        a2 = 32'h0000_0200;
        a3 = 32'h0000_0300;
        a4 = 32'h0000_0400;

        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0); wait_resp(1'b0, "T4 fill A");
        issue_read(4'b0001, a1, 32'd0, 32'd0, 32'd0); wait_resp(1'b0, "T4 fill B");
        issue_read(4'b0001, a2, 32'd0, 32'd0, 32'd0); wait_resp(1'b0, "T4 fill C");
        issue_read(4'b0001, a3, 32'd0, 32'd0, 32'd0); wait_resp(1'b0, "T4 fill D");

        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 touch A");
        issue_read(4'b0001, a4, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 insert E");

        miss_after_overflow = 0;
        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 recheck A"); if (!resp_hit) miss_after_overflow = miss_after_overflow + 1;
        issue_read(4'b0001, a1, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 recheck B"); if (!resp_hit) miss_after_overflow = miss_after_overflow + 1;
        issue_read(4'b0001, a2, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 recheck C"); if (!resp_hit) miss_after_overflow = miss_after_overflow + 1;
        issue_read(4'b0001, a3, 32'd0, 32'd0, 32'd0); wait_resp_any("T4 recheck D"); if (!resp_hit) miss_after_overflow = miss_after_overflow + 1;
        check_true(miss_after_overflow >= 1, "T4 overflow causes at least one eviction miss");

        // ------------------------------------------------------------------
        // Test 5: Back-to-back + memory backpressure edge case
        // ------------------------------------------------------------------
        mem_ready <= 1'b0;
        a0 = 32'h0000_5000;
        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0);
        expect_no_resp_cycles(20, "T5 no response when mem_ready=0");
        mem_ready <= 1'b1;
        wait_resp(1'b0, "T5 response after ready release");
        check_eq32(resp_rdata[31:0], expected_word(a0), "T5 lane0 data after stall");

        issue_write(4'b0001, 32'h0000_6000, 32'd0, 32'd0, 32'd0,
                    32'hABCD_1234, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b0, "T5 back-to-back write resp");
        issue_read(4'b0001, 32'h0000_6000, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b1, "T5 back-to-back read hit");
        check_eq32(resp_rdata[31:0], 32'hABCD_1234, "T5 back-to-back readback");

        $display("============================================================");
        $display("tb_l1_data_cache_optimized Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
        end else begin
            $display("SOME TESTS FAILED");
        end

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    initial begin
        #500000;
        $display("TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
