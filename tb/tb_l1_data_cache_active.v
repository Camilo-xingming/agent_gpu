`timescale 1ns / 1ps

module tb_l1_data_cache_active;

    localparam CLK_PERIOD      = 10;
    localparam THREADS         = 4;
    localparam LINE_SIZE_BYTES = 128;
    localparam WORDS_PER_LINE  = LINE_SIZE_BYTES / 4;
    localparam MAX_OUTSTANDING = 4;
    localparam MEM_WORDS       = 16384;

    reg clk;
    reg rst_n;

    reg                    req_valid;
    reg                    req_write;
    reg [THREADS*32-1:0]   req_addr;
    reg [THREADS*32-1:0]   req_wdata;
    reg [THREADS-1:0]      req_mask;
    wire [THREADS*32-1:0]  resp_rdata;
    wire                   resp_valid;
    wire                   resp_hit;

    wire                          mem_req;
    wire                          mem_write;
    wire [31:0]                   mem_addr;
    wire [LINE_SIZE_BYTES*8-1:0]  mem_wdata;
    reg  [LINE_SIZE_BYTES*8-1:0]  mem_rdata;
    reg                           mem_valid;
    reg                           mem_ready;
    reg  [3:0]                    mem_id;

    reg                           prefetch_enable;
    wire [31:0]                   prefetch_addr;
    wire                          prefetch_active;

    wire [31:0] stat_hits;
    wire [31:0] stat_misses;
    wire [31:0] stat_prefetch_hits;
    wire [31:0] stat_wcb_coalesces;

    integer pass_count;
    integer fail_count;
    integer timeout;
    integer i;
    integer m;
    integer wb_before;
    integer resp_cycles;
    integer read_before;
    integer dirty_wb_commits;

    reg pending_read;
    reg pending_writeback;
    reg [31:0] pending_addr;
    reg [3:0]  pending_id;
    reg [3:0]  matched_id;
    reg [LINE_SIZE_BYTES*8-1:0] pending_wdata;

    integer mem_write_count;
    integer mem_read_count;

    reg [31:0] mem_store [0:MEM_WORDS-1];

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

    function [31:0] mem_word_at;
        input [31:0] addr;
        begin
            mem_word_at = mem_store[(addr >> 2) & (MEM_WORDS-1)];
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
            rst_n            <= 1'b0;
            req_valid        <= 1'b0;
            req_write        <= 1'b0;
            req_addr         <= {(THREADS*32){1'b0}};
            req_wdata        <= {(THREADS*32){1'b0}};
            req_mask         <= {THREADS{1'b0}};
            mem_ready        <= 1'b1;
            prefetch_enable  <= 1'b0;
            mem_rdata        <= {(LINE_SIZE_BYTES*8){1'b0}};
            mem_valid        <= 1'b0;
            mem_id           <= 4'd0;
            pending_read     <= 1'b0;
            pending_writeback<= 1'b0;
            pending_addr     <= 32'd0;
            pending_id       <= 4'd0;
            pending_wdata    <= {(LINE_SIZE_BYTES*8){1'b0}};
            mem_write_count  = 0;
            mem_read_count   = 0;
            #(CLK_PERIOD*4);
            rst_n            <= 1'b1;
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
            if (resp_valid)
                check_true(resp_hit == exp_hit, {label, " resp_hit"});
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

    task wait_resp_cycles;
        input [255:0] label;
        output integer cycles;
        begin
            cycles = 0;
            while (!resp_valid && cycles < 300) begin
                cycles = cycles + 1;
                @(posedge clk);
            end
            check_true(resp_valid, {label, " resp_valid"});
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_valid         <= 1'b0;
            mem_rdata         <= {(LINE_SIZE_BYTES*8){1'b0}};
            mem_id            <= 4'd0;
            pending_read      <= 1'b0;
            pending_writeback <= 1'b0;
            pending_addr      <= 32'd0;
            pending_id        <= 4'd0;
            pending_wdata     <= {(LINE_SIZE_BYTES*8){1'b0}};
        end else begin
            mem_valid <= 1'b0;

            if (pending_writeback) begin
                mem_valid <= 1'b1;
                mem_id    <= pending_id;
                for (i = 0; i < WORDS_PER_LINE; i = i + 1)
                    mem_store[((pending_addr >> 2) + i) & (MEM_WORDS-1)] <= pending_wdata[i*32 +: 32];
                pending_writeback <= 1'b0;
            end else if (pending_read) begin
                mem_valid <= 1'b1;
                mem_id    <= pending_id;
                for (i = 0; i < WORDS_PER_LINE; i = i + 1)
                    mem_rdata[i*32 +: 32] <= mem_store[((pending_addr >> 2) + i) & (MEM_WORDS-1)];
                pending_read <= 1'b0;
            end

            if (mem_req && mem_ready) begin
                if (mem_write) begin
                    mem_write_count  = mem_write_count + 1;
                    pending_writeback<= 1'b1;
                    pending_addr     <= mem_addr;
                    pending_wdata    <= mem_wdata;
                    pending_id       <= 4'd0;
                end else begin
                    mem_read_count = mem_read_count + 1;
                    matched_id = 4'd0;
                    for (m = 0; m < MAX_OUTSTANDING; m = m + 1)
                        if (dut.mshr_valid[m] && (dut.mshr_addr[m] == mem_addr))
                            matched_id = dut.mshr_id[m];
                    pending_read <= 1'b1;
                    pending_addr <= mem_addr;
                    pending_id   <= matched_id;
                end
            end
        end
    end

    initial begin
        pass_count = 0;
        fail_count = 0;

        for (i = 0; i < MEM_WORDS; i = i + 1)
            mem_store[i] = 32'h1000_0000 + i;

        reset_dut();

        // ------------------------------------------------------------------
        // T1: Miss then hit on same address
        // ------------------------------------------------------------------
        a0 = 32'h0000_1000;
        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b0, "T1 first read miss");
        check_eq32(resp_rdata[31:0], mem_word_at(a0), "T1 lane0 miss data");

        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0);
        wait_resp(1'b1, "T1 second read hit");
        check_eq32(resp_rdata[31:0], mem_word_at(a0), "T1 lane0 hit data");

        // ------------------------------------------------------------------
        // T2: Dirty eviction triggers writeback
        // same set addresses: 0x000,0x100,0x200,0x300,0x400
        // Mark all resident ways dirty so victim choice cannot hide writeback
        // ------------------------------------------------------------------
        a0 = 32'h0000_0000;
        a1 = 32'h0000_0100;
        a2 = 32'h0000_0200;
        a3 = 32'h0000_0300;
        a4 = 32'h0000_0400;

        issue_write(4'b0001, a0, 32'd0, 32'd0, 32'd0,
                    32'hCAFE_BABE, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T2 write A");

        issue_write(4'b0001, a1, 32'd0, 32'd0, 32'd0,
                    32'hBEEF_1001, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T2 write B");

        issue_write(4'b0001, a2, 32'd0, 32'd0, 32'd0,
                    32'hBEEF_1002, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T2 write C");

        issue_write(4'b0001, a3, 32'd0, 32'd0, 32'd0,
                    32'hBEEF_1003, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T2 write D");

        wb_before = mem_write_count;
        issue_read(4'b0001, a4, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T2 insert E cause eviction");
        check_true(mem_write_count > wb_before, "T2 dirty eviction issued writeback");

        dirty_wb_commits = 0;
        if (mem_word_at(a0) == 32'hCAFE_BABE) dirty_wb_commits = dirty_wb_commits + 1;
        if (mem_word_at(a1) == 32'hBEEF_1001) dirty_wb_commits = dirty_wb_commits + 1;
        if (mem_word_at(a2) == 32'hBEEF_1002) dirty_wb_commits = dirty_wb_commits + 1;
        if (mem_word_at(a3) == 32'hBEEF_1003) dirty_wb_commits = dirty_wb_commits + 1;
        check_true(dirty_wb_commits >= 1, "T2 writeback committed one dirty victim line");

        // ------------------------------------------------------------------
        // T3: Clean eviction has no writeback and no extra stall
        // same set addresses with bit7=1: 0x080,0x180,0x280,0x380,0x480
        // ------------------------------------------------------------------
        a0 = 32'h0000_0080;
        a1 = 32'h0000_0180;
        a2 = 32'h0000_0280;
        a3 = 32'h0000_0380;
        a4 = 32'h0000_0480;

        issue_read(4'b0001, a0, 32'd0, 32'd0, 32'd0); wait_resp_any("T3 fill A");
        issue_read(4'b0001, a1, 32'd0, 32'd0, 32'd0); wait_resp_any("T3 fill B");
        issue_read(4'b0001, a2, 32'd0, 32'd0, 32'd0); wait_resp_any("T3 fill C");
        issue_read(4'b0001, a3, 32'd0, 32'd0, 32'd0); wait_resp_any("T3 fill D");

        wb_before = mem_write_count;
        read_before = mem_read_count;
        issue_read(4'b0001, a4, 32'd0, 32'd0, 32'd0);
        wait_resp_cycles("T3 clean eviction miss response", resp_cycles);
        check_true(mem_write_count == wb_before, "T3 clean eviction no dirty writeback");
        check_true(mem_read_count > read_before, "T3 clean eviction still fetches new line");
        check_true(resp_cycles < 80, "T3 clean eviction response not stalled");

        // ------------------------------------------------------------------
        // T4: Multi-warp same-line ordering/coherence
        // ------------------------------------------------------------------
        a0 = 32'h0000_9000;
        a1 = 32'h0000_9004;

        issue_read(4'b0011, a0, a1, 32'd0, 32'd0);
        wait_resp_any("T4 initial shared-line fill");

        issue_write(4'b0001, a0, 32'd0, 32'd0, 32'd0,
                    32'h55AA_11EE, 32'd0, 32'd0, 32'd0);
        wait_resp_any("T4 lane0 write same line");

        issue_read(4'b0011, a0, a1, 32'd0, 32'd0);
        wait_resp_any("T4 read-after-write same line");
        check_eq32(resp_rdata[0*32 +: 32], 32'h55AA_11EE, "T4 lane0 sees latest write");
        check_eq32(resp_rdata[1*32 +: 32], mem_word_at(a1), "T4 lane1 ordering unaffected");

        $display("============================================================");
        $display("tb_l1_data_cache_active Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    initial begin
        #700000;
        $display("TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
