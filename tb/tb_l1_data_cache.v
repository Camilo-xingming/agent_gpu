`timescale 1ns / 1ps

module tb_l1_data_cache;

    localparam CLK_PERIOD = 10;
    localparam THREADS = 4;
    localparam LINE_SIZE_BYTES = 128;
    localparam WORDS_PER_LINE = LINE_SIZE_BYTES / 4;

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

    wire                   mem_req;
    wire                   mem_write;
    wire [31:0]            mem_addr;
    wire [LINE_SIZE_BYTES*8-1:0] mem_wdata;
    reg  [LINE_SIZE_BYTES*8-1:0] mem_rdata;
    reg                    mem_valid;
    reg                    mem_ready;

    wire [31:0]            stat_hits;
    wire [31:0]            stat_misses;

    reg                    policy_create_valid;
    reg [2:0]              policy_id;
    reg [7:0]              policy_priority;
    wire [31:0]            policy_token_out;
    wire                   policy_token_valid;
    reg                    policy_apply_valid;
    reg [31:0]             policy_apply_addr;
    reg [2:0]              policy_apply_id;
    reg                    policy_discard_valid;
    reg [31:0]             policy_discard_addr;

    integer pass_count;
    integer fail_count;
    integer mem_word;
    integer timeout;

    reg pending_fill;
    reg pending_writeback;
    reg [31:0] pending_addr;

    reg [31:0] a0;
    reg [31:0] a1;
    reg [31:0] a2;
    reg [31:0] a3;
    reg [31:0] a4;

    function [31:0] expected_word;
        input [31:0] addr;
        begin
            expected_word = {addr[31:7], 7'b0} + {25'b0, addr[6:2], 2'b00};
        end
    endfunction

    l1_data_cache #(
        .CACHE_SIZE_KB   (1),
        .LINE_SIZE_BYTES (LINE_SIZE_BYTES),
        .NUM_WAYS        (4),
        .HIT_LATENCY     (2),
        .THREADS         (THREADS),
        .DATA_WIDTH      (32)
    ) dut (
        .clk                 (clk),
        .rst_n               (rst_n),
        .req_valid           (req_valid),
        .req_write           (req_write),
        .req_addr            (req_addr),
        .req_wdata           (req_wdata),
        .req_mask            (req_mask),
        .resp_rdata          (resp_rdata),
        .resp_valid          (resp_valid),
        .resp_hit            (resp_hit),
        .mem_req             (mem_req),
        .mem_write           (mem_write),
        .mem_addr            (mem_addr),
        .mem_wdata           (mem_wdata),
        .mem_rdata           (mem_rdata),
        .mem_valid           (mem_valid),
        .mem_ready           (mem_ready),
        .stat_hits           (stat_hits),
        .stat_misses         (stat_misses),
        .policy_create_valid (policy_create_valid),
        .policy_id           (policy_id),
        .policy_priority     (policy_priority),
        .policy_token_out    (policy_token_out),
        .policy_token_valid  (policy_token_valid),
        .policy_apply_valid  (policy_apply_valid),
        .policy_apply_addr   (policy_apply_addr),
        .policy_apply_id     (policy_apply_id),
        .policy_discard_valid(policy_discard_valid),
        .policy_discard_addr (policy_discard_addr)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_valid <= 1'b0;
            mem_rdata <= {(LINE_SIZE_BYTES*8){1'b0}};
            pending_fill <= 1'b0;
            pending_writeback <= 1'b0;
            pending_addr <= 32'b0;
        end else begin
            mem_valid <= 1'b0;

            if (pending_fill) begin
                mem_valid <= 1'b1;
                for (mem_word = 0; mem_word < WORDS_PER_LINE; mem_word = mem_word + 1) begin
                    mem_rdata[mem_word*32 +: 32] <=
                        {pending_addr[31:7], 7'b0} + (mem_word << 2);
                end
                pending_fill <= 1'b0;
            end else if (pending_writeback) begin
                mem_valid <= 1'b1;
                pending_writeback <= 1'b0;
            end

            if (mem_req && mem_ready) begin
                if (mem_write) begin
                    pending_writeback <= 1'b1;
                end else begin
                    pending_fill <= 1'b1;
                    pending_addr <= mem_addr;
                end
            end
        end
    end

    task reset_dut;
        begin
            rst_n <= 1'b0;
            req_valid <= 1'b0;
            req_write <= 1'b0;
            req_addr <= {(THREADS*32){1'b0}};
            req_wdata <= {(THREADS*32){1'b0}};
            req_mask <= {THREADS{1'b0}};
            mem_ready <= 1'b1;
            policy_create_valid <= 1'b0;
            policy_id <= 3'b0;
            policy_priority <= 8'b0;
            policy_apply_valid <= 1'b0;
            policy_apply_addr <= 32'b0;
            policy_apply_id <= 3'b0;
            policy_discard_valid <= 1'b0;
            policy_discard_addr <= 32'b0;
            #(CLK_PERIOD * 5);
            rst_n <= 1'b1;
            #(CLK_PERIOD * 2);
        end
    endtask

    task check_equal;
        input cond;
        input [255:0] msg;
        begin
            if (!cond) begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", msg);
            end else begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s", msg);
            end
        end
    endtask

    task issue_warp_read;
        input [THREADS-1:0] mask;
        input [31:0] addr0;
        input [31:0] addr1;
        input [31:0] addr2;
        input [31:0] addr3;
        begin
            @(posedge clk);
            req_valid <= 1'b1;
            req_write <= 1'b0;
            req_mask <= mask;
            req_addr <= {(THREADS*32){1'b0}};
            req_wdata <= {(THREADS*32){1'b0}};
            req_addr[0*32 +: 32] <= addr0;
            req_addr[1*32 +: 32] <= addr1;
            req_addr[2*32 +: 32] <= addr2;
            req_addr[3*32 +: 32] <= addr3;
            @(posedge clk);
            req_valid <= 1'b0;
            req_mask <= {THREADS{1'b0}};
        end
    endtask

    task wait_and_check_resp;
        input expect_hit;
        input [THREADS-1:0] mask;
        input [31:0] addr0;
        input [31:0] addr1;
        input [31:0] addr2;
        input [31:0] addr3;
        input [255:0] label;
        begin
            timeout = 0;
            while (!resp_valid && timeout < 200) begin
                timeout = timeout + 1;
                @(posedge clk);
            end
            check_equal(resp_valid, {label, " - response observed"});
            if (resp_valid) begin
                check_equal(resp_hit == expect_hit, {label, " - hit flag"});
                if (mask[0]) check_equal(resp_rdata[0*32 +: 32] == expected_word(addr0), {label, " - lane0 data"});
                if (mask[1]) check_equal(resp_rdata[1*32 +: 32] == expected_word(addr1), {label, " - lane1 data"});
                if (mask[2]) check_equal(resp_rdata[2*32 +: 32] == expected_word(addr2), {label, " - lane2 data"});
                if (mask[3]) check_equal(resp_rdata[3*32 +: 32] == expected_word(addr3), {label, " - lane3 data"});
            end
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        // Test 1: basic miss then hit
        a0 = 32'h0000_1000;
        issue_warp_read(4'b0001, a0, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a0, 32'b0, 32'b0, 32'b0, "hit/miss: first access miss");

        issue_warp_read(4'b0001, a0, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b1, 4'b0001, a0, 32'b0, 32'b0, 32'b0, "hit/miss: second access hit");

        // Test 2: different line miss
        a1 = 32'h0000_2080;
        issue_warp_read(4'b0001, a1, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a1, 32'b0, 32'b0, 32'b0, "new line miss");

        // Test 3: LRU replacement behavior in same set
        a0 = 32'h0000_0000;
        a1 = 32'h0000_0100;
        a2 = 32'h0000_0200;
        a3 = 32'h0000_0300;
        a4 = 32'h0000_0400;

        issue_warp_read(4'b0001, a0, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a0, 32'b0, 32'b0, 32'b0, "LRU fill way0");
        issue_warp_read(4'b0001, a1, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a1, 32'b0, 32'b0, 32'b0, "LRU fill way1");
        issue_warp_read(4'b0001, a2, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a2, 32'b0, 32'b0, 32'b0, "LRU fill way2");
        issue_warp_read(4'b0001, a3, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a3, 32'b0, 32'b0, 32'b0, "LRU fill way3");

        issue_warp_read(4'b0001, a0, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b1, 4'b0001, a0, 32'b0, 32'b0, 32'b0, "LRU touch way0");

        issue_warp_read(4'b0001, a4, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a4, 32'b0, 32'b0, 32'b0, "LRU insert way4");

        issue_warp_read(4'b0001, a1, 32'b0, 32'b0, 32'b0);
        wait_and_check_resp(1'b0, 4'b0001, a1, 32'b0, 32'b0, 32'b0, "LRU evicted line miss");

        // Test 4: multi-lane same-line access (bank-conflict-like pattern)
        a0 = 32'h0000_3000;
        a1 = 32'h0000_3010;
        a2 = 32'h0000_3020;
        a3 = 32'h0000_3030;

        issue_warp_read(4'b1111, a0, a1, a2, a3);
        wait_and_check_resp(1'b0, 4'b1111, a0, a1, a2, a3, "bank conflict pattern miss");

        issue_warp_read(4'b1111, a0, a1, a2, a3);
        wait_and_check_resp(1'b1, 4'b1111, a0, a1, a2, a3, "bank conflict pattern hit");

        check_equal(stat_hits > 0, "stat_hits increments");
        check_equal(stat_misses > 0, "stat_misses increments");

        $display("========================================");
        $display("tb_l1_data_cache done: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("========================================");

        if (fail_count != 0) begin
            $fatal(1, "tb_l1_data_cache failed");
        end
        $finish;
    end

endmodule
