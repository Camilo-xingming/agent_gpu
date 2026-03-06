`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_icache_lru_conflict;

    localparam SIZE_KB        = 1;
    localparam LINE_SIZE      = 16;
    localparam NUM_WAYS       = 2;
    localparam PREFETCH_DEPTH = 2;
    localparam ADDR_WIDTH     = 32;
    localparam DATA_WIDTH     = 32;
    localparam LINE_BITS      = LINE_SIZE * 8;
    localparam OFFSET_BITS    = $clog2(LINE_SIZE);
    localparam CLK_PERIOD     = 10;
    localparam MEM_LATENCY    = 4;

    localparam [ADDR_WIDTH-1:0] ADDR_A = 32'h0000_0000;
    localparam [ADDR_WIDTH-1:0] ADDR_B = 32'h0000_0200;
    localparam [ADDR_WIDTH-1:0] ADDR_C = 32'h0000_0400;

    reg clk;
    reg rst_n;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg                     fetch_req;
    reg  [ADDR_WIDTH-1:0]   fetch_addr;
    wire                    fetch_ready;
    wire [DATA_WIDTH-1:0]   fetch_data;
    wire [LINE_BITS-1:0]    fetch_line_data;
    wire                    fetch_valid;
    wire                    fetch_hit_bypass;
    wire [DATA_WIDTH-1:0]   fetch_hit_bypass_data;
    wire [LINE_BITS-1:0]    fetch_hit_bypass_line_data;

    reg                     fetch_req_b;
    reg  [ADDR_WIDTH-1:0]   fetch_addr_b;
    wire                    fetch_ready_b;
    wire [DATA_WIDTH-1:0]   fetch_data_b;
    wire [LINE_BITS-1:0]    fetch_line_data_b;
    wire                    fetch_valid_b;

    reg                     invalidate_req;
    reg  [ADDR_WIDTH-1:0]   invalidate_addr;
    reg                     invalidate_all;
    wire                    invalidate_done;

    wire                    mem_req_valid;
    wire [ADDR_WIDTH-1:0]   mem_req_addr;
    reg                     mem_req_ready;
    reg  [LINE_BITS-1:0]    mem_resp_data;
    reg                     mem_resp_valid;

    wire [31:0]             stat_hits;
    wire [31:0]             stat_misses;
    wire [31:0]             stat_prefetch_hits;

    icache #(
        .SIZE_KB        (SIZE_KB),
        .LINE_SIZE      (LINE_SIZE),
        .NUM_WAYS       (NUM_WAYS),
        .PREFETCH_DEPTH (PREFETCH_DEPTH),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .DATA_WIDTH     (DATA_WIDTH)
    ) dut (
        .clk                        (clk),
        .rst_n                      (rst_n),
        .fetch_req                  (fetch_req),
        .fetch_addr                 (fetch_addr),
        .fetch_ready                (fetch_ready),
        .fetch_data                 (fetch_data),
        .fetch_line_data            (fetch_line_data),
        .fetch_valid                (fetch_valid),
        .fetch_hit_bypass           (fetch_hit_bypass),
        .fetch_hit_bypass_data      (fetch_hit_bypass_data),
        .fetch_hit_bypass_line_data (fetch_hit_bypass_line_data),
        .fetch_req_b                (fetch_req_b),
        .fetch_addr_b               (fetch_addr_b),
        .fetch_ready_b              (fetch_ready_b),
        .fetch_data_b               (fetch_data_b),
        .fetch_line_data_b          (fetch_line_data_b),
        .fetch_valid_b              (fetch_valid_b),
        .invalidate_req             (invalidate_req),
        .invalidate_addr            (invalidate_addr),
        .invalidate_all             (invalidate_all),
        .invalidate_done            (invalidate_done),
        .mem_req_valid              (mem_req_valid),
        .mem_req_addr               (mem_req_addr),
        .mem_req_ready              (mem_req_ready),
        .mem_resp_data              (mem_resp_data),
        .mem_resp_valid             (mem_resp_valid),
        .stat_hits                  (stat_hits),
        .stat_misses                (stat_misses),
        .stat_prefetch_hits         (stat_prefetch_hits)
    );

    reg  [ADDR_WIDTH-1:0] pending_mem_addr;
    reg                   pending_mem_active;
    integer               pending_mem_countdown;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_resp_valid        <= 1'b0;
            mem_resp_data         <= {LINE_BITS{1'b0}};
            pending_mem_active    <= 1'b0;
            pending_mem_countdown <= 0;
        end else begin
            mem_resp_valid <= 1'b0;

            if (mem_req_valid && mem_req_ready && !pending_mem_active) begin
                pending_mem_addr      <= mem_req_addr;
                pending_mem_active    <= 1'b1;
                pending_mem_countdown <= MEM_LATENCY;
            end

            if (pending_mem_active) begin
                if (pending_mem_countdown == 0) begin
                    mem_resp_valid <= 1'b1;
                    mem_resp_data <= {
                        pending_mem_addr[15:0], 16'd3,
                        pending_mem_addr[15:0], 16'd2,
                        pending_mem_addr[15:0], 16'd1,
                        pending_mem_addr[15:0], 16'd0
                    };
                    pending_mem_active <= 1'b0;
                end else begin
                    pending_mem_countdown <= pending_mem_countdown - 1;
                end
            end
        end
    end

    integer pass_count;
    integer fail_count;
    integer i;

    task check;
        input [255:0] name;
        input         cond;
        begin
            if (cond) begin
                $display("[PASS] %0s", name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s", name);
                fail_count = fail_count + 1;
            end
        end
    endtask

    function [DATA_WIDTH-1:0] expected_word;
        input [ADDR_WIDTH-1:0] addr;
        reg [ADDR_WIDTH-1:0] line_addr;
        reg [1:0] word_idx;
        begin
            line_addr     = {addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            word_idx      = addr[3:2];
            expected_word = {line_addr[15:0], {14'b0, word_idx}};
        end
    endfunction

    task fetch_a;
        input  [ADDR_WIDTH-1:0] addr;
        output [DATA_WIDTH-1:0] data;
        output                  combo_hit;
        integer timeout;
        begin
            timeout = 0;
            while (!fetch_ready && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (!fetch_ready) begin
                $display("[FAIL] timeout waiting fetch_ready for addr=0x%08x", addr);
                fail_count = fail_count + 1;
            end

            @(posedge clk);
            fetch_req  <= 1'b1;
            fetch_addr <= addr;
            @(posedge clk);
            combo_hit = fetch_valid;
            data = fetch_data;
            fetch_req <= 1'b0;

            if (!combo_hit) begin
                timeout = 0;
                while (!fetch_valid && timeout < 100) begin
                    @(posedge clk);
                    timeout = timeout + 1;
                end
                if (!fetch_valid) begin
                    $display("[FAIL] timeout waiting fetch_valid for addr=0x%08x", addr);
                    fail_count = fail_count + 1;
                end
                data = fetch_data;
            end
            @(posedge clk);
        end
    endtask

    reg [DATA_WIDTH-1:0] rdata;
    reg combo_hit;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;

        fetch_req = 1'b0;
        fetch_addr = '0;
        fetch_req_b = 1'b0;
        fetch_addr_b = '0;

        invalidate_req = 1'b0;
        invalidate_addr = '0;
        invalidate_all = 1'b0;
        mem_req_ready = 1'b1;

        pass_count = 0;
        fail_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        fetch_a(ADDR_A, rdata, combo_hit);
        check("A initial access is miss", combo_hit === 1'b0);
        check("A data correct", rdata === expected_word(ADDR_A));
        check("stat_misses=1", stat_misses === 32'd1);

        fetch_a(ADDR_B, rdata, combo_hit);
        check("B initial access is miss", combo_hit === 1'b0);
        check("B data correct", rdata === expected_word(ADDR_B));
        check("stat_misses=2", stat_misses === 32'd2);
        i = 0;
        while ((!fetch_ready || !fetch_ready_b) && i < 100) begin
            @(posedge clk);
            i = i + 1;
        end

        @(posedge clk);
        fetch_req   <= 1'b1;
        fetch_addr  <= ADDR_B;
        fetch_req_b <= 1'b1;
        fetch_addr_b <= ADDR_A;
        @(posedge clk);
        check("bank-conflict port A hit", fetch_valid === 1'b1);
        check("bank-conflict port B hit", fetch_valid_b === 1'b1);
        check("bank-conflict port A data", fetch_data === expected_word(ADDR_B));
        check("bank-conflict port B data", fetch_data_b === expected_word(ADDR_A));
        fetch_req   <= 1'b0;
        fetch_req_b <= 1'b0;
        repeat (2) @(posedge clk);

        fetch_a(ADDR_C, rdata, combo_hit);
        check("C insert is miss", combo_hit === 1'b0);
        check("stat_misses=3", stat_misses === 32'd3);

        fetch_a(ADDR_A, rdata, combo_hit);
        check("A retained after C insert", combo_hit === 1'b1);
        check("A retained data correct", rdata === expected_word(ADDR_A));
        check("stat_misses still 3 after A hit", stat_misses === 32'd3);

        fetch_a(ADDR_B, rdata, combo_hit);
        check("B evicted after C insert", combo_hit === 1'b0);
        check("B refill data correct", rdata === expected_word(ADDR_B));
        check("stat_misses=4 after B miss", stat_misses === 32'd4);

        $display("[RESULT] pass=%0d fail=%0d", pass_count, fail_count);
        if (fail_count == 0)
            $display("*** ALL TESTS PASSED ***");
        else
            $display("*** TESTS FAILED ***");

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    initial begin
        #200000;
        $display("[TIMEOUT] tb_icache_lru_conflict");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
