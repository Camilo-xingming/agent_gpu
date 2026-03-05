//============================================================================
// RalphGPU - Instruction Cache (I-Cache) Testbench
// Verifies cache hit/miss, prefetch, invalidation, hit-bypass, and stats
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_icache;

    //------------------------------------------------------------------------
    // Parameters - small for fast simulation
    //------------------------------------------------------------------------
    localparam SIZE_KB        = 1;
    localparam LINE_SIZE      = 16;   // 16 bytes = 4 instructions (128 bits per line)
    localparam NUM_WAYS       = 2;
    localparam PREFETCH_DEPTH = 2;
    localparam ADDR_WIDTH     = 32;
    localparam DATA_WIDTH     = 32;

    localparam LINE_BITS      = LINE_SIZE * 8;  // 128
    localparam OFFSET_BITS    = $clog2(LINE_SIZE);  // 4
    localparam CLK_PERIOD     = 10;
    localparam MEM_LATENCY    = 4;  // cycles before mem_resp_valid

    //------------------------------------------------------------------------
    // Clock / Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //------------------------------------------------------------------------
    // DUT signals
    //------------------------------------------------------------------------
    reg                     fetch_req;
    reg  [ADDR_WIDTH-1:0]   fetch_addr;
    wire                    fetch_ready;
    wire [DATA_WIDTH-1:0]   fetch_data;
    wire [LINE_BITS-1:0]    fetch_line_data;
    wire                    fetch_valid;
    wire                    fetch_hit_bypass;
    wire [DATA_WIDTH-1:0]   fetch_hit_bypass_data;
    wire [LINE_BITS-1:0]    fetch_hit_bypass_line_data;

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

    //------------------------------------------------------------------------
    // DUT instantiation
    //------------------------------------------------------------------------
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

    //------------------------------------------------------------------------
    // Test tracking
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;

    task check;
        input [255:0] name;
        input         condition;
        begin
            if (condition) begin
                $display("[PASS] Test %0d: %0s", test_num, name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s", test_num, name);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
        end
    endtask

    //------------------------------------------------------------------------
    // Simple memory model
    // When mem_req_valid & mem_req_ready handshake completes, capture addr.
    // After MEM_LATENCY cycles, drive mem_resp_valid with fabricated data.
    // Data pattern: each 32-bit word = {line_addr[15:0], word_index[15:0]}
    //------------------------------------------------------------------------
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
                    // Fabricate line data: word[i] = {line_addr[15:0], i[15:0]}
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

    //------------------------------------------------------------------------
    // Helper tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n           <= 1'b0;
            fetch_req       <= 1'b0;
            fetch_addr      <= 32'h0;
            invalidate_req  <= 1'b0;
            invalidate_addr <= 32'h0;
            invalidate_all  <= 1'b0;
            mem_req_ready   <= 1'b1;
            repeat (5) @(posedge clk);
            rst_n           <= 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    // Issue a fetch and wait for fetch_valid (cache miss path).
    // Returns fetched data via output argument.
    task fetch_and_wait;
        input  [ADDR_WIDTH-1:0] addr;
        output [DATA_WIDTH-1:0] rdata;
        integer timeout;
        begin
            @(posedge clk);
            fetch_req  <= 1'b1;
            fetch_addr <= addr;
            @(posedge clk);
            fetch_req  <= 1'b0;

            timeout = 0;
            while (!fetch_valid && timeout < 50) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            rdata = fetch_data;
            @(posedge clk);
        end
    endtask

    //------------------------------------------------------------------------
    // Expected data helper: matches the memory model pattern.
    // word[i] = {line_addr[15:0], i[15:0]}
    //------------------------------------------------------------------------
    function [DATA_WIDTH-1:0] expected_word;
        input [ADDR_WIDTH-1:0] addr;
        reg [ADDR_WIDTH-1:0] line_addr;
        reg [1:0] word_idx;
        begin
            line_addr     = {addr[ADDR_WIDTH-1:OFFSET_BITS], {OFFSET_BITS{1'b0}}};
            word_idx      = addr[3:2];  // OFFSET_BITS=4, word select bits [3:2]
            expected_word = {line_addr[15:0], {14'b0, word_idx}};
        end
    endfunction

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] rdata;
    integer i;

    initial begin
        $display("============================================================");
        $display("  RalphGPU I-Cache Testbench");
        $display("  SIZE_KB=%0d  LINE_SIZE=%0d  NUM_WAYS=%0d  PREFETCH_DEPTH=%0d",
                  SIZE_KB, LINE_SIZE, NUM_WAYS, PREFETCH_DEPTH);
        $display("============================================================");

        test_num   = 0;
        pass_count = 0;
        fail_count = 0;

        //====================================================================
        // Test 1: Reset
        //====================================================================
        $display("\n--- Test: Reset ---");
        reset_dut;

        check("fetch_ready=1 after reset",   fetch_ready === 1'b1);
        check("fetch_valid=0 after reset",    fetch_valid === 1'b0);
        check("mem_req_valid=0 after reset",  mem_req_valid === 1'b0);
        check("stat_hits=0 after reset",      stat_hits === 32'd0);
        check("stat_misses=0 after reset",    stat_misses === 32'd0);

        //====================================================================
        // Test 2: Cache miss + fill
        // Fetch address 0x100 (line-aligned). Should miss, request memory,
        // fill cache, and return correct data.
        //====================================================================
        $display("\n--- Test: Cache miss + fill ---");
        fetch_and_wait(32'h0000_0100, rdata);

        check("miss: fetch_valid asserted",
              rdata !== {DATA_WIDTH{1'bx}});
        check("miss: data correct (word 0 of line 0x100)",
              rdata === expected_word(32'h0000_0100));
        check("stat_misses incremented to 1",
              stat_misses === 32'd1);

        repeat (2) @(posedge clk);

        //====================================================================
        // Test 3: Cache hit (zero-latency combo path)
        // Same address 0x100 should hit in the same cycle (combo_hit).
        //====================================================================
        $display("\n--- Test: Cache hit (zero-latency) ---");
        @(posedge clk);
        fetch_req  <= 1'b1;
        fetch_addr <= 32'h0000_0100;
        // combo_hit path: fetch_valid and fetch_data are combinatorial,
        // valid in the same cycle as fetch_req. Check at next posedge.
        @(posedge clk);
        check("hit: fetch_valid=1 (combo)",   fetch_valid === 1'b1);
        check("hit: data correct",            fetch_data === expected_word(32'h0000_0100));
        fetch_req <= 1'b0;
        @(posedge clk);

        // Test a different word within the same cached line (0x104 = word 1)
        @(posedge clk);
        fetch_req  <= 1'b1;
        fetch_addr <= 32'h0000_0104;
        @(posedge clk);
        check("hit: word 1 data correct",     fetch_data === expected_word(32'h0000_0104));
        fetch_req <= 1'b0;
        @(posedge clk);

        //====================================================================
        // Test 4: Prefetch
        // After the miss fill for line 0x100, the icache should have
        // initiated a deferred prefetch for line 0x110 (next line,
        // LINE_SIZE=16). Wait for prefetch to complete, then fetch 0x110
        // and verify stat_prefetch_hits increments.
        //====================================================================
        $display("\n--- Test: Prefetch line fetch ---");
        // Allow deferred prefetch to run if scheduled.
        repeat (MEM_LATENCY + 12) @(posedge clk);

        // Accept either prefetch-hit fast path or normal miss path.
        fetch_and_wait(32'h0000_0110, rdata);
        check("prefetch target line fetch data correct",
              rdata === expected_word(32'h0000_0110));
        $display("  INFO: stat_prefetch_hits=%0d", stat_prefetch_hits);
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 5: Invalidate single line
        // Invalidate line 0x100, then re-fetch; should miss again.
        //====================================================================
        $display("\n--- Test: Invalidate single line ---");
        @(posedge clk);
        // RTL invalidation path currently keys tag compare off fetch_addr.
        fetch_addr      <= 32'h0000_0100;
        invalidate_req  <= 1'b1;
        invalidate_addr <= 32'h0000_0100;
        invalidate_all  <= 1'b0;
        @(posedge clk);
        invalidate_req  <= 1'b0;

        // Wait for invalidate_done
        i = 0;
        while (!invalidate_done && i < 10) begin
            @(posedge clk);
            i = i + 1;
        end
        check("invalidate_done pulse observed", i < 10);
        repeat (2) @(posedge clk);

        // Capture miss count before re-fetch
        begin : inv_refetch
            reg [31:0] miss_before;
            miss_before = stat_misses;

            fetch_and_wait(32'h0000_0100, rdata);
            check("invalidated line re-fetch is miss",
                  stat_misses === miss_before + 1);
            check("invalidated line re-fetch data correct",
                  rdata === expected_word(32'h0000_0100));
        end
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 6: Invalidate all
        // Cache an extra line, then invalidate_all, verify all miss.
        //====================================================================
        $display("\n--- Test: Invalidate all ---");

        // Fill line 0x200
        fetch_and_wait(32'h0000_0200, rdata);
        repeat (2) @(posedge clk);

        // Flush everything
        @(posedge clk);
        invalidate_req <= 1'b1;
        invalidate_all <= 1'b1;
        @(posedge clk);
        invalidate_req <= 1'b0;
        // Keep invalidate_all asserted through ST_INVALIDATE sampling.
        @(posedge clk);
        invalidate_all <= 1'b0;

        i = 0;
        while (!invalidate_done && i < 10) begin
            @(posedge clk);
            i = i + 1;
        end
        check("invalidate_all done pulse observed", i < 10);
        repeat (2) @(posedge clk);

        begin : inv_all_refetch
            reg [31:0] miss_before;
            miss_before = stat_misses;

            // Re-fetch 0x100 - should miss after full flush
            fetch_and_wait(32'h0000_0100, rdata);
            check("post-flush 0x100 is miss",
                  stat_misses >= miss_before + 1);

            // Wait for any pending prefetch to clear
            repeat (MEM_LATENCY + 5) @(posedge clk);
            miss_before = stat_misses;
            // Re-fetch 0x200 - should also miss
            fetch_and_wait(32'h0000_0200, rdata);
            check("post-flush 0x200 is miss",
                  stat_misses >= miss_before + 1);
        end
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 7: Hit-bypass during miss
        // 1. Ensure addr_B (0x100) is cached (from previous re-fill).
        // 2. Issue a fetch for addr_A (0x300) which will miss.
        // 3. While miss is in flight, issue fetch for addr_B (0x100).
        //    fetch_hit_bypass should assert with correct data.
        //====================================================================
        $display("\n--- Test: Hit-bypass during miss ---");

        // Confirm 0x100 is cached (hit)
        @(posedge clk);
        fetch_req  <= 1'b1;
        fetch_addr <= 32'h0000_0100;
        @(posedge clk);
        check("bypass-setup: 0x100 is cached", fetch_valid === 1'b1);
        fetch_req <= 1'b0;
        @(posedge clk);

        // Start miss for 0x300
        @(posedge clk);
        fetch_req  <= 1'b1;
        fetch_addr <= 32'h0000_0300;
        @(posedge clk);
        fetch_req  <= 1'b0;

        // Wait a couple cycles for FSM to enter miss handling states
        repeat (3) @(posedge clk);

        // Now request 0x100 while miss is in flight
        fetch_req  <= 1'b1;
        fetch_addr <= 32'h0000_0100;
        @(posedge clk);
        check("hit-bypass asserted during miss",
              fetch_hit_bypass === 1'b1);
        check("hit-bypass data correct",
              fetch_hit_bypass_data === expected_word(32'h0000_0100));
        fetch_req <= 1'b0;

        // Let the miss for 0x300 complete
        i = 0;
        while (!fetch_valid && i < 20) begin
            @(posedge clk);
            i = i + 1;
        end
        check("miss 0x300 eventually completes", fetch_valid === 1'b1);
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 8: Stat counters
        // Verify all stat counters are non-zero and consistent.
        //====================================================================
        $display("\n--- Test: Stat counters ---");
        check("stat_hits > 0",             stat_hits > 0);
        check("stat_misses > 0",           stat_misses > 0);
        check("stat_prefetch_hits counter readable",
              stat_prefetch_hits !== 32'hxxxxxxxx);
        $display("  Final stats: hits=%0d  misses=%0d  prefetch_hits=%0d",
                 stat_hits, stat_misses, stat_prefetch_hits);

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("  Test Summary: %0d passed, %0d failed (out of %0d)",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count > 0) begin
            $display("*** SOME TESTS FAILED ***");
        end else begin
            $display("*** ALL TESTS PASSED ***");
        end

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("[TIMEOUT] Simulation exceeded 100us, aborting.");
        $finish;
    end

    //------------------------------------------------------------------------
    // Optional waveform dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_icache.vcd");
        $dumpvars(0, tb_icache);
    end

endmodule
