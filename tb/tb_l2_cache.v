//============================================================================
// RalphGPU - L2 Cache Bank Testbench
// Tests l2_cache_bank module with small parameters for simulation speed
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module tb_l2_cache;

    //------------------------------------------------------------------------
    // Parameters - small for fast simulation
    //------------------------------------------------------------------------
    localparam SIZE_BYTES   = 4096;     // 4KB
    localparam NUM_WAYS     = 2;
    localparam LINE_SIZE    = 16;       // 16 bytes = 128 bits per line
    localparam ADDR_WIDTH   = 32;
    localparam MSHR_ENTRIES = 2;
    localparam LINE_BITS    = LINE_SIZE * 8;  // 128

    localparam NUM_SETS     = SIZE_BYTES / (NUM_WAYS * LINE_SIZE);  // 128
    localparam OFFSET_BITS  = $clog2(LINE_SIZE);    // 4
    localparam INDEX_BITS   = $clog2(NUM_SETS);     // 7
    localparam TAG_BITS     = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS; // 21

    localparam CLK_PERIOD   = 10;

    //------------------------------------------------------------------------
    // Testbench signals
    //------------------------------------------------------------------------
    reg                         clk;
    reg                         rst_n;

    reg                         req_valid;
    reg                         req_write;
    reg  [ADDR_WIDTH-1:0]       req_addr;
    reg  [LINE_BITS-1:0]        req_wdata;
    reg  [LINE_SIZE-1:0]        req_wmask;
    wire                        req_ready;

    wire                        resp_valid;
    wire [LINE_BITS-1:0]        resp_rdata;

    wire                        hit;
    wire                        miss;
    wire                        writeback;

    wire                        mem_req_valid;
    wire                        mem_req_write;
    wire [ADDR_WIDTH-1:0]       mem_req_addr;
    wire [LINE_BITS-1:0]        mem_req_wdata;
    reg                         mem_req_ready;
    reg                         mem_fill_valid;
    reg  [LINE_BITS-1:0]        mem_fill_data;

    //------------------------------------------------------------------------
    // Test tracking
    //------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;
    integer test_num;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    l2_cache_bank #(
        .SIZE_BYTES     (SIZE_BYTES),
        .NUM_WAYS       (NUM_WAYS),
        .LINE_SIZE      (LINE_SIZE),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .MSHR_ENTRIES   (MSHR_ENTRIES)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .req_valid      (req_valid),
        .req_write      (req_write),
        .req_addr       (req_addr),
        .req_wdata      (req_wdata),
        .req_wmask      (req_wmask),
        .req_ready      (req_ready),
        .resp_valid     (resp_valid),
        .resp_rdata     (resp_rdata),
        .hit            (hit),
        .miss           (miss),
        .writeback      (writeback),
        .mem_req_valid  (mem_req_valid),
        .mem_req_write  (mem_req_write),
        .mem_req_addr   (mem_req_addr),
        .mem_req_wdata  (mem_req_wdata),
        .mem_req_ready  (mem_req_ready),
        .mem_fill_valid (mem_fill_valid),
        .mem_fill_data  (mem_fill_data)
    );

    //------------------------------------------------------------------------
    // Clock generation
    //------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //------------------------------------------------------------------------
    // Helper tasks
    //------------------------------------------------------------------------

    task check;
        input [255:0] test_name;
        input condition;
    begin
        if (condition) begin
            $display("[PASS] Test %0d: %0s", test_num, test_name);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s", test_num, test_name);
            fail_count = fail_count + 1;
        end
        test_num = test_num + 1;
    end
    endtask

    task reset_dut;
    begin
        rst_n = 0;
        req_valid = 0;
        req_write = 0;
        req_addr = 0;
        req_wdata = 0;
        req_wmask = 0;
        mem_req_ready = 1;
        mem_fill_valid = 0;
        mem_fill_data = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
    end
    endtask

    // Send a cache request and deassert req_valid once accepted
    task send_request;
        input                   is_write;
        input [ADDR_WIDTH-1:0]  addr;
        input [LINE_BITS-1:0]   wdata;
        input [LINE_SIZE-1:0]   wmask;
    begin
        @(posedge clk);
        req_valid = 1;
        req_write = is_write;
        req_addr  = addr;
        req_wdata = wdata;
        req_wmask = wmask;
        // Wait for request to be accepted (req_ready=1 means IDLE)
        @(posedge clk);
        req_valid = 0;
        req_write = 0;
        req_addr  = 0;
        req_wdata = 0;
        req_wmask = 0;
    end
    endtask

    // Wait for either resp_valid, miss, or hit with a timeout
    task wait_for_event;
        input integer max_cycles;
        integer cyc;
    begin
        for (cyc = 0; cyc < max_cycles; cyc = cyc + 1) begin
            @(posedge clk);
            if (resp_valid || miss || hit) begin
                cyc = max_cycles; // break
            end
        end
    end
    endtask

    // Wait for resp_valid with timeout
    task wait_resp;
        input integer max_cycles;
        integer cyc;
    begin
        for (cyc = 0; cyc < max_cycles; cyc = cyc + 1) begin
            @(posedge clk);
            if (resp_valid) begin
                cyc = max_cycles;
            end
        end
    end
    endtask

    //------------------------------------------------------------------------
    // Address helpers
    // Address layout: [tag(21)][index(7)][offset(4)]
    //------------------------------------------------------------------------
    function [ADDR_WIDTH-1:0] make_addr;
        input [TAG_BITS-1:0]    tag;
        input [INDEX_BITS-1:0]  idx;
        input [OFFSET_BITS-1:0] off;
    begin
        make_addr = {tag, idx, off};
    end
    endfunction

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("  L2 Cache Bank Testbench");
        $display("  SIZE_BYTES=%0d  NUM_WAYS=%0d  LINE_SIZE=%0d  MSHR=%0d",
                 SIZE_BYTES, NUM_WAYS, LINE_SIZE, MSHR_ENTRIES);
        $display("  NUM_SETS=%0d  OFFSET_BITS=%0d  INDEX_BITS=%0d  TAG_BITS=%0d",
                 NUM_SETS, OFFSET_BITS, INDEX_BITS, TAG_BITS);
        $display("============================================================");

        pass_count = 0;
        fail_count = 0;
        test_num   = 1;

        //====================================================================
        // Test 1: Reset
        //====================================================================
        $display("\n--- Test 1: Reset ---");
        reset_dut;

        check("req_ready=1 after reset",    req_ready === 1);
        check("resp_valid=0 after reset",    resp_valid === 0);
        check("hit=0 after reset",           hit === 0);
        check("miss=0 after reset",          miss === 0);
        check("writeback=0 after reset",     writeback === 0);
        check("mem_req_valid=0 after reset", mem_req_valid === 0);

        //====================================================================
        // Test 2: Read miss
        // Read address that has never been cached. Expect miss=1 and a
        // memory fill request (mem_req_valid=1, mem_req_write=0).
        //====================================================================
        $display("\n--- Test 2: Read miss ---");
        begin : test2_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  fill_a;
            addr_a = make_addr(21'h00_0001, 7'h00, 4'h0);  // tag=1, set=0
            fill_a = 128'hDEADBEEF_CAFEBABE_12345678_AABBCCDD;

            // Send read request
            send_request(0, addr_a, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});

            // FSM: IDLE(accept) -> TAG_CHECK(miss) -> FILL_REQ
            // In FILL_REQ the bank checks mem_fill_valid in the same cycle.
            // We need to provide fill data right when it reaches FILL_REQ.

            // Wait for miss pulse (TAG_CHECK cycle)
            wait_for_event(10);
            check("Read miss: miss asserted", miss === 1);

            // Drive fill data so FILL_REQ state sees it
            mem_fill_valid = 1;
            mem_fill_data  = fill_a;
            @(posedge clk);
            // FILL_REQ state: mem_req_valid should be high
            check("Read miss: mem_req_valid asserted", mem_req_valid === 1);
            check("Read miss: mem_req_write=0 (read)", mem_req_write === 0);
            // resp_valid should be asserted in FILL_REQ with the fill data
            check("Read miss: resp_valid after fill", resp_valid === 1);
            check("Read miss: resp_rdata matches fill",
                  resp_rdata === fill_a);
            mem_fill_valid = 0;
            mem_fill_data  = 0;
            @(posedge clk);
        end

        //====================================================================
        // Test 3: Read hit after fill
        // Same address should now be cached - expect hit=1.
        //====================================================================
        $display("\n--- Test 3: Read hit after fill ---");
        begin : test3_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  expected;
            addr_a   = make_addr(21'h00_0001, 7'h00, 4'h0);
            expected = 128'hDEADBEEF_CAFEBABE_12345678_AABBCCDD;

            send_request(0, addr_a, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Read hit: hit asserted", hit === 1);

            // Wait for S_HIT -> resp_valid
            @(posedge clk);
            check("Read hit: resp_valid", resp_valid === 1);
            check("Read hit: data matches cached line",
                  resp_rdata === expected);
            @(posedge clk);
        end

        //====================================================================
        // Test 4: Write hit
        // Write to the address we just cached. Should hit and mark dirty.
        //====================================================================
        $display("\n--- Test 4: Write hit ---");
        begin : test4_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  wdata;
            addr_a = make_addr(21'h00_0001, 7'h00, 4'h0);
            wdata  = 128'h11111111_22222222_33333333_44444444;

            send_request(1, addr_a, wdata, {LINE_SIZE{1'b1}});  // full mask
            wait_for_event(10);
            check("Write hit: hit asserted", hit === 1);

            @(posedge clk);
            check("Write hit: resp_valid", resp_valid === 1);
            check("Write hit: resp_rdata = written data",
                  resp_rdata === wdata);
            @(posedge clk);

            // Read back to verify
            send_request(0, addr_a, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Write hit readback: hit", hit === 1);
            @(posedge clk);
            check("Write hit readback: data matches",
                  resp_rdata === wdata);
            @(posedge clk);
        end

        //====================================================================
        // Test 5: Write miss (write-allocate)
        // Write to a new address that is not cached.
        // Bank: IDLE -> TAG_CHECK(miss) -> WRITE_ALLOC (no wb, victim invalid)
        //====================================================================
        $display("\n--- Test 5: Write miss (write-allocate) ---");
        begin : test5_block
            reg [ADDR_WIDTH-1:0] addr_b;
            reg [LINE_BITS-1:0]  wdata_b;
            addr_b   = make_addr(21'h00_0002, 7'h01, 4'h0);  // tag=2, set=1
            wdata_b  = 128'hAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDD;

            // Full mask write - write-allocate stores full line
            send_request(1, addr_b, wdata_b, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            check("Write miss: miss asserted", miss === 1);

            // S_WRITE_ALLOC happens next cycle, produces resp_valid
            @(posedge clk);
            check("Write miss: resp_valid after alloc", resp_valid === 1);
            check("Write miss: resp_rdata = written data",
                  resp_rdata === wdata_b);
            @(posedge clk);

            // Read back
            send_request(0, addr_b, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Write miss readback: hit", hit === 1);
            @(posedge clk);
            check("Write miss readback: data matches",
                  resp_rdata === wdata_b);
            @(posedge clk);
        end

        //====================================================================
        // Test 6: Eviction + writeback
        // With NUM_WAYS=2, filling a third line at the same set should
        // evict one existing dirty line and trigger writeback.
        //
        // Setup: use set 0.
        //   Way 0 already has tag=1 (dirty from Test 4).
        //   Fill way 1 with tag=2 (dirty write).
        //   Then access tag=3 - must evict a dirty way -> writeback.
        //====================================================================
        $display("\n--- Test 6: Eviction + writeback ---");
        begin : test6_block
            reg [ADDR_WIDTH-1:0] addr_t2, addr_t3;
            reg [LINE_BITS-1:0]  wdata_t2, wdata_t3;

            // Fill way 1 of set 0 with tag=2 via dirty write
            addr_t2  = make_addr(21'h00_0002, 7'h00, 4'h0);  // tag=2, set=0
            wdata_t2 = 128'hFEDCBA98_76543210_0F0F0F0F_F0F0F0F0;
            send_request(1, addr_t2, wdata_t2, {LINE_SIZE{1'b1}});
            // This is a miss on set 0 (tag=2 not present); way 1 is invalid
            // -> WRITE_ALLOC, no writeback
            wait_for_event(10);
            @(posedge clk); // let WRITE_ALLOC complete
            @(posedge clk);

            // Now set 0 has: way 0 = tag=1 (dirty), way 1 = tag=2 (dirty)
            // Access tag=3 on set 0 - must evict one dirty way -> writeback
            addr_t3  = make_addr(21'h00_0003, 7'h00, 4'h0);  // tag=3, set=0
            wdata_t3 = 128'h99887766_55443322_11009988_77665544;
            send_request(1, addr_t3, wdata_t3, {LINE_SIZE{1'b1}});

            wait_for_event(10);
            check("Eviction: miss asserted", miss === 1);
            check("Eviction: writeback asserted", writeback === 1);

            // S_WRITEBACK issues mem_req_valid+mem_req_write
            @(posedge clk);
            check("Eviction: mem_req_valid for writeback", mem_req_valid === 1);
            check("Eviction: mem_req_write=1", mem_req_write === 1);

            // After writeback, FSM goes to WRITE_ALLOC (write miss)
            @(posedge clk);
            check("Eviction: resp_valid after write-alloc", resp_valid === 1);
            check("Eviction: resp_rdata = new data",
                  resp_rdata === wdata_t3);
            @(posedge clk);

            // Read back tag=3 on set 0
            send_request(0, addr_t3, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Eviction readback: hit on tag=3", hit === 1);
            @(posedge clk);
            check("Eviction readback: correct data",
                  resp_rdata === wdata_t3);
            @(posedge clk);
        end

        //====================================================================
        // Test 7: Byte-masked write
        // Partial write using wmask - only masked bytes should change.
        //====================================================================
        $display("\n--- Test 7: Byte-masked write ---");
        begin : test7_block
            reg [ADDR_WIDTH-1:0] addr_m;
            reg [LINE_BITS-1:0]  orig_data, new_wdata, expected_data;
            reg [LINE_SIZE-1:0]  mask;

            // Use set 2, write full line first to establish known data
            addr_m    = make_addr(21'h00_0004, 7'h02, 4'h0);  // tag=4, set=2
            orig_data = 128'h00112233_44556677_8899AABB_CCDDEEFF;
            send_request(1, addr_m, orig_data, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            @(posedge clk);
            @(posedge clk);

            // Now do a partial write: only modify bytes 0-3 (lower 32 bits)
            new_wdata = 128'hFFFFFFFF_FFFFFFFF_FFFFFFFF_DEADBEEF;
            mask      = 16'h000F;  // only byte 0,1,2,3

            send_request(1, addr_m, new_wdata, mask);
            wait_for_event(10);
            check("Masked write: hit on existing line", hit === 1);
            @(posedge clk);
            check("Masked write: resp_valid", resp_valid === 1);

            // Expected: bytes 0-3 = DEADBEEF from new, rest = original
            expected_data = 128'h00112233_44556677_8899AABB_DEADBEEF;
            check("Masked write: partial data correct",
                  resp_rdata === expected_data);
            @(posedge clk);

            // Read back to verify
            send_request(0, addr_m, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            @(posedge clk);
            check("Masked write readback: data matches",
                  resp_rdata === expected_data);
            @(posedge clk);
        end

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("  L2 Cache Bank Testbench Results");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** SOME TESTS FAILED ***");
        $display("============================================================");
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("[TIMEOUT] Testbench exceeded 100us - aborting");
        $finish;
    end

    //------------------------------------------------------------------------
    // Waveform dump (optional)
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_l2_cache.vcd");
        $dumpvars(0, tb_l2_cache);
    end

endmodule
