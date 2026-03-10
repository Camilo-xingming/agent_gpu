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
        // Wait for request to be accepted
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
    
    // Automated Memory Responder
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_fill_valid <= 0;
            mem_fill_data <= 0;
        end else begin
            mem_fill_valid <= 0;
            if (mem_req_valid && mem_req_ready && !mem_req_write) begin
                mem_fill_valid <= 1;
                if (mem_req_addr[ADDR_WIDTH-1:INDEX_BITS+OFFSET_BITS] == 21'h00_0001) begin
                    mem_fill_data <= 128'hDEADBEEF_CAFEBABE_12345678_AABBCCDD;
                end else begin
                    mem_fill_data <= 128'hFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF;
                end
            end
        end
    end

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
        //====================================================================
        $display("\n--- Test 2: Read miss ---");
        begin : test2_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  fill_a;
            addr_a = make_addr(21'h00_0001, 7'h00, 4'h0);
            fill_a = 128'hDEADBEEF_CAFEBABE_12345678_AABBCCDD;

            send_request(0, addr_a, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Read miss: miss asserted", miss === 1);

            @(posedge clk);
            check("Read miss: mem_req_valid asserted", mem_req_valid === 1);
            check("Read miss: mem_req_write=0 (read)", mem_req_write === 0);

            wait_resp(10);
            check("Read miss: resp_valid after fill", resp_valid === 1);
            check("Read miss: resp_rdata matches fill", resp_rdata === fill_a);
        end

        $display("\n--- Test 3: Read hit after fill ---");
        begin : test3_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  expected;
            addr_a   = make_addr(21'h00_0001, 7'h00, 4'h0);
            expected = 128'hDEADBEEF_CAFEBABE_12345678_AABBCCDD;

            send_request(0, addr_a, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            check("Read hit: hit asserted", hit === 1);

            @(posedge clk);
            check("Read hit: resp_valid", resp_valid === 1);
            check("Read hit: data matches cached line",
                  resp_rdata === expected);
            @(posedge clk);
        end

        //====================================================================
        // Test 4: Write hit
        //====================================================================
        $display("\n--- Test 4: Write hit ---");
        begin : test4_block
            reg [ADDR_WIDTH-1:0] addr_a;
            reg [LINE_BITS-1:0]  wdata;
            addr_a = make_addr(21'h00_0001, 7'h00, 4'h0);
            wdata  = 128'h11111111_22222222_33333333_44444444;

            send_request(1, addr_a, wdata, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            check("Write hit: hit asserted", hit === 1);

            @(posedge clk);
            check("Write hit: resp_valid", resp_valid === 1);
            check("Write hit: resp_rdata = written data",
                  resp_rdata === wdata);
            @(posedge clk);

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
        //====================================================================
        $display("\n--- Test 5: Write miss (write-allocate) ---");
        begin : test5_block
            reg [ADDR_WIDTH-1:0] addr_b;
            reg [LINE_BITS-1:0]  wdata_b;
            addr_b   = make_addr(21'h00_0002, 7'h01, 4'h0);
            wdata_b  = 128'hAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDD;

            send_request(1, addr_b, wdata_b, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            check("Write miss: miss asserted", miss === 1);

            wait_resp(20);
            check("Write miss: resp_valid after alloc", resp_valid === 1);
            check("Write miss: resp_rdata = written data",
                  resp_rdata === wdata_b);
            @(posedge clk);

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
        //====================================================================
        $display("\n--- Test 6: Eviction + writeback ---");
        begin : test6_block
            reg [ADDR_WIDTH-1:0] addr_t2, addr_t3;
            reg [LINE_BITS-1:0]  wdata_t2, wdata_t3;

            addr_t2  = make_addr(21'h00_0002, 7'h00, 4'h0);
            wdata_t2 = 128'hFEDCBA98_76543210_0F0F0F0F_F0F0F0F0;
            send_request(1, addr_t2, wdata_t2, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            @(posedge clk);
            @(posedge clk);

            addr_t3  = make_addr(21'h00_0003, 7'h00, 4'h0);
            wdata_t3 = 128'h99887766_55443322_11009988_77665544;
            send_request(1, addr_t3, wdata_t3, {LINE_SIZE{1'b1}});

            wait_for_event(10);
            check("Eviction: miss asserted", miss === 1);
            check("Eviction: writeback asserted", writeback === 1);

            while (!mem_req_valid) @(posedge clk);
            check("Eviction: mem_req_valid for writeback", mem_req_valid === 1);
            check("Eviction: mem_req_write=1", mem_req_write === 1);

            wait_resp(30);
            check("Eviction: resp_valid after write-alloc", resp_valid === 1);
            check("Eviction: resp_rdata = new data",
                  resp_rdata === wdata_t3);
            @(posedge clk);

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
        //====================================================================
        $display("\n--- Test 7: Byte-masked write ---");
        begin : test7_block
            reg [ADDR_WIDTH-1:0] addr_m;
            reg [LINE_BITS-1:0]  orig_data, new_wdata, expected_data;
            reg [LINE_SIZE-1:0]  mask;

            addr_m    = make_addr(21'h00_0004, 7'h02, 4'h0);
            orig_data = 128'h00112233_44556677_8899AABB_CCDDEEFF;
            send_request(1, addr_m, orig_data, {LINE_SIZE{1'b1}});
            wait_for_event(10);
            @(posedge clk);
            @(posedge clk);

            new_wdata = 128'hFFFFFFFF_FFFFFFFF_FFFFFFFF_DEADBEEF;
            mask      = 16'h000F;

            send_request(1, addr_m, new_wdata, mask);
            wait_for_event(10);
            check("Masked write: hit on existing line", hit === 1);
            @(posedge clk);
            check("Masked write: resp_valid", resp_valid === 1);

            expected_data = 128'h00112233_44556677_8899AABB_DEADBEEF;
            check("Masked write: partial data correct",
                  resp_rdata === expected_data);
            @(posedge clk);

            send_request(0, addr_m, {LINE_BITS{1'b0}}, {LINE_SIZE{1'b0}});
            wait_for_event(10);
            @(posedge clk);
            check("Masked write readback: data matches",
                  resp_rdata === expected_data);
            @(posedge clk);
        end

        //====================================================================
        // Test 8: Back-to-back masked writes (#588 regression test)
        //====================================================================
        $display("\n--- Test 8: Back-to-back masked writes ---");
        begin : test8_block
            reg [ADDR_WIDTH-1:0] addr8;
            reg [LINE_BITS-1:0]  data8_1, data8_2, expected8;
            
            addr8 = make_addr(21'h00_0005, 7'h03, 4'h0);
            data8_1 = 128'hAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA;
            data8_2 = 128'h5555_5555_5555_5555_5555_5555_5555_5555;
            
            // 1. Initial write (fill line)
            send_request(1, addr8, data8_1, 16'hFFFF);
            wait_resp(20);
            @(posedge clk);
            
            // 2. Partial overwrite (lower 8 bytes)
            send_request(1, addr8, data8_2, 16'h00FF);
            wait_resp(10);
            @(posedge clk);
            
            // 3. Verify
            send_request(0, addr8, 128'b0, 16'h0000);
            wait_resp(10);
            expected8 = {64'hAAAA_AAAA_AAAA_AAAA, 64'h5555_5555_5555_5555};
            check("Multi-write masked data matches", resp_rdata === expected8);
            @(posedge clk);
        end

        //====================================================================
        // Summary
        //====================================================================
        $display("\n==========================================================");
        $display("  L2 Cache Bank Testbench Results");
        $display("  PASSED: %0d", pass_count);
        $display("  FAILED: %0d", fail_count);
        $display("  TOTAL:  %0d", pass_count + fail_count);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** SOME TESTS FAILED ***");
        $display("==========================================================");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("[TIMEOUT] Testbench exceeded 100us - aborting");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // Waveform dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_l2_cache.vcd");
        $dumpvars(0, tb_l2_cache);
    end

endmodule
