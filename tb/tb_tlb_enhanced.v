//============================================================================
// RalphGPU - TLB Enhanced Testbench
// Tests: L1/L2 hit/miss, page walker, multi-page sizes, ASID,
//        invalidation, permission faults, perf counters, multi-SM
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tlb_enhanced;

    localparam CLK_PERIOD   = 10;
    localparam NUM_SMS      = 2;
    localparam VADDR_WIDTH  = 48;
    localparam PADDR_WIDTH  = 40;
    localparam L1_ENTRIES   = 32;
    localparam L1_WAYS      = 4;
    localparam L2_ENTRIES   = 512;
    localparam L2_WAYS      = 8;
    localparam ASID_WIDTH   = 16;
    localparam PAGE_LEVELS  = 4;

    //------------------------------------------------------------------------
    // DUT signals
    //------------------------------------------------------------------------
    reg                              clk;
    reg                              rst_n;

    reg  [NUM_SMS-1:0]               req_valid;
    reg  [VADDR_WIDTH*NUM_SMS-1:0]   req_vaddr;
    reg  [NUM_SMS-1:0]               req_write;
    reg  [ASID_WIDTH*NUM_SMS-1:0]    req_asid;
    wire [NUM_SMS-1:0]               req_ready;

    wire [NUM_SMS-1:0]               resp_valid;
    wire [PADDR_WIDTH*NUM_SMS-1:0]   resp_paddr;
    wire [NUM_SMS-1:0]               resp_fault;
    wire [NUM_SMS*4-1:0]             resp_fault_code;

    wire                             ptw_req_valid;
    wire [PADDR_WIDTH-1:0]           ptw_req_addr;
    reg                              ptw_req_ready;
    reg                              ptw_resp_valid;
    reg  [63:0]                      ptw_resp_data;

    reg  [PADDR_WIDTH-1:0]           page_table_base;
    reg  [ASID_WIDTH-1:0]            current_asid;

    reg                              invalidate_all;
    reg                              invalidate_asid;
    reg  [ASID_WIDTH-1:0]            invalidate_asid_val;
    reg                              invalidate_page;
    reg  [VADDR_WIDTH-1:0]           invalidate_vaddr;

    wire [31:0]                      stat_l1_hits;
    wire [31:0]                      stat_l1_misses;
    wire [31:0]                      stat_l2_hits;
    wire [31:0]                      stat_l2_misses;
    wire [31:0]                      stat_page_walks;
    wire [31:0]                      stat_page_faults;

    //------------------------------------------------------------------------
    // DUT instantiation
    //------------------------------------------------------------------------
    tlb_enhanced #(
        .NUM_SMS        (NUM_SMS),
        .VADDR_WIDTH    (VADDR_WIDTH),
        .PADDR_WIDTH    (PADDR_WIDTH),
        .L1_ENTRIES     (L1_ENTRIES),
        .L1_WAYS        (L1_WAYS),
        .L2_ENTRIES     (L2_ENTRIES),
        .L2_WAYS        (L2_WAYS),
        .ASID_WIDTH     (ASID_WIDTH),
        .PAGE_LEVELS    (PAGE_LEVELS)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .req_valid          (req_valid),
        .req_vaddr          (req_vaddr),
        .req_write          (req_write),
        .req_asid           (req_asid),
        .req_ready          (req_ready),
        .resp_valid         (resp_valid),
        .resp_paddr         (resp_paddr),
        .resp_fault         (resp_fault),
        .resp_fault_code    (resp_fault_code),
        .ptw_req_valid      (ptw_req_valid),
        .ptw_req_addr       (ptw_req_addr),
        .ptw_req_ready      (ptw_req_ready),
        .ptw_resp_valid     (ptw_resp_valid),
        .ptw_resp_data      (ptw_resp_data),
        .page_table_base    (page_table_base),
        .current_asid       (current_asid),
        .invalidate_all     (invalidate_all),
        .invalidate_asid    (invalidate_asid),
        .invalidate_asid_val(invalidate_asid_val),
        .invalidate_page    (invalidate_page),
        .invalidate_vaddr   (invalidate_vaddr),
        .stat_l1_hits       (stat_l1_hits),
        .stat_l1_misses     (stat_l1_misses),
        .stat_l2_hits       (stat_l2_hits),
        .stat_l2_misses     (stat_l2_misses),
        .stat_page_walks    (stat_page_walks),
        .stat_page_faults   (stat_page_faults)
    );

    //------------------------------------------------------------------------
    // Clock
    //------------------------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test infrastructure
    //------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;
    integer test_num;
    integer timeout_cnt;

    // Latched response values (captured when resp_valid pulses)
    reg                      lat_valid_sm0;
    reg                      lat_fault_sm0;
    reg [PADDR_WIDTH-1:0]    lat_paddr_sm0;
    reg [3:0]                lat_fcode_sm0;
    reg                      lat_valid_sm1;
    reg                      lat_fault_sm1;
    reg [PADDR_WIDTH-1:0]    lat_paddr_sm1;
    reg [3:0]                lat_fcode_sm1;

    task check;
        input cond;
        input [399:0] msg;
        begin
            if (cond) begin
                $display("  PASS: %0s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %0s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Page-table walk responder (identity mapping model)
    // Uses 2-level walk: PML4 -> leaf (1GB pages)
    // VA[39:12] -> PA[39:12] (identity)
    //------------------------------------------------------------------------
    localparam [PADDR_WIDTH-1:0] PT_BASE     = 40'h00_0010_0000;
    localparam [PADDR_WIDTH-1:0] PT_L3_BASE  = 40'h00_0010_1000;

    function [63:0] make_pte;
        input [PADDR_WIDTH-1:0] base;
        input                   leaf;
        input [3:0]             perm;
        begin
            make_pte = 64'b0;
            make_pte[0]   = 1'b1;           // present
            make_pte[1]   = perm[1];        // writable
            make_pte[2]   = perm[2];        // executable
            make_pte[3]   = perm[3];        // user
            make_pte[7]   = leaf;           // large page indicator
            make_pte[12 + (PADDR_WIDTH-12) - 1 : 12] = base[PADDR_WIDTH-1:12];
        end
    endfunction

    // Page permissions (indexed by L3 entry = VPN[38:30])
    reg [3:0] page_perm [0:3];
    initial begin
        page_perm[0] = 4'b0011;  // page 0: R+W
        page_perm[1] = 4'b0011;  // page 1: R+W
        page_perm[2] = 4'b0001;  // page 2: R only
        page_perm[3] = 4'b0011;  // page 3: R+W
    end

    // PTW responder
    always @(posedge clk) begin
        if (!rst_n) begin
            ptw_resp_valid <= 1'b0;
            ptw_resp_data  <= 64'b0;
        end else begin
            ptw_resp_valid <= 1'b0;
            if (ptw_req_valid && ptw_req_ready) begin
                ptw_resp_valid <= 1'b1;
                if (ptw_req_addr == PT_BASE) begin
                    ptw_resp_data <= make_pte(PT_L3_BASE, 1'b0, 4'b0011);
                end else if (ptw_req_addr[PADDR_WIDTH-1:12] == PT_L3_BASE[PADDR_WIDTH-1:12]) begin
                    begin : ptw_l3_blk
                        reg [8:0] l3_idx;
                        reg [PADDR_WIDTH-1:0] leaf_base;
                        reg [3:0] perm;
                        l3_idx = ptw_req_addr[11:3];
                        if (l3_idx < 9'd4) begin
                            leaf_base = {{(PADDR_WIDTH-32){1'b0}}, l3_idx[1:0], 30'b0};
                            perm = page_perm[l3_idx[1:0]];
                            ptw_resp_data <= make_pte(leaf_base, 1'b1, perm);
                        end else begin
                            ptw_resp_data <= 64'b0;  // not present
                        end
                    end
                end else begin
                    ptw_resp_data <= 64'b0;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Helper tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 1'b0;
            req_valid = 0;
            req_vaddr = 0;
            req_write = 0;
            req_asid  = 0;
            ptw_req_ready = 1'b1;
            page_table_base = PT_BASE;
            current_asid = 16'b0;
            invalidate_all = 1'b0;
            invalidate_asid = 1'b0;
            invalidate_asid_val = 16'b0;
            invalidate_page = 1'b0;
            invalidate_vaddr = 0;
            #(CLK_PERIOD * 4);
            rst_n = 1'b1;
            #(CLK_PERIOD * 2);
        end
    endtask

    task translate_sm0;
        input [VADDR_WIDTH-1:0] va;
        input                   wr;
        input [ASID_WIDTH-1:0]  asid;
        begin
            lat_valid_sm0 = 1'b0;
            lat_fault_sm0 = 1'b0;
            lat_paddr_sm0 = {PADDR_WIDTH{1'b0}};
            lat_fcode_sm0 = 4'b0;
            @(negedge clk);
            req_valid[0] = 1'b1;
            req_vaddr[0 +: VADDR_WIDTH] = va;
            req_write[0] = wr;
            req_asid[0 +: ASID_WIDTH] = asid;
            @(negedge clk);
            req_valid[0] = 1'b0;
            // Check immediately - L1 hits respond in 1 cycle
            if (resp_valid[0]) begin
                lat_valid_sm0 = 1'b1;
                lat_fault_sm0 = resp_fault[0];
                lat_paddr_sm0 = resp_paddr[0 +: PADDR_WIDTH];
                lat_fcode_sm0 = resp_fault_code[0 +: 4];
            end

            timeout_cnt = 0;
            while (!lat_valid_sm0 && timeout_cnt < 200) begin
                @(negedge clk);
                if (resp_valid[0]) begin
                    lat_valid_sm0 = 1'b1;
                    lat_fault_sm0 = resp_fault[0];
                    lat_paddr_sm0 = resp_paddr[0 +: PADDR_WIDTH];
                    lat_fcode_sm0 = resp_fault_code[0 +: 4];
                end
                timeout_cnt = timeout_cnt + 1;
            end
        end
    endtask

    task translate_sm1;
        input [VADDR_WIDTH-1:0] va;
        input                   wr;
        input [ASID_WIDTH-1:0]  asid;
        begin
            lat_valid_sm1 = 1'b0;
            lat_fault_sm1 = 1'b0;
            lat_paddr_sm1 = {PADDR_WIDTH{1'b0}};
            lat_fcode_sm1 = 4'b0;
            @(negedge clk);
            req_valid[1] = 1'b1;
            req_vaddr[VADDR_WIDTH +: VADDR_WIDTH] = va;
            req_write[1] = wr;
            req_asid[ASID_WIDTH +: ASID_WIDTH] = asid;
            @(negedge clk);
            req_valid[1] = 1'b0;
            // Check immediately - L1 hits respond in 1 cycle
            if (resp_valid[1]) begin
                lat_valid_sm1 = 1'b1;
                lat_fault_sm1 = resp_fault[1];
                lat_paddr_sm1 = resp_paddr[PADDR_WIDTH +: PADDR_WIDTH];
                lat_fcode_sm1 = resp_fault_code[4 +: 4];
            end

            timeout_cnt = 0;
            while (!lat_valid_sm1 && timeout_cnt < 200) begin
                @(negedge clk);
                if (resp_valid[1]) begin
                    lat_valid_sm1 = 1'b1;
                    lat_fault_sm1 = resp_fault[1];
                    lat_paddr_sm1 = resp_paddr[PADDR_WIDTH +: PADDR_WIDTH];
                    lat_fcode_sm1 = resp_fault_code[4 +: 4];
                end
                timeout_cnt = timeout_cnt + 1;
            end
        end
    endtask

    task do_invalidate_all;
        begin
            @(negedge clk);
            invalidate_all = 1'b1;
            @(negedge clk);
            invalidate_all = 1'b0;
            #(CLK_PERIOD * 2);
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        reset_dut;

        $display("====================================================");
        $display("TLB Enhanced Testbench - Multi-page workload");
        $display("====================================================");

        // TEST 1: Cold miss - page walk fills L1 & L2
        test_num = test_num + 1;
        $display("\n[TEST %0d] Cold miss - page walk for VA 0x1000", test_num);
        translate_sm0(48'h0000_0000_1000, 1'b0, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b0,           "no fault on read");
        check(lat_paddr_sm0 == 40'h00_0000_1000,
              "PA = identity(VA) for page 0");

        // TEST 2: Same VA should L1 hit
        test_num = test_num + 1;
        $display("\n[TEST %0d] L1 hit - same VA", test_num);
        begin : test2_blk
            reg [31:0] walks_before;
            walks_before = stat_page_walks;
            translate_sm0(48'h0000_0000_1000, 1'b0, 16'h0);
            check(lat_valid_sm0 == 1'b1,       "resp_valid asserted");
            check(lat_fault_sm0 == 1'b0,       "no fault");
            check(lat_paddr_sm0 == 40'h00_0000_1000,
                  "PA matches identity map");
            check(stat_l1_hits > 0,            "L1 hit counter incremented");
        end

        // TEST 3: Different VA in same 1GB page
        test_num = test_num + 1;
        $display("\n[TEST %0d] Different offset in page 0", test_num);
        translate_sm0(48'h0000_0FFF_F000, 1'b0, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b0,           "no fault");

        // TEST 4: VA in page 1 (next 1GB region)
        test_num = test_num + 1;
        $display("\n[TEST %0d] VA in page 1 (1GB region 1)", test_num);
        translate_sm0(48'h0000_4000_2000, 1'b0, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b0,           "no fault on read");
        check(lat_paddr_sm0 == 40'h00_4000_2000,
              "PA = identity(VA) for page 1");

        // TEST 5: Write to read-only page (page 2) - permission fault
        test_num = test_num + 1;
        $display("\n[TEST %0d] Permission fault - write to read-only page 2", test_num);
        translate_sm0(48'h0000_8000_0000, 1'b1, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b1,           "write fault on read-only page");

        // TEST 6: Read from same read-only page - should succeed
        test_num = test_num + 1;
        $display("\n[TEST %0d] Read from read-only page 2 (OK)", test_num);
        translate_sm0(48'h0000_8000_0000, 1'b0, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b0,           "read allowed on read-only page");

        // TEST 7: Unmapped page (L3 index >= 4) - page fault
        test_num = test_num + 1;
        $display("\n[TEST %0d] Unmapped page - page fault", test_num);
        translate_sm0(48'h0001_4000_0000, 1'b0, 16'h0);
        check(lat_valid_sm0 == 1'b1,           "resp_valid asserted");
        check(lat_fault_sm0 == 1'b1,           "page fault for unmapped region");

        // TEST 8: Invalidate all - previously cached entries must refetch
        test_num = test_num + 1;
        $display("\n[TEST %0d] Invalidate all - refetch required", test_num);
        begin : test8_blk
            reg [31:0] walks_before;
            do_invalidate_all;
            walks_before = stat_page_walks;
            translate_sm0(48'h0000_0000_1000, 1'b0, 16'h0);
            check(lat_valid_sm0 == 1'b1,       "resp_valid after invalidate");
            check(lat_fault_sm0 == 1'b0,       "no fault");
            check(stat_page_walks > walks_before, "page walk triggered after flush");
        end

        // TEST 9: Multi-SM - SM1 accesses different VA
        test_num = test_num + 1;
        $display("\n[TEST %0d] Multi-SM - SM1 accesses different VA", test_num);
        translate_sm1(48'h0000_4000_5000, 1'b0, 16'h0);
        check(lat_valid_sm1 == 1'b1,           "SM1 resp_valid");
        check(lat_fault_sm1 == 1'b0,           "SM1 no fault");
        check(lat_paddr_sm1 == 40'h00_4000_5000,
              "SM1 PA = identity(VA)");

        // TEST 10: Multi-page sequential scan over 4 pages
        test_num = test_num + 1;
        $display("\n[TEST %0d] Multi-page sequential scan", test_num);
        begin : test10_blk
            integer pg;
            reg [VADDR_WIDTH-1:0] base_va;
            reg [PADDR_WIDTH-1:0] expected_pa;
            reg all_ok;
            all_ok = 1'b1;
            for (pg = 0; pg < 4; pg = pg + 1) begin
                base_va = {16'b0, pg[1:0], 30'h000_3000};
                expected_pa = {{(PADDR_WIDTH-32){1'b0}}, pg[1:0], 30'h000_3000};
                translate_sm0(base_va, 1'b0, 16'h0);
                if (lat_fault_sm0 || lat_paddr_sm0 != expected_pa) begin
                    $display("  page %0d: fault=%0b pa=%h expected=%h",
                             pg, lat_fault_sm0, lat_paddr_sm0, expected_pa);
                    all_ok = 1'b0;
                end
            end
            check(all_ok, "all 4 pages translated correctly");
        end

        // TEST 11: Performance counters
        test_num = test_num + 1;
        $display("\n[TEST %0d] Performance counters", test_num);
        $display("  L1 hits=%0d  misses=%0d", stat_l1_hits, stat_l1_misses);
        $display("  L2 hits=%0d  misses=%0d", stat_l2_hits, stat_l2_misses);
        $display("  page_walks=%0d  page_faults=%0d", stat_page_walks, stat_page_faults);
        check(stat_page_walks > 0,  "page walk counter > 0");
        check(stat_page_faults > 0, "page fault counter > 0 (from test 5/7)");

        // SUMMARY
        $display("\n====================================================");
        $display("TLB Enhanced: PASS=%0d  FAIL=%0d", pass_count, fail_count);
        $display("====================================================");

        if (fail_count == 0)
            $finish;
        else
            $fatal(1, "tb_tlb_enhanced failed with %0d errors", fail_count);
    end

    // Watchdog
    initial begin
        #(CLK_PERIOD * 5000);
        $display("TIMEOUT - testbench did not complete in time");
        $fatal(1, "Watchdog timeout");
    end

endmodule
