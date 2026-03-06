//============================================================================
// RalphGPU - TMA + WGMMA Integration Test
// Tests async copy (TMA) combined with WGMMA matrix multiply
// Pattern: cp.async -> commit -> wait -> WGMMA -> fence
//============================================================================

`timescale 1ns / 1ps

module tb_tma_wgmma_integration;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;

    //========================================================================
    // Async Copy Engine
    //========================================================================
    reg  [5:0]  ace_func;
    reg         ace_valid_in;
    reg  [31:0] ace_src_addr;
    reg  [13:0] ace_dst_addr;
    reg  [3:0]  ace_size;
    reg  [2:0]  ace_cache_hint;
    reg  [3:0]  ace_wait_count;
    wire        ace_ready;
    wire        ace_done;
    wire [3:0]  ace_pending_count;
    wire        ace_gmem_req_valid;
    wire [31:0] ace_gmem_req_addr;
    wire [4:0]  ace_gmem_req_size;
    wire [2:0]  ace_gmem_req_cache;
    reg         ace_gmem_resp_valid;
    reg  [127:0] ace_gmem_resp_data;
    wire        ace_smem_wr_en;
    wire [13:0] ace_smem_wr_addr;
    wire [127:0] ace_smem_wr_data;
    wire [4:0]  ace_smem_wr_size;

    //========================================================================
    // WGMMA
    //========================================================================
    reg  [5:0]  wgmma_func;
    reg         wgmma_valid_in;
    reg  [2:0]  wgmma_warpgroup_id;
    reg  [3:0]  wgmma_wait_count;
    reg  [63:0] wgmma_desc_a;
    reg  [63:0] wgmma_desc_b;
    reg  [31:0] wgmma_scale_d;
    reg  [511:0] wgmma_data_a;
    reg  [511:0] wgmma_data_b;
    reg  [1023:0] wgmma_accum_in;
    wire [1023:0] wgmma_accum_out;
    wire         wgmma_ready;
    wire         wgmma_done;
    wire [3:0]   wgmma_pending_ops;

    //========================================================================
    // Simulated shared memory (target of async copy, source for WGMMA)
    //========================================================================
    reg [127:0] shared_mem [0:255];

    //========================================================================
    // Simulated global memory (source for async copy)
    //========================================================================
    reg [127:0] global_mem [0:255];

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //========================================================================
    // DUT: Async Copy Engine
    //========================================================================
    async_copy_engine #(
        .MAX_GROUPS(8),
        .MAX_PENDING(16),
        .SHARED_MEM_ADDR_W(14),
        .GLOBAL_ADDR_W(32)
    ) u_ace (
        .clk            (clk),
        .rst_n          (rst_n),
        .func           (ace_func),
        .valid_in       (ace_valid_in),
        .src_addr       (ace_src_addr),
        .dst_addr       (ace_dst_addr),
        .size           (ace_size),
        .cache_hint     (ace_cache_hint),
        .wait_count     (ace_wait_count),
        .ready          (ace_ready),
        .done           (ace_done),
        .pending_count  (ace_pending_count),
        .gmem_req_valid (ace_gmem_req_valid),
        .gmem_req_addr  (ace_gmem_req_addr),
        .gmem_req_size  (ace_gmem_req_size),
        .gmem_req_cache (ace_gmem_req_cache),
        .gmem_resp_valid(ace_gmem_resp_valid),
        .gmem_resp_data (ace_gmem_resp_data),
        .smem_wr_en     (ace_smem_wr_en),
        .smem_wr_addr   (ace_smem_wr_addr),
        .smem_wr_data   (ace_smem_wr_data),
        .smem_wr_size   (ace_smem_wr_size)
    );

    //========================================================================
    // DUT: WGMMA
    //========================================================================
    wgmma #(
        .WARPGROUP_SIZE(4),
        .THREADS_PER_WARP(32),
        .MAX_PENDING_OPS(8)
    ) u_wgmma (
        .clk            (clk),
        .rst_n          (rst_n),
        .func           (wgmma_func),
        .valid_in       (wgmma_valid_in),
        .warpgroup_id   (wgmma_warpgroup_id),
        .wait_count     (wgmma_wait_count),
        .desc_a         (wgmma_desc_a),
        .desc_b         (wgmma_desc_b),
        .scale_d        (wgmma_scale_d),
        .data_a         (wgmma_data_a),
        .data_b         (wgmma_data_b),
        .accum_in       (wgmma_accum_in),
        .accum_out      (wgmma_accum_out),
        .ready          (wgmma_ready),
        .done           (wgmma_done),
        .pending_ops    (wgmma_pending_ops)
    );

    //========================================================================
    // Global memory responder
    //========================================================================
    reg [31:0] gmem_pending_addr;
    reg gmem_resp_pending;
    reg [3:0] gmem_delay;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ace_gmem_resp_valid <= 1'b0;
            ace_gmem_resp_data <= 128'b0;
            gmem_resp_pending <= 1'b0;
        end else begin
            ace_gmem_resp_valid <= 1'b0;

            if (ace_gmem_req_valid && !gmem_resp_pending) begin
                gmem_pending_addr <= ace_gmem_req_addr;
                gmem_resp_pending <= 1'b1;
                gmem_delay <= 4'd3;  // 3 cycle latency
            end else if (gmem_resp_pending) begin
                if (gmem_delay == 0) begin
                    ace_gmem_resp_valid <= 1'b1;
                    ace_gmem_resp_data <= global_mem[gmem_pending_addr[11:4]];
                    gmem_resp_pending <= 1'b0;
                end else begin
                    gmem_delay <= gmem_delay - 1;
                end
            end
        end
    end

    //========================================================================
    // Shared memory capture
    //========================================================================
    always @(posedge clk) begin
        if (ace_smem_wr_en) begin
            shared_mem[ace_smem_wr_addr[11:4]] <= ace_smem_wr_data;
            $display("[SMEM] Write addr=0x%04x data=0x%032h",
                     ace_smem_wr_addr, ace_smem_wr_data);
        end
    end

    //========================================================================
    // Test helpers
    //========================================================================
    task check_result;
        input [255:0] test_name;
        input [31:0] expected;
        input [31:0] actual;
        begin
            if (expected == actual) begin
                $display("[PASS] Test %0d: %0s", test_num, test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s - expected 0x%h, got 0x%h",
                         test_num, test_name, expected, actual);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
        end
    endtask

    // Issue cp.async
    task issue_cpasync;
        input [5:0] op_func;
        input [31:0] src;
        input [13:0] dst;
        input [3:0] sz;
        begin
            @(posedge clk);
            ace_func <= op_func;
            ace_src_addr <= src;
            ace_dst_addr <= dst;
            ace_size <= sz;
            ace_valid_in <= 1'b1;
            @(posedge clk);
            ace_valid_in <= 1'b0;
        end
    endtask

    // Issue WGMMA operation
    task issue_wgmma;
        input [5:0] op_func;
        input [2:0] wg_id;
        begin
            @(posedge clk);
            wgmma_func <= op_func;
            wgmma_warpgroup_id <= wg_id;
            wgmma_valid_in <= 1'b1;
            @(posedge clk);
            wgmma_valid_in <= 1'b0;
        end
    endtask

    // Wait for async copy to complete
    task wait_ace_ready;
        begin
            while (!ace_ready) @(posedge clk);
        end
    endtask

    // Wait for WGMMA to complete
    task wait_wgmma_ready;
        begin
            while (!wgmma_ready) @(posedge clk);
        end
    endtask

    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU TMA + WGMMA Integration Test");
        $display("Tests: cp.async -> commit -> wait -> WGMMA -> fence");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        ace_valid_in = 0;
        ace_func = 0;
        ace_src_addr = 0;
        ace_dst_addr = 0;
        ace_size = 0;
        ace_cache_hint = 0;
        ace_wait_count = 0;
        wgmma_valid_in = 0;
        wgmma_func = 0;
        wgmma_warpgroup_id = 0;
        wgmma_wait_count = 0;
        wgmma_desc_a = 64'h0000_0010_0000_0000;
        wgmma_desc_b = 64'h0000_0010_0000_0100;
        wgmma_scale_d = 32'h3F800000;  // 1.0
        wgmma_data_a = 512'b0;
        wgmma_data_b = 512'b0;
        wgmma_accum_in = 1024'b0;
        test_num = 1;
        pass_count = 0;
        fail_count = 0;

        // Initialize global memory with matrix data
        for (i = 0; i < 256; i = i + 1) begin
            // Matrix A and B test patterns
            global_mem[i] = {4{i[7:0], 8'h01, 8'h02, 8'h03}};
            shared_mem[i] = 128'hDEAD_BEEF_DEAD_BEEF_DEAD_BEEF_DEAD_BEEF;
        end

        // Special test data for matrices
        global_mem[0] = 128'h0001_0002_0003_0004_0005_0006_0007_0008;  // Matrix A row 0
        global_mem[1] = 128'h0009_000A_000B_000C_000D_000E_000F_0010;  // Matrix A row 1
        global_mem[16] = 128'h0011_0012_0013_0014_0015_0016_0017_0018; // Matrix B row 0
        global_mem[17] = 128'h0019_001A_001B_001C_001D_001E_001F_0020; // Matrix B row 1

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Test 1: Initial states
        //==================================================================
        check_result("ACE ready after reset", 1, ace_ready);
        check_result("WGMMA ready after reset", 1, wgmma_ready);
        check_result("No ACE pending", 0, ace_pending_count);
        check_result("No WGMMA pending", 0, wgmma_pending_ops);

        //==================================================================
        // Test 2: TMA Phase - Load Matrix A to shared memory
        //==================================================================
        $display("\n--- Phase 1: TMA - Load Matrix A ---");

        // cp.async to load Matrix A (16 bytes at a time)
        issue_cpasync(`CPASYNC_CA, 32'h0000_0000, 14'h0000, 4'd16);  // A row 0
        issue_cpasync(`CPASYNC_CA, 32'h0000_0010, 14'h0010, 4'd16);  // A row 1

        // Commit the copy group
        @(posedge clk);
        ace_func <= `CPASYNC_COMMIT;
        ace_valid_in <= 1'b1;
        @(posedge clk);
        ace_valid_in <= 1'b0;

        // Wait for copies to complete
        repeat(50) @(posedge clk);
        wait_ace_ready();

        check_result("Matrix A loaded", 0, ace_pending_count);
        check_result("Shared mem[0] has A[0]", global_mem[0], shared_mem[0]);

        //==================================================================
        // Test 3: TMA Phase - Load Matrix B to shared memory
        //==================================================================
        $display("\n--- Phase 2: TMA - Load Matrix B ---");

        issue_cpasync(`CPASYNC_CA, 32'h0000_0100, 14'h0100, 4'd16);  // B row 0
        issue_cpasync(`CPASYNC_CA, 32'h0000_0110, 14'h0110, 4'd16);  // B row 1

        @(posedge clk);
        ace_func <= `CPASYNC_COMMIT;
        ace_valid_in <= 1'b1;
        @(posedge clk);
        ace_valid_in <= 1'b0;

        repeat(50) @(posedge clk);
        wait_ace_ready();

        check_result("Matrix B loaded", 0, ace_pending_count);

        //==================================================================
        // Test 4: Wait for all TMA operations
        //==================================================================
        $display("\n--- Phase 3: Wait All TMA ---");

        @(posedge clk);
        ace_func <= `CPASYNC_WAIT_ALL;
        ace_valid_in <= 1'b1;
        @(posedge clk);
        ace_valid_in <= 1'b0;

        repeat(20) @(posedge clk);
        wait_ace_ready();

        check_result("All TMA complete", 0, ace_pending_count);

        //==================================================================
        // Test 5: WGMMA Phase - Matrix multiply
        //==================================================================
        $display("\n--- Phase 4: WGMMA Compute ---");

        // Setup WGMMA data from shared memory
        wgmma_data_a <= {shared_mem[1], shared_mem[0]};
        wgmma_data_b <= {shared_mem[17], shared_mem[16]};
        wgmma_accum_in <= 1024'b0;

        // Issue WGMMA M64N8K16
        issue_wgmma(`WGMMA_M64N8K16, 3'd0);

        repeat(20) @(posedge clk);
        wait_wgmma_ready();

        check_result("WGMMA compute done", 0, wgmma_pending_ops);

        //==================================================================
        // Test 6: WGMMA Commit and Wait
        //==================================================================
        $display("\n--- Phase 5: WGMMA Commit/Wait ---");

        issue_wgmma(`WGMMA_COMMIT_GROUP, 3'd0);
        @(posedge clk);
        @(posedge clk);

        wgmma_wait_count <= 4'd0;
        issue_wgmma(`WGMMA_WAIT_GROUP, 3'd0);

        repeat(10) @(posedge clk);
        wait_wgmma_ready();

        check_result("WGMMA wait complete", 0, wgmma_pending_ops);

        //==================================================================
        // Test 7: WGMMA Fence
        //==================================================================
        $display("\n--- Phase 6: WGMMA Fence ---");

        issue_wgmma(`WGMMA_FENCE, 3'd0);

        repeat(20) @(posedge clk);
        wait_wgmma_ready();

        check_result("WGMMA fence complete", 0, wgmma_pending_ops);
        check_result("WGMMA ready after fence", 1, wgmma_ready);

        //==================================================================
        // Test 8: Full pipeline - TMA + WGMMA together
        //==================================================================
        $display("\n--- Phase 7: Full Pipeline Test ---");

        // Start new TMA while WGMMA is idle
        issue_cpasync(`CPASYNC_CA, 32'h0000_0200, 14'h0200, 4'd16);

        // Start WGMMA with previous data
        issue_wgmma(`WGMMA_M64N8K16, 3'd1);

        // Commit TMA
        @(posedge clk);
        ace_func <= `CPASYNC_COMMIT;
        ace_valid_in <= 1'b1;
        @(posedge clk);
        ace_valid_in <= 1'b0;

        // Wait for both to complete
        repeat(100) @(posedge clk);
        wait_ace_ready();
        wait_wgmma_ready();

        check_result("Parallel TMA+WGMMA done (ACE)", 0, ace_pending_count);
        check_result("Parallel TMA+WGMMA done (WGMMA)", 0, wgmma_pending_ops);

        //==================================================================
        // Test 9: Multiple WGMMA with TMA interleaved
        //==================================================================
        $display("\n--- Phase 8: Interleaved TMA+WGMMA ---");

        // Issue TMA
        issue_cpasync(`CPASYNC_CA, 32'h0000_0300, 14'h0300, 4'd16);

        // Issue WGMMA
        issue_wgmma(`WGMMA_M64N16K16, 3'd0);

        // More TMA
        issue_cpasync(`CPASYNC_CA, 32'h0000_0310, 14'h0310, 4'd16);

        // Another WGMMA
        issue_wgmma(`WGMMA_M64N32K16, 3'd0);

        // Commit and wait all
        @(posedge clk);
        ace_func <= `CPASYNC_COMMIT;
        ace_valid_in <= 1'b1;
        @(posedge clk);
        ace_valid_in <= 1'b0;

        issue_wgmma(`WGMMA_FENCE, 3'd0);

        repeat(150) @(posedge clk);
        wait_ace_ready();
        wait_wgmma_ready();

        check_result("Interleaved complete (ACE)", 0, ace_pending_count);
        check_result("Interleaved complete (WGMMA)", 0, wgmma_pending_ops);

        //==================================================================
        // Results Summary
        //==================================================================
        #100;
        $display("\n============================================================");
        $display("TMA + WGMMA Integration Test Results");
        $display("============================================================");
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        $display("============================================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout
    initial begin
        #200000;
        $display("ERROR: Test timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_tma_wgmma_integration.vcd");
        $dumpvars(0, tb_tma_wgmma_integration);
    end

endmodule
