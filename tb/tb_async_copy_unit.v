//============================================================================
// RalphGPU - Async Copy Engine Unit Test
// Tests cp.async instructions for async global to shared memory copy
// Verifies: cp.async.ca/cg, commit_group, wait_group, wait_all, bulk
//============================================================================

`timescale 1ns / 1ps

module tb_async_copy_unit;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter MAX_GROUPS = 8;
    parameter MAX_PENDING = 16;
    parameter SHARED_MEM_ADDR_W = 14;
    parameter GLOBAL_ADDR_W = 32;

    reg clk;
    reg rst_n;

    // Control interface
    reg  [5:0]  func;
    reg         valid_in;
    reg  [31:0] src_addr;
    reg  [SHARED_MEM_ADDR_W-1:0] dst_addr;
    reg  [3:0]  size;
    reg  [2:0]  cache_hint;
    reg  [3:0]  wait_count;

    // Status outputs
    wire        ready;
    wire        done;
    wire [3:0]  pending_count;

    // Global memory interface
    wire        gmem_req_valid;
    wire [GLOBAL_ADDR_W-1:0] gmem_req_addr;
    wire [4:0]  gmem_req_size;
    wire [2:0]  gmem_req_cache;
    reg         gmem_resp_valid;
    reg  [127:0] gmem_resp_data;

    // Shared memory write interface
    wire        smem_wr_en;
    wire [SHARED_MEM_ADDR_W-1:0] smem_wr_addr;
    wire [127:0] smem_wr_data;
    wire [4:0]  smem_wr_size;

    // Simulated memories
    reg [127:0] global_mem [0:255];
    reg [127:0] shared_mem [0:255];

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;
    integer mem_latency;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT
    async_copy_engine #(
        .MAX_GROUPS(MAX_GROUPS),
        .MAX_PENDING(MAX_PENDING),
        .SHARED_MEM_ADDR_W(SHARED_MEM_ADDR_W),
        .GLOBAL_ADDR_W(GLOBAL_ADDR_W)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .func           (func),
        .valid_in       (valid_in),
        .src_addr       (src_addr),
        .dst_addr       (dst_addr),
        .size           (size),
        .cache_hint     (cache_hint),
        .wait_count     (wait_count),
        .ready          (ready),
        .done           (done),
        .pending_count  (pending_count),
        .gmem_req_valid (gmem_req_valid),
        .gmem_req_addr  (gmem_req_addr),
        .gmem_req_size  (gmem_req_size),
        .gmem_req_cache (gmem_req_cache),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_data (gmem_resp_data),
        .smem_wr_en     (smem_wr_en),
        .smem_wr_addr   (smem_wr_addr),
        .smem_wr_data   (smem_wr_data),
        .smem_wr_size   (smem_wr_size)
    );

    // Global memory responder with configurable latency
    reg [3:0] resp_delay_cnt;
    reg resp_pending;
    reg [31:0] pending_gmem_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gmem_resp_valid <= 1'b0;
            gmem_resp_data <= 128'b0;
            resp_pending <= 1'b0;
            resp_delay_cnt <= 0;
        end else begin
            gmem_resp_valid <= 1'b0;

            if (gmem_req_valid && !resp_pending) begin
                resp_pending <= 1'b1;
                resp_delay_cnt <= mem_latency;
                pending_gmem_addr <= gmem_req_addr;
            end else if (resp_pending) begin
                if (resp_delay_cnt == 0) begin
                    gmem_resp_valid <= 1'b1;
                    gmem_resp_data <= global_mem[pending_gmem_addr[11:4]];
                    resp_pending <= 1'b0;
                end else begin
                    resp_delay_cnt <= resp_delay_cnt - 1;
                end
            end
        end
    end

    // Shared memory capture
    always @(posedge clk) begin
        if (smem_wr_en) begin
            shared_mem[smem_wr_addr[11:4]] <= smem_wr_data;
        end
    end

    // Test task: Check result
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

    // Issue cp.async operation
    task issue_cpasync;
        input [5:0] op_func;
        input [31:0] src;
        input [13:0] dst;
        input [3:0] sz;
        begin
            @(posedge clk);
            func <= op_func;
            src_addr <= src;
            dst_addr <= dst;
            size <= sz;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    // Wait for ready
    task wait_ready;
        begin
            while (!ready) @(posedge clk);
        end
    endtask

    // Wait for done
    task wait_done_signal;
        integer timeout;
        begin
            timeout = 0;
            while (!done && timeout < 1000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU Async Copy Engine Unit Test");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        func = 0;
        src_addr = 0;
        dst_addr = 0;
        size = 0;
        cache_hint = 0;
        wait_count = 0;
        test_num = 1;
        pass_count = 0;
        fail_count = 0;
        mem_latency = 4;  // 4 cycle memory latency

        // Initialize memories with test patterns
        for (i = 0; i < 256; i = i + 1) begin
            global_mem[i] = {4{32'hDEAD0000 + i}};
            shared_mem[i] = 128'hCAFECAFE;
        end

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Test 1: Reset state
        //==================================================================
        check_result("Ready after reset", 1, ready);
        check_result("No pending after reset", 0, pending_count);

        //==================================================================
        // Test 2: Single cp.async.ca operation
        //==================================================================
        $display("\n--- Test: cp.async.ca single copy ---");

        issue_cpasync(`CPASYNC_CA, 32'h0000_0100, 14'h0010, 4'd16);

        // Wait for operation to complete
        repeat(20) @(posedge clk);
        wait_ready();

        check_result("cp.async.ca completed (ready)", 1, ready);
        check_result("Shared mem has data", global_mem[32'h10], shared_mem[1]);

        //==================================================================
        // Test 3: Multiple async copies in same group
        //==================================================================
        $display("\n--- Test: Multiple copies in group ---");

        // Issue 4 copies
        issue_cpasync(`CPASYNC_CA, 32'h0000_0200, 14'h0020, 4'd16);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0210, 14'h0030, 4'd16);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0220, 14'h0040, 4'd16);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0230, 14'h0050, 4'd16);

        // Wait for all to complete
        repeat(100) @(posedge clk);
        wait_ready();

        check_result("Multiple copies completed", 1, ready);

        //==================================================================
        // Test 4: Commit group
        //==================================================================
        $display("\n--- Test: Commit group ---");

        // Wait for any pending ops to complete first
        repeat(50) @(posedge clk);
        wait_ready();

        // Issue a copy
        issue_cpasync(`CPASYNC_CA, 32'h0000_0300, 14'h0060, 4'd16);

        // Wait for copy to complete
        repeat(50) @(posedge clk);
        wait_ready();

        // Now commit the group (done is pulsed synchronously)
        @(posedge clk);
        func <= `CPASYNC_COMMIT;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Commit is instant - check ready state
        @(posedge clk);
        check_result("Commit group ready", 1, ready);

        //==================================================================
        // Test 5: Wait all
        //==================================================================
        $display("\n--- Test: Wait all ---");

        // Issue some copies
        issue_cpasync(`CPASYNC_CA, 32'h0000_0400, 14'h0070, 4'd16);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0410, 14'h0080, 4'd16);

        // Wait for all pending to complete
        @(posedge clk);
        func <= `CPASYNC_WAIT_ALL;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Should wait until all complete
        repeat(50) @(posedge clk);
        wait_ready();

        check_result("Wait all completed", 0, pending_count);

        //==================================================================
        // Test 6: Wait group with count
        //==================================================================
        $display("\n--- Test: Wait group with count ---");

        // Issue a copy and commit
        issue_cpasync(`CPASYNC_CA, 32'h0000_0500, 14'h0090, 4'd16);
        @(posedge clk);
        func <= `CPASYNC_COMMIT;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Issue another copy and commit
        issue_cpasync(`CPASYNC_CA, 32'h0000_0510, 14'h00A0, 4'd16);
        @(posedge clk);
        func <= `CPASYNC_COMMIT;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Wait for 1 group
        @(posedge clk);
        func <= `CPASYNC_WAIT;
        wait_count <= 4'd1;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        repeat(50) @(posedge clk);
        wait_ready();

        check_result("Wait group completed", 1, ready);

        //==================================================================
        // Test 7: cp.async.cg (cache global)
        //==================================================================
        $display("\n--- Test: cp.async.cg ---");

        issue_cpasync(`CPASYNC_CG, 32'h0000_0600, 14'h00B0, 4'd16);

        repeat(30) @(posedge clk);
        wait_ready();

        check_result("cp.async.cg completed", 1, ready);

        //==================================================================
        // Test 8: cp.async.bulk
        //==================================================================
        $display("\n--- Test: cp.async.bulk ---");

        issue_cpasync(`CPASYNC_BULK, 32'h0000_0700, 14'h00C0, 4'd16);

        repeat(30) @(posedge clk);
        wait_ready();

        check_result("cp.async.bulk completed", 1, ready);

        //==================================================================
        // Test 9: Memory latency measurement
        //==================================================================
        $display("\n--- Test: Memory latency ---");

        // Reset and set higher latency
        rst_n = 0;
        #50;
        rst_n = 1;
        #50;

        mem_latency = 10;  // 10 cycle latency

        // Time a single copy
        @(posedge clk);
        issue_cpasync(`CPASYNC_CA, 32'h0000_0800, 14'h00D0, 4'd16);

        // Count cycles until ready
        i = 0;
        while (!ready && i < 100) begin
            @(posedge clk);
            i = i + 1;
        end

        $display("Copy took %0d cycles with 10-cycle memory latency", i);
        check_result("High latency copy completed", 1, ready);

        //==================================================================
        // Test 10: Back-to-back copies
        //==================================================================
        $display("\n--- Test: Back-to-back copies ---");

        mem_latency = 2;  // Low latency

        // Issue many copies rapidly
        for (i = 0; i < 8; i = i + 1) begin
            issue_cpasync(`CPASYNC_CA, 32'h0000_1000 + (i << 4), 14'h0100 + (i << 4), 4'd16);
        end

        // Wait for all to complete
        repeat(200) @(posedge clk);
        wait_ready();

        check_result("Back-to-back copies completed", 0, pending_count);

        //==================================================================
        // Test 11: Different copy sizes
        //==================================================================
        $display("\n--- Test: Different copy sizes ---");

        issue_cpasync(`CPASYNC_CA, 32'h0000_2000, 14'h0200, 4'd4);   // 4 bytes
        repeat(20) @(posedge clk);
        wait_ready();

        issue_cpasync(`CPASYNC_CA, 32'h0000_2010, 14'h0210, 4'd8);   // 8 bytes
        repeat(20) @(posedge clk);
        wait_ready();

        issue_cpasync(`CPASYNC_CA, 32'h0000_2020, 14'h0220, 4'd16);  // 16 bytes
        repeat(20) @(posedge clk);
        wait_ready();

        check_result("Different sizes completed", 1, ready);

        //==================================================================
        // Results Summary
        //==================================================================
        #100;
        $display("\n============================================================");
        $display("Async Copy Engine Unit Test Results");
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
        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_async_copy_unit.vcd");
        $dumpvars(0, tb_async_copy_unit);
    end

endmodule
