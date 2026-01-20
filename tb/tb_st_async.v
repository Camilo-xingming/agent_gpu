//============================================================================
// RalphGPU - st.async Testbench
// Tests asynchronous store operations (st.async.global, st.async.shared)
//============================================================================

`timescale 1ns/1ps
`include "gpu_defines.vh"

module tb_st_async;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter SHARED_MEM_ADDR_W = 14;
    parameter GLOBAL_ADDR_W = 32;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg  [5:0]                   opcode;
    reg  [5:0]                   func;
    reg                          valid_in;
    reg  [31:0]                  src_addr;
    reg  [SHARED_MEM_ADDR_W-1:0] dst_addr;
    reg  [3:0]                   size;
    reg  [2:0]                   cache_hint;
    reg  [3:0]                   wait_count;

    // TMA Interface (not used in this test)
    reg  [63:0]                  tensor_desc;
    reg  [31:0]                  tensor_coord_x;
    reg  [31:0]                  tensor_coord_y;

    // st.async Interface
    reg                          is_store;
    reg  [31:0]                  store_gmem_addr;
    reg  [127:0]                 store_data;

    // Status
    wire                         ready;
    wire                         done;
    wire [3:0]                   pending_count;
    wire                         tma_busy;

    // Global memory read interface
    wire                         gmem_req_valid;
    wire [GLOBAL_ADDR_W-1:0]     gmem_req_addr;
    wire [4:0]                   gmem_req_size;
    wire [2:0]                   gmem_req_cache;
    reg                          gmem_resp_valid;
    reg  [127:0]                 gmem_resp_data;

    // Global memory write interface
    wire                         gmem_wr_valid;
    wire [GLOBAL_ADDR_W-1:0]     gmem_wr_addr;
    wire [127:0]                 gmem_wr_data;
    wire [4:0]                   gmem_wr_size;
    reg                          gmem_wr_done;

    // Shared memory write interface
    wire                         smem_wr_en;
    wire [SHARED_MEM_ADDR_W-1:0] smem_wr_addr;
    wire [127:0]                 smem_wr_data;
    wire [4:0]                   smem_wr_size;

    // Shared memory read interface
    wire                         smem_rd_en;
    wire [SHARED_MEM_ADDR_W-1:0] smem_rd_addr;
    reg  [127:0]                 smem_rd_data;
    reg                          smem_rd_valid;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    async_copy_engine #(
        .MAX_GROUPS(8),
        .MAX_PENDING(16),
        .SHARED_MEM_ADDR_W(SHARED_MEM_ADDR_W),
        .GLOBAL_ADDR_W(GLOBAL_ADDR_W)
    ) u_ace (
        .clk(clk),
        .rst_n(rst_n),
        // Control
        .opcode(opcode),
        .func(func),
        .valid_in(valid_in),
        .src_addr(src_addr),
        .dst_addr(dst_addr),
        .size(size),
        .cache_hint(cache_hint),
        .wait_count(wait_count),
        // TMA
        .tensor_desc(tensor_desc),
        .tensor_coord_x(tensor_coord_x),
        .tensor_coord_y(tensor_coord_y),
        // st.async
        .is_store(is_store),
        .store_gmem_addr(store_gmem_addr),
        .store_data(store_data),
        // Status
        .ready(ready),
        .done(done),
        .pending_count(pending_count),
        .tma_busy(tma_busy),
        // Global memory read
        .gmem_req_valid(gmem_req_valid),
        .gmem_req_addr(gmem_req_addr),
        .gmem_req_size(gmem_req_size),
        .gmem_req_cache(gmem_req_cache),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_data(gmem_resp_data),
        // Global memory write
        .gmem_wr_valid(gmem_wr_valid),
        .gmem_wr_addr(gmem_wr_addr),
        .gmem_wr_data(gmem_wr_data),
        .gmem_wr_size(gmem_wr_size),
        .gmem_wr_done(gmem_wr_done),
        // Shared memory write
        .smem_wr_en(smem_wr_en),
        .smem_wr_addr(smem_wr_addr),
        .smem_wr_data(smem_wr_data),
        .smem_wr_size(smem_wr_size),
        // Shared memory read
        .smem_rd_en(smem_rd_en),
        .smem_rd_addr(smem_rd_addr),
        .smem_rd_data(smem_rd_data),
        .smem_rd_valid(smem_rd_valid)
    );

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_count;
    integer pass_count;
    integer fail_count;

    // Global memory model
    reg [127:0] gmem [0:255];  // 256 x 128-bit = 4KB

    //------------------------------------------------------------------------
    // Global Memory Model
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        gmem_wr_done <= 1'b0;
        if (gmem_wr_valid) begin
            // Write to global memory
            gmem[gmem_wr_addr[11:4]] <= gmem_wr_data;
            gmem_wr_done <= 1'b1;
            $display("[GMEM] Write: addr=0x%08x data=0x%032x size=%0d",
                     gmem_wr_addr, gmem_wr_data[31:0], gmem_wr_size);
        end
    end

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 0;
            opcode = 6'b0;
            func = 6'b0;
            valid_in = 0;
            src_addr = 32'b0;
            dst_addr = {SHARED_MEM_ADDR_W{1'b0}};
            size = 4'd4;
            cache_hint = 3'b0;
            wait_count = 4'b0;
            tensor_desc = 64'b0;
            tensor_coord_x = 32'b0;
            tensor_coord_y = 32'b0;
            is_store = 0;
            store_gmem_addr = 32'b0;
            store_data = 128'b0;
            gmem_resp_valid = 0;
            gmem_resp_data = 128'b0;
            smem_rd_data = 128'b0;
            smem_rd_valid = 0;

            repeat (5) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);
        end
    endtask

    task test_st_async_global;
        input [31:0] gmem_addr;
        input [31:0] data_val;
        input [3:0]  data_size;
        integer timeout;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: st.async.global ---", test_count);
            $display("  addr=0x%08x data=0x%08x size=%0d", gmem_addr, data_val, data_size);

            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_GLOBAL;
            valid_in <= 1;
            is_store <= 1;
            store_gmem_addr <= gmem_addr;
            store_data <= {96'b0, data_val};
            size <= data_size;
            dst_addr <= 14'h0;  // SMEM addr (not used for st.async.global with direct data)

            @(posedge clk);
            valid_in <= 0;
            is_store <= 0;
            opcode <= 6'b0;

            // Wait for gmem_wr_valid (the actual write)
            timeout = 0;
            while (!gmem_wr_valid && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end

            if (timeout >= 100) begin
                $display("FAIL: Timeout waiting for gmem_wr_valid");
                fail_count = fail_count + 1;
            end else begin
                // Wait for write completion
                @(posedge clk);
                @(posedge clk);
                @(posedge clk);

                // Verify global memory
                if (gmem[gmem_addr[11:4]][31:0] == data_val) begin
                    $display("PASS: Data written correctly to global memory");
                    pass_count = pass_count + 1;
                end else begin
                    $display("FAIL: Expected 0x%08x, got 0x%08x", data_val, gmem[gmem_addr[11:4]][31:0]);
                    fail_count = fail_count + 1;
                end
            end
        end
    endtask

    task test_st_async_shared;
        input [13:0] smem_addr;
        input [31:0] data_val;
        input [3:0]  data_size;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: st.async.shared ---", test_count);
            $display("  addr=0x%04x data=0x%08x size=%0d", smem_addr, data_val, data_size);

            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_SHARED;
            valid_in <= 1;
            dst_addr <= smem_addr;
            store_data <= {96'b0, data_val};
            size <= data_size;

            @(posedge clk);
            valid_in <= 0;
            opcode <= 6'b0;

            // Wait for SMEM write
            wait (smem_wr_en);
            @(posedge clk);

            if (smem_wr_addr == smem_addr && smem_wr_data[31:0] == data_val) begin
                $display("PASS: st.async.shared issued correctly");
                $display("  smem_wr_addr=0x%04x smem_wr_data=0x%08x", smem_wr_addr, smem_wr_data[31:0]);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Incorrect SMEM write");
                $display("  Expected addr=0x%04x data=0x%08x", smem_addr, data_val);
                $display("  Got addr=0x%04x data=0x%08x", smem_wr_addr, smem_wr_data[31:0]);
                fail_count = fail_count + 1;
            end

            wait (!smem_wr_en);
            @(posedge clk);
        end
    endtask

    task test_st_async_commit_wait;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: st.async commit/wait ---", test_count);

            // Issue several stores
            $display("  Issuing 3 st.async.global operations");

            // Store 1
            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_GLOBAL;
            valid_in <= 1;
            is_store <= 1;
            store_gmem_addr <= 32'h0000_1000;
            store_data <= 128'hDEAD_BEEF;
            size <= 4;
            @(posedge clk);
            valid_in <= 0;
            is_store <= 0;
            opcode <= 6'b0;
            // st.async returns immediately (async), wait for store completion
            wait (gmem_wr_done);
            @(posedge clk);

            // Store 2
            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_GLOBAL;
            valid_in <= 1;
            is_store <= 1;
            store_gmem_addr <= 32'h0000_1010;
            store_data <= 128'hCAFE_BABE;
            size <= 4;
            @(posedge clk);
            valid_in <= 0;
            is_store <= 0;
            opcode <= 6'b0;
            // Wait for store completion
            wait (gmem_wr_done);
            @(posedge clk);

            // Commit
            $display("  Committing store group");
            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_COMMIT;
            valid_in <= 1;
            @(posedge clk);
            valid_in <= 0;
            opcode <= 6'b0;
            repeat (5) @(posedge clk);  // Commit is immediate

            // Wait for all
            $display("  Waiting for all stores to complete");
            @(posedge clk);
            opcode <= `OP_CPASYNC;
            func <= `CPASYNC_WAIT_ALL;  // Same wait mechanism
            valid_in <= 1;
            @(posedge clk);
            valid_in <= 0;
            opcode <= 6'b0;

            // Wait for completion
            repeat (20) @(posedge clk);

            if (pending_count == 0) begin
                $display("PASS: All stores completed, pending_count=0");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Stores not completed, pending_count=%0d", pending_count);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_mixed_load_store;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Mixed cp.async (load) and st.async (store) ---", test_count);

            // cp.async load
            $display("  Issuing cp.async.cg (load)");
            @(posedge clk);
            opcode <= `OP_CPASYNC;
            func <= `CPASYNC_CG;
            valid_in <= 1;
            src_addr <= 32'h0000_2000;
            dst_addr <= 14'h100;
            size <= 4;
            @(posedge clk);
            valid_in <= 0;
            opcode <= 6'b0;

            // Provide load response
            wait (gmem_req_valid);
            @(posedge clk);
            gmem_resp_valid <= 1;
            gmem_resp_data <= 128'h12345678_AABBCCDD_11223344_55667788;
            @(posedge clk);
            gmem_resp_valid <= 0;

            // Wait for load to complete
            wait (smem_wr_en);
            @(posedge clk);
            @(posedge clk);

            // st.async store
            $display("  Issuing st.async.global (store)");
            @(posedge clk);
            opcode <= `OP_ST_ASYNC;
            func <= `ST_ASYNC_GLOBAL;
            valid_in <= 1;
            is_store <= 1;
            store_gmem_addr <= 32'h0000_0300;  // Use address within gmem range
            store_data <= 128'hFEED_FACE;
            size <= 4;
            @(posedge clk);
            valid_in <= 0;
            is_store <= 0;
            opcode <= 6'b0;

            // Wait for store
            wait (gmem_wr_valid);
            @(posedge clk);
            @(posedge clk);

            if (gmem[32'h0300 >> 4][31:0] == 32'hFEED_FACE) begin
                $display("PASS: Mixed load/store operations work correctly");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Store data mismatch");
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU st.async Testbench");
        $display("============================================================");

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Initialize global memory
        for (integer i = 0; i < 256; i = i + 1) begin
            gmem[i] = 128'b0;
        end

        reset_dut();

        // Test 1: Basic st.async.global
        test_st_async_global(32'h0000_0100, 32'hDEAD_BEEF, 4);

        // Test 2: st.async.global with different address
        test_st_async_global(32'h0000_0200, 32'hCAFE_BABE, 4);

        // Test 3: st.async.shared
        test_st_async_shared(14'h0400, 32'h1234_5678, 4);

        // Test 4: st.async.shared different address
        test_st_async_shared(14'h0500, 32'hABCD_EF01, 4);

        // Test 5: Commit and wait
        test_st_async_commit_wait();

        // Test 6: Mixed load/store
        test_mixed_load_store();

        // Summary
        $display("\n============================================================");
        $display("Test Summary: %0d/%0d tests passed", pass_count, test_count);
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("FAILURES: %0d", fail_count);
        end
        $display("============================================================");

        #100;
        $finish;
    end

    // Timeout watchdog
    initial begin
        #50000;
        $display("ERROR: Test timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_st_async.vcd");
        $dumpvars(0, tb_st_async);
    end

endmodule
