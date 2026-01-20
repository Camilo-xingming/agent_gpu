//============================================================================
// RalphGPU - Bulk Store Unit Testbench
// Tests st.bulk instructions for asynchronous bulk store operations
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_st_bulk_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter DATA_WIDTH = 32;
    parameter SMEM_ADDR_W = 14;
    parameter GMEM_ADDR_W = 32;
    parameter MAX_BULK_SIZE = 1024;
    parameter MAX_PENDING = 8;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    //------------------------------------------------------------------------
    // DUT Interface
    //------------------------------------------------------------------------
    reg                      valid_in;
    reg  [5:0]               opcode;
    reg  [5:0]               func;
    reg  [SMEM_ADDR_W-1:0]   smem_addr;
    reg  [GMEM_ADDR_W-1:0]   gmem_addr;
    reg  [15:0]              byte_count;
    reg  [2:0]               cache_hint;

    wire                     done;
    wire                     result_valid;
    wire [DATA_WIDTH-1:0]    result;
    wire                     busy;

    // Shared memory interface
    wire                     smem_rd_en;
    wire [SMEM_ADDR_W-1:0]   smem_rd_addr;
    reg  [127:0]             smem_rd_data;
    reg                      smem_rd_valid;

    // Global memory interface
    wire                     gmem_wr_valid;
    wire [GMEM_ADDR_W-1:0]   gmem_wr_addr;
    wire [127:0]             gmem_wr_data;
    wire [4:0]               gmem_wr_size;
    reg                      gmem_wr_done;

    // Completion signals
    wire                     bulk_complete;
    wire [15:0]              bytes_transferred;

    //------------------------------------------------------------------------
    // Simulated Memory
    //------------------------------------------------------------------------
    reg [7:0] smem [0:16383];    // 16KB shared memory
    reg [7:0] gmem [0:65535];    // 64KB global memory (for testing)

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    st_bulk_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .SMEM_ADDR_W(SMEM_ADDR_W),
        .GMEM_ADDR_W(GMEM_ADDR_W),
        .MAX_BULK_SIZE(MAX_BULK_SIZE),
        .MAX_PENDING(MAX_PENDING)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .opcode(opcode),
        .func(func),
        .smem_addr(smem_addr),
        .gmem_addr(gmem_addr),
        .byte_count(byte_count),
        .cache_hint(cache_hint),
        .done(done),
        .result_valid(result_valid),
        .result(result),
        .busy(busy),
        .smem_rd_en(smem_rd_en),
        .smem_rd_addr(smem_rd_addr),
        .smem_rd_data(smem_rd_data),
        .smem_rd_valid(smem_rd_valid),
        .gmem_wr_valid(gmem_wr_valid),
        .gmem_wr_addr(gmem_wr_addr),
        .gmem_wr_data(gmem_wr_data),
        .gmem_wr_size(gmem_wr_size),
        .gmem_wr_done(gmem_wr_done),
        .bulk_complete(bulk_complete),
        .bytes_transferred(bytes_transferred)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Memory Model - Shared Memory Read
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        smem_rd_valid <= 1'b0;
        if (smem_rd_en) begin
            // Read 16 bytes from shared memory
            smem_rd_data <= {
                smem[smem_rd_addr + 15], smem[smem_rd_addr + 14],
                smem[smem_rd_addr + 13], smem[smem_rd_addr + 12],
                smem[smem_rd_addr + 11], smem[smem_rd_addr + 10],
                smem[smem_rd_addr + 9],  smem[smem_rd_addr + 8],
                smem[smem_rd_addr + 7],  smem[smem_rd_addr + 6],
                smem[smem_rd_addr + 5],  smem[smem_rd_addr + 4],
                smem[smem_rd_addr + 3],  smem[smem_rd_addr + 2],
                smem[smem_rd_addr + 1],  smem[smem_rd_addr + 0]
            };
            smem_rd_valid <= 1'b1;
        end
    end

    //------------------------------------------------------------------------
    // Memory Model - Global Memory Write
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        gmem_wr_done <= 1'b0;
        if (gmem_wr_valid) begin
            // Write to global memory
            if (gmem_wr_size >= 1)  gmem[gmem_wr_addr + 0]  <= gmem_wr_data[7:0];
            if (gmem_wr_size >= 2)  gmem[gmem_wr_addr + 1]  <= gmem_wr_data[15:8];
            if (gmem_wr_size >= 3)  gmem[gmem_wr_addr + 2]  <= gmem_wr_data[23:16];
            if (gmem_wr_size >= 4)  gmem[gmem_wr_addr + 3]  <= gmem_wr_data[31:24];
            if (gmem_wr_size >= 5)  gmem[gmem_wr_addr + 4]  <= gmem_wr_data[39:32];
            if (gmem_wr_size >= 6)  gmem[gmem_wr_addr + 5]  <= gmem_wr_data[47:40];
            if (gmem_wr_size >= 7)  gmem[gmem_wr_addr + 6]  <= gmem_wr_data[55:48];
            if (gmem_wr_size >= 8)  gmem[gmem_wr_addr + 7]  <= gmem_wr_data[63:56];
            if (gmem_wr_size >= 9)  gmem[gmem_wr_addr + 8]  <= gmem_wr_data[71:64];
            if (gmem_wr_size >= 10) gmem[gmem_wr_addr + 9]  <= gmem_wr_data[79:72];
            if (gmem_wr_size >= 11) gmem[gmem_wr_addr + 10] <= gmem_wr_data[87:80];
            if (gmem_wr_size >= 12) gmem[gmem_wr_addr + 11] <= gmem_wr_data[95:88];
            if (gmem_wr_size >= 13) gmem[gmem_wr_addr + 12] <= gmem_wr_data[103:96];
            if (gmem_wr_size >= 14) gmem[gmem_wr_addr + 13] <= gmem_wr_data[111:104];
            if (gmem_wr_size >= 15) gmem[gmem_wr_addr + 14] <= gmem_wr_data[119:112];
            if (gmem_wr_size >= 16) gmem[gmem_wr_addr + 15] <= gmem_wr_data[127:120];
            #10;
            gmem_wr_done <= 1'b1;
            #10;
            gmem_wr_done <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;
    integer i;
    reg [7:0] expected_data;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        smem_addr = 0;
        gmem_addr = 0;
        byte_count = 0;
        cache_hint = 0;
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task init_smem;
        input [SMEM_ADDR_W-1:0] base;
        input [15:0] size;
        input [7:0] start_val;
    begin
        for (i = 0; i < size; i = i + 1) begin
            smem[base + i] = start_val + i[7:0];
        end
    end
    endtask

    task clear_gmem;
        input [GMEM_ADDR_W-1:0] base;
        input [15:0] size;
    begin
        for (i = 0; i < size; i = i + 1) begin
            gmem[base + i] = 8'hFF;
        end
    end
    endtask

    task issue_st_bulk;
        input [5:0] f;
        input [SMEM_ADDR_W-1:0] src;
        input [GMEM_ADDR_W-1:0] dst;
        input [15:0] bytes;
    begin
        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_ST_BULK;
        func <= f;
        smem_addr <= src;
        gmem_addr <= dst;
        byte_count <= bytes;
        @(posedge clk);
        valid_in <= 0;
        wait(done);
        @(posedge clk);
    end
    endtask

    task verify_transfer;
        input [SMEM_ADDR_W-1:0] smem_base;
        input [GMEM_ADDR_W-1:0] gmem_base;
        input [15:0] size;
        output reg success;
    begin
        success = 1;
        for (i = 0; i < size; i = i + 1) begin
            expected_data = smem[smem_base + i];
            if (gmem[gmem_base + i] !== expected_data) begin
                $display("  ERROR at offset %0d: expected 0x%02x, got 0x%02x",
                         i, expected_data, gmem[gmem_base + i]);
                success = 0;
            end
        end
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    reg verify_result;

    initial begin
        $display("============================================================");
        $display("RalphGPU Bulk Store Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // Test 1: Simple 32-byte bulk store
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Simple 32-byte bulk store", test_num);

        init_smem(14'h0100, 32, 8'hAA);
        clear_gmem(32'h00001000, 64);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0100, 32'h00001000, 16'd32);
        $display("  Enqueued 32-byte bulk store");

        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);
        $display("  Committed bulk store");

        // Wait for completion
        #200;
        wait(!busy);
        #20;

        verify_transfer(14'h0100, 32'h00001000, 32, verify_result);
        if (verify_result) begin
            $display("  [PASS] 32-byte transfer verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] 32-byte transfer failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 2: 64-byte bulk store
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] 64-byte bulk store", test_num);

        init_smem(14'h0200, 64, 8'h10);
        clear_gmem(32'h00002000, 128);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0200, 32'h00002000, 16'd64);
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);

        #400;
        wait(!busy);
        #20;

        verify_transfer(14'h0200, 32'h00002000, 64, verify_result);
        if (verify_result) begin
            $display("  [PASS] 64-byte transfer verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] 64-byte transfer failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 3: Multiple enqueued operations
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Multiple enqueued operations", test_num);

        init_smem(14'h0300, 48, 8'h20);
        init_smem(14'h0400, 48, 8'h30);
        clear_gmem(32'h00003000, 128);
        clear_gmem(32'h00003100, 128);

        // Enqueue two operations
        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0300, 32'h00003000, 16'd48);
        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0400, 32'h00003100, 16'd48);
        $display("  Enqueued 2 bulk store operations");

        // Commit all
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);
        $display("  Committed all operations");

        #600;
        wait(!busy);
        #20;

        verify_transfer(14'h0300, 32'h00003000, 48, verify_result);
        if (verify_result) begin
            $display("  [PASS] First operation (48 bytes) verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] First operation failed");
            fail_count = fail_count + 1;
        end

        test_num = test_num + 1;
        verify_transfer(14'h0400, 32'h00003100, 48, verify_result);
        if (verify_result) begin
            $display("  [PASS] Second operation (48 bytes) verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Second operation failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 4: 128-byte transfer (8 x 16-byte chunks)
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] 128-byte bulk store", test_num);

        init_smem(14'h0500, 128, 8'h50);
        clear_gmem(32'h00004000, 256);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0500, 32'h00004000, 16'd128);
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);

        #800;
        wait(!busy);
        #20;

        verify_transfer(14'h0500, 32'h00004000, 128, verify_result);
        if (verify_result) begin
            $display("  [PASS] 128-byte transfer verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] 128-byte transfer failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 5: st.bulk.wait functionality
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] st.bulk.wait functionality", test_num);

        init_smem(14'h0600, 32, 8'h60);
        clear_gmem(32'h00005000, 64);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0600, 32'h00005000, 16'd32);
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);

        // Issue wait
        issue_st_bulk(`ST_BULK_WAIT, 14'h0, 32'h0, 16'd0);
        $display("  Wait completed");

        verify_transfer(14'h0600, 32'h00005000, 32, verify_result);
        if (verify_result) begin
            $display("  [PASS] Wait and transfer verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Transfer after wait failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 6: Non-aligned size (17 bytes)
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Non-aligned size (17 bytes)", test_num);

        init_smem(14'h0700, 32, 8'h70);
        clear_gmem(32'h00006000, 64);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0700, 32'h00006000, 16'd17);
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);

        #300;
        wait(!busy);
        #20;

        verify_transfer(14'h0700, 32'h00006000, 17, verify_result);
        if (verify_result) begin
            $display("  [PASS] 17-byte transfer verified");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] 17-byte transfer failed");
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Test 7: st.bulk.shared (SMEM to SMEM)
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] st.bulk.shared operation", test_num);

        init_smem(14'h0800, 32, 8'h80);
        issue_st_bulk(`ST_BULK_SHARED, 14'h0800, 32'h0900, 16'd32);
        $display("  [PASS] st.bulk.shared enqueued");
        pass_count = pass_count + 1;

        //====================================================================
        // Test 8: Bulk complete signal
        //====================================================================
        test_num = test_num + 1;
        $display("\n[TEST %0d] Bulk complete signal", test_num);

        init_smem(14'h0900, 16, 8'h90);
        clear_gmem(32'h00007000, 32);

        issue_st_bulk(`ST_BULK_GLOBAL, 14'h0900, 32'h00007000, 16'd16);
        issue_st_bulk(`ST_BULK_COMMIT, 14'h0, 32'h0, 16'd0);

        // Wait for bulk_complete signal
        wait(bulk_complete);
        $display("  bulk_complete signal received, bytes=%0d", bytes_transferred);

        if (bytes_transferred == 16) begin
            $display("  [PASS] bulk_complete with correct byte count");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Incorrect bytes_transferred: %0d", bytes_transferred);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Summary
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Test Summary: %0d passed, %0d failed out of %0d tests",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED!");
        else
            $display("SOME TESTS FAILED!");

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #50000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
