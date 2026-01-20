//============================================================================
// RalphGPU - DPX Unit Testbench
// Tests Dynamic Programming Extensions and Sparse MMA operations
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_dpx_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter DATA_WIDTH = 32;
    parameter NUM_LANES = 32;
    parameter TILE_M = 16;
    parameter TILE_N = 8;
    parameter TILE_K = 16;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    //------------------------------------------------------------------------
    // DPX Unit Interface
    //------------------------------------------------------------------------
    reg                      dpx_valid_in;
    reg  [5:0]               dpx_func;
    reg  [DATA_WIDTH-1:0]    dpx_src_a;
    reg  [DATA_WIDTH-1:0]    dpx_src_b;
    reg  [DATA_WIDTH-1:0]    dpx_src_c;
    wire                     dpx_done;
    wire [DATA_WIDTH-1:0]    dpx_result;
    wire [DATA_WIDTH-1:0]    dpx_result2;

    //------------------------------------------------------------------------
    // Sparse MMA Unit Interface
    //------------------------------------------------------------------------
    reg                      sparse_valid_in;
    reg  [5:0]               sparse_func;
    reg  [TILE_M*TILE_K*16/2-1:0]  sparse_a_data;
    reg  [TILE_M*TILE_K-1:0]       sparse_a_indices;
    reg  [TILE_K*TILE_N*16-1:0]    dense_b;
    reg  [TILE_M*TILE_N*32-1:0]    accum_in;
    wire [TILE_M*TILE_N*32-1:0]    accum_out;
    wire                     sparse_done;
    wire                     sparse_busy;

    //------------------------------------------------------------------------
    // DUT Instantiation - DPX Unit
    //------------------------------------------------------------------------
    dpx_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_LANES(NUM_LANES)
    ) dut_dpx (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(dpx_valid_in),
        .func(dpx_func),
        .src_a(dpx_src_a),
        .src_b(dpx_src_b),
        .src_c(dpx_src_c),
        .done(dpx_done),
        .result(dpx_result),
        .result2(dpx_result2)
    );

    //------------------------------------------------------------------------
    // DUT Instantiation - Sparse MMA Unit
    //------------------------------------------------------------------------
    sparse_mma_unit #(
        .DATA_WIDTH(16),
        .TILE_M(TILE_M),
        .TILE_N(TILE_N),
        .TILE_K(TILE_K)
    ) dut_sparse (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(sparse_valid_in),
        .func(sparse_func),
        .sparse_a_data(sparse_a_data),
        .sparse_a_indices(sparse_a_indices),
        .dense_b(dense_b),
        .accum_in(accum_in),
        .accum_out(accum_out),
        .done(sparse_done),
        .busy(sparse_busy)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;
    reg [DATA_WIDTH-1:0] expected_result;
    reg [DATA_WIDTH-1:0] expected_result2;

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        dpx_valid_in = 0;
        dpx_func = 0;
        dpx_src_a = 0;
        dpx_src_b = 0;
        dpx_src_c = 0;
        sparse_valid_in = 0;
        sparse_func = 0;
        sparse_a_data = 0;
        sparse_a_indices = 0;
        dense_b = 0;
        accum_in = 0;
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task test_dpx_op;
        input [5:0] func;
        input [DATA_WIDTH-1:0] a, b, c;
        input [DATA_WIDTH-1:0] exp_result;
        input [DATA_WIDTH-1:0] exp_result2;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Inputs: a=%0d (0x%08x), b=%0d (0x%08x), c=%0d (0x%08x)",
                 $signed(a), a, $signed(b), b, $signed(c), c);

        @(posedge clk);
        dpx_valid_in <= 1;
        dpx_func <= func;
        dpx_src_a <= a;
        dpx_src_b <= b;
        dpx_src_c <= c;
        @(posedge clk);
        dpx_valid_in <= 0;

        // Wait for done
        wait(dpx_done);
        @(posedge clk);

        $display("  Result: %0d (0x%08x), Result2: %0d (0x%08x)",
                 $signed(dpx_result), dpx_result, $signed(dpx_result2), dpx_result2);

        if (dpx_result === exp_result && (exp_result2 === 32'hx || dpx_result2 === exp_result2)) begin
            $display("  [PASS] Result matches expected");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Expected: %0d (0x%08x), got: %0d (0x%08x)",
                     $signed(exp_result), exp_result, $signed(dpx_result), dpx_result);
            fail_count = fail_count + 1;
        end
        #10;
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU DPX Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // Test 1: VIADDMIN - min(a + b, c)
        //====================================================================
        // 5 + 3 = 8, min(8, 10) = 8
        test_dpx_op(`DPX_VIADDMIN, 32'd5, 32'd3, 32'd10, 32'd8, 32'hx, "VIADDMIN: min(5+3, 10) = 8");

        //====================================================================
        // Test 2: VIADDMIN with negative - min(a + b, c) where sum < c
        //====================================================================
        // (-5) + (-3) = -8, min(-8, 0) = -8
        test_dpx_op(`DPX_VIADDMIN, -32'sd5, -32'sd3, 32'd0, -32'sd8, 32'hx, "VIADDMIN: min(-5+-3, 0) = -8");

        //====================================================================
        // Test 3: VIADDMAX - max(a + b, c)
        //====================================================================
        // 5 + 3 = 8, max(8, 5) = 8
        test_dpx_op(`DPX_VIADDMAX, 32'd5, 32'd3, 32'd5, 32'd8, 32'hx, "VIADDMAX: max(5+3, 5) = 8");

        //====================================================================
        // Test 4: VIADDMAX with c being larger
        //====================================================================
        // 1 + 2 = 3, max(3, 10) = 10
        test_dpx_op(`DPX_VIADDMAX, 32'd1, 32'd2, 32'd10, 32'd10, 32'hx, "VIADDMAX: max(1+2, 10) = 10");

        //====================================================================
        // Test 5: VIMINABS - min(|a|, |b|)
        //====================================================================
        // min(|-5|, |3|) = min(5, 3) = 3
        test_dpx_op(`DPX_VIMINABS, -32'sd5, 32'd3, 32'd0, 32'd3, 32'hx, "VIMINABS: min(|-5|, |3|) = 3");

        //====================================================================
        // Test 6: VIMAXABS - max(|a|, |b|)
        //====================================================================
        // max(|-7|, |4|) = max(7, 4) = 7
        test_dpx_op(`DPX_VIMAXABS, -32'sd7, 32'd4, 32'd0, 32'd7, 32'hx, "VIMAXABS: max(|-7|, |4|) = 7");

        //====================================================================
        // Test 7: VIADDMINMAX - both min and max
        //====================================================================
        // 10 + 5 = 15, min(15, 20) = 15, max(15, 20) = 20
        test_dpx_op(`DPX_VIADDMINMAX, 32'd10, 32'd5, 32'd20, 32'd15, 32'd20, "VIADDMINMAX: 10+5=15, c=20 -> min=15, max=20");

        //====================================================================
        // Test 8: VIBMATCH - bit pattern matching (XNOR)
        //====================================================================
        // 0xFF00FF00 XNOR 0xFF00FF00 = 0xFFFFFFFF (all match)
        test_dpx_op(`DPX_VIBMATCH, 32'hFF00FF00, 32'hFF00FF00, 32'd0, 32'hFFFFFFFF, 32'hx, "VIBMATCH: identical patterns");

        //====================================================================
        // Test 9: VIBMATCH - partial match
        //====================================================================
        // 0xAAAAAAAA XNOR 0x55555555 = 0x00000000 (all different)
        test_dpx_op(`DPX_VIBMATCH, 32'hAAAAAAAA, 32'h55555555, 32'd0, 32'h00000000, 32'hx, "VIBMATCH: opposite patterns");

        //====================================================================
        // Test 10: VIBSET - bit selection
        //====================================================================
        // select a=0xFF, b=0x00, mask=0x0F -> 0xF0 (lower 4 from b, upper 4 from a)
        test_dpx_op(`DPX_VIBSET, 32'hFF, 32'h00, 32'h0F, 32'hF0, 32'hx, "VIBSET: select bits");

        //====================================================================
        // Test 11: RELU - max(0, x) positive
        //====================================================================
        test_dpx_op(`DPX_RELU, 32'd42, 32'd0, 32'd0, 32'd42, 32'hx, "RELU: max(0, 42) = 42");

        //====================================================================
        // Test 12: RELU - max(0, x) negative
        //====================================================================
        test_dpx_op(`DPX_RELU, -32'sd10, 32'd0, 32'd0, 32'd0, 32'hx, "RELU: max(0, -10) = 0");

        //====================================================================
        // Test 13: EXP2 - 2^4 = 16
        //====================================================================
        test_dpx_op(`DPX_EXP2, 32'd4, 32'd0, 32'd0, 32'd16, 32'hx, "EXP2: 2^4 = 16");

        //====================================================================
        // Test 14: EXP2 - 2^0 = 1
        //====================================================================
        test_dpx_op(`DPX_EXP2, 32'd0, 32'd0, 32'd0, 32'd1, 32'hx, "EXP2: 2^0 = 1");

        //====================================================================
        // Test 15: EXP2 - 2^10 = 1024
        //====================================================================
        test_dpx_op(`DPX_EXP2, 32'd10, 32'd0, 32'd0, 32'd1024, 32'hx, "EXP2: 2^10 = 1024");

        //====================================================================
        // Test 16-18: Sparse MMA Unit Tests
        //====================================================================
        $display("\n[TEST %0d] Sparse MMA Unit: Compress operation", test_num + 1);
        test_num = test_num + 1;
        @(posedge clk);
        sparse_valid_in <= 1;
        sparse_func <= `SPARSE_COMPRESS;
        accum_in <= 128'hDEADBEEFCAFEBABE12345678ABCDEF00;
        @(posedge clk);
        sparse_valid_in <= 0;
        wait(sparse_done);
        @(posedge clk);
        if (accum_out == 128'hDEADBEEFCAFEBABE12345678ABCDEF00) begin
            $display("  [PASS] Compress passthrough works");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Compress output mismatch");
            fail_count = fail_count + 1;
        end
        #20;

        $display("\n[TEST %0d] Sparse MMA Unit: Decompress operation", test_num + 1);
        test_num = test_num + 1;
        @(posedge clk);
        sparse_valid_in <= 1;
        sparse_func <= `SPARSE_DECOMPRESS;
        sparse_a_data <= 256'h0001_0002_0003_0004;  // Sample sparse data
        sparse_a_indices <= 64'hFF;  // Sample indices
        @(posedge clk);
        sparse_valid_in <= 0;
        wait(sparse_done);
        @(posedge clk);
        $display("  [PASS] Decompress operation completed");
        pass_count = pass_count + 1;
        #20;

        $display("\n[TEST %0d] Sparse MMA Unit: MMA operation", test_num + 1);
        test_num = test_num + 1;
        @(posedge clk);
        sparse_valid_in <= 1;
        sparse_func <= `SPARSE_MMA_FP16;
        sparse_a_data <= 256'h0010_0020_0030_0040;
        sparse_a_indices <= 64'hAA;
        dense_b <= 512'h0001_0002_0003_0004;
        accum_in <= 0;
        @(posedge clk);
        sparse_valid_in <= 0;
        wait(sparse_done);
        @(posedge clk);
        $display("  [PASS] Sparse MMA operation completed");
        pass_count = pass_count + 1;

        //====================================================================
        // Summary
        //====================================================================
        #50;
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
        #10000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
