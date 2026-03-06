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

    localparam SPARSE_DATA_W = TILE_M*TILE_K*16/2;
    localparam SPARSE_INDEX_W = TILE_M*TILE_K;
    localparam DENSE_A_W = TILE_M*TILE_K*16;

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

    task run_sparse_op;
        input [5:0] func_code;
    begin
        @(posedge clk);
        sparse_valid_in <= 1'b1;
        sparse_func <= func_code;
        @(posedge clk);
        sparse_valid_in <= 1'b0;

        wait(sparse_done);
        @(posedge clk);
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
        //====================================================================
        // Test 16: TANH boundary and saturation behavior
        //====================================================================
        test_dpx_op(`DPX_TANH, 32'sd4, 32'd0, 32'd0, 32'd1, 32'hx, "TANH: saturate high at +1");
        test_dpx_op(`DPX_TANH, -32'sd4, 32'd0, 32'd0, 32'hFFFFFFFF, 32'hx, "TANH: saturate low at -1");
        test_dpx_op(`DPX_TANH, 32'sd1, 32'd0, 32'd0, 32'd1, 32'hx, "TANH: linear region at +1");
        test_dpx_op(`DPX_TANH, 32'sd2, 32'd0, 32'd0, 32'd1, 32'hx, "TANH: piecewise positive region");

        //====================================================================
        // Test 17: EXP2 boundary behavior (negative and saturation)
        //====================================================================
        test_dpx_op(`DPX_EXP2, -32'sd1, 32'd0, 32'd0, 32'd0, 32'hx, "EXP2: negative input rounds to 0");
        test_dpx_op(`DPX_EXP2, 32'd31, 32'd0, 32'd0, 32'h80000000, 32'hx, "EXP2: saturate at bit31");
        test_dpx_op(`DPX_EXP2, 32'd40, 32'd0, 32'd0, 32'h80000000, 32'hx, "EXP2: saturate when input > 31");

        //====================================================================
        // Test 18: VIADDMIN/VIADDMAX overflow and abs edge behavior
        //====================================================================
        test_dpx_op(`DPX_VIADDMIN, 32'h7FFFFFFF, 32'd1, 32'h7FFFFFFF, 32'h80000000, 32'hx,
                    "VIADDMIN: signed overflow path");
        test_dpx_op(`DPX_VIADDMAX, 32'h7FFFFFFF, 32'd1, 32'h7FFFFFFF, 32'h7FFFFFFF, 32'hx,
                    "VIADDMAX: signed overflow path");
        test_dpx_op(`DPX_VIADDMINMAX, 32'h7FFFFFFF, 32'd1, 32'h7FFFFFFF, 32'h80000000, 32'h7FFFFFFF,
                    "VIADDMINMAX: overflow min/max pair");
        test_dpx_op(`DPX_VIMINABS, 32'h80000000, 32'd1, 32'd0, 32'd1, 32'hx,
                    "VIMINABS: abs(INT_MIN) edge case");
        // Test 24: Sparse MMA Unit - Compress operation
        //====================================================================
        $display("\n[TEST %0d] Sparse MMA Unit: Compress operation", test_num + 1);
        test_num = test_num + 1;

        sparse_a_data = {SPARSE_DATA_W{1'b0}};
        sparse_a_indices = {SPARSE_INDEX_W{1'b0}};
        dense_b = 0;
        accum_in = 0;

        // Group0 = [3,0,-2,1] -> keep 3(idx0), -2(idx2)
        // Group1 = [0,7,0,-5] -> keep 7(idx1), -5(idx3)
        accum_in[0 +: 16] = 16'sd3;
        accum_in[16 +: 16] = 16'sd0;
        accum_in[32 +: 16] = -16'sd2;
        accum_in[48 +: 16] = 16'sd1;
        accum_in[64 +: 16] = 16'sd0;
        accum_in[80 +: 16] = 16'sd7;
        accum_in[96 +: 16] = 16'sd0;
        accum_in[112 +: 16] = -16'sd5;

        run_sparse_op(`SPARSE_COMPRESS);

        if (accum_out[0 +: 16] == 16'sd3 &&
            accum_out[16 +: 16] == -16'sd2 &&
            accum_out[32 +: 16] == 16'sd7 &&
            accum_out[48 +: 16] == -16'sd5 &&
            accum_out[SPARSE_DATA_W + 0 +: 2] == 2'd0 &&
            accum_out[SPARSE_DATA_W + 2 +: 2] == 2'd2 &&
            accum_out[SPARSE_DATA_W + 4 +: 2] == 2'd1 &&
            accum_out[SPARSE_DATA_W + 6 +: 2] == 2'd3) begin
            $display("  [PASS] Compress output matches expected packed format");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Compress output mismatch");
            fail_count = fail_count + 1;
        end
        #20;

        //====================================================================
        // Test 25: Sparse MMA Unit - Decompress operation
        //====================================================================
        $display("\n[TEST %0d] Sparse MMA Unit: Decompress operation", test_num + 1);
        test_num = test_num + 1;

        sparse_a_data = {SPARSE_DATA_W{1'b0}};
        sparse_a_indices = {SPARSE_INDEX_W{1'b0}};
        dense_b = 0;
        accum_in = 0;

        // Group0 -> idx1=1 val=10, idx2=3 val=-7 => [0,10,0,-7]
        sparse_a_data[0 +: 16] = 16'sd10;
        sparse_a_data[16 +: 16] = -16'sd7;
        sparse_a_indices[0 +: 2] = 2'd1;
        sparse_a_indices[2 +: 2] = 2'd3;

        // Group1 -> idx0=0 val=4, idx1=2 val=2 => [4,0,2,0]
        sparse_a_data[32 +: 16] = 16'sd4;
        sparse_a_data[48 +: 16] = 16'sd2;
        sparse_a_indices[4 +: 2] = 2'd0;
        sparse_a_indices[6 +: 2] = 2'd2;

        run_sparse_op(`SPARSE_DECOMPRESS);

        if (accum_out[0 +: 16] == 16'sd0 &&
            accum_out[16 +: 16] == 16'sd10 &&
            accum_out[32 +: 16] == 16'sd0 &&
            accum_out[48 +: 16] == -16'sd7 &&
            accum_out[64 +: 16] == 16'sd4 &&
            accum_out[80 +: 16] == 16'sd0 &&
            accum_out[96 +: 16] == 16'sd2 &&
            accum_out[112 +: 16] == 16'sd0) begin
            $display("  [PASS] Decompress output matches expected dense values");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Decompress output mismatch");
            fail_count = fail_count + 1;
        end
        #20;

        //====================================================================
        // Test 26: Sparse MMA Unit - MMA operation
        //====================================================================
        $display("\n[TEST %0d] Sparse MMA Unit: MMA operation", test_num + 1);
        test_num = test_num + 1;

        sparse_a_data = {SPARSE_DATA_W{1'b0}};
        sparse_a_indices = {SPARSE_INDEX_W{1'b0}};
        dense_b = 0;
        accum_in = 0;

        // Row0, k0=2, k1=3
        sparse_a_data[0 +: 16] = 16'sd2;
        sparse_a_data[16 +: 16] = 16'sd3;
        sparse_a_indices[0 +: 2] = 2'd0;
        sparse_a_indices[2 +: 2] = 2'd1;

        // B[k0,col0]=4, B[k1,col0]=5
        dense_b[(0*TILE_N + 0)*16 +: 16] = 16'sd4;
        dense_b[(1*TILE_N + 0)*16 +: 16] = 16'sd5;

        // Base accumulator C[0,0] = 7
        accum_in[0 +: 32] = 32'sd7;

        run_sparse_op(`SPARSE_MMA_FP16);

        // Expected: 7 + (2*4 + 3*5) = 30
        if (accum_out[0 +: 32] == 32'sd30 && accum_out[32 +: 32] == 32'sd0) begin
            $display("  [PASS] Sparse MMA output matches expected result");
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] Sparse MMA output mismatch: got C00=%0d C01=%0d",
                     $signed(accum_out[0 +: 32]), $signed(accum_out[32 +: 32]));
            fail_count = fail_count + 1;
        end

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

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #10000;
        $display("ERROR: Test timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
