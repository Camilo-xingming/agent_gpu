//============================================================================
// Testbench: FP4/FP8 Tensor Core e2e Verification
// Issue #128: Comprehensive functional verification of FP4 (E2M1/E3M0)
// and FP8 (E4M3/E5M2) tensor core data paths
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_fp4_fp8;

    localparam CLK_PERIOD = 10;
    localparam NUM_LANES = 1;
    localparam DATA_WIDTH = 32;

    reg clk, rst_n;
    reg op_valid;
    wire op_ready;
    reg [3:0] op_type;
    reg [31:0] frag_a, frag_b, frag_c;
    wire [31:0] result_data;
    wire result_valid;
    reg result_ready;

    tensor_core #(
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TC_NUM_CORES(1),
        .TC_LATENCY(2),
        .TC_DATA_DEFAULT(`TC_DATA_FP16),
        .TC_USE_OP_TYPE(1),
        .TC_FP4_FORMAT(`TC_FP4_E2M1),
        .TC_FP8_FORMAT(`TC_FP8_E4M3)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .op_valid(op_valid),
        .op_ready(op_ready),
        .op_type(op_type),
        .frag_a(frag_a),
        .frag_b(frag_b),
        .frag_c(frag_c),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_data(result_data)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;
    integer timeout_cnt;

    //------------------------------------------------------------------------
    // Issue op and wait for result
    //------------------------------------------------------------------------
    task issue_and_wait;
        input [3:0]  dtype;
        input [31:0] a_val, b_val, c_val;
        input integer max_wait;
        begin
            // Wait for ready
            timeout_cnt = 0;
            while (!op_ready && timeout_cnt < max_wait) begin
                @(posedge clk);
                timeout_cnt = timeout_cnt + 1;
            end

            op_type = dtype;
            frag_a = a_val;
            frag_b = b_val;
            frag_c = c_val;
            @(posedge clk);
            op_valid = 1'b1;
            @(posedge clk);
            op_valid = 1'b0;

            timeout_cnt = 0;
            while (!result_valid && timeout_cnt < max_wait) begin
                @(posedge clk);
                timeout_cnt = timeout_cnt + 1;
            end
            #1;
        end
    endtask

    //------------------------------------------------------------------------
    // Check result (exact match)
    //------------------------------------------------------------------------
    task check_exact;
        input [31:0] expected;
        input [8*64-1:0] label;
        begin
            test_num = test_num + 1;
            if (!result_valid) begin
                $display("FAIL test %0d: %0s — timeout", test_num, label);
                fail_count = fail_count + 1;
            end else if (result_data === expected) begin
                $display("PASS test %0d: %0s", test_num, label);
                pass_count = pass_count + 1;
                // Consume result
                result_ready = 1'b1;
                @(posedge clk);
                result_ready = 1'b0;
            end else begin
                $display("FAIL test %0d: %0s — got 0x%08h exp 0x%08h", test_num, label, result_data, expected);
                fail_count = fail_count + 1;
                result_ready = 1'b1;
                @(posedge clk);
                result_ready = 1'b0;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Check result with tolerance (for FP rounding differences)
    //------------------------------------------------------------------------
    task check_fp32_tol;
        input [31:0] expected;
        input [31:0] tolerance_ulp;
        input [8*64-1:0] label;
        reg [31:0] diff;
        begin
            test_num = test_num + 1;
            if (!result_valid) begin
                $display("FAIL test %0d: %0s — timeout", test_num, label);
                fail_count = fail_count + 1;
            end else begin
                // Simple absolute difference on bit pattern (for same-sign, close values)
                if (result_data > expected)
                    diff = result_data - expected;
                else
                    diff = expected - result_data;

                if (diff <= tolerance_ulp || result_data === expected) begin
                    $display("PASS test %0d: %0s (got 0x%08h)", test_num, label, result_data);
                    pass_count = pass_count + 1;
                end else begin
                    $display("FAIL test %0d: %0s — got 0x%08h exp 0x%08h (diff=%0d)", test_num, label, result_data, expected, diff);
                    fail_count = fail_count + 1;
                end
                result_ready = 1'b1;
                @(posedge clk);
                result_ready = 1'b0;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_tensor_fp4_fp8.vcd");
        $dumpvars(0, tb_tensor_fp4_fp8);

        rst_n = 0;
        op_valid = 0;
        op_type = 0;
        frag_a = 0;
        frag_b = 0;
        frag_c = 0;
        result_ready = 0;  // Low by default; check tasks pulse it to consume
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);


        //====================================================================
        // Section 1: FP4 E2M1 Tests
        //====================================================================
        $display("\n=== Section 1: FP4 E2M1 Tests ===");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "FP4 E2M1: all 1.0, dot8=8.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h33333333, 32'h33333333, 32'h00000000, 50);
        check_fp32_tol(32'h41900000, 32'd16, "FP4 E2M1: all 1.5, dot8=18.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h44444444, 32'h44444444, 32'h00000000, 50);
        check_fp32_tol(32'h42000000, 32'd16, "FP4 E2M1: all 2.0, dot8=32.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h55555555, 32'h55555555, 32'h00000000, 50);
        check_fp32_tol(32'h42900000, 32'd16, "FP4 E2M1: all 3.0, dot8=72.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h44444444, 32'h00000000, 50);
        check_fp32_tol(32'h41800000, 32'd16, "FP4 E2M1: A=1.0 B=2.0, dot8=16.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h41200000, 50);
        check_fp32_tol(32'h41900000, 32'd16, "FP4 E2M1: dot8+C=10.0 -> 18.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h00000000, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h00000000, 32'd16, "FP4 E2M1: A=0, dot8=0.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h00000000, 32'h00000000, 50);
        check_fp32_tol(32'h00000000, 32'd16, "FP4 E2M1: B=0, dot8=0.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'hAAAAAAAA, 32'hAAAAAAAA, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "FP4 E2M1: all -1.0, dot8=8.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'hAAAAAAAA, 32'h00000000, 50);
        check_fp32_tol(32'hc1000000, 32'd16, "FP4 E2M1: +1*-1, dot8=-8.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'hAAAAAAAA, 32'h41a00000, 50);
        check_fp32_tol(32'h41400000, 32'd16, "FP4 E2M1: dot8=-8+C=20 -> 12.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h11111111, 32'h11111111, 32'h00000000, 50);
        check_fp32_tol(32'h40000000, 32'd16, "FP4 E2M1: denorm (0.5) 0x1*0x1 x8 = 2.0");

        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h23452345, 32'h54325432, 32'h00000000, 50);
        check_fp32_tol(32'h41c00000, 32'd16, "FP4 E2M1: mixed pattern");


        //====================================================================
        // Section 2: FP4 E3M0 Tests
        //====================================================================
        $display("\n=== Section 2: FP4 E3M0 Tests ===");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h22222222, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h40000000, 32'd16, "FP4 E3M0: all 0.5, dot8=2.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h33333333, 32'h33333333, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "FP4 E3M0: all 1.0, dot8=8.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h44444444, 32'h44444444, 32'h00000000, 50);
        check_fp32_tol(32'h42000000, 32'd16, "FP4 E3M0: all 2.0, dot8=32.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h33333333, 32'h33333333, 32'h40a00000, 50);
        check_fp32_tol(32'h41500000, 32'd16, "FP4 E3M0: dot8+C=5 -> 13.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h00000000, 32'h33333333, 32'h00000000, 50);
        check_fp32_tol(32'h00000000, 32'd16, "FP4 E3M0: A=0, dot8=0.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'hBBBBBBBB, 32'hBBBBBBBB, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "FP4 E3M0: all -1.0, dot8=8.0");


        //====================================================================
        // Section 3: FP8 E4M3 Tests
        //====================================================================
        $display("\n=== Section 3: FP8 E4M3 Tests ===");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h38383838, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "FP8 E4M3: all 1.0, dot4=4.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h40404040, 32'h40404040, 32'h00000000, 50);
        check_fp32_tol(32'h41800000, 32'd16, "FP8 E4M3: all 2.0, dot4=16.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h3C3C3C3C, 32'h3C3C3C3C, 32'h00000000, 50);
        check_fp32_tol(32'h41100000, 32'd16, "FP8 E4M3: all 1.5, dot4=9.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h40404040, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "FP8 E4M3: A=1 B=2, dot4=8.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h38383838, 32'h42c80000, 50);
        check_fp32_tol(32'h42d00000, 32'd16, "FP8 E4M3: dot4+C=100 -> 104.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h00000000, 32'h38383838, 32'h00000000, 50);
        check_fp32_tol(32'h00000000, 32'd16, "FP8 E4M3: A=0, dot4=0.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'hB8B8B8B8, 32'hB8B8B8B8, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "FP8 E4M3: all -1.0, dot4=4.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'hB8B8B8B8, 32'h00000000, 50);
        check_fp32_tol(32'hc0800000, 32'd16, "FP8 E4M3: +1*-1, dot4=-4.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h08080808, 32'h08080808, 32'h00000000, 50);
        check_fp32_tol(32'h3a800000, 32'd16, "FP8 E4M3: small vals 0x08");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h70707070, 32'h38383838, 32'h00000000, 50);
        check_fp32_tol(32'h44000000, 32'd16, "FP8 E4M3: large vals 0x70");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38403C44, 32'h40383C38, 32'h00000000, 50);
        check_fp32_tol(32'h41140000, 32'd16, "FP8 E4M3: mixed pattern");


        //====================================================================
        // Section 4: FP8 E5M2 Tests
        //====================================================================
        $display("\n=== Section 4: FP8 E5M2 Tests ===");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h3C3C3C3C, 32'h3C3C3C3C, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "FP8 E5M2: all 1.0, dot4=4.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h40404040, 32'h40404040, 32'h00000000, 50);
        check_fp32_tol(32'h41800000, 32'd16, "FP8 E5M2: all 2.0, dot4=16.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h3E3E3E3E, 32'h3E3E3E3E, 32'h00000000, 50);
        check_fp32_tol(32'h41100000, 32'd16, "FP8 E5M2: all 1.5, dot4=9.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h3C3C3C3C, 32'h3C3C3C3C, 32'h42480000, 50);
        check_fp32_tol(32'h42580000, 32'd16, "FP8 E5M2: dot4+C=50 -> 54.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'hBCBCBCBC, 32'hBCBCBCBC, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "FP8 E5M2: all -1.0, dot4=4.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h00000000, 32'h3C3C3C3C, 32'h00000000, 50);
        check_fp32_tol(32'h00000000, 32'd16, "FP8 E5M2: A=0, dot4=0.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h3C403E44, 32'h403C3E3C, 32'h00000000, 50);
        check_fp32_tol(32'h41240000, 32'd16, "FP8 E5M2: mixed pattern");


        //====================================================================
        // Section 5: Back-to-Back Pipeline Tests
        //====================================================================
        $display("\n=== Section 5: Back-to-Back Pipeline ===");

        // Rapid FP4 -> FP8 -> FP4 switching
        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "B2B #1: FP4 E2M1 dot8=8.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h38383838, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "B2B #2: FP8 E4M3 dot4=4.0");

        issue_and_wait(`TC_DATA_FP4_E3M0, 32'h33333333, 32'h33333333, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "B2B #3: FP4 E3M0 dot8=8.0");

        issue_and_wait(`TC_DATA_FP8_E5M2, 32'h3C3C3C3C, 32'h3C3C3C3C, 32'h00000000, 50);
        check_fp32_tol(32'h40800000, 32'd16, "B2B #4: FP8 E5M2 dot4=4.0");

        // Rapid same-type sequence
        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h40404040, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "B2B #5: FP8 E4M3 1*2 dot4=8.0");

        issue_and_wait(`TC_DATA_FP8_E4M3, 32'h40404040, 32'h40404040, 32'h00000000, 50);
        check_fp32_tol(32'h41800000, 32'd16, "B2B #6: FP8 E4M3 2*2 dot4=16.0");

        //====================================================================
        // Section 6: Reset Recovery
        //====================================================================
        $display("\n=== Section 6: Reset Recovery ===");

        // Issue op then reset mid-flight
        @(posedge clk);
        op_type = `TC_DATA_FP4_E2M1;
        frag_a = 32'h55555555;
        frag_b = 32'h55555555;
        frag_c = 32'h00000000;
        op_valid = 1'b1;
        @(posedge clk);
        op_valid = 1'b0;
        rst_n = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        test_num = test_num + 1;
        if (!result_valid && op_ready) begin
            $display("PASS test %0d: Reset clears pipeline", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: Reset state: valid=%b ready=%b", test_num, result_valid, op_ready);
            fail_count = fail_count + 1;
        end

        // Normal op after reset
        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "Post-reset FP4 E2M1 dot8=8.0");

        //====================================================================
        // Section 7: Accumulator Chaining
        //====================================================================
        $display("\n=== Section 7: Accumulator Chaining ===");

        // Chain: C=0 -> result -> feed as C to next op
        // Step 1: FP4 E2M1 all 1.0 dot8 = 8.0
        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h00000000, 50);
        check_fp32_tol(32'h41000000, 32'd16, "Chain step 1: dot8=8.0");

        // Step 2: Feed 8.0 as C, dot8(1*1)+8 = 16.0
        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h41000000, 50);
        check_fp32_tol(32'h41800000, 32'd16, "Chain step 2: dot8+8=16.0");

        // Step 3: Feed 16.0 as C, dot8(1*1)+16 = 24.0
        issue_and_wait(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h41800000, 50);
        check_fp32_tol(32'h41C00000, 32'd16, "Chain step 3: dot8+16=24.0");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n========================================");
        $display("FP4/FP8 Tensor Core Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED (%0d failures)", fail_count);
        $display("========================================");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #500000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
