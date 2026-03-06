//============================================================================
// Testbench: tensor_core — FP4/FP8/FP16/BF16/INT8/INT4 e2e verification
// Gemini #1: FP4/FP8 Tensor Core e2e verification
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_core_e2e;

    localparam NUM_LANES = 1;
    localparam DATA_WIDTH = 32;
    localparam TC_LATENCY = 2;
    localparam TC_NUM_CORES = 1;

    reg  clk, rst_n;
    reg  op_valid;
    wire op_ready;
    reg  [3:0] op_type;
    reg  [NUM_LANES*DATA_WIDTH-1:0] frag_a, frag_b, frag_c;
    wire result_valid;
    reg  result_ready;
    wire [NUM_LANES*DATA_WIDTH-1:0] result_data;

    tensor_core #(
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .TC_NUM_CORES(TC_NUM_CORES),
        .TC_LATENCY(TC_LATENCY),
        .TC_USE_OP_TYPE(1)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .op_valid(op_valid), .op_ready(op_ready),
        .op_type(op_type),
        .frag_a(frag_a), .frag_b(frag_b), .frag_c(frag_c),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_data(result_data)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    //----------------------------------------------------------------------
    // Task: submit one MMA op and check result
    // Drives op_valid, waits for result_valid, checks lane 0
    //----------------------------------------------------------------------
    task check_mma;
        input [3:0]   t_type;
        input [31:0]  t_a, t_b, t_c;
        input [31:0]  expected;
        input [255:0] name;
        integer timeout;
        begin
            test_num = test_num + 1;
            // Wait for ready
            @(posedge clk);
            while (!op_ready) @(posedge clk);

            // Submit operation
            op_valid <= 1'b1;
            op_type  <= t_type;
            frag_a   <= t_a;
            frag_b   <= t_b;
            frag_c   <= t_c;
            @(posedge clk);
            op_valid <= 1'b0;

            // Wait for result (result_ready always asserted)
            timeout = 0;
            while (!result_valid && timeout < 50) begin
                @(posedge clk);
                #1;
                timeout = timeout + 1;
            end

            if (!result_valid) begin
                $display("FAIL #%0d %0s: TIMEOUT waiting for result", test_num, name);
                fail_count = fail_count + 1;
            end else if (result_data[31:0] === expected) begin
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL #%0d %0s: got %08h, expected %08h",
                         test_num, name, result_data[31:0], expected);
                fail_count = fail_count + 1;
            end
            @(posedge clk); // gap between tests
        end
    endtask

    initial begin
        rst_n = 0; op_valid = 0; result_ready = 1; // always ready to accept
        op_type = 0; frag_a = 0; frag_b = 0; frag_c = 0;
        #30;
        rst_n = 1;
        #10;

        //==================================================================
        // INT8 dot4: baseline
        // a={4,1,3,2} b={1,3,2,1} c=10
        // dot = 4*1 + 1*3 + 3*2 + 2*1 = 15, result=25
        //==================================================================
        check_mma(`TC_DATA_INT8, 32'h02030104, 32'h01020301, 32'd10,
                  32'h00000019, "int8 dot4 basic");

        // INT8: all zeros → c passthrough
        check_mma(`TC_DATA_INT8, 32'h00000000, 32'h00000000, 32'd42,
                  32'h0000002A, "int8 zero+c");

        // INT8: negative values
        // a={-1, 2, 0, 0} = {0xFF, 0x02, 0x00, 0x00}
        // b={3, 4, 0, 0} = {0x03, 0x04, 0x00, 0x00}
        // dot = (-1)*3 + 2*4 = -3+8 = 5, c=0 → 5
        check_mma(`TC_DATA_INT8, 32'h000002FF, 32'h00000403, 32'd0,
                  32'h00000005, "int8 signed");

        //==================================================================
        // INT4 dot8: 8 nibbles per word
        // All 1s: 8 * (1*1) = 8
        // FP4 value 1 in INT4 = just integer 1 sign-extended
        // a=0x11111111 → nibbles all 1
        // b=0x11111111 → nibbles all 1
        // dot8 = 8*(1*1) = 8
        //==================================================================
        check_mma(`TC_DATA_INT4, 32'h11111111, 32'h11111111, 32'd0,
                  32'h00000008, "int4 dot8 ones");

        // INT4: with negatives
        // a = {0,0,0,0, 0,0, 2, -1} nibbles = 0x0000002F (F=-1 in 4-bit signed)
        // b = {0,0,0,0, 0,0, 3, 2} nibbles = 0x00000032
        // dot = (-1)*2 + 2*3 + 0... = -2+6 = 4
        check_mma(`TC_DATA_INT4, 32'h0000002F, 32'h00000032, 32'd0,
                  32'h00000004, "int4 signed");

        //==================================================================
        // FP16 dot2: a={2.0, 1.5} b={1.0, 2.0} c=1.0
        // 2.0_fp16=0x4000, 1.5_fp16=0x3E00
        // 1.0_fp16=0x3C00, 2.0_fp16=0x4000
        // dot = 2.0*1.0 + 1.5*2.0 = 2.0+3.0 = 5.0
        // result = 5.0 + 1.0 = 6.0 = 0x40C00000
        //==================================================================
        check_mma(`TC_DATA_FP16, 32'h3E004000, 32'h40003C00, 32'h3F800000,
                  32'h40C00000, "fp16 dot2");

        // FP16: 1.0*1.0 + 1.0*1.0 + 0.0 = 2.0
        check_mma(`TC_DATA_FP16, 32'h3C003C00, 32'h3C003C00, 32'h00000000,
                  32'h40000000, "fp16 ones");

        //==================================================================
        // BF16 dot2: a={1.0_bf16, 2.0_bf16} b={2.0_bf16, 1.0_bf16} c=0
        // 1.0_bf16=0x3F80, 2.0_bf16=0x4000
        // dot = 1.0*2.0 + 2.0*1.0 = 4.0 = 0x40800000
        //==================================================================
        check_mma(`TC_DATA_BF16, 32'h40003F80, 32'h3F804000, 32'h00000000,
                  32'h40800000, "bf16 dot2");

        //==================================================================
        // FP8 E4M3 dot4
        // 1.0=0x38, 1.5=0x3C, 2.0=0x40
        // a={1.0, 2.0, 1.5, 1.0} b={1.0, 1.0, 1.0, 2.0}
        // a32=0x383C4038, b32=0x38383840
        // dot = 1.0*2.0 + 2.0*1.0 + 1.5*1.0 + 1.0*1.0 = 6.5
        // 6.5 = 0x40D00000
        //==================================================================
        check_mma(`TC_DATA_FP8_E4M3, 32'h383C4038, 32'h38383840, 32'h00000000,
                  32'h40D00000, "fp8 e4m3 dot4");

        // FP8 E4M3: all 1.0s → 4.0
        check_mma(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h38383838, 32'h00000000,
                  32'h40800000, "fp8 e4m3 ones");

        // FP8 E4M3: with accumulator
        // a={1.0,1.0,1.0,1.0} b={1.0,1.0,1.0,1.0} c=10.0(0x41200000)
        // dot=4.0, result=14.0=0x41600000
        check_mma(`TC_DATA_FP8_E4M3, 32'h38383838, 32'h38383838, 32'h41200000,
                  32'h41600000, "fp8 e4m3 +acc");

        //==================================================================
        // FP8 E5M2 dot4
        // 1.0=0x3C, 2.0=0x40
        // a={1.0, 1.0, 2.0, 1.0}, b={2.0, 1.0, 1.0, 1.0}
        // a32=0x3C3C403C, b32=0x3C3C3C40
        // dot = 1.0*2.0 + 2.0*1.0 + 1.0*1.0 + 1.0*1.0 = 6.0
        // 6.0 = 0x40C00000
        //==================================================================
        check_mma(`TC_DATA_FP8_E5M2, 32'h3C3C403C, 32'h3C3C3C40, 32'h00000000,
                  32'h40C00000, "fp8 e5m2 dot4");

        // FP8 E5M2: all 1.0s → 4.0
        check_mma(`TC_DATA_FP8_E5M2, 32'h3C3C3C3C, 32'h3C3C3C3C, 32'h00000000,
                  32'h40800000, "fp8 e5m2 ones");

        //==================================================================
        // FP4 E2M1 dot8
        // 1.0 = 0x2 (0_01_0)
        // All 1.0s: a=0x22222222, b=0x22222222
        // dot8 = 8 * 1.0 = 8.0 = 0x41000000
        //==================================================================
        check_mma(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h00000000,
                  32'h41000000, "fp4 e2m1 ones");

        // FP4 E2M1: mixed values
        // 1.5 = 0x3 (0_01_1), 2.0 = 0x4 (0_10_0)
        // a = {1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0}
        //   = nibbles: 2,2,2,2,3,3,2,2 from MSN to LSN
        //   = a32 = 0x22223322
        // b = all 1.0 = 0x22222222
        // dot = 1.0 + 1.0 + 1.5 + 1.5 + 1.0 + 1.0 + 1.0 + 1.0 = 9.0
        // 9.0 = 0x41100000
        check_mma(`TC_DATA_FP4_E2M1, 32'h22223322, 32'h22222222, 32'h00000000,
                  32'h41100000, "fp4 e2m1 mixed");

        // FP4 E2M1: with accumulator
        // all 1.0 * 1.0 = 8.0, c = 2.0 → 10.0 = 0x41200000
        check_mma(`TC_DATA_FP4_E2M1, 32'h22222222, 32'h22222222, 32'h40000000,
                  32'h41200000, "fp4 e2m1 +acc");

        //==================================================================
        // FP4 E3M0 dot8
        // E3M0: S_EEE (no mantissa), bias=3
        // 1.0 = 0_011 = 0x3 (exp=3, val = 2^(3-3) = 1.0)
        // 2.0 = 0_100 = 0x4 (exp=4, val = 2^(4-3) = 2.0)
        // All 1.0s: nibble = 0x3 → a32 = 0x33333333
        // dot8 = 8 * 1.0 = 8.0 = 0x41000000
        //==================================================================
        check_mma(`TC_DATA_FP4_E3M0, 32'h33333333, 32'h33333333, 32'h00000000,
                  32'h41000000, "fp4 e3m0 ones");

        //==================================================================
        // FP8 E4M3 small exponent tests (exp4 < 7, regression for underflow bug)
        //==================================================================

        // FP8 E4M3: 0.5 × 0.5 × 4 + 0 = 1.0
        // 0.5 in E4M3: exp=6, man=0 → 0_0110_000 = 0x30
        // a={0.5, 0.5, 0.5, 0.5} = 0x30303030
        // dot4 = 4 * 0.25 = 1.0 = 0x3F800000
        check_mma(`TC_DATA_FP8_E4M3, 32'h30303030, 32'h30303030, 32'h00000000,
                  32'h3F800000, "fp8 e4m3 0.5s");

        // FP8 E4M3: 0.25 × 1.0 × 4 + 0 = 1.0
        // 0.25 in E4M3: exp=5, man=0 → 0_0101_000 = 0x28
        // 1.0 in E4M3: exp=7, man=0 → 0_0111_000 = 0x38
        // a={0.25,0.25,0.25,0.25}=0x28282828, b={1.0,1.0,1.0,1.0}=0x38383838
        // dot4 = 4 * 0.25 = 1.0
        check_mma(`TC_DATA_FP8_E4M3, 32'h28282828, 32'h38383838, 32'h00000000,
                  32'h3F800000, "fp8 e4m3 0.25*1");

        // FP8 E4M3: mixed small exponents
        // 0.125 = 2^(-3), E4M3: exp=4, man=0 → 0_0100_000 = 0x20
        // a={0.125,0.25,0.5,1.0}=0x20283038, b={1.0,1.0,1.0,1.0}=0x38383838
        // dot4 = 0.125+0.25+0.5+1.0 = 1.875 = 0x3FF00000
        check_mma(`TC_DATA_FP8_E4M3, 32'h20283038, 32'h38383838, 32'h00000000,
                  32'h3FF00000, "fp8 e4m3 mixed exp");

        // FP8 E4M3: smallest normal × smallest normal
        // 2^(-6) in E4M3: exp=1, man=0 → 0_0001_000 = 0x08
        // a=b={2^-6, 0, 0, 0} = 0x00000008
        // dot = 2^(-12) = 0x39800000
        check_mma(`TC_DATA_FP8_E4M3, 32'h00000008, 32'h00000008, 32'h00000000,
                  32'h39800000, "fp8 e4m3 tiny");

        // FP8 E4M3: 0.75 (exp=6, man=100 → 0x34) × 2.0 (0x40)
        // a={0.75,0,0,0}=0x00000034, b={2.0,0,0,0}=0x00000040
        // dot = 1.5 = 0x3FC00000
        check_mma(`TC_DATA_FP8_E4M3, 32'h00000034, 32'h00000040, 32'h00000000,
                  32'h3FC00000, "fp8 e4m3 0.75*2");

        //==================================================================
        // FP4 E2M1 denormal test
        //==================================================================

        // FP4 E2M1 denormal: 0.5 = 0x1 (0_00_1)
        // a={0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5} = 0x11111111
        // b=all 1.0 = 0x22222222
        // dot8 = 8 * 0.5 = 4.0 = 0x40800000
        check_mma(`TC_DATA_FP4_E2M1, 32'h11111111, 32'h22222222, 32'h00000000,
                  32'h40800000, "fp4 e2m1 denorm");

        // FP4 E2M1: 0.5 × 0.5 × 8 = 2.0
        check_mma(`TC_DATA_FP4_E2M1, 32'h11111111, 32'h11111111, 32'h00000000,
                  32'h40000000, "fp4 e2m1 0.5sq");

        //==================================================================
        // Edge cases
        //==================================================================

        // FP8 with zero inputs → c passthrough
        check_mma(`TC_DATA_FP8_E4M3, 32'h00000000, 32'h00000000, 32'h42480000,
                  32'h42480000, "fp8 zero+c");

        // FP16 with zero → c passthrough
        check_mma(`TC_DATA_FP16, 32'h00000000, 32'h00000000, 32'h41A00000,
                  32'h41A00000, "fp16 zero+c");

        //==================================================================
        // Summary
        //==================================================================
        #50;
        $display("===== Tensor Core E2E Tests: %0d/%0d passed =====",
                 pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("FAILURES: %0d", fail_count);
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
