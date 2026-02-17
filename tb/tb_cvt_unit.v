//============================================================================
// RalphGPU - CVT Unit Standalone Testbench
// Tests: FP32<->S32/U32, FP64<->S64/U64, FP32<->FP64, FP16<->FP32
// Includes saturation, rounding, special values
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_cvt_unit;

    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg  [1:0]  rnd_mode;
    reg         saturate;
    reg         ftz;
    reg         valid_in;
    reg  [63:0] src;
    reg  [2:0]  src_type;

    wire [63:0] dst;
    wire        valid_out;
    wire        overflow_flag;  // renamed to avoid keyword
    wire        inexact_flag;

    cvt_unit uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .func      (func),
        .rnd_mode  (rnd_mode),
        .saturate  (saturate),
        .ftz       (ftz),
        .valid_in  (valid_in),
        .src       (src),
        .src_type  (src_type),
        .dst       (dst),
        .valid_out (valid_out),
        .overflow  (overflow_flag),
        .inexact   (inexact_flag)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Test counters
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // FP constants
    //------------------------------------------------------------------------
    // FP32: 1.0=32'h3F800000, 2.0=32'h40000000, -1.0=32'hBF800000
    //       0.5=32'h3F000000, 3.5=32'h40600000, 100.0=32'h42C80000
    //       0.0=32'h00000000, Inf=32'h7F800000, NaN=32'h7FC00000
    //       42.0=32'h42280000, -42.0=32'hC2280000
    //       255.0=32'h437F0000, 65535.0=32'h477FFF00
    //
    // FP64: 1.0=64'h3FF0000000000000, 2.0=64'h4000000000000000
    //       -1.0=64'hBFF0000000000000, 100.0=64'h4059000000000000
    //       42.0=64'h4045000000000000
    //       1.5=64'h3FF8000000000000
    //
    // FP16: 1.0=16'h3C00, 2.0=16'h4000, -1.0=16'hBC00
    //       0.5=16'h3800, Inf=16'h7C00, NaN=16'h7E00

    //------------------------------------------------------------------------
    // Task: Apply stimulus and wait for result (1-cycle registered)
    //------------------------------------------------------------------------
    task apply_cvt;
        input [5:0]  t_func;
        input [63:0] t_src;
        input [2:0]  t_src_type;
        input [1:0]  t_rnd;
        input        t_sat;
        begin
            @(posedge clk);
            func     <= t_func;
            src      <= t_src;
            src_type <= t_src_type;
            rnd_mode <= t_rnd;
            saturate <= t_sat;
            ftz      <= 1'b0;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
            @(posedge clk); // result available
        end
    endtask

    // Check 64-bit result (full)
    task check_result;
        input [63:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (dst == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%016x", test_name, dst);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%016x, got 0x%016x",
                         test_name, expected, dst);
            end
        end
    endtask

    // Check lower 32 bits only
    task check_result32;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (dst[31:0] == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%08x", test_name, dst[31:0]);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%08x, got 0x%08x",
                         test_name, expected, dst[31:0]);
            end
        end
    endtask

    // Check lower 16 bits only
    task check_result16;
        input [15:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (dst[15:0] == expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: got 0x%04x", test_name, dst[15:0]);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: expected 0x%04x, got 0x%04x",
                         test_name, expected, dst[15:0]);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU CVT Unit - Comprehensive Test Suite");
        $display("============================================================");

        // Reset
        rst_n = 0;
        valid_in = 0;
        func = 0;
        rnd_mode = 0;
        saturate = 0;
        ftz = 0;
        src = 0;
        src_type = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        //====================================================================
        // 1. CVT_S32_F32 (FP32 -> signed int32)
        //====================================================================
        $display("\n--- CVT_S32_F32 ---");

        // 42.0 -> 42
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h42280000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd42, "cvt.s32.f32(42.0) = 42");

        // -42.0 -> -42
        apply_cvt(`CVT_S32_F32, {32'h0, 32'hC2280000}, 3'd0, 2'b00, 1'b0);
        check_result32(-32'd42, "cvt.s32.f32(-42.0) = -42");

        // 0.0 -> 0
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h00000000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd0, "cvt.s32.f32(0.0) = 0");

        // 1.0 -> 1
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h3F800000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd1, "cvt.s32.f32(1.0) = 1");

        // 3.5 -> 4 (round to nearest)
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h40600000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd4, "cvt.s32.f32(3.5) = 4 (rnd)");

        // 3.5 -> 3 (round toward zero)
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h40600000}, 3'd0, 2'b01, 1'b0);
        check_result32(32'd3, "cvt.s32.f32(3.5) = 3 (rtz)");

        // NaN -> 0 (with overflow)
        apply_cvt(`CVT_S32_F32, {32'h0, 32'h7FC00000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd0, "cvt.s32.f32(NaN) = 0");

        //====================================================================
        // 2. CVT_U32_F32 (FP32 -> unsigned int32)
        //====================================================================
        $display("\n--- CVT_U32_F32 ---");

        // 42.0 -> 42
        apply_cvt(`CVT_U32_F32, {32'h0, 32'h42280000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd42, "cvt.u32.f32(42.0) = 42");

        // 0.0 -> 0
        apply_cvt(`CVT_U32_F32, {32'h0, 32'h00000000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd0, "cvt.u32.f32(0.0) = 0");

        // 255.0 -> 255
        apply_cvt(`CVT_U32_F32, {32'h0, 32'h437F0000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'd255, "cvt.u32.f32(255.0) = 255");

        //====================================================================
        // 3. CVT_F32_S32 (signed int32 -> FP32)
        //====================================================================
        $display("\n--- CVT_F32_S32 ---");

        // 42 -> 42.0
        apply_cvt(`CVT_F32_S32, {32'h0, 32'd42}, 3'd4, 2'b00, 1'b0);
        check_result32(32'h42280000, "cvt.f32.s32(42) = 42.0");

        // 0 -> 0.0
        apply_cvt(`CVT_F32_S32, {32'h0, 32'd0}, 3'd4, 2'b00, 1'b0);
        check_result32(32'h00000000, "cvt.f32.s32(0) = 0.0");

        // 1 -> 1.0
        apply_cvt(`CVT_F32_S32, {32'h0, 32'd1}, 3'd4, 2'b00, 1'b0);
        check_result32(32'h3F800000, "cvt.f32.s32(1) = 1.0");

        // -1 -> -1.0
        apply_cvt(`CVT_F32_S32, {32'h0, 32'hFFFFFFFF}, 3'd4, 2'b00, 1'b0);
        check_result32(32'hBF800000, "cvt.f32.s32(-1) = -1.0");

        //====================================================================
        // 4. CVT_F32_U32 (unsigned int32 -> FP32)
        //====================================================================
        $display("\n--- CVT_F32_U32 ---");

        // 42 -> 42.0
        apply_cvt(`CVT_F32_U32, {32'h0, 32'd42}, 3'd5, 2'b00, 1'b0);
        check_result32(32'h42280000, "cvt.f32.u32(42) = 42.0");

        // 0 -> 0.0
        apply_cvt(`CVT_F32_U32, {32'h0, 32'd0}, 3'd5, 2'b00, 1'b0);
        check_result32(32'h00000000, "cvt.f32.u32(0) = 0.0");

        //====================================================================
        // 5. CVT_F32_F64 (FP64 -> FP32)
        //====================================================================
        $display("\n--- CVT_F32_F64 ---");

        // 1.0 (f64) -> 1.0 (f32)
        apply_cvt(`CVT_F32_F64, 64'h3FF0_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'h3F800000, "cvt.f32.f64(1.0) = 1.0f");

        // 2.0 (f64) -> 2.0 (f32)
        apply_cvt(`CVT_F32_F64, 64'h4000_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'h40000000, "cvt.f32.f64(2.0) = 2.0f");

        // -1.0 (f64) -> -1.0 (f32)
        apply_cvt(`CVT_F32_F64, 64'hBFF0_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'hBF800000, "cvt.f32.f64(-1.0) = -1.0f");

        // 0.0 (f64) -> 0.0 (f32)
        apply_cvt(`CVT_F32_F64, 64'h0000_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'h00000000, "cvt.f32.f64(0.0) = 0.0f");

        // NaN (f64) -> NaN (f32)
        apply_cvt(`CVT_F32_F64, 64'h7FF8_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'h7FC00000, "cvt.f32.f64(NaN) = NaN_f32");

        // +Inf (f64) -> +Inf (f32)
        apply_cvt(`CVT_F32_F64, 64'h7FF0_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result32(32'h7F800000, "cvt.f32.f64(Inf) = Inf_f32");

        //====================================================================
        // 6. CVT_F64_F32 (FP32 -> FP64)
        //====================================================================
        $display("\n--- CVT_F64_F32 ---");

        // 1.0 (f32) -> 1.0 (f64)
        apply_cvt(`CVT_F64_F32, {32'h0, 32'h3F800000}, 3'd0, 2'b00, 1'b0);
        check_result(64'h3FF0_0000_0000_0000, "cvt.f64.f32(1.0f) = 1.0");

        // 2.0 (f32) -> 2.0 (f64)
        apply_cvt(`CVT_F64_F32, {32'h0, 32'h40000000}, 3'd0, 2'b00, 1'b0);
        check_result(64'h4000_0000_0000_0000, "cvt.f64.f32(2.0f) = 2.0");

        // -1.0 (f32) -> -1.0 (f64)
        apply_cvt(`CVT_F64_F32, {32'h0, 32'hBF800000}, 3'd0, 2'b00, 1'b0);
        check_result(64'hBFF0_0000_0000_0000, "cvt.f64.f32(-1.0f) = -1.0");

        // 0.0 -> 0.0
        apply_cvt(`CVT_F64_F32, {32'h0, 32'h00000000}, 3'd0, 2'b00, 1'b0);
        check_result(64'h0000_0000_0000_0000, "cvt.f64.f32(0.0f) = 0.0");

        // NaN (f32) -> NaN (f64)
        apply_cvt(`CVT_F64_F32, {32'h0, 32'h7FC00000}, 3'd0, 2'b00, 1'b0);
        // NaN f32: sign=0, exp=FF, man=400000
        // f64 NaN: sign=0, exp=7FF, man has bit set
        check_result(64'h7FF8_0000_0000_0000, "cvt.f64.f32(NaN) = NaN_f64");

        // +Inf (f32) -> +Inf (f64)
        apply_cvt(`CVT_F64_F32, {32'h0, 32'h7F800000}, 3'd0, 2'b00, 1'b0);
        check_result(64'h7FF0_0000_0000_0000, "cvt.f64.f32(Inf) = Inf_f64");

        //====================================================================
        // 7. CVT_F32_F16 (FP16 -> FP32) — func code maps to fp16_to_fp32
        //====================================================================
        $display("\n--- CVT_F32_F16 ---");

        // 1.0 (f16) -> 1.0 (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'h3C00}, 3'd0, 2'b00, 1'b0);
        check_result32(32'h3F800000, "cvt.f32.f16(1.0) = 1.0f");

        // 0.5 (f16) -> 0.5 (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'h3800}, 3'd0, 2'b00, 1'b0);
        check_result32(32'h3F000000, "cvt.f32.f16(0.5) = 0.5f");

        // -1.0 (f16) -> -1.0 (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'hBC00}, 3'd0, 2'b00, 1'b0);
        check_result32(32'hBF800000, "cvt.f32.f16(-1.0) = -1.0f");

        // 0.0 (f16) -> 0.0 (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'h0000}, 3'd0, 2'b00, 1'b0);
        check_result32(32'h00000000, "cvt.f32.f16(0.0) = 0.0f");

        // Inf (f16) -> Inf (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'h7C00}, 3'd0, 2'b00, 1'b0);
        check_result32(32'h7F800000, "cvt.f32.f16(Inf) = Inf_f32");

        // NaN (f16) -> NaN (f32)
        apply_cvt(`CVT_F32_F16, {48'h0, 16'h7E00}, 3'd0, 2'b00, 1'b0);
        check_result32(32'h7FC00000, "cvt.f32.f16(NaN) = NaN_f32");

        //====================================================================
        // 8. CVT_F16_F32 (FP32 -> FP16)
        //====================================================================
        $display("\n--- CVT_F16_F32 ---");

        // 1.0 (f32) -> 1.0 (f16)
        apply_cvt(`CVT_F16_F32, {32'h0, 32'h3F800000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'h3C00, "cvt.f16.f32(1.0f) = 1.0_f16");

        // 2.0 -> 2.0
        apply_cvt(`CVT_F16_F32, {32'h0, 32'h40000000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'h4000, "cvt.f16.f32(2.0f) = 2.0_f16");

        // -1.0 -> -1.0
        apply_cvt(`CVT_F16_F32, {32'h0, 32'hBF800000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'hBC00, "cvt.f16.f32(-1.0f) = -1.0_f16");

        // 0.0 -> 0.0
        apply_cvt(`CVT_F16_F32, {32'h0, 32'h00000000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'h0000, "cvt.f16.f32(0.0f) = 0.0_f16");

        // Inf (f32) -> Inf (f16)
        apply_cvt(`CVT_F16_F32, {32'h0, 32'h7F800000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'h7C00, "cvt.f16.f32(Inf) = Inf_f16");

        // NaN (f32) -> NaN (f16)
        apply_cvt(`CVT_F16_F32, {32'h0, 32'h7FC00000}, 3'd0, 2'b00, 1'b0);
        check_result16(16'h7E00, "cvt.f16.f32(NaN) = NaN_f16");

        //====================================================================
        // 9. CVT_S64_F64 (FP64 -> signed int64)
        //====================================================================
        $display("\n--- CVT_S64_F64 ---");

        // 42.0 -> 42
        apply_cvt(`CVT_S64_F64, 64'h4045_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result(64'd42, "cvt.s64.f64(42.0) = 42");

        // 0.0 -> 0
        apply_cvt(`CVT_S64_F64, 64'h0000_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result(64'd0, "cvt.s64.f64(0.0) = 0");

        // 1.0 -> 1
        apply_cvt(`CVT_S64_F64, 64'h3FF0_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result(64'd1, "cvt.s64.f64(1.0) = 1");

        //====================================================================
        // 10. CVT_F64_S64 (signed int64 -> FP64)
        //====================================================================
        $display("\n--- CVT_F64_S64 ---");

        // 42 -> 42.0
        apply_cvt(`CVT_F64_S64, 64'd42, 3'd6, 2'b00, 1'b0);
        check_result(64'h4045_0000_0000_0000, "cvt.f64.s64(42) = 42.0");

        // 0 -> 0.0
        apply_cvt(`CVT_F64_S64, 64'd0, 3'd6, 2'b00, 1'b0);
        check_result(64'h0000_0000_0000_0000, "cvt.f64.s64(0) = 0.0");

        // 1 -> 1.0
        apply_cvt(`CVT_F64_S64, 64'd1, 3'd6, 2'b00, 1'b0);
        check_result(64'h3FF0_0000_0000_0000, "cvt.f64.s64(1) = 1.0");

        //====================================================================
        // 11. CVT_U64_F64 (FP64 -> unsigned int64)
        //====================================================================
        $display("\n--- CVT_U64_F64 ---");

        // 42.0 -> 42
        apply_cvt(`CVT_U64_F64, 64'h4045_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result(64'd42, "cvt.u64.f64(42.0) = 42");

        // 0.0 -> 0
        apply_cvt(`CVT_U64_F64, 64'h0000_0000_0000_0000, 3'd0, 2'b00, 1'b0);
        check_result(64'd0, "cvt.u64.f64(0.0) = 0");

        //====================================================================
        // 12. CVT_F64_U64 (unsigned int64 -> FP64)
        //====================================================================
        $display("\n--- CVT_F64_U64 ---");

        // 42 -> 42.0
        apply_cvt(`CVT_F64_U64, 64'd42, 3'd7, 2'b00, 1'b0);
        check_result(64'h4045_0000_0000_0000, "cvt.f64.u64(42) = 42.0");

        // 0 -> 0.0
        apply_cvt(`CVT_F64_U64, 64'd0, 3'd7, 2'b00, 1'b0);
        check_result(64'h0000_0000_0000_0000, "cvt.f64.u64(0) = 0.0");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("CVT Unit Test Summary");
        $display("============================================================");
        $display("Total: %0d  PASS: %0d  FAIL: %0d", test_count, pass_count, fail_count);
        if (fail_count == 0)
            $display("*** ALL TESTS PASSED ***");
        else
            $display("*** %0d TESTS FAILED ***", fail_count);
        $display("============================================================");

        $finish;
    end

    // Timeout
    initial begin
        #500000;
        $display("[TIMEOUT] Test did not complete within timeout");
        $finish;
    end

endmodule
