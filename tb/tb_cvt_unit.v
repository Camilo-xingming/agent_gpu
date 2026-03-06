//============================================================================
// RalphGPU - CVT Unit Testbench
// Tests cvt_unit with 12 cases: s32<->f32, u32<->f32, f16<->f32
//============================================================================

`timescale 1ns/1ps
`include "../rtl/gpu_defines.vh"

module tb_cvt_unit;

    //------------------------------------------------------------------------
    // Signals
    //------------------------------------------------------------------------
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
    wire        overflow;
    wire        inexact;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    cvt_unit u_cvt (
        .clk      (clk),
        .rst_n    (rst_n),
        .func     (func),
        .rnd_mode (rnd_mode),
        .saturate (saturate),
        .ftz      (ftz),
        .valid_in (valid_in),
        .src      (src),
        .src_type (src_type),
        .dst      (dst),
        .valid_out(valid_out),
        .overflow (overflow),
        .inexact  (inexact)
    );

    //------------------------------------------------------------------------
    // Clock: 10ns period
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Counters
    //------------------------------------------------------------------------
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // Test task: 1-cycle pipeline (same as fpu64)
    //------------------------------------------------------------------------
    task run_test;
        input [5:0]   t_func;
        input [1:0]   t_rnd;
        input [63:0]  t_src;
        input [2:0]   t_src_type;
        input [63:0]  expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;

            @(negedge clk);
            func      = t_func;
            rnd_mode  = t_rnd;
            saturate  = 1'b0;
            ftz       = 1'b0;
            src       = t_src;
            src_type  = t_src_type;
            valid_in  = 1'b1;

            @(posedge clk);
            @(negedge clk);
            valid_in  = 1'b0;

            @(posedge clk);
            @(negedge clk);

            if (dst === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%016h == 0x%016h", test_name, dst, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got 0x%016h, expected 0x%016h", test_name, dst, expected);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Test sequence
    //------------------------------------------------------------------------
    initial begin
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

        $display("============================================================");
        $display("CVT Unit Testbench — 12 tests");
        $display("============================================================");

        //--------------------------------------------------------------------
        // CVT_S32_F32: float -> signed int (truncate toward zero)
        //--------------------------------------------------------------------
        // 3.7 -> 3
        run_test(`CVT_S32_F32, 2'b01, {32'b0, 32'h406CCCCD}, 3'd0,
                 {32'b0, 32'd3}, "cvt_s32_f32: 3.7->3");

        // -2.5 -> -2 (truncate)
        run_test(`CVT_S32_F32, 2'b01, {32'b0, 32'hC0200000}, 3'd0,
                 {32'b0, 32'hFFFFFFFE}, "cvt_s32_f32: -2.5->-2");

        // 100.9 -> 100
        run_test(`CVT_S32_F32, 2'b01, {32'b0, 32'h42C9CCCD}, 3'd0,
                 {32'b0, 32'd100}, "cvt_s32_f32: 100.9->100");

        //--------------------------------------------------------------------
        // CVT_U32_F32: float -> unsigned int (truncate)
        //--------------------------------------------------------------------
        // 5.9 -> 5
        run_test(`CVT_U32_F32, 2'b01, {32'b0, 32'h40BCCCCD}, 3'd0,
                 {32'b0, 32'd5}, "cvt_u32_f32: 5.9->5");

        // 255.1 -> 255
        run_test(`CVT_U32_F32, 2'b01, {32'b0, 32'h437F199A}, 3'd0,
                 {32'b0, 32'd255}, "cvt_u32_f32: 255.1->255");

        // 0.0 -> 0
        run_test(`CVT_U32_F32, 2'b01, {32'b0, 32'h00000000}, 3'd0,
                 {32'b0, 32'd0}, "cvt_u32_f32: 0.0->0");

        //--------------------------------------------------------------------
        // CVT_F32_S32: signed int -> float
        //--------------------------------------------------------------------
        // 42 -> 42.0 (0x42280000)
        run_test(`CVT_F32_S32, 2'b00, {32'b0, 32'd42}, 3'd4,
                 {32'b0, 32'h42280000}, "cvt_f32_s32: 42->42.0");

        // -7 -> -7.0 (0xC0E00000)
        run_test(`CVT_F32_S32, 2'b00, {32'b0, 32'hFFFFFFF9}, 3'd4,
                 {32'b0, 32'hC0E00000}, "cvt_f32_s32: -7->-7.0");

        // 0 -> 0.0
        run_test(`CVT_F32_S32, 2'b00, {32'b0, 32'd0}, 3'd4,
                 {32'b0, 32'h00000000}, "cvt_f32_s32: 0->0.0");

        //--------------------------------------------------------------------
        // CVT_F32_F16: fp16 -> fp32
        //--------------------------------------------------------------------
        // FP16 1.0 (0x3C00) -> FP32 1.0 (0x3F800000)
        run_test(`CVT_F32_F16, 2'b00, {48'b0, 16'h3C00}, 3'd0,
                 {32'b0, 32'h3F800000}, "cvt_f32_f16: fp16(1.0)->fp32");

        // FP16 0.5 (0x3800) -> FP32 0.5 (0x3F000000)
        run_test(`CVT_F32_F16, 2'b00, {48'b0, 16'h3800}, 3'd0,
                 {32'b0, 32'h3F000000}, "cvt_f32_f16: fp16(0.5)->fp32");

        // FP16 -2.0 (0xC000) -> FP32 -2.0 (0xC0000000)
        run_test(`CVT_F32_F16, 2'b00, {48'b0, 16'hC000}, 3'd0,
                 {32'b0, 32'hC0000000}, "cvt_f32_f16: fp16(-2.0)->fp32");

        //--------------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------------
        $display("============================================================");
        $display("Results: %0d/%0d PASSED, %0d FAILED",
                 pass_count, test_count, fail_count);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL PASSED");
        else
            $display("SOME TESTS FAILED");

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // VCD dump
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_cvt_unit.vcd");
        $dumpvars(0, tb_cvt_unit);
    end

    //------------------------------------------------------------------------
    // Timeout
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
