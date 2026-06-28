//============================================================================
// Testbench: video_unit — SIMD 4x8, 2x16, DP4A, DP2A verification
// Gemini #3: Video SIMD verification beyond dp4a
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_video_unit;

    reg        clk, rst_n;
    reg [5:0]  func;
    reg        valid_in, is_signed;
    reg [31:0] operand_a, operand_b, operand_c;

    wire [31:0] result;
    wire        valid_out;
    wire        overflow, saturate_flag;

    video_unit dut (
        .clk(clk), .rst_n(rst_n),
        .func(func), .valid_in(valid_in), .is_signed(is_signed),
        .operand_a(operand_a), .operand_b(operand_b), .operand_c(operand_c),
        .result(result), .valid_out(valid_out),
        .overflow(overflow), .saturate_flag(saturate_flag)
    );

    // Clock: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    //----------------------------------------------------------------------
    // Task: drive one pipelined operation, wait for result, check
    // Pipeline: stage 1 registers inputs, stage 2 computes result
    // Need #1 after final posedge to read non-blocking result
    //----------------------------------------------------------------------
    task check;
        input [5:0]  t_func;
        input        t_signed;
        input [31:0] t_a, t_b, t_c;
        input [31:0] expected;
        input [255:0] name;
        begin
            test_num = test_num + 1;
            @(posedge clk);
            func      <= t_func;
            is_signed <= t_signed;
            valid_in  <= 1'b1;
            operand_a <= t_a;
            operand_b <= t_b;
            operand_c <= t_c;
            @(posedge clk);
            // DUT stage 1 latches our inputs
            @(posedge clk);
            // DUT stage 2 computes result (non-blocking)
            #1; // let non-blocking assignments settle
            valid_in <= 1'b0;
            if (result === expected) begin
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL #%0d %0s: got %08h, expected %08h", test_num, name, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Check result plus overflow/saturate behavior for boundary cases.
    task check_with_flags;
        input [5:0]  t_func;
        input        t_signed;
        input [31:0] t_a, t_b, t_c;
        input [31:0] expected;
        input        expected_overflow;
        input        expected_saturate;
        input [255:0] name;
        begin
            test_num = test_num + 1;
            @(posedge clk);
            func      <= t_func;
            is_signed <= t_signed;
            valid_in  <= 1'b1;
            operand_a <= t_a;
            operand_b <= t_b;
            operand_c <= t_c;
            @(posedge clk);
            @(posedge clk);
            #1;
            valid_in <= 1'b0;

            if (valid_out !== 1'b1) begin
                $display("FAIL #%0d %0s: valid_out=%b expected=1", test_num, name, valid_out);
                fail_count = fail_count + 1;
            end else if ((result === expected) &&
                         (overflow === expected_overflow) &&
                         (saturate_flag === expected_saturate)) begin
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL #%0d %0s: got result=%08h ovf=%b sat=%b, expected result=%08h ovf=%b sat=%b",
                         test_num,
                         name,
                         result,
                         overflow,
                         saturate_flag,
                         expected,
                         expected_overflow,
                         expected_saturate);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        // Reset
        rst_n = 0; valid_in = 0; func = 0;
        is_signed = 0; operand_a = 0; operand_b = 0; operand_c = 0;
        #20;
        rst_n = 1;
        #10;

        //==================================================================
        // SIMD 4x8-bit operations
        //==================================================================

        // VADD4: {0x40,0x30,0x20,0x10} + {0x04,0x03,0x02,0x01}
        //      = {0x44,0x33,0x22,0x11}
        check(`VIDEO_VADD4, 0, 32'h40302010, 32'h04030201, 0,
              32'h44332211, "vadd4 basic");

        // VADD4: byte overflow wraps (0xFF + 0x01 = 0x00)
        check(`VIDEO_VADD4, 0, 32'hFF800140, 32'h01010101, 0,
              32'h00810241, "vadd4 wrap");

        // VSUB4: {0x40,0x30,0x20,0x10} - {0x04,0x03,0x02,0x01}
        //      = {0x3C,0x2D,0x1E,0x0F}
        check(`VIDEO_VSUB4, 0, 32'h40302010, 32'h04030201, 0,
              32'h3C2D1E0F, "vsub4 basic");

        // VSUB4: underflow wraps (0x00 - 0x01 = 0xFF)
        check(`VIDEO_VSUB4, 0, 32'h00100020, 32'h01200030, 0,
              32'hFFF000F0, "vsub4 wrap");

        // VABSDIFF4: |{0x80,0x05,0x20,0x10} - {0x40,0x0A,0x10,0x20}|
        //          = {0x40,0x05,0x10,0x10}  (unsigned byte absdiff)
        check(`VIDEO_VABSDIFF4, 0, 32'h80052010, 32'h400A1020, 0,
              32'h40051010, "vabsdiff4 basic");

        // VABSDIFF4: same operands = zero
        check(`VIDEO_VABSDIFF4, 0, 32'hAABBCCDD, 32'hAABBCCDD, 0,
              32'h00000000, "vabsdiff4 same");

        //==================================================================
        // SIMD 2x16-bit operations
        //==================================================================

        // VADD2: {0x0200, 0x0100} + {0x0020, 0x0010}
        //      = {0x0220, 0x0110}
        check(`VIDEO_VADD2, 0, 32'h02000100, 32'h00200010, 0,
              32'h02200110, "vadd2 basic");

        // VADD2: overflow wraps (0xFFFF + 1 = 0x0000)
        check(`VIDEO_VADD2, 0, 32'hFFFF8000, 32'h00018001, 0,
              32'h00000001, "vadd2 wrap");

        // VSUB2: {0x0500, 0x0300} - {0x0200, 0x0100}
        //      = {0x0300, 0x0200}
        check(`VIDEO_VSUB2, 0, 32'h05000300, 32'h02000100, 0,
              32'h03000200, "vsub2 basic");

        // VMUL2: {5, 3} * {6, 4} = {30, 12}  (low 16 bits)
        check(`VIDEO_VMUL2, 0, 32'h00050003, 32'h00060004, 0,
              32'h001E000C, "vmul2 basic");

        // VMUL2: {0x0010, 0x0100} * {0x0100, 0x0010}
        //      = {0x1000, 0x1000}
        check(`VIDEO_VMUL2, 0, 32'h00100100, 32'h01000010, 0,
              32'h10001000, "vmul2 shift");

        //==================================================================
        // DP4A - 4-element dot product with accumulate (INT8 ML)
        //==================================================================

        // DP4A unsigned: a={1,2,3,4} b={5,6,7,8} c=100
        // dot = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        // result = 100 + 70 = 170 = 0xAA
        check(`VIDEO_DP4A, 0, 32'h04030201, 32'h08070605, 32'd100,
              32'h000000AA, "dp4a unsigned");

        // DP4A unsigned: all ones → 4
        check(`VIDEO_DP4A, 0, 32'h01010101, 32'h01010101, 32'd0,
              32'h00000004, "dp4a u ones");

        // DP4A unsigned: max bytes 255*255 = 65025 = 0xFE01
        check(`VIDEO_DP4A, 0, 32'h000000FF, 32'h000000FF, 32'd0,
              32'h0000FE01, "dp4a u max");

        // DP4A signed: a={-1,-2,3,4} b={5,6,-7,8}
        // -1=0xFF, -2=0xFE, -7=0xF9
        // dot = (-1)*5 + (-2)*6 + 3*(-7) + 4*8 = -5-12-21+32 = -6
        // c=10 → result = 4
        check(`VIDEO_DP4A, 1, 32'h0403FEFF, 32'h08F90605, 32'd10,
              32'h00000004, "dp4a signed");

        // DP4A signed: -128 * 2 = -256
        check(`VIDEO_DP4A, 1, 32'h00000080, 32'h00000002, 32'd0,
              32'hFFFFFF00, "dp4a s neg");

        // DP4A via ALU path
        check(`VIDEO_DP4A_ALU, 0, 32'h04030201, 32'h08070605, 32'd100,
              32'h000000AA, "dp4a_alu");

        // DP4A boundary: unsigned accumulate overflow wraps; no saturation flags.
        check_with_flags(`VIDEO_DP4A, 0, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF,
                         32'h0003F803, 1'b0, 1'b0, "dp4a u wrap boundary");

        // DP4A boundary: signed positive overflow wraps through sign bit.
        check_with_flags(`VIDEO_DP4A, 1, 32'h7F7F7F7F, 32'h7F7F7F7F, 32'h7FFF1000,
                         32'h80000C04, 1'b0, 1'b0, "dp4a s wrap pos");

        // DP4A boundary: signed negative underflow wraps; no saturation.
        check_with_flags(`VIDEO_DP4A, 1, 32'h80808080, 32'h7F7F7F7F, 32'h80000010,
                         32'h7FFF0210, 1'b0, 1'b0, "dp4a s wrap neg");

        // DP4A ALU-routed variants hit the same wrap/flag behavior.
        check_with_flags(`VIDEO_DP4A_ALU, 0, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF,
                         32'h0003F803, 1'b0, 1'b0, "dp4a alu u wrap");
        check_with_flags(`VIDEO_DP4A_ALU, 1, 32'h7F7F7F7F, 32'h7F7F7F7F, 32'h7FFF1000,
                         32'h80000C04, 1'b0, 1'b0, "dp4a alu s wrap");

        //==================================================================
        // DP2A - 2-element dot product with accumulate (INT16)
        //==================================================================

        // NOTE: lane packing is {half1, half0} in [31:16] and [15:0].

        // DP2A unsigned: a={5, 3} b={6, 4} c=10
        // dot = 3*4 + 5*6 = 12+30 = 42, result = 52 = 0x34
        check(`VIDEO_DP2A, 0, 32'h00050003, 32'h00060004, 32'd10,
              32'h00000034, "dp2a unsigned");

        // DP2A unsigned: a={0x0200, 0x0100} b={0x0004, 0x0003}
        // dot = 256*3 + 512*4 = 768+2048 = 2816 = 0x0B00
        check(`VIDEO_DP2A, 0, 32'h02000100, 32'h00040003, 32'd0,
              32'h00000B00, "dp2a u large");

        // DP2A signed: a={2, -1} b={-4, 3} c=100
        // -1=0xFFFF, -4=0xFFFC
        // dot = (-1)*3 + 2*(-4) = -3-8 = -11, result = 89 = 0x59
        check(`VIDEO_DP2A, 1, 32'h0002FFFF, 32'hFFFC0003, 32'd100,
              32'h00000059, "dp2a signed");

        // DP2A signed: negative result
        // a={-200, -100} b={4, 3} c=0
        // -100=0xFF9C, -200=0xFF38
        // dot = (-100)*3 + (-200)*4 = -300-800 = -1100 = 0xFFFFFBB4
        check(`VIDEO_DP2A, 1, 32'hFF38FF9C, 32'h00040003, 32'd0,
              32'hFFFFFBB4, "dp2a s neg");

        // DP2A via ALU path
        check(`VIDEO_DP2A_ALU, 0, 32'h00050003, 32'h00060004, 32'd10,
              32'h00000034, "dp2a_alu");

        // DP2A boundary: unsigned accumulate overflow wraps; no saturation flags.
        check_with_flags(`VIDEO_DP2A, 0, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF,
                         32'hFFFC0001, 1'b0, 1'b0, "dp2a u wrap boundary");

        // DP2A boundary: signed positive overflow wraps.
        check_with_flags(`VIDEO_DP2A, 1, 32'h7FFF0001, 32'h7FFF0001, 32'h7FFFFFFF,
                         32'hBFFF0001, 1'b0, 1'b0, "dp2a s wrap pos");

        // DP2A boundary: signed negative underflow wraps.
        check_with_flags(`VIDEO_DP2A, 1, 32'h8000FFFF, 32'h7FFF7FFF, 32'h80000000,
                         32'h40000001, 1'b0, 1'b0, "dp2a s wrap neg");

        // DP2A ALU-routed variants hit the same wrap/flag behavior.
        check_with_flags(`VIDEO_DP2A_ALU, 0, 32'hFFFFFFFF, 32'hFFFFFFFF, 32'hFFFFFFFF,
                         32'hFFFC0001, 1'b0, 1'b0, "dp2a alu u wrap");
        check_with_flags(`VIDEO_DP2A_ALU, 1, 32'h8000FFFF, 32'h7FFF7FFF, 32'h80000000,
                         32'h40000001, 1'b0, 1'b0, "dp2a alu s wrap");

        //==================================================================
        // Scalar 32-bit operations
        //==================================================================

        // VADD unsigned
        check(`VIDEO_VADD, 0, 32'd100, 32'd200, 0,
              32'd300, "vadd unsigned");

        // VADD signed: (-10) + 25 = 15
        check(`VIDEO_VADD, 1, 32'hFFFFFFF6, 32'd25, 0,
              32'd15, "vadd signed");

        // VSUB unsigned
        check(`VIDEO_VSUB, 0, 32'd500, 32'd200, 0,
              32'd300, "vsub unsigned");

        // VABSDIFF unsigned: |100 - 250| = 150
        check(`VIDEO_VABSDIFF, 0, 32'd100, 32'd250, 0,
              32'd150, "vabsdiff unsigned");

        // VABSDIFF signed: |-50 - 30| = 80
        check(`VIDEO_VABSDIFF, 1, 32'hFFFFFFCE, 32'd30, 0,
              32'd80, "vabsdiff signed");

        // VMIN unsigned
        check(`VIDEO_VMIN, 0, 32'd42, 32'd99, 0,
              32'd42, "vmin unsigned");

        // VMIN signed: min(-5, 3) = -5
        check(`VIDEO_VMIN, 1, 32'hFFFFFFFB, 32'd3, 0,
              32'hFFFFFFFB, "vmin signed");

        // VMAX unsigned
        check(`VIDEO_VMAX, 0, 32'd42, 32'd99, 0,
              32'd99, "vmax unsigned");

        // VMAX signed: max(-5, 3) = 3
        check(`VIDEO_VMAX, 1, 32'hFFFFFFFB, 32'd3, 0,
              32'd3, "vmax signed");

        // VSHL: 1 << 4 = 16
        check(`VIDEO_VSHL, 0, 32'd1, 32'd4, 0,
              32'd16, "vshl");

        // VSHR unsigned: 0x80 >> 3 = 0x10
        check(`VIDEO_VSHR, 0, 32'h80, 32'd3, 0,
              32'h10, "vshr unsigned");

        // VSHR signed (arithmetic): 0xFFFFFF00 >>> 4 = 0xFFFFFFF0
        check(`VIDEO_VSHR, 1, 32'hFFFFFF00, 32'd4, 0,
              32'hFFFFFFF0, "vshr signed");

        // VMAD unsigned: 5*10+7 = 57
        check(`VIDEO_VMAD, 0, 32'd5, 32'd10, 32'd7,
              32'd57, "vmad unsigned");

        // VMAD signed: (-3)*4+100 = 88
        check(`VIDEO_VMAD, 1, 32'hFFFFFFFD, 32'd4, 32'd100,
              32'd88, "vmad signed");

        //==================================================================
        // Summary
        //==================================================================
        #20;
        $display("===== Video Unit Tests: %0d/%0d passed =====",
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
        #50000;
        $display("TIMEOUT");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
