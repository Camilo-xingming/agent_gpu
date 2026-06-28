//============================================================================
// RalphGPU - Extended ALU Testbench
// 测试所有新增的整数运算指令
//============================================================================

`timescale 1ns/1ps
`include "../rtl/gpu_defines.vh"

module tb_alu_extended;

    //------------------------------------------------------------------------
    // 信号定义
    //------------------------------------------------------------------------
    reg  [5:0]  func;
    reg  [31:0] operand_a;
    reg  [31:0] operand_b;
    reg  [31:0] operand_c;
    reg         pred_in;
    wire [31:0] result;
    wire        zero;
    wire        negative;
    wire        overflow;

    //------------------------------------------------------------------------
    // DUT实例化
    //------------------------------------------------------------------------
    alu u_alu (
        .func      (func),
        .operand_a (operand_a),
        .operand_b (operand_b),
        .operand_c (operand_c),
        .pred_in   (pred_in),
        .result    (result),
        .zero      (zero),
        .negative  (negative),
        .overflow  (overflow)
    );

    //------------------------------------------------------------------------
    // 测试计数器
    //------------------------------------------------------------------------
    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    //------------------------------------------------------------------------
    // 测试任务
    //------------------------------------------------------------------------
    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            #1;
            if (result === expected) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: 0x%08h == 0x%08h", test_name, result, expected);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: got 0x%08h, expected 0x%08h", test_name, result, expected);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 主测试
    //------------------------------------------------------------------------
    initial begin
        $display("============================================");
        $display("RalphGPU Extended ALU Testbench");
        $display("============================================");

        // 初始化
        func = 6'b0;
        operand_a = 32'b0;
        operand_b = 32'b0;
        operand_c = 32'b0;
        pred_in = 1'b0;
        #10;

        //====================================================================
        // 测试 ABS (绝对值)
        //====================================================================
        $display("\n--- Testing ABS ---");
        func = `FUNC_ABS;

        operand_a = 32'h00000005;  // 5
        check_result(32'h00000005, "abs(5)");

        operand_a = 32'hFFFFFFFB;  // -5
        check_result(32'h00000005, "abs(-5)");

        operand_a = 32'h80000000;  // MIN_INT
        check_result(32'h80000000, "abs(MIN_INT)");  // 溢出情况

        operand_a = 32'h00000000;  // 0
        check_result(32'h00000000, "abs(0)");

        //====================================================================
        // 测试 NEG (取反)
        //====================================================================
        $display("\n--- Testing NEG ---");
        func = `FUNC_NEG;

        operand_a = 32'h00000005;  // 5
        check_result(32'hFFFFFFFB, "neg(5)");

        operand_a = 32'hFFFFFFFB;  // -5
        check_result(32'h00000005, "neg(-5)");

        operand_a = 32'h00000000;  // 0
        check_result(32'h00000000, "neg(0)");

        //====================================================================
        // 测试 MIN_S (有符号最小值)
        //====================================================================
        $display("\n--- Testing MIN_S ---");
        func = `FUNC_MIN_S;

        operand_a = 32'h00000005;  operand_b = 32'h0000000A;
        check_result(32'h00000005, "min_s(5, 10)");

        operand_a = 32'hFFFFFFF6;  operand_b = 32'h00000005;  // -10, 5
        check_result(32'hFFFFFFF6, "min_s(-10, 5)");

        operand_a = 32'h00000005;  operand_b = 32'h00000005;
        check_result(32'h00000005, "min_s(5, 5)");

        //====================================================================
        // 测试 MIN_U (无符号最小值)
        //====================================================================
        $display("\n--- Testing MIN_U ---");
        func = `FUNC_MIN_U;

        operand_a = 32'h00000005;  operand_b = 32'h0000000A;
        check_result(32'h00000005, "min_u(5, 10)");

        operand_a = 32'hFFFFFFF6;  operand_b = 32'h00000005;  // 大数 vs 5
        check_result(32'h00000005, "min_u(0xFFFFFFF6, 5)");

        //====================================================================
        // 测试 MAX_S (有符号最大值)
        //====================================================================
        $display("\n--- Testing MAX_S ---");
        func = `FUNC_MAX_S;

        operand_a = 32'h00000005;  operand_b = 32'h0000000A;
        check_result(32'h0000000A, "max_s(5, 10)");

        operand_a = 32'hFFFFFFF6;  operand_b = 32'h00000005;  // -10, 5
        check_result(32'h00000005, "max_s(-10, 5)");

        //====================================================================
        // 测试 MAX_U (无符号最大值)
        //====================================================================
        $display("\n--- Testing MAX_U ---");
        func = `FUNC_MAX_U;

        operand_a = 32'h00000005;  operand_b = 32'h0000000A;
        check_result(32'h0000000A, "max_u(5, 10)");

        operand_a = 32'hFFFFFFF6;  operand_b = 32'h00000005;
        check_result(32'hFFFFFFF6, "max_u(0xFFFFFFF6, 5)");

        //====================================================================
        // 测试 POPC (位计数)
        //====================================================================
        $display("\n--- Testing POPC ---");
        func = `FUNC_POPC;

        operand_a = 32'h00000000;
        check_result(32'h00000000, "popc(0x00000000)");

        operand_a = 32'hFFFFFFFF;
        check_result(32'h00000020, "popc(0xFFFFFFFF)");  // 32

        operand_a = 32'h0000000F;
        check_result(32'h00000004, "popc(0x0000000F)");  // 4

        operand_a = 32'hAAAAAAAA;
        check_result(32'h00000010, "popc(0xAAAAAAAA)");  // 16

        //====================================================================
        // 测试 CLZ (前导零计数)
        //====================================================================
        $display("\n--- Testing CLZ ---");
        func = `FUNC_CLZ;

        operand_a = 32'h80000000;
        check_result(32'h00000000, "clz(0x80000000)");  // 0

        operand_a = 32'h00000001;
        check_result(32'h0000001F, "clz(0x00000001)");  // 31

        operand_a = 32'h00000000;
        check_result(32'h00000020, "clz(0x00000000)");  // 32

        operand_a = 32'h0000FFFF;
        check_result(32'h00000010, "clz(0x0000FFFF)");  // 16

        //====================================================================
        // 测试 BREV (位反转)
        //====================================================================
        $display("\n--- Testing BREV ---");
        func = `FUNC_BREV;

        operand_a = 32'h00000001;
        check_result(32'h80000000, "brev(0x00000001)");

        operand_a = 32'h80000000;
        check_result(32'h00000001, "brev(0x80000000)");

        operand_a = 32'hF0F0F0F0;
        check_result(32'h0F0F0F0F, "brev(0xF0F0F0F0)");

        operand_a = 32'h12345678;
        check_result(32'h1E6A2C48, "brev(0x12345678)");

        //====================================================================
        // 测试 BFE_U (无符号位域提取)
        //====================================================================
        $display("\n--- Testing BFE_U ---");
        func = `FUNC_BFE_U;

        operand_a = 32'hABCDEF12;
        operand_b = {16'b0, 8'd4, 8'd8};  // len=4, pos=8
        check_result(32'h0000000F, "bfe_u(0xABCDEF12, pos=8, len=4)");

        operand_a = 32'hFFFFFFFF;
        operand_b = {16'b0, 8'd8, 8'd0};  // len=8, pos=0
        check_result(32'h000000FF, "bfe_u(0xFFFFFFFF, pos=0, len=8)");

        //====================================================================
        // 测试 BFI (位域插入)
        //====================================================================
        $display("\n--- Testing BFI ---");
        func = `FUNC_BFI;

        operand_a = 32'h0000000F;  // 要插入的值
        operand_b = 32'h00000000;  // 目标
        operand_c = {16'b0, 8'd4, 8'd8};  // len=4, pos=8
        check_result(32'h00000F00, "bfi(0xF into 0 at pos=8, len=4)");

        operand_a = 32'h000000FF;
        operand_b = 32'hFFFF0000;
        operand_c = {16'b0, 8'd8, 8'd0};  // len=8, pos=0
        check_result(32'hFFFF00FF, "bfi(0xFF into 0xFFFF0000 at pos=0, len=8)");

        //====================================================================
        // 测试 SELP (谓词选择)
        //====================================================================
        $display("\n--- Testing SELP ---");
        func = `FUNC_SELP;

        operand_a = 32'hAAAAAAAA;
        operand_b = 32'h55555555;
        pred_in = 1'b1;
        check_result(32'hAAAAAAAA, "selp(A, B, true)");

        pred_in = 1'b0;
        check_result(32'h55555555, "selp(A, B, false)");

        //====================================================================
        // 测试 SLCT (符号选择)
        //====================================================================
        $display("\n--- Testing SLCT ---");
        func = `FUNC_SLCT;

        operand_a = 32'hAAAAAAAA;
        operand_b = 32'hFFFFFFFF;  // 负数
        check_result(32'hAAAAAAAA, "slct(A, B) where B<0");

        operand_b = 32'h00000001;  // 正数
        check_result(32'h00000001, "slct(A, B) where B>=0");

        //====================================================================
        // 测试 SAD (绝对差值和)
        //====================================================================
        $display("\n--- Testing SAD ---");
        func = `FUNC_SAD;

        operand_a = 32'h0000000A;  // 10
        operand_b = 32'h00000005;  // 5
        operand_c = 32'h00000003;  // 3
        check_result(32'h00000008, "sad(10, 5, 3) = |10-5| + 3 = 8");

        operand_a = 32'h00000005;  // 5
        operand_b = 32'h0000000A;  // 10
        operand_c = 32'h00000000;  // 0
        check_result(32'h00000005, "sad(5, 10, 0) = |5-10| + 0 = 5");

        //====================================================================
        // 新增指令测试：BMSK/SZEXT/FNS/SHF/LOP3/CNOT
        //====================================================================
        $display("\n--- Testing BMSK ---");
        func = `FUNC_BMSK;
        operand_a = 32'd2;   // pos
        operand_b = 32'd3;   // len
        check_result(32'h0000001C, "bmsk(pos=2,len=3)");

        $display("\n--- Testing SZEXT ---");
        func = `FUNC_SZEXT;
        operand_a = 32'h00000F80; operand_b = 32'd8;
        check_result(32'hFFFFFF80, "szext width=8 negative");
        operand_a = 32'h0000070F; operand_b = 32'd12;
        check_result(32'h0000070F, "szext width=12 positive");

        $display("\n--- Testing FNS ---");
        func = `FUNC_FNS;
        operand_a = 32'h00000000;
        check_result(32'hFFFFFFFF, "fns(0) -> -1");
        operand_a = 32'h00000010;
        check_result(32'h00000004, "fns(0x10) -> 4");

        $display("\n--- Testing SHF (funnel) ---");
        func = `FUNC_SHF_L;
        operand_a = 32'h00000001; operand_b = 32'h00000000; operand_c = 32'd1;
        check_result(32'h00000002, "shf.l (A=1,B=0,sh=1)");
        func = `FUNC_SHF_R;
        operand_a = 32'h89ABCDEF; operand_b = 32'h01234567; operand_c = 32'd4;
        check_result(32'h789ABCDE, "shf.r (B=0x01234567,A=0x89ABCDEF,sh=4)");

        $display("\n--- Testing LOP3 (LUT=0xCA default) ---");
        func = `FUNC_LOP3;
        operand_a = 32'hFFFF0000;
        operand_b = 32'h12345678;
        operand_c = 32'hDEADBEEF;
        check_result(32'h1234BEEF, "lop3 default LUT 0xCA");

        $display("\n--- Testing CNOT ---");
        func = `FUNC_CNOT;
        operand_a = 32'h0F0F0F0F;
        operand_b = 32'hFFFF0000;
        check_result(32'hF0F00000, "cnot(~A & B)");

        //====================================================================
        // 新增 DP4A/DP2A (ALU 路径)
        //====================================================================
        $display("\n--- Testing DP4A/DP2A ---");
        func = `VIDEO_DP4A_ALU;
        operand_a = 32'h01010101;  // all 1
        operand_b = 32'h02020202;  // all 2
        operand_c = 32'h00000000;
        check_result(32'h00000008, "dp4a (1*2)*4 = 8");

        func = `VIDEO_DP2A_ALU;
        operand_a = 32'h00010002;  // halfwords 1,2
        operand_b = 32'h00030004;  // halfwords 3,4
        operand_c = 32'h00000001;
        check_result(32'h0000000C, "dp2a 1*3 + 2*4 + 1 (signed 16-bit)");

        //====================================================================
        // cvt.pack
        //====================================================================
        $display("\n--- Testing CVT.PACK ---");
        func = `CVT_PACK;
        operand_a = 32'hAAAA5555;
        operand_b = 32'h12345678;
        check_result(32'h56785555, "cvt.pack lower16 A + lower16 B");

        //====================================================================
        // 测试总结
        //====================================================================
        $display("\n============================================");
        $display("Test Summary:");
        $display("  Total:  %0d", test_count);
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("============================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end

        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
