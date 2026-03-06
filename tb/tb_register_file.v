//============================================================================
// RalphGPU - Register File Test
// 验证SIMD寄存器文件的读写操作和写掩码功能
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_register_file;

    //------------------------------------------------------------------------
    // 参数
    //------------------------------------------------------------------------
    localparam NUM_REGS   = `NUM_REGS;         // 32
    localparam NUM_LANES  = `THREADS_PER_WARP; // 32
    localparam DATA_WIDTH = `DATA_WIDTH;       // 32
    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam WARP_ID_W  = $clog2(NUM_WARPS);

    //------------------------------------------------------------------------
    // 时钟和复位
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT 信号
    //------------------------------------------------------------------------
    reg  [WARP_ID_W-1:0]            warp_id;
    reg  [4:0]                      rd_addr_a, rd_addr_b, rd_addr_c;
    wire [NUM_LANES*DATA_WIDTH-1:0] rd_data_a, rd_data_b, rd_data_c;
    reg                             wr_en;
    reg  [WARP_ID_W-1:0]            wr_warp;
    reg  [4:0]                      wr_addr;
    reg  [NUM_LANES*DATA_WIDTH-1:0] wr_data;
    reg  [NUM_LANES-1:0]            wr_mask;

    //------------------------------------------------------------------------
    // DUT 实例化
    //------------------------------------------------------------------------
    register_file #(
        .NUM_REGS   (NUM_REGS),
        .NUM_LANES  (NUM_LANES),
        .DATA_WIDTH (DATA_WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .warp_id   (warp_id),
        .rd_addr_a (rd_addr_a),
        .rd_data_a (rd_data_a),
        .rd_addr_b (rd_addr_b),
        .rd_data_b (rd_data_b),
        .rd_addr_c (rd_addr_c),
        .rd_data_c (rd_data_c),
        .wr_en     (wr_en),
        .wr_warp   (wr_warp),
        .wr_addr   (wr_addr),
        .wr_data   (wr_data),
        .wr_mask   (wr_mask)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;
    integer i;
    reg [DATA_WIDTH-1:0] lane_data;

    //------------------------------------------------------------------------
    // 辅助函数
    //------------------------------------------------------------------------
    function [NUM_LANES*DATA_WIDTH-1:0] make_simd_data;
        input [DATA_WIDTH-1:0] base_value;
        integer j;
        begin
            make_simd_data = 0;
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                make_simd_data[j*DATA_WIDTH +: DATA_WIDTH] = base_value + j;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // 测试任务 - 写寄存器
    //------------------------------------------------------------------------
    task write_reg;
        input [4:0] addr;
        input [NUM_LANES*DATA_WIDTH-1:0] data;
        input [NUM_LANES-1:0] mask;
        begin
            @(posedge clk);
            wr_en   <= 1;
            wr_warp <= 0;
            wr_addr <= addr;
            wr_data <= data;
            wr_mask <= mask;
            @(posedge clk);
            wr_en   <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // 测试任务 - 检查读数据
    //------------------------------------------------------------------------
    task check_read;
        input [4:0] addr;
        input [NUM_LANES*DATA_WIDTH-1:0] expected;
        input [127:0] test_name;
        reg match;
        begin
            @(posedge clk);
            rd_addr_a <= addr;
            @(posedge clk);
            #1;  // 等待组合逻辑

            match = (rd_data_a === expected);
            if (match) begin
                $display("[PASS] %s: reg[%0d] correct", test_name, addr);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s: reg[%0d] mismatch", test_name, addr);
                $display("       First lane: got 0x%08X, expected 0x%08X",
                         rd_data_a[31:0], expected[31:0]);
                failed = failed + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 临时变量 (声明在模块级)
    //------------------------------------------------------------------------
    reg test_ok;
    reg ports_ok;
    reg all_ok;
    reg [NUM_LANES*DATA_WIDTH-1:0] single_data;

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Register File Test");
        $display("============================================================");

        // 初始化
        rst_n     = 0;
        wr_en     = 0;
        warp_id   = 0;
        wr_warp   = 0;
        wr_addr   = 0;
        wr_data   = 0;
        wr_mask   = 0;
        rd_addr_a = 0;
        rd_addr_b = 0;
        rd_addr_c = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // 测试1: 复位后读取应为0
        //====================================================================
        $display("\n--- Reset Value Tests ---");

        check_read(5'd0, {NUM_LANES*DATA_WIDTH{1'b0}}, "Reset r0");
        check_read(5'd15, {NUM_LANES*DATA_WIDTH{1'b0}}, "Reset r15");
        check_read(5'd31, {NUM_LANES*DATA_WIDTH{1'b0}}, "Reset r31");

        //====================================================================
        // 测试2: 基本写读测试 (全掩码)
        //====================================================================
        $display("\n--- Basic Write/Read Tests ---");

        // 写r1，每个lane值不同
        write_reg(5'd1, make_simd_data(32'h1000), {NUM_LANES{1'b1}});
        check_read(5'd1, make_simd_data(32'h1000), "Write/Read r1");

        // 写r2
        write_reg(5'd2, make_simd_data(32'h2000), {NUM_LANES{1'b1}});
        check_read(5'd2, make_simd_data(32'h2000), "Write/Read r2");

        // 验证r1未被覆盖
        check_read(5'd1, make_simd_data(32'h1000), "r1 unchanged");

        //====================================================================
        // 测试3: 写掩码测试
        //====================================================================
        $display("\n--- Write Mask Tests ---");

        // 先全写r5
        write_reg(5'd5, make_simd_data(32'hAAAA0000), {NUM_LANES{1'b1}});

        // 只写偶数lane
        write_reg(5'd5, make_simd_data(32'hBBBB0000), 32'h55555555);

        // 验证：偶数lane应该是新值，奇数lane应该是旧值
        @(posedge clk);
        rd_addr_a <= 5'd5;
        @(posedge clk);
        #1;

        test_ok = 1;

        // 检查lane 0 (偶数，应该是新值)
        if (rd_data_a[0*32 +: 32] !== 32'hBBBB0000) test_ok = 0;

        // 检查lane 1 (奇数，应该是旧值)
        if (rd_data_a[1*32 +: 32] !== 32'hAAAA0001) test_ok = 0;

        // 检查lane 2 (偶数，应该是新值+2)
        if (rd_data_a[2*32 +: 32] !== 32'hBBBB0002) test_ok = 0;

        if (test_ok) begin
            $display("[PASS] Write mask: selective update works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Write mask: incorrect values");
            $display("       Lane0: 0x%08X (exp 0xBBBB0000)", rd_data_a[0*32 +: 32]);
            $display("       Lane1: 0x%08X (exp 0xAAAA0001)", rd_data_a[1*32 +: 32]);
            $display("       Lane2: 0x%08X (exp 0xBBBB0002)", rd_data_a[2*32 +: 32]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试4: 空掩码不应写入
        //====================================================================
        $display("\n--- Empty Mask Test ---");

        write_reg(5'd10, make_simd_data(32'hCCCC0000), {NUM_LANES{1'b1}});
        write_reg(5'd10, make_simd_data(32'hDDDD0000), {NUM_LANES{1'b0}});  // 空掩码
        check_read(5'd10, make_simd_data(32'hCCCC0000), "Empty mask no-op");

        //====================================================================
        // 测试5: 多端口读取
        //====================================================================
        $display("\n--- Multi-Port Read Test ---");

        write_reg(5'd20, make_simd_data(32'h20000000), {NUM_LANES{1'b1}});
        write_reg(5'd21, make_simd_data(32'h21000000), {NUM_LANES{1'b1}});
        write_reg(5'd22, make_simd_data(32'h22000000), {NUM_LANES{1'b1}});

        @(posedge clk);
        rd_addr_a <= 5'd20;
        rd_addr_b <= 5'd21;
        rd_addr_c <= 5'd22;
        @(posedge clk);
        #1;

        ports_ok = (rd_data_a[31:0] === 32'h20000000) &&
                  (rd_data_b[31:0] === 32'h21000000) &&
                  (rd_data_c[31:0] === 32'h22000000);

        if (ports_ok) begin
            $display("[PASS] Multi-port read: all 3 ports work");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Multi-port read error");
            $display("       Port A: 0x%08X, Port B: 0x%08X, Port C: 0x%08X",
                     rd_data_a[31:0], rd_data_b[31:0], rd_data_c[31:0]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试6: 所有32个寄存器
        //====================================================================
        $display("\n--- All Registers Test ---");

        for (i = 0; i < NUM_REGS; i = i + 1) begin
            write_reg(i[4:0], make_simd_data(32'hF0000000 + i*32'h100), {NUM_LANES{1'b1}});
        end

        all_ok = 1;

        for (i = 0; i < NUM_REGS; i = i + 1) begin
            @(posedge clk);
            rd_addr_a <= i[4:0];
            @(posedge clk);
            #1;
            if (rd_data_a[31:0] !== 32'hF0000000 + i*32'h100) begin
                all_ok = 0;
                $display("       reg[%0d] mismatch: got 0x%08X", i, rd_data_a[31:0]);
            end
        end

        if (all_ok) begin
            $display("[PASS] All 32 registers write/read correctly");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Some registers failed");
            failed = failed + 1;
        end

        //====================================================================
        // 测试7: 单lane掩码测试
        //====================================================================
        $display("\n--- Single Lane Mask Test ---");

        // 清零r30
        write_reg(5'd30, {NUM_LANES*DATA_WIDTH{1'b0}}, {NUM_LANES{1'b1}});

        // 只写lane 15
        single_data = 0;
        single_data[15*32 +: 32] = 32'hDEADBEEF;
        write_reg(5'd30, single_data, 32'h00008000);  // 只有bit 15

        @(posedge clk);
        rd_addr_a <= 5'd30;
        @(posedge clk);
        #1;

        if (rd_data_a[15*32 +: 32] === 32'hDEADBEEF &&
            rd_data_a[14*32 +: 32] === 32'h0 &&
            rd_data_a[16*32 +: 32] === 32'h0) begin
            $display("[PASS] Single lane mask works (lane 15)");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Single lane mask error");
            failed = failed + 1;
        end

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Register File Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_register_file.vcd");
        $dumpvars(0, tb_register_file);
    end

    initial begin
        #100000;
        $display("ERROR: Timeout!");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
