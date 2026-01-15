//============================================================================
// RalphGPU - Shared Memory Test
// 验证32-bank共享内存的读写和冲突检测
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_shared_memory;

    //------------------------------------------------------------------------
    // 参数
    //------------------------------------------------------------------------
    localparam SIZE_KB    = `SHARED_MEM_KB;    // 16KB
    localparam NUM_BANKS  = 32;
    localparam DATA_WIDTH = 32;
    localparam ADDR_WIDTH = `SHARED_MEM_ADDR_W; // 14

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
    reg                              req_valid;
    reg                              req_write;
    reg  [NUM_BANKS*ADDR_WIDTH-1:0]  req_addr;
    reg  [NUM_BANKS*DATA_WIDTH-1:0]  req_wdata;
    reg  [NUM_BANKS-1:0]             req_mask;
    wire                             resp_valid;
    wire [NUM_BANKS*DATA_WIDTH-1:0]  resp_rdata;
    wire                             bank_conflict;

    //------------------------------------------------------------------------
    // DUT 实例化
    //------------------------------------------------------------------------
    shared_memory #(
        .SIZE_KB    (SIZE_KB),
        .NUM_BANKS  (NUM_BANKS),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .req_valid    (req_valid),
        .req_write    (req_write),
        .req_addr     (req_addr),
        .req_wdata    (req_wdata),
        .req_mask     (req_mask),
        .resp_valid   (resp_valid),
        .resp_rdata   (resp_rdata),
        .bank_conflict(bank_conflict)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;
    integer i;
    reg read_ok;
    reg consistent;
    reg selective_ok;

    //------------------------------------------------------------------------
    // 辅助函数 - 设置地址
    //------------------------------------------------------------------------
    task set_addresses;
        input [ADDR_WIDTH-1:0] base;
        input [31:0] stride;
        integer j;
        begin
            for (j = 0; j < NUM_BANKS; j = j + 1) begin
                req_addr[j*ADDR_WIDTH +: ADDR_WIDTH] = base + j*stride;
            end
        end
    endtask

    task set_write_data;
        input [DATA_WIDTH-1:0] base;
        integer j;
        begin
            for (j = 0; j < NUM_BANKS; j = j + 1) begin
                req_wdata[j*DATA_WIDTH +: DATA_WIDTH] = base + j;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // 写操作任务
    //------------------------------------------------------------------------
    task smem_write;
        input [ADDR_WIDTH-1:0] base_addr;
        input [31:0] stride;
        input [DATA_WIDTH-1:0] base_data;
        input [NUM_BANKS-1:0] mask;
        begin
            @(posedge clk);
            req_valid <= 1;
            req_write <= 1;
            set_addresses(base_addr, stride);
            set_write_data(base_data);
            req_mask <= mask;
            @(posedge clk);
            req_valid <= 0;
            req_write <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // 读操作任务
    //------------------------------------------------------------------------
    task smem_read;
        input [ADDR_WIDTH-1:0] base_addr;
        input [31:0] stride;
        input [NUM_BANKS-1:0] mask;
        begin
            @(posedge clk);
            req_valid <= 1;
            req_write <= 0;
            set_addresses(base_addr, stride);
            req_mask <= mask;
            @(posedge clk);
            req_valid <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Shared Memory Test");
        $display("============================================================");

        // 初始化
        rst_n     = 0;
        req_valid = 0;
        req_write = 0;
        req_addr  = 0;
        req_wdata = 0;
        req_mask  = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // 测试1: 无冲突写入 (stride=1, 每lane访问不同bank)
        //====================================================================
        $display("\n--- No-Conflict Write Test ---");

        // 地址0,1,2,3... 分别映射到bank 0,1,2,3...
        smem_write(14'd0, 32'd1, 32'hA000, {NUM_BANKS{1'b1}});

        if (!bank_conflict) begin
            $display("[PASS] No conflict with stride=1");
            passed = passed + 1;
        end else begin
            $display("[FAIL] False conflict detected with stride=1");
            failed = failed + 1;
        end

        //====================================================================
        // 测试2: 无冲突读取
        //====================================================================
        $display("\n--- No-Conflict Read Test ---");

        smem_read(14'd0, 32'd1, {NUM_BANKS{1'b1}});
        @(posedge clk);  // 等待resp_valid

        if (resp_valid && !bank_conflict) begin
            // 检查数据
            read_ok = 1;
            for (i = 0; i < NUM_BANKS; i = i + 1) begin
                if (resp_rdata[i*DATA_WIDTH +: DATA_WIDTH] !== 32'hA000 + i) begin
                    read_ok = 0;
                    $display("       Lane %0d: got 0x%08X, expected 0x%08X",
                             i, resp_rdata[i*DATA_WIDTH +: DATA_WIDTH], 32'hA000 + i);
                end
            end
            if (read_ok) begin
                $display("[PASS] No-conflict read data correct");
                passed = passed + 1;
            end else begin
                $display("[FAIL] Read data mismatch");
                failed = failed + 1;
            end
        end else begin
            $display("[FAIL] Read response not valid or conflict detected");
            failed = failed + 1;
        end

        //====================================================================
        // 测试3: Bank冲突检测 (所有lane访问同一bank)
        //====================================================================
        $display("\n--- Bank Conflict Detection Test ---");

        // stride=32: 所有地址低5位相同，都映射到bank 0
        @(posedge clk);
        req_valid <= 1;
        req_write <= 0;
        for (i = 0; i < NUM_BANKS; i = i + 1) begin
            req_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i * 32;  // 0, 32, 64, 96...
        end
        req_mask <= {NUM_BANKS{1'b1}};
        @(posedge clk);
        req_valid <= 0;

        if (bank_conflict) begin
            $display("[PASS] Bank conflict correctly detected (all access bank 0)");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Bank conflict not detected");
            failed = failed + 1;
        end

        //====================================================================
        // 测试4: 部分冲突 (两个lane访问同一bank)
        //====================================================================
        $display("\n--- Partial Conflict Test ---");

        @(posedge clk);
        req_valid <= 1;
        req_write <= 0;
        // Lane 0和Lane 1都访问bank 0
        req_addr[0*ADDR_WIDTH +: ADDR_WIDTH] = 14'd0;   // bank 0
        req_addr[1*ADDR_WIDTH +: ADDR_WIDTH] = 14'd32;  // bank 0
        for (i = 2; i < NUM_BANKS; i = i + 1) begin
            req_addr[i*ADDR_WIDTH +: ADDR_WIDTH] = i;   // 各自的bank
        end
        req_mask <= {NUM_BANKS{1'b1}};
        @(posedge clk);
        req_valid <= 0;

        if (bank_conflict) begin
            $display("[PASS] Partial conflict detected (lanes 0,1 -> bank 0)");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Partial conflict not detected");
            failed = failed + 1;
        end

        //====================================================================
        // 测试5: 掩码禁用冲突lane
        //====================================================================
        $display("\n--- Mask Disable Conflict Test ---");

        @(posedge clk);
        req_valid <= 1;
        req_write <= 0;
        req_addr[0*ADDR_WIDTH +: ADDR_WIDTH] = 14'd0;
        req_addr[1*ADDR_WIDTH +: ADDR_WIDTH] = 14'd32;  // 与lane 0冲突
        req_mask <= 32'hFFFFFFFE;  // 禁用lane 0
        @(posedge clk);
        req_valid <= 0;

        // 禁用一个冲突lane后应该无冲突
        if (!bank_conflict) begin
            $display("[PASS] No conflict when one conflicting lane is masked");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Conflict still reported with masked lane");
            failed = failed + 1;
        end

        //====================================================================
        // 测试6: 写后读一致性
        //====================================================================
        $display("\n--- Write-Read Consistency Test ---");

        // 写入特定模式
        smem_write(14'd64, 32'd1, 32'hBEEF0000, {NUM_BANKS{1'b1}});
        #10;

        // 读回
        smem_read(14'd64, 32'd1, {NUM_BANKS{1'b1}});
        @(posedge clk);

        consistent = 1;
        for (i = 0; i < 4; i = i + 1) begin  // 只检查前4个lane
            if (resp_rdata[i*DATA_WIDTH +: DATA_WIDTH] !== 32'hBEEF0000 + i) begin
                consistent = 0;
            end
        end
        if (consistent) begin
            $display("[PASS] Write-read consistency verified");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Write-read inconsistency");
            failed = failed + 1;
        end

        //====================================================================
        // 测试7: 边界地址测试
        //====================================================================
        $display("\n--- Boundary Address Test ---");

        // 写入高地址区域
        smem_write(14'h3F00, 32'd1, 32'hFFFF0000, 32'h0000000F);  // 只写前4个lane
        #10;
        smem_read(14'h3F00, 32'd1, 32'h0000000F);
        @(posedge clk);

        if (resp_rdata[0*32 +: 32] === 32'hFFFF0000) begin
            $display("[PASS] High address access works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] High address access error");
            failed = failed + 1;
        end

        //====================================================================
        // 测试8: 选择性写入测试
        //====================================================================
        $display("\n--- Selective Write Test ---");

        // 先全写
        smem_write(14'd128, 32'd1, 32'h11110000, {NUM_BANKS{1'b1}});

        // 只更新偶数lane
        smem_write(14'd128, 32'd1, 32'h22220000, 32'h55555555);

        smem_read(14'd128, 32'd1, {NUM_BANKS{1'b1}});
        @(posedge clk);

        selective_ok = 1;
        // Lane 0 (偶): 应该是0x22220000
        if (resp_rdata[0*32 +: 32] !== 32'h22220000) selective_ok = 0;
        // Lane 1 (奇): 应该是0x11110001
        if (resp_rdata[1*32 +: 32] !== 32'h11110001) selective_ok = 0;

        if (selective_ok) begin
            $display("[PASS] Selective write with mask works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Selective write error");
            $display("       Lane0: 0x%08X (exp 0x22220000)", resp_rdata[0*32 +: 32]);
            $display("       Lane1: 0x%08X (exp 0x11110001)", resp_rdata[1*32 +: 32]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Shared Memory Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        $finish;
    end

    //------------------------------------------------------------------------
    // 波形输出
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_shared_memory.vcd");
        $dumpvars(0, tb_shared_memory);
    end

    initial begin
        #50000;
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule
