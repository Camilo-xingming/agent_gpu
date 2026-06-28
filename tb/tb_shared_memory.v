//============================================================================
// RalphGPU - Shared Memory Test
// 验证32-bank共享内存的读写、冲突统计和warp级原子操作
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
    // Atomic port signals
    //------------------------------------------------------------------------
    reg                    atomic_req_valid;
    reg                    atomic_req_write;
    reg  [ADDR_WIDTH-1:0]  atomic_req_addr;
    reg  [DATA_WIDTH-1:0]  atomic_req_wdata;
    reg                    atomic_req_mask;
    wire                   atomic_resp_valid;
    wire [DATA_WIDTH-1:0]  atomic_resp_rdata;

    //------------------------------------------------------------------------
    // Async write port signals (for cp.async)
    //------------------------------------------------------------------------
    reg                     async_wr_en;
    reg  [ADDR_WIDTH-1:0]   async_wr_addr;
    reg  [127:0]            async_wr_data;
    reg  [4:0]              async_wr_size;  // 5 bits to hold values up to 16

    //------------------------------------------------------------------------
    // Warp-atomic emulation vectors (testbench side)
    //------------------------------------------------------------------------
    reg [NUM_BANKS*32-1:0] atomic_addr_vec;
    reg [NUM_BANKS*32-1:0] atomic_operand_a_vec;
    reg [NUM_BANKS*32-1:0] atomic_operand_b_vec;
    reg [NUM_BANKS*32-1:0] atomic_old_values;

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
        .bank_conflict(bank_conflict),
        // Atomic port
        .atomic_req_valid(atomic_req_valid),
        .atomic_req_write(atomic_req_write),
        .atomic_req_addr (atomic_req_addr),
        .atomic_req_wdata(atomic_req_wdata),
        .atomic_req_mask (atomic_req_mask),
        .atomic_resp_valid(atomic_resp_valid),
        .atomic_resp_rdata(atomic_resp_rdata),
        // Async copy write port
        .async_wr_en  (async_wr_en),
        .async_wr_addr(async_wr_addr),
        .async_wr_data(async_wr_data),
        .async_wr_size(async_wr_size)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;
    integer i;
    integer resp_valid_count;
    integer resp_before;
    integer resp_after;
    integer issued_reqs;
    integer conflict_cycles;
    integer clean_cycles;
    reg read_ok;
    reg consistent;
    reg selective_ok;
    reg [31:0] lane0_value;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid_count <= 0;
        end else if (resp_valid) begin
            resp_valid_count <= resp_valid_count + 1;
        end
    end

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

    task atomic_clear_vectors;
        begin
            atomic_addr_vec      = {NUM_BANKS*32{1'b0}};
            atomic_operand_a_vec = {NUM_BANKS*32{1'b0}};
            atomic_operand_b_vec = {NUM_BANKS*32{1'b0}};
            atomic_old_values    = {NUM_BANKS*32{1'b0}};
        end
    endtask

    task atomic_set_lane;
        input integer lane_idx;
        input [31:0] lane_addr;
        input [31:0] lane_op_a;
        input [31:0] lane_op_b;
        begin
            atomic_addr_vec[lane_idx*32 +: 32]      = lane_addr;
            atomic_operand_a_vec[lane_idx*32 +: 32] = lane_op_a;
            atomic_operand_b_vec[lane_idx*32 +: 32] = lane_op_b;
        end
    endtask

    task atomic_port_read;
        input [ADDR_WIDTH-1:0] addr_i;
        output [31:0] data_o;
        begin
            @(posedge clk);
            atomic_req_valid <= 1'b1;
            atomic_req_write <= 1'b0;
            atomic_req_addr  <= addr_i;
            atomic_req_wdata <= 32'b0;
            atomic_req_mask  <= 1'b1;

            @(posedge clk);
            atomic_req_valid <= 1'b0;
            atomic_req_mask  <= 1'b0;

            wait (atomic_resp_valid == 1'b1);
            data_o = atomic_resp_rdata;
            @(posedge clk);
        end
    endtask

    task atomic_port_write;
        input [ADDR_WIDTH-1:0] addr_i;
        input [31:0] data_i;
        begin
            @(posedge clk);
            atomic_req_valid <= 1'b1;
            atomic_req_write <= 1'b1;
            atomic_req_addr  <= addr_i;
            atomic_req_wdata <= data_i;
            atomic_req_mask  <= 1'b1;

            @(posedge clk);
            atomic_req_valid <= 1'b0;
            atomic_req_mask  <= 1'b0;

            wait (atomic_resp_valid == 1'b1);
            @(posedge clk);
        end
    endtask

    task run_warp_atomic_add;
        input [NUM_BANKS-1:0] mask_i;
        integer lane;
        reg [31:0] addr_lane;
        reg [31:0] old_value;
        reg [31:0] new_value;
        begin
            for (lane = 0; lane < NUM_BANKS; lane = lane + 1) begin
                if (mask_i[lane]) begin
                    addr_lane = atomic_addr_vec[lane*32 +: 32];
                    atomic_port_read(addr_lane[ADDR_WIDTH-1:0], old_value);
                    atomic_old_values[lane*32 +: 32] = old_value;
                    new_value = old_value + atomic_operand_a_vec[lane*32 +: 32];
                    atomic_port_write(addr_lane[ADDR_WIDTH-1:0], new_value);
                end
            end
        end
    endtask

    task run_warp_atomic_max_u;
        input [NUM_BANKS-1:0] mask_i;
        integer lane;
        reg [31:0] addr_lane;
        reg [31:0] old_value;
        reg [31:0] op_value;
        reg [31:0] new_value;
        begin
            for (lane = 0; lane < NUM_BANKS; lane = lane + 1) begin
                if (mask_i[lane]) begin
                    addr_lane = atomic_addr_vec[lane*32 +: 32];
                    op_value  = atomic_operand_a_vec[lane*32 +: 32];
                    atomic_port_read(addr_lane[ADDR_WIDTH-1:0], old_value);
                    atomic_old_values[lane*32 +: 32] = old_value;
                    new_value = (old_value > op_value) ? old_value : op_value;
                    atomic_port_write(addr_lane[ADDR_WIDTH-1:0], new_value);
                end
            end
        end
    endtask

    task run_warp_atomic_cas;
        input [NUM_BANKS-1:0] mask_i;
        integer lane;
        reg [31:0] addr_lane;
        reg [31:0] old_value;
        reg [31:0] cmp_value;
        reg [31:0] swap_value;
        begin
            for (lane = 0; lane < NUM_BANKS; lane = lane + 1) begin
                if (mask_i[lane]) begin
                    addr_lane  = atomic_addr_vec[lane*32 +: 32];
                    cmp_value  = atomic_operand_a_vec[lane*32 +: 32];
                    swap_value = atomic_operand_b_vec[lane*32 +: 32];
                    atomic_port_read(addr_lane[ADDR_WIDTH-1:0], old_value);
                    atomic_old_values[lane*32 +: 32] = old_value;
                    if (old_value == cmp_value) begin
                        atomic_port_write(addr_lane[ADDR_WIDTH-1:0], swap_value);
                    end
                end
            end
        end
    endtask

    task read_lane0_word;
        input [ADDR_WIDTH-1:0] addr_i;
        output [31:0] data_o;
        begin
            smem_read(addr_i, 32'd1, 32'h00000001);
            @(posedge clk);
            data_o = resp_rdata[0*DATA_WIDTH +: DATA_WIDTH];
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

        atomic_req_valid = 0;
        atomic_req_write = 0;
        atomic_req_addr  = 0;
        atomic_req_wdata = 0;
        atomic_req_mask  = 0;

        async_wr_en   = 0;
        async_wr_addr = 0;
        async_wr_data = 0;
        async_wr_size = 0;

        atomic_clear_vectors();

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
        // 测试9: Async Write Port - 4 bytes
        //====================================================================
        $display("\n--- Async Write Port Test (4 bytes) ---");

        @(posedge clk);
        async_wr_en <= 1;
        async_wr_addr <= 14'd256;  // Word address
        async_wr_data <= 128'hDEADBEEF_12345678_CAFEBABE_87654321;
        async_wr_size <= 5'd4;  // 4 bytes = 1 word
        @(posedge clk);
        async_wr_en <= 0;
        #20;

        // Read back using normal interface
        smem_read(14'd256, 32'd1, 32'h00000001);  // Only lane 0
        @(posedge clk);

        if (resp_rdata[0*32 +: 32] === 32'h87654321) begin  // LSB 32 bits
            $display("[PASS] Async write 4 bytes works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Async write 4 bytes failed: got 0x%08X, exp 0x87654321",
                     resp_rdata[0*32 +: 32]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试10: Async Write Port - 8 bytes
        //====================================================================
        $display("\n--- Async Write Port Test (8 bytes) ---");

        @(posedge clk);
        async_wr_en <= 1;
        async_wr_addr <= 14'd512;
        async_wr_data <= 128'hAAAABBBB_CCCCDDDD_11112222_33334444;
        async_wr_size <= 5'd8;  // 8 bytes = 2 words
        @(posedge clk);
        async_wr_en <= 0;
        #20;

        smem_read(14'd512, 32'd1, 32'h00000003);  // Lane 0 and 1
        @(posedge clk);

        if (resp_rdata[0*32 +: 32] === 32'h33334444 &&
            resp_rdata[1*32 +: 32] === 32'h11112222) begin
            $display("[PASS] Async write 8 bytes works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Async write 8 bytes failed:");
            $display("       Word0: got 0x%08X, exp 0x33334444", resp_rdata[0*32 +: 32]);
            $display("       Word1: got 0x%08X, exp 0x11112222", resp_rdata[1*32 +: 32]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试11: Async Write Port - 16 bytes
        //====================================================================
        $display("\n--- Async Write Port Test (16 bytes) ---");

        @(posedge clk);
        async_wr_en <= 1;
        async_wr_addr <= 14'd768;
        async_wr_data <= 128'hFEDCBA98_76543210_01234567_89ABCDEF;
        async_wr_size <= 5'd16;  // 16 bytes = 4 words
        @(posedge clk);
        async_wr_en <= 0;
        #20;

        smem_read(14'd768, 32'd1, 32'h0000000F);  // Lanes 0-3
        @(posedge clk);

        if (resp_rdata[0*32 +: 32] === 32'h89ABCDEF &&
            resp_rdata[1*32 +: 32] === 32'h01234567 &&
            resp_rdata[2*32 +: 32] === 32'h76543210 &&
            resp_rdata[3*32 +: 32] === 32'hFEDCBA98) begin
            $display("[PASS] Async write 16 bytes works");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Async write 16 bytes failed:");
            $display("       Word0: got 0x%08X, exp 0x89ABCDEF", resp_rdata[0*32 +: 32]);
            $display("       Word1: got 0x%08X, exp 0x01234567", resp_rdata[1*32 +: 32]);
            $display("       Word2: got 0x%08X, exp 0x76543210", resp_rdata[2*32 +: 32]);
            $display("       Word3: got 0x%08X, exp 0xFEDCBA98", resp_rdata[3*32 +: 32]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试12: 同bank访问冲突周期计数 + 响应完整性
        //====================================================================
        $display("\n--- Bank Conflict Stall Accounting Test ---");

        conflict_cycles = 0;
        issued_reqs = 8;
        req_valid <= 0;
        req_mask  <= 0;
        repeat (2) @(posedge clk);
        resp_before = resp_valid_count;

        for (i = 0; i < issued_reqs; i = i + 1) begin
            @(posedge clk);
            req_valid <= 1;
            req_write <= 0;
            set_addresses(14'd1024 + i*14'd64, 32'd32);  // 全lane落在同bank，冲突
            req_mask <= {NUM_BANKS{1'b1}};
            #1;
            if (bank_conflict) conflict_cycles = conflict_cycles + 1;
        end

        @(posedge clk);
        req_valid <= 0;
        req_mask  <= 0;
        repeat (2) @(posedge clk);
        resp_after = resp_valid_count;

        if (conflict_cycles == issued_reqs) begin
            $display("[PASS] Conflict stall cycles counted correctly (%0d)", conflict_cycles);
            passed = passed + 1;
        end else begin
            $display("[FAIL] Conflict cycle count mismatch: got %0d exp %0d", conflict_cycles, issued_reqs);
            failed = failed + 1;
        end

        if ((resp_after - resp_before) == issued_reqs) begin
            $display("[PASS] Conflict traffic keeps response accounting (%0d/%0d)", resp_after-resp_before, issued_reqs);
            passed = passed + 1;
        end else begin
            $display("[FAIL] Conflict traffic response count mismatch: got %0d exp %0d", resp_after-resp_before, issued_reqs);
            failed = failed + 1;
        end

        //====================================================================
        // 测试13: 无冲突访问满带宽（无冲突周期 + 响应数）
        //====================================================================
        $display("\n--- Conflict-Free Full Bandwidth Test ---");

        clean_cycles = 0;
        issued_reqs = 8;
        req_valid <= 0;
        req_mask  <= 0;
        repeat (2) @(posedge clk);
        resp_before = resp_valid_count;

        for (i = 0; i < issued_reqs; i = i + 1) begin
            @(posedge clk);
            req_valid <= 1;
            req_write <= 0;
            set_addresses(14'd2048 + i*NUM_BANKS, 32'd1); // 连续地址 -> 32 bank并行
            req_mask <= {NUM_BANKS{1'b1}};
            #1;
            if (!bank_conflict) clean_cycles = clean_cycles + 1;
        end

        @(posedge clk);
        req_valid <= 0;
        req_mask  <= 0;
        repeat (2) @(posedge clk);
        resp_after = resp_valid_count;

        if (clean_cycles == issued_reqs) begin
            $display("[PASS] Conflict-free cycles counted correctly (%0d)", clean_cycles);
            passed = passed + 1;
        end else begin
            $display("[FAIL] Conflict-free cycle count mismatch: got %0d exp %0d", clean_cycles, issued_reqs);
            failed = failed + 1;
        end

        if ((resp_after - resp_before) == issued_reqs) begin
            $display("[PASS] Conflict-free request bandwidth maintained (%0d/%0d)", resp_after-resp_before, issued_reqs);
            passed = passed + 1;
        end else begin
            $display("[FAIL] Conflict-free response count mismatch: got %0d exp %0d", resp_after-resp_before, issued_reqs);
            failed = failed + 1;
        end

        //====================================================================
        // 测试14: Warp-level atomic add 顺序正确
        //====================================================================
        $display("\n--- Warp-level Atomic ADD Ordering Test ---");

        smem_write(14'd3000, 32'd1, 32'd10, 32'h00000001); // lane0 seed=10
        #10;

        atomic_clear_vectors();
        atomic_set_lane(0, 32'd3000, 32'd1, 32'd0);
        atomic_set_lane(1, 32'd3000, 32'd2, 32'd0);
        atomic_set_lane(2, 32'd3000, 32'd3, 32'd0);
        atomic_set_lane(3, 32'd3000, 32'd4, 32'd0);
        run_warp_atomic_add(32'h0000000F);

        read_lane0_word(14'd3000, lane0_value);
        if (lane0_value == 32'd20 &&
            atomic_old_values[31:0]   == 32'd10 &&
            atomic_old_values[63:32]  == 32'd11 &&
            atomic_old_values[95:64]  == 32'd13 &&
            atomic_old_values[127:96] == 32'd16) begin
            $display("[PASS] Atomic ADD ordering/result correct");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Atomic ADD ordering/result mismatch");
            $display("       Final: %0d (exp 20)", lane0_value);
            failed = failed + 1;
        end

        //====================================================================
        // 测试15: Warp-level atomic max
        //====================================================================
        $display("\n--- Warp-level Atomic MAX Test ---");

        smem_write(14'd3010, 32'd1, 32'd7, 32'h00000001); // lane0 seed=7
        #10;

        atomic_clear_vectors();
        atomic_set_lane(0, 32'd3010, 32'd5, 32'd0);
        atomic_set_lane(1, 32'd3010, 32'd12, 32'd0);
        atomic_set_lane(2, 32'd3010, 32'd9, 32'd0);
        atomic_set_lane(3, 32'd3010, 32'd20, 32'd0);
        run_warp_atomic_max_u(32'h0000000F);

        read_lane0_word(14'd3010, lane0_value);
        if (lane0_value == 32'd20 &&
            atomic_old_values[31:0]   == 32'd7 &&
            atomic_old_values[63:32]  == 32'd7 &&
            atomic_old_values[95:64]  == 32'd12 &&
            atomic_old_values[127:96] == 32'd12) begin
            $display("[PASS] Atomic MAX ordering/result correct");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Atomic MAX ordering/result mismatch");
            $display("       Final: %0d (exp 20)", lane0_value);
            failed = failed + 1;
        end

        //====================================================================
        // 测试16: Warp-level atomic CAS
        //====================================================================
        $display("\n--- Warp-level Atomic CAS Test ---");

        smem_write(14'd3020, 32'd1, 32'd100, 32'h00000001); // lane0 seed=100
        #10;

        atomic_clear_vectors();
        atomic_set_lane(0, 32'd3020, 32'd100, 32'd200); // success
        atomic_set_lane(1, 32'd3020, 32'd100, 32'd300); // fail after lane0 write
        atomic_set_lane(2, 32'd3020, 32'd200, 32'd400); // success
        run_warp_atomic_cas(32'h00000007);

        read_lane0_word(14'd3020, lane0_value);
        if (lane0_value == 32'd400 &&
            atomic_old_values[31:0]  == 32'd100 &&
            atomic_old_values[63:32] == 32'd200 &&
            atomic_old_values[95:64] == 32'd200) begin
            $display("[PASS] Atomic CAS sequencing/result correct");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Atomic CAS sequencing/result mismatch");
            $display("       Final: %0d (exp 400)", lane0_value);
            failed = failed + 1;
        end

        //====================================================================
        // 测试17: 并发lane原子加法无数据破坏（不同地址）
        //====================================================================
        $display("\n--- Concurrent Warp Atomic No-Corruption Test ---");

        smem_write(14'd3200, 32'd1, 32'd1000, 32'h000000FF); // lane0..7 seed
        #10;

        atomic_clear_vectors();
        for (i = 0; i < 8; i = i + 1) begin
            atomic_set_lane(i, 32'd3200 + i, i + 1, 32'd0);
        end
        run_warp_atomic_add(32'h000000FF);

        smem_read(14'd3200, 32'd1, 32'h000000FF);
        @(posedge clk);

        read_ok = 1;
        for (i = 0; i < 8; i = i + 1) begin
            if (resp_rdata[i*32 +: 32] !== (32'd1000 + i + (i + 1))) begin
                read_ok = 0;
                $display("       Lane %0d: got %0d exp %0d",
                         i, resp_rdata[i*32 +: 32], 32'd1000 + i + (i + 1));
            end
        end

        if (read_ok) begin
            $display("[PASS] Concurrent warp atomics keep data integrity");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Concurrent warp atomics data corruption detected");
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

        if (failed > 0) $fatal(1, "Test Failed");
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
        $fatal(1, "Timeout");
    end

endmodule
