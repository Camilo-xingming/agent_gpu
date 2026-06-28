//============================================================================
// RalphGPU - Warp Scheduler Test
// 验证轮询调度策略和Warp状态管理
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_warp_scheduler;

    //------------------------------------------------------------------------
    // 参数
    //------------------------------------------------------------------------
    localparam NUM_WARPS = `WARPS_PER_SM;   // 4
    localparam WARP_ID_W = `WARP_ID_WIDTH;  // 2

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
    // Scheduler DUT 信号
    //------------------------------------------------------------------------
    reg  [NUM_WARPS-1:0] warp_valid;
    reg  [NUM_WARPS-1:0] warp_ready;
    reg  [NUM_WARPS-1:0] warp_waiting;
    wire [WARP_ID_W-1:0] active_warp_id;
    wire                 warp_selected;
    wire [NUM_WARPS-1:0] warp_active_oh;

    //------------------------------------------------------------------------
    // Scheduler DUT 实例化
    //------------------------------------------------------------------------
    warp_scheduler #(
        .NUM_WARPS (NUM_WARPS),
        .WARP_ID_W (WARP_ID_W)
    ) dut_scheduler (
        .clk           (clk),
        .rst_n         (rst_n),
        .warp_valid    (warp_valid),
        .warp_ready    (warp_ready),
        .warp_waiting  (warp_waiting),
        .active_warp_id(active_warp_id),
        .warp_selected (warp_selected),
        .warp_active_oh(warp_active_oh)
    );

    //------------------------------------------------------------------------
    // Warp State DUT 信号
    //------------------------------------------------------------------------
    reg         ws_alloc_en;
    reg  [1:0]  ws_alloc_warp_id;
    reg  [31:0] ws_alloc_pc;
    reg         ws_dealloc_en;
    reg  [1:0]  ws_dealloc_warp_id;
    reg         ws_pc_update_en;
    reg  [1:0]  ws_pc_update_warp;
    reg  [31:0] ws_pc_update_value;
    reg         ws_pc_is_branch;
    reg         ws_sync_start;
    reg  [1:0]  ws_sync_warp_id;
    reg         ws_sync_complete;
    wire [NUM_WARPS-1:0] ws_warp_valid;
    wire [NUM_WARPS-1:0] ws_warp_ready;
    wire [NUM_WARPS-1:0] ws_warp_waiting;
    wire [NUM_WARPS*32-1:0] ws_warp_pc_flat;

    // Unpack the PC array for easier access in testbench
    wire [31:0] ws_warp_pc [0:NUM_WARPS-1];
    genvar gi;
    generate
        for (gi = 0; gi < NUM_WARPS; gi = gi + 1) begin : unpack_pc
            assign ws_warp_pc[gi] = ws_warp_pc_flat[gi*32 +: 32];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Warp State DUT 实例化
    //------------------------------------------------------------------------
    warp_state #(
        .NUM_WARPS (NUM_WARPS)
    ) dut_warp_state (
        .clk              (clk),
        .rst_n            (rst_n),
        .alloc_en         (ws_alloc_en),
        .alloc_warp_id    (ws_alloc_warp_id),
        .alloc_pc         (ws_alloc_pc),
        .dealloc_en       (ws_dealloc_en),
        .dealloc_warp_id  (ws_dealloc_warp_id),
        .pc_update_en     (ws_pc_update_en),
        .pc_update_warp   (ws_pc_update_warp),
        .pc_update_value  (ws_pc_update_value),
        .pc_is_branch     (ws_pc_is_branch),
        .sync_start       (ws_sync_start),
        .sync_warp_id     (ws_sync_warp_id),
        .sync_complete    (ws_sync_complete),
        .warp_valid       (ws_warp_valid),
        .warp_ready       (ws_warp_ready),
        .warp_waiting     (ws_warp_waiting),
        .warp_pc_flat     (ws_warp_pc_flat)
    );

    //------------------------------------------------------------------------
    // 测试变量
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;
    integer i;
    reg [WARP_ID_W-1:0] last_selected;
    // 使用独立计数器代替位向量，避免索引问题
    reg [3:0] sel_cnt_0, sel_cnt_1, sel_cnt_2, sel_cnt_3;
    reg rr_ok;
    integer starve_last_0, starve_last_1, starve_last_2, starve_last_3;
    integer starve_max_gap_0, starve_max_gap_1, starve_max_gap_2, starve_max_gap_3;
    integer starve_seen_0, starve_seen_1, starve_seen_2, starve_seen_3;
    integer bounded_cycles;
    reg starvation_ok;

    //------------------------------------------------------------------------
    // 测试用例
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Warp Scheduler Test");
        $display("============================================================");

        // 初始化
        rst_n = 0;
        warp_valid = 0;
        warp_ready = 0;
        warp_waiting = 0;

        ws_alloc_en = 0;
        ws_alloc_warp_id = 0;
        ws_alloc_pc = 0;
        ws_dealloc_en = 0;
        ws_dealloc_warp_id = 0;
        ws_pc_update_en = 0;
        ws_pc_update_warp = 0;
        ws_pc_update_value = 0;
        ws_pc_is_branch = 0;
        ws_sync_start = 0;
        ws_sync_warp_id = 0;
        ws_sync_complete = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // 调度器测试
        //====================================================================
        $display("\n=== Warp Scheduler Tests ===");

        //====================================================================
        // 测试1: 无有效Warp时不应选择
        //====================================================================
        $display("\n--- No Valid Warps Test ---");

        warp_valid = 4'b0000;
        warp_ready = 4'b1111;
        warp_waiting = 4'b0000;

        @(posedge clk);
        @(posedge clk);

        if (!warp_selected) begin
            $display("[PASS] No selection when no warp valid");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Selected warp when none valid");
            failed = failed + 1;
        end

        //====================================================================
        // 测试2: 单个有效Warp
        //====================================================================
        $display("\n--- Single Valid Warp Test ---");

        warp_valid = 4'b0010;  // Warp 1 valid
        warp_ready = 4'b0010;
        warp_waiting = 4'b0000;

        @(posedge clk);
        @(posedge clk);

        if (warp_selected && active_warp_id === 2'd1 && warp_active_oh === 4'b0010) begin
            $display("[PASS] Single warp (1) correctly selected");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp 1 not selected correctly");
            $display("       selected=%b, id=%d, oh=%b", warp_selected, active_warp_id, warp_active_oh);
            failed = failed + 1;
        end

        //====================================================================
        // 测试3: 轮询调度验证
        //====================================================================
        $display("\n--- Round-Robin Scheduling Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b1111;
        warp_waiting = 4'b0000;
        sel_cnt_0 = 0; sel_cnt_1 = 0; sel_cnt_2 = 0; sel_cnt_3 = 0;

        // 运行多个周期，收集选择结果
        for (i = 0; i < 8; i = i + 1) begin
            @(posedge clk);
            if (warp_selected) begin
                case (active_warp_id)
                    2'd0: sel_cnt_0 = sel_cnt_0 + 1;
                    2'd1: sel_cnt_1 = sel_cnt_1 + 1;
                    2'd2: sel_cnt_2 = sel_cnt_2 + 1;
                    2'd3: sel_cnt_3 = sel_cnt_3 + 1;
                endcase
                $display("       Cycle %0d: Selected Warp %0d", i, active_warp_id);
            end
        end

        // 验证每个Warp都被选中了
        rr_ok = (sel_cnt_0 >= 1) && (sel_cnt_1 >= 1) &&
               (sel_cnt_2 >= 1) && (sel_cnt_3 >= 1);

        if (rr_ok) begin
            $display("[PASS] Round-robin: all warps selected at least once");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Round-robin failed");
            $display("       Counts: W0=%0d, W1=%0d, W2=%0d, W3=%0d",
                     sel_cnt_0, sel_cnt_1, sel_cnt_2, sel_cnt_3);
            failed = failed + 1;
        end

        //====================================================================
        // 测试4: 等待状态排除
        //====================================================================
        $display("\n--- Waiting Exclusion Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b1111;
        warp_waiting = 4'b0101;  // Warp 0和2在等待

        @(posedge clk);
        @(posedge clk);
        @(posedge clk);

        // 收集选择结果
        sel_cnt_0 = 0; sel_cnt_1 = 0; sel_cnt_2 = 0; sel_cnt_3 = 0;
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            if (warp_selected) begin
                case (active_warp_id)
                    2'd0: sel_cnt_0 = sel_cnt_0 + 1;
                    2'd1: sel_cnt_1 = sel_cnt_1 + 1;
                    2'd2: sel_cnt_2 = sel_cnt_2 + 1;
                    2'd3: sel_cnt_3 = sel_cnt_3 + 1;
                endcase
            end
        end

        if (sel_cnt_0 == 0 && sel_cnt_2 == 0 &&
            (sel_cnt_1 > 0 || sel_cnt_3 > 0)) begin
            $display("[PASS] Waiting warps (0,2) excluded from scheduling");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Waiting warps incorrectly scheduled");
            failed = failed + 1;
        end


        //====================================================================
        // 测试5: 长窗口公平性 + 饥饿防护
        //====================================================================
        $display("\n--- Fairness + Starvation Bound Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b1111;
        warp_waiting = 4'b0000;

        starve_last_0 = -NUM_WARPS;
        starve_last_1 = -NUM_WARPS;
        starve_last_2 = -NUM_WARPS;
        starve_last_3 = -NUM_WARPS;
        starve_max_gap_0 = 0;
        starve_max_gap_1 = 0;
        starve_max_gap_2 = 0;
        starve_max_gap_3 = 0;
        starve_seen_0 = 0;
        starve_seen_1 = 0;
        starve_seen_2 = 0;
        starve_seen_3 = 0;

        for (i = 0; i < 64; i = i + 1) begin
            @(posedge clk);
            if (warp_selected) begin
                case (active_warp_id)
                    2'd0: begin
                        if ((i - starve_last_0) > starve_max_gap_0) starve_max_gap_0 = i - starve_last_0;
                        starve_last_0 = i;
                        starve_seen_0 = starve_seen_0 + 1;
                    end
                    2'd1: begin
                        if ((i - starve_last_1) > starve_max_gap_1) starve_max_gap_1 = i - starve_last_1;
                        starve_last_1 = i;
                        starve_seen_1 = starve_seen_1 + 1;
                    end
                    2'd2: begin
                        if ((i - starve_last_2) > starve_max_gap_2) starve_max_gap_2 = i - starve_last_2;
                        starve_last_2 = i;
                        starve_seen_2 = starve_seen_2 + 1;
                    end
                    2'd3: begin
                        if ((i - starve_last_3) > starve_max_gap_3) starve_max_gap_3 = i - starve_last_3;
                        starve_last_3 = i;
                        starve_seen_3 = starve_seen_3 + 1;
                    end
                endcase
            end
        end

        starvation_ok = (starve_seen_0 > 0) && (starve_seen_1 > 0) && (starve_seen_2 > 0) && (starve_seen_3 > 0) &&
                        (starve_max_gap_0 <= (NUM_WARPS * 2)) && (starve_max_gap_1 <= (NUM_WARPS * 2)) &&
                        (starve_max_gap_2 <= (NUM_WARPS * 2)) && (starve_max_gap_3 <= (NUM_WARPS * 2));

        if (starvation_ok) begin
            $display("[PASS] Fairness/starvation bound respected under sustained contention");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Fairness/starvation bound violation");
            $display("       seen: W0=%0d W1=%0d W2=%0d W3=%0d",
                     starve_seen_0, starve_seen_1, starve_seen_2, starve_seen_3);
            $display("       max_gap: W0=%0d W1=%0d W2=%0d W3=%0d",
                     starve_max_gap_0, starve_max_gap_1, starve_max_gap_2, starve_max_gap_3);
            failed = failed + 1;
        end

        //====================================================================
        // 测试6: 解阻塞后应在有界周期内被调度（防饥饿）
        //====================================================================
        $display("\n--- Unblock Scheduling Latency Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b0111;     // Warp3先不可发射
        warp_waiting = 4'b0000;

        // 先让低编号warp形成连续流
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
        end

        // Warp3解阻塞，必须在有界周期内得到调度
        warp_ready[3] = 1'b1;
        bounded_cycles = 0;
        while (bounded_cycles < (NUM_WARPS + 2) && !(warp_selected && active_warp_id == 2'd3)) begin
            @(posedge clk);
            bounded_cycles = bounded_cycles + 1;
        end

        if (warp_selected && active_warp_id == 2'd3) begin
            $display("[PASS] Unblocked warp scheduled within bounded cycles");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Unblocked warp starved too long");
            failed = failed + 1;
        end

        //====================================================================
        // 测试7: stalled warp 跳过 + 恢复后可调度
        //====================================================================
        $display("\n--- Stall Skip And Recovery Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b1111;
        warp_waiting = 4'b1010;   // Warp1/Warp3 stalled
        sel_cnt_0 = 0; sel_cnt_1 = 0; sel_cnt_2 = 0; sel_cnt_3 = 0;

        for (i = 0; i < 8; i = i + 1) begin
            @(posedge clk);
            if (warp_selected) begin
                case (active_warp_id)
                    2'd0: sel_cnt_0 = sel_cnt_0 + 1;
                    2'd1: sel_cnt_1 = sel_cnt_1 + 1;
                    2'd2: sel_cnt_2 = sel_cnt_2 + 1;
                    2'd3: sel_cnt_3 = sel_cnt_3 + 1;
                endcase
            end
        end

        if (sel_cnt_1 == 0 && sel_cnt_3 == 0 && sel_cnt_0 > 0 && sel_cnt_2 > 0) begin
            $display("[PASS] Scheduler skips stalled warps");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Stalled warp was incorrectly scheduled");
            failed = failed + 1;
        end

        // 释放Warp1，验证可在有界周期内恢复调度
        warp_waiting[1] = 1'b0;
        bounded_cycles = 0;
        while (bounded_cycles < (NUM_WARPS + 2) && !(warp_selected && active_warp_id == 2'd1)) begin
            @(posedge clk);
            bounded_cycles = bounded_cycles + 1;
        end

        if (warp_selected && active_warp_id == 2'd1) begin
            $display("[PASS] Recovered warp scheduled after stall release");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Recovered warp not scheduled in time");
            failed = failed + 1;
        end

        //====================================================================
        // 测试8: 全部stalled，再单warp解除stalled
        //====================================================================
        $display("\n--- All Stalled Then One Unblocks Test ---");

        warp_valid = 4'b1111;
        warp_ready = 4'b1111;
        warp_waiting = 4'b1111;

        @(posedge clk);
        @(posedge clk);

        if (!warp_selected) begin
            $display("[PASS] No warp selected when all stalled");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp selected while all are stalled");
            failed = failed + 1;
        end

        warp_waiting = 4'b1011;   // only warp2 unblocked
        bounded_cycles = 0;
        while (bounded_cycles < 3 && !(warp_selected && active_warp_id == 2'd2)) begin
            @(posedge clk);
            bounded_cycles = bounded_cycles + 1;
        end

        if (warp_selected && active_warp_id == 2'd2) begin
            $display("[PASS] Single unblocked warp selected promptly");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Single unblocked warp not selected");
            failed = failed + 1;
        end
        //====================================================================
        // Warp状态管理测试
        //====================================================================
        $display("\n=== Warp State Tests ===");

        //====================================================================
        // 测试5: Warp分配
        //====================================================================
        $display("\n--- Warp Allocation Test ---");

        // 分配Warp 0
        @(posedge clk);
        ws_alloc_en <= 1;
        ws_alloc_warp_id <= 2'd0;
        ws_alloc_pc <= 32'h1000;
        @(posedge clk);
        ws_alloc_en <= 0;

        @(posedge clk);

        if (ws_warp_valid[0] && ws_warp_ready[0] && ws_warp_pc[0] === 32'h1000) begin
            $display("[PASS] Warp 0 allocated with PC=0x1000");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp 0 allocation error");
            $display("       valid=%b, ready=%b, PC=0x%08X",
                     ws_warp_valid[0], ws_warp_ready[0], ws_warp_pc[0]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试6: 非分支PC更新（+4）
        //====================================================================
        $display("\n--- Sequential PC Update Test (+4) ---");

        @(posedge clk);
        ws_pc_update_en <= 1;
        ws_pc_update_warp <= 2'd0;
        ws_pc_update_value <= 32'hDEAD_BEEF;
        ws_pc_is_branch <= 0;
        @(posedge clk);
        ws_pc_update_en <= 0;

        @(posedge clk);

        if (ws_warp_pc[0] === 32'h1004) begin
            $display("[PASS] PC updated to 0x1004");
            passed = passed + 1;
        end else begin
            $display("[FAIL] PC update error: got 0x%08X", ws_warp_pc[0]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试6b: 分支PC更新（使用目标地址）
        //====================================================================
        $display("\n--- Branch PC Update Test (target redirect) ---");

        @(posedge clk);
        ws_pc_update_en <= 1;
        ws_pc_update_warp <= 2'd0;
        ws_pc_update_value <= 32'h1400;
        ws_pc_is_branch <= 1;
        @(posedge clk);
        ws_pc_update_en <= 0;
        ws_pc_is_branch <= 0;

        @(posedge clk);

        if (ws_warp_pc[0] === 32'h1400) begin
            $display("[PASS] Branch update redirected PC to 0x1400");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Branch PC update error: got 0x%08X", ws_warp_pc[0]);
            failed = failed + 1;
        end

        //====================================================================
        // 测试7: 同步等待
        //====================================================================
        $display("\n--- Sync Wait Test ---");

        // 分配Warp 1
        @(posedge clk);
        ws_alloc_en <= 1;
        ws_alloc_warp_id <= 2'd1;
        ws_alloc_pc <= 32'h2000;
        @(posedge clk);
        ws_alloc_en <= 0;

        // Warp 0进入同步等待
        @(posedge clk);
        ws_sync_start <= 1;
        ws_sync_warp_id <= 2'd0;
        @(posedge clk);
        ws_sync_start <= 0;

        @(posedge clk);

        if (ws_warp_waiting[0] && !ws_warp_ready[0]) begin
            $display("[PASS] Warp 0 entered sync waiting state");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp 0 not in waiting state");
            failed = failed + 1;
        end

        //====================================================================
        // 测试8: 同步完成
        //====================================================================
        $display("\n--- Sync Complete Test ---");

        @(posedge clk);
        ws_sync_complete <= 1;
        @(posedge clk);
        ws_sync_complete <= 0;

        @(posedge clk);

        if (!ws_warp_waiting[0] && ws_warp_ready[0]) begin
            $display("[PASS] Warp 0 exited waiting state after sync complete");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp 0 still in waiting state");
            failed = failed + 1;
        end

        //====================================================================
        // 测试9: Warp释放
        //====================================================================
        $display("\n--- Warp Deallocation Test ---");

        @(posedge clk);
        ws_dealloc_en <= 1;
        ws_dealloc_warp_id <= 2'd0;
        @(posedge clk);
        ws_dealloc_en <= 0;

        @(posedge clk);

        if (!ws_warp_valid[0] && !ws_warp_ready[0]) begin
            $display("[PASS] Warp 0 deallocated");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Warp 0 not deallocated");
            failed = failed + 1;
        end

        //====================================================================
        // 测试10: 多Warp同时分配
        //====================================================================
        $display("\n--- Multi-Warp Allocation Test ---");

        // 分配所有4个Warp
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            ws_alloc_en <= 1;
            ws_alloc_warp_id <= i[1:0];
            ws_alloc_pc <= 32'h3000 + i*32'h100;
            @(posedge clk);
            ws_alloc_en <= 0;
        end

        @(posedge clk);

        if (ws_warp_valid === 4'b1111) begin
            $display("[PASS] All 4 warps allocated");
            passed = passed + 1;
        end else begin
            $display("[FAIL] Not all warps allocated: valid=%b", ws_warp_valid);
            failed = failed + 1;
        end

        //====================================================================
        // 测试总结
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("Warp Scheduler Test Summary: %0d PASSED, %0d FAILED", passed, failed);
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
        $dumpfile("tb_warp_scheduler.vcd");
        $dumpvars(0, tb_warp_scheduler);
    end

    initial begin
        #50000;
        $display("ERROR: Timeout!");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
