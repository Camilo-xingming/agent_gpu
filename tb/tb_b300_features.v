//============================================================================
// RalphGPU - B300 Gap Features Comprehensive Testbench
// Tests all implemented B300/Blackwell architecture features:
// - TMA/Async Memory (cp.async, st.async, multimem, mbarrier)
// - Tensor Operations (WGMMA, FP6)
// - Synchronization (bar.warp.sync, barrier.cluster)
// - Cache Policy (createpolicy, applypriority, discard)
// - Stack/Debug (alloca, stacksave, brkpt, nanosleep)
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_b300_features;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter CLK_PERIOD = 10;

    //------------------------------------------------------------------------
    // Test Control
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    integer pass_count;
    integer fail_count;
    integer test_num;

    //------------------------------------------------------------------------
    // DUT Signals - Decoder
    //------------------------------------------------------------------------
    reg  [31:0] instruction;
    reg         valid_in;
    wire        valid_out;
    wire [5:0]  opcode;
    wire [4:0]  rd, ra, rb, rc;
    wire [5:0]  func;
    wire [15:0] imm16;
    wire [20:0] imm21;

    // Control signals
    wire use_imm, alu_op, mul_op, div_op;
    wire mem_read, mem_write, mem_shared;
    wire branch_op, sync_op, special_reg, exit_op;
    wire reg_write, pred_write;
    wire [2:0] pred_addr;
    wire fp32_op, fp32_special, fp64_op, fp16_op, cvt_op;
    wire mem_param, mem_const, mem_local, mem_vector;
    wire [1:0] vec_size;
    wire atomic_op, reduce_op;
    wire shfl_op, vote_op, redux_op;
    wire wmma_load, wmma_store, wmma_mma, mma_op;
    wire call_op, membar_op;
    wire video_op;
    wire tex_op, txq_op, surf_ld, surf_st, surf_red;
    wire cpasync_op, prefetch_op;
    wire wgmma_load, wgmma_store, wgmma_mma;
    wire [2:0] cache_hint;
    wire mbarrier_op;
    wire bar_warp_sync;
    wire cache_policy_op;
    wire stack_op, debug_op, misc_op;
    wire st_async_op, multimem_op;
    wire barrier_cluster_op;

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    //------------------------------------------------------------------------
    // DUT Instantiation - Decoder
    //------------------------------------------------------------------------
    decoder u_decoder (
        .clk(clk),
        .rst_n(rst_n),
        .instruction(instruction),
        .valid_in(valid_in),
        .valid_out(valid_out),
        .opcode(opcode),
        .rd(rd), .ra(ra), .rb(rb), .rc(rc),
        .func(func),
        .imm16(imm16),
        .imm21(imm21),
        .use_imm(use_imm),
        .alu_op(alu_op),
        .mul_op(mul_op),
        .div_op(div_op),
        .mem_read(mem_read),
        .mem_write(mem_write),
        .mem_shared(mem_shared),
        .branch_op(branch_op),
        .sync_op(sync_op),
        .special_reg(special_reg),
        .exit_op(exit_op),
        .reg_write(reg_write),
        .pred_write(pred_write),
        .pred_addr(pred_addr),
        .fp32_op(fp32_op),
        .fp32_special(fp32_special),
        .fp64_op(fp64_op),
        .fp16_op(fp16_op),
        .cvt_op(cvt_op),
        .mem_param(mem_param),
        .mem_const(mem_const),
        .mem_local(mem_local),
        .mem_vector(mem_vector),
        .vec_size(vec_size),
        .atomic_op(atomic_op),
        .reduce_op(reduce_op),
        .shfl_op(shfl_op),
        .vote_op(vote_op),
        .redux_op(redux_op),
        .wmma_load(wmma_load),
        .wmma_store(wmma_store),
        .wmma_mma(wmma_mma),
        .mma_op(mma_op),
        .call_op(call_op),
        .membar_op(membar_op),
        .video_op(video_op),
        .tex_op(tex_op),
        .txq_op(txq_op),
        .surf_ld(surf_ld),
        .surf_st(surf_st),
        .surf_red(surf_red),
        .cpasync_op(cpasync_op),
        .prefetch_op(prefetch_op),
        .wgmma_load(wgmma_load),
        .wgmma_store(wgmma_store),
        .wgmma_mma(wgmma_mma),
        .cache_hint(cache_hint),
        .mbarrier_op(mbarrier_op),
        .bar_warp_sync(bar_warp_sync),
        .cache_policy_op(cache_policy_op),
        .stack_op(stack_op),
        .debug_op(debug_op),
        .misc_op(misc_op),
        .st_async_op(st_async_op),
        .multimem_op(multimem_op),
        .barrier_cluster_op(barrier_cluster_op)
    );

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 0;
            valid_in = 0;
            instruction = 32'h0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(2) @(posedge clk);
        end
    endtask

    task decode_instruction;
        input [31:0] inst;
        begin
            @(posedge clk);
            instruction = inst;
            valid_in = 1;
            @(posedge clk);  // Decoder latches on this edge and outputs appear
            #1;              // Small delay for output to settle after clock edge
            // Check outputs HERE while valid_out is still high
            // Don't set valid_in=0 until after we've checked
        end
    endtask

    task clear_valid;
        begin
            valid_in = 0;
            @(posedge clk);  // Clear valid_out on next cycle
        end
    endtask

    task check_result;
        input [255:0] test_name;
        input expected;
        input actual;
        begin
            test_num = test_num + 1;
            if (expected == actual) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: %s", test_num, test_name);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %s - Expected %b, Got %b", test_num, test_name, expected, actual);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Build instruction helper
    //------------------------------------------------------------------------
    function [31:0] build_inst;
        input [5:0] op;
        input [4:0] dst;
        input [4:0] src_a;
        input [4:0] src_b;
        input [5:0] fn;
        begin
            build_inst = {op, dst, src_a, src_b, 5'b0, fn};
        end
    endfunction

    //------------------------------------------------------------------------
    // Main Test
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU B300 Features Comprehensive Test");
        $display("============================================================");

        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        reset_dut();

        //====================================================================
        // Phase 1: TMA/Async Memory Tests
        //====================================================================
        $display("\n--- Phase 1: TMA/Async Memory Operations ---");

        // Test 1.1: cp.async operation
        decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
        check_result("cp.async.ca decode", 1'b1, cpasync_op);

        // Test 1.2: cp.async commit
        decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
        check_result("cp.async.commit decode", 1'b1, cpasync_op);

        // Test 1.3: cp.async wait
        decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_WAIT));
        check_result("cp.async.wait decode", 1'b1, cpasync_op);

        // Test 1.4: st.async.global
        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL));
        check_result("st.async.global decode", 1'b1, st_async_op);
        check_result("st.async.global mem_write", 1'b1, mem_write);

        // Test 1.5: st.async.shared
        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_SHARED));
        check_result("st.async.shared decode", 1'b1, st_async_op);
        check_result("st.async.shared mem_shared", 1'b1, mem_shared);

        // Test 1.6: st.async.commit
        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd0, 5'd0, `ST_ASYNC_COMMIT));
        check_result("st.async.commit decode", 1'b1, st_async_op);

        // Test 1.7: st.async.wait
        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd0, 5'd0, `ST_ASYNC_WAIT));
        check_result("st.async.wait decode", 1'b1, st_async_op);

        // Test 1.8: multimem.ld
        decode_instruction(build_inst(`OP_MULTIMEM, 5'd1, 5'd2, 5'd0, `MULTIMEM_LD));
        check_result("multimem.ld decode", 1'b1, multimem_op);
        check_result("multimem.ld mem_read", 1'b1, mem_read);
        check_result("multimem.ld reg_write", 1'b1, reg_write);

        // Test 1.9: multimem.st
        decode_instruction(build_inst(`OP_MULTIMEM, 5'd0, 5'd1, 5'd2, `MULTIMEM_ST));
        check_result("multimem.st decode", 1'b1, multimem_op);
        check_result("multimem.st mem_write", 1'b1, mem_write);

        // Test 1.10: multimem.red
        decode_instruction(build_inst(`OP_MULTIMEM, 5'd0, 5'd1, 5'd2, `MULTIMEM_RED));
        check_result("multimem.red decode", 1'b1, multimem_op);

        // Test 1.11: mbarrier.init
        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INIT));
        check_result("mbarrier.init decode", 1'b1, mbarrier_op);

        // Test 1.12: mbarrier.arrive
        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
        check_result("mbarrier.arrive decode", 1'b1, mbarrier_op);

        // Test 1.13: mbarrier.test_wait
        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_TEST_WAIT));
        check_result("mbarrier.test_wait decode", 1'b1, mbarrier_op);

        // Test 1.14: mbarrier.try_wait
        decode_instruction(build_inst(`OP_MBARRIER, 5'd1, 5'd2, 5'd0, `MBAR_TRY_WAIT));
        check_result("mbarrier.try_wait decode", 1'b1, mbarrier_op);

        //====================================================================
        // Phase 2: Tensor Operations Tests
        //====================================================================
        $display("\n--- Phase 2: Tensor Operations ---");

        // Test 2.1: WGMMA load
        decode_instruction(build_inst(`OP_WGMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00));
        check_result("wgmma.load decode", 1'b1, wgmma_load);

        // Test 2.2: WGMMA store
        decode_instruction(build_inst(`OP_WGMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00));
        check_result("wgmma.store decode", 1'b1, wgmma_store);

        // Test 2.3: WGMMA mma
        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
        check_result("wgmma.mma decode", 1'b1, wgmma_mma);

        // Test 2.4: WGMMA fence (uses MMA opcode with fence func)
        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd0, 5'd0, `WGMMA_FENCE));
        check_result("wgmma.fence decode", 1'b1, valid_out);

        // Test 2.5: WGMMA commit
        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd0, 5'd0, `WGMMA_COMMIT_GROUP));
        check_result("wgmma.commit decode", 1'b1, valid_out);

        // Test 2.6: WGMMA wait
        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd0, 5'd0, `WGMMA_WAIT_GROUP));
        check_result("wgmma.wait decode", 1'b1, valid_out);

        //====================================================================
        // Phase 3: Synchronization Tests
        //====================================================================
        $display("\n--- Phase 3: Synchronization Operations ---");

        // Test 3.1: bar.warp.sync
        decode_instruction(build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00));
        check_result("bar.warp.sync decode", 1'b1, bar_warp_sync);
        check_result("bar.warp.sync sync_op", 1'b1, sync_op);

        // Test 3.2: barrier.cluster.arrive
        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_ARRIVE));
        check_result("barrier.cluster.arrive decode", 1'b1, barrier_cluster_op);
        check_result("barrier.cluster.arrive sync_op", 1'b1, sync_op);

        // Test 3.3: barrier.cluster.wait
        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_WAIT));
        check_result("barrier.cluster.wait decode", 1'b1, barrier_cluster_op);

        // Test 3.4: barrier.cluster.sync
        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
        check_result("barrier.cluster.sync decode", 1'b1, barrier_cluster_op);

        // Test 3.5: barrier.cluster.init
        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd1, 5'd0, `CLUSTER_BARRIER_INIT));
        check_result("barrier.cluster.init decode", 1'b1, barrier_cluster_op);

        //====================================================================
        // Phase 4: Cache Policy Tests
        //====================================================================
        $display("\n--- Phase 4: Cache Policy Operations ---");

        // Test 4.1: createpolicy
        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd1, 5'd0, 5'd0, `CACHE_CREATEPOLICY));
        check_result("createpolicy decode", 1'b1, cache_policy_op);

        // Test 4.2: applypriority
        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd0, 5'd1, 5'd0, `CACHE_APPLYPRIORITY));
        check_result("applypriority decode", 1'b1, cache_policy_op);

        // Test 4.3: discard
        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd0, 5'd1, 5'd0, `CACHE_DISCARD));
        check_result("discard decode", 1'b1, cache_policy_op);

        //====================================================================
        // Phase 5: Stack/Debug Operations Tests
        //====================================================================
        $display("\n--- Phase 5: Stack/Debug Operations ---");

        // Test 5.1: alloca
        decode_instruction(build_inst(`OP_STACK, 5'd1, 5'd2, 5'd0, `STACK_ALLOCA));
        check_result("alloca decode", 1'b1, stack_op);

        // Test 5.2: stacksave
        decode_instruction(build_inst(`OP_STACK, 5'd1, 5'd0, 5'd0, `STACK_SAVE));
        check_result("stacksave decode", 1'b1, stack_op);

        // Test 5.3: stackrestore
        decode_instruction(build_inst(`OP_STACK, 5'd0, 5'd1, 5'd0, `STACK_RESTORE));
        check_result("stackrestore decode", 1'b1, stack_op);

        // Test 5.4: brkpt
        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd0, 5'd0, `DEBUG_BRKPT));
        check_result("brkpt decode", 1'b1, debug_op);

        // Test 5.5: trap
        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd0, 5'd0, `DEBUG_TRAP));
        check_result("trap decode", 1'b1, debug_op);

        // Test 5.6: pmevent
        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd1, 5'd0, `DEBUG_PMEVENT));
        check_result("pmevent decode", 1'b1, debug_op);

        // Test 5.7: nanosleep
        decode_instruction(build_inst(`OP_MISC, 5'd0, 5'd1, 5'd0, `MISC_NANOSLEEP));
        check_result("nanosleep decode", 1'b1, misc_op);

        // Test 5.8: setmaxnreg
        decode_instruction(build_inst(`OP_MISC, 5'd0, 5'd1, 5'd0, `MISC_SETMAXNREG));
        check_result("setmaxnreg decode", 1'b1, misc_op);

        //====================================================================
        // Phase 6: Texture/Video Operations Tests
        //====================================================================
        $display("\n--- Phase 6: Texture/Video Operations ---");

        // Test 6.1: texture sample
        decode_instruction(build_inst(`OP_TEX, 5'd1, 5'd2, 5'd3, 6'h00));
        check_result("tex decode", 1'b1, tex_op);

        // Test 6.2: texture query
        decode_instruction(build_inst(`OP_TXQ, 5'd1, 5'd2, 5'd0, 6'h00));
        check_result("txq decode", 1'b1, txq_op);

        // Test 6.3: surface load
        decode_instruction(build_inst(`OP_SULD, 5'd1, 5'd2, 5'd3, 6'h00));
        check_result("suld decode", 1'b1, surf_ld);

        // Test 6.4: surface store
        decode_instruction(build_inst(`OP_SUST, 5'd0, 5'd1, 5'd2, 6'h00));
        check_result("sust decode", 1'b1, surf_st);

        // Test 6.5: surface reduction
        decode_instruction(build_inst(`OP_SURED, 5'd0, 5'd1, 5'd2, 6'h00));
        check_result("sured decode", 1'b1, surf_red);

        // Test 6.6: video operation
        decode_instruction(build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, `VIDEO_VADD));
        check_result("video decode", 1'b1, video_op);

        //====================================================================
        // Phase 7: Performance Benchmark Tests
        //====================================================================
        $display("\n--- Phase 7: Performance Benchmarks ---");

        // Performance Test 7.1: Decoder throughput - decode 100 instructions
        begin : perf_decoder_throughput
            integer i;
            integer start_time, end_time;
            real throughput;
            reg [31:0] test_insts [0:9];

            // Pre-build a variety of instructions
            test_insts[0] = build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA);
            test_insts[1] = build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16);
            test_insts[2] = build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE);
            test_insts[3] = build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00);
            test_insts[4] = build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC);
            test_insts[5] = build_inst(`OP_TEX, 5'd1, 5'd2, 5'd3, 6'h00);
            test_insts[6] = build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, `VIDEO_VADD);
            test_insts[7] = build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL);
            test_insts[8] = build_inst(`OP_MULTIMEM, 5'd1, 5'd2, 5'd0, `MULTIMEM_LD);
            test_insts[9] = build_inst(`OP_STACK, 5'd1, 5'd2, 5'd0, `STACK_ALLOCA);

            start_time = $time;

            // Decode 100 instructions (10 rounds x 10 instructions)
            for (i = 0; i < 100; i = i + 1) begin
                @(posedge clk);
                instruction = test_insts[i % 10];
                valid_in = 1;
                @(posedge clk);
                #1;
            end
            valid_in = 0;

            end_time = $time;

            throughput = 100.0 / ((end_time - start_time) / 1000.0);  // Instructions per ns
            $display("[PERF] Decoder throughput: 100 instructions in %0d ps (%.2f instr/ns)",
                     end_time - start_time, throughput);

            // Check throughput is >= 1 instruction per 2 clock cycles (0.1 instr/ns at 10ns period)
            test_num = test_num + 1;
            if (throughput >= 0.04) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: Decoder throughput >= 0.04 instr/ns", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: Decoder throughput too low (%.2f)", test_num, throughput);
            end
        end

        // Performance Test 7.2: Instruction mix latency
        begin : perf_instruction_mix
            integer i;
            integer mix_errors;
            reg [31:0] mixed_insts [0:19];

            // Create a realistic instruction mix
            mixed_insts[0]  = build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA);
            mixed_insts[1]  = build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT);
            mixed_insts[2]  = build_inst(`OP_WGMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00);
            mixed_insts[3]  = build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16);
            mixed_insts[4]  = build_inst(`OP_WGMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00);
            mixed_insts[5]  = build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INIT);
            mixed_insts[6]  = build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE);
            mixed_insts[7]  = build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00);
            mixed_insts[8]  = build_inst(`OP_TEX, 5'd1, 5'd2, 5'd3, 6'h00);
            mixed_insts[9]  = build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, `VIDEO_VADD);
            mixed_insts[10] = build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL);
            mixed_insts[11] = build_inst(`OP_MULTIMEM, 5'd1, 5'd2, 5'd0, `MULTIMEM_LD);
            mixed_insts[12] = build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC);
            mixed_insts[13] = build_inst(`OP_CACHE_POLICY, 5'd1, 5'd0, 5'd0, `CACHE_CREATEPOLICY);
            mixed_insts[14] = build_inst(`OP_STACK, 5'd1, 5'd2, 5'd0, `STACK_ALLOCA);
            mixed_insts[15] = build_inst(`OP_DEBUG, 5'd0, 5'd0, 5'd0, `DEBUG_BRKPT);
            mixed_insts[16] = build_inst(`OP_MISC, 5'd0, 5'd1, 5'd0, `MISC_NANOSLEEP);
            mixed_insts[17] = build_inst(`OP_SULD, 5'd1, 5'd2, 5'd3, 6'h00);
            mixed_insts[18] = build_inst(`OP_SUST, 5'd0, 5'd1, 5'd2, 6'h00);
            mixed_insts[19] = build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_WAIT);

            mix_errors = 0;

            // Decode each and verify valid_out
            for (i = 0; i < 20; i = i + 1) begin
                @(posedge clk);
                instruction = mixed_insts[i];
                valid_in = 1;
                @(posedge clk);
                #1;
                if (!valid_out) begin
                    mix_errors = mix_errors + 1;
                    $display("[WARN] Instruction mix index %0d failed valid_out", i);
                end
            end
            valid_in = 0;

            test_num = test_num + 1;
            if (mix_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: All 20 mixed instructions decoded", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d mixed instructions failed", test_num, mix_errors);
            end
        end

        // Performance Test 7.3: Back-to-back same opcode (stress test)
        begin : perf_stress_same_opcode
            integer i;
            integer stress_errors;
            reg [31:0] stress_inst;

            stress_inst = build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16);
            stress_errors = 0;

            // Decode same instruction 50 times back-to-back
            for (i = 0; i < 50; i = i + 1) begin
                @(posedge clk);
                instruction = stress_inst;
                valid_in = 1;
                @(posedge clk);
                #1;
                if (!wgmma_mma) stress_errors = stress_errors + 1;
            end
            valid_in = 0;

            test_num = test_num + 1;
            if (stress_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: 50x WGMMA stress test", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d WGMMA decodes failed", test_num, stress_errors);
            end
        end

        // Performance Test 7.4: Async/Sync interleave pattern
        begin : perf_async_sync_interleave
            integer i;
            integer interleave_errors;

            interleave_errors = 0;

            // Test typical async copy followed by sync pattern
            for (i = 0; i < 10; i = i + 1) begin
                // cp.async
                decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
                if (!cpasync_op) interleave_errors = interleave_errors + 1;

                // commit
                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) interleave_errors = interleave_errors + 1;

                // mbarrier arrive
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                if (!mbarrier_op) interleave_errors = interleave_errors + 1;

                // wait
                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_WAIT));
                if (!cpasync_op) interleave_errors = interleave_errors + 1;
            end

            test_num = test_num + 1;
            if (interleave_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: Async/sync interleave pattern (40 ops)", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d interleave errors", test_num, interleave_errors);
            end
        end

        // Performance Test 7.5: TMA + WGMMA typical kernel pattern
        begin : perf_tma_wgmma_pattern
            integer i;
            integer kernel_errors;

            kernel_errors = 0;

            // Simulate typical tensor kernel pattern
            for (i = 0; i < 5; i = i + 1) begin
                // TMA load phase
                decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
                if (!cpasync_op) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) kernel_errors = kernel_errors + 1;

                // Barrier
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                if (!mbarrier_op) kernel_errors = kernel_errors + 1;

                // WGMMA compute phase
                decode_instruction(build_inst(`OP_WGMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00));
                if (!wgmma_load) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
                if (!wgmma_mma) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
                if (!wgmma_mma) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_WGMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00));
                if (!wgmma_store) kernel_errors = kernel_errors + 1;

                // Sync before next iteration
                decode_instruction(build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00));
                if (!bar_warp_sync) kernel_errors = kernel_errors + 1;
            end

            test_num = test_num + 1;
            if (kernel_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: TMA+WGMMA kernel pattern (40 ops)", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d kernel pattern errors", test_num, kernel_errors);
            end
        end

        //====================================================================
        // Phase 8: Combination Tests
        //====================================================================
        $display("\n--- Phase 8: Combination Tests ---");

        // Combination Test 8.1: All memory operations sequence
        begin : combo_memory_ops
            integer combo_errors;
            combo_errors = 0;

            // Global load (existing)
            decode_instruction(build_inst(`OP_LD_GLOBAL, 5'd1, 5'd2, 5'd0, 6'h00));
            if (!mem_read) combo_errors = combo_errors + 1;

            // Async copy
            decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
            if (!cpasync_op) combo_errors = combo_errors + 1;

            // Async store
            decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL));
            if (!st_async_op || !mem_write) combo_errors = combo_errors + 1;

            // Multimem load
            decode_instruction(build_inst(`OP_MULTIMEM, 5'd1, 5'd2, 5'd0, `MULTIMEM_LD));
            if (!multimem_op || !mem_read) combo_errors = combo_errors + 1;

            // Texture load
            decode_instruction(build_inst(`OP_TEX, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!tex_op || !mem_read) combo_errors = combo_errors + 1;

            // Surface load
            decode_instruction(build_inst(`OP_SULD, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!surf_ld || !mem_read) combo_errors = combo_errors + 1;

            test_num = test_num + 1;
            if (combo_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: All memory operations sequence", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d memory op combo errors", test_num, combo_errors);
            end
        end

        // Combination Test 8.2: All synchronization operations
        begin : combo_sync_ops
            integer combo_errors;
            combo_errors = 0;

            // Block barrier
            decode_instruction(build_inst(`OP_BAR_SYNC, 5'd0, 5'd0, 5'd0, 6'h00));
            if (!sync_op) combo_errors = combo_errors + 1;

            // Warp barrier
            decode_instruction(build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00));
            if (!bar_warp_sync || !sync_op) combo_errors = combo_errors + 1;

            // Cluster barrier
            decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
            if (!barrier_cluster_op || !sync_op) combo_errors = combo_errors + 1;

            // mbarrier
            decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
            if (!mbarrier_op) combo_errors = combo_errors + 1;

            // Memory barrier
            decode_instruction(build_inst(`OP_MEMBAR, 5'd0, 5'd0, 5'd0, 6'h00));
            if (!membar_op || !sync_op) combo_errors = combo_errors + 1;

            test_num = test_num + 1;
            if (combo_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: All synchronization operations", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d sync combo errors", test_num, combo_errors);
            end
        end

        // Combination Test 8.3: All tensor operations
        begin : combo_tensor_ops
            integer combo_errors;
            combo_errors = 0;

            // WMMA load
            decode_instruction(build_inst(`OP_WMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!wmma_load || !mem_read) combo_errors = combo_errors + 1;

            // WMMA mma
            decode_instruction(build_inst(`OP_WMMA_MMA, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!wmma_mma || !reg_write) combo_errors = combo_errors + 1;

            // WMMA store
            decode_instruction(build_inst(`OP_WMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00));
            if (!wmma_store || !mem_write) combo_errors = combo_errors + 1;

            // WGMMA load
            decode_instruction(build_inst(`OP_WGMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!wgmma_load || !mem_read) combo_errors = combo_errors + 1;

            // WGMMA mma
            decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
            if (!wgmma_mma || !reg_write) combo_errors = combo_errors + 1;

            // WGMMA store
            decode_instruction(build_inst(`OP_WGMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00));
            if (!wgmma_store || !mem_write) combo_errors = combo_errors + 1;

            // MMA
            decode_instruction(build_inst(`OP_MMA, 5'd1, 5'd2, 5'd3, 6'h00));
            if (!mma_op || !reg_write) combo_errors = combo_errors + 1;

            test_num = test_num + 1;
            if (combo_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: All tensor operations", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d tensor combo errors", test_num, combo_errors);
            end
        end

        // Combination Test 8.4: Complete B300 kernel simulation
        begin : combo_b300_kernel
            integer i;
            integer kernel_errors;
            kernel_errors = 0;

            for (i = 0; i < 3; i = i + 1) begin
                // Initialize barriers
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INIT));
                if (!mbarrier_op) kernel_errors = kernel_errors + 1;

                // Cluster barrier init
                decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd1, 5'd0, `CLUSTER_BARRIER_INIT));
                if (!barrier_cluster_op) kernel_errors = kernel_errors + 1;

                // TMA loads
                decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
                if (!cpasync_op) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) kernel_errors = kernel_errors + 1;

                // Wait for data
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                if (!mbarrier_op) kernel_errors = kernel_errors + 1;

                // Tensor compute
                decode_instruction(build_inst(`OP_WGMMA_LOAD, 5'd1, 5'd2, 5'd3, 6'h00));
                if (!wgmma_load) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
                if (!wgmma_mma) kernel_errors = kernel_errors + 1;

                decode_instruction(build_inst(`OP_WGMMA_STORE, 5'd0, 5'd1, 5'd2, 6'h00));
                if (!wgmma_store) kernel_errors = kernel_errors + 1;

                // Async store results
                decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL));
                if (!st_async_op) kernel_errors = kernel_errors + 1;

                // Cluster sync
                decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
                if (!barrier_cluster_op) kernel_errors = kernel_errors + 1;
            end

            test_num = test_num + 1;
            if (kernel_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: Complete B300 kernel simulation (30 ops)", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: %0d B300 kernel errors", test_num, kernel_errors);
            end
        end

        //====================================================================
        // Phase 9: Extended Complex Test Cases (100 additional tests)
        //====================================================================
        $display("\n--- Phase 9: Extended Complex Test Cases ---");

        //--------------------------------------------------------------------
        // 9.1: Register Field Boundary Tests (10 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.1: Register Field Boundary Tests ---");

        // Test all register field values
        begin : reg_boundary_tests
            integer err_cnt;
            err_cnt = 0;

            // Test rd=0 (minimum)
            decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_CA));
            if (rd != 5'd0) err_cnt = err_cnt + 1;
            test_num = test_num + 1;
            if (rd == 5'd0) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: rd=0 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: rd=0 boundary", test_num); end

            // Test rd=31 (maximum)
            decode_instruction(build_inst(`OP_CPASYNC, 5'd31, 5'd0, 5'd0, `CPASYNC_CA));
            test_num = test_num + 1;
            if (rd == 5'd31) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: rd=31 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: rd=31 boundary", test_num); end

            // Test ra=0 and ra=31
            decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd0, 5'd0, `WGMMA_M64N8K16));
            test_num = test_num + 1;
            if (ra == 5'd0) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: ra=0 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: ra=0 boundary", test_num); end

            decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd31, 5'd0, `WGMMA_M64N8K16));
            test_num = test_num + 1;
            if (ra == 5'd31) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: ra=31 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: ra=31 boundary", test_num); end

            // Test rb=0 and rb=31
            decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INIT));
            test_num = test_num + 1;
            if (rb == 5'd0) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: rb=0 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: rb=0 boundary", test_num); end

            decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd31, `MBAR_INIT));
            test_num = test_num + 1;
            if (rb == 5'd31) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: rb=31 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: rb=31 boundary", test_num); end

            // Test func=0 and func=63
            decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, 6'h00));
            test_num = test_num + 1;
            if (func == 6'h00) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: func=0 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: func=0 boundary", test_num); end

            decode_instruction(build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, 6'h3F));
            test_num = test_num + 1;
            if (func == 6'h3F) begin pass_count = pass_count + 1; $display("[PASS] Test %0d: func=63 boundary", test_num); end
            else begin fail_count = fail_count + 1; $display("[FAIL] Test %0d: func=63 boundary", test_num); end

            // Test all registers at max
            decode_instruction(build_inst(`OP_TEX, 5'd31, 5'd31, 5'd31, 6'h3F));
            test_num = test_num + 1;
            if (rd == 5'd31 && ra == 5'd31 && rb == 5'd31 && func == 6'h3F) begin
                pass_count = pass_count + 1; $display("[PASS] Test %0d: all fields max", test_num);
            end else begin
                fail_count = fail_count + 1; $display("[FAIL] Test %0d: all fields max", test_num);
            end

            // Test all registers at min
            decode_instruction(build_inst(`OP_SULD, 5'd0, 5'd0, 5'd0, 6'h00));
            test_num = test_num + 1;
            if (rd == 5'd0 && ra == 5'd0 && rb == 5'd0 && func == 6'h00) begin
                pass_count = pass_count + 1; $display("[PASS] Test %0d: all fields min", test_num);
            end else begin
                fail_count = fail_count + 1; $display("[FAIL] Test %0d: all fields min", test_num);
            end
        end

        //--------------------------------------------------------------------
        // 9.2: Mbarrier Function Code Exhaustive Tests (8 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.2: Mbarrier Function Code Tests ---");

        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE_DROP));
        test_num = test_num + 1;
        if (mbarrier_op && func == `MBAR_ARRIVE_DROP) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.arrive_drop", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.arrive_drop", test_num);
        end

        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE_TX));
        test_num = test_num + 1;
        if (mbarrier_op && func == `MBAR_ARRIVE_TX) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.arrive_tx", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.arrive_tx", test_num);
        end

        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INVALIDATE));
        test_num = test_num + 1;
        if (mbarrier_op && func == `MBAR_INVALIDATE) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.invalidate", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.invalidate", test_num);
        end

        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE_NOCOMP));
        test_num = test_num + 1;
        if (mbarrier_op && func == `MBAR_ARRIVE_NOCOMP) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.arrive_noComplete", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.arrive_noComplete", test_num);
        end

        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_EXPECT_TX));
        test_num = test_num + 1;
        if (mbarrier_op && func == `MBAR_EXPECT_TX) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.expect_tx", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.expect_tx", test_num);
        end

        // Mbarrier with different register combinations
        decode_instruction(build_inst(`OP_MBARRIER, 5'd15, 5'd20, 5'd25, `MBAR_TEST_WAIT));
        test_num = test_num + 1;
        if (mbarrier_op && reg_write && rd == 5'd15 && ra == 5'd20 && rb == 5'd25) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.test_wait with regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.test_wait with regs", test_num);
        end

        decode_instruction(build_inst(`OP_MBARRIER, 5'd10, 5'd11, 5'd12, `MBAR_TRY_WAIT));
        test_num = test_num + 1;
        if (mbarrier_op && reg_write && rd == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.try_wait reg_write", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.try_wait reg_write", test_num);
        end

        // Mbarrier init with different expected counts
        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd30, 5'd0, `MBAR_INIT));
        test_num = test_num + 1;
        if (mbarrier_op && ra == 5'd30) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: mbarrier.init with count reg", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: mbarrier.init with count reg", test_num);
        end

        //--------------------------------------------------------------------
        // 9.3: WGMMA Tile Size Variations (10 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.3: WGMMA Tile Size Variations ---");

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N8K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N8K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N8K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N16K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N16K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N16K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N16K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N32K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N32K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N32K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N32K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N64K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N64K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N64K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N64K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N128K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N128K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N128K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N128K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N256K16));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_M64N256K16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA M64N256K16", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA M64N256K16", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd0, 5'd0, `WGMMA_FENCE));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_FENCE) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA fence", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA fence", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd0, 5'd0, `WGMMA_COMMIT_GROUP));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_COMMIT_GROUP) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA commit_group", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA commit_group", test_num);
        end

        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd0, 5'd5, 5'd0, `WGMMA_WAIT_GROUP));
        test_num = test_num + 1;
        if (wgmma_mma && func == `WGMMA_WAIT_GROUP) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA wait_group", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA wait_group", test_num);
        end

        // WGMMA with various register combinations
        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd28, 5'd29, 5'd30, `WGMMA_M64N8K16));
        test_num = test_num + 1;
        if (wgmma_mma && rd == 5'd28 && ra == 5'd29 && rb == 5'd30) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: WGMMA high regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: WGMMA high regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.4: Async Copy Cache Hint Variations (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.4: Async Copy Cache Hint Variations ---");

        decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
        test_num = test_num + 1;
        if (cpasync_op && cache_hint == 3'b000) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: cp.async.ca cache_hint=0", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: cp.async.ca cache_hint=0", test_num);
        end

        decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CG));
        test_num = test_num + 1;
        if (cpasync_op && cache_hint == 3'b001) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: cp.async.cg cache_hint=1", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: cp.async.cg cache_hint=1", test_num);
        end

        decode_instruction(build_inst(`OP_PREFETCH, 5'd0, 5'd1, 5'd0, 6'b000000));
        test_num = test_num + 1;
        if (prefetch_op && cache_hint == 3'b000) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: prefetch hint=0", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: prefetch hint=0", test_num);
        end

        decode_instruction(build_inst(`OP_PREFETCH, 5'd0, 5'd1, 5'd0, 6'b000001));
        test_num = test_num + 1;
        if (prefetch_op && cache_hint == 3'b001) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: prefetch hint=1", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: prefetch hint=1", test_num);
        end

        decode_instruction(build_inst(`OP_PREFETCH, 5'd0, 5'd1, 5'd0, 6'b000010));
        test_num = test_num + 1;
        if (prefetch_op && cache_hint == 3'b010) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: prefetch hint=2", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: prefetch hint=2", test_num);
        end

        decode_instruction(build_inst(`OP_PREFETCH, 5'd0, 5'd1, 5'd0, 6'b000111));
        test_num = test_num + 1;
        if (prefetch_op && cache_hint == 3'b111) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: prefetch hint=7", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: prefetch hint=7", test_num);
        end

        //--------------------------------------------------------------------
        // 9.5: Barrier Cluster All Variants (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.5: Barrier Cluster Variants ---");

        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd1, 5'd2, `CLUSTER_BARRIER_ARRIVE));
        test_num = test_num + 1;
        if (barrier_cluster_op && sync_op && func == `CLUSTER_BARRIER_ARRIVE) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster.arrive", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster.arrive", test_num);
        end

        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_WAIT));
        test_num = test_num + 1;
        if (barrier_cluster_op && sync_op && func == `CLUSTER_BARRIER_WAIT) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster.wait", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster.wait", test_num);
        end

        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
        test_num = test_num + 1;
        if (barrier_cluster_op && sync_op && func == `CLUSTER_BARRIER_SYNC) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster.sync", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster.sync", test_num);
        end

        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd10, 5'd0, `CLUSTER_BARRIER_INIT));
        test_num = test_num + 1;
        if (barrier_cluster_op && sync_op && func == `CLUSTER_BARRIER_INIT && ra == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster.init with count", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster.init with count", test_num);
        end

        // Barrier with high register values
        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd31, 5'd31, 5'd31, `CLUSTER_BARRIER_ARRIVE));
        test_num = test_num + 1;
        if (barrier_cluster_op && ra == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster high regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster high regs", test_num);
        end

        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd15, 5'd16, `CLUSTER_BARRIER_SYNC));
        test_num = test_num + 1;
        if (barrier_cluster_op && ra == 5'd15 && rb == 5'd16) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: barrier.cluster mid regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: barrier.cluster mid regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.6: Multimem Variations (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.6: Multimem Variations ---");

        decode_instruction(build_inst(`OP_MULTIMEM, 5'd5, 5'd10, 5'd0, `MULTIMEM_LD));
        test_num = test_num + 1;
        if (multimem_op && mem_read && mem_shared && reg_write && rd == 5'd5 && ra == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem.ld regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem.ld regs", test_num);
        end

        decode_instruction(build_inst(`OP_MULTIMEM, 5'd0, 5'd20, 5'd21, `MULTIMEM_ST));
        test_num = test_num + 1;
        if (multimem_op && mem_write && mem_shared && ra == 5'd20 && rb == 5'd21) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem.st regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem.st regs", test_num);
        end

        decode_instruction(build_inst(`OP_MULTIMEM, 5'd0, 5'd25, 5'd26, `MULTIMEM_RED));
        test_num = test_num + 1;
        if (multimem_op && mem_write && mem_shared && ra == 5'd25 && rb == 5'd26) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem.red regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem.red regs", test_num);
        end

        // Multimem with boundary registers
        decode_instruction(build_inst(`OP_MULTIMEM, 5'd31, 5'd31, 5'd31, `MULTIMEM_LD));
        test_num = test_num + 1;
        if (multimem_op && rd == 5'd31 && ra == 5'd31 && rb == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem max regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem max regs", test_num);
        end

        decode_instruction(build_inst(`OP_MULTIMEM, 5'd0, 5'd0, 5'd0, `MULTIMEM_ST));
        test_num = test_num + 1;
        if (multimem_op && rd == 5'd0 && ra == 5'd0 && rb == 5'd0) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem min regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem min regs", test_num);
        end

        decode_instruction(build_inst(`OP_MULTIMEM, 5'd16, 5'd17, 5'd18, `MULTIMEM_RED));
        test_num = test_num + 1;
        if (multimem_op && rd == 5'd16 && ra == 5'd17) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: multimem mid regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: multimem mid regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.7: St.async Variations (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.7: St.async Variations ---");

        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd5, 5'd10, `ST_ASYNC_GLOBAL));
        test_num = test_num + 1;
        if (st_async_op && mem_write && !mem_shared && ra == 5'd5 && rb == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async.global regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async.global regs", test_num);
        end

        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd15, 5'd20, `ST_ASYNC_SHARED));
        test_num = test_num + 1;
        if (st_async_op && mem_write && mem_shared && ra == 5'd15 && rb == 5'd20) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async.shared regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async.shared regs", test_num);
        end

        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd0, 5'd0, `ST_ASYNC_COMMIT));
        test_num = test_num + 1;
        if (st_async_op && !mem_write && func == `ST_ASYNC_COMMIT) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async.commit", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async.commit", test_num);
        end

        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd3, 5'd0, `ST_ASYNC_WAIT));
        test_num = test_num + 1;
        if (st_async_op && !mem_write && func == `ST_ASYNC_WAIT) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async.wait", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async.wait", test_num);
        end

        // St.async with boundary registers
        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd31, 5'd31, 5'd31, `ST_ASYNC_GLOBAL));
        test_num = test_num + 1;
        if (st_async_op && ra == 5'd31 && rb == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async max regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async max regs", test_num);
        end

        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd0, 5'd0, `ST_ASYNC_SHARED));
        test_num = test_num + 1;
        if (st_async_op && ra == 5'd0 && rb == 5'd0) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: st.async min regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: st.async min regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.8: Cache Policy Variations (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.8: Cache Policy Variations ---");

        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd5, 5'd10, 5'd0, `CACHE_CREATEPOLICY));
        test_num = test_num + 1;
        if (cache_policy_op && reg_write && rd == 5'd5 && ra == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: createpolicy regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: createpolicy regs", test_num);
        end

        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd0, 5'd15, 5'd20, `CACHE_APPLYPRIORITY));
        test_num = test_num + 1;
        if (cache_policy_op && !reg_write && ra == 5'd15 && rb == 5'd20) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: applypriority regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: applypriority regs", test_num);
        end

        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd0, 5'd25, 5'd0, `CACHE_DISCARD));
        test_num = test_num + 1;
        if (cache_policy_op && !reg_write && ra == 5'd25) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: discard regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: discard regs", test_num);
        end

        // Cache policy with boundary registers
        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd31, 5'd31, 5'd31, `CACHE_CREATEPOLICY));
        test_num = test_num + 1;
        if (cache_policy_op && rd == 5'd31 && ra == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: cache_policy max regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: cache_policy max regs", test_num);
        end

        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd0, 5'd0, 5'd0, `CACHE_APPLYPRIORITY));
        test_num = test_num + 1;
        if (cache_policy_op && ra == 5'd0) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: cache_policy min regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: cache_policy min regs", test_num);
        end

        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd16, 5'd17, 5'd18, `CACHE_DISCARD));
        test_num = test_num + 1;
        if (cache_policy_op && ra == 5'd17) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: cache_policy mid regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: cache_policy mid regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.9: Stack Operations Extended (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.9: Stack Operations Extended ---");

        decode_instruction(build_inst(`OP_STACK, 5'd5, 5'd10, 5'd0, `STACK_ALLOCA));
        test_num = test_num + 1;
        if (stack_op && reg_write && rd == 5'd5 && ra == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: alloca regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: alloca regs", test_num);
        end

        decode_instruction(build_inst(`OP_STACK, 5'd15, 5'd0, 5'd0, `STACK_SAVE));
        test_num = test_num + 1;
        if (stack_op && reg_write && rd == 5'd15) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: stacksave regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: stacksave regs", test_num);
        end

        decode_instruction(build_inst(`OP_STACK, 5'd0, 5'd20, 5'd0, `STACK_RESTORE));
        test_num = test_num + 1;
        if (stack_op && !reg_write && ra == 5'd20) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: stackrestore regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: stackrestore regs", test_num);
        end

        // Stack with boundary registers
        decode_instruction(build_inst(`OP_STACK, 5'd31, 5'd31, 5'd0, `STACK_ALLOCA));
        test_num = test_num + 1;
        if (stack_op && rd == 5'd31 && ra == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: stack max regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: stack max regs", test_num);
        end

        decode_instruction(build_inst(`OP_STACK, 5'd0, 5'd0, 5'd0, `STACK_SAVE));
        test_num = test_num + 1;
        if (stack_op && rd == 5'd0) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: stack min regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: stack min regs", test_num);
        end

        decode_instruction(build_inst(`OP_STACK, 5'd16, 5'd17, 5'd0, `STACK_RESTORE));
        test_num = test_num + 1;
        if (stack_op && ra == 5'd17) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: stack mid regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: stack mid regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.10: Debug/Misc Operations Extended (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.10: Debug/Misc Operations Extended ---");

        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd5, 5'd0, `DEBUG_BRKPT));
        test_num = test_num + 1;
        if (debug_op && !reg_write && ra == 5'd5) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: brkpt regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: brkpt regs", test_num);
        end

        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd10, 5'd0, `DEBUG_TRAP));
        test_num = test_num + 1;
        if (debug_op && !reg_write && ra == 5'd10) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: trap regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: trap regs", test_num);
        end

        decode_instruction(build_inst(`OP_DEBUG, 5'd0, 5'd15, 5'd0, `DEBUG_PMEVENT));
        test_num = test_num + 1;
        if (debug_op && !reg_write && ra == 5'd15) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: pmevent regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: pmevent regs", test_num);
        end

        decode_instruction(build_inst(`OP_MISC, 5'd0, 5'd20, 5'd0, `MISC_NANOSLEEP));
        test_num = test_num + 1;
        if (misc_op && !reg_write && ra == 5'd20) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: nanosleep regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: nanosleep regs", test_num);
        end

        decode_instruction(build_inst(`OP_MISC, 5'd0, 5'd25, 5'd0, `MISC_SETMAXNREG));
        test_num = test_num + 1;
        if (misc_op && !reg_write && ra == 5'd25) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: setmaxnreg regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: setmaxnreg regs", test_num);
        end

        // Debug/misc with boundary registers
        decode_instruction(build_inst(`OP_DEBUG, 5'd31, 5'd31, 5'd31, `DEBUG_PMEVENT));
        test_num = test_num + 1;
        if (debug_op && ra == 5'd31) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: debug max regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: debug max regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.11: Texture/Surface Extended (8 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.11: Texture/Surface Extended ---");

        decode_instruction(build_inst(`OP_TEX, 5'd5, 5'd10, 5'd15, 6'h00));
        test_num = test_num + 1;
        if (tex_op && mem_read && reg_write && rd == 5'd5 && ra == 5'd10 && rb == 5'd15) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: tex regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: tex regs", test_num);
        end

        decode_instruction(build_inst(`OP_TXQ, 5'd20, 5'd25, 5'd0, 6'h01));
        test_num = test_num + 1;
        if (txq_op && reg_write && rd == 5'd20 && ra == 5'd25 && func == 6'h01) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: txq regs func", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: txq regs func", test_num);
        end

        decode_instruction(build_inst(`OP_SULD, 5'd1, 5'd2, 5'd3, 6'h02));
        test_num = test_num + 1;
        if (surf_ld && mem_read && reg_write && func == 6'h02) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: suld func", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: suld func", test_num);
        end

        decode_instruction(build_inst(`OP_SUST, 5'd0, 5'd4, 5'd5, 6'h03));
        test_num = test_num + 1;
        if (surf_st && mem_write && !reg_write && func == 6'h03) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: sust func", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: sust func", test_num);
        end

        decode_instruction(build_inst(`OP_SURED, 5'd6, 5'd7, 5'd8, 6'h04));
        test_num = test_num + 1;
        if (surf_red && mem_read && mem_write && reg_write && func == 6'h04) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: sured func", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: sured func", test_num);
        end

        // Texture with boundary registers
        decode_instruction(build_inst(`OP_TEX, 5'd31, 5'd31, 5'd31, 6'h3F));
        test_num = test_num + 1;
        if (tex_op && rd == 5'd31 && ra == 5'd31 && rb == 5'd31 && func == 6'h3F) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: tex max all", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: tex max all", test_num);
        end

        decode_instruction(build_inst(`OP_SULD, 5'd0, 5'd0, 5'd0, 6'h00));
        test_num = test_num + 1;
        if (surf_ld && rd == 5'd0 && ra == 5'd0 && rb == 5'd0) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: suld min all", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: suld min all", test_num);
        end

        decode_instruction(build_inst(`OP_SUST, 5'd16, 5'd17, 5'd18, 6'h20));
        test_num = test_num + 1;
        if (surf_st && ra == 5'd17 && rb == 5'd18 && func == 6'h20) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: sust mid regs", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: sust mid regs", test_num);
        end

        //--------------------------------------------------------------------
        // 9.12: Video Operations Extended (6 tests)
        //--------------------------------------------------------------------
        $display("\n--- 9.12: Video Operations Extended ---");

        decode_instruction(build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, `VIDEO_VADD));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_VADD) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video vadd", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video vadd", test_num);
        end

        decode_instruction(build_inst(`OP_VIDEO, 5'd4, 5'd5, 5'd6, `VIDEO_VSUB));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_VSUB) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video vsub", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video vsub", test_num);
        end

        decode_instruction(build_inst(`OP_VIDEO, 5'd7, 5'd8, 5'd9, `VIDEO_VABSDIFF));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_VABSDIFF) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video vabsdiff", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video vabsdiff", test_num);
        end

        decode_instruction(build_inst(`OP_VIDEO, 5'd10, 5'd11, 5'd12, `VIDEO_VMIN));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_VMIN) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video vmin", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video vmin", test_num);
        end

        decode_instruction(build_inst(`OP_VIDEO, 5'd13, 5'd14, 5'd15, `VIDEO_VMAX));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_VMAX) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video vmax", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video vmax", test_num);
        end

        decode_instruction(build_inst(`OP_VIDEO, 5'd31, 5'd31, 5'd31, `VIDEO_DP4A_ALU));
        test_num = test_num + 1;
        if (video_op && reg_write && func == `VIDEO_DP4A_ALU) begin
            pass_count = pass_count + 1; $display("[PASS] Test %0d: video dp4a", test_num);
        end else begin
            fail_count = fail_count + 1; $display("[FAIL] Test %0d: video dp4a", test_num);
        end

        //--------------------------------------------------------------------
        // 9.13: Complex Kernel Simulation - GEMM Pattern (8 tests as 1 block)
        //--------------------------------------------------------------------
        $display("\n--- 9.13: Complex GEMM Kernel Pattern ---");

        begin : gemm_kernel_pattern
            integer i, j;
            integer gemm_errors;
            gemm_errors = 0;

            // Simulate GEMM kernel with TMA + WGMMA
            for (i = 0; i < 2; i = i + 1) begin
                // Prologue: Initialize mbarriers
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_INIT));
                if (!mbarrier_op) gemm_errors = gemm_errors + 1;

                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd2, 5'd0, `MBAR_INIT));
                if (!mbarrier_op) gemm_errors = gemm_errors + 1;

                // Load A matrix
                decode_instruction(build_inst(`OP_CPASYNC, 5'd4, 5'd8, 5'd0, `CPASYNC_CA));
                if (!cpasync_op) gemm_errors = gemm_errors + 1;

                // Load B matrix
                decode_instruction(build_inst(`OP_CPASYNC, 5'd5, 5'd9, 5'd0, `CPASYNC_CA));
                if (!cpasync_op) gemm_errors = gemm_errors + 1;

                // Commit group
                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) gemm_errors = gemm_errors + 1;

                // Signal arrival
                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                if (!mbarrier_op) gemm_errors = gemm_errors + 1;

                // Compute loop
                for (j = 0; j < 3; j = j + 1) begin
                    // WGMMA multiply-accumulate
                    decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd10, 5'd4, 5'd5, `WGMMA_M64N16K16));
                    if (!wgmma_mma) gemm_errors = gemm_errors + 1;
                end

                // Warp sync before store
                decode_instruction(build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd0, 5'd0, 6'h00));
                if (!bar_warp_sync) gemm_errors = gemm_errors + 1;

                // Store result
                decode_instruction(build_inst(`OP_WGMMA_STORE, 5'd0, 5'd10, 5'd12, 6'h00));
                if (!wgmma_store) gemm_errors = gemm_errors + 1;

                // Cluster sync
                decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
                if (!barrier_cluster_op) gemm_errors = gemm_errors + 1;
            end

            test_num = test_num + 1;
            if (gemm_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: GEMM kernel pattern (28 ops)", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: GEMM kernel pattern - %0d errors", test_num, gemm_errors);
            end
        end

        //--------------------------------------------------------------------
        // 9.14: Flash Attention Style Pattern (8 tests as 1 block)
        //--------------------------------------------------------------------
        $display("\n--- 9.14: Flash Attention Pattern ---");

        begin : flash_attention_pattern
            integer i;
            integer fa_errors;
            fa_errors = 0;

            for (i = 0; i < 2; i = i + 1) begin
                // Load Q, K tiles
                decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd10, 5'd0, `CPASYNC_CA));
                if (!cpasync_op) fa_errors = fa_errors + 1;

                decode_instruction(build_inst(`OP_CPASYNC, 5'd2, 5'd11, 5'd0, `CPASYNC_CA));
                if (!cpasync_op) fa_errors = fa_errors + 1;

                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) fa_errors = fa_errors + 1;

                decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                if (!mbarrier_op) fa_errors = fa_errors + 1;

                // Q*K^T
                decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd3, 5'd1, 5'd2, `WGMMA_M64N64K16));
                if (!wgmma_mma) fa_errors = fa_errors + 1;

                // Load V
                decode_instruction(build_inst(`OP_CPASYNC, 5'd4, 5'd12, 5'd0, `CPASYNC_CA));
                if (!cpasync_op) fa_errors = fa_errors + 1;

                decode_instruction(build_inst(`OP_CPASYNC, 5'd0, 5'd0, 5'd0, `CPASYNC_COMMIT));
                if (!cpasync_op) fa_errors = fa_errors + 1;

                // Softmax * V
                decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd5, 5'd3, 5'd4, `WGMMA_M64N64K16));
                if (!wgmma_mma) fa_errors = fa_errors + 1;

                // Async store
                decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd5, 5'd13, `ST_ASYNC_GLOBAL));
                if (!st_async_op) fa_errors = fa_errors + 1;

                // Sync
                decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
                if (!barrier_cluster_op) fa_errors = fa_errors + 1;
            end

            test_num = test_num + 1;
            if (fa_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: Flash Attention pattern (20 ops)", test_num);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: Flash Attention pattern - %0d errors", test_num, fa_errors);
            end
        end

        //--------------------------------------------------------------------
        // 9.15: Performance Stress - Rapid Decode Test (1 test covering 200 ops)
        //--------------------------------------------------------------------
        $display("\n--- 9.15: Performance Stress Test ---");

        begin : perf_stress_200
            integer i;
            integer stress_errors;
            integer start_time, end_time;
            real ops_per_ns;

            stress_errors = 0;
            start_time = $time;

            // Decode 200 mixed operations rapidly
            for (i = 0; i < 200; i = i + 1) begin
                case (i % 10)
                    0: begin
                        decode_instruction(build_inst(`OP_CPASYNC, 5'd1, 5'd2, 5'd3, `CPASYNC_CA));
                        if (!cpasync_op) stress_errors = stress_errors + 1;
                    end
                    1: begin
                        decode_instruction(build_inst(`OP_WGMMA_MMA, 5'd1, 5'd2, 5'd3, `WGMMA_M64N8K16));
                        if (!wgmma_mma) stress_errors = stress_errors + 1;
                    end
                    2: begin
                        decode_instruction(build_inst(`OP_MBARRIER, 5'd0, 5'd1, 5'd0, `MBAR_ARRIVE));
                        if (!mbarrier_op) stress_errors = stress_errors + 1;
                    end
                    3: begin
                        decode_instruction(build_inst(`OP_BAR_WARP_SYNC, 5'd0, 5'd1, 5'd0, 6'h00));
                        if (!bar_warp_sync) stress_errors = stress_errors + 1;
                    end
                    4: begin
                        decode_instruction(build_inst(`OP_BARRIER_CLUSTER, 5'd0, 5'd0, 5'd0, `CLUSTER_BARRIER_SYNC));
                        if (!barrier_cluster_op) stress_errors = stress_errors + 1;
                    end
                    5: begin
                        decode_instruction(build_inst(`OP_TEX, 5'd1, 5'd2, 5'd3, 6'h00));
                        if (!tex_op) stress_errors = stress_errors + 1;
                    end
                    6: begin
                        decode_instruction(build_inst(`OP_VIDEO, 5'd1, 5'd2, 5'd3, `VIDEO_VADD));
                        if (!video_op) stress_errors = stress_errors + 1;
                    end
                    7: begin
                        decode_instruction(build_inst(`OP_ST_ASYNC, 5'd0, 5'd1, 5'd2, `ST_ASYNC_GLOBAL));
                        if (!st_async_op) stress_errors = stress_errors + 1;
                    end
                    8: begin
                        decode_instruction(build_inst(`OP_MULTIMEM, 5'd1, 5'd2, 5'd0, `MULTIMEM_LD));
                        if (!multimem_op) stress_errors = stress_errors + 1;
                    end
                    9: begin
                        decode_instruction(build_inst(`OP_CACHE_POLICY, 5'd1, 5'd0, 5'd0, `CACHE_CREATEPOLICY));
                        if (!cache_policy_op) stress_errors = stress_errors + 1;
                    end
                endcase
            end

            end_time = $time;
            ops_per_ns = 200.0 / ((end_time - start_time) / 1000.0);

            test_num = test_num + 1;
            if (stress_errors == 0) begin
                pass_count = pass_count + 1;
                $display("[PASS] Test %0d: Stress test 200 ops (%.2f ops/ns)", test_num, ops_per_ns);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] Test %0d: Stress test - %0d errors", test_num, stress_errors);
            end
        end

        //====================================================================
        // Summary
        //====================================================================
        $display("\n============================================================");
        $display("B300 Features Test Summary");
        $display("============================================================");
        $display("  Passed: %0d", pass_count);
        $display("  Failed: %0d", fail_count);
        $display("  Total:  %0d", test_num);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("*** ALL B300 FEATURE TESTS PASSED ***");
        end else begin
            $display("*** SOME TESTS FAILED ***");
        end

        $finish;
    end

    // Timeout watchdog
    initial begin
        #2000000;  // Increased timeout for extended tests
        $display("TIMEOUT: Test took too long");
        $finish;
    end

endmodule
