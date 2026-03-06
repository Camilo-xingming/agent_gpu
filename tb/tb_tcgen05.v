//============================================================================
// RalphGPU - Blackwell tcgen05 Instruction Testbench
// Tests 5th-gen Tensor Core (tcgen05) instruction decoding
// Verifies: tcgen05.mma, tcgen05.ld, tcgen05.st, tcgen05.cp,
//           tcgen05.alloc, tcgen05.dealloc, tcgen05.commit, tcgen05.wait
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_tcgen05;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg  [31:0] instruction;
    reg         valid_in;

    // Standard decoder outputs
    wire        valid_out;
    wire [5:0]  opcode;
    wire [4:0]  rd, ra, rb, rc;
    wire [5:0]  func;
    wire [15:0] imm16;
    wire [20:0] imm21;
    wire        use_imm;
    wire        alu_op, mul_op, div_op;
    wire        mem_read, mem_write, mem_shared;
    wire        branch_op, sync_op, special_reg, exit_op;
    wire        reg_write, pred_write;
    wire [2:0]  pred_addr;

    // FP and extended signals
    wire        fp32_op, fp32_special, fp64_op, fp16_op, cvt_op;
    wire        mem_param, mem_const, mem_local, mem_vector;
    wire [1:0]  vec_size;
    wire        atomic_op, reduce_op;
    wire        shfl_op, vote_op, redux_op;
    wire        wmma_load, wmma_store, wmma_mma, mma_op;
    wire        call_op, membar_op, video_op;
    wire        tex_op, txq_op, surf_ld, surf_st, surf_red;
    wire        cpasync_op, prefetch_op;
    wire        wgmma_load, wgmma_store, wgmma_mma;
    wire [2:0]  cache_hint;
    wire        mbarrier_op, bar_warp_sync, cache_policy_op;
    wire        stack_op, debug_op, misc_op;
    wire        st_async_op, multimem_op, barrier_cluster_op;
    wire        match_sync_op, elect_sync_op, red_async_op;
    wire        dpx_op, sparse_mma_op;

    // tcgen05 specific outputs
    wire        tcgen05_op;
    wire        tcgen05_mma;
    wire        tcgen05_ld;
    wire        tcgen05_st;
    wire        tcgen05_cp;
    wire        tcgen05_alloc;
    wire        tcgen05_dealloc;
    wire        tcgen05_commit;
    wire        tcgen05_wait;
    wire [15:0] tmem_addr;
    wire [3:0]  tcgen05_dtype;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    decoder dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .instruction     (instruction),
        .valid_in        (valid_in),
        .valid_out       (valid_out),
        .opcode          (opcode),
        .rd              (rd),
        .ra              (ra),
        .rb              (rb),
        .rc              (rc),
        .func            (func),
        .imm16           (imm16),
        .imm21           (imm21),
        .use_imm         (use_imm),
        .alu_op          (alu_op),
        .mul_op          (mul_op),
        .div_op          (div_op),
        .mem_read        (mem_read),
        .mem_write       (mem_write),
        .mem_shared      (mem_shared),
        .branch_op       (branch_op),
        .sync_op         (sync_op),
        .special_reg     (special_reg),
        .exit_op         (exit_op),
        .reg_write       (reg_write),
        .pred_write      (pred_write),
        .pred_addr       (pred_addr),
        .fp32_op         (fp32_op),
        .fp32_special    (fp32_special),
        .fp64_op         (fp64_op),
        .fp16_op         (fp16_op),
        .cvt_op          (cvt_op),
        .mem_param       (mem_param),
        .mem_const       (mem_const),
        .mem_local       (mem_local),
        .mem_vector      (mem_vector),
        .vec_size        (vec_size),
        .atomic_op       (atomic_op),
        .reduce_op       (reduce_op),
        .shfl_op         (shfl_op),
        .vote_op         (vote_op),
        .redux_op        (redux_op),
        .wmma_load       (wmma_load),
        .wmma_store      (wmma_store),
        .wmma_mma        (wmma_mma),
        .mma_op          (mma_op),
        .call_op         (call_op),
        .membar_op       (membar_op),
        .video_op        (video_op),
        .tex_op          (tex_op),
        .txq_op          (txq_op),
        .surf_ld         (surf_ld),
        .surf_st         (surf_st),
        .surf_red        (surf_red),
        .cpasync_op      (cpasync_op),
        .prefetch_op     (prefetch_op),
        .wgmma_load      (wgmma_load),
        .wgmma_store     (wgmma_store),
        .wgmma_mma       (wgmma_mma),
        .cache_hint      (cache_hint),
        .mbarrier_op     (mbarrier_op),
        .bar_warp_sync   (bar_warp_sync),
        .cache_policy_op (cache_policy_op),
        .stack_op        (stack_op),
        .debug_op        (debug_op),
        .misc_op         (misc_op),
        .st_async_op     (st_async_op),
        .multimem_op     (multimem_op),
        .barrier_cluster_op (barrier_cluster_op),
        .match_sync_op   (match_sync_op),
        .elect_sync_op   (elect_sync_op),
        .red_async_op    (red_async_op),
        .dpx_op          (dpx_op),
        .sparse_mma_op   (sparse_mma_op),
        .tcgen05_op      (tcgen05_op),
        .tcgen05_mma     (tcgen05_mma),
        .tcgen05_ld      (tcgen05_ld),
        .tcgen05_st      (tcgen05_st),
        .tcgen05_cp      (tcgen05_cp),
        .tcgen05_alloc   (tcgen05_alloc),
        .tcgen05_dealloc (tcgen05_dealloc),
        .tcgen05_commit  (tcgen05_commit),
        .tcgen05_wait    (tcgen05_wait),
        .tmem_addr       (tmem_addr),
        .tcgen05_dtype   (tcgen05_dtype)
    );

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer passed = 0;
    integer failed = 0;

    //------------------------------------------------------------------------
    // Helper Function - Construct tcgen05 Instruction
    // tcgen05 uses OP_MMA opcode with func[4]=1 to distinguish from regular MMA
    // Format: [31:26]=OP_MMA, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]=rc (dtype), [5:0]=func
    // tcgen05 func codes: 6'b01xxxx where xxxx=operation (0=mma, 1=ld, 2=st, etc.)
    //------------------------------------------------------------------------
    function [31:0] make_tcgen05_inst;
        input [4:0]  dst;       // rd - destination register or TMEM base
        input [4:0]  src_a;     // ra - source register A
        input [4:0]  src_b;     // rb - source register B
        input [3:0]  dtype;     // data type (in rc[3:0])
        input [5:0]  fn;        // function code (should have bit[4]=1 for tcgen05)
        input [9:0]  tmem_low;  // unused - kept for compatibility
        begin
            // [31:26]=OP_MMA, [25:21]=rd, [20:16]=ra, [15:11]=rb, [10:6]={1'b0, dtype}, [5:0]=func
            make_tcgen05_inst = {`OP_MMA, dst, src_a, src_b, 1'b0, dtype, fn};
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Task
    //------------------------------------------------------------------------
    task decode_and_check_tcgen05;
        input [31:0]  inst;
        input         exp_tcgen05_op;
        input         exp_tcgen05_mma;
        input         exp_tcgen05_ld;
        input         exp_tcgen05_st;
        input         exp_tcgen05_cp;
        input         exp_tcgen05_alloc;
        input         exp_tcgen05_dealloc;
        input         exp_tcgen05_commit;
        input         exp_tcgen05_wait;
        input         exp_reg_write;
        input         exp_mem_shared;
        input         exp_sync_op;
        input         exp_mbarrier_op;
        input [127:0] test_name;
        reg all_match;
        begin
            @(posedge clk);
            instruction <= inst;
            valid_in    <= 1;

            @(posedge clk);
            valid_in <= 0;

            @(posedge clk);  // Wait for output

            all_match = (tcgen05_op === exp_tcgen05_op) &&
                       (tcgen05_mma === exp_tcgen05_mma) &&
                       (tcgen05_ld === exp_tcgen05_ld) &&
                       (tcgen05_st === exp_tcgen05_st) &&
                       (tcgen05_cp === exp_tcgen05_cp) &&
                       (tcgen05_alloc === exp_tcgen05_alloc) &&
                       (tcgen05_dealloc === exp_tcgen05_dealloc) &&
                       (tcgen05_commit === exp_tcgen05_commit) &&
                       (tcgen05_wait === exp_tcgen05_wait) &&
                       (reg_write === exp_reg_write) &&
                       (mem_shared === exp_mem_shared) &&
                       (sync_op === exp_sync_op) &&
                       (mbarrier_op === exp_mbarrier_op);

            if (valid_out && all_match) begin
                $display("[PASS] %s: opcode=0x%02X, func=0x%02X, rd=%0d, ra=%0d",
                         test_name, opcode, func, rd, ra);
                passed = passed + 1;
            end else begin
                $display("[FAIL] %s", test_name);
                $display("       Got: tcgen05=%b mma=%b ld=%b st=%b cp=%b alloc=%b dealloc=%b commit=%b wait=%b",
                         tcgen05_op, tcgen05_mma, tcgen05_ld, tcgen05_st, tcgen05_cp,
                         tcgen05_alloc, tcgen05_dealloc, tcgen05_commit, tcgen05_wait);
                $display("       Got: reg_wr=%b mem_sh=%b sync=%b mbar=%b",
                         reg_write, mem_shared, sync_op, mbarrier_op);
                $display("       Exp: tcgen05=%b mma=%b ld=%b st=%b cp=%b alloc=%b dealloc=%b commit=%b wait=%b",
                         exp_tcgen05_op, exp_tcgen05_mma, exp_tcgen05_ld, exp_tcgen05_st, exp_tcgen05_cp,
                         exp_tcgen05_alloc, exp_tcgen05_dealloc, exp_tcgen05_commit, exp_tcgen05_wait);
                $display("       Exp: reg_wr=%b mem_sh=%b sync=%b mbar=%b",
                         exp_reg_write, exp_mem_shared, exp_sync_op, exp_mbarrier_op);
                failed = failed + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Test Cases
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Blackwell tcgen05 Instruction Unit Test");
        $display("============================================================");

        rst_n = 0;
        instruction = 0;
        valid_in = 0;

        #100;
        rst_n = 1;
        #20;

        //====================================================================
        // Test 1: tcgen05.alloc - TMEM Column Allocation
        //====================================================================
        $display("\n--- Test 1: tcgen05.alloc (TMEM Allocation) ---");

        // tcgen05.alloc rd, num_cols
        // Returns TMEM base address in rd
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd1, 5'd0, 5'd0, 4'd0, `TCGEN05_ALLOC, 10'd64),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            1,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            1,  // reg_write (returns base address)
            0,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.ALLOC r1, 64"
        );

        //====================================================================
        // Test 2: tcgen05.ld - Load from TMEM
        //====================================================================
        $display("\n--- Test 2: tcgen05.ld (TMEM Load) ---");

        // tcgen05.ld rd, [tmem_addr]
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd2, 5'd0, 5'd0, 4'd0, `TCGEN05_LD, 10'd128),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            1,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            1,  // reg_write
            0,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.LD r2, [0x80]"
        );

        //====================================================================
        // Test 3: tcgen05.st - Store to TMEM
        //====================================================================
        $display("\n--- Test 3: tcgen05.st (TMEM Store) ---");

        // tcgen05.st [tmem_addr], rs
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd3, 5'd0, 4'd0, `TCGEN05_ST, 10'd256),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            1,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write (no write)
            0,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.ST [0x100], r3"
        );

        //====================================================================
        // Test 4: tcgen05.mma - Per-Thread Async MMA
        //====================================================================
        $display("\n--- Test 4: tcgen05.mma (Async MMA) ---");

        // tcgen05.mma d-tmem, a-desc, b-desc (FP16 data type)
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd4, 5'd5, `TCGEN05_FP16, `TCGEN05_MMA, 10'd0),
            1,  // tcgen05_op
            1,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write (result to TMEM)
            1,  // mem_shared (operands from SMEM)
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.MMA FP16"
        );

        //====================================================================
        // Test 5: tcgen05.mma with BF16 data type
        //====================================================================
        $display("\n--- Test 5: tcgen05.mma BF16 ---");

        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd6, 5'd7, `TCGEN05_BF16, `TCGEN05_MMA, 10'd0),
            1,  // tcgen05_op
            1,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            1,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.MMA BF16"
        );

        //====================================================================
        // Test 6: tcgen05.cp - Async Tensor Copy
        //====================================================================
        $display("\n--- Test 6: tcgen05.cp (Tensor Copy) ---");

        // tcgen05.cp dst_tmem, src_smem
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd8, 5'd0, 4'd0, `TCGEN05_CP, 10'd512),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            1,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            1,  // mem_shared (source from SMEM)
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.CP [tmem], [smem]"
        );

        //====================================================================
        // Test 7: tcgen05.commit - Signal Completion via mbarrier
        //====================================================================
        $display("\n--- Test 7: tcgen05.commit (Completion Signal) ---");

        // tcgen05.commit mbar_addr
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd9, 5'd0, 4'd0, `TCGEN05_COMMIT, 10'd0),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            1,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            0,  // mem_shared
            0,  // sync_op
            1,  // mbarrier_op (uses mbarrier)
            "TCGEN05.COMMIT mbar"
        );

        //====================================================================
        // Test 8: tcgen05.wait - Wait for TMEM Operations
        //====================================================================
        $display("\n--- Test 8: tcgen05.wait (Sync Wait) ---");

        // tcgen05.wait
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd0, 5'd0, 4'd0, `TCGEN05_WAIT, 10'd0),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            1,  // tcgen05_wait
            0,  // reg_write
            0,  // mem_shared
            1,  // sync_op (stalls warp)
            0,  // mbarrier_op
            "TCGEN05.WAIT"
        );

        //====================================================================
        // Test 9: tcgen05.dealloc - TMEM Deallocation
        //====================================================================
        $display("\n--- Test 9: tcgen05.dealloc (TMEM Deallocation) ---");

        // tcgen05.dealloc tmem_base
        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd1, 5'd0, 4'd0, `TCGEN05_DEALLOC, 10'd0),
            1,  // tcgen05_op
            0,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            1,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            0,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.DEALLOC r1"
        );

        //====================================================================
        // Test 10: tcgen05.mma with FP8 E4M3 data type
        //====================================================================
        $display("\n--- Test 10: tcgen05.mma FP8_E4M3 ---");

        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd10, 5'd11, `TCGEN05_FP8_E4M3, `TCGEN05_MMA, 10'd0),
            1,  // tcgen05_op
            1,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            1,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.MMA FP8_E4M3"
        );

        //====================================================================
        // Test 11: tcgen05.mma with FP6 E3M2 (Blackwell new)
        //====================================================================
        $display("\n--- Test 11: tcgen05.mma FP6_E3M2 (Blackwell) ---");

        decode_and_check_tcgen05(
            make_tcgen05_inst(5'd0, 5'd12, 5'd13, `TCGEN05_FP6_E3M2, `TCGEN05_MMA, 10'd0),
            1,  // tcgen05_op
            1,  // tcgen05_mma
            0,  // tcgen05_ld
            0,  // tcgen05_st
            0,  // tcgen05_cp
            0,  // tcgen05_alloc
            0,  // tcgen05_dealloc
            0,  // tcgen05_commit
            0,  // tcgen05_wait
            0,  // reg_write
            1,  // mem_shared
            0,  // sync_op
            0,  // mbarrier_op
            "TCGEN05.MMA FP6_E3M2"
        );

        //====================================================================
        // Test Summary
        //====================================================================
        #100;
        $display("\n============================================================");
        $display("tcgen05 Decoder Test Summary: %0d PASSED, %0d FAILED", passed, failed);
        $display("============================================================");

        if (failed == 0) begin
            $display("*** ALL TCGEN05 TESTS PASSED ***");
            $display("C[0] = 0xCAFE");  // Success marker for test harness
        end else begin
            $display("*** SOME TCGEN05 TESTS FAILED ***");
            $display("C[0] = 0xDEAD");  // Failure marker
        end

        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

    //------------------------------------------------------------------------
    // Waveform Output
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_tcgen05.vcd");
        $dumpvars(0, tb_tcgen05);
    end

    initial begin
        #50000;
        $display("ERROR: Timeout!");
        if (failed > 0) $fatal(1, "Test Failed");
        $finish;
    end

endmodule
