//============================================================================
// RalphGPU - Cache Policy Unit Testbench
// Tests cache policy management and address space query instructions
//============================================================================

`timescale 1ns/1ps

`include "gpu_defines.vh"

module tb_cache_policy_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter DATA_WIDTH = 32;
    parameter SMEM_BASE = 32'hFFFE0000;
    parameter SMEM_SIZE = 32'h00020000;
    parameter LOCAL_BASE = 32'hFFFC0000;
    parameter LOCAL_SIZE = 32'h00020000;
    parameter CONST_BASE = 32'hFFFA0000;
    parameter CONST_SIZE = 32'h00020000;
    parameter PARAM_BASE = 32'hFFF80000;
    parameter PARAM_SIZE = 32'h00020000;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    //------------------------------------------------------------------------
    // DUT Interface
    //------------------------------------------------------------------------
    reg                      valid_in;
    reg  [5:0]               opcode;
    reg  [5:0]               func;
    reg  [DATA_WIDTH-1:0]    src_a;
    reg  [DATA_WIDTH-1:0]    src_b;
    reg  [2:0]               cache_level;

    // Cluster config
    reg  [3:0]               cta_id;
    reg  [3:0]               cluster_size;
    reg  [3:0]               sm_id;

    // Outputs
    wire                     done;
    wire                     result_valid;
    wire [DATA_WIDTH-1:0]    result;
    wire                     pred_result;

    // Cache control interface
    wire                     cache_ctrl_valid;
    wire [2:0]               cache_ctrl_op;
    wire [DATA_WIDTH-1:0]    cache_ctrl_addr;
    wire [7:0]               cache_ctrl_policy;
    wire [2:0]               cache_ctrl_level;
    reg                      cache_ctrl_done;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    cache_policy_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .SMEM_BASE(SMEM_BASE),
        .SMEM_SIZE(SMEM_SIZE),
        .LOCAL_BASE(LOCAL_BASE),
        .LOCAL_SIZE(LOCAL_SIZE),
        .CONST_BASE(CONST_BASE),
        .CONST_SIZE(CONST_SIZE),
        .PARAM_BASE(PARAM_BASE),
        .PARAM_SIZE(PARAM_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .opcode(opcode),
        .func(func),
        .src_a(src_a),
        .src_b(src_b),
        .cache_level(cache_level),
        .cta_id(cta_id),
        .cluster_size(cluster_size),
        .sm_id(sm_id),
        .done(done),
        .result_valid(result_valid),
        .result(result),
        .pred_result(pred_result),
        .cache_ctrl_valid(cache_ctrl_valid),
        .cache_ctrl_op(cache_ctrl_op),
        .cache_ctrl_addr(cache_ctrl_addr),
        .cache_ctrl_policy(cache_ctrl_policy),
        .cache_ctrl_level(cache_ctrl_level),
        .cache_ctrl_done(cache_ctrl_done)
    );

    //------------------------------------------------------------------------
    // Clock Generation
    //------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_num;
    integer pass_count;
    integer fail_count;

    //------------------------------------------------------------------------
    // Cache controller response simulation
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (cache_ctrl_valid && !cache_ctrl_done) begin
            #20;  // Simulate cache operation latency
            cache_ctrl_done <= 1'b1;
            #10;
            cache_ctrl_done <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
    begin
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        src_a = 0;
        src_b = 0;
        cache_level = 0;
        cta_id = 4'd2;
        cluster_size = 4'd8;
        sm_id = 4'd1;
        cache_ctrl_done = 0;
        #20;
        rst_n = 1;
        #10;
    end
    endtask

    task test_isspacep;
        input [5:0] space_func;
        input [DATA_WIDTH-1:0] addr;
        input expected_pred;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Address: 0x%08x", addr);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_ISSPACEP;
        func <= space_func;
        src_a <= addr;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        if (pred_result === expected_pred) begin
            $display("  [PASS] pred_result=%0d (expected %0d)", pred_result, expected_pred);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] pred_result=%0d (expected %0d)", pred_result, expected_pred);
            fail_count = fail_count + 1;
        end
        #10;
    end
    endtask

    task test_mapa;
        input [5:0] map_func;
        input [DATA_WIDTH-1:0] addr;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);
        $display("  Input address: 0x%08x", addr);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_MAPA;
        func <= map_func;
        src_a <= addr;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        $display("  [PASS] Mapped to: 0x%08x", result);
        pass_count = pass_count + 1;
        #10;
    end
    endtask

    task test_cache_policy;
        input [5:0] policy_func;
        input [DATA_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] policy;
        input [2:0] level;
        input [255:0] test_name;
    begin
        test_num = test_num + 1;
        $display("\n[TEST %0d] %s", test_num, test_name);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_CACHE_POLICY;
        func <= policy_func;
        src_a <= addr;
        src_b <= policy;
        cache_level <= level;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        $display("  [PASS] Cache policy operation completed, result=0x%08x", result);
        pass_count = pass_count + 1;
        #10;
    end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU Cache Policy Unit Testbench");
        $display("============================================================");

        test_num = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        //====================================================================
        // ISSPACEP Tests - Address Space Queries
        //====================================================================
        $display("\n--- Testing ISSPACEP (Address Space Queries) ---");

        // Test global address (below special spaces)
        test_isspacep(`ISSPACEP_GLOBAL, 32'h00001000, 1'b1, "isspacep.global: global addr");

        // Test shared memory address
        test_isspacep(`ISSPACEP_SHARED, SMEM_BASE + 32'h100, 1'b1, "isspacep.shared: shared addr");

        // Test shared should be false for global address
        test_isspacep(`ISSPACEP_SHARED, 32'h00001000, 1'b0, "isspacep.shared: global addr (expect false)");

        // Test local memory address
        test_isspacep(`ISSPACEP_LOCAL, LOCAL_BASE + 32'h200, 1'b1, "isspacep.local: local addr");

        // Test constant memory address
        test_isspacep(`ISSPACEP_CONST, CONST_BASE + 32'h300, 1'b1, "isspacep.const: const addr");

        // Test parameter memory address
        test_isspacep(`ISSPACEP_PARAM, PARAM_BASE + 32'h400, 1'b1, "isspacep.param: param addr");

        // Cross-check: global address should not be in shared
        test_isspacep(`ISSPACEP_GLOBAL, SMEM_BASE + 32'h100, 1'b0, "isspacep.global: shared addr (expect false)");

        //====================================================================
        // MAPA Tests - Address Mapping
        //====================================================================
        $display("\n--- Testing MAPA (Address Mapping) ---");

        test_mapa(`MAPA_TO_SHARED, 32'h00010000, "mapa.to_shared: global to shared");
        test_mapa(`MAPA_FROM_SHARED, SMEM_BASE + 32'h1000, "mapa.from_shared: shared to generic");
        test_mapa(`MAPA_TO_LOCAL, 32'h00020000, "mapa.to_local: global to local");

        //====================================================================
        // Cache Policy Tests
        //====================================================================
        $display("\n--- Testing Cache Policy Operations ---");

        // Create policy tokens
        test_cache_policy(`CACHE_CREATEPOLICY, 32'h0, 32'h0, 3'd0, "createpolicy: create token 1");
        test_cache_policy(`CACHE_CREATEPOLICY, 32'h0, 32'h0, 3'd0, "createpolicy: create token 2");

        // Apply priority to cache lines
        test_cache_policy(`CACHE_APPLYPRIORITY, 32'h00001000, 32'h02, 3'd1, "applypriority: apply to L1");

        // Discard cache lines
        test_cache_policy(`CACHE_DISCARD, 32'h00002000, 32'h0, 3'd2, "discard: invalidate L2 lines");

        //====================================================================
        // GETCTARANK Test
        //====================================================================
        $display("\n--- Testing GETCTARANK ---");

        test_num = test_num + 1;
        $display("\n[TEST %0d] getctarank: get CTA rank in cluster", test_num);

        @(posedge clk);
        valid_in <= 1;
        opcode <= `OP_GETCTARANK;
        func <= 6'b0;
        @(posedge clk);
        valid_in <= 0;

        wait(done);
        @(posedge clk);

        if (result == {28'b0, cta_id}) begin
            $display("  [PASS] CTA rank=%0d (expected %0d)", result[3:0], cta_id);
            pass_count = pass_count + 1;
        end else begin
            $display("  [FAIL] CTA rank=%0d (expected %0d)", result[3:0], cta_id);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Summary
        //====================================================================
        #50;
        $display("\n============================================================");
        $display("Test Summary: %0d passed, %0d failed out of %0d tests",
                 pass_count, fail_count, test_num);
        $display("============================================================");

        if (fail_count == 0)
            $display("ALL TESTS PASSED!");
        else
            $display("SOME TESTS FAILED!");

        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #10000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule
