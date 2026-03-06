`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_sm_wbq_bank;
    // Matching sm_wbq_bank defaults
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam SIMD_WIDTH = NUM_LANES * 32;
    localparam WARP_ID_W  = $clog2(`WARPS_PER_SM);

    reg clk;
    reg rst_n;

    // FU pipeline inputs
    reg [WARP_ID_W-1:0]  alu_warp;
    reg [4:0]            alu_rd;
    reg [NUM_LANES-1:0]  alu_mask;
    reg [SIMD_WIDTH-1:0] alu_data;
    reg                  alu_valid;

    reg [WARP_ID_W-1:0]  mul_warp;
    reg [4:0]            mul_rd;
    reg [NUM_LANES-1:0]  mul_mask;
    reg [SIMD_WIDTH-1:0] mul_data;
    reg                  mul_valid;

    // Pop signals
    reg alu_pop, mul_pop;

    // Outputs
    wire [WARP_ID_W-1:0]  alu_wbq_warp;
    wire [4:0]            alu_wbq_rd;
    wire [NUM_LANES-1:0]  alu_wbq_mask;
    wire [SIMD_WIDTH-1:0] alu_wbq_data;
    wire                  alu_wbq_full, alu_wbq_empty;

    wire [WARP_ID_W-1:0]  mul_wbq_warp;
    wire [4:0]            mul_wbq_rd;
    wire [NUM_LANES-1:0]  mul_wbq_mask;
    wire [SIMD_WIDTH-1:0] mul_wbq_data;
    wire                  mul_wbq_full, mul_wbq_empty;

    // Instantiate DUT
    sm_wbq_bank #(
        .ALU_WBQ_DEPTH(2),
        .MUL_WBQ_DEPTH(2)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .alu_warp(alu_warp), .alu_rd(alu_rd), .alu_mask(alu_mask), .alu_data(alu_data), .alu_valid(alu_valid),
        .mul_warp(mul_warp), .mul_rd(mul_rd), .mul_mask(mul_mask), .mul_data(mul_data), .mul_valid(mul_valid),
        .fpu32_valid(1'b0), .fpu64_valid(1'b0), .fp16_valid(1'b0), .sfu_valid(1'b0),
        .shfl_valid(1'b0), .video_valid(1'b0), .special_valid(1'b0),
        .alu_pop(alu_pop), .mul_pop(mul_pop), .fpu32_pop(1'b0), .fpu64_pop(1'b0), .fp16_pop(1'b0), .sfu_pop(1'b0),
        .shfl_pop(1'b0), .video_pop(1'b0), .special_pop(1'b0),
        // Outputs
        .alu_wbq_warp(alu_wbq_warp), .alu_wbq_rd(alu_wbq_rd), .alu_wbq_mask(alu_wbq_mask), .alu_wbq_data(alu_wbq_data),
        .alu_wbq_full(alu_wbq_full), .alu_wbq_empty(alu_wbq_empty),
        .mul_wbq_warp(mul_wbq_warp), .mul_wbq_rd(mul_wbq_rd), .mul_wbq_mask(mul_wbq_mask), .mul_wbq_data(mul_wbq_data),
        .mul_wbq_full(mul_wbq_full), .mul_wbq_empty(mul_wbq_empty)
    );

    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task check_alu;
        input [WARP_ID_W-1:0] exp_warp;
        input [4:0]           exp_rd;
        input [SIMD_WIDTH-1:0] exp_data;
        begin
            if (alu_wbq_warp === exp_warp && alu_wbq_rd === exp_rd && alu_wbq_data === exp_data)
                pass_count = pass_count + 1;
            else begin
                $display("FAIL: ALU Data Mismatch! Got warp=%0d rd=%0d", alu_wbq_warp, alu_wbq_rd);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        clk = 0; rst_n = 0;
        alu_valid = 0; mul_valid = 0; alu_pop = 0; mul_pop = 0;
        alu_mask = -1; mul_mask = -1;
        #20 rst_n = 1;

        // Test 1: Full and integrity
        @(negedge clk);
        alu_valid = 1; alu_warp = 1; alu_rd = 10; alu_data = 32'hAAAA_AAAA;
        @(negedge clk);
        alu_warp = 2; alu_rd = 11; alu_data = 32'hBBBB_BBBB;
        @(negedge clk);
        alu_valid = 0;
        
        #1;
        if (alu_wbq_full) $display("PASS: ALU WBQ is full");
        else begin $display("FAIL: ALU WBQ not full"); fail_count = fail_count + 1; end

        check_alu(1, 10, 32'hAAAA_AAAA);
        alu_pop = 1;
        @(negedge clk);
        #1;
        check_alu(2, 11, 32'hBBBB_BBBB);
        @(negedge clk);
        alu_pop = 0;

        #1;
        if (alu_wbq_empty) $display("PASS: ALU WBQ is empty");
        else begin $display("FAIL: ALU WBQ not empty"); fail_count = fail_count + 1; end

        $display("SM WBQ Bank Test: %0d PASSED, %0d FAILED", pass_count, fail_count);
        if (fail_count == 0) $display("*** ALL TESTS PASSED ***");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end
endmodule
