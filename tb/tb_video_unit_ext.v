`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_video_unit_ext;
    reg clk, rst_n;
    reg [5:0] func;
    reg valid_in, is_signed;
    reg [31:0] op_a, op_b, op_c;
    wire [31:0] result;
    wire valid_out, overflow, saturate_flag;

    video_unit dut (
        .clk(clk), .rst_n(rst_n),
        .func(func), .valid_in(valid_in), .is_signed(is_signed),
        .operand_a(op_a), .operand_b(op_b), .operand_c(op_c),
        .result(result), .valid_out(valid_out),
        .overflow(overflow), .saturate_flag(saturate_flag)
    );

    // SIMD Unit for integration test
    localparam NL = 4;
    reg [NL*32-1:0] sim_a, sim_b, sim_c;
    reg [NL-1:0] sim_mask;
    wire [NL*32-1:0] sim_res;
    wire sim_valid;

    video_simd_unit #(.NUM_LANES(NL)) dut_simd (
        .clk(clk), .rst_n(rst_n),
        .func(func), .valid_in(valid_in), .is_signed(is_signed),
        .operand_a(sim_a), .operand_b(sim_b), .operand_c(sim_c),
        .lane_mask(sim_mask),
        .result(sim_res), .valid_out(sim_valid)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass = 0, fail = 0;

    task check_unit;
        input [31:0] exp;
        input [255:0] msg;
        begin
            @(posedge clk); // Stage 1
            @(posedge clk); // Stage 2
            #1;
            if (result === exp) begin
                pass = pass + 1;
                $display("[PASS] %0s", msg);
            end else begin
                fail = fail + 1;
                $display("[FAIL] %0s: got %h exp %h", msg, result, exp);
            end
        end
    endtask

    initial begin
        rst_n = 0; valid_in = 0; #20 rst_n = 1; #10;

        $display("--- Ext Test 1: Saturated ADD (Theoretical) ---");
        // Note: PTX vadd.u32.u32.u32.sat is handled in scalar ALU, 
        // but SIMD saturation (vadd4.sat) should be here.
        // Current RTL doesn't have .sat input yet, this test documents the gap.
        
        $display("--- Ext Test 2: SIMD Lane Masking ---");
        @(negedge clk);
        func = `VIDEO_VADD; is_signed = 0;
        valid_in = 1;
        sim_a = {32'd40, 32'd30, 32'd20, 32'd10};
        sim_b = {32'd4, 32'd3, 32'd2, 32'd1};
        sim_mask = 4'b1010; // Only lanes 1 and 3 active
        
        @(posedge clk); valid_in = 0;
        repeat(2) @(posedge clk); #1;
        
        if (sim_res[31:0] === 32'h0 && sim_res[63:32] === 32'd22 && 
            sim_res[95:64] === 32'h0 && sim_res[127:96] === 32'd44) begin
            $display("[PASS] SIMD Masking: Lanes 0 and 2 suppressed");
            pass = pass + 1;
        end else begin
            $display("[FAIL] SIMD Masking: got %h", sim_res);
            fail = fail + 1;
        end

        $display("Summary: %0d PASS, %0d FAIL", pass, fail);
        if (fail > 0) $fatal(1, "Test Failed");
        $finish;
    end
endmodule
