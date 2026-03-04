`timescale 1ns / 1ps

module tb_fma_int32;
    localparam NUM_UNITS = 4;

    reg                         clk;
    reg                         rst_n;
    reg                         valid_in;
    reg [NUM_UNITS*32-1:0]      a;
    reg [NUM_UNITS*32-1:0]      b;
    reg [NUM_UNITS*32-1:0]      c;
    reg                         is_signed;

    wire                        valid_out;
    wire [NUM_UNITS*32-1:0]     result;

    integer pass_count;
    integer fail_count;
    integer i;

    fma_int32 #(
        .NUM_UNITS(NUM_UNITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .a(a),
        .b(b),
        .c(c),
        .is_signed(is_signed),
        .valid_out(valid_out),
        .result(result)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    function [31:0] fma_model;
        input [31:0] aa;
        input [31:0] bb;
        input [31:0] cc;
        input        signed_mode;
        reg signed [63:0] signed_sum;
        reg [63:0] unsigned_sum;
        begin
            if (signed_mode) begin
                signed_sum = ($signed(aa) * $signed(bb)) + $signed({{32{cc[31]}}, cc});
                fma_model = signed_sum[31:0];
            end else begin
                unsigned_sum = (aa * bb) + {32'b0, cc};
                fma_model = unsigned_sum[31:0];
            end
        end
    endfunction

    task clear_inputs;
        begin
            valid_in = 1'b0;
            a = {NUM_UNITS*32{1'b0}};
            b = {NUM_UNITS*32{1'b0}};
            c = {NUM_UNITS*32{1'b0}};
            is_signed = 1'b0;
        end
    endtask

    task check_current_result;
        input [255:0] case_name;
        input [NUM_UNITS*32-1:0] expected;
        reg [31:0] exp_lane;
        reg [31:0] got_lane;
        begin
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                exp_lane = expected[i*32 +: 32];
                got_lane = result[i*32 +: 32];
                if (got_lane !== exp_lane) begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] %s lane%0d exp=0x%08x got=0x%08x", case_name, i, exp_lane, got_lane);
                end else begin
                    pass_count = pass_count + 1;
                end
            end
        end
    endtask

    task run_vector_case;
        input [255:0] case_name;
        input         signed_mode;
        input [NUM_UNITS*32-1:0] vec_a;
        input [NUM_UNITS*32-1:0] vec_b;
        input [NUM_UNITS*32-1:0] vec_c;
        input [NUM_UNITS*32-1:0] expected;
        integer wait_cycles;
        begin
            @(posedge clk);
            valid_in <= 1'b1;
            a <= vec_a;
            b <= vec_b;
            c <= vec_c;
            is_signed <= signed_mode;

            @(posedge clk);
            valid_in <= 1'b0;
            a <= {NUM_UNITS*32{1'b0}};
            b <= {NUM_UNITS*32{1'b0}};
            c <= {NUM_UNITS*32{1'b0}};

            wait_cycles = 0;
            while (!valid_out && wait_cycles < 8) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end

            if (!valid_out) begin
                fail_count = fail_count + 1;
                $display("[FAIL] %s timeout waiting valid_out", case_name);
            end else begin
                check_current_result(case_name, expected);
            end
        end
    endtask

    reg [NUM_UNITS*32-1:0] exp;
    reg [NUM_UNITS*32-1:0] vec_a;
    reg [NUM_UNITS*32-1:0] vec_b;
    reg [NUM_UNITS*32-1:0] vec_c;
    reg [NUM_UNITS*32-1:0] exp_a;
    reg [NUM_UNITS*32-1:0] exp_b;
    integer t;
    integer pulse_count;

    initial begin
        pass_count = 0;
        fail_count = 0;

        clear_inputs();
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        if (valid_out !== 1'b0) begin
            fail_count = fail_count + 1;
            $display("[FAIL] reset valid_out should be 0");
        end else begin
            pass_count = pass_count + 1;
        end

        // Basic unsigned a*b+c
        vec_a = {32'd9, 32'd7, 32'd5, 32'd3};
        vec_b = {32'd8, 32'd6, 32'd4, 32'd2};
        vec_c = {32'd7, 32'd5, 32'd3, 32'd1};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b0);
        end
        run_vector_case("basic unsigned", 1'b0, vec_a, vec_b, vec_c, exp);

        // Signed with negative operands
        vec_a = {32'hFFFF_FFF9, 32'h0000_0008, 32'hFFFF_FFFD, 32'hFFFF_FFFE}; // -7, 8, -3, -2
        vec_b = {32'h0000_0003, 32'hFFFF_FFFC, 32'hFFFF_FFFA, 32'h0000_0006}; // 3, -4, -6, 6
        vec_c = {32'hFFFF_FFF0, 32'h0000_0002, 32'hFFFF_FFFF, 32'h0000_0001}; // -16,2,-1,1
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b1);
        end
        run_vector_case("signed mixed", 1'b1, vec_a, vec_b, vec_c, exp);

        // Zero behavior
        vec_a = {32'd0, 32'd123, 32'd0, 32'd456};
        vec_b = {32'd0, 32'd0, 32'd789, 32'd0};
        vec_c = {32'd0, 32'd5, 32'd6, 32'd7};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b0);
        end
        run_vector_case("zero operands", 1'b0, vec_a, vec_b, vec_c, exp);

        // Overflow / underflow wrap-around edges (non-saturating low32 behavior)
        vec_a = {32'hFFFF_FFFF, 32'h8000_0000, 32'h7FFF_FFFF, 32'hFFFF_FFFF};
        vec_b = {32'h0000_0002, 32'h0000_0002, 32'h0000_0002, 32'hFFFF_FFFF};
        vec_c = {32'h0000_0001, 32'hFFFF_FFFF, 32'h0000_0001, 32'h0000_0001};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b1);
        end
        run_vector_case("signed wrap edges", 1'b1, vec_a, vec_b, vec_c, exp);

        // Unsigned max/wrap edges
        vec_a = {32'hFFFF_FFFF, 32'hFFFF_FFFF, 32'h8000_0000, 32'h7FFF_FFFF};
        vec_b = {32'hFFFF_FFFF, 32'h0000_0002, 32'h0000_0002, 32'h0000_0002};
        vec_c = {32'h0000_0001, 32'h0000_0001, 32'hFFFF_FFFF, 32'hFFFF_FFFF};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b0);
        end
        run_vector_case("unsigned wrap edges", 1'b0, vec_a, vec_b, vec_c, exp);

        // Signed min/max edges
        vec_a = {32'h8000_0000, 32'h7FFF_FFFF, 32'h8000_0000, 32'h7FFF_FFFF};
        vec_b = {32'h0000_0001, 32'h0000_0001, 32'hFFFF_FFFF, 32'hFFFF_FFFF};
        vec_c = {32'h0000_0000, 32'h0000_0000, 32'h7FFF_FFFF, 32'h8000_0000};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b1);
        end
        run_vector_case("signed min/max edges", 1'b1, vec_a, vec_b, vec_c, exp);

        // Single-shot request should produce exactly one valid_out pulse
        vec_a = {32'd13, 32'd11, 32'd7, 32'd5};
        vec_b = {32'd4, 32'd3, 32'd2, 32'd1};
        vec_c = {32'd3, 32'd2, 32'd1, 32'd0};
        for (i = 0; i < NUM_UNITS; i = i + 1) begin
            exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b0);
        end

        @(posedge clk);
        valid_in <= 1'b1;
        a <= vec_a;
        b <= vec_b;
        c <= vec_c;
        is_signed <= 1'b0;
        @(posedge clk);
        valid_in <= 1'b0;
        a <= {NUM_UNITS*32{1'b0}};
        b <= {NUM_UNITS*32{1'b0}};
        c <= {NUM_UNITS*32{1'b0}};

        pulse_count = 0;
        for (t = 0; t < 8; t = t + 1) begin
            @(posedge clk);
            if (valid_out) begin
                pulse_count = pulse_count + 1;
                if (pulse_count == 1)
                    check_current_result("single-shot pulse", exp);
            end
        end
        if (pulse_count == 1)
            pass_count = pass_count + 1;
        else begin
            fail_count = fail_count + 1;
            $display("[FAIL] single-shot pulse expected 1 valid_out pulse, got %0d", pulse_count);
        end

        // Back-to-back throughput check (pipeline should emit consecutive valid_out)
        exp_a[0*32 +: 32] = fma_model(32'd10, 32'd1, 32'd1, 1'b0);
        exp_a[1*32 +: 32] = fma_model(32'd20, 32'd2, 32'd2, 1'b0);
        exp_a[2*32 +: 32] = fma_model(32'd30, 32'd3, 32'd3, 1'b0);
        exp_a[3*32 +: 32] = fma_model(32'd40, 32'd4, 32'd4, 1'b0);

        exp_b[0*32 +: 32] = fma_model(32'd6, 32'd2, 32'd2, 1'b0);
        exp_b[1*32 +: 32] = fma_model(32'd5, 32'd3, 32'd2, 1'b0);
        exp_b[2*32 +: 32] = fma_model(32'd4, 32'd4, 32'd2, 1'b0);
        exp_b[3*32 +: 32] = fma_model(32'd3, 32'd5, 32'd2, 1'b0);

        // transaction A
        @(posedge clk);
        valid_in <= 1'b1;
        a <= {32'd40, 32'd30, 32'd20, 32'd10};
        b <= {32'd4, 32'd3, 32'd2, 32'd1};
        c <= {32'd4, 32'd3, 32'd2, 32'd1};
        is_signed <= 1'b0;

        // transaction B (next cycle)
        @(posedge clk);
        valid_in <= 1'b1;
        a <= {32'd3, 32'd4, 32'd5, 32'd6};
        b <= {32'd5, 32'd4, 32'd3, 32'd2};
        c <= {32'd2, 32'd2, 32'd2, 32'd2};
        is_signed <= 1'b0;

        @(posedge clk);
        valid_in <= 1'b0;
        a <= {NUM_UNITS*32{1'b0}};
        b <= {NUM_UNITS*32{1'b0}};
        c <= {NUM_UNITS*32{1'b0}};

        t = 0;
        while (!valid_out && t < 8) begin
            @(posedge clk);
            t = t + 1;
        end
        if (!valid_out) begin
            fail_count = fail_count + 1;
            $display("[FAIL] throughput case A timeout");
        end else begin
            check_current_result("throughput case A", exp_a);
        end

        @(posedge clk);
        if (!valid_out) begin
            fail_count = fail_count + 1;
            $display("[FAIL] throughput case B missing consecutive valid_out");
        end else begin
            check_current_result("throughput case B", exp_b);
        end

        // Random regression (unsigned then signed)
        for (t = 0; t < 20; t = t + 1) begin
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                vec_a[i*32 +: 32] = $random;
                vec_b[i*32 +: 32] = $random;
                vec_c[i*32 +: 32] = $random;
                exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b0);
            end
            run_vector_case("random unsigned", 1'b0, vec_a, vec_b, vec_c, exp);
        end

        for (t = 0; t < 20; t = t + 1) begin
            for (i = 0; i < NUM_UNITS; i = i + 1) begin
                vec_a[i*32 +: 32] = $random;
                vec_b[i*32 +: 32] = $random;
                vec_c[i*32 +: 32] = $random;
                exp[i*32 +: 32] = fma_model(vec_a[i*32 +: 32], vec_b[i*32 +: 32], vec_c[i*32 +: 32], 1'b1);
            end
            run_vector_case("random signed", 1'b1, vec_a, vec_b, vec_c, exp);
        end

        $display("============================================================");
        $display("tb_fma_int32 Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_fma_int32 failed");
        end
    end

    initial begin
        #500000;
        $fatal(1, "tb_fma_int32 timeout");
    end

    initial begin
        $dumpfile("tb_fma_int32.vcd");
        $dumpvars(0, tb_fma_int32);
    end
endmodule
