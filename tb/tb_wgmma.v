`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_wgmma;
    localparam CLK_PERIOD = 10;

    reg         clk;
    reg         rst_n;
    reg  [5:0]  func;
    reg         valid_in;
    reg  [2:0]  warpgroup_id;
    reg  [3:0]  wait_count;
    reg  [63:0] desc_a;
    reg  [63:0] desc_b;
    reg  [31:0] scale_d;
    reg  [511:0] data_a;
    reg  [511:0] data_b;
    reg  [1023:0] accum_in;
    wire [1023:0] accum_out;
    wire        ready;
    wire        done;
    wire [3:0]  pending_ops;

    integer pass_count;
    integer fail_count;
    reg [31:0] done_pulses;

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    wgmma dut (
        .clk(clk),
        .rst_n(rst_n),
        .func(func),
        .valid_in(valid_in),
        .warpgroup_id(warpgroup_id),
        .wait_count(wait_count),
        .desc_a(desc_a),
        .desc_b(desc_b),
        .scale_d(scale_d),
        .data_a(data_a),
        .data_b(data_b),
        .accum_in(accum_in),
        .accum_out(accum_out),
        .ready(ready),
        .done(done),
        .pending_ops(pending_ops)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_pulses <= 32'd0;
        end else if (done) begin
            done_pulses <= done_pulses + 1;
        end
    end

    function [31:0] lane32;
        input [1023:0] vec;
        input integer lane;
        begin
            lane32 = vec[lane*32 +: 32];
        end
    endfunction

    function [31:0] expected_int8_lane;
        input [31:0] acc;
        input signed [7:0] a;
        input signed [7:0] b;
        reg signed [31:0] prod;
        begin
            prod = a * b;
            expected_int8_lane = acc + prod + prod + prod + prod;
        end
    endfunction

    task check_eq32;
        input [255:0] name;
        input [31:0] exp;
        input [31:0] got;
        begin
            if (exp === got) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s exp=0x%08h got=0x%08h", name, exp, got);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s exp=0x%08h got=0x%08h", name, exp, got);
            end
        end
    endtask

    task check_true;
        input [255:0] name;
        input cond;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s", name);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", name);
            end
        end
    endtask

    task clear_vectors;
        begin
            data_a = 512'b0;
            data_b = 512'b0;
            accum_in = 1024'b0;
        end
    endtask

    task set_int8_dtype;
        begin
            desc_a = {8'h00, 4'h0, 4'h5, 16'h0010, 32'h00001000};
            desc_b = {8'h00, 4'h0, 4'h5, 16'h0010, 32'h00002000};
        end
    endtask

    task set_lane_int8;
        input integer lane;
        input signed [7:0] a;
        input signed [7:0] b;
        begin
            data_a[lane*8 +: 8] = a[7:0];
            data_b[lane*8 +: 8] = b[7:0];
        end
    endtask

    task set_lane_accum;
        input integer lane;
        input [31:0] val;
        begin
            accum_in[lane*32 +: 32] = val;
        end
    endtask

    task issue_cmd;
        input [5:0] cmd_func;
        input [3:0] cmd_wait;
        begin
            @(posedge clk);
            while (!ready) @(posedge clk);
            func <= cmd_func;
            wait_count <= cmd_wait;
            warpgroup_id <= 3'd0;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
            func <= 6'b0;
            wait_count <= 4'b0;
        end
    endtask

    task wait_ready_timeout;
        input integer timeout_cycles;
        integer c;
        begin
            c = 0;
            while (!ready && c < timeout_cycles) begin
                @(posedge clk);
                c = c + 1;
            end
            if (!ready) begin
                $fatal(1, "Timeout waiting for ready");
            end
        end
    endtask

    initial begin
        $dumpfile("tb_wgmma.vcd");
        $dumpvars(0, tb_wgmma);

        rst_n = 1'b0;
        valid_in = 1'b0;
        func = 6'b0;
        warpgroup_id = 3'b0;
        wait_count = 4'b0;
        desc_a = 64'b0;
        desc_b = 64'b0;
        scale_d = 32'h3F800000;
        data_a = 512'b0;
        data_b = 512'b0;
        accum_in = 1024'b0;
        pass_count = 0;
        fail_count = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        check_true("reset: ready asserted", ready);
        check_eq32("reset: pending_ops", 32'd0, {28'd0, pending_ops});

        // Test 2: INT8 MMA dispatch + writeback correctness
        clear_vectors();
        set_int8_dtype();
        set_lane_int8(0, 8'sd2, 8'sd3);
        set_lane_int8(1, -8'sd4, 8'sd5);
        issue_cmd(`WGMMA_M64N8K16, 4'd0);
        @(posedge clk);
        check_true("mma #1 accepted (pending>0)", pending_ops > 0);
        wait_ready_timeout(100);
        check_eq32("mma #1 pending_ops back to 0", 32'd0, {28'd0, pending_ops});
        check_eq32("mma #1 lane0", expected_int8_lane(32'd0, 8'sd2, 8'sd3), lane32(accum_out, 0));
        check_eq32("mma #1 lane1", expected_int8_lane(32'd0, -8'sd4, 8'sd5), lane32(accum_out, 1));

        // Test 3: Partial-tile edge (inactive lanes preserve C tile)
        clear_vectors();
        set_int8_dtype();
        set_lane_accum(8, 32'h12345678);
        set_lane_int8(0, 8'sd1, 8'sd1);
        set_lane_int8(1, 8'sd2, 8'sd2);
        set_lane_int8(2, 8'sd3, 8'sd3);
        set_lane_int8(3, 8'sd4, 8'sd4);
        issue_cmd(`WGMMA_M64N16K16, 4'd0);
        @(posedge clk);
        check_true("mma #2 accepted (pending>0)", pending_ops > 0);
        wait_ready_timeout(100);
        check_eq32("partial tile lane8 preserved", 32'h12345678, lane32(accum_out, 8));
        check_eq32("partial tile lane0 updated", expected_int8_lane(32'd0, 8'sd1, 8'sd1), lane32(accum_out, 0));

        // Test 4: Overflow edge stimulus
        clear_vectors();
        set_int8_dtype();
        set_lane_accum(0, 32'h7FFFFF00);
        set_lane_int8(0, 8'sd127, 8'sd127);
        issue_cmd(`WGMMA_M64N32K16, 4'd0);
        @(posedge clk);
        check_true("mma #3 accepted (pending>0)", pending_ops > 0);
        wait_ready_timeout(100);
        check_eq32("overflow wrap behavior", expected_int8_lane(32'h7FFFFF00, 8'sd127, 8'sd127), lane32(accum_out, 0));

        // Test 5: wait_group/commit/fence/unknown control paths
        begin
            reg [31:0] done_before;

            done_before = done_pulses;
            issue_cmd(`WGMMA_WAIT_GROUP, 4'd0);
            repeat (2) @(posedge clk);
            check_true("wait_group done pulse observed", done_pulses > done_before);
            check_eq32("wait_group keeps pending=0", 32'd0, {28'd0, pending_ops});

            done_before = done_pulses;
            issue_cmd(`WGMMA_COMMIT_GROUP, 4'd0);
            repeat (2) @(posedge clk);
            check_true("commit_group done pulse observed", done_pulses > done_before);
            check_eq32("commit_group keeps pending=0", 32'd0, {28'd0, pending_ops});

            done_before = done_pulses;
            issue_cmd(`WGMMA_FENCE, 4'd0);
            wait_ready_timeout(20);
            repeat (2) @(posedge clk);
            check_eq32("fence keeps pending=0", 32'd0, {28'd0, pending_ops});

            done_before = done_pulses;
            issue_cmd(6'h3F, 4'd0);
            repeat (2) @(posedge clk);
            check_true("unknown opcode done pulse observed", done_pulses > done_before);
            check_eq32("unknown opcode pending=0", 32'd0, {28'd0, pending_ops});
            check_true("unknown opcode ready", ready);
        end

        $display("============================================================");
        $display("tb_wgmma Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_wgmma failed");
        end
    end

    initial begin
        #200000;
        $fatal(1, "tb_wgmma timeout");
    end

endmodule
