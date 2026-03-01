//============================================================================
// Testbench for command_queue ring buffer (Issue #271)
//============================================================================

`timescale 1ns / 1ps

module tb_command_queue;

    localparam CLK_PERIOD = 10;
    localparam DEPTH = 4;
    localparam PTR_W = $clog2(DEPTH);

    reg clk;
    reg rst_n;

    reg        push_valid;
    wire       push_ready;
    reg [31:0] push_kernel_pc;
    reg [31:0] push_grid_dim_x;
    reg [31:0] push_grid_dim_y;
    reg [31:0] push_grid_dim_z;
    reg [31:0] push_block_dim_x;
    reg [31:0] push_block_dim_y;
    reg [31:0] push_block_dim_z;
    reg [31:0] push_shared_mem_size;
    reg [31:0] push_kernel_params_base;
    reg [31:0] push_fence_id;
    reg [31:0] push_flags;

    wire       pop_valid;
    reg        pop_ready;
    wire [31:0] pop_kernel_pc;
    wire [31:0] pop_grid_dim_x;
    wire [31:0] pop_grid_dim_y;
    wire [31:0] pop_grid_dim_z;
    wire [31:0] pop_block_dim_x;
    wire [31:0] pop_block_dim_y;
    wire [31:0] pop_block_dim_z;
    wire [31:0] pop_shared_mem_size;
    wire [31:0] pop_kernel_params_base;
    wire [31:0] pop_fence_id;
    wire [31:0] pop_flags;

    wire [PTR_W-1:0] head;
    wire [PTR_W-1:0] tail;
    wire [PTR_W:0]   count;

    integer total_tests;
    integer passed_tests;
    integer failed_tests;

    command_queue #(
        .DEPTH(DEPTH),
        .PTR_WIDTH(PTR_W)
    ) u_queue (
        .clk                    (clk),
        .rst_n                  (rst_n),
        .push_valid             (push_valid),
        .push_ready             (push_ready),
        .push_kernel_pc         (push_kernel_pc),
        .push_grid_dim_x        (push_grid_dim_x),
        .push_grid_dim_y        (push_grid_dim_y),
        .push_grid_dim_z        (push_grid_dim_z),
        .push_block_dim_x       (push_block_dim_x),
        .push_block_dim_y       (push_block_dim_y),
        .push_block_dim_z       (push_block_dim_z),
        .push_shared_mem_size   (push_shared_mem_size),
        .push_kernel_params_base(push_kernel_params_base),
        .push_fence_id          (push_fence_id),
        .push_flags             (push_flags),
        .pop_valid              (pop_valid),
        .pop_ready              (pop_ready),
        .pop_kernel_pc          (pop_kernel_pc),
        .pop_grid_dim_x         (pop_grid_dim_x),
        .pop_grid_dim_y         (pop_grid_dim_y),
        .pop_grid_dim_z         (pop_grid_dim_z),
        .pop_block_dim_x        (pop_block_dim_x),
        .pop_block_dim_y        (pop_block_dim_y),
        .pop_block_dim_z        (pop_block_dim_z),
        .pop_shared_mem_size    (pop_shared_mem_size),
        .pop_kernel_params_base (pop_kernel_params_base),
        .pop_fence_id           (pop_fence_id),
        .pop_flags              (pop_flags),
        .head                   (head),
        .tail                   (tail),
        .count                  (count)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    task check_val(input [255:0] name, input [31:0] actual, input [31:0] expected);
        begin
            total_tests = total_tests + 1;
            if (actual === expected) begin
                passed_tests = passed_tests + 1;
                $display("[PASS] %0s: 0x%08h", name, actual);
            end else begin
                failed_tests = failed_tests + 1;
                $display("[FAIL] %0s: expected 0x%08h got 0x%08h", name, expected, actual);
            end
        end
    endtask

    task drive_push(input [31:0] base);
        begin
            @(negedge clk);
            if (!push_ready) begin
                failed_tests = failed_tests + 1;
                total_tests = total_tests + 1;
                $display("[FAIL] push_ready was low before push");
            end
            push_kernel_pc          <= 32'h1000_0000 + base;
            push_grid_dim_x         <= base + 32'd1;
            push_grid_dim_y         <= base + 32'd2;
            push_grid_dim_z         <= base + 32'd3;
            push_block_dim_x        <= base + 32'd4;
            push_block_dim_y        <= base + 32'd5;
            push_block_dim_z        <= base + 32'd6;
            push_shared_mem_size    <= base + 32'd7;
            push_kernel_params_base <= 32'h2000_0000 + base;
            push_fence_id           <= base + 32'd8;
            push_flags              <= base + 32'd9;
            push_valid              <= 1'b1;
            @(posedge clk);
            #1;
            push_valid <= 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    task expect_front(input [31:0] base);
        begin
            check_val("front.kernel_pc", pop_kernel_pc, 32'h1000_0000 + base);
            check_val("front.grid_x", pop_grid_dim_x, base + 32'd1);
            check_val("front.block_z", pop_block_dim_z, base + 32'd6);
            check_val("front.shared_mem", pop_shared_mem_size, base + 32'd7);
            check_val("front.params_base", pop_kernel_params_base, 32'h2000_0000 + base);
            check_val("front.fence", pop_fence_id, base + 32'd8);
        end
    endtask

    task drive_pop;
        begin
            @(negedge clk);
            if (!pop_valid) begin
                failed_tests = failed_tests + 1;
                total_tests = total_tests + 1;
                $display("[FAIL] pop_valid was low before pop");
            end
            pop_ready <= 1'b1;
            @(posedge clk);
            #1;
            pop_ready <= 1'b0;
            @(posedge clk);
            #1;
        end
    endtask

    initial begin
        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;

        rst_n = 1'b0;
        push_valid = 1'b0;
        pop_ready = 1'b0;
        push_kernel_pc = 32'b0;
        push_grid_dim_x = 32'b0;
        push_grid_dim_y = 32'b0;
        push_grid_dim_z = 32'b0;
        push_block_dim_x = 32'b0;
        push_block_dim_y = 32'b0;
        push_block_dim_z = 32'b0;
        push_shared_mem_size = 32'b0;
        push_kernel_params_base = 32'b0;
        push_fence_id = 32'b0;
        push_flags = 32'b0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // Reset state
        check_val("reset.count", count, 0);
        check_val("reset.head", head, 0);
        check_val("reset.tail", tail, 0);
        check_val("reset.pop_valid", {31'b0, pop_valid}, 0);

        // Push A/B and validate order
        drive_push(32'd1);
        check_val("after_pushA.count", count, 1);
        check_val("after_pushA.pop_valid", {31'b0, pop_valid}, 1);
        expect_front(32'd1);

        drive_push(32'd10);
        check_val("after_pushB.count", count, 2);
        expect_front(32'd1);

        drive_pop;
        check_val("after_popA.count", count, 1);
        expect_front(32'd10);

        // Simultaneous push + pop, occupancy should stay constant
        @(negedge clk);
        push_kernel_pc          <= 32'h1000_0020;
        push_grid_dim_x         <= 32'd21;
        push_grid_dim_y         <= 32'd22;
        push_grid_dim_z         <= 32'd23;
        push_block_dim_x        <= 32'd24;
        push_block_dim_y        <= 32'd25;
        push_block_dim_z        <= 32'd26;
        push_shared_mem_size    <= 32'd27;
        push_kernel_params_base <= 32'h2000_0020;
        push_fence_id           <= 32'd28;
        push_flags              <= 32'd29;
        push_valid              <= 1'b1;
        pop_ready               <= 1'b1;
        @(posedge clk);
        #1;
        push_valid <= 1'b0;
        pop_ready  <= 1'b0;
        @(posedge clk);
        #1;

        check_val("simul_push_pop.count", count, 1);
        check_val("simul_push_pop.front_pc", pop_kernel_pc, 32'h1000_0020);

        // Drain remaining entry
        drive_pop;
        check_val("after_drain.count", count, 0);
        check_val("after_drain.pop_valid", {31'b0, pop_valid}, 0);

        $display("====================================");
        $display("Total: %0d  Passed: %0d  Failed: %0d", total_tests, passed_tests, failed_tests);
        $display("====================================");

        if (failed_tests == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

    initial begin
        #100000;
        $display("[TIMEOUT] tb_command_queue timed out");
        $finish;
    end

endmodule
