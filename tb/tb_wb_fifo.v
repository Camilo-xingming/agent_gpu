`timescale 1ns / 1ps

module tb_wb_fifo;

    localparam WIDTH = 32;
    localparam DEPTH = 4;

    reg                 clk;
    reg                 rst_n;
    reg                 push;
    reg [WIDTH-1:0]     push_data;
    reg                 pop;

    wire [WIDTH-1:0]    pop_data;
    wire                full;
    wire                empty;
    wire                dropped;

    wb_fifo #(
        .WIDTH(WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .push(push),
        .push_data(push_data),
        .pop(pop),
        .pop_data(pop_data),
        .full(full),
        .empty(empty),
        .dropped(dropped)
    );

    always #5 clk = ~clk;

    task automatic assert_true(input bit cond, input string msg);
    begin
        if (!cond) begin
            $display("FAIL: %s (t=%0t)", msg, $time);
            $fatal(1);
        end
    end
    endtask

    task automatic push_word(input [WIDTH-1:0] data);
    begin
        @(negedge clk);
        push = 1'b1;
        push_data = data;
        pop = 1'b0;
        #1;
        assert_true(!dropped, "push should not drop when space is available");
        @(posedge clk);
        #1;
        push = 1'b0;
    end
    endtask

    task automatic pop_word(input [WIDTH-1:0] exp_data);
    begin
        @(negedge clk);
        push = 1'b0;
        pop = 1'b1;
        #1;
        assert_true(!empty, "pop_word requires non-empty fifo");
        assert_true(pop_data === exp_data, "pop data mismatch");
        @(posedge clk);
        #1;
        pop = 1'b0;
    end
    endtask

    task automatic push_pop_word(
        input [WIDTH-1:0] push_val,
        input [WIDTH-1:0] exp_pop_val,
        input bit exp_drop
    );
    begin
        @(negedge clk);
        push = 1'b1;
        push_data = push_val;
        pop = 1'b1;
        #1;
        assert_true(pop_data === exp_pop_val, "simultaneous push/pop pop_data mismatch");
        assert_true(dropped == exp_drop, "unexpected dropped state during simultaneous push/pop");
        @(posedge clk);
        #1;
        push = 1'b0;
        pop = 1'b0;
    end
    endtask

    integer i;
    reg [WIDTH-1:0] drain_expect [0:DEPTH-1];

    initial begin
        $dumpfile("tb_wb_fifo.vcd");
        $dumpvars(0, tb_wb_fifo);

        clk = 1'b0;
        rst_n = 1'b0;
        push = 1'b0;
        pop = 1'b0;
        push_data = {WIDTH{1'b0}};

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        #1;

        // Reset + empty/full baseline
        assert_true(empty, "fifo should be empty after reset");
        assert_true(!full, "fifo should not be full after reset");
        assert_true(pop_data == {WIDTH{1'b0}}, "pop_data should be zero when empty");

        // Basic enqueue/dequeue flow + back-to-back operations
        push_word(32'h0000_0011);
        push_word(32'h0000_0022);
        push_word(32'h0000_0033);
        assert_true(!empty, "fifo should not be empty after pushes");
        assert_true(!full, "fifo should not be full with 3/4 entries");

        pop_word(32'h0000_0011);
        pop_word(32'h0000_0022);
        push_word(32'h0000_0044);
        pop_word(32'h0000_0033);
        pop_word(32'h0000_0044);
        assert_true(empty, "fifo should be empty after draining basic flow");

        // Underflow protection: pop on empty should hold zero and remain empty
        @(negedge clk);
        pop = 1'b1;
        push = 1'b0;
        #1;
        assert_true(empty, "empty should remain high on underflow pop");
        assert_true(pop_data == {WIDTH{1'b0}}, "underflow pop_data should stay zero");
        assert_true(!dropped, "underflow must not assert dropped");
        @(posedge clk);
        #1;
        pop = 1'b0;

        // Fill to full boundary
        push_word(32'hA0);
        push_word(32'hA1);
        push_word(32'hA2);
        push_word(32'hA3);
        assert_true(full, "fifo should assert full at depth boundary");
        assert_true(!empty, "fifo full implies non-empty");

        // Overflow protection: push without pop while full should set dropped
        @(negedge clk);
        push = 1'b1;
        push_data = 32'hDEAD_BEEF;
        pop = 1'b0;
        #1;
        assert_true(full, "fifo should still be full before overflow cycle commits");
        assert_true(dropped, "overflow push must assert dropped");
        @(posedge clk);
        #1;
        push = 1'b0;
        assert_true(full, "overflow push must not change occupancy");

        // Back-to-back operation at full: concurrent push+pop must not drop
        push_pop_word(32'hA4, 32'h0000_00A0, 1'b0);
        assert_true(full, "fifo should remain full after push+pop at full boundary");

        // Drain and verify ordering after overflow + concurrent operation
        drain_expect[0] = 32'h0000_00A1;
        drain_expect[1] = 32'h0000_00A2;
        drain_expect[2] = 32'h0000_00A3;
        drain_expect[3] = 32'h0000_00A4;

        for (i = 0; i < DEPTH; i = i + 1) begin
            pop_word(drain_expect[i]);
        end

        assert_true(empty, "fifo should be empty after final drain");
        assert_true(!full, "fifo should deassert full after final drain");

        $display("PASS: tb_wb_fifo all scenarios passed");
        $finish;
    end

endmodule
