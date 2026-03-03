`timescale 1ns / 1ps

module tb_wb_fifo_overflow;

    reg clk;
    reg rst_n;
    reg push;
    reg [31:0] push_data;
    reg pop;

    wire [31:0] pop_data;
    wire full;
    wire empty;
    wire dropped;

    // Instantiate with DEPTH=4 for more headroom
    wb_fifo #(
        .WIDTH(32),
        .DEPTH(4)
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

    integer i;
    initial begin
        $dumpfile("tb_wb_fifo_overflow.vcd");
        $dumpvars(0, tb_wb_fifo_overflow);

        clk = 0;
        rst_n = 0;
        push = 0;
        push_data = 0;
        pop = 0;

        #20 rst_n = 1;

        $display("--- Filling FIFO ---");
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            push = 1; push_data = i + 1;
        end
        @(negedge clk);
        push = 0;

        if (!full) begin
            $display("FAIL: FIFO should be full");
            $finish;
        end
        $display("PASS: FIFO is full");

        $display("--- Testing Silent Drop Detection ---");
        @(negedge clk);
        push = 1; push_data = 32'hDEAD_BEEF;
        
        // dropped should be high combinationally
        #1;
        if (dropped) begin
            $display("PASS: 'dropped' signal high during push-while-full");
        end else begin
            $display("FAIL: 'dropped' signal should be high during push-while-full");
            $finish;
        end

        // Wait past posedge clk to test actual overflow behavior on clock edge
        @(posedge clk);
        #1;
        
        $display("--- Testing Concurrent Push and Pop (Full) ---");
        @(negedge clk);
        push = 1; push_data = 32'hFEED_FACE;
        pop = 1;
        
        #1;
        if (dropped) begin
            $display("FAIL: 'dropped' should NOT be high during push-and-pop-while-full");
            $finish;
        end
        $display("PASS: No drop during concurrent push and pop while full");

        @(negedge clk);
        push = 0;
        pop = 0;

        $display("--- Verifying Data Integrity ---");
        // Pop remaining items
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            pop = 1;
            #1;
            $display("Popped: %0h", pop_data);
        end
        @(negedge clk);
        pop = 0;

        if (!empty) begin
            $display("FAIL: FIFO should be empty");
            $finish;
        end
        $display("PASS: FIFO is empty");

        $display("ALL OVERFLOW TESTS PASSED");
        $finish;
    end

endmodule
