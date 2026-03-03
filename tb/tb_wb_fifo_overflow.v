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

    // Instantiate with DEPTH=4
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
    reg [31:0] expected_data [0:3];
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
            if (i < 3) expected_data[i] = i + 2; // Data shifts due to concurrent push/pop later
        end
        // Correction: Initial state 1, 2, 3, 4. 
        // We will push 1, 2, 3, 4.
        // Then we do a concurrent push(FEEDFACE) and pop.
        // So expected sequence: 2, 3, 4, FEEDFACE (since 1 is popped first)
        expected_data[0] = 32'd2;
        expected_data[1] = 32'd3;
        expected_data[2] = 32'd4;
        expected_data[3] = 32'hFEED_FACE;

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
        
        #1;
        if (dropped) begin
            $display("PASS: 'dropped' signal high during push-while-full");
        end else begin
            $display("FAIL: 'dropped' signal should be high during push-while-full");
            $finish;
        end
        push = 0;
        
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
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            pop = 1;
            #1;
            if (pop_data !== expected_data[i]) begin
                $display("FAIL: Data mismatch at index %0d! Got %0h, expected %0h", i, pop_data, expected_data[i]);
                $finish;
            end
            $display("Popped: %0h (Correct)", pop_data);
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
