`timescale 1ns / 1ps

module tb_fpga_demo;
    reg clk;
    reg rst_n;
    wire [3:0] led;
    wire uart_tx;

    integer i;
    integer led_toggle_count;
    integer uart_edge_count;
    reg [3:0] led_last;
    reg uart_last;

    ralph_gpu_fpga_demo_top #(
        .CLK_HZ(1000000),
        .UART_BAUD(100000),
        .HEARTBEAT_HZ(500),
        .DEMO_GAP_MS(1)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .led_o(led),
        .uart_tx_o(uart_tx)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        led_toggle_count = 0;
        uart_edge_count = 0;
        led_last = 4'b0000;
        uart_last = 1'b1;

        $dumpfile("tb_fpga_demo.vcd");
        $dumpvars(0, tb_fpga_demo);

        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        for (i = 0; i < 12000; i = i + 1) begin
            @(posedge clk);

            if (led != led_last) begin
                led_toggle_count = led_toggle_count + 1;
                led_last = led;
            end

            if (uart_tx != uart_last) begin
                uart_edge_count = uart_edge_count + 1;
                uart_last = uart_tx;
            end
        end

        if (led_toggle_count == 0) begin
            $fatal(1, "FPGA demo LED did not toggle");
        end

        if (uart_edge_count == 0) begin
            $fatal(1, "FPGA demo UART did not toggle");
        end

        $display("PASS: tb_fpga_demo (led_toggles=%0d, uart_edges=%0d)", led_toggle_count, uart_edge_count);
        $finish;
    end
endmodule
