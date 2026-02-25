`timescale 1ns / 1ps

module de10_nano_demo_top (
    input  wire       CLOCK_50,
    input  wire [1:0] KEY,
    output wire [7:0] LED,
    output wire       UART_TX
);
    wire [3:0] demo_led;

    ralph_gpu_fpga_demo_top #(
        .CLK_HZ(50000000),
        .UART_BAUD(115200),
        .HEARTBEAT_HZ(2),
        .DEMO_GAP_MS(500)
    ) u_demo (
        .clk_i(CLOCK_50),
        .rst_ni(KEY[0]),
        .led_o(demo_led),
        .uart_tx_o(UART_TX)
    );

    assign LED[3:0] = demo_led;
    assign LED[7:4] = 4'b0000;
endmodule
