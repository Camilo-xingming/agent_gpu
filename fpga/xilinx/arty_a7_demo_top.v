`timescale 1ns / 1ps

module arty_a7_demo_top (
    input  wire       CLK100MHZ,
    input  wire       CPU_RESETN,
    output wire [3:0] LED,
    output wire       UART_TXD
);
    ralph_gpu_fpga_demo_top #(
        .CLK_HZ(100000000),
        .UART_BAUD(115200),
        .HEARTBEAT_HZ(2),
        .DEMO_GAP_MS(500)
    ) u_demo (
        .clk_i(CLK100MHZ),
        .rst_ni(CPU_RESETN),
        .led_o(LED),
        .uart_tx_o(UART_TXD)
    );
endmodule
