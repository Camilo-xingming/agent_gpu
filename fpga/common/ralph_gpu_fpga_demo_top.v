`timescale 1ns / 1ps

module ralph_gpu_fpga_demo_top #(
    parameter integer CLK_HZ       = 50000000,
    parameter integer UART_BAUD    = 115200,
    parameter integer HEARTBEAT_HZ = 2,
    parameter integer DEMO_GAP_MS  = 500
)(
    input  wire       clk_i,
    input  wire       rst_ni,
    output reg  [3:0] led_o,
    output wire       uart_tx_o
);
    localparam integer UART_BAUD_SAFE    = (UART_BAUD < 1) ? 1 : UART_BAUD;
    localparam integer HEARTBEAT_HZ_SAFE = (HEARTBEAT_HZ < 1) ? 1 : HEARTBEAT_HZ;
    localparam integer DEMO_GAP_MS_SAFE  = (DEMO_GAP_MS < 1) ? 1 : DEMO_GAP_MS;
    localparam integer CYCLES_PER_MS     = ((CLK_HZ / 1000) > 0) ? (CLK_HZ / 1000) : 1;
    localparam integer FRAME_GAP_CYCLES  = CYCLES_PER_MS * DEMO_GAP_MS_SAFE;
    localparam integer UART_CLKS_PER_BIT = ((CLK_HZ / UART_BAUD_SAFE) > 0) ? (CLK_HZ / UART_BAUD_SAFE) : 1;
    localparam integer HEARTBEAT_DIV     = ((CLK_HZ / (HEARTBEAT_HZ_SAFE * 2)) > 0) ? (CLK_HZ / (HEARTBEAT_HZ_SAFE * 2)) : 1;
    localparam integer DEMO_MSG_LEN      = 10;

    localparam [1:0] UART_ST_GAP   = 2'd0;
    localparam [1:0] UART_ST_ISSUE = 2'd1;
    localparam [1:0] UART_ST_WAIT  = 2'd2;

    reg [31:0] heartbeat_counter;

    reg [1:0]  uart_state;
    reg [31:0] gap_counter;
    reg [3:0]  msg_index;
    reg        tx_valid;
    reg [7:0]  tx_data;
    wire       tx_busy;
    wire       tx_done;

    function [7:0] demo_char;
        input [3:0] index;
        begin
            case (index)
                4'd0: demo_char = 8'h52; // R
                4'd1: demo_char = 8'h41; // A
                4'd2: demo_char = 8'h4C; // L
                4'd3: demo_char = 8'h50; // P
                4'd4: demo_char = 8'h48; // H
                4'd5: demo_char = 8'h47; // G
                4'd6: demo_char = 8'h50; // P
                4'd7: demo_char = 8'h55; // U
                4'd8: demo_char = 8'h0D; // \r
                4'd9: demo_char = 8'h0A; // \n
                default: demo_char = 8'h20;
            endcase
        end
    endfunction

    uart_tx #(
        .CLKS_PER_BIT(UART_CLKS_PER_BIT)
    ) u_uart_tx (
        .clk(clk_i),
        .rst_n(rst_ni),
        .data_valid(tx_valid),
        .data_in(tx_data),
        .tx(uart_tx_o),
        .busy(tx_busy),
        .done(tx_done)
    );

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            heartbeat_counter <= 32'd0;
            led_o             <= 4'b0001;
        end else begin
            if (heartbeat_counter == HEARTBEAT_DIV - 1) begin
                heartbeat_counter <= 32'd0;
                led_o             <= {led_o[2:0], led_o[3]};
            end else begin
                heartbeat_counter <= heartbeat_counter + 1'b1;
            end
        end
    end

    always @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            uart_state  <= UART_ST_GAP;
            gap_counter <= 32'd0;
            msg_index   <= 4'd0;
            tx_valid    <= 1'b0;
            tx_data     <= 8'h52;
        end else begin
            tx_valid <= 1'b0;

            case (uart_state)
                UART_ST_GAP: begin
                    if (gap_counter == FRAME_GAP_CYCLES - 1) begin
                        gap_counter <= 32'd0;
                        msg_index   <= 4'd0;
                        uart_state  <= UART_ST_ISSUE;
                    end else begin
                        gap_counter <= gap_counter + 1'b1;
                    end
                end

                UART_ST_ISSUE: begin
                    if (!tx_busy) begin
                        tx_data    <= demo_char(msg_index);
                        tx_valid   <= 1'b1;
                        uart_state <= UART_ST_WAIT;
                    end
                end

                UART_ST_WAIT: begin
                    if (tx_done) begin
                        if (msg_index == DEMO_MSG_LEN - 1) begin
                            uart_state <= UART_ST_GAP;
                        end else begin
                            msg_index  <= msg_index + 1'b1;
                            uart_state <= UART_ST_ISSUE;
                        end
                    end
                end

                default: begin
                    uart_state <= UART_ST_GAP;
                end
            endcase
        end
    end

endmodule
