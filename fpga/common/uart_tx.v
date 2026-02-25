`timescale 1ns / 1ps

module uart_tx #(
    parameter integer CLKS_PER_BIT = 434
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       data_valid,
    input  wire [7:0] data_in,
    output reg        tx,
    output reg        busy,
    output reg        done
);
    localparam integer CLKS_PER_BIT_SAFE = (CLKS_PER_BIT < 1) ? 1 : CLKS_PER_BIT;

    localparam [1:0] ST_IDLE  = 2'd0;
    localparam [1:0] ST_START = 2'd1;
    localparam [1:0] ST_DATA  = 2'd2;
    localparam [1:0] ST_STOP  = 2'd3;

    reg [1:0]  state;
    reg [31:0] clk_count;
    reg [2:0]  bit_index;
    reg [7:0]  shift_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_IDLE;
            clk_count  <= 32'd0;
            bit_index  <= 3'd0;
            shift_reg  <= 8'd0;
            tx         <= 1'b1;
            busy       <= 1'b0;
            done       <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    tx        <= 1'b1;
                    busy      <= 1'b0;
                    clk_count <= 32'd0;
                    bit_index <= 3'd0;
                    if (data_valid) begin
                        busy      <= 1'b1;
                        shift_reg <= data_in;
                        tx        <= 1'b0;
                        state     <= ST_START;
                    end
                end

                ST_START: begin
                    busy <= 1'b1;
                    tx   <= 1'b0;
                    if (clk_count == CLKS_PER_BIT_SAFE - 1) begin
                        clk_count <= 32'd0;
                        state     <= ST_DATA;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                ST_DATA: begin
                    busy <= 1'b1;
                    tx   <= shift_reg[bit_index];
                    if (clk_count == CLKS_PER_BIT_SAFE - 1) begin
                        clk_count <= 32'd0;
                        if (bit_index == 3'd7) begin
                            bit_index <= 3'd0;
                            state     <= ST_STOP;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                ST_STOP: begin
                    busy <= 1'b1;
                    tx   <= 1'b1;
                    if (clk_count == CLKS_PER_BIT_SAFE - 1) begin
                        clk_count <= 32'd0;
                        busy      <= 1'b0;
                        done      <= 1'b1;
                        state     <= ST_IDLE;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
