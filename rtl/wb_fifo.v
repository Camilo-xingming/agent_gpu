//============================================================================
// RalphGPU - Write-Back FIFO
//
// Generic synchronous FIFO used for write-back queues (WBQs) and other
// internal buffering (tensor issue queue, atomic request queue, etc.).
//
// RALPH-6: Extracted from streaming_multiprocessor_v2.v (was inline module).
//============================================================================

`timescale 1ns / 1ps

module wb_fifo #(
    parameter WIDTH = 32,
    parameter DEPTH = 2
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             push,
    input  wire [WIDTH-1:0] push_data,
    input  wire             pop,
    output wire [WIDTH-1:0] pop_data,
    output wire             full,
    output wire             empty,
    output wire             dropped
);
    localparam PTR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam COUNT_W = $clog2(DEPTH + 1);
    /* verilator lint_off WIDTHTRUNC */
    /* verilator lint_off WIDTHEXPAND */
    localparam [COUNT_W-1:0] DEPTH_VAL = DEPTH;
    localparam [PTR_W-1:0] PTR_LAST = DEPTH - 1;
    /* verilator lint_on WIDTHEXPAND */
    /* verilator lint_on WIDTHTRUNC */

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_W-1:0] head;
    reg [PTR_W-1:0] tail;
    reg [COUNT_W-1:0] count;

    wire push_fire = push && (!full || pop);
    wire pop_fire = pop && !empty;

    assign full = (count == DEPTH_VAL);
    assign empty = (count == 0);
    assign pop_data = empty ? {WIDTH{1'b0}} : mem[head];

    function [PTR_W-1:0] ptr_inc;
        input [PTR_W-1:0] ptr;
        begin
            if (ptr == PTR_LAST) begin
                ptr_inc = {PTR_W{1'b0}};
            end else begin
                ptr_inc = ptr + {{(PTR_W-1){1'b0}}, 1'b1};
            end
        end
    endfunction

    integer mi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= {PTR_W{1'b0}};
            tail <= {PTR_W{1'b0}};
            count <= {COUNT_W{1'b0}};
            for (mi = 0; mi < DEPTH; mi = mi + 1) begin
                mem[mi] <= {WIDTH{1'b0}};
            end
        end else begin
            if (push_fire) begin
                mem[tail] <= push_data;
                tail <= ptr_inc(tail);
            end

            if (pop_fire) begin
                head <= ptr_inc(head);
            end

            case ({push_fire, pop_fire})
                2'b10: count <= count + {{(COUNT_W-1){1'b0}}, 1'b1};
                2'b01: count <= count - {{(COUNT_W-1){1'b0}}, 1'b1};
                default: count <= count;
            endcase
        end
    end

    // Detect silent drop: push attempted while full with no pop
    assign dropped = push && full && !pop;

    `ifdef SIMULATION
    always @(posedge clk) begin
        if (rst_n && dropped) begin
            $display("[WBQ] FATAL: push while full without pop at time %0t", $time);
            $error("[WBQ] Data silently dropped!");
            $finish;
        end
    end
    `endif

endmodule
