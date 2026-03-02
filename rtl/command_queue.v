//============================================================================
// RalphGPU - Command Queue Ring Buffer
// Issue #271: Command Processor Phase 2 queue module
//============================================================================

`timescale 1ns / 1ps

module command_queue #(
    parameter DEPTH = 8,
    parameter PTR_WIDTH = $clog2(DEPTH)
)(
    input  wire        clk,
    input  wire        rst_n,

    // Push side
    input  wire        push_valid,
    output wire        push_ready,
    input  wire [31:0] push_kernel_pc,
    input  wire [31:0] push_grid_dim_x,
    input  wire [31:0] push_grid_dim_y,
    input  wire [31:0] push_grid_dim_z,
    input  wire [31:0] push_block_dim_x,
    input  wire [31:0] push_block_dim_y,
    input  wire [31:0] push_block_dim_z,
    input  wire [31:0] push_shared_mem_size,
    input  wire [31:0] push_kernel_params_base,
    input  wire [31:0] push_fence_id,
    input  wire [31:0] push_flags,

    // Pop side
    output wire        pop_valid,
    input  wire        pop_ready,
    output wire [31:0] pop_kernel_pc,
    output wire [31:0] pop_grid_dim_x,
    output wire [31:0] pop_grid_dim_y,
    output wire [31:0] pop_grid_dim_z,
    output wire [31:0] pop_block_dim_x,
    output wire [31:0] pop_block_dim_y,
    output wire [31:0] pop_block_dim_z,
    output wire [31:0] pop_shared_mem_size,
    output wire [31:0] pop_kernel_params_base,
    output wire [31:0] pop_fence_id,
    output wire [31:0] pop_flags,

    // Queue status
    output reg  [PTR_WIDTH-1:0] head,
    output reg  [PTR_WIDTH-1:0] tail,
    output reg  [PTR_WIDTH:0]   count
);

    localparam [PTR_WIDTH-1:0] PTR_LAST = PTR_WIDTH'(DEPTH - 1);

    reg [31:0] queue_kernel_pc          [0:DEPTH-1];
    reg [31:0] queue_grid_dim_x         [0:DEPTH-1];
    reg [31:0] queue_grid_dim_y         [0:DEPTH-1];
    reg [31:0] queue_grid_dim_z         [0:DEPTH-1];
    reg [31:0] queue_block_dim_x        [0:DEPTH-1];
    reg [31:0] queue_block_dim_y        [0:DEPTH-1];
    reg [31:0] queue_block_dim_z        [0:DEPTH-1];
    reg [31:0] queue_shared_mem_size    [0:DEPTH-1];
    reg [31:0] queue_kernel_params_base [0:DEPTH-1];
    reg [31:0] queue_fence_id           [0:DEPTH-1];
    reg [31:0] queue_flags              [0:DEPTH-1];

    wire push_fire = push_valid && push_ready;
    wire pop_fire  = pop_valid && pop_ready;

    function [PTR_WIDTH-1:0] ptr_inc;
        input [PTR_WIDTH-1:0] ptr;
        begin
            if (ptr == PTR_LAST)
                ptr_inc = {PTR_WIDTH{1'b0}};
            else
                ptr_inc = ptr + 1'b1;
        end
    endfunction

    assign push_ready = (count < DEPTH);
    assign pop_valid  = (count != {PTR_WIDTH+1{1'b0}});

    assign pop_kernel_pc          = pop_valid ? queue_kernel_pc[head]          : 32'b0;
    assign pop_grid_dim_x         = pop_valid ? queue_grid_dim_x[head]         : 32'b0;
    assign pop_grid_dim_y         = pop_valid ? queue_grid_dim_y[head]         : 32'b0;
    assign pop_grid_dim_z         = pop_valid ? queue_grid_dim_z[head]         : 32'b0;
    assign pop_block_dim_x        = pop_valid ? queue_block_dim_x[head]        : 32'b0;
    assign pop_block_dim_y        = pop_valid ? queue_block_dim_y[head]        : 32'b0;
    assign pop_block_dim_z        = pop_valid ? queue_block_dim_z[head]        : 32'b0;
    assign pop_shared_mem_size    = pop_valid ? queue_shared_mem_size[head]    : 32'b0;
    assign pop_kernel_params_base = pop_valid ? queue_kernel_params_base[head] : 32'b0;
    assign pop_fence_id           = pop_valid ? queue_fence_id[head]           : 32'b0;
    assign pop_flags              = pop_valid ? queue_flags[head]              : 32'b0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head  <= {PTR_WIDTH{1'b0}};
            tail  <= {PTR_WIDTH{1'b0}};
            count <= {PTR_WIDTH+1{1'b0}};
        end else begin
            if (push_fire) begin
                queue_kernel_pc[tail]           <= push_kernel_pc;
                queue_grid_dim_x[tail]          <= push_grid_dim_x;
                queue_grid_dim_y[tail]          <= push_grid_dim_y;
                queue_grid_dim_z[tail]          <= push_grid_dim_z;
                queue_block_dim_x[tail]         <= push_block_dim_x;
                queue_block_dim_y[tail]         <= push_block_dim_y;
                queue_block_dim_z[tail]         <= push_block_dim_z;
                queue_shared_mem_size[tail]     <= push_shared_mem_size;
                queue_kernel_params_base[tail]  <= push_kernel_params_base;
                queue_fence_id[tail]            <= push_fence_id;
                queue_flags[tail]               <= push_flags;
                tail                            <= ptr_inc(tail);
            end

            if (pop_fire)
                head <= ptr_inc(head);

            case ({push_fire, pop_fire})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule
