//============================================================================
// RalphGPU - Enhanced Reconvergence Stack
// Per-warp divergence tracking with proper SIMT reconvergence
// Based on NVIDIA IPDOM (Immediate Post-Dominator) reconvergence
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module reconvergence_stack #(
    parameter NUM_WARPS     = `WARPS_PER_SM,
    parameter STACK_DEPTH   = 16,           // Stack entries per warp
    parameter NUM_THREADS   = `THREADS_PER_WARP,
    parameter ADDR_WIDTH    = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Warp Selection
    //------------------------------------------------------------------------
    input  wire [$clog2(NUM_WARPS)-1:0] warp_id,

    //------------------------------------------------------------------------
    // Branch/Divergence Interface
    //------------------------------------------------------------------------
    input  wire                     branch_valid,
    input  wire [ADDR_WIDTH-1:0]    branch_target,
    input  wire [ADDR_WIDTH-1:0]    fallthrough_pc,     // PC + 4
    input  wire [NUM_THREADS-1:0]   branch_taken_mask,  // Threads taking branch
    input  wire [NUM_THREADS-1:0]   active_mask,        // Currently active threads
    input  wire                     is_uniform,         // All threads same direction

    //------------------------------------------------------------------------
    // Current PC Interface
    //------------------------------------------------------------------------
    input  wire [ADDR_WIDTH-1:0]    current_pc,

    //------------------------------------------------------------------------
    // Reconvergence Output
    //------------------------------------------------------------------------
    output wire [ADDR_WIDTH-1:0]    next_pc,
    output wire [NUM_THREADS-1:0]   next_active_mask,
    output wire                     pc_valid,
    output wire                     at_reconvergence,   // At reconvergence point
    output wire                     diverged,           // Threads currently diverged

    //------------------------------------------------------------------------
    // Stack Status
    //------------------------------------------------------------------------
    output wire                     stack_overflow,
    output wire                     stack_empty,
    output wire [$clog2(STACK_DEPTH):0] stack_depth
);

    //------------------------------------------------------------------------
    // Stack Entry Format
    //------------------------------------------------------------------------
    localparam ENTRY_PC_LSB     = 0;
    localparam ENTRY_PC_MSB     = ADDR_WIDTH - 1;
    localparam ENTRY_MASK_LSB   = ENTRY_PC_MSB + 1;
    localparam ENTRY_MASK_MSB   = ENTRY_MASK_LSB + NUM_THREADS - 1;
    localparam ENTRY_RPC_LSB    = ENTRY_MASK_MSB + 1;
    localparam ENTRY_RPC_MSB    = ENTRY_RPC_LSB + ADDR_WIDTH - 1;
    localparam ENTRY_WIDTH      = ENTRY_RPC_MSB + 1;

    //------------------------------------------------------------------------
    // Per-Warp Stack Storage
    //------------------------------------------------------------------------
    reg [ENTRY_WIDTH-1:0]   stack_mem [0:NUM_WARPS-1][0:STACK_DEPTH-1];
    reg [$clog2(STACK_DEPTH):0] stack_ptr [0:NUM_WARPS-1];

    // Current warp's stack pointer
    reg [$clog2(STACK_DEPTH):0] curr_sp;
    always @(*) curr_sp = stack_ptr[warp_id];

    //------------------------------------------------------------------------
    // Stack Entry Access
    //------------------------------------------------------------------------
    reg [ENTRY_WIDTH-1:0] top_entry;
    always @(*) begin
        if (curr_sp > 0)
            top_entry = stack_mem[warp_id][curr_sp - 1];
        else
            top_entry = {ENTRY_WIDTH{1'b0}};
    end
    wire [ADDR_WIDTH-1:0]   top_pc      = top_entry[ENTRY_PC_MSB:ENTRY_PC_LSB];
    wire [NUM_THREADS-1:0]  top_mask    = top_entry[ENTRY_MASK_MSB:ENTRY_MASK_LSB];
    wire [ADDR_WIDTH-1:0]   top_rpc     = top_entry[ENTRY_RPC_MSB:ENTRY_RPC_LSB];

    //------------------------------------------------------------------------
    // Divergence Detection
    //------------------------------------------------------------------------
    wire [NUM_THREADS-1:0] taken_threads     = branch_taken_mask & active_mask;
    wire [NUM_THREADS-1:0] not_taken_threads = ~branch_taken_mask & active_mask;

    wire all_taken     = (taken_threads == active_mask);
    wire none_taken    = (taken_threads == {NUM_THREADS{1'b0}});
    wire threads_diverge = !all_taken && !none_taken && !is_uniform;

    //------------------------------------------------------------------------
    // Reconvergence Point Detection
    //------------------------------------------------------------------------
    // Check if current PC matches top of stack reconvergence point
    wire at_rpc = (curr_sp > 0) && (current_pc == top_rpc);

    // Check if current PC matches top entry PC (execution path)
    wire at_top_pc = (curr_sp > 0) && (current_pc == top_pc);

    //------------------------------------------------------------------------
    // FSM States
    //------------------------------------------------------------------------
    localparam ST_NORMAL     = 3'd0;
    localparam ST_PUSH_TAKEN = 3'd1;
    localparam ST_PUSH_NOT   = 3'd2;
    localparam ST_DIVERGED   = 3'd3;
    localparam ST_RECONVERGE = 3'd4;

    reg [2:0] state [0:NUM_WARPS-1];
    wire [2:0] curr_state = state[warp_id];

    //------------------------------------------------------------------------
    // Output Registers
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0]    next_pc_r;
    reg [NUM_THREADS-1:0]   next_mask_r;
    reg                     pc_valid_r;
    reg                     at_reconv_r;
    reg                     diverged_r;

    //------------------------------------------------------------------------
    // Main Logic
    //------------------------------------------------------------------------
    integer rst_w, rst_s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all warps
            for (rst_w = 0; rst_w < NUM_WARPS; rst_w = rst_w + 1) begin
                stack_ptr[rst_w] <= 0;
                state[rst_w] <= ST_NORMAL;
                for (rst_s = 0; rst_s < STACK_DEPTH; rst_s = rst_s + 1) begin
                    stack_mem[rst_w][rst_s] <= {ENTRY_WIDTH{1'b0}};
                end
            end
            next_pc_r <= 0;
            next_mask_r <= {NUM_THREADS{1'b1}};
            pc_valid_r <= 1'b0;
            at_reconv_r <= 1'b0;
            diverged_r <= 1'b0;
        end else begin
            pc_valid_r <= 1'b0;
            at_reconv_r <= 1'b0;

            case (curr_state)
                ST_NORMAL: begin
                    // Check for reconvergence first
                    if (at_rpc) begin
                        // Pop and restore mask
                        next_mask_r <= top_mask | active_mask;  // Merge masks
                        next_pc_r <= current_pc;
                        pc_valid_r <= 1'b1;
                        at_reconv_r <= 1'b1;
                        stack_ptr[warp_id] <= curr_sp - 1;
                        diverged_r <= (curr_sp > 1);
                        state[warp_id] <= ST_NORMAL;
                    end
                    // Handle new branch
                    else if (branch_valid) begin
                        if (threads_diverge) begin
                            // Push not-taken path with reconvergence point
                            if (curr_sp < STACK_DEPTH) begin
                                stack_mem[warp_id][curr_sp] <= {
                                    fallthrough_pc,     // RPC: immediate post-dominator
                                    not_taken_threads,  // Mask: not-taken threads
                                    fallthrough_pc      // PC: continue at fallthrough
                                };
                                stack_ptr[warp_id] <= curr_sp + 1;

                                // Execute taken path with taken threads
                                next_pc_r <= branch_target;
                                next_mask_r <= taken_threads;
                                pc_valid_r <= 1'b1;
                                diverged_r <= 1'b1;
                                state[warp_id] <= ST_DIVERGED;
                            end
                        end else if (all_taken) begin
                            // All threads take branch
                            next_pc_r <= branch_target;
                            next_mask_r <= active_mask;
                            pc_valid_r <= 1'b1;
                        end else begin
                            // No threads take branch (fall through)
                            next_pc_r <= fallthrough_pc;
                            next_mask_r <= active_mask;
                            pc_valid_r <= 1'b1;
                        end
                    end
                end

                ST_DIVERGED: begin
                    // Continue executing diverged path
                    // Check for reconvergence
                    if (at_rpc) begin
                        // Pop and switch to other path or reconverge
                        if (top_mask != {NUM_THREADS{1'b0}}) begin
                            // Execute the saved path
                            next_pc_r <= top_pc;
                            next_mask_r <= top_mask;
                            pc_valid_r <= 1'b1;
                            // Don't pop yet - wait until we reach RPC again
                        end else begin
                            // Reconverge
                            stack_ptr[warp_id] <= curr_sp - 1;
                            next_mask_r <= active_mask;
                            at_reconv_r <= 1'b1;
                            diverged_r <= (curr_sp > 1);
                            state[warp_id] <= ST_NORMAL;
                        end
                    end
                    // Handle nested divergence
                    else if (branch_valid && threads_diverge) begin
                        if (curr_sp < STACK_DEPTH) begin
                            // Push for nested divergence
                            stack_mem[warp_id][curr_sp] <= {
                                fallthrough_pc,
                                not_taken_threads,
                                fallthrough_pc
                            };
                            stack_ptr[warp_id] <= curr_sp + 1;

                            next_pc_r <= branch_target;
                            next_mask_r <= taken_threads;
                            pc_valid_r <= 1'b1;
                        end
                    end else if (branch_valid) begin
                        // Uniform branch in diverged state
                        if (all_taken) begin
                            next_pc_r <= branch_target;
                            next_mask_r <= active_mask;
                            pc_valid_r <= 1'b1;
                        end else begin
                            next_pc_r <= fallthrough_pc;
                            next_mask_r <= active_mask;
                            pc_valid_r <= 1'b1;
                        end
                    end
                end

                ST_RECONVERGE: begin
                    // Complete reconvergence
                    state[warp_id] <= ST_NORMAL;
                    diverged_r <= (curr_sp > 0);
                end

                default: state[warp_id] <= ST_NORMAL;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign next_pc          = next_pc_r;
    assign next_active_mask = next_mask_r;
    assign pc_valid         = pc_valid_r;
    assign at_reconvergence = at_reconv_r;
    assign diverged         = diverged_r;
    assign stack_overflow   = (curr_sp >= STACK_DEPTH);
    assign stack_empty      = (curr_sp == 0);
    assign stack_depth      = curr_sp;

endmodule


//============================================================================
// SIMT Stack Controller
// Integrates reconvergence stack with warp scheduler
//============================================================================
module simt_stack_controller #(
    parameter NUM_WARPS     = `WARPS_PER_SM,
    parameter STACK_DEPTH   = 16,
    parameter NUM_THREADS   = `THREADS_PER_WARP,
    parameter ADDR_WIDTH    = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Warp Interface (per warp)
    //------------------------------------------------------------------------
    input  wire [NUM_WARPS-1:0]             warp_valid,
    input  wire [ADDR_WIDTH*NUM_WARPS-1:0]  warp_pc,
    input  wire [NUM_THREADS*NUM_WARPS-1:0] warp_active_mask,

    //------------------------------------------------------------------------
    // Branch Interface
    //------------------------------------------------------------------------
    input  wire                     branch_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] branch_warp_id,
    input  wire [ADDR_WIDTH-1:0]    branch_target,
    input  wire [ADDR_WIDTH-1:0]    branch_fallthrough,
    input  wire [NUM_THREADS-1:0]   branch_taken_mask,
    input  wire                     branch_uniform,

    //------------------------------------------------------------------------
    // Scheduler Interface
    //------------------------------------------------------------------------
    output wire [NUM_WARPS-1:0]     warp_diverged,      // Per-warp divergence status
    output wire [NUM_WARPS-1:0]     warp_at_barrier,    // At reconvergence point

    //------------------------------------------------------------------------
    // PC Update Interface
    //------------------------------------------------------------------------
    output wire                     pc_update_valid,
    output wire [$clog2(NUM_WARPS)-1:0] pc_update_warp,
    output wire [ADDR_WIDTH-1:0]    pc_update_value,
    output wire [NUM_THREADS-1:0]   mask_update_value
);

    //------------------------------------------------------------------------
    // Per-Warp Reconvergence Stacks
    //------------------------------------------------------------------------
    wire [ADDR_WIDTH-1:0]   stack_next_pc       [0:NUM_WARPS-1];
    wire [NUM_THREADS-1:0]  stack_next_mask     [0:NUM_WARPS-1];
    wire [NUM_WARPS-1:0]    stack_pc_valid;
    wire [NUM_WARPS-1:0]    stack_at_reconv;
    wire [NUM_WARPS-1:0]    stack_diverged;
    wire [NUM_WARPS-1:0]    stack_overflow;
    wire [NUM_WARPS-1:0]    stack_empty;

    genvar w;
    generate
        for (w = 0; w < NUM_WARPS; w = w + 1) begin : gen_stacks
            reconvergence_stack #(
                .NUM_WARPS      (1),
                .STACK_DEPTH    (STACK_DEPTH),
                .NUM_THREADS    (NUM_THREADS),
                .ADDR_WIDTH     (ADDR_WIDTH)
            ) u_stack (
                .clk                (clk),
                .rst_n              (rst_n),
                .warp_id            (1'b0),
                .branch_valid       (branch_valid && (branch_warp_id == w)),
                .branch_target      (branch_target),
                .fallthrough_pc     (branch_fallthrough),
                .branch_taken_mask  (branch_taken_mask),
                .active_mask        (warp_active_mask[w*NUM_THREADS +: NUM_THREADS]),
                .is_uniform         (branch_uniform),
                .current_pc         (warp_pc[w*ADDR_WIDTH +: ADDR_WIDTH]),
                .next_pc            (stack_next_pc[w]),
                .next_active_mask   (stack_next_mask[w]),
                .pc_valid           (stack_pc_valid[w]),
                .at_reconvergence   (stack_at_reconv[w]),
                .diverged           (stack_diverged[w]),
                .stack_overflow     (stack_overflow[w]),
                .stack_empty        (stack_empty[w]),
                .stack_depth        ()
            );
        end
    endgenerate

    //------------------------------------------------------------------------
    // Output Aggregation
    //------------------------------------------------------------------------
    assign warp_diverged   = stack_diverged;
    assign warp_at_barrier = stack_at_reconv;

    // PC update arbitration (priority to lowest warp ID)
    reg                     pc_update_valid_r;
    reg [$clog2(NUM_WARPS)-1:0] pc_update_warp_r;
    reg [ADDR_WIDTH-1:0]    pc_update_value_r;
    reg [NUM_THREADS-1:0]   mask_update_value_r;

    integer pc_i;
    always @(*) begin
        pc_update_valid_r = 1'b0;
        pc_update_warp_r = 0;
        pc_update_value_r = 0;
        mask_update_value_r = {NUM_THREADS{1'b1}};

        for (pc_i = 0; pc_i < NUM_WARPS; pc_i = pc_i + 1) begin
            if (stack_pc_valid[pc_i] && !pc_update_valid_r) begin
                pc_update_valid_r = 1'b1;
                pc_update_warp_r = pc_i[$clog2(NUM_WARPS)-1:0];
                pc_update_value_r = stack_next_pc[pc_i];
                mask_update_value_r = stack_next_mask[pc_i];
            end
        end
    end

    assign pc_update_valid  = pc_update_valid_r;
    assign pc_update_warp   = pc_update_warp_r;
    assign pc_update_value  = pc_update_value_r;
    assign mask_update_value = mask_update_value_r;

endmodule
