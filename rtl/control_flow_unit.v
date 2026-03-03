//============================================================================
// RalphGPU - Control Flow Unit
// Branch, Call, Return with hardware stack support
// Supports divergent execution with convergence tracking
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module control_flow_unit #(
    parameter STACK_DEPTH = 8,                              // Call stack depth per warp
    parameter NUM_WARPS   = `WARPS_PER_SM,                  // Number of warps
    parameter WARP_ID_W   = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1
)(
    input  wire        clk,
    input  wire        rst_n,

    // Current state
    input  wire [31:0] pc_current,
    input  wire [WARP_ID_W-1:0] warp_id,
    input  wire [31:0] active_mask,     // Currently active threads

    // Branch control
    input  wire        branch_valid,
    input  wire [5:0]  branch_type,     // Branch type/condition
    input  wire [31:0] branch_target,   // Target PC for branch
    input  wire [31:0] branch_cond,     // Condition mask (per-thread)
    input  wire        is_uniform,      // Uniform branch (all threads same)

    // Call/Return control
    input  wire        call_valid,
    input  wire [31:0] call_target,
    input  wire        ret_valid,

    // Divergence info
    input  wire [31:0] diverge_mask,    // Threads that want to diverge

    // Output
    output reg  [31:0] next_pc,
    output reg  [31:0] next_active_mask,
    output reg         pc_valid,
    output reg         stall,           // Stall for stack operations

    // Convergence point management
    output reg  [31:0] reconverge_pc,
    output reg         at_reconverge,

    // Stack status
    output wire        stack_overflow,
    output wire        stack_underflow
);

    //------------------------------------------------------------------------
    // Call stack per warp
    //------------------------------------------------------------------------
    reg [31:0] return_stack [0:NUM_WARPS-1][0:STACK_DEPTH-1];
    reg [2:0]  stack_ptr    [0:NUM_WARPS-1];

    // Divergence stack for SIMT reconvergence
    reg [31:0] div_stack_pc   [0:NUM_WARPS-1][0:STACK_DEPTH-1];
    reg [31:0] div_stack_mask [0:NUM_WARPS-1][0:STACK_DEPTH-1];
    reg [2:0]  div_stack_ptr  [0:NUM_WARPS-1];

    wire [2:0] curr_stack_ptr     = stack_ptr[warp_id];
    wire [2:0] curr_div_stack_ptr = div_stack_ptr[warp_id];

    assign stack_overflow  = (curr_stack_ptr == STACK_DEPTH - 1);
    assign stack_underflow = (curr_stack_ptr == 0) && ret_valid;

    //------------------------------------------------------------------------
    // Branch type definitions
    //------------------------------------------------------------------------
    localparam [5:0] BR_UNCONDITIONAL = 6'b000000;  // bra
    localparam [5:0] BR_IF_TRUE       = 6'b000001;  // @p bra
    localparam [5:0] BR_IF_FALSE      = 6'b000010;  // @!p bra
    localparam [5:0] BR_IF_OVERFLOW   = 6'b000101;  // bra.ovf
    localparam [5:0] BR_UNIFORM       = 6'b000011;  // bra.uni
    localparam [5:0] BR_INDIRECT      = 6'b000100;  // brx.idx

    //------------------------------------------------------------------------
    // Control flow FSM
    //------------------------------------------------------------------------
    localparam IDLE       = 2'b00;
    localparam DIVERGE    = 2'b01;
    localparam CONVERGE   = 2'b10;
    localparam CALL_RET   = 2'b11;

    reg [1:0] state;

    //------------------------------------------------------------------------
    // Divergence handling logic
    //------------------------------------------------------------------------
    wire threads_diverge = (diverge_mask != 32'b0) &&
                          (diverge_mask != active_mask) && !is_uniform;

    wire [31:0] taken_mask     = diverge_mask & active_mask;
    wire [31:0] not_taken_mask = ~diverge_mask & active_mask;

    // Check if we're at a reconvergence point
    wire [2:0] div_stack_top_idx = (curr_div_stack_ptr > 0) ? (curr_div_stack_ptr - 3'd1) : 3'd0;
    wire [2:0] stack_top_idx = (curr_stack_ptr > 0) ? (curr_stack_ptr - 3'd1) : 3'd0;
    wire at_div_top = (curr_div_stack_ptr > 0) &&
                     (div_stack_pc[warp_id][div_stack_top_idx] == pc_current);

    //------------------------------------------------------------------------
    // Main control logic
    //------------------------------------------------------------------------
    // Reset indices
    integer rst_i, rst_j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all stacks
            for (rst_i = 0; rst_i < NUM_WARPS; rst_i = rst_i + 1) begin
                stack_ptr[rst_i] <= 3'b0;
                div_stack_ptr[rst_i] <= 3'b0;
                for (rst_j = 0; rst_j < STACK_DEPTH; rst_j = rst_j + 1) begin
                    return_stack[rst_i][rst_j] <= 32'b0;
                    div_stack_pc[rst_i][rst_j] <= 32'b0;
                    div_stack_mask[rst_i][rst_j] <= 32'b0;
                end
            end
            state <= IDLE;
            next_pc <= 32'b0;
            next_active_mask <= 32'hFFFFFFFF;
            pc_valid <= 1'b0;
            stall <= 1'b0;
            reconverge_pc <= 32'b0;
            at_reconverge <= 1'b0;
        end else begin
            pc_valid <= 1'b0;
            stall <= 1'b0;
            at_reconverge <= 1'b0;

            case (state)
                IDLE: begin
                    //--------------------------------------------------
                    // Check for reconvergence first
                    //--------------------------------------------------
                    if (at_div_top) begin
                        // Pop divergence stack and restore mask
                        at_reconverge <= 1'b1;
                        next_active_mask <= div_stack_mask[warp_id][div_stack_top_idx];
                        div_stack_ptr[warp_id] <= curr_div_stack_ptr - 1;
                        next_pc <= pc_current;
                        pc_valid <= 1'b1;
                    end

                    //--------------------------------------------------
                    // Handle CALL
                    //--------------------------------------------------
                    else if (call_valid && !stack_overflow) begin
                        // Push return address
                        return_stack[warp_id][curr_stack_ptr] <= pc_current + 4;
                        stack_ptr[warp_id] <= curr_stack_ptr + 1;
                        next_pc <= call_target;
                        next_active_mask <= active_mask;
                        pc_valid <= 1'b1;
                        state <= CALL_RET;
                    end

                    //--------------------------------------------------
                    // Handle RET
                    //--------------------------------------------------
                    else if (ret_valid && !stack_underflow) begin
                        // Pop return address
                        next_pc <= return_stack[warp_id][stack_top_idx];
                        stack_ptr[warp_id] <= curr_stack_ptr - 1;
                        next_active_mask <= active_mask;
                        pc_valid <= 1'b1;
                        state <= CALL_RET;
                    end

                    //--------------------------------------------------
                    // Handle BRANCH
                    //--------------------------------------------------
                    else if (branch_valid) begin
                        case (branch_type)
                            BR_UNCONDITIONAL, BR_UNIFORM: begin
                                // All threads take the branch
                                next_pc <= branch_target;
                                next_active_mask <= active_mask;
                                pc_valid <= 1'b1;
                            end

                            BR_IF_TRUE, BR_IF_OVERFLOW: begin
                                if (threads_diverge) begin
                                    // Push reconvergence point
                                    div_stack_pc[warp_id][curr_div_stack_ptr] <= pc_current + 4;
                                    div_stack_mask[warp_id][curr_div_stack_ptr] <= not_taken_mask;
                                    div_stack_ptr[warp_id] <= curr_div_stack_ptr + 1;
                                    // Execute taken path first
                                    next_pc <= branch_target;
                                    next_active_mask <= taken_mask;
                                    reconverge_pc <= pc_current + 4;
                                    pc_valid <= 1'b1;
                                    state <= DIVERGE;
                                end else if (taken_mask != 32'b0) begin
                                    // All active threads take branch
                                    next_pc <= branch_target;
                                    next_active_mask <= taken_mask;
                                    pc_valid <= 1'b1;
                                end else begin
                                    // No threads take branch - continue
                                    next_pc <= pc_current + 4;
                                    next_active_mask <= active_mask;
                                    pc_valid <= 1'b1;
                                end
                            end

                            BR_IF_FALSE: begin
                                if (threads_diverge) begin
                                    // Push reconvergence point
                                    div_stack_pc[warp_id][curr_div_stack_ptr] <= pc_current + 4;
                                    div_stack_mask[warp_id][curr_div_stack_ptr] <= taken_mask;
                                    div_stack_ptr[warp_id] <= curr_div_stack_ptr + 1;
                                    // Execute not-taken path first
                                    next_pc <= branch_target;
                                    next_active_mask <= not_taken_mask;
                                    reconverge_pc <= pc_current + 4;
                                    pc_valid <= 1'b1;
                                    state <= DIVERGE;
                                end else if (not_taken_mask != 32'b0) begin
                                    // All active threads take branch
                                    next_pc <= branch_target;
                                    next_active_mask <= not_taken_mask;
                                    pc_valid <= 1'b1;
                                end else begin
                                    // No threads take branch - continue
                                    next_pc <= pc_current + 4;
                                    next_active_mask <= active_mask;
                                    pc_valid <= 1'b1;
                                end
                            end

                            BR_INDIRECT: begin
                                // Indirect branch - target from register
                                next_pc <= branch_target;
                                next_active_mask <= active_mask;
                                pc_valid <= 1'b1;
                            end

                            default: begin
                                // Unknown branch type - continue sequentially
                                next_pc <= pc_current + 4;
                                next_active_mask <= active_mask;
                                pc_valid <= 1'b1;
                            end
                        endcase
                    end
                end

                DIVERGE: begin
                    // Wait for divergent path to complete
                    // Will reconverge when reaching div_stack_pc
                    state <= IDLE;
                end

                CONVERGE: begin
                    // Restore previous active mask
                    state <= IDLE;
                end

                CALL_RET: begin
                    // Single cycle for call/ret
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

//============================================================================
// Simplified Branch Unit (combinatorial for pipeline integration)
//============================================================================
module branch_unit (
    input  wire [31:0] pc_current,
    input  wire [31:0] offset,         // Signed offset (21-bit extended)
    input  wire [31:0] target_reg,     // For indirect branches
    input  wire        taken,          // Branch condition result
    input  wire        is_indirect,
    input  wire        is_call,

    output wire [31:0] next_pc,
    output wire [31:0] return_addr
);

    wire [31:0] branch_target = is_indirect ? target_reg :
                               (pc_current + {{11{offset[20]}}, offset});

    assign next_pc = taken ? branch_target : (pc_current + 4);
    assign return_addr = pc_current + 4;

endmodule

//============================================================================
// Predicate Register File (for conditional execution)
//============================================================================
module predicate_rf #(
    parameter NUM_PREDICATES = 8,
    parameter NUM_THREADS    = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    // Write port
    input  wire        wr_en,
    input  wire [2:0]  wr_addr,
    input  wire [NUM_THREADS-1:0] wr_data,

    // Read ports
    input  wire [2:0]  rd_addr_a,
    input  wire [2:0]  rd_addr_b,
    output wire [NUM_THREADS-1:0] rd_data_a,
    output wire [NUM_THREADS-1:0] rd_data_b
);

    // Predicate storage - one bit per thread per predicate
    reg [NUM_THREADS-1:0] pred_regs [0:NUM_PREDICATES-1];

    // Special predicates
    // p0 is always true (all 1s)
    // p7 is often used as uniform predicate

    integer pred_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pred_i = 0; pred_i < NUM_PREDICATES; pred_i = pred_i + 1) begin
                pred_regs[pred_i] <= {NUM_THREADS{1'b1}};  // Initialize to all true
            end
        end else if (wr_en) begin
            pred_regs[wr_addr] <= wr_data;
        end
    end

    assign rd_data_a = pred_regs[rd_addr_a];
    assign rd_data_b = pred_regs[rd_addr_b];

endmodule

//============================================================================
// Convergence Barrier - Ensures SIMT threads reconverge
//============================================================================
module convergence_barrier #(
    parameter NUM_THREADS = 32
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire [31:0] active_mask,
    input  wire [31:0] arrived_mask,  // Threads that have arrived at barrier
    input  wire        barrier_id,    // Barrier identifier

    output wire        all_arrived,   // All active threads arrived
    output reg  [31:0] waiting_mask   // Threads waiting at barrier
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            waiting_mask <= 32'b0;
        end else begin
            waiting_mask <= arrived_mask & active_mask;
        end
    end

    assign all_arrived = ((waiting_mask & active_mask) == active_mask);

endmodule
