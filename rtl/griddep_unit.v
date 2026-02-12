//============================================================================
// RalphGPU - Grid Dependency Control Unit
// Implements griddepcontrol instructions for grid-level synchronization
// PTX Instructions: griddepcontrol.wait, griddepcontrol.launch_dependent,
//                   griddepcontrol.signal, griddepcontrol.get_token
//
// This unit manages dependencies between grids for programmatic dependent launch.
// Reference: NVIDIA PTX ISA 8.5+, Hopper/Blackwell Architecture
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module griddep_unit #(
    parameter DATA_WIDTH = 32,
    parameter MAX_GRIDS = 8,           // Max concurrent grids
    parameter TOKEN_WIDTH = 16         // Grid token width
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   opcode,         // OP_GRIDDEPCTRL
    input  wire [5:0]                   func,           // GRIDDEP_* function
    input  wire [DATA_WIDTH-1:0]        src_a,          // Dependency token / count
    input  wire [DATA_WIDTH-1:0]        src_b,          // Additional operand

    //------------------------------------------------------------------------
    // Grid Configuration (from scheduler)
    //------------------------------------------------------------------------
    input  wire [3:0]                   grid_id,        // Current grid ID
    input  wire [TOKEN_WIDTH-1:0]       grid_token,     // Current grid's dependency token

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          result_valid,
    output reg  [DATA_WIDTH-1:0]        result,
    output reg                          pred_result,    // Predicate for dependency check
    output wire                         stall_grid,     // Stall current grid

    //------------------------------------------------------------------------
    // Grid Control Interface (to scheduler)
    //------------------------------------------------------------------------
    output reg                          signal_complete,     // Signal grid completion
    output reg  [3:0]                   signal_grid_id,      // Which grid is signaling
    output reg  [TOKEN_WIDTH-1:0]       signal_token,        // Dependency token for signal

    output reg                          launch_dep_req,      // Request to launch dependent grid
    output reg  [TOKEN_WIDTH-1:0]       launch_dep_token,    // Token of grid to launch

    //------------------------------------------------------------------------
    // Dependency Status (from scheduler)
    //------------------------------------------------------------------------
    input  wire [MAX_GRIDS-1:0]         dep_satisfied,  // Which dependencies are satisfied
    input  wire [MAX_GRIDS-1:0]         grid_active,    // Which grids are active
    input  wire [TOKEN_WIDTH-1:0]       current_tokens [0:MAX_GRIDS-1]  // Current grid tokens
);

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE      = 2'd0;
    localparam ST_EXECUTE   = 2'd1;
    localparam ST_WAIT_DEP  = 2'd2;
    localparam ST_COMPLETE  = 2'd3;

    reg [1:0] state;

    //------------------------------------------------------------------------
    // Internal registers
    //------------------------------------------------------------------------
    reg [5:0]              saved_func;
    reg [DATA_WIDTH-1:0]   saved_src_a;
    reg [DATA_WIDTH-1:0]   saved_src_b;
    reg [3:0]              wait_grid_id;
    reg [TOKEN_WIDTH-1:0]  wait_token;
    reg                    waiting;

    //------------------------------------------------------------------------
    // Token management
    //------------------------------------------------------------------------
    reg [TOKEN_WIDTH-1:0]  next_token;          // Next available token
    reg [MAX_GRIDS-1:0]    grid_completed;      // Completion status per grid

    //------------------------------------------------------------------------
    // Stall signal for grid that's waiting on dependency
    //------------------------------------------------------------------------
    assign stall_grid = waiting && (state == ST_WAIT_DEP);

    //------------------------------------------------------------------------
    // Dependency satisfaction check
    //------------------------------------------------------------------------
    wire dep_check_result;
    localparam GRID_IDX_W = $clog2(MAX_GRIDS);
    reg [GRID_IDX_W-1:0] check_grid_idx;

    // Token to check - use wait_token in WAIT_DEP state, saved_src_a in EXECUTE
    wire [TOKEN_WIDTH-1:0] token_to_check = (state == ST_WAIT_DEP) ? wait_token : saved_src_a[TOKEN_WIDTH-1:0];

    // Find grid by token
    localparam INVALID_GRID = {GRID_IDX_W{1'b1}};  // All 1s = invalid
    always @(*) begin
        check_grid_idx = INVALID_GRID;  // Invalid
        for (integer i = 0; i < MAX_GRIDS; i = i + 1) begin
            if (current_tokens[i] == token_to_check && grid_active[i]) begin
                check_grid_idx = i[GRID_IDX_W-1:0];
            end
        end
    end

    assign dep_check_result = (check_grid_idx != INVALID_GRID) ?
                              dep_satisfied[check_grid_idx] || grid_completed[check_grid_idx] :
                              1'b1;  // No such grid = dependency satisfied

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            result_valid <= 1'b0;
            result <= 0;
            pred_result <= 1'b0;
            signal_complete <= 1'b0;
            signal_grid_id <= 0;
            signal_token <= 0;
            launch_dep_req <= 1'b0;
            launch_dep_token <= 0;
            saved_func <= 0;
            saved_src_a <= 0;
            saved_src_b <= 0;
            wait_grid_id <= 0;
            wait_token <= 0;
            waiting <= 1'b0;
            next_token <= 16'd1;
            grid_completed <= 0;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            signal_complete <= 1'b0;
            launch_dep_req <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (valid_in && opcode == `OP_GRIDDEPCTRL) begin
                        saved_func <= func;
                        saved_src_a <= src_a;
                        saved_src_b <= src_b;
                        state <= ST_EXECUTE;
                    end
                end

                ST_EXECUTE: begin
                    case (saved_func)
                        `GRIDDEP_WAIT: begin
                            // Wait for grid dependency to be satisfied
                            wait_token <= saved_src_a[TOKEN_WIDTH-1:0];

                            // Check if dependency is already satisfied
                            if (dep_check_result) begin
                                result <= 32'd0;  // Success, no wait needed
                                pred_result <= 1'b1;  // Dependency satisfied
                                result_valid <= 1'b1;
                                done <= 1'b1;
                                state <= ST_IDLE;
                                `ifdef SIMULATION
                                $display("[GRIDDEP] WAIT: token=%0d - already satisfied", saved_src_a[TOKEN_WIDTH-1:0]);
                                `endif
                            end else begin
                                // Need to wait
                                waiting <= 1'b1;
                                state <= ST_WAIT_DEP;
                                `ifdef SIMULATION
                                $display("[GRIDDEP] WAIT: token=%0d - waiting", saved_src_a[TOKEN_WIDTH-1:0]);
                                `endif
                            end
                        end

                        `GRIDDEP_LAUNCH_DEP: begin
                            // Launch dependent grid (request to scheduler)
                            launch_dep_req <= 1'b1;
                            launch_dep_token <= saved_src_a[TOKEN_WIDTH-1:0];
                            result <= {16'b0, saved_src_a[TOKEN_WIDTH-1:0]};  // Echo token
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[GRIDDEP] LAUNCH_DEP: token=%0d", saved_src_a[TOKEN_WIDTH-1:0]);
                            `endif
                        end

                        `GRIDDEP_SIGNAL: begin
                            // Signal grid completion
                            signal_complete <= 1'b1;
                            signal_grid_id <= grid_id;
                            signal_token <= grid_token;
                            grid_completed[grid_id[GRID_IDX_W-1:0]] <= 1'b1;
                            result <= {28'b0, grid_id};
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[GRIDDEP] SIGNAL: grid_id=%0d token=%0d", grid_id, grid_token);
                            `endif
                        end

                        `GRIDDEP_GET_TOKEN: begin
                            // Allocate and return a new dependency token
                            result <= {16'b0, next_token};
                            next_token <= next_token + 1;
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[GRIDDEP] GET_TOKEN: allocated token=%0d", next_token);
                            `endif
                        end

                        default: begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    endcase
                end

                ST_WAIT_DEP: begin
                    // Check if dependency is now satisfied
                    if (dep_check_result) begin
                        waiting <= 1'b0;
                        result <= 32'd0;  // Success
                        pred_result <= 1'b1;
                        result_valid <= 1'b1;
                        done <= 1'b1;
                        state <= ST_IDLE;
                        `ifdef SIMULATION
                        $display("[GRIDDEP] WAIT: token=%0d - satisfied", wait_token);
                        `endif
                    end
                    // Otherwise keep waiting
                end

                ST_COMPLETE: begin
                    done <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
