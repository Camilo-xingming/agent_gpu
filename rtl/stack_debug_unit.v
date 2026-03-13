//============================================================================
// RalphGPU - Stack and Debug Unit
// Implements stack management and debug/monitoring instructions
// PTX Instructions: alloca, stacksave, stackrestore, brkpt, trap, pmevent,
//                   nanosleep, setmaxnreg
//
// Reference: NVIDIA PTX ISA 8.5+
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module stack_debug_unit #(
    parameter DATA_WIDTH = 32,
    parameter NUM_WARPS = 4,
    parameter STACK_SIZE_PER_WARP = 4096,  // 4KB per warp local stack
    parameter WARP_ID_W = 2
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   opcode,         // OP_STACK, OP_DEBUG, OP_MISC
    input  wire [5:0]                   func,           // Function code
    input  wire [WARP_ID_W-1:0]         warp_id,        // Warp issuing the operation
    input  wire [DATA_WIDTH-1:0]        src_a,          // Operand (size for alloca, delay for nanosleep)
    input  wire [DATA_WIDTH-1:0]        src_b,          // Additional operand

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          result_valid,
    output reg  [DATA_WIDTH-1:0]        result,

    //------------------------------------------------------------------------
    // Debug/Exception Interface
    //------------------------------------------------------------------------
    output reg                          brkpt_hit,      // Breakpoint triggered
    output reg                          trap_hit,       // Trap triggered
    output reg  [15:0]                  trap_code,      // Trap code
    output reg                          pmevent_pulse,  // Performance event pulse
    output reg  [7:0]                   pmevent_id,     // Event ID for performance monitoring

    //------------------------------------------------------------------------
    // Warp Stall Interface (for nanosleep)
    //------------------------------------------------------------------------
    output reg  [NUM_WARPS-1:0]         warp_stall,     // Stall specific warps

    //------------------------------------------------------------------------
    // Configuration Interface
    //------------------------------------------------------------------------
    input  wire [7:0]                   max_reg_limit,  // System-wide max register limit
    output reg [NUM_WARPS*8-1:0] warp_max_regs  // Per-warp register limits
);

    //------------------------------------------------------------------------
    // Per-warp stack pointers
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] stack_ptr [0:NUM_WARPS-1];
    reg [DATA_WIDTH-1:0] stack_base [0:NUM_WARPS-1];
    reg [DATA_WIDTH-1:0] stack_limit [0:NUM_WARPS-1];

    //------------------------------------------------------------------------
    // Nanosleep timer per warp
    //------------------------------------------------------------------------
    reg [31:0] sleep_counter [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] sleeping;

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE = 2'd0;
    localparam ST_EXECUTE = 2'd1;
    localparam ST_SLEEP_WAIT = 2'd2;
    localparam ST_DONE = 2'd3;

    reg [1:0] state;
    reg [5:0] saved_opcode;
    reg [5:0] saved_func;
    reg [WARP_ID_W-1:0] saved_warp;
    reg [DATA_WIDTH-1:0] saved_src_a;

    //------------------------------------------------------------------------
    // Initialization
    //------------------------------------------------------------------------
    integer i;

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            result_valid <= 1'b0;
            result <= 0;
            brkpt_hit <= 1'b0;
            trap_hit <= 1'b0;
            trap_code <= 16'b0;
            pmevent_pulse <= 1'b0;
            pmevent_id <= 8'b0;
            warp_stall <= {NUM_WARPS{1'b0}};
            sleeping <= {NUM_WARPS{1'b0}};
            saved_opcode <= 6'b0;
            saved_func <= 6'b0;
            saved_warp <= 0;
            saved_src_a <= 0;

            // Initialize per-warp state
            for (i = 0; i < NUM_WARPS; i = i + 1) begin
                stack_ptr[i] <= i * STACK_SIZE_PER_WARP;  // Stack base for each warp
                stack_base[i] <= i * STACK_SIZE_PER_WARP;
                stack_limit[i] <= (i + 1) * STACK_SIZE_PER_WARP;
                sleep_counter[i] <= 32'b0;
                warp_max_regs[i*8 +: 8] <= max_reg_limit;  // Default to system limit
            end
        end else begin
            // Clear single-cycle signals
            done <= 1'b0;
            result_valid <= 1'b0;
            brkpt_hit <= 1'b0;
            trap_hit <= 1'b0;
            pmevent_pulse <= 1'b0;

            // Process sleep counters and update stall signals
            for (i = 0; i < NUM_WARPS; i = i + 1) begin
                if (sleeping[i]) begin
                    if (sleep_counter[i] > 0) begin
                        sleep_counter[i] <= sleep_counter[i] - 1;
                        warp_stall[i] <= 1'b1;
                    end else begin
                        sleeping[i] <= 1'b0;
                        warp_stall[i] <= 1'b0;
                    end
                end
            end

            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        saved_opcode <= opcode;
                        saved_func <= func;
                        saved_warp <= warp_id;
                        saved_src_a <= src_a;
                        state <= ST_EXECUTE;
                    end
                end

                ST_EXECUTE: begin
                    case (saved_opcode)
                        //====================================================
                        // Stack Operations
                        //====================================================
                        `OP_STACK: begin
                            case (saved_func)
                                `STACK_ALLOCA: begin
                                    // alloca: allocate space on stack, return pointer
                                    // Align allocation to 16 bytes
                                    if (stack_ptr[saved_warp] + ((saved_src_a + 15) & ~32'hF) <= stack_limit[saved_warp]) begin
                                        result <= stack_ptr[saved_warp];
                                        stack_ptr[saved_warp] <= stack_ptr[saved_warp] + ((saved_src_a + 15) & ~32'hF);
                                        result_valid <= 1'b1;
                                        `ifdef SIMULATION
                                        $display("[STACK] Warp %0d ALLOCA: size=%0d ptr=0x%08x new_sp=0x%08x", // keep
                                                 saved_warp, saved_src_a, stack_ptr[saved_warp],
                                                 stack_ptr[saved_warp] + ((saved_src_a + 15) & ~32'hF));
                                        `endif
                                    end else begin
                                        // Stack overflow - return 0 (null pointer)
                                        result <= 32'b0;
                                        result_valid <= 1'b1;
                                        `ifdef SIMULATION
                                        $display("[STACK] Warp %0d ALLOCA: OVERFLOW! size=%0d", saved_warp, saved_src_a); // keep
                                        `endif
                                    end
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end

                                `STACK_SAVE: begin
                                    // stacksave: return current stack pointer
                                    result <= stack_ptr[saved_warp];
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[STACK] Warp %0d SAVE: sp=0x%08x", saved_warp, stack_ptr[saved_warp]); // keep
                                    `endif
                                end

                                `STACK_RESTORE: begin
                                    // stackrestore: restore stack pointer from operand
                                    // Validate the pointer is within bounds
                                    if (saved_src_a >= stack_base[saved_warp] &&
                                        saved_src_a <= stack_limit[saved_warp]) begin
                                        stack_ptr[saved_warp] <= saved_src_a;
                                        `ifdef SIMULATION
                                        $display("[STACK] Warp %0d RESTORE: sp=0x%08x", saved_warp, saved_src_a); // keep
                                        `endif
                                    end else begin
                                        `ifdef SIMULATION
                                        $display("[STACK] Warp %0d RESTORE: INVALID sp=0x%08x", saved_warp, saved_src_a); // keep
                                        `endif
                                    end
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end

                                default: begin
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end
                            endcase
                        end

                        //====================================================
                        // Debug Operations
                        //====================================================
                        `OP_DEBUG: begin
                            case (saved_func)
                                `DEBUG_BRKPT: begin
                                    // brkpt: trigger breakpoint
                                    brkpt_hit <= 1'b1;
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[DEBUG] Warp %0d BRKPT hit", saved_warp); // keep
                                    `endif
                                end

                                `DEBUG_TRAP: begin
                                    // trap: software trap with code
                                    trap_hit <= 1'b1;
                                    trap_code <= saved_src_a[15:0];
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[DEBUG] Warp %0d TRAP code=%0d", saved_warp, saved_src_a[15:0]); // keep
                                    `endif
                                end

                                `DEBUG_PMEVENT: begin
                                    // pmevent: signal performance monitoring event
                                    pmevent_pulse <= 1'b1;
                                    pmevent_id <= saved_src_a[7:0];
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[DEBUG] Warp %0d PMEVENT id=%0d", saved_warp, saved_src_a[7:0]); // keep
                                    `endif
                                end

                                default: begin
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end
                            endcase
                        end

                        //====================================================
                        // Misc Operations
                        //====================================================
                        `OP_MISC: begin
                            case (saved_func)
                                `MISC_NANOSLEEP: begin
                                    // nanosleep: stall warp for specified cycles
                                    // Convert nanoseconds to cycles (simplified: 1ns = 1 cycle)
                                    sleep_counter[saved_warp] <= saved_src_a;
                                    sleeping[saved_warp] <= 1'b1;
                                    warp_stall[saved_warp] <= 1'b1;
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[MISC] Warp %0d NANOSLEEP cycles=%0d", saved_warp, saved_src_a); // keep
                                    `endif
                                end

                                `MISC_SETMAXNREG: begin
                                    // setmaxnreg: set maximum register count for warp
                                    // Clamp to system limit
                                    if (saved_src_a <= {24'd0, max_reg_limit}) begin
                                        warp_max_regs[saved_warp*8 +: 8] <= saved_src_a[7:0];
                                    end else begin
                                        warp_max_regs[saved_warp*8 +: 8] <= max_reg_limit;
                                    end
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[MISC] Warp %0d SETMAXNREG regs=%0d", saved_warp, saved_src_a[7:0]); // keep
                                    `endif
                                end

                                default: begin
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end
                            endcase
                        end

                        default: begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    endcase
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
