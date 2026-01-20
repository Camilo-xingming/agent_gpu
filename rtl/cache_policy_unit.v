//============================================================================
// RalphGPU - Cache Policy Unit
// Implements cache policy management for Hopper+ architecture
// PTX Instructions: createpolicy, applypriority, discard
// Also includes address space queries: isspacep, mapa, getctarank
//
// Reference: NVIDIA PTX ISA 8.5+, Hopper/Blackwell Architecture
//============================================================================

`include "gpu_defines.vh"

module cache_policy_unit #(
    parameter DATA_WIDTH = 32,
    parameter SMEM_BASE = 32'hFFFE0000,   // Shared memory base (top of address space)
    parameter SMEM_SIZE = 32'h00020000,   // 128KB shared memory window
    parameter LOCAL_BASE = 32'hFFFC0000,  // Local memory base
    parameter LOCAL_SIZE = 32'h00020000,  // 128KB local memory window
    parameter CONST_BASE = 32'hFFFA0000,  // Constant memory base
    parameter CONST_SIZE = 32'h00020000,  // 128KB constant memory window
    parameter PARAM_BASE = 32'hFFF80000,  // Parameter memory base
    parameter PARAM_SIZE = 32'h00020000,  // 128KB parameter memory window
    parameter NUM_SMs = 2                 // Number of SMs in cluster
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   opcode,         // OP_CACHE_POLICY, OP_ISSPACEP, OP_MAPA, OP_GETCTARANK
    input  wire [5:0]                   func,           // Function code within opcode
    input  wire [DATA_WIDTH-1:0]        src_a,          // Source address or value
    input  wire [DATA_WIDTH-1:0]        src_b,          // Additional operand
    input  wire [2:0]                   cache_level,    // Cache level (L1/L2/etc)

    //------------------------------------------------------------------------
    // Cluster Configuration (for getctarank)
    //------------------------------------------------------------------------
    input  wire [3:0]                   cta_id,         // This CTA's ID within cluster
    input  wire [3:0]                   cluster_size,   // Number of CTAs in cluster
    input  wire [3:0]                   sm_id,          // This SM's ID

    //------------------------------------------------------------------------
    // Result Interface
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          result_valid,
    output reg  [DATA_WIDTH-1:0]        result,
    output reg                          pred_result,    // Predicate result for isspacep

    //------------------------------------------------------------------------
    // Cache Control Interface (to L1/L2 cache controllers)
    //------------------------------------------------------------------------
    output reg                          cache_ctrl_valid,
    output reg  [2:0]                   cache_ctrl_op,      // 0=createpolicy, 1=applypriority, 2=discard
    output reg  [DATA_WIDTH-1:0]        cache_ctrl_addr,
    output reg  [7:0]                   cache_ctrl_policy,  // Policy token
    output reg  [2:0]                   cache_ctrl_level,   // Target cache level
    input  wire                         cache_ctrl_done
);

    //------------------------------------------------------------------------
    // Internal registers
    //------------------------------------------------------------------------
    reg [5:0] saved_opcode;
    reg [5:0] saved_func;
    reg [DATA_WIDTH-1:0] saved_addr;
    reg [7:0] policy_counter;  // Policy token generator

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE = 2'd0;
    localparam ST_EXECUTE = 2'd1;
    localparam ST_WAIT_CACHE = 2'd2;
    localparam ST_DONE = 2'd3;

    reg [1:0] state;

    //------------------------------------------------------------------------
    // Address space detection (combinational)
    // Note: Using offset comparison to avoid overflow issues
    //------------------------------------------------------------------------
    wire in_global = (src_a < PARAM_BASE);  // Below special spaces = global
    wire in_shared = (src_a >= SMEM_BASE) && ((src_a - SMEM_BASE) < SMEM_SIZE);
    wire in_local  = (src_a >= LOCAL_BASE) && ((src_a - LOCAL_BASE) < LOCAL_SIZE);
    wire in_const  = (src_a >= CONST_BASE) && ((src_a - CONST_BASE) < CONST_SIZE);
    wire in_param  = (src_a >= PARAM_BASE) && ((src_a - PARAM_BASE) < PARAM_SIZE);

    //------------------------------------------------------------------------
    // Address mapping (combinational)
    // Convert between address spaces
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] mapped_addr;

    always @(*) begin
        mapped_addr = src_a;  // Default passthrough

        case (saved_func)
            `MAPA_TO_GLOBAL: begin
                // Map shared address to global (add SM offset)
                if (in_shared) begin
                    mapped_addr = (src_a - SMEM_BASE) + (sm_id * SMEM_SIZE);
                end
            end
            `MAPA_TO_SHARED: begin
                // Map global to shared (compute offset)
                mapped_addr = (src_a % SMEM_SIZE) + SMEM_BASE;
            end
            `MAPA_FROM_SHARED: begin
                // Map shared to generic pointer
                mapped_addr = src_a;  // Same in this implementation
            end
            `MAPA_TO_LOCAL: begin
                // Map global to local space
                mapped_addr = (src_a % LOCAL_SIZE) + LOCAL_BASE;
            end
            default: mapped_addr = src_a;
        endcase
    end

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
            cache_ctrl_valid <= 1'b0;
            cache_ctrl_op <= 3'b0;
            cache_ctrl_addr <= 0;
            cache_ctrl_policy <= 8'b0;
            cache_ctrl_level <= 3'b0;
            saved_opcode <= 6'b0;
            saved_func <= 6'b0;
            saved_addr <= 0;
            policy_counter <= 8'd1;  // Start from 1 (0 = invalid policy)
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            cache_ctrl_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (valid_in) begin
                        saved_opcode <= opcode;
                        saved_func <= func;
                        saved_addr <= src_a;
                        state <= ST_EXECUTE;
                    end
                end

                ST_EXECUTE: begin
                    case (saved_opcode)
                        //====================================================
                        // Cache Policy Operations
                        //====================================================
                        `OP_CACHE_POLICY: begin
                            case (saved_func)
                                `CACHE_CREATEPOLICY: begin
                                    // Create a new cache policy token
                                    // Returns token in result register
                                    result <= {24'b0, policy_counter};
                                    policy_counter <= policy_counter + 1;
                                    result_valid <= 1'b1;
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                    `ifdef SIMULATION
                                    $display("[CACHE_POLICY] CREATEPOLICY: token=%0d", policy_counter);
                                    `endif
                                end

                                `CACHE_APPLYPRIORITY: begin
                                    // Apply priority to cache lines at address
                                    cache_ctrl_valid <= 1'b1;
                                    cache_ctrl_op <= 3'd1;  // applypriority
                                    cache_ctrl_addr <= saved_addr;
                                    cache_ctrl_policy <= src_b[7:0];  // Policy from src_b
                                    cache_ctrl_level <= cache_level;
                                    state <= ST_WAIT_CACHE;
                                    `ifdef SIMULATION
                                    $display("[CACHE_POLICY] APPLYPRIORITY: addr=0x%08x policy=%0d level=%0d",
                                             saved_addr, src_b[7:0], cache_level);
                                    `endif
                                end

                                `CACHE_DISCARD: begin
                                    // Mark cache lines for eviction (invalidate without writeback)
                                    cache_ctrl_valid <= 1'b1;
                                    cache_ctrl_op <= 3'd2;  // discard
                                    cache_ctrl_addr <= saved_addr;
                                    cache_ctrl_level <= cache_level;
                                    state <= ST_WAIT_CACHE;
                                    `ifdef SIMULATION
                                    $display("[CACHE_POLICY] DISCARD: addr=0x%08x level=%0d",
                                             saved_addr, cache_level);
                                    `endif
                                end

                                default: begin
                                    done <= 1'b1;
                                    state <= ST_IDLE;
                                end
                            endcase
                        end

                        //====================================================
                        // Address Space Query (isspacep)
                        //====================================================
                        `OP_ISSPACEP: begin
                            case (saved_func)
                                `ISSPACEP_GLOBAL: begin
                                    pred_result <= in_global;
                                    result <= {31'b0, in_global};
                                end
                                `ISSPACEP_SHARED: begin
                                    pred_result <= in_shared;
                                    result <= {31'b0, in_shared};
                                end
                                `ISSPACEP_LOCAL: begin
                                    pred_result <= in_local;
                                    result <= {31'b0, in_local};
                                end
                                `ISSPACEP_CONST: begin
                                    pred_result <= in_const;
                                    result <= {31'b0, in_const};
                                end
                                `ISSPACEP_PARAM: begin
                                    pred_result <= in_param;
                                    result <= {31'b0, in_param};
                                end
                                default: begin
                                    pred_result <= 1'b0;
                                    result <= 0;
                                end
                            endcase
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[ISSPACEP] addr=0x%08x func=%0d result=%0d",
                                     saved_addr, saved_func, pred_result);
                            `endif
                        end

                        //====================================================
                        // Address Mapping (mapa)
                        //====================================================
                        `OP_MAPA: begin
                            result <= mapped_addr;
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[MAPA] src=0x%08x func=%0d mapped=0x%08x",
                                     saved_addr, saved_func, mapped_addr);
                            `endif
                        end

                        //====================================================
                        // Get CTA Rank in Cluster (getctarank)
                        //====================================================
                        `OP_GETCTARANK: begin
                            // Return this CTA's rank within the cluster
                            result <= {28'b0, cta_id};
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[GETCTARANK] cta_id=%0d cluster_size=%0d sm_id=%0d",
                                     cta_id, cluster_size, sm_id);
                            `endif
                        end

                        default: begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    endcase
                end

                ST_WAIT_CACHE: begin
                    // Wait for cache controller to acknowledge
                    if (cache_ctrl_done) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
