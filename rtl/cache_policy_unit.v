//============================================================================
// RalphGPU - Cache Policy Unit
// Implements cache policy management for Hopper+ architecture
// PTX Instructions: createpolicy, applypriority, discard, isspacep, mapa, getctarank
//
// All operations use OP_CACHE_POLICY opcode with func codes:
//   0x00: CACHE_CREATEPOLICY  - Create cache policy token
//   0x01: CACHE_APPLYPRIORITY - Apply priority to cache lines
//   0x02: CACHE_DISCARD       - Mark cache lines for eviction
//   0x04: CACHE_ISSPACEP      - Test address space membership (space type in src_b[2:0])
//   0x05: CACHE_MAPA          - Map address between spaces (map type in src_b[2:0])
//   0x06: CACHE_GETCTARANK    - Get CTA rank within cluster
//
// Reference: NVIDIA PTX ISA 8.5+, Hopper/Blackwell Architecture
//============================================================================

`timescale 1ns / 1ps
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
    input  wire [5:0]                   opcode,         // OP_CACHE_POLICY
    input  wire [5:0]                   func,           // Function code (see header)
    input  wire [DATA_WIDTH-1:0]        src_a,          // Source address or value
    input  wire [DATA_WIDTH-1:0]        src_b,          // Additional operand (space/map type, policy)
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
    reg [5:0] saved_func;
    reg [DATA_WIDTH-1:0] saved_addr;
    reg [DATA_WIDTH-1:0] saved_src_b;
    reg [7:0] policy_counter;  // Policy token generator

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE = 2'd0;
    localparam ST_EXECUTE = 2'd1;
    localparam ST_WAIT_CACHE = 2'd2;

    reg [1:0] state;

    //------------------------------------------------------------------------
    // Address space detection (combinational)
    //------------------------------------------------------------------------
    wire in_global = (src_a < PARAM_BASE);
    wire in_shared = (src_a >= SMEM_BASE) && ((src_a - SMEM_BASE) < SMEM_SIZE);
    wire in_local  = (src_a >= LOCAL_BASE) && ((src_a - LOCAL_BASE) < LOCAL_SIZE);
    wire in_const  = (src_a >= CONST_BASE) && ((src_a - CONST_BASE) < CONST_SIZE);
    wire in_param  = (src_a >= PARAM_BASE) && ((src_a - PARAM_BASE) < PARAM_SIZE);

    //------------------------------------------------------------------------
    // Address mapping (combinational)
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] mapped_addr;

    always @(*) begin
        mapped_addr = src_a;
        case (src_b[2:0])  // Map type in src_b[2:0]
            3'd0: mapped_addr = (in_shared) ? ((src_a - SMEM_BASE) + (sm_id * SMEM_SIZE)) : src_a;  // to_global
            3'd1: mapped_addr = (src_a % SMEM_SIZE) + SMEM_BASE;  // to_shared
            3'd2: mapped_addr = src_a;  // from_shared (generic)
            3'd3: mapped_addr = (src_a % LOCAL_SIZE) + LOCAL_BASE;  // to_local
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
            saved_func <= 6'b0;
            saved_addr <= 0;
            saved_src_b <= 0;
            policy_counter <= 8'd1;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            cache_ctrl_valid <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (valid_in && opcode == `OP_CACHE_POLICY) begin
                        saved_func <= func;
                        saved_addr <= src_a;
                        saved_src_b <= src_b;
                        state <= ST_EXECUTE;
                    end
                end

                ST_EXECUTE: begin
                    case (saved_func)
                        `CACHE_CREATEPOLICY: begin
                            result <= {24'b0, policy_counter};
                            policy_counter <= policy_counter + 1;
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[CACHE_POLICY] CREATEPOLICY: token=%0d", policy_counter); // keep
                            `endif
                        end

                        `CACHE_APPLYPRIORITY: begin
                            cache_ctrl_valid <= 1'b1;
                            cache_ctrl_op <= 3'd1;
                            cache_ctrl_addr <= saved_addr;
                            cache_ctrl_policy <= saved_src_b[7:0];
                            cache_ctrl_level <= cache_level;
                            state <= ST_WAIT_CACHE;
                            `ifdef SIMULATION
                            $display("[CACHE_POLICY] APPLYPRIORITY: addr=0x%08x policy=%0d", saved_addr, saved_src_b[7:0]); // keep
                            `endif
                        end

                        `CACHE_DISCARD: begin
                            cache_ctrl_valid <= 1'b1;
                            cache_ctrl_op <= 3'd2;
                            cache_ctrl_addr <= saved_addr;
                            cache_ctrl_level <= cache_level;
                            state <= ST_WAIT_CACHE;
                            `ifdef SIMULATION
                            $display("[CACHE_POLICY] DISCARD: addr=0x%08x", saved_addr); // keep
                            `endif
                        end

                        `CACHE_ISSPACEP: begin
                            // Space type in saved_src_b[2:0]
                            case (saved_src_b[2:0])
                                3'd0: begin pred_result <= in_global; result <= {31'b0, in_global}; end
                                3'd1: begin pred_result <= in_shared; result <= {31'b0, in_shared}; end
                                3'd2: begin pred_result <= in_local;  result <= {31'b0, in_local};  end
                                3'd3: begin pred_result <= in_const;  result <= {31'b0, in_const};  end
                                3'd4: begin pred_result <= in_param;  result <= {31'b0, in_param};  end
                                default: begin pred_result <= 1'b0; result <= 0; end
                            endcase
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[ISSPACEP] addr=0x%08x space=%0d result=%0d", saved_addr, saved_src_b[2:0], pred_result); // keep
                            `endif
                        end

                        `CACHE_MAPA: begin
                            result <= mapped_addr;
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[MAPA] src=0x%08x map_type=%0d result=0x%08x", saved_addr, saved_src_b[2:0], mapped_addr); // keep
                            `endif
                        end

                        `CACHE_GETCTARANK: begin
                            result <= {28'b0, cta_id};
                            result_valid <= 1'b1;
                            done <= 1'b1;
                            state <= ST_IDLE;
                            `ifdef SIMULATION
                            $display("[GETCTARANK] cta_id=%0d", cta_id); // keep
                            `endif
                        end

                        default: begin
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end
                    endcase
                end

                ST_WAIT_CACHE: begin
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
