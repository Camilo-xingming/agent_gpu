//============================================================================
// RalphGPU - Warp Collective Unit
// Implements warp-level collective operations for Hopper/Blackwell:
// - match.sync.any/all: Predicate matching across warp threads
// - elect.sync: Leader election within participating threads
// - red.async: Asynchronous reduction to shared memory
//
// Reference: PTX ISA 8.5+, CUDA Programming Guide (Warp-Level Primitives)
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module warp_collective_unit #(
    parameter NUM_WARPS = 4,
    parameter NUM_LANES = 32,
    parameter SHARED_MEM_ADDR_W = 14
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Issue Interface
    //------------------------------------------------------------------------
    input  wire                         valid_in,
    input  wire [5:0]                   opcode,
    input  wire [5:0]                   func,
    input  wire [1:0]                   warp_id,
    input  wire [31:0]                  thread_mask,     // Active thread mask
    input  wire [31:0]                  membermask,      // Participating thread mask (from ra)
    input  wire [31:0]                  src_data,        // Source data (value to match/reduce)
    input  wire [SHARED_MEM_ADDR_W-1:0] dst_addr,        // Destination address (for red.async)

    //------------------------------------------------------------------------
    // Per-Lane Data Input (for parallel comparison/reduction)
    // Packed as 32 lanes x 32 bits = 1024 bits
    //------------------------------------------------------------------------
    input  wire [NUM_LANES*32-1:0]      lane_data_packed,

    //------------------------------------------------------------------------
    // Results
    //------------------------------------------------------------------------
    output reg                          done,
    output reg                          stall_warp,      // Stall warp until completion
    output reg  [31:0]                  result_mask,     // Result mask (for match.sync)
    output reg  [31:0]                  result_data,     // Result data (elected lane id, reduction result)
    output reg                          pred_result,     // Predicate result (for elect.sync)

    //------------------------------------------------------------------------
    // Shared Memory Interface (for red.async)
    //------------------------------------------------------------------------
    output reg                          smem_red_valid,
    output reg  [SHARED_MEM_ADDR_W-1:0] smem_red_addr,
    output reg  [31:0]                  smem_red_data,
    output reg  [2:0]                   smem_red_op,     // Reduction operation
    input  wire                         smem_red_done,

    //------------------------------------------------------------------------
    // mbarrier signal interface (for red.async completion)
    //------------------------------------------------------------------------
    output reg                          mbarrier_arrive_trigger,
    output reg  [3:0]                   mbarrier_id
);

    //------------------------------------------------------------------------
    // Internal State
    //------------------------------------------------------------------------
    localparam ST_IDLE        = 3'd0;
    localparam ST_MATCH       = 3'd1;
    localparam ST_ELECT       = 3'd2;
    localparam ST_RED_COMPUTE = 3'd3;
    localparam ST_RED_WRITE   = 3'd4;
    localparam ST_COMPLETE    = 3'd5;

    reg [2:0]  state;
    reg [5:0]  saved_func;
    reg [31:0] saved_membermask;
    reg [31:0] saved_src_data;
    reg [SHARED_MEM_ADDR_W-1:0] saved_dst_addr;
    reg [1:0]  saved_warp_id;

    // Reduction accumulator
    reg [31:0] reduction_acc;
    reg [5:0]  red_lane_idx;

    // Per-lane data copy for reduction
    reg [31:0] lane_data_copy [0:NUM_LANES-1];

    // Extract lane data from packed input
    function [31:0] get_lane_data;
        input integer lane_idx;
        begin
            get_lane_data = lane_data_packed[lane_idx*32 +: 32];
        end
    endfunction

    //------------------------------------------------------------------------
    // match.sync logic
    // match.sync.any membermask, a, b: Returns mask of threads where Ra == Rb
    // match.sync.all membermask, a, b: Same, but also checks all participating threads match
    //------------------------------------------------------------------------
    wire [31:0] match_result;
    wire        all_threads_match;

    // Compare each lane's value against the source data
    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : match_gen
            assign match_result[i] = saved_membermask[i] && (lane_data_copy[i] == saved_src_data);
        end
    endgenerate

    // Check if ALL participating threads have the same value
    assign all_threads_match = (match_result == saved_membermask);

    //------------------------------------------------------------------------
    // elect.sync logic
    // elect.sync.one membermask: Elects one thread from participating threads
    // Returns: predicate true for elected thread, false for others
    //          result_data = lane id of elected thread
    //------------------------------------------------------------------------
    wire [4:0]  elected_lane;
    wire        has_participating_thread;

    // Find first set bit (lowest numbered participating thread)
    assign elected_lane =
        saved_membermask[0]  ? 5'd0  :
        saved_membermask[1]  ? 5'd1  :
        saved_membermask[2]  ? 5'd2  :
        saved_membermask[3]  ? 5'd3  :
        saved_membermask[4]  ? 5'd4  :
        saved_membermask[5]  ? 5'd5  :
        saved_membermask[6]  ? 5'd6  :
        saved_membermask[7]  ? 5'd7  :
        saved_membermask[8]  ? 5'd8  :
        saved_membermask[9]  ? 5'd9  :
        saved_membermask[10] ? 5'd10 :
        saved_membermask[11] ? 5'd11 :
        saved_membermask[12] ? 5'd12 :
        saved_membermask[13] ? 5'd13 :
        saved_membermask[14] ? 5'd14 :
        saved_membermask[15] ? 5'd15 :
        saved_membermask[16] ? 5'd16 :
        saved_membermask[17] ? 5'd17 :
        saved_membermask[18] ? 5'd18 :
        saved_membermask[19] ? 5'd19 :
        saved_membermask[20] ? 5'd20 :
        saved_membermask[21] ? 5'd21 :
        saved_membermask[22] ? 5'd22 :
        saved_membermask[23] ? 5'd23 :
        saved_membermask[24] ? 5'd24 :
        saved_membermask[25] ? 5'd25 :
        saved_membermask[26] ? 5'd26 :
        saved_membermask[27] ? 5'd27 :
        saved_membermask[28] ? 5'd28 :
        saved_membermask[29] ? 5'd29 :
        saved_membermask[30] ? 5'd30 :
        saved_membermask[31] ? 5'd31 : 5'd0;

    assign has_participating_thread = |saved_membermask;

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            stall_warp <= 1'b0;
            result_mask <= 32'b0;
            result_data <= 32'b0;
            pred_result <= 1'b0;
            smem_red_valid <= 1'b0;
            smem_red_addr <= 0;
            smem_red_data <= 32'b0;
            smem_red_op <= 3'b0;
            mbarrier_arrive_trigger <= 1'b0;
            mbarrier_id <= 4'b0;
            saved_func <= 6'b0;
            saved_membermask <= 32'b0;
            saved_src_data <= 32'b0;
            saved_dst_addr <= 0;
            saved_warp_id <= 2'b0;
            reduction_acc <= 32'b0;
            red_lane_idx <= 6'b0;
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                lane_data_copy[j] <= 32'b0;
            end
        end else begin
            done <= 1'b0;
            smem_red_valid <= 1'b0;
            mbarrier_arrive_trigger <= 1'b0;

            case (state)
                ST_IDLE: begin
                    stall_warp <= 1'b0;
                    if (valid_in) begin
                        // Save inputs
                        saved_func <= func;
                        saved_membermask <= membermask & thread_mask;  // Only active threads
                        saved_src_data <= src_data;
                        saved_dst_addr <= dst_addr;
                        saved_warp_id <= warp_id;

                        // Copy lane data from packed input
                        for (j = 0; j < NUM_LANES; j = j + 1) begin
                            lane_data_copy[j] <= lane_data_packed[j*32 +: 32];
                        end

                        case (opcode)
                            `OP_MATCH_SYNC: begin
                                state <= ST_MATCH;
                                stall_warp <= 1'b1;
                                `ifdef SIMULATION
                                $display("[WARP_COLL] match.sync: membermask=0x%08x src=0x%08x", // keep
                                         membermask & thread_mask, src_data);
                                `endif
                            end

                            `OP_ELECT_SYNC: begin
                                state <= ST_ELECT;
                                stall_warp <= 1'b1;
                                `ifdef SIMULATION
                                $display("[WARP_COLL] elect.sync: membermask=0x%08x", // keep
                                         membermask & thread_mask);
                                `endif
                            end

                            `OP_RED_ASYNC: begin
                                state <= ST_RED_COMPUTE;
                                stall_warp <= 1'b0;  // Async - don't stall
                                red_lane_idx <= 6'd0;
                                // Initialize accumulator based on operation
                                case (func)
                                    `RED_ASYNC_ADD: reduction_acc <= 32'b0;
                                    `RED_ASYNC_MIN: reduction_acc <= 32'hFFFFFFFF;
                                    `RED_ASYNC_MAX: reduction_acc <= 32'b0;
                                    `RED_ASYNC_AND: reduction_acc <= 32'hFFFFFFFF;
                                    `RED_ASYNC_OR:  reduction_acc <= 32'b0;
                                    `RED_ASYNC_XOR: reduction_acc <= 32'b0;
                                    default:        reduction_acc <= 32'b0;
                                endcase
                                `ifdef SIMULATION
                                $display("[WARP_COLL] red.async: func=%0d membermask=0x%08x dst=0x%04x", // keep
                                         func, membermask & thread_mask, dst_addr);
                                `endif
                            end

                            default: begin
                                done <= 1'b1;  // Unknown opcode, complete immediately
                            end
                        endcase
                    end
                end

                ST_MATCH: begin
                    // match.sync completes in one cycle
                    case (saved_func)
                        `MATCH_ANY: begin
                            result_mask <= match_result;
                            result_data <= {27'b0, elected_lane};  // Return first matching lane
                            pred_result <= |match_result;  // True if any match
                        end
                        `MATCH_ALL: begin
                            result_mask <= all_threads_match ? saved_membermask : 32'b0;
                            result_data <= all_threads_match ? saved_src_data : 32'b0;
                            pred_result <= all_threads_match;
                        end
                        default: begin
                            result_mask <= match_result;
                            result_data <= 32'b0;
                            pred_result <= |match_result;
                        end
                    endcase

                    `ifdef SIMULATION
                    $display("[WARP_COLL] match.sync done: result_mask=0x%08x pred=%b", // keep
                             match_result, |match_result);
                    `endif

                    done <= 1'b1;
                    stall_warp <= 1'b0;
                    state <= ST_IDLE;
                end

                ST_ELECT: begin
                    // elect.sync completes in one cycle
                    if (has_participating_thread) begin
                        // Generate per-lane predicate: only elected lane gets true
                        result_mask <= (1'b1 << elected_lane);
                        result_data <= {27'b0, elected_lane};
                        pred_result <= 1'b1;  // Election succeeded
                    end else begin
                        result_mask <= 32'b0;
                        result_data <= 32'b0;
                        pred_result <= 1'b0;  // No participating threads
                    end

                    `ifdef SIMULATION
                    $display("[WARP_COLL] elect.sync done: elected_lane=%0d mask=0x%08x", // keep
                             elected_lane, (1'b1 << elected_lane));
                    `endif

                    done <= 1'b1;
                    stall_warp <= 1'b0;
                    state <= ST_IDLE;
                end

                ST_RED_COMPUTE: begin
                    // Sequential reduction across participating lanes
                    if (red_lane_idx < NUM_LANES) begin
                        if (saved_membermask[red_lane_idx]) begin
                            case (saved_func)
                                `RED_ASYNC_ADD: reduction_acc <= reduction_acc + lane_data_copy[red_lane_idx];
                                `RED_ASYNC_MIN: reduction_acc <= (lane_data_copy[red_lane_idx] < reduction_acc) ?
                                                                 lane_data_copy[red_lane_idx] : reduction_acc;
                                `RED_ASYNC_MAX: reduction_acc <= (lane_data_copy[red_lane_idx] > reduction_acc) ?
                                                                 lane_data_copy[red_lane_idx] : reduction_acc;
                                `RED_ASYNC_AND: reduction_acc <= reduction_acc & lane_data_copy[red_lane_idx];
                                `RED_ASYNC_OR:  reduction_acc <= reduction_acc | lane_data_copy[red_lane_idx];
                                `RED_ASYNC_XOR: reduction_acc <= reduction_acc ^ lane_data_copy[red_lane_idx];
                                default: ; // lint: CASEINCOMPLETE
                            endcase
                        end
                        red_lane_idx <= red_lane_idx + 1;
                    end else begin
                        // Reduction complete, write to shared memory
                        state <= ST_RED_WRITE;
                        smem_red_valid <= 1'b1;
                        smem_red_addr <= saved_dst_addr;
                        smem_red_data <= reduction_acc;
                        smem_red_op <= saved_func[2:0];

                        `ifdef SIMULATION
                        $display("[WARP_COLL] red.async computed: result=0x%08x -> smem[0x%04x]", // keep
                                 reduction_acc, saved_dst_addr);
                        `endif
                    end
                end

                ST_RED_WRITE: begin
                    smem_red_valid <= 1'b1;
                    if (smem_red_done) begin
                        smem_red_valid <= 1'b0;
                        // Optionally trigger mbarrier arrive
                        mbarrier_arrive_trigger <= 1'b1;
                        mbarrier_id <= 4'd0;  // Default barrier ID

                        result_data <= reduction_acc;
                        done <= 1'b1;
                        state <= ST_IDLE;

                        `ifdef SIMULATION
                        $display("[WARP_COLL] red.async done: wrote 0x%08x to smem[0x%04x]", // keep
                                 reduction_acc, saved_dst_addr);
                        `endif
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
