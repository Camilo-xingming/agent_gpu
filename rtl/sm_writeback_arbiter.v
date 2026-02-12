//============================================================================
// sm_writeback_arbiter.v — Writeback Arbiter for Streaming Multiprocessor
//
// Extracted from streaming_multiprocessor_v2.v to reduce god-module complexity.
// Implements round-robin arbitration across 17 FU writeback sources,
// selecting one per cycle to write back to the register file.
//
// Interface:
//   Inputs:  Per-FU queue outputs (warp/rd/mask/data) + empty flags
//            + non-queued sources (memory latches, atomic, mbarrier, etc.)
//   Outputs: wb_valid, wb_warp_id, wb_rd, wb_data, wb_mask
//            wb_found, wb_sel (for external latch-clearing logic)
//            Per-FU pop signals (WBQ drain)
//============================================================================
`timescale 1ns / 1ps
module sm_writeback_arbiter #(
    parameter NUM_WARPS   = 4,
    parameter NUM_LANES   = 32,
    parameter DATA_WIDTH  = 32,
    parameter SM_ID       = 0
)(
    input  wire clk,
    input  wire rst_n,

    // --- Queued FU sources (WBQ outputs) ---
    // ALU
    input  wire                             alu_wbq_empty,
    input  wire [WARP_ID_W-1:0]             alu_wbq_warp,
    input  wire [4:0]                       alu_wbq_rd,
    input  wire [NUM_LANES-1:0]             alu_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            alu_wbq_data,
    // MUL
    input  wire                             mul_wbq_empty,
    input  wire [WARP_ID_W-1:0]             mul_wbq_warp,
    input  wire [4:0]                       mul_wbq_rd,
    input  wire [NUM_LANES-1:0]             mul_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            mul_wbq_data,
    // FPU32
    input  wire                             fpu32_wbq_empty,
    input  wire [WARP_ID_W-1:0]             fpu32_wbq_warp,
    input  wire [4:0]                       fpu32_wbq_rd,
    input  wire [NUM_LANES-1:0]             fpu32_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            fpu32_wbq_data,
    // FPU64
    input  wire                             fpu64_wbq_empty,
    input  wire [WARP_ID_W-1:0]             fpu64_wbq_warp,
    input  wire [4:0]                       fpu64_wbq_rd,
    input  wire [NUM_LANES-1:0]             fpu64_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            fpu64_wbq_data,
    // FP16
    input  wire                             fp16_wbq_empty,
    input  wire [WARP_ID_W-1:0]             fp16_wbq_warp,
    input  wire [4:0]                       fp16_wbq_rd,
    input  wire [NUM_LANES-1:0]             fp16_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            fp16_wbq_data,
    // SFU
    input  wire                             sfu_wbq_empty,
    input  wire [WARP_ID_W-1:0]             sfu_wbq_warp,
    input  wire [4:0]                       sfu_wbq_rd,
    input  wire [NUM_LANES-1:0]             sfu_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            sfu_wbq_data,
    // Tensor
    input  wire                             tensor_wbq_empty,
    input  wire [WARP_ID_W-1:0]             tensor_wbq_warp,
    input  wire [4:0]                       tensor_wbq_rd,
    input  wire [NUM_LANES-1:0]             tensor_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            tensor_wbq_data,
    // Shuffle
    input  wire                             shfl_wbq_empty,
    input  wire [WARP_ID_W-1:0]             shfl_wbq_warp,
    input  wire [4:0]                       shfl_wbq_rd,
    input  wire [NUM_LANES-1:0]             shfl_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            shfl_wbq_data,
    // Special registers
    input  wire                             special_wbq_empty,
    input  wire [WARP_ID_W-1:0]             special_wbq_warp,
    input  wire [4:0]                       special_wbq_rd,
    input  wire [NUM_LANES-1:0]             special_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            special_wbq_data,
    // Video
    input  wire                             video_wbq_empty,
    input  wire [WARP_ID_W-1:0]             video_wbq_warp,
    input  wire [4:0]                       video_wbq_rd,
    input  wire [NUM_LANES-1:0]             video_wbq_mask,
    input  wire [SIMD_WIDTH-1:0]            video_wbq_data,

    // --- Non-queued sources (directly latched) ---
    // Memory (case 7)
    input  wire                             gmem_resp_latched,
    input  wire [WARP_ID_W-1:0]             gmem_resp_warp,
    input  wire [4:0]                       gmem_resp_rd,
    input  wire [SIMD_WIDTH-1:0]            gmem_resp_data,
    input  wire [NUM_LANES-1:0]             gmem_resp_mask,
    input  wire                             smem_resp_latched,
    input  wire [WARP_ID_W-1:0]             smem_resp_warp,
    input  wire [4:0]                       smem_resp_rd,
    input  wire [SIMD_WIDTH-1:0]            smem_resp_data,
    input  wire [NUM_LANES-1:0]             smem_resp_mask,
    input  wire                             store_pending_valid,
    input  wire [WARP_ID_W-1:0]             store_warp_pending,
    input  wire [NUM_LANES-1:0]             store_mask_pending,
    // Atomic (case 9)
    input  wire                             atomic_valid_out,
    input  wire [WARP_ID_W-1:0]             atomic_warp_pending,
    input  wire [4:0]                       atomic_rd_pending,
    input  wire [SIMD_WIDTH-1:0]            atomic_result,
    input  wire [NUM_LANES-1:0]             atomic_mask_pending,
    // mbarrier (case 11)
    input  wire                             mbarrier_result_latched,
    input  wire [WARP_ID_W-1:0]             mbarrier_wb_warp,
    input  wire [4:0]                       mbarrier_wb_rd,
    input  wire [31:0]                      mbarrier_wb_result,
    input  wire [NUM_LANES-1:0]             mbarrier_wb_mask,
    // Texture (case 12)
    input  wire                             tex_result_valid_latched,
    input  wire [WARP_ID_W-1:0]             tex_warp_pending,
    input  wire [4:0]                       tex_rd_pending,
    input  wire [127:0]                     tex_result_latched,
    input  wire [NUM_LANES-1:0]             tex_mask_pending,
    // Cache policy (case 14)
    input  wire                             cache_policy_token_valid_r,
    input  wire [WARP_ID_W-1:0]             cache_policy_wb_warp,
    input  wire [4:0]                       cache_policy_wb_rd,
    input  wire [NUM_LANES-1:0]             cache_policy_wb_mask,
    input  wire [31:0]                      cache_policy_token_r,
    // Stack (case 15)
    input  wire                             stack_result_valid_r,
    input  wire [WARP_ID_W-1:0]             stack_wb_warp,
    input  wire [4:0]                       stack_wb_rd,
    input  wire [NUM_LANES-1:0]             stack_wb_mask,
    input  wire [31:0]                      stack_result_r,
    // Multimem (case 16)
    input  wire                             multimem_result_valid_r,
    input  wire [WARP_ID_W-1:0]             multimem_wb_warp,
    input  wire [4:0]                       multimem_wb_rd,
    input  wire [NUM_LANES-1:0]             multimem_wb_mask,
    input  wire [31:0]                      multimem_result_r,

    // --- Outputs ---
    output reg                              wb_valid,
    output reg  [WARP_ID_W-1:0]             wb_warp_id,
    output reg  [4:0]                       wb_rd,
    output reg  [SIMD_WIDTH-1:0]            wb_data,
    output reg  [NUM_LANES-1:0]             wb_mask,
    output wire                             wb_found,
    output wire [4:0]                       wb_sel,

    // WBQ pop signals (active when selected by arbiter)
    output wire                             alu_wbq_pop,
    output wire                             mul_wbq_pop,
    output wire                             fpu32_wbq_pop,
    output wire                             fpu64_wbq_pop,
    output wire                             fp16_wbq_pop,
    output wire                             sfu_wbq_pop,
    output wire                             tensor_wbq_pop,
    output wire                             shfl_wbq_pop,
    output wire                             special_wbq_pop,
    output wire                             video_wbq_pop,
    output wire                             tex_wbq_pop
);

    localparam WARP_ID_W  = $clog2(NUM_WARPS);
    localparam SIMD_WIDTH = NUM_LANES * DATA_WIDTH;

    // Round-robin priority state
    reg [3:0] wb_arb_priority;

    // Collect all ready FU outputs for round-robin arbitration
    wire [16:0] fu_ready;
    assign fu_ready[0]  = !alu_wbq_empty;
    assign fu_ready[1]  = !mul_wbq_empty;
    assign fu_ready[2]  = !fpu32_wbq_empty;
    assign fu_ready[3]  = !fpu64_wbq_empty;
    assign fu_ready[4]  = !fp16_wbq_empty;
    assign fu_ready[5]  = !sfu_wbq_empty;
    assign fu_ready[6]  = !tensor_wbq_empty;
    assign fu_ready[7]  = gmem_resp_latched || smem_resp_latched || store_pending_valid;
    assign fu_ready[8]  = !shfl_wbq_empty;
    assign fu_ready[9]  = atomic_valid_out;
    assign fu_ready[10] = !special_wbq_empty;
    assign fu_ready[11] = mbarrier_result_latched;
    assign fu_ready[12] = tex_result_valid_latched;
    assign fu_ready[13] = !video_wbq_empty;
    assign fu_ready[14] = cache_policy_token_valid_r;
    assign fu_ready[15] = stack_result_valid_r;
    assign fu_ready[16] = multimem_result_valid_r;

    // Round-robin selection
    reg [4:0] wb_sel_r;
    reg       wb_found_r;
    integer   wb_i;

    always @(*) begin
        wb_found_r = 1'b0;
        wb_sel_r = 0;
        for (wb_i = 0; wb_i < 17; wb_i = wb_i + 1) begin
            if (!wb_found_r && fu_ready[(wb_arb_priority + wb_i) % 17]) begin
                wb_sel_r = (wb_arb_priority + wb_i) % 17;
                wb_found_r = 1'b1;
            end
        end
    end

    assign wb_found = wb_found_r;
    assign wb_sel   = wb_sel_r;

    // WBQ pop signals
    assign alu_wbq_pop     = wb_found && (wb_sel == 4'd0);
    assign mul_wbq_pop     = wb_found && (wb_sel == 4'd1);
    assign fpu32_wbq_pop   = wb_found && (wb_sel == 4'd2);
    assign fpu64_wbq_pop   = wb_found && (wb_sel == 4'd3);
    assign fp16_wbq_pop    = wb_found && (wb_sel == 4'd4);
    assign sfu_wbq_pop     = wb_found && (wb_sel == 4'd5);
    assign tensor_wbq_pop  = wb_found && (wb_sel == 4'd6);
    assign shfl_wbq_pop    = wb_found && (wb_sel == 4'd8);
    assign special_wbq_pop = wb_found && (wb_sel == 4'd10);
    assign tex_wbq_pop     = wb_found && (wb_sel == 4'd12);
    assign video_wbq_pop   = wb_found && (wb_sel == 4'd13);

    // Writeback arbiter FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_valid <= 1'b0;
            wb_arb_priority <= 0;
        end else begin
            if (wb_found) begin
                wb_valid <= 1'b1;
                wb_arb_priority <= (wb_sel + 1) % 17;

                case (wb_sel)
                    4'd0: begin  // ALU
                        wb_warp_id <= alu_wbq_warp;
                        wb_rd      <= alu_wbq_rd;
                        wb_data    <= alu_wbq_data;
                        wb_mask    <= alu_wbq_mask;
                    end
                    4'd1: begin  // MUL
                        wb_warp_id <= mul_wbq_warp;
                        wb_rd      <= mul_wbq_rd;
                        wb_data    <= mul_wbq_data;
                        wb_mask    <= mul_wbq_mask;
                    end
                    4'd2: begin  // FPU32
                        wb_warp_id <= fpu32_wbq_warp;
                        wb_rd      <= fpu32_wbq_rd;
                        wb_data    <= fpu32_wbq_data;
                        wb_mask    <= fpu32_wbq_mask;
                    end
                    4'd3: begin  // FPU64
                        wb_warp_id <= fpu64_wbq_warp;
                        wb_rd      <= fpu64_wbq_rd;
                        wb_data    <= fpu64_wbq_data;
                        wb_mask    <= fpu64_wbq_mask;
                    end
                    4'd4: begin  // FP16
                        wb_warp_id <= fp16_wbq_warp;
                        wb_rd      <= fp16_wbq_rd;
                        wb_data    <= fp16_wbq_data;
                        wb_mask    <= fp16_wbq_mask;
                    end
                    4'd5: begin  // SFU
                        wb_warp_id <= sfu_wbq_warp;
                        wb_rd      <= sfu_wbq_rd;
                        wb_data    <= sfu_wbq_data;
                        wb_mask    <= sfu_wbq_mask;
                    end
                    4'd6: begin  // Tensor
                        wb_warp_id <= tensor_wbq_warp;
                        wb_rd      <= tensor_wbq_rd;
                        wb_data    <= tensor_wbq_data;
                        wb_mask    <= tensor_wbq_mask;
                    end
                    4'd7: begin  // Memory
                        if (smem_resp_latched) begin
                            wb_warp_id <= smem_resp_warp;
                            wb_rd      <= smem_resp_rd;
                            wb_data    <= smem_resp_data;
                            wb_mask    <= smem_resp_mask;
                        end else if (gmem_resp_latched) begin
                            wb_warp_id <= gmem_resp_warp;
                            wb_rd      <= gmem_resp_rd;
                            wb_data    <= gmem_resp_data;
                            wb_mask    <= gmem_resp_mask;
                        end else begin
                            // Store completion (no register writeback)
                            wb_warp_id <= store_warp_pending;
                            wb_rd      <= 0;
                            wb_data    <= 0;
                            wb_mask    <= store_mask_pending;
                        end
                    end
                    4'd8: begin  // Shuffle
                        wb_warp_id <= shfl_wbq_warp;
                        wb_rd      <= shfl_wbq_rd;
                        wb_data    <= shfl_wbq_data;
                        wb_mask    <= shfl_wbq_mask;
                    end
                    4'd9: begin  // Atomic
                        wb_warp_id <= atomic_warp_pending;
                        wb_rd      <= atomic_rd_pending;
                        wb_data    <= atomic_result;
                        wb_mask    <= atomic_mask_pending;
                    end
                    4'd10: begin  // Special registers
                        wb_warp_id <= special_wbq_warp;
                        wb_rd      <= special_wbq_rd;
                        wb_data    <= special_wbq_data;
                        wb_mask    <= special_wbq_mask;
                    end
                    4'd11: begin  // mbarrier
                        wb_warp_id <= mbarrier_wb_warp;
                        wb_rd      <= mbarrier_wb_rd;
                        wb_data    <= {NUM_LANES{mbarrier_wb_result}};
                        wb_mask    <= mbarrier_wb_mask;
                    end
                    4'd12: begin  // Texture
                        wb_warp_id <= tex_warp_pending;
                        wb_rd      <= tex_rd_pending;
                        wb_data    <= {(NUM_LANES/4){tex_result_latched}};
                        wb_mask    <= tex_mask_pending;
                    end
                    4'd13: begin  // Video SIMD
                        wb_warp_id <= video_wbq_warp;
                        wb_rd      <= video_wbq_rd;
                        wb_data    <= video_wbq_data;
                        wb_mask    <= video_wbq_mask;
                    end
                    4'd14: begin  // Cache policy
                        wb_warp_id <= cache_policy_wb_warp;
                        wb_rd      <= cache_policy_wb_rd;
                        wb_data    <= {NUM_LANES{cache_policy_token_r}};
                        wb_mask    <= cache_policy_wb_mask;
                    end
                    4'd15: begin  // Stack
                        wb_warp_id <= stack_wb_warp;
                        wb_rd      <= stack_wb_rd;
                        wb_data    <= {NUM_LANES{stack_result_r}};
                        wb_mask    <= stack_wb_mask;
                    end
                    5'd16: begin  // Multimem
                        wb_warp_id <= multimem_wb_warp;
                        wb_rd      <= multimem_wb_rd;
                        wb_data    <= {NUM_LANES{multimem_result_r}};
                        wb_mask    <= multimem_wb_mask;
                    end
                    default: ; // no-op
                endcase
            end else begin
                wb_valid <= 1'b0;
            end
        end
    end

endmodule
