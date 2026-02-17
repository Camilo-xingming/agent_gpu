//============================================================================
// RalphGPU - Writeback Queue Bank
// 9 independent wb_fifo instances (one per FU) with pack/unpack logic.
// Extracted from streaming_multiprocessor_v2.v (RALPH-6 God Module Refactor)
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module sm_wbq_bank #(
    parameter NUM_LANES  = `THREADS_PER_WARP,
    parameter SIMD_WIDTH = NUM_LANES * 32,
    parameter WARP_ID_W  = $clog2(`WARPS_PER_SM),
    // Per-FU FIFO depths
    parameter ALU_WBQ_DEPTH   = 4,
    parameter MUL_WBQ_DEPTH   = 4,
    parameter FPU32_WBQ_DEPTH = 8,
    parameter FPU64_WBQ_DEPTH = 8,
    parameter FP16_WBQ_DEPTH  = 4,
    parameter SFU_WBQ_DEPTH   = 16,
    parameter SHFL_WBQ_DEPTH  = 4,
    parameter VIDEO_WBQ_DEPTH = 4
)(
    input  wire             clk,
    input  wire             rst_n,

    //------------------------------------------------------------------------
    // FU pipeline outputs (push side)
    //------------------------------------------------------------------------
    // ALU (1-cycle)
    input  wire [WARP_ID_W-1:0]  alu_warp,
    input  wire [4:0]            alu_rd,
    input  wire [NUM_LANES-1:0]  alu_mask,
    input  wire [SIMD_WIDTH-1:0] alu_data,
    input  wire                  alu_valid,
    // MUL (2-cycle)
    input  wire [WARP_ID_W-1:0]  mul_warp,
    input  wire [4:0]            mul_rd,
    input  wire [NUM_LANES-1:0]  mul_mask,
    input  wire [SIMD_WIDTH-1:0] mul_data,
    input  wire                  mul_valid,
    // FPU32 (1-cycle)
    input  wire [WARP_ID_W-1:0]  fpu32_warp,
    input  wire [4:0]            fpu32_rd,
    input  wire [NUM_LANES-1:0]  fpu32_mask,
    input  wire [SIMD_WIDTH-1:0] fpu32_data,
    input  wire                  fpu32_valid,
    // FPU64 (5-cycle)
    input  wire [WARP_ID_W-1:0]  fpu64_warp,
    input  wire [4:0]            fpu64_rd,
    input  wire [NUM_LANES-1:0]  fpu64_mask,
    input  wire [SIMD_WIDTH-1:0] fpu64_data,
    input  wire                  fpu64_valid,
    // FP16 (3-cycle)
    input  wire [WARP_ID_W-1:0]  fp16_warp,
    input  wire [4:0]            fp16_rd,
    input  wire [NUM_LANES-1:0]  fp16_mask,
    input  wire [SIMD_WIDTH-1:0] fp16_data,
    input  wire                  fp16_valid,
    // SFU (8-cycle)
    input  wire [WARP_ID_W-1:0]  sfu_warp,
    input  wire [4:0]            sfu_rd,
    input  wire [NUM_LANES-1:0]  sfu_mask,
    input  wire [SIMD_WIDTH-1:0] sfu_data,
    input  wire                  sfu_valid,
    // Shuffle (1-cycle)
    input  wire [WARP_ID_W-1:0]  shfl_warp,
    input  wire [4:0]            shfl_rd,
    input  wire [NUM_LANES-1:0]  shfl_mask,
    input  wire [SIMD_WIDTH-1:0] shfl_data,
    input  wire                  shfl_valid,
    // Video (2-cycle)
    input  wire [WARP_ID_W-1:0]  video_warp,
    input  wire [4:0]            video_rd,
    input  wire [NUM_LANES-1:0]  video_mask,
    input  wire [SIMD_WIDTH-1:0] video_data,
    input  wire                  video_valid,
    // Special register (1-cycle)
    input  wire [WARP_ID_W-1:0]  special_warp,
    input  wire [4:0]            special_rd,
    input  wire [NUM_LANES-1:0]  special_mask,
    input  wire [SIMD_WIDTH-1:0] special_data,
    input  wire                  special_valid,

    //------------------------------------------------------------------------
    // Arbiter interface (pop side)
    //------------------------------------------------------------------------
    input  wire                  alu_pop,
    input  wire                  mul_pop,
    input  wire                  fpu32_pop,
    input  wire                  fpu64_pop,
    input  wire                  fp16_pop,
    input  wire                  sfu_pop,
    input  wire                  shfl_pop,
    input  wire                  video_pop,
    input  wire                  special_pop,

    //------------------------------------------------------------------------
    // Per-FU unpacked outputs (to arbiter + stall logic)
    //------------------------------------------------------------------------
    output wire [WARP_ID_W-1:0]  alu_wbq_warp,
    output wire [4:0]            alu_wbq_rd,
    output wire [NUM_LANES-1:0]  alu_wbq_mask,
    output wire [SIMD_WIDTH-1:0] alu_wbq_data,
    output wire                  alu_wbq_full,
    output wire                  alu_wbq_empty,
    output wire                  alu_wbq_dropped,

    output wire [WARP_ID_W-1:0]  mul_wbq_warp,
    output wire [4:0]            mul_wbq_rd,
    output wire [NUM_LANES-1:0]  mul_wbq_mask,
    output wire [SIMD_WIDTH-1:0] mul_wbq_data,
    output wire                  mul_wbq_full,
    output wire                  mul_wbq_empty,
    output wire                  mul_wbq_dropped,

    output wire [WARP_ID_W-1:0]  fpu32_wbq_warp,
    output wire [4:0]            fpu32_wbq_rd,
    output wire [NUM_LANES-1:0]  fpu32_wbq_mask,
    output wire [SIMD_WIDTH-1:0] fpu32_wbq_data,
    output wire                  fpu32_wbq_full,
    output wire                  fpu32_wbq_empty,
    output wire                  fpu32_wbq_dropped,

    output wire [WARP_ID_W-1:0]  fpu64_wbq_warp,
    output wire [4:0]            fpu64_wbq_rd,
    output wire [NUM_LANES-1:0]  fpu64_wbq_mask,
    output wire [SIMD_WIDTH-1:0] fpu64_wbq_data,
    output wire                  fpu64_wbq_full,
    output wire                  fpu64_wbq_empty,
    output wire                  fpu64_wbq_dropped,

    output wire [WARP_ID_W-1:0]  fp16_wbq_warp,
    output wire [4:0]            fp16_wbq_rd,
    output wire [NUM_LANES-1:0]  fp16_wbq_mask,
    output wire [SIMD_WIDTH-1:0] fp16_wbq_data,
    output wire                  fp16_wbq_full,
    output wire                  fp16_wbq_empty,
    output wire                  fp16_wbq_dropped,

    output wire [WARP_ID_W-1:0]  sfu_wbq_warp,
    output wire [4:0]            sfu_wbq_rd,
    output wire [NUM_LANES-1:0]  sfu_wbq_mask,
    output wire [SIMD_WIDTH-1:0] sfu_wbq_data,
    output wire                  sfu_wbq_full,
    output wire                  sfu_wbq_empty,
    output wire                  sfu_wbq_dropped,

    output wire [WARP_ID_W-1:0]  shfl_wbq_warp,
    output wire [4:0]            shfl_wbq_rd,
    output wire [NUM_LANES-1:0]  shfl_wbq_mask,
    output wire [SIMD_WIDTH-1:0] shfl_wbq_data,
    output wire                  shfl_wbq_full,
    output wire                  shfl_wbq_empty,
    output wire                  shfl_wbq_dropped,

    output wire [WARP_ID_W-1:0]  video_wbq_warp,
    output wire [4:0]            video_wbq_rd,
    output wire [NUM_LANES-1:0]  video_wbq_mask,
    output wire [SIMD_WIDTH-1:0] video_wbq_data,
    output wire                  video_wbq_full,
    output wire                  video_wbq_empty,
    output wire                  video_wbq_dropped,

    output wire [WARP_ID_W-1:0]  special_wbq_warp,
    output wire [4:0]            special_wbq_rd,
    output wire [NUM_LANES-1:0]  special_wbq_mask,
    output wire [SIMD_WIDTH-1:0] special_wbq_data,
    output wire                  special_wbq_full,
    output wire                  special_wbq_empty,
    output wire                  special_wbq_dropped
);

    //------------------------------------------------------------------------
    // Packet format (same as SM v2 pack_wb)
    //------------------------------------------------------------------------
    localparam WB_DATA_LSB = 0;
    localparam WB_DATA_MSB = SIMD_WIDTH - 1;
    localparam WB_MASK_LSB = WB_DATA_MSB + 1;
    localparam WB_MASK_MSB = WB_MASK_LSB + NUM_LANES - 1;
    localparam WB_RD_LSB   = WB_MASK_MSB + 1;
    localparam WB_RD_MSB   = WB_RD_LSB + 5 - 1;
    localparam WB_WARP_LSB = WB_RD_MSB + 1;
    localparam WB_WARP_MSB = WB_WARP_LSB + WARP_ID_W - 1;
    localparam WB_PKT_W    = WB_WARP_MSB + 1;

    //------------------------------------------------------------------------
    // Pack helper
    //------------------------------------------------------------------------
    function [WB_PKT_W-1:0] pack_wb;
        input [WARP_ID_W-1:0]  warp_id;
        input [4:0]            rd;
        input [NUM_LANES-1:0]  mask;
        input [SIMD_WIDTH-1:0] data;
        begin
            pack_wb = {warp_id, rd, mask, data};
        end
    endfunction

    //------------------------------------------------------------------------
    // Pack FU outputs into WB packets
    //------------------------------------------------------------------------
    wire [WB_PKT_W-1:0] alu_pkt     = pack_wb(alu_warp,     alu_rd,     alu_mask,     alu_data);
    wire [WB_PKT_W-1:0] mul_pkt     = pack_wb(mul_warp,     mul_rd,     mul_mask,     mul_data);
    wire [WB_PKT_W-1:0] fpu32_pkt   = pack_wb(fpu32_warp,   fpu32_rd,   fpu32_mask,   fpu32_data);
    wire [WB_PKT_W-1:0] fpu64_pkt   = pack_wb(fpu64_warp,   fpu64_rd,   fpu64_mask,   fpu64_data);
    wire [WB_PKT_W-1:0] fp16_pkt    = pack_wb(fp16_warp,    fp16_rd,    fp16_mask,    fp16_data);
    wire [WB_PKT_W-1:0] sfu_pkt     = pack_wb(sfu_warp,     sfu_rd,     sfu_mask,     sfu_data);
    wire [WB_PKT_W-1:0] shfl_pkt    = pack_wb(shfl_warp,    shfl_rd,    shfl_mask,    shfl_data);
    wire [WB_PKT_W-1:0] video_pkt   = pack_wb(video_warp,   video_rd,   video_mask,   video_data);
    wire [WB_PKT_W-1:0] special_pkt = pack_wb(special_warp, special_rd, special_mask, special_data);

    //------------------------------------------------------------------------
    // FIFO output wires
    //------------------------------------------------------------------------
    wire [WB_PKT_W-1:0] alu_out, mul_out, fpu32_out, fpu64_out;
    wire [WB_PKT_W-1:0] fp16_out, sfu_out, shfl_out, video_out, special_out;

    //------------------------------------------------------------------------
    // 9x wb_fifo instances
    //------------------------------------------------------------------------
    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(ALU_WBQ_DEPTH)) u_alu_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(alu_valid), .push_data(alu_pkt),
        .pop(alu_pop), .pop_data(alu_out),
        .full(alu_wbq_full), .empty(alu_wbq_empty), .dropped(alu_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(MUL_WBQ_DEPTH)) u_mul_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(mul_valid), .push_data(mul_pkt),
        .pop(mul_pop), .pop_data(mul_out),
        .full(mul_wbq_full), .empty(mul_wbq_empty), .dropped(mul_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(FPU32_WBQ_DEPTH)) u_fpu32_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(fpu32_valid), .push_data(fpu32_pkt),
        .pop(fpu32_pop), .pop_data(fpu32_out),
        .full(fpu32_wbq_full), .empty(fpu32_wbq_empty), .dropped(fpu32_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(FPU64_WBQ_DEPTH)) u_fpu64_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(fpu64_valid), .push_data(fpu64_pkt),
        .pop(fpu64_pop), .pop_data(fpu64_out),
        .full(fpu64_wbq_full), .empty(fpu64_wbq_empty), .dropped(fpu64_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(FP16_WBQ_DEPTH)) u_fp16_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(fp16_valid), .push_data(fp16_pkt),
        .pop(fp16_pop), .pop_data(fp16_out),
        .full(fp16_wbq_full), .empty(fp16_wbq_empty), .dropped(fp16_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(SFU_WBQ_DEPTH)) u_sfu_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(sfu_valid), .push_data(sfu_pkt),
        .pop(sfu_pop), .pop_data(sfu_out),
        .full(sfu_wbq_full), .empty(sfu_wbq_empty), .dropped(sfu_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(SHFL_WBQ_DEPTH)) u_shfl_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(shfl_valid), .push_data(shfl_pkt),
        .pop(shfl_pop), .pop_data(shfl_out),
        .full(shfl_wbq_full), .empty(shfl_wbq_empty), .dropped(shfl_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(VIDEO_WBQ_DEPTH)) u_video_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(video_valid), .push_data(video_pkt),
        .pop(video_pop), .pop_data(video_out),
        .full(video_wbq_full), .empty(video_wbq_empty), .dropped(video_wbq_dropped)
    );

    wb_fifo #(.WIDTH(WB_PKT_W), .DEPTH(ALU_WBQ_DEPTH)) u_special_wbq (
        .clk(clk), .rst_n(rst_n),
        .push(special_valid), .push_data(special_pkt),
        .pop(special_pop), .pop_data(special_out),
        .full(special_wbq_full), .empty(special_wbq_empty), .dropped(special_wbq_dropped)
    );

    //------------------------------------------------------------------------
    // Unpack FIFO outputs
    //------------------------------------------------------------------------
    assign alu_wbq_warp     = alu_out[WB_WARP_MSB:WB_WARP_LSB];
    assign alu_wbq_rd       = alu_out[WB_RD_MSB:WB_RD_LSB];
    assign alu_wbq_mask     = alu_out[WB_MASK_MSB:WB_MASK_LSB];
    assign alu_wbq_data     = alu_out[WB_DATA_MSB:WB_DATA_LSB];

    assign mul_wbq_warp     = mul_out[WB_WARP_MSB:WB_WARP_LSB];
    assign mul_wbq_rd       = mul_out[WB_RD_MSB:WB_RD_LSB];
    assign mul_wbq_mask     = mul_out[WB_MASK_MSB:WB_MASK_LSB];
    assign mul_wbq_data     = mul_out[WB_DATA_MSB:WB_DATA_LSB];

    assign fpu32_wbq_warp   = fpu32_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu32_wbq_rd     = fpu32_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu32_wbq_mask   = fpu32_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu32_wbq_data   = fpu32_out[WB_DATA_MSB:WB_DATA_LSB];

    assign fpu64_wbq_warp   = fpu64_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu64_wbq_rd     = fpu64_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu64_wbq_mask   = fpu64_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu64_wbq_data   = fpu64_out[WB_DATA_MSB:WB_DATA_LSB];

    assign fp16_wbq_warp    = fp16_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fp16_wbq_rd      = fp16_out[WB_RD_MSB:WB_RD_LSB];
    assign fp16_wbq_mask    = fp16_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fp16_wbq_data    = fp16_out[WB_DATA_MSB:WB_DATA_LSB];

    assign sfu_wbq_warp     = sfu_out[WB_WARP_MSB:WB_WARP_LSB];
    assign sfu_wbq_rd       = sfu_out[WB_RD_MSB:WB_RD_LSB];
    assign sfu_wbq_mask     = sfu_out[WB_MASK_MSB:WB_MASK_LSB];
    assign sfu_wbq_data     = sfu_out[WB_DATA_MSB:WB_DATA_LSB];

    assign shfl_wbq_warp    = shfl_out[WB_WARP_MSB:WB_WARP_LSB];
    assign shfl_wbq_rd      = shfl_out[WB_RD_MSB:WB_RD_LSB];
    assign shfl_wbq_mask    = shfl_out[WB_MASK_MSB:WB_MASK_LSB];
    assign shfl_wbq_data    = shfl_out[WB_DATA_MSB:WB_DATA_LSB];

    assign video_wbq_warp   = video_out[WB_WARP_MSB:WB_WARP_LSB];
    assign video_wbq_rd     = video_out[WB_RD_MSB:WB_RD_LSB];
    assign video_wbq_mask   = video_out[WB_MASK_MSB:WB_MASK_LSB];
    assign video_wbq_data   = video_out[WB_DATA_MSB:WB_DATA_LSB];

    assign special_wbq_warp = special_out[WB_WARP_MSB:WB_WARP_LSB];
    assign special_wbq_rd   = special_out[WB_RD_MSB:WB_RD_LSB];
    assign special_wbq_mask = special_out[WB_MASK_MSB:WB_MASK_LSB];
    assign special_wbq_data = special_out[WB_DATA_MSB:WB_DATA_LSB];

endmodule
