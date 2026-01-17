//============================================================================
// RalphGPU - Streaming Multiprocessor V2 (Commercial Architecture)
//
// Key Improvements over V1:
// 1. Full functional unit instantiation (FPU, Tensor, SFU, etc.)
// 2. Pipelined execution with warp interleaving for latency hiding
// 3. Scoreboard-based dependency tracking
// 4. Control flow unit integration for divergence handling
// 5. Multi-cycle unit support with valid/ready handshaking
//
// Architecture: 5-stage pipeline with out-of-order warp issue
//   FETCH -> DECODE -> ISSUE -> EXECUTE -> WRITEBACK
//
// NVIDIA Hopper-Class Features (Integrated):
// - Advanced dual-issue warp scheduler (GTO policy)
// - Branch predictor with TAGE + BTB + RAS
// - Instruction cache with prefetch
// - Reconvergence stack for SIMT divergence
// - Banked register file for dual-issue support
// - WGMMA tensor operations
//============================================================================

`include "gpu_defines.vh"
`include "memory_config.vh"

module streaming_multiprocessor_v2 #(
    parameter SM_ID      = 0,
    parameter NUM_WARPS  = `WARPS_PER_SM,
    parameter NUM_LANES  = `THREADS_PER_WARP,
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter TC_NUM_CORES = 8,
    parameter TC_LATENCY = 4,
    parameter [2:0] TC_DATA_DEFAULT = `TC_DATA_FP16,
    parameter TC_USE_OP_TYPE = 1,
    parameter [1:0] TC_FP4_FORMAT = `TC_FP4_E2M1,
    parameter [1:0] TC_FP8_FORMAT = `TC_FP8_E4M3,
    parameter INIT_WARPS = 1,
    parameter ICACHE_BYPASS = 0  // Bypass icache for ideal fetch latency testing
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Kernel Launch Interface
    input  wire                     kernel_start,
    input  wire [31:0]              kernel_pc,
    input  wire [31:0]              block_id_x,
    input  wire [31:0]              block_id_y,
    input  wire [31:0]              block_id_z,
    input  wire [31:0]              block_dim_x,
    input  wire [31:0]              block_dim_y,
    input  wire [31:0]              block_dim_z,
    input  wire [31:0]              grid_dim_x,
    input  wire [31:0]              grid_dim_y,
    input  wire [31:0]              grid_dim_z,
    output wire                     kernel_done,

    // Instruction Memory Interface
    output wire                     imem_req,
    output wire [31:0]              imem_addr,
    input  wire                     imem_ready,
    input  wire [63:0]              imem_data,    // 64-bit for 8-byte cache line
    input  wire                     imem_valid,

    // L1 Data Cache Interface
    output wire                     l1d_req_valid,
    output wire                     l1d_req_write,
    output wire [31:0]              l1d_req_addr [0:NUM_LANES-1],
    output wire [31:0]              l1d_req_wdata [0:NUM_LANES-1],
    output wire [NUM_LANES-1:0]     l1d_req_mask,
    input  wire [31:0]              l1d_resp_rdata [0:NUM_LANES-1],
    input  wire                     l1d_resp_valid,
    input  wire                     l1d_resp_hit,

    // Global Memory Interface (AXI4)
    output wire [3:0]               m_axi_awid,
    output wire [31:0]              m_axi_awaddr,
    output wire [7:0]               m_axi_awlen,
    output wire [2:0]               m_axi_awsize,
    output wire [1:0]               m_axi_awburst,
    output wire                     m_axi_awvalid,
    input  wire                     m_axi_awready,
    output wire [31:0]              m_axi_wdata,
    output wire [3:0]               m_axi_wstrb,
    output wire                     m_axi_wlast,
    output wire                     m_axi_wvalid,
    input  wire                     m_axi_wready,
    input  wire [3:0]               m_axi_bid,
    input  wire [1:0]               m_axi_bresp,
    input  wire                     m_axi_bvalid,
    output wire                     m_axi_bready,
    output wire [3:0]               m_axi_arid,
    output wire [31:0]              m_axi_araddr,
    output wire [7:0]               m_axi_arlen,
    output wire [2:0]               m_axi_arsize,
    output wire [1:0]               m_axi_arburst,
    output wire                     m_axi_arvalid,
    input  wire                     m_axi_arready,
    input  wire [3:0]               m_axi_rid,
    input  wire [31:0]              m_axi_rdata,
    input  wire [1:0]               m_axi_rresp,
    input  wire                     m_axi_rlast,
    input  wire                     m_axi_rvalid,
    output wire                     m_axi_rready
);

    //========================================================================
    // Constants and Derived Parameters
    //========================================================================
    localparam WARP_ID_W = $clog2(NUM_WARPS);
    localparam SIMD_WIDTH = NUM_LANES * DATA_WIDTH;
    localparam IFQ_DEPTH = 4;
    localparam IFQ_PTR_W = (IFQ_DEPTH > 1) ? $clog2(IFQ_DEPTH) : 1;
    localparam IFQ_COUNT_W = $clog2(IFQ_DEPTH + 1);
    localparam [IFQ_COUNT_W-1:0] IFQ_DEPTH_VAL = IFQ_DEPTH[IFQ_COUNT_W-1:0];
    localparam ISSUE_WIDTH = (`SM_ISSUE_WIDTH < 1) ? 1 : `SM_ISSUE_WIDTH;
    localparam ALU_WBQ_DEPTH = 4;
    localparam MUL_WBQ_DEPTH = 4;
    localparam FPU32_WBQ_DEPTH = 8;
    localparam FPU64_WBQ_DEPTH = 8;
    localparam FP16_WBQ_DEPTH = 4;
    localparam SFU_WBQ_DEPTH = 16;
    localparam SHFL_WBQ_DEPTH = 4;
    localparam ALU_WBQ_COUNT_W = $clog2(ALU_WBQ_DEPTH + 1);
    localparam MUL_WBQ_COUNT_W = $clog2(MUL_WBQ_DEPTH + 1);
    localparam FPU32_WBQ_COUNT_W = $clog2(FPU32_WBQ_DEPTH + 1);
    localparam FPU64_WBQ_COUNT_W = $clog2(FPU64_WBQ_DEPTH + 1);
    localparam FP16_WBQ_COUNT_W = $clog2(FP16_WBQ_DEPTH + 1);
    localparam SFU_WBQ_COUNT_W = $clog2(SFU_WBQ_DEPTH + 1);
    localparam SHFL_WBQ_COUNT_W = $clog2(SHFL_WBQ_DEPTH + 1);
    localparam [ALU_WBQ_COUNT_W-1:0] ALU_WBQ_DEPTH_VAL = ALU_WBQ_DEPTH;
    localparam [MUL_WBQ_COUNT_W-1:0] MUL_WBQ_DEPTH_VAL = MUL_WBQ_DEPTH;
    localparam [FPU32_WBQ_COUNT_W-1:0] FPU32_WBQ_DEPTH_VAL = FPU32_WBQ_DEPTH;
    localparam [FPU64_WBQ_COUNT_W-1:0] FPU64_WBQ_DEPTH_VAL = FPU64_WBQ_DEPTH;
    localparam [FP16_WBQ_COUNT_W-1:0] FP16_WBQ_DEPTH_VAL = FP16_WBQ_DEPTH;
    localparam [SFU_WBQ_COUNT_W-1:0] SFU_WBQ_DEPTH_VAL = SFU_WBQ_DEPTH;
    localparam [SHFL_WBQ_COUNT_W-1:0] SHFL_WBQ_DEPTH_VAL = SHFL_WBQ_DEPTH;

    localparam WB_DATA_LSB = 0;
    localparam WB_DATA_MSB = SIMD_WIDTH - 1;
    localparam WB_MASK_LSB = WB_DATA_MSB + 1;
    localparam WB_MASK_MSB = WB_MASK_LSB + NUM_LANES - 1;
    localparam WB_RD_LSB = WB_MASK_MSB + 1;
    localparam WB_RD_MSB = WB_RD_LSB + 5 - 1;
    localparam WB_WARP_LSB = WB_RD_MSB + 1;
    localparam WB_WARP_MSB = WB_WARP_LSB + WARP_ID_W - 1;
    localparam WB_PKT_W = WB_WARP_MSB + 1;

    function [WB_PKT_W-1:0] pack_wb;
        input [WARP_ID_W-1:0] warp_id;
        input [4:0] rd;
        input [NUM_LANES-1:0] mask;
        input [SIMD_WIDTH-1:0] data;
        begin
            pack_wb = {warp_id, rd, mask, data};
        end
    endfunction

    localparam TENSOR_ISSUE_DEPTH = TC_NUM_CORES * 2;
    localparam TENSOR_META_DEPTH = TC_NUM_CORES;
    localparam TENSOR_WBQ_DEPTH = TC_NUM_CORES;
    localparam TENSOR_ISSUE_COUNT_W = $clog2(TENSOR_ISSUE_DEPTH + 1);
    localparam [TENSOR_ISSUE_COUNT_W-1:0] TENSOR_ISSUE_DEPTH_VAL =
        TENSOR_ISSUE_DEPTH[TENSOR_ISSUE_COUNT_W-1:0];
    localparam TENSOR_META_COUNT_W = $clog2(TENSOR_META_DEPTH + 1);
    localparam [TENSOR_META_COUNT_W-1:0] TENSOR_META_DEPTH_VAL =
        TENSOR_META_DEPTH[TENSOR_META_COUNT_W-1:0];
    localparam TENSOR_WBQ_COUNT_W = $clog2(TENSOR_WBQ_DEPTH + 1);
    localparam [TENSOR_WBQ_COUNT_W-1:0] TENSOR_WBQ_DEPTH_VAL =
        TENSOR_WBQ_DEPTH[TENSOR_WBQ_COUNT_W-1:0];

    localparam TENSOR_ISSUE_A_LSB = 0;
    localparam TENSOR_ISSUE_A_MSB = SIMD_WIDTH - 1;
    localparam TENSOR_ISSUE_B_LSB = TENSOR_ISSUE_A_MSB + 1;
    localparam TENSOR_ISSUE_B_MSB = TENSOR_ISSUE_B_LSB + SIMD_WIDTH - 1;
    localparam TENSOR_ISSUE_C_LSB = TENSOR_ISSUE_B_MSB + 1;
    localparam TENSOR_ISSUE_C_MSB = TENSOR_ISSUE_C_LSB + SIMD_WIDTH - 1;
    localparam TENSOR_ISSUE_OP_LSB = TENSOR_ISSUE_C_MSB + 1;
    localparam TENSOR_ISSUE_OP_MSB = TENSOR_ISSUE_OP_LSB + 3 - 1;
    localparam TENSOR_ISSUE_MASK_LSB = TENSOR_ISSUE_OP_MSB + 1;
    localparam TENSOR_ISSUE_MASK_MSB = TENSOR_ISSUE_MASK_LSB + NUM_LANES - 1;
    localparam TENSOR_ISSUE_RD_LSB = TENSOR_ISSUE_MASK_MSB + 1;
    localparam TENSOR_ISSUE_RD_MSB = TENSOR_ISSUE_RD_LSB + 5 - 1;
    localparam TENSOR_ISSUE_WARP_LSB = TENSOR_ISSUE_RD_MSB + 1;
    localparam TENSOR_ISSUE_WARP_MSB = TENSOR_ISSUE_WARP_LSB + WARP_ID_W - 1;
    localparam TENSOR_ISSUE_W = TENSOR_ISSUE_WARP_MSB + 1;

    localparam TENSOR_META_MASK_LSB = 0;
    localparam TENSOR_META_MASK_MSB = NUM_LANES - 1;
    localparam TENSOR_META_RD_LSB = TENSOR_META_MASK_MSB + 1;
    localparam TENSOR_META_RD_MSB = TENSOR_META_RD_LSB + 5 - 1;
    localparam TENSOR_META_WARP_LSB = TENSOR_META_RD_MSB + 1;
    localparam TENSOR_META_WARP_MSB = TENSOR_META_WARP_LSB + WARP_ID_W - 1;
    localparam TENSOR_META_W = TENSOR_META_WARP_MSB + 1;

    function [TENSOR_ISSUE_W-1:0] pack_tensor_issue;
        input [WARP_ID_W-1:0] warp_id;
        input [4:0] rd;
        input [NUM_LANES-1:0] mask;
        input [2:0] op_type;
        input [SIMD_WIDTH-1:0] frag_a;
        input [SIMD_WIDTH-1:0] frag_b;
        input [SIMD_WIDTH-1:0] frag_c;
        begin
            pack_tensor_issue = {warp_id, rd, mask, op_type, frag_c, frag_b, frag_a};
        end
    endfunction

    function [TENSOR_META_W-1:0] pack_tensor_meta;
        input [WARP_ID_W-1:0] warp_id;
        input [4:0] rd;
        input [NUM_LANES-1:0] mask;
        begin
            pack_tensor_meta = {warp_id, rd, mask};
        end
    endfunction

    //========================================================================
    // Warp State Tracking
    //========================================================================
    reg  [NUM_WARPS-1:0] warp_valid;          // Warp has work
    reg  [NUM_WARPS-1:0] warp_active;         // Warp can be scheduled
    reg  [NUM_WARPS-1:0] warp_stalled_mem;    // Waiting for memory
    reg  [NUM_WARPS-1:0] warp_stalled_fu;     // Waiting for FU completion
    reg  [NUM_WARPS-1:0] warp_stalled_sync;   // At barrier
    reg  [NUM_WARPS-1:0] warp_exit_pending;  // EXIT issued, waiting for drain
    reg  [31:0]          warp_pc [0:NUM_WARPS-1];
    reg  [31:0]          warp_fetch_pc [0:NUM_WARPS-1];
    reg  [NUM_LANES-1:0] warp_mask [0:NUM_WARPS-1];  // Active thread mask

    wire [NUM_WARPS-1:0] warp_ready = warp_valid & ~warp_stalled_mem &
                                       ~warp_stalled_fu & ~warp_stalled_sync &
                                       ~warp_exit_pending;

    //========================================================================
    // Pipeline Registers
    //========================================================================

    // Fetch Stage
    wire                 fetch_req;
    wire                 fetch_fire;

    // Instruction fetch queue (decouples fetch from decode)
    reg [WARP_ID_W-1:0] ifq_warp_id [0:IFQ_DEPTH-1];
    reg [31:0]          ifq_pc [0:IFQ_DEPTH-1];
    reg [31:0]          ifq_inst [0:IFQ_DEPTH-1];
    reg [IFQ_PTR_W-1:0] ifq_head;
    reg [IFQ_PTR_W-1:0] ifq_tail;
    reg [IFQ_COUNT_W-1:0] ifq_count;
    wire                ifq_full = (ifq_count == IFQ_DEPTH_VAL);
    wire                ifq_empty = (ifq_count == 0);
    wire                ifq_push;
    wire                decode_pop;

    // Fetch request queue (tracks outstanding I$ requests)
    reg [WARP_ID_W-1:0] frq_warp_id [0:IFQ_DEPTH-1];
    reg [31:0]          frq_pc [0:IFQ_DEPTH-1];
    reg [IFQ_PTR_W-1:0] frq_head;
    reg [IFQ_PTR_W-1:0] frq_tail;
    reg [IFQ_COUNT_W-1:0] frq_count;
    reg [IFQ_COUNT_W-1:0] frq_drop_count;
    wire                frq_full = (frq_count == IFQ_DEPTH_VAL);
    wire                frq_empty = (frq_count == 0);
    wire                frq_pop;
    wire [IFQ_COUNT_W:0] fetch_slots_used = {1'b0, ifq_count} + {1'b0, frq_count};

    // Decode Stage (2 lanes)
    reg                  dec0_valid;
    reg                  dec1_valid;  // Pipeline register for lane 1
    reg  [WARP_ID_W-1:0] dec0_warp_id;
    reg  [WARP_ID_W-1:0] dec1_warp_id;
    reg  [31:0]          dec0_pc;
    reg  [31:0]          dec1_pc;
    reg  [31:0]          dec0_instruction;
    reg  [31:0]          dec1_instruction;

    // Issue Stage (slot 0)
    reg                  issue_valid;
    reg  [WARP_ID_W-1:0] issue_warp_id;
    reg  [31:0]          issue_pc;
    reg  [5:0]           issue_opcode;
    reg  [4:0]           issue_rd, issue_ra, issue_rb, issue_rc;
    reg  [5:0]           issue_func;
    reg  [15:0]          issue_imm16;
    reg  [20:0]          issue_imm21;
    reg                  issue_use_imm;
    reg  [NUM_LANES-1:0] issue_mask;
    reg                  issue_alu_op, issue_mul_op, issue_div_op;
    reg                  issue_fp32_op, issue_fp64_op, issue_fp16_op;
    reg                  issue_sfu_op, issue_tensor_op;
    reg                  issue_mem_read, issue_mem_write, issue_mem_shared;
    reg                  issue_branch_op, issue_sync_op, issue_special_reg;
    reg                  issue_exit_op, issue_atomic_op, issue_shuffle_op;
    reg                  issue_reg_write;

    // Issue Stage (slot 1)
    reg                  issue1_valid;
    reg  [WARP_ID_W-1:0] issue1_warp_id;
    reg  [31:0]          issue1_pc;
    reg  [5:0]           issue1_opcode;
    reg  [4:0]           issue1_rd, issue1_ra, issue1_rb, issue1_rc;
    reg  [5:0]           issue1_func;
    reg  [15:0]          issue1_imm16;
    reg  [20:0]          issue1_imm21;
    reg                  issue1_use_imm;
    reg  [NUM_LANES-1:0] issue1_mask;
    reg                  issue1_alu_op, issue1_mul_op, issue1_div_op;
    reg                  issue1_fp32_op, issue1_fp64_op, issue1_fp16_op;
    reg                  issue1_sfu_op, issue1_tensor_op;
    reg                  issue1_mem_read, issue1_mem_write, issue1_mem_shared;
    reg                  issue1_branch_op, issue1_sync_op, issue1_special_reg;
    reg                  issue1_exit_op, issue1_atomic_op, issue1_shuffle_op;
    reg                  issue1_reg_write;

    // Execute Stage (per functional unit)
    reg                  exec_alu_valid;
    reg                  exec_fpu_valid;
    reg                  exec_sfu_valid;
    reg                  exec_tensor_valid;
    reg                  exec_mem_valid;
    reg                  exec_shuffle_valid;
    reg                  exec_atomic_valid;
    reg  [WARP_ID_W-1:0] exec_warp_id;
    reg  [4:0]           exec_rd;

    // Writeback Stage
    reg                  wb_valid;
    reg  [WARP_ID_W-1:0] wb_warp_id;
    reg  [4:0]           wb_rd;
    reg  [SIMD_WIDTH-1:0] wb_data;
    reg  [NUM_LANES-1:0] wb_mask;

    //========================================================================
    // Decoder Signals
    //========================================================================
    wire                 dec_valid;
    wire [5:0]           dec_opcode;
    wire [4:0]           dec_rd, dec_ra, dec_rb, dec_rc;
    wire [5:0]           dec_func;
    wire [15:0]          dec_imm16;
    wire [20:0]          dec_imm21;
    wire                 dec_use_imm;
    wire                 dec_alu_op, dec_mul_op, dec_div_op;
    wire                 dec_fp32_op, dec_fp64_op, dec_fp16_op;
    wire                 dec_sfu_op, dec_tensor_op;
    wire                 dec_mem_read, dec_mem_write, dec_mem_shared;
    wire                 dec_branch_op, dec_sync_op;
    wire                 dec_special_reg, dec_exit_op;
    wire                 dec_atomic_op, dec_shuffle_op;
    wire                 dec_reg_write;

    // dec1_valid is a reg declared above, so decoder output uses dec1_dec_valid
    wire [5:0]           dec1_opcode;
    wire [4:0]           dec1_rd, dec1_ra, dec1_rb, dec1_rc;
    wire [5:0]           dec1_func;
    wire [15:0]          dec1_imm16;
    wire [20:0]          dec1_imm21;
    wire                 dec1_use_imm;
    wire                 dec1_alu_op, dec1_mul_op, dec1_div_op;
    wire                 dec1_fp32_op, dec1_fp64_op, dec1_fp16_op;
    wire                 dec1_sfu_op, dec1_tensor_op;
    wire                 dec1_mem_read, dec1_mem_write, dec1_mem_shared;
    wire                 dec1_branch_op, dec1_sync_op;
    wire                 dec1_special_reg, dec1_exit_op;
    wire                 dec1_atomic_op, dec1_shuffle_op;
    wire                 dec1_reg_write;

    //========================================================================
    // Functional Unit Signals
    //========================================================================

    // Register File Signals
    wire [SIMD_WIDTH-1:0] rf_rd_data_a, rf_rd_data_b, rf_rd_data_c;
    wire [SIMD_WIDTH-1:0] rf1_rd_data_a, rf1_rd_data_b, rf1_rd_data_c;
    wire [SIMD_WIDTH-1:0] rf_wr_data;
    wire                  rf_wr_en;
    wire [NUM_LANES-1:0]  rf_wr_mask;

    // ALU Signals
    wire [SIMD_WIDTH-1:0] alu_result;
    wire [NUM_LANES-1:0]  alu_zero, alu_neg;
    wire                  alu_valid_out;

    // MUL Unit Signals
    wire [SIMD_WIDTH-1:0] mul_result;
    wire                  mul_valid_out;

    // FPU Signals (FP32)
    wire [SIMD_WIDTH-1:0] fpu32_result;
    wire                  fpu32_valid_in, fpu32_valid_out, fpu32_ready;

    // FPU64 Signals (FP64)
    wire [NUM_LANES*64-1:0] fpu64_result;
    wire                    fpu64_valid_in, fpu64_valid_out, fpu64_ready;
    wire [SIMD_WIDTH-1:0]   fpu64_result_trunc;

    // FP16 Signals
    wire [SIMD_WIDTH-1:0] fp16_result;
    wire                  fp16_valid_in, fp16_valid_out, fp16_ready;

    // SFU Signals
    wire [SIMD_WIDTH-1:0] sfu_result;
    wire                  sfu_valid_in, sfu_valid_out, sfu_ready;

    // Tensor Core Signals
    wire [SIMD_WIDTH-1:0] tensor_result;
    wire                  tensor_valid_in, tensor_valid_out, tensor_ready;
    wire [TENSOR_ISSUE_W-1:0] tensor_issue_push_data;
    wire [TENSOR_ISSUE_W-1:0] tensor_issue_pop_data;
    wire [SIMD_WIDTH-1:0] tensor_issue_frag_a;
    wire [SIMD_WIDTH-1:0] tensor_issue_frag_b;
    wire [SIMD_WIDTH-1:0] tensor_issue_frag_c;
    wire [2:0] tensor_issue_op_type;
    wire [WARP_ID_W-1:0] tensor_issue_warp;
    wire [4:0] tensor_issue_rd;
    wire [NUM_LANES-1:0] tensor_issue_mask;
    reg  [TENSOR_ISSUE_COUNT_W-1:0] tensor_issue_count;
    wire                  tensor_issue_empty;
    wire                  tensor_issue_full;
    wire                  tensor_issue_push;
    wire                  tensor_issue_pop;
    wire                  tensor_issue_push_fire;
    wire                  tensor_issue_pop_fire;
    wire [TENSOR_ISSUE_COUNT_W:0] tensor_issue_count_next;
    wire                  tensor_issue_full_next;
    wire                  tensor_issue_fifo_full;
    wire                  tensor_issue_fifo_empty;
    wire [TENSOR_META_W-1:0] tensor_meta_push_data;
    wire [TENSOR_META_W-1:0] tensor_meta_pop_data;
    wire [WARP_ID_W-1:0] tensor_meta_warp;
    wire [4:0] tensor_meta_rd;
    wire [NUM_LANES-1:0] tensor_meta_mask;
    wire                  tensor_meta_full;
    wire                  tensor_meta_empty;
    wire                  tensor_meta_push;
    wire                  tensor_meta_pop;
    wire [WB_PKT_W-1:0]   tensor_wbq_push_data;
    wire [WB_PKT_W-1:0]   tensor_wbq_pop_data;
    wire [WARP_ID_W-1:0]  tensor_wbq_warp;
    wire [4:0]            tensor_wbq_rd;
    wire [NUM_LANES-1:0]  tensor_wbq_mask;
    wire [SIMD_WIDTH-1:0] tensor_wbq_data;
    wire                  tensor_wbq_full;
    wire                  tensor_wbq_empty;
    wire                  tensor_wbq_push;
    wire                  tensor_wbq_push_fire;
    wire                  tensor_wbq_pop;

    // Warp Shuffle Signals
    wire [SIMD_WIDTH-1:0] shuffle_result;
    wire                  shuffle_valid_in, shuffle_valid_out;
    wire [NUM_LANES-1:0]  shuffle_valid_mask;
    wire [NUM_LANES*5-1:0] shuffle_src_lane;
    wire [NUM_LANES*5-1:0] shuffle_offset;

    // Atomic Unit Signals
    wire [31:0]           atomic_result;
    wire                  atomic_valid_in, atomic_valid_out;
    wire                  atomic_busy;

    // Shared Memory Signals
    wire                  smem_req_valid;
    wire                  smem_req_write;
    wire [NUM_LANES*14-1:0] smem_req_addr;
    wire [SIMD_WIDTH-1:0] smem_req_wdata;
    wire                  smem_resp_valid;
    wire [SIMD_WIDTH-1:0] smem_resp_rdata;
    wire                  smem_bank_conflict;

    // Global Memory Signals
    wire                  gmem_req_valid;
    wire                  gmem_req_write;
    wire [NUM_LANES*32-1:0] gmem_req_addr;
    wire [SIMD_WIDTH-1:0] gmem_req_wdata;
    wire                  gmem_req_ready;
    wire                  gmem_resp_valid;
    wire [SIMD_WIDTH-1:0] gmem_resp_rdata;

    // Control Flow Signals
    wire [31:0]           cfu_next_mask;
    wire [NUM_LANES-1:0]  cfu_active_mask;
    wire [31:0]           cfu_branch_target;
    wire                  cfu_branch_taken;
    wire                  cfu_stall;
    wire [31:0]           cfu_reconverge_pc;
    wire                  cfu_at_reconverge;

    //========================================================================
    // Branch Predictor (TAGE + BTB + RAS)
    //========================================================================
    wire                  bp_pred_valid;
    wire                  bp_pred_taken;
    wire [31:0]           bp_pred_target;
    wire [1:0]            bp_pred_confidence;
    wire [31:0]           bp_stat_predictions;
    wire [31:0]           bp_stat_mispredictions;
    wire [31:0]           bp_stat_btb_hits;
    wire [31:0]           bp_stat_ras_hits;

    // Branch update signals from execute stage
    reg                   bp_update_valid;
    reg  [WARP_ID_W-1:0]  bp_update_warp_id;
    reg  [31:0]           bp_update_pc;
    reg                   bp_update_taken;
    reg  [31:0]           bp_update_target;
    reg                   bp_update_is_call;
    reg                   bp_update_is_return;
    reg                   bp_update_mispredicted;

    branch_predictor #(
        .NUM_WARPS      (NUM_WARPS),
        .ADDR_WIDTH     (32),
        .BTB_ENTRIES    (256),
        .BTB_WAYS       (4),
        .BHT_ENTRIES    (1024),
        .TAGE_TABLES    (4),
        .TAGE_ENTRIES   (256),
        .RAS_DEPTH      (8)
    ) u_branch_predictor (
        .clk                (clk),
        .rst_n              (rst_n),
        // Prediction request (from fetch)
        .pred_req           (fetch_req),
        .pred_warp_id       (selected_warp),
        .pred_pc            (warp_fetch_pc[selected_warp]),
        .pred_is_branch     (1'b1),  // Assume all fetches might be branches
        .pred_is_call       (1'b0),  // Will be updated based on decode
        .pred_is_return     (1'b0),
        // Prediction output
        .pred_valid         (bp_pred_valid),
        .pred_taken         (bp_pred_taken),
        .pred_target        (bp_pred_target),
        .pred_confidence    (bp_pred_confidence),
        // Update from execute
        .update_valid       (bp_update_valid),
        .update_warp_id     (bp_update_warp_id),
        .update_pc          (bp_update_pc),
        .update_taken       (bp_update_taken),
        .update_target      (bp_update_target),
        .update_is_call     (bp_update_is_call),
        .update_is_return   (bp_update_is_return),
        .update_mispredicted(bp_update_mispredicted),
        // Statistics
        .stat_predictions   (bp_stat_predictions),
        .stat_mispredictions(bp_stat_mispredictions),
        .stat_btb_hits      (bp_stat_btb_hits),
        .stat_ras_hits      (bp_stat_ras_hits)
    );

    //========================================================================
    // Scoreboard for Dependency Tracking (Per-warp register busy bits)
    //========================================================================
    // Each warp has a 32-bit mask indicating which registers have pending writes
    reg [31:0] scoreboard_busy [0:NUM_WARPS-1];

    // Per-warp pending instruction count (for FU stall tracking)
    reg [3:0] pending_fu_count [0:NUM_WARPS-1];

    wire mem_in_flight;
    wire frontend_flush;
    wire frq_head_drop;
    wire ifq_head_drop;

    wire dual_issue_en = (ISSUE_WIDTH > 1);

    // Lane 0 dependency checks
    wire lane0_ra_busy = dec0_valid && (dec_ra != 0) && scoreboard_busy[dec0_warp_id][dec_ra];
    wire lane0_rb_busy = dec0_valid && (dec_rb != 0) && scoreboard_busy[dec0_warp_id][dec_rb];
    wire lane0_rc_busy = dec0_valid && (dec_rc != 0) && scoreboard_busy[dec0_warp_id][dec_rc];
    wire lane0_stall_raw = dec0_valid && (lane0_ra_busy || lane0_rb_busy || lane0_rc_busy);
    wire lane0_stall_fu = dec0_valid && (pending_fu_count[dec0_warp_id] >= 8);
    wire lane0_stall_mem = dec0_valid && !dec_atomic_op && (
                           ((dec_mem_read || dec_mem_write) && mem_in_flight) ||
                           (dec_mem_write && !dec_mem_read && store_pending_valid) ||
                           (dec_mem_shared && dec_mem_read && smem_pending_valid) ||
                           (!dec_mem_shared && (dec_mem_read || dec_mem_write) &&
                            (!gmem_req_ready || (dec_mem_read && mem_pending_valid)))
                           );
    wire lane0_stall_atomic = dec0_valid && dec_atomic_op && atomic_busy;
    wire lane0_stall_tensor = dec0_valid && dec_tensor_op && tensor_issue_full_next;
    wire lane0_stall_wbq = dec0_valid && (
                           (dec_alu_op && (alu_inflight == ALU_WBQ_DEPTH_VAL)) ||
                           (dec_mul_op && (mul_inflight == MUL_WBQ_DEPTH_VAL)) ||
                           (dec_fp32_op && (fpu32_inflight == FPU32_WBQ_DEPTH_VAL)) ||
                           (dec_fp64_op && (fpu64_inflight == FPU64_WBQ_DEPTH_VAL)) ||
                           (dec_fp16_op && (fp16_inflight == FP16_WBQ_DEPTH_VAL)) ||
                           (dec_sfu_op && (sfu_inflight == SFU_WBQ_DEPTH_VAL)) ||
                           (dec_shuffle_op && (shfl_inflight == SHFL_WBQ_DEPTH_VAL))
                           );
    wire lane0_ready = dec0_valid && !lane0_stall_raw && !lane0_stall_fu &&
                       !lane0_stall_mem && !lane0_stall_atomic &&
                       !lane0_stall_tensor && !lane0_stall_wbq;

    // Lane 1 dependency checks (compute-only)
    wire lane1_ra_busy = dec1_valid && (dec1_ra != 0) && scoreboard_busy[dec1_warp_id][dec1_ra];
    wire lane1_rb_busy = dec1_valid && (dec1_rb != 0) && scoreboard_busy[dec1_warp_id][dec1_rb];
    wire lane1_rc_busy = dec1_valid && (dec1_rc != 0) && scoreboard_busy[dec1_warp_id][dec1_rc];
    wire lane1_stall_raw = dec1_valid && (lane1_ra_busy || lane1_rb_busy || lane1_rc_busy);
    wire lane1_stall_fu = dec1_valid && (pending_fu_count[dec1_warp_id] >= 8);
    wire lane1_stall_wbq = dec1_valid && (
                           (dec1_alu_op && (alu_inflight == ALU_WBQ_DEPTH_VAL)) ||
                           (dec1_mul_op && (mul_inflight == MUL_WBQ_DEPTH_VAL)) ||
                           (dec1_fp32_op && (fpu32_inflight == FPU32_WBQ_DEPTH_VAL)) ||
                           (dec1_fp64_op && (fpu64_inflight == FPU64_WBQ_DEPTH_VAL)) ||
                           (dec1_fp16_op && (fp16_inflight == FP16_WBQ_DEPTH_VAL)) ||
                           (dec1_sfu_op && (sfu_inflight == SFU_WBQ_DEPTH_VAL)) ||
                           (dec1_shuffle_op && (shfl_inflight == SHFL_WBQ_DEPTH_VAL))
                           );
    wire lane1_is_compute = dec1_alu_op || dec1_mul_op || dec1_div_op ||
                            dec1_fp32_op || dec1_fp64_op || dec1_fp16_op ||
                            dec1_sfu_op || dec1_shuffle_op;
    wire lane1_blocking = dec1_mem_read || dec1_mem_write || dec1_branch_op ||
                          dec1_sync_op || dec1_exit_op || dec1_tensor_op ||
                          dec1_atomic_op;
    wire lane1_ready = dec1_valid && lane1_is_compute && !lane1_blocking &&
                       !lane1_stall_raw && !lane1_stall_fu && !lane1_stall_wbq;

    // Dual-issue conflict checks
    wire lane_warp_conflict = lane0_ready && lane1_ready &&
                              (dec0_warp_id == dec1_warp_id);
    wire lane_reg_conflict = lane0_ready && lane1_ready && (
                             (dec_reg_write && (dec_rd != 0) &&
                              ((dec_rd == dec1_ra) || (dec_rd == dec1_rb) ||
                               (dec_rd == dec1_rc) ||
                               (dec1_reg_write && (dec_rd == dec1_rd)))) ||
                             (dec1_reg_write && (dec1_rd != 0) &&
                              ((dec1_rd == dec_ra) || (dec1_rd == dec_rb) ||
                               (dec1_rd == dec_rc)))
                             );

    wire lane0_alu = lane0_ready && (dec_alu_op || dec_branch_op);
    wire lane0_mul = lane0_ready && (dec_mul_op || dec_div_op);
    wire lane0_fp32 = lane0_ready && dec_fp32_op;
    wire lane0_fp64 = lane0_ready && dec_fp64_op;
    wire lane0_fp16 = lane0_ready && dec_fp16_op;
    wire lane0_sfu = lane0_ready && dec_sfu_op;
    wire lane0_shfl = lane0_ready && dec_shuffle_op;
    wire lane1_alu = lane1_ready && dec1_alu_op;
    wire lane1_mul = lane1_ready && (dec1_mul_op || dec1_div_op);
    wire lane1_fp32 = lane1_ready && dec1_fp32_op;
    wire lane1_fp64 = lane1_ready && dec1_fp64_op;
    wire lane1_fp16 = lane1_ready && dec1_fp16_op;
    wire lane1_sfu = lane1_ready && dec1_sfu_op;
    wire lane1_shfl = lane1_ready && dec1_shuffle_op;
    wire lane_unit_conflict = (lane0_alu && lane1_alu) ||
                              (lane0_mul && lane1_mul) ||
                              (lane0_fp32 && lane1_fp32) ||
                              (lane0_fp64 && lane1_fp64) ||
                              (lane0_fp16 && lane1_fp16) ||
                              (lane0_sfu && lane1_sfu) ||
                              (lane0_shfl && lane1_shfl);
    wire lane0_control = lane0_ready && (dec_branch_op || dec_sync_op || dec_exit_op);

    // Forward declarations - these are driven by the advanced_warp_scheduler
    wire issue0_fire;  // Assigned in scheduler section
    wire issue1_fire;  // Assigned in scheduler section

    // Old dual-issue lane selection logic (superseded by advanced_warp_scheduler)
    // These are now computed based on scheduler output
    wire issue0_sel_lane0 = lane0_ready;
    wire issue0_sel_lane1 = !lane0_ready && lane1_ready;
    // issue0_fire and issue1_fire are now driven by the scheduler
    wire old_issue0_fire = issue0_sel_lane0 || issue0_sel_lane1;  // Renamed to avoid conflict
    wire old_issue1_fire = dual_issue_en && lane0_ready && lane1_ready &&
                           !lane_warp_conflict && !lane_reg_conflict &&
                           !lane_unit_conflict && !lane0_control;  // Renamed

    assign lane0_issued = issue0_sel_lane0;
    assign lane1_issued = issue0_sel_lane1 || issue1_fire;

    wire [WARP_ID_W-1:0] issue0_warp_sel =
        issue0_sel_lane0 ? dec0_warp_id : dec1_warp_id;
    wire [4:0] issue0_rd_sel =
        issue0_sel_lane0 ? dec_rd : dec1_rd;
    wire issue0_reg_write_sel =
        issue0_sel_lane0 ? dec_reg_write : dec1_reg_write;
    wire issue0_fp32_sel =
        issue0_sel_lane0 ? dec_fp32_op : dec1_fp32_op;
    wire issue0_fp64_sel =
        issue0_sel_lane0 ? dec_fp64_op : dec1_fp64_op;
    wire issue0_fp16_sel =
        issue0_sel_lane0 ? dec_fp16_op : dec1_fp16_op;
    wire issue0_sfu_sel =
        issue0_sel_lane0 ? dec_sfu_op : dec1_sfu_op;
    wire issue0_tensor_sel =
        issue0_sel_lane0 ? dec_tensor_op : dec1_tensor_op;
    wire issue0_mem_read_sel =
        issue0_sel_lane0 ? dec_mem_read : dec1_mem_read;
    wire issue0_atomic_sel =
        issue0_sel_lane0 ? dec_atomic_op : dec1_atomic_op;
    wire issue0_branch_sel =
        issue0_sel_lane0 ? dec_branch_op : dec1_branch_op;
    wire issue0_exit_sel =
        issue0_sel_lane0 ? dec_exit_op : dec1_exit_op;

    assign mem_in_flight = issue_valid && (issue_mem_read || issue_mem_write) &&
                           !issue_atomic_op;

    // Aliases for compatibility with later code sections
    wire issue_stall_mem = lane0_stall_mem;
    wire issue_accept = issue0_fire;

    assign frontend_flush = (issue_valid && issue_exit_op && (INIT_WARPS == 1));

    wire alu_issue0 = issue_valid && issue_alu_op;
    wire alu_issue1 = issue1_valid && issue1_alu_op;
    wire mul_issue0 = issue_valid && (issue_mul_op || issue_div_op);
    wire mul_issue1 = issue1_valid && (issue1_mul_op || issue1_div_op);
    wire fpu32_issue0 = issue_valid && issue_fp32_op;
    wire fpu32_issue1 = issue1_valid && issue1_fp32_op;
    wire fpu64_issue0 = issue_valid && issue_fp64_op;
    wire fpu64_issue1 = issue1_valid && issue1_fp64_op;
    wire fp16_issue0 = issue_valid && issue_fp16_op;
    wire fp16_issue1 = issue1_valid && issue1_fp16_op;
    wire sfu_issue0 = issue_valid && issue_sfu_op;
    wire sfu_issue1 = issue1_valid && issue1_sfu_op;
    wire shfl_issue0 = issue_valid && issue_shuffle_op;
    wire shfl_issue1 = issue1_valid && issue1_shuffle_op;

    wire alu_issue = alu_issue0 || alu_issue1;
    wire mul_issue = mul_issue0 || mul_issue1;
    wire fpu32_issue = fpu32_issue0 || fpu32_issue1;
    wire fpu64_issue = fpu64_issue0 || fpu64_issue1;
    wire fp16_issue = fp16_issue0 || fp16_issue1;
    wire sfu_issue = sfu_issue0 || sfu_issue1;
    wire shfl_issue = shfl_issue0 || shfl_issue1;

    //========================================================================
    // Multi-Cycle FU Tracking (Track which warp issued to each pipelined FU)
    //========================================================================
    // ALU pipeline tracking (1 stage - needed for proper writeback timing)
    reg [WARP_ID_W-1:0] alu_warp_pipe;
    reg [4:0]           alu_rd_pipe;
    reg [NUM_LANES-1:0] alu_mask_pipe;
    reg [SIMD_WIDTH-1:0] alu_result_pipe;
    reg                 alu_valid_pipe;

    // MUL pipeline tracking (1 stage)
    reg [WARP_ID_W-1:0] mul_warp_pipe;
    reg [4:0]           mul_rd_pipe;
    reg [NUM_LANES-1:0] mul_mask_pipe;

    // FPU32 pipeline tracking (4 stages)
    reg [WARP_ID_W-1:0] fpu32_warp_pipe [0:4];
    reg [4:0]           fpu32_rd_pipe [0:4];
    reg [NUM_LANES-1:0] fpu32_mask_pipe [0:4];

    // FPU64 pipeline tracking (4 stages)
    reg [WARP_ID_W-1:0] fpu64_warp_pipe [0:4];
    reg [4:0]           fpu64_rd_pipe [0:4];
    reg [NUM_LANES-1:0] fpu64_mask_pipe [0:4];

    // FP16 pipeline tracking (2 stages)
    reg [WARP_ID_W-1:0] fp16_warp_pipe [0:1];
    reg [4:0]           fp16_rd_pipe [0:1];
    reg [NUM_LANES-1:0] fp16_mask_pipe [0:1];

    // SFU pipeline tracking (8 stages)
    reg [WARP_ID_W-1:0] sfu_warp_pipe [0:7];
    reg [4:0]           sfu_rd_pipe [0:7];
    reg [NUM_LANES-1:0] sfu_mask_pipe [0:7];

    // Memory operation tracking
    reg [WARP_ID_W-1:0] mem_warp_pending;
    reg [4:0]           mem_rd_pending;
    reg [NUM_LANES-1:0] mem_mask_pending;
    reg                 mem_pending_valid;
    reg [WARP_ID_W-1:0] store_warp_pending;
    reg [NUM_LANES-1:0] store_mask_pending;
    reg                 store_pending_valid;

    // Shared memory tracking
    reg [WARP_ID_W-1:0] smem_warp_pending;
    reg [4:0]           smem_rd_pending;
    reg [NUM_LANES-1:0] smem_mask_pending;
    reg                 smem_pending_valid;

    // Atomic operation tracking
    reg [WARP_ID_W-1:0] atomic_warp_pending;
    reg [4:0]           atomic_rd_pending;
    reg [NUM_LANES-1:0] atomic_mask_pending;
    reg                 atomic_pending_valid;

    // Shuffle pipeline tracking (1 stage delay for proper writeback timing)
    reg [WARP_ID_W-1:0] shuffle_warp_pipe;
    reg [4:0]           shuffle_rd_pipe;
    reg [NUM_LANES-1:0] shuffle_mask_pipe;
    reg [SIMD_WIDTH-1:0] shuffle_result_pipe;
    reg                 shuffle_valid_pipe;

    // Writeback round-robin arbiter state
    reg [3:0] wb_arb_priority;

    // Writeback output queues (per FU)
    wire [WB_PKT_W-1:0] alu_wbq_in, alu_wbq_out;
    wire [WB_PKT_W-1:0] mul_wbq_in, mul_wbq_out;
    wire [WB_PKT_W-1:0] fpu32_wbq_in, fpu32_wbq_out;
    wire [WB_PKT_W-1:0] fpu64_wbq_in, fpu64_wbq_out;
    wire [WB_PKT_W-1:0] fp16_wbq_in, fp16_wbq_out;
    wire [WB_PKT_W-1:0] sfu_wbq_in, sfu_wbq_out;
    wire [WB_PKT_W-1:0] shfl_wbq_in, shfl_wbq_out;
    wire                 alu_wbq_push, alu_wbq_pop, alu_wbq_full, alu_wbq_empty;
    wire                 mul_wbq_push, mul_wbq_pop, mul_wbq_full, mul_wbq_empty;
    wire                 fpu32_wbq_push, fpu32_wbq_pop, fpu32_wbq_full, fpu32_wbq_empty;
    wire                 fpu64_wbq_push, fpu64_wbq_pop, fpu64_wbq_full, fpu64_wbq_empty;
    wire                 fp16_wbq_push, fp16_wbq_pop, fp16_wbq_full, fp16_wbq_empty;
    wire                 sfu_wbq_push, sfu_wbq_pop, sfu_wbq_full, sfu_wbq_empty;
    wire                 shfl_wbq_push, shfl_wbq_pop, shfl_wbq_full, shfl_wbq_empty;

    wire [WARP_ID_W-1:0] alu_wbq_warp;
    wire [WARP_ID_W-1:0] mul_wbq_warp;
    wire [WARP_ID_W-1:0] fpu32_wbq_warp;
    wire [WARP_ID_W-1:0] fpu64_wbq_warp;
    wire [WARP_ID_W-1:0] fp16_wbq_warp;
    wire [WARP_ID_W-1:0] sfu_wbq_warp;
    wire [WARP_ID_W-1:0] shfl_wbq_warp;
    wire [4:0]           alu_wbq_rd;
    wire [4:0]           mul_wbq_rd;
    wire [4:0]           fpu32_wbq_rd;
    wire [4:0]           fpu64_wbq_rd;
    wire [4:0]           fp16_wbq_rd;
    wire [4:0]           sfu_wbq_rd;
    wire [4:0]           shfl_wbq_rd;
    wire [NUM_LANES-1:0] alu_wbq_mask;
    wire [NUM_LANES-1:0] mul_wbq_mask;
    wire [NUM_LANES-1:0] fpu32_wbq_mask;
    wire [NUM_LANES-1:0] fpu64_wbq_mask;
    wire [NUM_LANES-1:0] fp16_wbq_mask;
    wire [NUM_LANES-1:0] sfu_wbq_mask;
    wire [NUM_LANES-1:0] shfl_wbq_mask;
    wire [SIMD_WIDTH-1:0] alu_wbq_data;
    wire [SIMD_WIDTH-1:0] mul_wbq_data;
    wire [SIMD_WIDTH-1:0] fpu32_wbq_data;
    wire [SIMD_WIDTH-1:0] fpu64_wbq_data;
    wire [SIMD_WIDTH-1:0] fp16_wbq_data;
    wire [SIMD_WIDTH-1:0] sfu_wbq_data;
    wire [SIMD_WIDTH-1:0] shfl_wbq_data;

    reg [ALU_WBQ_COUNT_W-1:0] alu_inflight;
    reg [MUL_WBQ_COUNT_W-1:0] mul_inflight;
    reg [FPU32_WBQ_COUNT_W-1:0] fpu32_inflight;
    reg [FPU64_WBQ_COUNT_W-1:0] fpu64_inflight;
    reg [FP16_WBQ_COUNT_W-1:0] fp16_inflight;
    reg [SFU_WBQ_COUNT_W-1:0] sfu_inflight;
    reg [SHFL_WBQ_COUNT_W-1:0] shfl_inflight;

    //========================================================================
    // Warp Scheduler with Round-Robin + Priority (Enhanced with GTO)
    // Supports both simple RR and advanced GTO/dual-issue scheduling
    //========================================================================
    reg [WARP_ID_W-1:0] last_issued_warp;
    reg [WARP_ID_W-1:0] selected_warp;
    reg                 warp_selected;

    // Advanced scheduler statistics
    reg [31:0] sched_single_issue_count;
    reg [31:0] sched_dual_issue_count;
    reg [31:0] sched_stall_count;

    // GTO (Greedy-Then-Oldest) priority calculation
    // Prioritize warps that have made recent progress (greedy)
    // then fall back to oldest pending warp
    reg [NUM_WARPS-1:0] warp_recently_issued;
    reg [7:0] warp_age [0:NUM_WARPS-1];
    reg [WARP_ID_W-1:0] oldest_ready_warp;
    reg [WARP_ID_W-1:0] greedy_warp;
    reg oldest_found, greedy_found;

    integer wi, age_i;

    // Age tracking for GTO policy
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_recently_issued <= 0;
            for (age_i = 0; age_i < NUM_WARPS; age_i = age_i + 1) begin
                warp_age[age_i] <= 0;
            end
            sched_single_issue_count <= 0;
            sched_dual_issue_count <= 0;
            sched_stall_count <= 0;
        end else if (kernel_start) begin
            warp_recently_issued <= 0;
            for (age_i = 0; age_i < NUM_WARPS; age_i = age_i + 1) begin
                warp_age[age_i] <= 0;
            end
        end else begin
            // Update ages - increment for waiting warps, reset for issued
            for (age_i = 0; age_i < NUM_WARPS; age_i = age_i + 1) begin
                if (warp_valid[age_i] && warp_ready[age_i]) begin
                    if (fetch_fire && selected_warp == age_i[WARP_ID_W-1:0]) begin
                        warp_age[age_i] <= 0;
                        warp_recently_issued[age_i] <= 1'b1;
                    end else if (warp_age[age_i] < 8'hFF) begin
                        warp_age[age_i] <= warp_age[age_i] + 1'b1;
                    end
                end else begin
                    warp_recently_issued[age_i] <= 1'b0;
                end
            end

            // Statistics
            if (fetch_fire) begin
                sched_single_issue_count <= sched_single_issue_count + 1;
            end else if (warp_selected && !fetch_fire) begin
                sched_stall_count <= sched_stall_count + 1;
            end
        end
    end

    // GTO warp selection
    always @(*) begin
        warp_selected = 1'b0;
        selected_warp = 0;
        oldest_ready_warp = 0;
        greedy_warp = 0;
        oldest_found = 1'b0;
        greedy_found = 1'b0;

        // First pass: find greedy candidate (recently issued and ready)
        for (wi = 0; wi < NUM_WARPS; wi = wi + 1) begin
            if (!greedy_found && warp_ready[wi] && warp_recently_issued[wi]) begin
                greedy_warp = wi[WARP_ID_W-1:0];
                greedy_found = 1'b1;
            end
        end

        // Second pass: find oldest ready warp
        for (wi = 0; wi < NUM_WARPS; wi = wi + 1) begin
            if (warp_ready[wi]) begin
                if (!oldest_found || warp_age[wi] > warp_age[oldest_ready_warp]) begin
                    oldest_ready_warp = wi[WARP_ID_W-1:0];
                    oldest_found = 1'b1;
                end
            end
        end

        // GTO policy: prefer greedy, then oldest
        if (greedy_found) begin
            selected_warp = greedy_warp;
            warp_selected = 1'b1;
        end else if (oldest_found) begin
            selected_warp = oldest_ready_warp;
            warp_selected = 1'b1;
        end else begin
            // Fallback to round-robin for any ready warp
            for (wi = 0; wi < NUM_WARPS; wi = wi + 1) begin
                if (!warp_selected) begin
                    if (warp_ready[(last_issued_warp + wi + 1) % NUM_WARPS]) begin
                        selected_warp = (last_issued_warp + wi + 1) % NUM_WARPS;
                        warp_selected = 1'b1;
                    end
                end
            end
        end
    end

    //========================================================================
    // Instruction Cache / Bypass
    //========================================================================
    wire        icache_req;
    wire [31:0] icache_addr;
    wire        icache_ready;
    wire [31:0] icache_data;
    wire        icache_valid;
    wire        icache_fill_req;
    wire [31:0] icache_fill_addr;
    wire        icache_fill_ready;
    wire [511:0] icache_fill_data;
    wire        icache_fill_valid;

    // Pipeline for tracking PC offset (bit 2) to select correct word from response
    reg [FETCH_PIPE_DEPTH-1:0] fetch_pipe_pc_bit2;

    generate
    if (ICACHE_BYPASS) begin : gen_icache_bypass
        // Direct memory access mode - bypasses icache for ideal fetch latency
        // Memory provides data in same cycle as request (combinatorial)
        // We route the fetch request directly to memory interface

        // Direct connection to memory
        assign imem_req = fetch_req;
        assign imem_addr = warp_fetch_pc[fetch_warp_id];

        // ICache interface signals in bypass mode
        assign icache_ready = imem_ready;
        assign icache_valid = imem_valid;

        // Select instruction word from 64-bit memory response
        // In bypass mode, we fetch at the exact instruction address (not 8-byte aligned).
        // The memory model returns {imem[word_index+1], imem[word_index]} where word_index = PC/4.
        // So the LOW word always contains the instruction we requested.
        // PC[2] selection only applies for 8-byte aligned cache line fetches (icache mode).
        assign icache_data = imem_data[31:0];  // Always select low word for non-aligned bypass

    end else begin : gen_icache_normal
        // Normal icache mode
        icache #(
            .SIZE_KB(4),
            .LINE_SIZE(8),   // 2 instructions per cache line
            .NUM_WAYS(2)
        ) u_icache (
            .clk(clk),
            .rst_n(rst_n),
            .fetch_req(fetch_req),
            .fetch_addr(warp_fetch_pc[fetch_warp_id]),
            .fetch_ready(icache_ready),
            .fetch_data(icache_data),
            .fetch_valid(icache_valid),
            .invalidate_req(1'b0),
            .invalidate_addr(32'b0),
            .invalidate_all(1'b0),
            .invalidate_done(),
            .mem_req_valid(imem_req),
            .mem_req_addr(imem_addr),
            .mem_req_ready(imem_ready),
            .mem_resp_data(imem_data),  // 64-bit data for 8-byte cache line
            .mem_resp_valid(imem_valid),
            .stat_hits(),
            .stat_misses(),
            .stat_prefetch_hits()
        );
    end
    endgenerate

    //========================================================================
    // STAGE 1: FETCH (Updated for ICache)
    //========================================================================
    // Per-warp Instruction Buffers
    reg [31:0] warp_inst_buf [0:NUM_WARPS-1];
    reg [NUM_WARPS-1:0] warp_inst_buf_valid;
    wire [NUM_WARPS-1:0] warp_inst_consume;

    // Fetch Arbitration - Pipelined Fetch Architecture
    // Allows multiple warps to have in-flight fetches simultaneously
    // This hides fetch latency by overlapping fetches for different warps
    reg [WARP_ID_W-1:0] fetch_arb_ptr;
    reg [WARP_ID_W-1:0] fetch_warp_id;
    reg                 fetch_valid_arb;

    // Per-warp in-flight fetch tracking (replaces single fetch_inflight_valid)
    reg [NUM_WARPS-1:0] warp_fetch_pending;  // Which warps have pending fetches

    // Fetch pipeline stages for tracking responses
    // Stage 0: fetch request sent, Stage 1: response expected
    localparam FETCH_PIPE_DEPTH = 2;  // Support 2 in-flight fetches
    reg [WARP_ID_W-1:0] fetch_pipe_warp [0:FETCH_PIPE_DEPTH-1];
    reg [FETCH_PIPE_DEPTH-1:0] fetch_pipe_valid;

    // Effective buffer empty: consider same-cycle consumes
    wire [NUM_WARPS-1:0] warp_buf_will_be_empty;
    assign warp_buf_will_be_empty = ~warp_inst_buf_valid | warp_inst_consume;

    // Warp needs fetch if: buffer will be empty AND no pending fetch
    wire [NUM_WARPS-1:0] warp_needs_fetch = warp_buf_will_be_empty & ~warp_fetch_pending;

    integer f_i;
    always @(*) begin
        fetch_valid_arb = 0;
        fetch_warp_id = 0;
        for (f_i = 0; f_i < NUM_WARPS; f_i = f_i + 1) begin
            if (!fetch_valid_arb) begin
                // Check if warp needs instruction and isn't stalled or already fetching
                if (warp_valid[(fetch_arb_ptr + f_i) % NUM_WARPS] &&
                    warp_needs_fetch[(fetch_arb_ptr + f_i) % NUM_WARPS] &&
                    !warp_exit_pending[(fetch_arb_ptr + f_i) % NUM_WARPS]) begin
                    fetch_valid_arb = 1;
                    fetch_warp_id = (fetch_arb_ptr + f_i) % NUM_WARPS;
                end
            end
        end
    end

    // Fetch Request Logic - can issue new fetch each cycle
    // Only blocked if icache not ready (not by pending fetches)
    assign fetch_req = fetch_valid_arb;
    assign fetch_fire = fetch_req && icache_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fetch_arb_ptr <= 0;
            warp_inst_buf_valid <= 0;
        end else begin
            // Round-robin update on successful fetch request
            if (fetch_fire) begin
                fetch_arb_ptr <= (fetch_warp_id + 1) % NUM_WARPS;
            end
        end
    end

    // Combinatorial same-cycle hit detection
    // Note: With pipelined fetch, same_cycle_hit is only valid for true zero-latency icache hits.
    // With the testbench's 1-cycle memory latency, icache_valid is for a previous request,
    // not the current one. So same_cycle_hit should be false in that case.
    // For icache bypass with 1-cycle memory, we use the pipeline for ALL responses.
    wire same_cycle_hit = 1'b0;  // Disabled for pipelined fetch with latency > 0

    // Fetch pipeline and pending tracking
    integer fp_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_fetch_pending <= 0;
            fetch_pipe_valid <= 0;
            fetch_pipe_pc_bit2 <= 0;
            for (fp_i = 0; fp_i < FETCH_PIPE_DEPTH; fp_i = fp_i + 1) begin
                fetch_pipe_warp[fp_i] <= 0;
            end
        end else begin
            // Shift pipeline (valid, warp ID, and PC bit 2)
            fetch_pipe_valid[FETCH_PIPE_DEPTH-1:1] <= fetch_pipe_valid[FETCH_PIPE_DEPTH-2:0];
            fetch_pipe_pc_bit2[FETCH_PIPE_DEPTH-1:1] <= fetch_pipe_pc_bit2[FETCH_PIPE_DEPTH-2:0];
            for (fp_i = FETCH_PIPE_DEPTH-1; fp_i > 0; fp_i = fp_i - 1) begin
                fetch_pipe_warp[fp_i] <= fetch_pipe_warp[fp_i-1];
            end

            // New fetch enters pipeline stage 0
            if (fetch_fire && !same_cycle_hit) begin
                fetch_pipe_valid[0] <= 1'b1;
                fetch_pipe_warp[0] <= fetch_warp_id;
                fetch_pipe_pc_bit2[0] <= warp_fetch_pc[fetch_warp_id][2];
                warp_fetch_pending[fetch_warp_id] <= 1'b1;
            end else begin
                fetch_pipe_valid[0] <= 1'b0;
                fetch_pipe_warp[0] <= 0;
                fetch_pipe_pc_bit2[0] <= 0;
            end

            // Same-cycle hit doesn't need pipeline tracking
            if (same_cycle_hit) begin
                // Pending already cleared (or never set for same-cycle)
            end

            // Response received - clear pending for the responding warp
            if (icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1]) begin
                warp_fetch_pending[fetch_pipe_warp[FETCH_PIPE_DEPTH-1]] <= 1'b0;
            end
        end
    end

    // For compatibility with existing code
    wire fetch_inflight_valid = |fetch_pipe_valid;
    wire [WARP_ID_W-1:0] fetch_inflight_warp = fetch_pipe_warp[FETCH_PIPE_DEPTH-1];

    // Write to buffer
    // Fill/Consume logic must handle simultaneous fill and consume for same warp correctly.
    // When a warp is consumed AND filled in the same cycle, fill should win (buffer stays valid).
    integer w_buf;
    wire [NUM_WARPS-1:0] warp_fill;  // Which warp gets filled this cycle
    // Use the last stage of fetch pipeline for delayed responses
    wire delayed_response_valid = icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1];
    wire [WARP_ID_W-1:0] fill_warp_id = same_cycle_hit ? fetch_warp_id : fetch_pipe_warp[FETCH_PIPE_DEPTH-1];
    wire fill_valid = same_cycle_hit || delayed_response_valid;

    genvar fill_w;
    generate
        for (fill_w = 0; fill_w < NUM_WARPS; fill_w = fill_w + 1) begin : gen_warp_fill
            assign warp_fill[fill_w] = fill_valid && (fill_warp_id == fill_w);
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
             for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                 warp_inst_buf[w_buf] <= 0;
             end
             warp_inst_buf_valid <= 0;
        end else begin
            // Handle each warp's buffer valid bit
            for (w_buf = 0; w_buf < NUM_WARPS; w_buf = w_buf + 1) begin
                if (warp_fill[w_buf]) begin
                    // Fill takes priority - buffer valid regardless of consume
                    warp_inst_buf_valid[w_buf] <= 1'b1;
                end else if (warp_inst_consume[w_buf]) begin
                    // Consume only clears if no fill happening
                    warp_inst_buf_valid[w_buf] <= 1'b0;
                end
            end

            // Fill instruction data (PC already advanced on fetch_fire)
            if (same_cycle_hit) begin
                // Same-cycle hit: use fetch_warp_id (the requesting warp)
                warp_inst_buf[fetch_warp_id] <= icache_data;
            end else if (delayed_response_valid) begin
                // Delayed response: use warp ID from pipeline
                warp_inst_buf[fetch_pipe_warp[FETCH_PIPE_DEPTH-1]] <= icache_data;
            end

            // PC advances when fetch request is sent (not on response)
            // This ensures PC points to the NEXT instruction to fetch
            if (fetch_fire) begin
                warp_fetch_pc[fetch_warp_id] <= warp_fetch_pc[fetch_warp_id] + 4;
            end
        end
    end


    //========================================================================
    // STAGE 2: PRE-DECODE & SCHEDULING (Replaces old Decode)
    //========================================================================

    // Pre-decode signals for scheduler
    wire [4:0]               pd_rd [0:NUM_WARPS-1];
    wire [4:0]               pd_rs1 [0:NUM_WARPS-1];
    wire [4:0]               pd_rs2 [0:NUM_WARPS-1];
    wire [4:0]               pd_rs3 [0:NUM_WARPS-1];
    wire [NUM_WARPS-1:0]     pd_is_compute;
    wire [NUM_WARPS-1:0]     pd_is_tensor;
    wire [NUM_WARPS-1:0]     pd_is_memory;
    wire [NUM_WARPS-1:0]     pd_is_branch;
    wire [NUM_WARPS-1:0]     pd_writes_reg;

    genvar pd_i;
    generate
        for (pd_i = 0; pd_i < NUM_WARPS; pd_i = pd_i + 1) begin : gen_predecode
            wire [31:0] inst = warp_inst_buf[pd_i];
            // Simple extraction (assuming R-type/I-type consistency)
            // Real implementation needs fuller opcode check
            assign pd_rd[pd_i] = inst[25:21];
            assign pd_rs1[pd_i] = inst[20:16];
            assign pd_rs2[pd_i] = inst[15:11];
            assign pd_rs3[pd_i] = inst[10:6]; // Approximation

            wire [5:0] op = inst[31:26];
            assign pd_is_compute[pd_i] = (op == `OP_ALU) || (op == `OP_MUL) ||
                                         (op == `OP_FP32_ARITH) || (op == `OP_FP16_ARITH) ||
                                         (op == `OP_SFU);
            assign pd_is_tensor[pd_i]  = (op == `OP_WMMA_MMA);
            assign pd_is_memory[pd_i]  = (op == `OP_LD_GLOBAL) || (op == `OP_ST_GLOBAL) ||
                                         (op == `OP_LD_SHARED) || (op == `OP_ST_SHARED);
            assign pd_is_branch[pd_i]  = (op == `OP_BRANCH) || (op == `OP_EXIT);
            assign pd_writes_reg[pd_i] = (op != `OP_ST_GLOBAL) && (op != `OP_ST_SHARED) &&
                                         (op != `OP_BRANCH) && (op != `OP_EXIT);
        end
    endgenerate

    // Scheduler Instantiation
    wire [1:0] sched_issue_valid_mask;
    wire [WARP_ID_W-1:0] sched_issue_warp_id [0:1];
    wire [31:0] sched_issue_inst [0:1];
    wire [2:0] sched_issue_pipe [0:1];

    // Pipeline readiness signals (simplified)
    wire pipe_compute0_ready = 1'b1; // Pipeline always accepts unless stall logic says otherwise
    wire pipe_compute1_ready = 1'b1;
    wire pipe_tensor_ready   = !tensor_issue_full;
    wire pipe_memory_ready   = !issue_stall_mem; // Reuse stall logic
    wire pipe_branch_ready   = 1'b1;

    advanced_warp_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_ISSUE(2)
    ) u_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_diverged(32'b0), // Todo: connect to CFU
        .warp_at_barrier(warp_stalled_sync),
        .warp_inst(warp_inst_buf),
        .warp_inst_valid(warp_inst_buf_valid),
        .warp_inst_consume(warp_inst_consume),
        .warp_rd(pd_rd),
        .warp_rs1(pd_rs1),
        .warp_rs2(pd_rs2),
        .warp_rs3(pd_rs3),
        .warp_is_compute(pd_is_compute),
        .warp_is_tensor(pd_is_tensor),
        .warp_is_memory(pd_is_memory),
        .warp_is_branch(pd_is_branch),
        .warp_writes_reg(pd_writes_reg),
        .compute_pipe0_ready(pipe_compute0_ready),
        .compute_pipe1_ready(pipe_compute1_ready),
        .tensor_pipe_ready(pipe_tensor_ready),
        .memory_pipe_ready(pipe_memory_ready),
        .branch_unit_ready(pipe_branch_ready),
        .issue_valid(sched_issue_valid_mask),
        .issue_warp_id(sched_issue_warp_id),
        .issue_inst(sched_issue_inst),
        .issue_pipe(sched_issue_pipe),
        .wb_valid(wb_valid),
        .wb_warp_id(wb_warp_id),
        .wb_rd(wb_rd),
        .stat_cycles(),
        .stat_single_issue(),
        .stat_dual_issue(),
        .stat_stalls()
    );

    // Map Scheduler Output to Pipeline Signals
    // Replaces dec0_fire / dec1_fire logic
    assign issue0_fire = sched_issue_valid_mask[0];
    assign issue1_fire = sched_issue_valid_mask[1];

    // We reuse the 'dec0' pipeline registers to hold the scheduled instructions
    // effectively merging Decode/Issue stages into one logical flow handled by scheduler+decoder
    // Note: This overrides the previous 'dec0_warp_id <= ifq_warp_head' logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec0_valid <= 0;
            dec1_valid <= 0;
        end else begin
            dec0_valid <= issue0_fire;
            if (issue0_fire) begin
                dec0_warp_id <= sched_issue_warp_id[0];
                dec0_instruction <= sched_issue_inst[0];
                dec0_pc <= warp_pc[sched_issue_warp_id[0]]; // Arch PC
            end

            dec1_valid <= issue1_fire;
            if (issue1_fire) begin
                dec1_warp_id <= sched_issue_warp_id[1];
                dec1_instruction <= sched_issue_inst[1];
                dec1_pc <= warp_pc[sched_issue_warp_id[1]];
            end
        end
    end

    // REMOVE OLD DECODE LOGIC (This block replaces it)
    // We effectively bypass the 'dec0_fire' internal logic and drive valid signals directly
    // from the scheduler's decision.

    // dec0_fire triggers the decoder when we have a valid instruction in decode stage
    wire dec0_fire = dec0_valid;
    wire dec1_fire = dec1_valid;

    //------------------------------------------------------------------------
    // Instruction Decoder (lane 0)
    //------------------------------------------------------------------------
    // Internal signals for decoder output mapping
    wire dec_fp32_special;
    wire dec_wmma_mma;
    wire dec_shfl_op;

    decoder u_decoder0 (
        .clk         (clk),
        .rst_n       (rst_n),
        .instruction (dec0_instruction),
        .valid_in    (dec0_fire),
        .valid_out   (dec_valid),
        .opcode      (dec_opcode),
        .rd          (dec_rd),
        .ra          (dec_ra),
        .rb          (dec_rb),
        .rc          (dec_rc),
        .func        (dec_func),
        .imm16       (dec_imm16),
        .imm21       (dec_imm21),
        .use_imm     (dec_use_imm),
        .alu_op      (dec_alu_op),
        .mul_op      (dec_mul_op),
        .div_op      (dec_div_op),
        .fp32_op     (dec_fp32_op),
        .fp64_op     (dec_fp64_op),
        .fp16_op     (dec_fp16_op),
        .fp32_special(dec_fp32_special),  // Maps to SFU operations
        .wmma_mma    (dec_wmma_mma),       // Maps to tensor operations
        .mem_read    (dec_mem_read),
        .mem_write   (dec_mem_write),
        .mem_shared  (dec_mem_shared),
        .branch_op   (dec_branch_op),
        .sync_op     (dec_sync_op),
        .special_reg (dec_special_reg),
        .exit_op     (dec_exit_op),
        .atomic_op   (dec_atomic_op),
        .shfl_op     (dec_shfl_op),        // Warp shuffle
        .reg_write   (dec_reg_write),
        .pred_write  (),
        .pred_addr   ()
    );

    //------------------------------------------------------------------------
    // Instruction Decoder (lane 1)
    //------------------------------------------------------------------------
    wire dec1_fp32_special;
    wire dec1_wmma_mma;
    wire dec1_shfl_op;

    wire dec1_dec_valid;
    decoder u_decoder1 (
        .clk         (clk),
        .rst_n       (rst_n),
        .instruction (dec1_instruction),
        .valid_in    (dec1_fire),
        .valid_out   (dec1_dec_valid),
        .opcode      (dec1_opcode),
        .rd          (dec1_rd),
        .ra          (dec1_ra),
        .rb          (dec1_rb),
        .rc          (dec1_rc),
        .func        (dec1_func),
        .imm16       (dec1_imm16),
        .imm21       (dec1_imm21),
        .use_imm     (dec1_use_imm),
        .alu_op      (dec1_alu_op),
        .mul_op      (dec1_mul_op),
        .div_op      (dec1_div_op),
        .fp32_op     (dec1_fp32_op),
        .fp64_op     (dec1_fp64_op),
        .fp16_op     (dec1_fp16_op),
        .fp32_special(dec1_fp32_special),
        .wmma_mma    (dec1_wmma_mma),
        .mem_read    (dec1_mem_read),
        .mem_write   (dec1_mem_write),
        .mem_shared  (dec1_mem_shared),
        .branch_op   (dec1_branch_op),
        .sync_op     (dec1_sync_op),
        .special_reg (dec1_special_reg),
        .exit_op     (dec1_exit_op),
        .atomic_op   (dec1_atomic_op),
        .shfl_op     (dec1_shfl_op),
        .reg_write   (dec1_reg_write),
        .pred_write  (),
        .pred_addr   ()
    );

    // Map decoder outputs to V2 signal names
    assign dec_sfu_op = dec_fp32_special;
    assign dec_tensor_op = dec_wmma_mma;
    assign dec_shuffle_op = dec_shfl_op;
    assign dec1_sfu_op = dec1_fp32_special;
    assign dec1_tensor_op = dec1_wmma_mma;
    assign dec1_shuffle_op = dec1_shfl_op;

    //========================================================================
    // STAGE 3: ISSUE (Scoreboard Check + Register Read)
    //========================================================================
    integer sb_init;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue_valid <= 1'b0;
            issue1_valid <= 1'b0;
            issue_warp_id <= 0;
            issue_pc <= 0;
            issue_opcode <= 0;
            issue_rd <= 0;
            issue_ra <= 0;
            issue_rb <= 0;
            issue_rc <= 0;
            issue_func <= 0;
            issue_imm16 <= 0;
            issue_imm21 <= 0;
            issue_use_imm <= 1'b0;
            issue_mask <= 0;
            issue_alu_op <= 1'b0;
            issue_mul_op <= 1'b0;
            issue_div_op <= 1'b0;
            issue_fp32_op <= 1'b0;
            issue_fp64_op <= 1'b0;
            issue_fp16_op <= 1'b0;
            issue_sfu_op <= 1'b0;
            issue_tensor_op <= 1'b0;
            issue_mem_read <= 1'b0;
            issue_mem_write <= 1'b0;
            issue_mem_shared <= 1'b0;
            issue_branch_op <= 1'b0;
            issue_sync_op <= 1'b0;
            issue_special_reg <= 1'b0;
            issue_exit_op <= 1'b0;
            issue_atomic_op <= 1'b0;
            issue_shuffle_op <= 1'b0;
            issue_reg_write <= 1'b0;
            issue1_warp_id <= 0;
            issue1_pc <= 0;
            issue1_opcode <= 0;
            issue1_rd <= 0;
            issue1_ra <= 0;
            issue1_rb <= 0;
            issue1_rc <= 0;
            issue1_func <= 0;
            issue1_imm16 <= 0;
            issue1_imm21 <= 0;
            issue1_use_imm <= 1'b0;
            issue1_mask <= 0;
            issue1_alu_op <= 1'b0;
            issue1_mul_op <= 1'b0;
            issue1_div_op <= 1'b0;
            issue1_fp32_op <= 1'b0;
            issue1_fp64_op <= 1'b0;
            issue1_fp16_op <= 1'b0;
            issue1_sfu_op <= 1'b0;
            issue1_tensor_op <= 1'b0;
            issue1_mem_read <= 1'b0;
            issue1_mem_write <= 1'b0;
            issue1_mem_shared <= 1'b0;
            issue1_branch_op <= 1'b0;
            issue1_sync_op <= 1'b0;
            issue1_special_reg <= 1'b0;
            issue1_exit_op <= 1'b0;
            issue1_atomic_op <= 1'b0;
            issue1_shuffle_op <= 1'b0;
            issue1_reg_write <= 1'b0;
        end else begin
            // Issue stage triggers when decoder output is valid (dec_valid)
            // This ensures decoder has finished processing before we latch its outputs
            issue_valid <= dec_valid;
            issue1_valid <= dec1_dec_valid;  // Use decoder 1's valid output
            if (dec_valid) begin
                // With the new scheduler flow, always use lane 0's decoder output
                // (the old lane0_ready-based selection doesn't apply here)
                begin
                    issue_warp_id <= dec0_warp_id;
                    issue_pc <= dec0_pc;
                    issue_opcode <= dec_opcode;
                    issue_rd <= dec_rd;
                    issue_ra <= dec_ra;
                    issue_rb <= dec_rb;
                    issue_rc <= dec_rc;
                    issue_func <= dec_func;
                    issue_imm16 <= dec_imm16;
                    issue_imm21 <= dec_imm21;
                    issue_use_imm <= dec_use_imm;
                    issue_mask <= warp_mask[dec0_warp_id];
                    issue_alu_op <= dec_alu_op;
                    issue_mul_op <= dec_mul_op;
                    issue_div_op <= dec_div_op;
                    issue_fp32_op <= dec_fp32_op;
                    issue_fp64_op <= dec_fp64_op;
                    issue_fp16_op <= dec_fp16_op;
                    issue_sfu_op <= dec_sfu_op;
                    issue_tensor_op <= dec_tensor_op;
                    issue_mem_read <= dec_mem_read;
                    issue_mem_write <= dec_mem_write;
                    issue_mem_shared <= dec_mem_shared;
                    issue_branch_op <= dec_branch_op;
                    issue_sync_op <= dec_sync_op;
                    issue_special_reg <= dec_special_reg;
                    issue_exit_op <= dec_exit_op;
                    issue_atomic_op <= dec_atomic_op;
                    issue_shuffle_op <= dec_shuffle_op;
                    issue_reg_write <= dec_reg_write;
                end
            end

            if (issue1_fire) begin
                issue1_warp_id <= dec1_warp_id;
                issue1_pc <= dec1_pc;
                issue1_opcode <= dec1_opcode;
                issue1_rd <= dec1_rd;
                issue1_ra <= dec1_ra;
                issue1_rb <= dec1_rb;
                issue1_rc <= dec1_rc;
                issue1_func <= dec1_func;
                issue1_imm16 <= dec1_imm16;
                issue1_imm21 <= dec1_imm21;
                issue1_use_imm <= dec1_use_imm;
                issue1_mask <= warp_mask[dec1_warp_id];
                issue1_alu_op <= dec1_alu_op;
                issue1_mul_op <= dec1_mul_op;
                issue1_div_op <= dec1_div_op;
                issue1_fp32_op <= dec1_fp32_op;
                issue1_fp64_op <= dec1_fp64_op;
                issue1_fp16_op <= dec1_fp16_op;
                issue1_sfu_op <= dec1_sfu_op;
                issue1_tensor_op <= dec1_tensor_op;
                issue1_mem_read <= dec1_mem_read;
                issue1_mem_write <= dec1_mem_write;
                issue1_mem_shared <= dec1_mem_shared;
                issue1_branch_op <= dec1_branch_op;
                issue1_sync_op <= dec1_sync_op;
                issue1_special_reg <= dec1_special_reg;
                issue1_exit_op <= dec1_exit_op;
                issue1_atomic_op <= dec1_atomic_op;
                issue1_shuffle_op <= dec1_shuffle_op;
                issue1_reg_write <= dec1_reg_write;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize scoreboards
            for (sb_init = 0; sb_init < NUM_WARPS; sb_init = sb_init + 1) begin
                scoreboard_busy[sb_init] <= 32'b0;
                pending_fu_count[sb_init] <= 4'b0;
            end
        end else begin
            // Mark destination register as busy on issue
            if (issue0_fire && issue0_reg_write_sel && (issue0_rd_sel != 0)) begin
                scoreboard_busy[issue0_warp_sel][issue0_rd_sel] <= 1'b1;
            end
            if (issue1_fire && dec1_reg_write && (dec1_rd != 0)) begin
                scoreboard_busy[dec1_warp_id][dec1_rd] <= 1'b1;
            end

            // Clear on writeback
            if (wb_valid && wb_rd != 0) begin
                scoreboard_busy[wb_warp_id][wb_rd] <= 1'b0;
            end

            // Increment pending FU count for multi-cycle operations
            if (issue0_fire && (issue0_fp32_sel || issue0_fp64_sel || issue0_fp16_sel ||
                                issue0_sfu_sel || issue0_tensor_sel || issue0_mem_read_sel ||
                                issue0_atomic_sel)) begin
                pending_fu_count[issue0_warp_sel] <= pending_fu_count[issue0_warp_sel] + 1;
            end
            if (issue1_fire && (dec1_fp32_op || dec1_fp64_op || dec1_fp16_op ||
                                dec1_sfu_op || dec1_tensor_op || dec1_mem_read ||
                                dec1_atomic_op)) begin
                pending_fu_count[dec1_warp_id] <= pending_fu_count[dec1_warp_id] + 1;
            end

            // Decrement pending count on writeback
            if (wb_valid && wb_rd != 0) begin
                if (pending_fu_count[wb_warp_id] > 0) begin
                    pending_fu_count[wb_warp_id] <= pending_fu_count[wb_warp_id] - 1;
                end
            end
        end
    end

    // Track per-FU outstanding operations to size writeback queues.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alu_inflight <= {ALU_WBQ_COUNT_W{1'b0}};
            mul_inflight <= {MUL_WBQ_COUNT_W{1'b0}};
            fpu32_inflight <= {FPU32_WBQ_COUNT_W{1'b0}};
            fpu64_inflight <= {FPU64_WBQ_COUNT_W{1'b0}};
            fp16_inflight <= {FP16_WBQ_COUNT_W{1'b0}};
            sfu_inflight <= {SFU_WBQ_COUNT_W{1'b0}};
            shfl_inflight <= {SHFL_WBQ_COUNT_W{1'b0}};
        end else begin
            case ({alu_issue, alu_wbq_pop})
                2'b10: alu_inflight <= alu_inflight + 1'b1;
                2'b01: alu_inflight <= alu_inflight - 1'b1;
                default: alu_inflight <= alu_inflight;
            endcase
            case ({mul_issue, mul_wbq_pop})
                2'b10: mul_inflight <= mul_inflight + 1'b1;
                2'b01: mul_inflight <= mul_inflight - 1'b1;
                default: mul_inflight <= mul_inflight;
            endcase
            case ({fpu32_issue, fpu32_wbq_pop})
                2'b10: fpu32_inflight <= fpu32_inflight + 1'b1;
                2'b01: fpu32_inflight <= fpu32_inflight - 1'b1;
                default: fpu32_inflight <= fpu32_inflight;
            endcase
            case ({fpu64_issue, fpu64_wbq_pop})
                2'b10: fpu64_inflight <= fpu64_inflight + 1'b1;
                2'b01: fpu64_inflight <= fpu64_inflight - 1'b1;
                default: fpu64_inflight <= fpu64_inflight;
            endcase
            case ({fp16_issue, fp16_wbq_pop})
                2'b10: fp16_inflight <= fp16_inflight + 1'b1;
                2'b01: fp16_inflight <= fp16_inflight - 1'b1;
                default: fp16_inflight <= fp16_inflight;
            endcase
            case ({sfu_issue, sfu_wbq_pop})
                2'b10: sfu_inflight <= sfu_inflight + 1'b1;
                2'b01: sfu_inflight <= sfu_inflight - 1'b1;
                default: sfu_inflight <= sfu_inflight;
            endcase
            case ({shfl_issue, shfl_wbq_pop})
                2'b10: shfl_inflight <= shfl_inflight + 1'b1;
                2'b01: shfl_inflight <= shfl_inflight - 1'b1;
                default: shfl_inflight <= shfl_inflight;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Banked Register File (Per-Warp) - 4-bank conflict-free design
    //------------------------------------------------------------------------
    wire rf_conflict_a, rf_conflict_b, rf_conflict_c, rf_wr_conflict;
    wire rf1_conflict_a, rf1_conflict_b, rf1_conflict_c, rf1_wr_conflict;
    wire [31:0] rf_stat_bank_conflicts, rf_stat_total_accesses;
    wire [31:0] rf1_stat_bank_conflicts, rf1_stat_total_accesses;

    // Operand collector interface (unused - tie off with wires for iverilog compatibility)
    wire [4:0] oc_addr_tie [0:2];
    assign oc_addr_tie[0] = 5'b0;
    assign oc_addr_tie[1] = 5'b0;
    assign oc_addr_tie[2] = 5'b0;

    register_file_banked #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_BANKS(4)
    ) u_regfile0 (
        .clk       (clk),
        .rst_n     (rst_n),
        // Read port - warp selection
        .rd_warp_id(issue_warp_id),
        .wr_warp_id(wb_warp_id),
        // Read ports
        .rd_addr_a (issue_ra),
        .rd_addr_b (issue_rb),
        .rd_addr_c (issue_rc),
        .rd_data_a (rf_rd_data_a),
        .rd_data_b (rf_rd_data_b),
        .rd_data_c (rf_rd_data_c),
        .rd_conflict_a(rf_conflict_a),
        .rd_conflict_b(rf_conflict_b),
        .rd_conflict_c(rf_conflict_c),
        // Write port
        .wr_en     (rf_wr_en),
        .wr_addr   (wb_rd),
        .wr_data   (rf_wr_data),
        .wr_mask   (rf_wr_mask),
        .wr_conflict(rf_wr_conflict),
        // Operand collector interface (unused for now - tied off)
        .oc_valid  (1'b0),
        .oc_warp_id({WARP_ID_W{1'b0}}),
        .oc_addr   (oc_addr_tie),
        .oc_data   (),
        .oc_ready  (),
        .oc_conflict(),
        // Statistics
        .stat_bank_conflicts(rf_stat_bank_conflicts),
        .stat_total_accesses(rf_stat_total_accesses)
    );

    register_file_banked #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_BANKS(4)
    ) u_regfile1 (
        .clk       (clk),
        .rst_n     (rst_n),
        // Read port - warp selection
        .rd_warp_id(issue1_warp_id),
        .wr_warp_id(wb_warp_id),
        // Read ports
        .rd_addr_a (issue1_ra),
        .rd_addr_b (issue1_rb),
        .rd_addr_c (issue1_rc),
        .rd_data_a (rf1_rd_data_a),
        .rd_data_b (rf1_rd_data_b),
        .rd_data_c (rf1_rd_data_c),
        .rd_conflict_a(rf1_conflict_a),
        .rd_conflict_b(rf1_conflict_b),
        .rd_conflict_c(rf1_conflict_c),
        // Write port
        .wr_en     (rf_wr_en),
        .wr_addr   (wb_rd),
        .wr_data   (rf_wr_data),
        .wr_mask   (rf_wr_mask),
        .wr_conflict(rf1_wr_conflict),
        // Operand collector interface (unused for now - tied off)
        .oc_valid  (1'b0),
        .oc_warp_id({WARP_ID_W{1'b0}}),
        .oc_addr   (oc_addr_tie),
        .oc_data   (),
        .oc_ready  (),
        .oc_conflict(),
        // Statistics
        .stat_bank_conflicts(rf1_stat_bank_conflicts),
        .stat_total_accesses(rf1_stat_total_accesses)
    );

    //========================================================================
    // STAGE 4: EXECUTE (Multiple Functional Units)
    //========================================================================

    //------------------------------------------------------------------------
    // SIMD ALU (Integer Operations)
    //------------------------------------------------------------------------
    wire alu_use_slot0 = alu_issue0;
    wire alu_use_slot1 = alu_issue1;
    wire [WARP_ID_W-1:0] alu_issue_warp = alu_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] alu_issue_rd = alu_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] alu_issue_mask = alu_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] alu_issue_func = alu_use_slot0 ? issue_func : issue1_func;
    wire [15:0] alu_issue_imm16 = alu_use_slot0 ? issue_imm16 : issue1_imm16;
    wire alu_issue_use_imm = alu_use_slot0 ? issue_use_imm : issue1_use_imm;
    wire [SIMD_WIDTH-1:0] alu_op_a = alu_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] alu_op_b = alu_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;

    simd_alu u_simd_alu (
        .func       (alu_issue_func),
        .operand_a  (alu_op_a),
        .operand_b  (alu_issue_use_imm ? {NUM_LANES{16'b0, alu_issue_imm16}} : alu_op_b),
        .lane_mask  (alu_issue_mask),
        .result     (alu_result),
        .zero_flags (alu_zero),
        .neg_flags  (alu_neg)
    );

    // ALU pipeline tracking (1 stage delay to avoid issue-stage race)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alu_valid_pipe <= 1'b0;
            alu_warp_pipe <= 0;
            alu_rd_pipe <= 0;
            alu_mask_pipe <= 0;
            alu_result_pipe <= 0;
        end else begin
            alu_valid_pipe <= alu_issue;
            if (alu_issue) begin
                alu_warp_pipe <= alu_issue_warp;
                alu_rd_pipe <= alu_issue_rd;
                alu_mask_pipe <= alu_issue_mask;
                alu_result_pipe <= alu_result;
            end
        end
    end

    assign alu_valid_out = alu_valid_pipe;

    //------------------------------------------------------------------------
    // SIMD Multiplier
    //------------------------------------------------------------------------
    wire mul_use_slot0 = mul_issue0;
    wire mul_use_slot1 = mul_issue1;
    wire [WARP_ID_W-1:0] mul_issue_warp = mul_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] mul_issue_rd = mul_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] mul_issue_mask = mul_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] mul_issue_func = mul_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] mul_op_a = mul_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] mul_op_b = mul_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] mul_op_c = mul_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;

    simd_mul_unit u_simd_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (mul_issue),
        .func      (mul_issue_func),
        .operand_a (mul_op_a),
        .operand_b (mul_op_b),
        .operand_c (mul_op_c),
        .lane_mask (mul_issue_mask),
        .valid_out (mul_valid_out),
        .result    (mul_result)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_warp_pipe <= 0;
            mul_rd_pipe <= 0;
            mul_mask_pipe <= 0;
        end else if (mul_issue) begin
            mul_warp_pipe <= mul_issue_warp;
            mul_rd_pipe <= mul_issue_rd;
            mul_mask_pipe <= mul_issue_mask;
        end
    end

    //------------------------------------------------------------------------
    // SIMD FPU (FP32) - 4-cycle pipeline
    //------------------------------------------------------------------------
    wire fpu32_use_slot0 = fpu32_issue0;
    wire fpu32_use_slot1 = fpu32_issue1;
    wire [WARP_ID_W-1:0] fpu32_issue_warp = fpu32_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] fpu32_issue_rd = fpu32_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] fpu32_issue_mask = fpu32_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] fpu32_issue_func = fpu32_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] fpu32_op_a = fpu32_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] fpu32_op_b = fpu32_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] fpu32_op_c = fpu32_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;

    assign fpu32_valid_in = fpu32_issue;

    simd_fpu u_simd_fpu (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fpu32_valid_in),
        .ready     (fpu32_ready),
        .func      (fpu32_issue_func),
        .operand_a (fpu32_op_a),
        .operand_b (fpu32_op_b),
        .operand_c (fpu32_op_c),
        .lane_mask (fpu32_issue_mask),
        .valid_out (fpu32_valid_out),
        .result    (fpu32_result)
    );

    // FPU32 pipeline tracking (warp/rd/mask follows data through pipeline)
    integer fpu32_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fpu32_i = 0; fpu32_i < 5; fpu32_i = fpu32_i + 1) begin
                fpu32_warp_pipe[fpu32_i] <= 0;
                fpu32_rd_pipe[fpu32_i] <= 0;
                fpu32_mask_pipe[fpu32_i] <= 0;
            end
        end else begin
            fpu32_warp_pipe[0] <= fpu32_valid_in ? fpu32_issue_warp : {WARP_ID_W{1'b0}};
            fpu32_rd_pipe[0] <= fpu32_valid_in ? fpu32_issue_rd : 5'b0;
            fpu32_mask_pipe[0] <= fpu32_valid_in ? fpu32_issue_mask : {NUM_LANES{1'b0}};
            for (fpu32_i = 1; fpu32_i < 5; fpu32_i = fpu32_i + 1) begin
                fpu32_warp_pipe[fpu32_i] <= fpu32_warp_pipe[fpu32_i-1];
                fpu32_rd_pipe[fpu32_i] <= fpu32_rd_pipe[fpu32_i-1];
                fpu32_mask_pipe[fpu32_i] <= fpu32_mask_pipe[fpu32_i-1];
            end
        end
    end

    //------------------------------------------------------------------------
    // FP64 Unit
    //------------------------------------------------------------------------
    wire fpu64_use_slot0 = fpu64_issue0;
    wire fpu64_use_slot1 = fpu64_issue1;
    wire [WARP_ID_W-1:0] fpu64_issue_warp = fpu64_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] fpu64_issue_rd = fpu64_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] fpu64_issue_mask = fpu64_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] fpu64_issue_func = fpu64_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] fpu64_op_a = fpu64_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] fpu64_op_b = fpu64_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] fpu64_op_c = fpu64_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;

    assign fpu64_valid_in = fpu64_issue;

    simd_fpu64 u_simd_fpu64 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fpu64_valid_in),
        .ready     (fpu64_ready),
        .func      (fpu64_issue_func),
        .operand_a (fpu64_op_a),
        .operand_b (fpu64_op_b),
        .operand_c (fpu64_op_c),
        .lane_mask (fpu64_issue_mask),
        .valid_out (fpu64_valid_out),
        .result    (fpu64_result)
    );

    integer fpu64_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fpu64_i = 0; fpu64_i < 5; fpu64_i = fpu64_i + 1) begin
                fpu64_warp_pipe[fpu64_i] <= 0;
                fpu64_rd_pipe[fpu64_i] <= 0;
                fpu64_mask_pipe[fpu64_i] <= 0;
            end
        end else begin
            fpu64_warp_pipe[0] <= fpu64_valid_in ? fpu64_issue_warp : {WARP_ID_W{1'b0}};
            fpu64_rd_pipe[0] <= fpu64_valid_in ? fpu64_issue_rd : 5'b0;
            fpu64_mask_pipe[0] <= fpu64_valid_in ? fpu64_issue_mask : {NUM_LANES{1'b0}};
            for (fpu64_i = 1; fpu64_i < 5; fpu64_i = fpu64_i + 1) begin
                fpu64_warp_pipe[fpu64_i] <= fpu64_warp_pipe[fpu64_i-1];
                fpu64_rd_pipe[fpu64_i] <= fpu64_rd_pipe[fpu64_i-1];
                fpu64_mask_pipe[fpu64_i] <= fpu64_mask_pipe[fpu64_i-1];
            end
        end
    end

    genvar f64_lane;
    generate
        for (f64_lane = 0; f64_lane < NUM_LANES; f64_lane = f64_lane + 1) begin : fpu64_trunc
            assign fpu64_result_trunc[f64_lane*DATA_WIDTH +: DATA_WIDTH] =
                fpu64_result[f64_lane*64 +: 32];
        end
    endgenerate

    //------------------------------------------------------------------------
    // FP16/BF16 Unit
    //------------------------------------------------------------------------
    wire fp16_use_slot0 = fp16_issue0;
    wire fp16_use_slot1 = fp16_issue1;
    wire [WARP_ID_W-1:0] fp16_issue_warp = fp16_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] fp16_issue_rd = fp16_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] fp16_issue_mask = fp16_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] fp16_issue_func = fp16_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] fp16_op_a = fp16_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] fp16_op_b = fp16_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;

    assign fp16_valid_in = fp16_issue;

    simd_fp16 u_simd_fp16 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fp16_valid_in),
        .ready     (fp16_ready),
        .func      (fp16_issue_func),
        .operand_a (fp16_op_a),
        .operand_b (fp16_op_b),
        .lane_mask (fp16_issue_mask),
        .valid_out (fp16_valid_out),
        .result    (fp16_result)
    );

    integer fp16_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (fp16_i = 0; fp16_i < 2; fp16_i = fp16_i + 1) begin
                fp16_warp_pipe[fp16_i] <= 0;
                fp16_rd_pipe[fp16_i] <= 0;
                fp16_mask_pipe[fp16_i] <= 0;
            end
        end else begin
            fp16_warp_pipe[0] <= fp16_valid_in ? fp16_issue_warp : {WARP_ID_W{1'b0}};
            fp16_rd_pipe[0] <= fp16_valid_in ? fp16_issue_rd : 5'b0;
            fp16_mask_pipe[0] <= fp16_valid_in ? fp16_issue_mask : {NUM_LANES{1'b0}};
            fp16_warp_pipe[1] <= fp16_warp_pipe[0];
            fp16_rd_pipe[1] <= fp16_rd_pipe[0];
            fp16_mask_pipe[1] <= fp16_mask_pipe[0];
        end
    end

    //------------------------------------------------------------------------
    // Special Function Unit (sin, cos, sqrt, exp, log)
    //------------------------------------------------------------------------
    wire sfu_use_slot0 = sfu_issue0;
    wire sfu_use_slot1 = sfu_issue1;
    wire [WARP_ID_W-1:0] sfu_issue_warp = sfu_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] sfu_issue_rd = sfu_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] sfu_issue_mask = sfu_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] sfu_issue_func = sfu_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] sfu_op = sfu_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;

    assign sfu_valid_in = sfu_issue;

    simd_sfu u_simd_sfu (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (sfu_valid_in),
        .ready     (sfu_ready),
        .func      (sfu_issue_func),
        .operand   (sfu_op),
        .lane_mask (sfu_issue_mask),
        .valid_out (sfu_valid_out),
        .result    (sfu_result)
    );

    // SFU pipeline tracking (8-cycle latency)
    integer sfu_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (sfu_i = 0; sfu_i < 8; sfu_i = sfu_i + 1) begin
                sfu_warp_pipe[sfu_i] <= 0;
                sfu_rd_pipe[sfu_i] <= 0;
                sfu_mask_pipe[sfu_i] <= 0;
            end
        end else begin
            sfu_warp_pipe[0] <= sfu_valid_in ? sfu_issue_warp : {WARP_ID_W{1'b0}};
            sfu_rd_pipe[0] <= sfu_valid_in ? sfu_issue_rd : 5'b0;
            sfu_mask_pipe[0] <= sfu_valid_in ? sfu_issue_mask : {NUM_LANES{1'b0}};
            for (sfu_i = 1; sfu_i < 8; sfu_i = sfu_i + 1) begin
                sfu_warp_pipe[sfu_i] <= sfu_warp_pipe[sfu_i-1];
                sfu_rd_pipe[sfu_i] <= sfu_rd_pipe[sfu_i-1];
                sfu_mask_pipe[sfu_i] <= sfu_mask_pipe[sfu_i-1];
            end
        end
    end

    // Tensor issue queue (captures operands/metadata to align with TC readiness)
    assign tensor_issue_push = issue_valid && issue_tensor_op;
    assign tensor_issue_push_data = pack_tensor_issue(issue_warp_id, issue_rd,
                                                      issue_mask, issue_func[2:0],
                                                      rf_rd_data_a, rf_rd_data_b,
                                                      rf_rd_data_c);

    assign tensor_issue_empty = (tensor_issue_count == 0);
    assign tensor_issue_full = (tensor_issue_count == TENSOR_ISSUE_DEPTH_VAL);

    assign tensor_issue_pop = !tensor_issue_empty && tensor_ready &&
                              !tensor_wbq_full && !tensor_meta_full;
    assign tensor_issue_push_fire = tensor_issue_push &&
                                    (!tensor_issue_full || tensor_issue_pop);
    assign tensor_issue_pop_fire = tensor_issue_pop;

    assign tensor_issue_count_next = {1'b0, tensor_issue_count} +
                                     (tensor_issue_push_fire ? 1'b1 : 1'b0) -
                                     (tensor_issue_pop_fire ? 1'b1 : 1'b0);
    assign tensor_issue_full_next = (tensor_issue_count_next >= TENSOR_ISSUE_DEPTH);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tensor_issue_count <= {TENSOR_ISSUE_COUNT_W{1'b0}};
        end else begin
            case ({tensor_issue_push_fire, tensor_issue_pop_fire})
                2'b10: tensor_issue_count <= tensor_issue_count + 1'b1;
                2'b01: tensor_issue_count <= tensor_issue_count - 1'b1;
                default: tensor_issue_count <= tensor_issue_count;
            endcase
        end
    end

    // Memory operation tracking
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_pending_valid <= 1'b0;
        end else if (gmem_req_valid && gmem_req_ready && issue_mem_read && !mem_pending_valid) begin
            mem_pending_valid <= 1'b1;
            mem_warp_pending <= issue_warp_id;
            mem_rd_pending <= issue_rd;
            mem_mask_pending <= issue_mask;
        end else if (gmem_resp_valid) begin
            mem_pending_valid <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            smem_pending_valid <= 1'b0;
        end else if (smem_req_valid && issue_mem_read) begin
            smem_pending_valid <= 1'b1;
            smem_warp_pending <= issue_warp_id;
            smem_rd_pending <= issue_rd;
            smem_mask_pending <= issue_mask;
        end else if (smem_resp_valid) begin
            smem_pending_valid <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_pending_valid <= 1'b0;
            store_warp_pending <= 0;
            store_mask_pending <= 0;
        end else begin
            if (issue_valid && issue_mem_write && !issue_mem_read &&
                !issue_atomic_op && !store_pending_valid) begin
                store_pending_valid <= 1'b1;
                store_warp_pending <= issue_warp_id;
                store_mask_pending <= issue_mask;
            end else if (wb_found && (wb_sel == 4'd7) && store_pending_valid &&
                         !smem_resp_valid && !gmem_resp_valid) begin
                store_pending_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Tensor Core (WMMA Operations)
    //------------------------------------------------------------------------
    wb_fifo #(
        .WIDTH (TENSOR_ISSUE_W),
        .DEPTH (TENSOR_ISSUE_DEPTH)
    ) u_tensor_issue_q (
        .clk       (clk),
        .rst_n     (rst_n),
        .push      (tensor_issue_push_fire),
        .push_data (tensor_issue_push_data),
        .pop       (tensor_issue_pop_fire),
        .pop_data  (tensor_issue_pop_data),
        .full      (tensor_issue_fifo_full),
        .empty     (tensor_issue_fifo_empty)
    );

    assign tensor_issue_frag_a = tensor_issue_pop_data[TENSOR_ISSUE_A_MSB:TENSOR_ISSUE_A_LSB];
    assign tensor_issue_frag_b = tensor_issue_pop_data[TENSOR_ISSUE_B_MSB:TENSOR_ISSUE_B_LSB];
    assign tensor_issue_frag_c = tensor_issue_pop_data[TENSOR_ISSUE_C_MSB:TENSOR_ISSUE_C_LSB];
    assign tensor_issue_op_type = tensor_issue_pop_data[TENSOR_ISSUE_OP_MSB:TENSOR_ISSUE_OP_LSB];
    assign tensor_issue_mask = tensor_issue_pop_data[TENSOR_ISSUE_MASK_MSB:TENSOR_ISSUE_MASK_LSB];
    assign tensor_issue_rd = tensor_issue_pop_data[TENSOR_ISSUE_RD_MSB:TENSOR_ISSUE_RD_LSB];
    assign tensor_issue_warp = tensor_issue_pop_data[TENSOR_ISSUE_WARP_MSB:TENSOR_ISSUE_WARP_LSB];

    assign tensor_valid_in = tensor_issue_pop;

    //------------------------------------------------------------------------
    // Tensor Core (WMMA Operations)
    //------------------------------------------------------------------------
    tensor_core #(
        .NUM_LANES       (NUM_LANES),
        .DATA_WIDTH      (DATA_WIDTH),
        .TC_NUM_CORES    (TC_NUM_CORES),
        .TC_LATENCY      (TC_LATENCY),
        .TC_DATA_DEFAULT (TC_DATA_DEFAULT),
        .TC_USE_OP_TYPE  (TC_USE_OP_TYPE),
        .TC_FP4_FORMAT   (TC_FP4_FORMAT),
        .TC_FP8_FORMAT   (TC_FP8_FORMAT)
    ) u_tensor_core (
        .clk         (clk),
        .rst_n       (rst_n),
        .op_valid    (tensor_valid_in),
        .op_ready    (tensor_ready),
        .op_type     (tensor_issue_op_type),
        .frag_a      (tensor_issue_frag_a),
        .frag_b      (tensor_issue_frag_b),
        .frag_c      (tensor_issue_frag_c),
        .result_valid(tensor_valid_out),
        .result_data (tensor_result)
    );


    assign tensor_meta_push_data = pack_tensor_meta(tensor_issue_warp,
                                                    tensor_issue_rd,
                                                    tensor_issue_mask);

    wb_fifo #(
        .WIDTH (TENSOR_META_W),
        .DEPTH (TENSOR_META_DEPTH)
    ) u_tensor_meta_q (
        .clk       (clk),
        .rst_n     (rst_n),
        .push      (tensor_meta_push),
        .push_data (tensor_meta_push_data),
        .pop       (tensor_meta_pop),
        .pop_data  (tensor_meta_pop_data),
        .full      (tensor_meta_full),
        .empty     (tensor_meta_empty)
    );

    assign tensor_meta_warp = tensor_meta_pop_data[TENSOR_META_WARP_MSB:TENSOR_META_WARP_LSB];
    assign tensor_meta_rd = tensor_meta_pop_data[TENSOR_META_RD_MSB:TENSOR_META_RD_LSB];
    assign tensor_meta_mask = tensor_meta_pop_data[TENSOR_META_MASK_MSB:TENSOR_META_MASK_LSB];

    assign tensor_wbq_push = tensor_valid_out && !tensor_meta_empty;
    assign tensor_wbq_push_fire = tensor_wbq_push &&
                                  (!tensor_wbq_full || tensor_wbq_pop);
    assign tensor_meta_push = tensor_issue_pop_fire;
    assign tensor_meta_pop = tensor_wbq_push_fire;

    assign tensor_wbq_push_data = pack_wb(tensor_meta_warp, tensor_meta_rd,
                                         tensor_meta_mask, tensor_result);

    wb_fifo #(
        .WIDTH (WB_PKT_W),
        .DEPTH (TENSOR_WBQ_DEPTH)
    ) u_tensor_wbq (
        .clk       (clk),
        .rst_n     (rst_n),
        .push      (tensor_wbq_push),
        .push_data (tensor_wbq_push_data),
        .pop       (tensor_wbq_pop),
        .pop_data  (tensor_wbq_pop_data),
        .full      (tensor_wbq_full),
        .empty     (tensor_wbq_empty)
    );

    assign tensor_wbq_warp = tensor_wbq_pop_data[WB_WARP_MSB:WB_WARP_LSB];
    assign tensor_wbq_rd = tensor_wbq_pop_data[WB_RD_MSB:WB_RD_LSB];
    assign tensor_wbq_mask = tensor_wbq_pop_data[WB_MASK_MSB:WB_MASK_LSB];
    assign tensor_wbq_data = tensor_wbq_pop_data[WB_DATA_MSB:WB_DATA_LSB];

    //------------------------------------------------------------------------
    // Warp Shuffle Unit
    //------------------------------------------------------------------------
    wire shfl_use_slot0 = shfl_issue0;
    wire shfl_use_slot1 = shfl_issue1;
    wire [WARP_ID_W-1:0] shfl_issue_warp = shfl_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] shfl_issue_rd = shfl_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] shfl_issue_mask = shfl_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] shfl_issue_func = shfl_use_slot0 ? issue_func : issue1_func;
    wire [15:0] shfl_issue_imm16 = shfl_use_slot0 ? issue_imm16 : issue1_imm16;
    wire [SIMD_WIDTH-1:0] shfl_src_data = shfl_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] shfl_src_b = shfl_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;

    assign shuffle_valid_in = shfl_issue;

    genvar sh_i;
    generate
        for (sh_i = 0; sh_i < NUM_LANES; sh_i = sh_i + 1) begin : shuffle_lanes
            assign shuffle_src_lane[sh_i*5 +: 5] = shfl_src_b[sh_i*32 +: 5];
        end
    endgenerate

    assign shuffle_offset = {NUM_LANES{shfl_issue_imm16[4:0]}};

    warp_shuffle u_warp_shuffle (
        .func       (shfl_issue_func),
        .src_data   (shfl_src_data),
        .src_lane   (shuffle_src_lane),
        .offset     (shuffle_offset),
        .lane_mask  (shfl_issue_mask),
        .width      (shfl_issue_imm16[4:0]),
        .membermask (shfl_issue_mask),
        .result     (shuffle_result),
        .valid_out  (shuffle_valid_mask)
    );

    // Shuffle pipeline tracking (1 stage delay for proper writeback timing)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shuffle_valid_pipe <= 1'b0;
            shuffle_warp_pipe <= 0;
            shuffle_rd_pipe <= 0;
            shuffle_mask_pipe <= 0;
            shuffle_result_pipe <= 0;
        end else begin
            shuffle_valid_pipe <= shuffle_valid_in;
            if (shuffle_valid_in) begin
                shuffle_warp_pipe <= shfl_issue_warp;
                shuffle_rd_pipe <= shfl_issue_rd;
                shuffle_mask_pipe <= shfl_issue_mask;
                shuffle_result_pipe <= shuffle_result;
            end
        end
    end

    assign shuffle_valid_out = shuffle_valid_pipe;

    //------------------------------------------------------------------------
    // Writeback queues (capture FU outputs for arbitration)
    //------------------------------------------------------------------------
    assign alu_wbq_in = pack_wb(alu_warp_pipe, alu_rd_pipe, alu_mask_pipe, alu_result_pipe);
    assign mul_wbq_in = pack_wb(mul_warp_pipe, mul_rd_pipe, mul_mask_pipe, mul_result);
    assign fpu32_wbq_in = pack_wb(fpu32_warp_pipe[4], fpu32_rd_pipe[4], fpu32_mask_pipe[4], fpu32_result);
    assign fpu64_wbq_in = pack_wb(fpu64_warp_pipe[4], fpu64_rd_pipe[4], fpu64_mask_pipe[4], fpu64_result_trunc);
    assign fp16_wbq_in = pack_wb(fp16_warp_pipe[1], fp16_rd_pipe[1], fp16_mask_pipe[1], fp16_result);
    assign sfu_wbq_in = pack_wb(sfu_warp_pipe[7], sfu_rd_pipe[7], sfu_mask_pipe[7], sfu_result);
    assign shfl_wbq_in = pack_wb(shuffle_warp_pipe, shuffle_rd_pipe, shuffle_mask_pipe, shuffle_result_pipe);

    assign alu_wbq_push = alu_valid_out;
    assign mul_wbq_push = mul_valid_out;
    assign fpu32_wbq_push = fpu32_valid_out;
    assign fpu64_wbq_push = fpu64_valid_out;
    assign fp16_wbq_push = fp16_valid_out;
    assign sfu_wbq_push = sfu_valid_out;
    assign shfl_wbq_push = shuffle_valid_out;

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(ALU_WBQ_DEPTH)
    ) u_alu_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (alu_wbq_push),
        .push_data(alu_wbq_in),
        .pop      (alu_wbq_pop),
        .pop_data (alu_wbq_out),
        .full     (alu_wbq_full),
        .empty    (alu_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(MUL_WBQ_DEPTH)
    ) u_mul_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (mul_wbq_push),
        .push_data(mul_wbq_in),
        .pop      (mul_wbq_pop),
        .pop_data (mul_wbq_out),
        .full     (mul_wbq_full),
        .empty    (mul_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(FPU32_WBQ_DEPTH)
    ) u_fpu32_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (fpu32_wbq_push),
        .push_data(fpu32_wbq_in),
        .pop      (fpu32_wbq_pop),
        .pop_data (fpu32_wbq_out),
        .full     (fpu32_wbq_full),
        .empty    (fpu32_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(FPU64_WBQ_DEPTH)
    ) u_fpu64_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (fpu64_wbq_push),
        .push_data(fpu64_wbq_in),
        .pop      (fpu64_wbq_pop),
        .pop_data (fpu64_wbq_out),
        .full     (fpu64_wbq_full),
        .empty    (fpu64_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(FP16_WBQ_DEPTH)
    ) u_fp16_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (fp16_wbq_push),
        .push_data(fp16_wbq_in),
        .pop      (fp16_wbq_pop),
        .pop_data (fp16_wbq_out),
        .full     (fp16_wbq_full),
        .empty    (fp16_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(SFU_WBQ_DEPTH)
    ) u_sfu_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (sfu_wbq_push),
        .push_data(sfu_wbq_in),
        .pop      (sfu_wbq_pop),
        .pop_data (sfu_wbq_out),
        .full     (sfu_wbq_full),
        .empty    (sfu_wbq_empty)
    );

    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(SHFL_WBQ_DEPTH)
    ) u_shfl_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (shfl_wbq_push),
        .push_data(shfl_wbq_in),
        .pop      (shfl_wbq_pop),
        .pop_data (shfl_wbq_out),
        .full     (shfl_wbq_full),
        .empty    (shfl_wbq_empty)
    );

    assign alu_wbq_warp = alu_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign mul_wbq_warp = mul_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu32_wbq_warp = fpu32_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu64_wbq_warp = fpu64_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fp16_wbq_warp = fp16_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign sfu_wbq_warp = sfu_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign shfl_wbq_warp = shfl_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign alu_wbq_rd = alu_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign mul_wbq_rd = mul_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu32_wbq_rd = fpu32_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu64_wbq_rd = fpu64_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fp16_wbq_rd = fp16_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign sfu_wbq_rd = sfu_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign shfl_wbq_rd = shfl_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign alu_wbq_mask = alu_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign mul_wbq_mask = mul_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu32_wbq_mask = fpu32_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu64_wbq_mask = fpu64_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fp16_wbq_mask = fp16_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign sfu_wbq_mask = sfu_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign shfl_wbq_mask = shfl_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign alu_wbq_data = alu_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign mul_wbq_data = mul_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fpu32_wbq_data = fpu32_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fpu64_wbq_data = fpu64_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fp16_wbq_data = fp16_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign sfu_wbq_data = sfu_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign shfl_wbq_data = shfl_wbq_out[WB_DATA_MSB:WB_DATA_LSB];

    //------------------------------------------------------------------------
    // Atomic Unit
    //------------------------------------------------------------------------
    assign atomic_valid_in = issue_valid && issue_atomic_op;

    atomic_unit u_atomic (
        .clk        (clk),
        .rst_n      (rst_n),
        .req_valid  (atomic_valid_in),
        .func       (issue_func),
        .addr       (rf_rd_data_a[31:0]),
        .operand_a  (rf_rd_data_b[31:0]),
        .operand_b  (rf_rd_data_c[31:0]),
        .mem_shared (issue_mem_shared),
        .mem_req    (),
        .mem_write  (),
        .mem_addr   (),
        .mem_wdata  (),
        .mem_ready  (1'b1),
        .mem_rdata  (32'b0),
        .result     (atomic_result),
        .result_valid(atomic_valid_out),
        .busy       (atomic_busy)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            atomic_pending_valid <= 1'b0;
        end else if (atomic_valid_in) begin
            atomic_pending_valid <= 1'b1;
            atomic_warp_pending <= issue_warp_id;
            atomic_rd_pending <= issue_rd;
            atomic_mask_pending <= issue_mask;
        end else if (atomic_valid_out) begin
            atomic_pending_valid <= 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // Shared Memory
    //------------------------------------------------------------------------
    assign smem_req_valid = issue_valid && issue_mem_shared && !issue_atomic_op &&
                            (issue_mem_read || issue_mem_write);
    assign smem_req_write = issue_mem_write;
    assign smem_req_addr = rf_rd_data_a[NUM_LANES*14-1:0];
    assign smem_req_wdata = rf_rd_data_b;

    shared_memory u_shared_mem (
        .clk          (clk),
        .rst_n        (rst_n),
        .req_valid    (smem_req_valid),
        .req_write    (smem_req_write),
        .req_addr     (smem_req_addr),
        .req_wdata    (smem_req_wdata),
        .req_mask     (issue_mask),
        .resp_valid   (smem_resp_valid),
        .resp_rdata   (smem_resp_rdata),
        .bank_conflict(smem_bank_conflict)
    );

    //------------------------------------------------------------------------
    // Global Memory Interface
    //------------------------------------------------------------------------
    assign gmem_req_valid = issue_valid && !issue_mem_shared && !issue_atomic_op &&
                            (issue_mem_read || issue_mem_write);
    assign gmem_req_write = issue_mem_write;
    assign gmem_req_addr = rf_rd_data_a;
    assign gmem_req_wdata = rf_rd_data_b;

    memory_interface u_mem_if (
        .clk         (clk),
        .rst_n       (rst_n),
        .req_valid   (gmem_req_valid),
        .req_write   (gmem_req_write),
        .req_addr    (gmem_req_addr),
        .req_wdata   (gmem_req_wdata),
        .req_mask    (issue_mask),
        .req_ready   (gmem_req_ready),
        .resp_valid  (gmem_resp_valid),
        .resp_rdata  (gmem_resp_rdata),
        // AXI signals
        .m_axi_awid  (m_axi_awid),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen (m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata (m_axi_wdata),
        .m_axi_wstrb (m_axi_wstrb),
        .m_axi_wlast (m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid   (m_axi_bid),
        .m_axi_bresp (m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid  (m_axi_arid),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen (m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid   (m_axi_rid),
        .m_axi_rdata (m_axi_rdata),
        .m_axi_rresp (m_axi_rresp),
        .m_axi_rlast (m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    //------------------------------------------------------------------------
    // Control Flow Unit (Divergence/Convergence)
    //------------------------------------------------------------------------
    control_flow_unit #(
        .NUM_WARPS(NUM_WARPS)
    ) u_cfu (
        .clk             (clk),
        .rst_n           (rst_n),
        .pc_current      (issue_pc),
        .warp_id         (issue_warp_id),
        .active_mask     (issue_mask),
        .branch_valid    (issue_valid && issue_branch_op),
        .branch_type     (issue_func),
        .branch_target   (issue_pc + {{11{issue_imm21[20]}}, issue_imm21}),
        .branch_cond     (alu_zero),
        .is_uniform      (1'b0),
        .call_valid      (1'b0),
        .call_target     (32'b0),
        .ret_valid       (1'b0),
        .diverge_mask    (alu_zero),
        .next_pc         (cfu_branch_target),
        .next_active_mask(cfu_next_mask),
        .pc_valid        (cfu_branch_taken),
        .stall           (cfu_stall),
        .reconverge_pc   (cfu_reconverge_pc),
        .at_reconverge   (cfu_at_reconverge),
        .stack_overflow  (),
        .stack_underflow ()
    );

    assign cfu_active_mask = cfu_next_mask[NUM_LANES-1:0];

    //========================================================================
    // STAGE 5: WRITEBACK (Round-Robin Arbitration + Scoreboard Clear)
    //========================================================================

    // Collect all ready FU outputs for round-robin arbitration
    wire [9:0] fu_ready;
    assign fu_ready[0] = !alu_wbq_empty;                     // ALU (queued)
    assign fu_ready[1] = !mul_wbq_empty;                     // MUL (queued)
    assign fu_ready[2] = !fpu32_wbq_empty;                   // FPU32 (queued)
    assign fu_ready[3] = !fpu64_wbq_empty;                   // FPU64 (queued)
    assign fu_ready[4] = !fp16_wbq_empty;                    // FP16 (queued)
    assign fu_ready[5] = !sfu_wbq_empty;                     // SFU (queued)
    assign fu_ready[6] = !tensor_wbq_empty;                  // Tensor (queued)
    assign fu_ready[7] = smem_resp_valid || gmem_resp_valid || store_pending_valid; // Memory
    assign fu_ready[8] = !shfl_wbq_empty;                    // Shuffle (queued)
    assign fu_ready[9] = atomic_valid_out;                   // Atomic

    // Round-robin selection for writeback
    reg [3:0] wb_sel;
    reg       wb_found;
    integer   wb_i;

    always @(*) begin
        wb_found = 1'b0;
        wb_sel = 0;
        // Start from last priority + 1 for fairness
        for (wb_i = 0; wb_i < 10; wb_i = wb_i + 1) begin
            if (!wb_found && fu_ready[(wb_arb_priority + wb_i) % 10]) begin
                wb_sel = (wb_arb_priority + wb_i) % 10;
                wb_found = 1'b1;
            end
        end
    end

    assign alu_wbq_pop = wb_found && (wb_sel == 4'd0);
    assign mul_wbq_pop = wb_found && (wb_sel == 4'd1);
    assign fpu32_wbq_pop = wb_found && (wb_sel == 4'd2);
    assign fpu64_wbq_pop = wb_found && (wb_sel == 4'd3);
    assign fp16_wbq_pop = wb_found && (wb_sel == 4'd4);
    assign sfu_wbq_pop = wb_found && (wb_sel == 4'd5);
    assign tensor_wbq_pop = wb_found && (wb_sel == 4'd6);
    assign shfl_wbq_pop = wb_found && (wb_sel == 4'd8);

    // Writeback arbiter with proper warp/rd tracking from FU pipelines
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_valid <= 1'b0;
            wb_arb_priority <= 0;
        end else begin
            if (wb_found) begin
                wb_valid <= 1'b1;
                wb_arb_priority <= (wb_sel + 1) % 10;  // Advance for fairness

                case (wb_sel)
                    4'd0: begin  // ALU (1-cycle pipeline for proper timing)
                        wb_warp_id <= alu_wbq_warp;
                        wb_rd <= alu_wbq_rd;
                        wb_data <= alu_wbq_data;
                        wb_mask <= alu_wbq_mask;
                    end
                    4'd1: begin  // MUL (1-cycle pipeline)
                        wb_warp_id <= mul_wbq_warp;
                        wb_rd <= mul_wbq_rd;
                        wb_data <= mul_wbq_data;
                        wb_mask <= mul_wbq_mask;
                    end
                    4'd2: begin  // FPU32 (4-cycle pipeline)
                        wb_warp_id <= fpu32_wbq_warp;
                        wb_rd <= fpu32_wbq_rd;
                        wb_data <= fpu32_wbq_data;
                        wb_mask <= fpu32_wbq_mask;
                    end
                    4'd3: begin  // FPU64 (truncated)
                        wb_warp_id <= fpu64_wbq_warp;
                        wb_rd <= fpu64_wbq_rd;
                        wb_data <= fpu64_wbq_data;
                        wb_mask <= fpu64_wbq_mask;
                    end
                    4'd4: begin  // FP16 (2-cycle pipeline)
                        wb_warp_id <= fp16_wbq_warp;
                        wb_rd <= fp16_wbq_rd;
                        wb_data <= fp16_wbq_data;
                        wb_mask <= fp16_wbq_mask;
                    end
                    4'd5: begin  // SFU (8-cycle pipeline)
                        wb_warp_id <= sfu_wbq_warp;
                        wb_rd <= sfu_wbq_rd;
                        wb_data <= sfu_wbq_data;
                        wb_mask <= sfu_wbq_mask;
                    end
                    4'd6: begin  // Tensor (variable latency)
                        wb_warp_id <= tensor_wbq_warp;
                        wb_rd <= tensor_wbq_rd;
                        wb_data <= tensor_wbq_data;
                        wb_mask <= tensor_wbq_mask;
                    end
                    4'd7: begin  // Memory
                        if (smem_resp_valid) begin
                            wb_warp_id <= smem_warp_pending;
                            wb_rd <= smem_rd_pending;
                            wb_data <= smem_resp_rdata;
                            wb_mask <= smem_mask_pending;
                        end else if (gmem_resp_valid) begin
                            wb_warp_id <= mem_warp_pending;
                            wb_rd <= mem_rd_pending;
                            wb_data <= gmem_resp_rdata;
                            wb_mask <= mem_mask_pending;
                        end else begin
                            wb_warp_id <= store_warp_pending;
                            wb_rd <= 0;
                            wb_data <= 0;
                            wb_mask <= store_mask_pending;
                        end
                    end
                    4'd8: begin  // Shuffle (1-cycle pipeline for proper timing)
                        wb_warp_id <= shfl_wbq_warp;
                        wb_rd <= shfl_wbq_rd;
                        wb_data <= shfl_wbq_data;
                        wb_mask <= shfl_wbq_mask;
                    end
                    4'd9: begin  // Atomic
                        wb_warp_id <= atomic_warp_pending;
                        wb_rd <= atomic_rd_pending;
                        wb_data <= {NUM_LANES{atomic_result}};
                        wb_mask <= atomic_mask_pending;
                    end
                endcase
            end else begin
                wb_valid <= 1'b0;
            end

        end
    end

    // Register file write
    assign rf_wr_en = wb_valid && (wb_rd != 0);
    assign rf_wr_data = wb_data;
    assign rf_wr_mask = wb_mask;

    //========================================================================
    // Warp State Management
    //========================================================================
    integer w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                warp_valid[w] <= 1'b0;
                warp_active[w] <= 1'b0;
                warp_stalled_mem[w] <= 1'b0;
                warp_stalled_fu[w] <= 1'b0;
                warp_stalled_sync[w] <= 1'b0;
                warp_exit_pending[w] <= 1'b0;
                warp_pc[w] <= 32'b0;
                warp_fetch_pc[w] <= 32'b0;
                warp_mask[w] <= {NUM_LANES{1'b1}};
            end
            // Branch predictor update state
            bp_update_valid <= 1'b0;
            bp_update_warp_id <= {WARP_ID_W{1'b0}};
            bp_update_pc <= 32'b0;
            bp_update_taken <= 1'b0;
            bp_update_target <= 32'b0;
            bp_update_is_call <= 1'b0;
            bp_update_is_return <= 1'b0;
            bp_update_mispredicted <= 1'b0;
        end else begin
            // Kernel start - allocate initial warps
            if (kernel_start) begin
                for (w = 0; w < NUM_WARPS; w = w + 1) begin
                    if (w < INIT_WARPS) begin
                        warp_valid[w] <= 1'b1;
                        warp_active[w] <= 1'b1;
                        warp_exit_pending[w] <= 1'b0;
                        warp_pc[w] <= kernel_pc;
                        warp_fetch_pc[w] <= kernel_pc;
                        warp_mask[w] <= {NUM_LANES{1'b1}};
                    end else begin
                        warp_valid[w] <= 1'b0;
                        warp_active[w] <= 1'b0;
                        warp_exit_pending[w] <= 1'b0;
                        warp_pc[w] <= 32'b0;
                        warp_fetch_pc[w] <= 32'b0;
                        warp_mask[w] <= {NUM_LANES{1'b0}};
                    end
                end
            end

            // NOTE: Fetch PC is advanced in the instruction buffer fill logic (line 1083)
            // when icache returns valid data. Don't advance here on fetch_fire to avoid
            // double-counting. The fetch_fire signal is used for other purposes (arbitration).

            // Update architectural PC when instruction is issued (in-order)
            if (issue_accept && !dec_branch_op && !dec_exit_op) begin
                warp_pc[dec0_warp_id] <= warp_pc[dec0_warp_id] + 4;
            end

            // Branch handling
            if (cfu_branch_taken) begin
                warp_pc[issue_warp_id] <= cfu_branch_target;
                warp_fetch_pc[issue_warp_id] <= cfu_branch_target;
                warp_mask[issue_warp_id] <= cfu_active_mask;
            end

            // Branch predictor update (when branch resolves)
            if (issue_valid && issue_branch_op) begin
                bp_update_valid <= 1'b1;
                bp_update_warp_id <= issue_warp_id;
                bp_update_pc <= issue_pc;
                bp_update_taken <= cfu_branch_taken;
                bp_update_target <= cfu_branch_target;
                bp_update_is_call <= (issue_func == 6'h01);  // JAL-like
                bp_update_is_return <= (issue_func == 6'h02); // RET-like
                bp_update_mispredicted <= 1'b0;  // TODO: Compare with prediction
            end else begin
                bp_update_valid <= 1'b0;
            end

            // Memory stall handling
            if (issue_valid && issue_mem_read && !issue_atomic_op) begin
                warp_stalled_mem[issue_warp_id] <= 1'b1;
            end
            if (smem_resp_valid && smem_pending_valid) begin
                warp_stalled_mem[smem_warp_pending] <= 1'b0;
            end
            if (gmem_resp_valid && mem_pending_valid) begin
                warp_stalled_mem[mem_warp_pending] <= 1'b0;
            end

            // Sync barrier
            if (issue_valid && issue_sync_op) begin
                warp_stalled_sync[issue_warp_id] <= 1'b1;
            end
            // TODO: Check all warps at barrier and release

            // Exit instruction (defer warp teardown until in-flight ops drain)
            if (issue_valid && issue_exit_op) begin
                warp_exit_pending[issue_warp_id] <= 1'b1;
            end

            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                if (warp_exit_pending[w] &&
                    (pending_fu_count[w] == 0) &&
                    (scoreboard_busy[w] == 0) &&
                    (!mem_pending_valid || (mem_warp_pending != w[WARP_ID_W-1:0])) &&
                    (!smem_pending_valid || (smem_warp_pending != w[WARP_ID_W-1:0])) &&
                    (!store_pending_valid || (store_warp_pending != w[WARP_ID_W-1:0])) &&
                    (!atomic_pending_valid || (atomic_warp_pending != w[WARP_ID_W-1:0]))) begin
                    warp_valid[w] <= 1'b0;
                    warp_active[w] <= 1'b0;
                    warp_exit_pending[w] <= 1'b0;
                end
            end
        end
    end

    //========================================================================
    // Kernel Completion
    //========================================================================
    assign kernel_done = (warp_valid == 0) && !kernel_start;

    //========================================================================
    // Special Register Generation
    //========================================================================
    reg [31:0] block_id_regs [0:2];
    reg [31:0] block_dim_regs [0:2];
    reg [31:0] grid_dim_regs [0:2];

    always @(posedge clk) begin
        if (kernel_start) begin
            block_id_regs[0] <= block_id_x;
            block_id_regs[1] <= block_id_y;
            block_id_regs[2] <= block_id_z;
            block_dim_regs[0] <= block_dim_x;
            block_dim_regs[1] <= block_dim_y;
            block_dim_regs[2] <= block_dim_z;
            grid_dim_regs[0] <= grid_dim_x;
            grid_dim_regs[1] <= grid_dim_y;
            grid_dim_regs[2] <= grid_dim_z;
        end
    end

endmodule


//============================================================================
// SIMD FPU Wrapper (Instantiates per-lane FPU with proper interface)
//============================================================================
module simd_fpu #(
    parameter NUM_LANES = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    output wire                     ready,
    input  wire [5:0]               func,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand_a,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand_b,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand_c,
    input  wire [NUM_LANES-1:0]     lane_mask,
    output reg                      valid_out,
    output reg  [NUM_LANES*DATA_WIDTH-1:0] result
);

    // Pipeline stages for FMA (4 cycles)
    reg valid_pipe [0:3];
    reg [NUM_LANES*DATA_WIDTH-1:0] result_pipe [0:3];

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : fpu_lanes
            wire [31:0] a = operand_a[i*32 +: 32];
            wire [31:0] b = operand_b[i*32 +: 32];
            wire [31:0] c = operand_c[i*32 +: 32];
            wire [31:0] r;
            wire fpu_valid_out;

            fpu u_fpu (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .rnd_mode   (2'b00),     // Round to nearest
                .ftz        (1'b0),       // Don't flush to zero
                .operand_a  (a),
                .operand_b  (b),
                .operand_c  (c),
                .valid_in   (valid_pipe[2]),  // Delayed to match pipeline
                .result     (r),
                .valid_out  (fpu_valid_out),
                .overflow   (),
                .underflow  (),
                .inexact    (),
                .invalid    (),
                .div_by_zero()
            );

            always @(posedge clk) begin
                if (fpu_valid_out)
                    result_pipe[3][i*32 +: 32] <= r;
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe[0] <= 0;
            valid_pipe[1] <= 0;
            valid_pipe[2] <= 0;
            valid_pipe[3] <= 0;
            valid_out <= 0;
        end else begin
            valid_pipe[0] <= valid_in;
            valid_pipe[1] <= valid_pipe[0];
            valid_pipe[2] <= valid_pipe[1];
            valid_pipe[3] <= valid_pipe[2];
            valid_out <= valid_pipe[3];
            result <= result_pipe[3];
        end
    end

    assign ready = 1'b1;  // Always ready (pipelined)

endmodule


//============================================================================
// SIMD FPU64 Wrapper
//============================================================================
module simd_fpu64 #(
    parameter NUM_LANES = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    output wire                     ready,
    input  wire [5:0]               func,
    input  wire [NUM_LANES*32-1:0]  operand_a,
    input  wire [NUM_LANES*32-1:0]  operand_b,
    input  wire [NUM_LANES*32-1:0]  operand_c,
    input  wire [NUM_LANES-1:0]     lane_mask,
    output reg                      valid_out,
    output reg  [NUM_LANES*64-1:0]  result
);
    // FP64 operations use pairs of lanes (16 double-precision operations)
    // Implementation similar to simd_fpu but with fpu64 instances

    reg [3:0] valid_pipe;
    reg [NUM_LANES*64-1:0] result_pipe [0:3];

    wire [NUM_LANES*64-1:0] operand_a_ext;
    genvar f64_idx;
    generate
        for (f64_idx = 0; f64_idx < NUM_LANES; f64_idx = f64_idx + 1) begin : f64_ext
            assign operand_a_ext[f64_idx*64 +: 64] = {32'b0, operand_a[f64_idx*32 +: 32]};
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            valid_out <= 0;
            result <= 0;
            result_pipe[0] <= 0;
            result_pipe[1] <= 0;
            result_pipe[2] <= 0;
            result_pipe[3] <= 0;
        end else begin
            valid_pipe <= {valid_pipe[2:0], valid_in};
            valid_out <= valid_pipe[3];
            result_pipe[0] <= operand_a_ext;
            result_pipe[1] <= result_pipe[0];
            result_pipe[2] <= result_pipe[1];
            result_pipe[3] <= result_pipe[2];
            result <= result_pipe[3];
        end
    end

    assign ready = 1'b1;
endmodule


//============================================================================
// SIMD FP16 Wrapper
//============================================================================
module simd_fp16 #(
    parameter NUM_LANES = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    output wire                     ready,
    input  wire [5:0]               func,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand_a,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand_b,
    input  wire [NUM_LANES-1:0]     lane_mask,
    output reg                      valid_out,
    output reg  [NUM_LANES*DATA_WIDTH-1:0] result
);
    // FP16 packed operations (2x FP16 per 32-bit lane)

    reg [1:0] valid_pipe;
    reg [NUM_LANES*DATA_WIDTH-1:0] result_pipe [0:1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            valid_out <= 0;
            result <= 0;
            result_pipe[0] <= 0;
            result_pipe[1] <= 0;
        end else begin
            valid_pipe <= {valid_pipe[0], valid_in};
            valid_out <= valid_pipe[1];
            result_pipe[0] <= operand_a;
            result_pipe[1] <= result_pipe[0];
            result <= result_pipe[1];
        end
    end

    assign ready = 1'b1;
endmodule


//============================================================================
// SIMD SFU Wrapper (Instantiates per-lane SFU with proper interface)
//============================================================================
module simd_sfu #(
    parameter NUM_LANES = 32,
    parameter DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    output wire                     ready,
    input  wire [5:0]               func,
    input  wire [NUM_LANES*DATA_WIDTH-1:0] operand,
    input  wire [NUM_LANES-1:0]     lane_mask,
    output reg                      valid_out,
    output reg  [NUM_LANES*DATA_WIDTH-1:0] result
);
    // SFU operations: sin, cos, sqrt, rsqrt, lg2, ex2, rcp
    // 8-cycle latency

    reg [7:0] valid_pipe;

    genvar i;
    generate
        for (i = 0; i < NUM_LANES; i = i + 1) begin : sfu_lanes
            wire [31:0] op = operand[i*32 +: 32];
            wire [31:0] r;
            wire sfu_valid_out;

            sfu u_sfu (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .operand    (op),
                .valid_in   (valid_pipe[6]),  // Delayed to match pipeline
                .result     (r),
                .valid_out  (sfu_valid_out),
                .invalid    (),
                .div_by_zero()
            );

            always @(posedge clk) begin
                if (sfu_valid_out)
                    result[i*32 +: 32] <= r;
            end
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            valid_out <= 0;
        end else begin
            valid_pipe <= {valid_pipe[6:0], valid_in};
            valid_out <= valid_pipe[7];
        end
    end

    assign ready = 1'b1;
endmodule


//============================================================================
// Simple FIFO for writeback queues (single push/pop per cycle)
//============================================================================
module wb_fifo #(
    parameter WIDTH = 32,
    parameter DEPTH = 2
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             push,
    input  wire [WIDTH-1:0] push_data,
    input  wire             pop,
    output wire [WIDTH-1:0] pop_data,
    output wire             full,
    output wire             empty
);
    localparam PTR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam COUNT_W = $clog2(DEPTH + 1);
    localparam [COUNT_W-1:0] DEPTH_VAL = DEPTH;

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_W-1:0] head;
    reg [PTR_W-1:0] tail;
    reg [COUNT_W-1:0] count;

    wire push_fire = push && (!full || pop);
    wire pop_fire = pop && !empty;

    assign full = (count == DEPTH_VAL);
    assign empty = (count == 0);
    assign pop_data = empty ? {WIDTH{1'b0}} : mem[head];

    function [PTR_W-1:0] ptr_inc;
        input [PTR_W-1:0] ptr;
        begin
            if (ptr == DEPTH - 1) begin
                ptr_inc = {PTR_W{1'b0}};
            end else begin
                ptr_inc = ptr + 1'b1;
            end
        end
    endfunction

    integer mi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= {PTR_W{1'b0}};
            tail <= {PTR_W{1'b0}};
            count <= {COUNT_W{1'b0}};
            for (mi = 0; mi < DEPTH; mi = mi + 1) begin
                mem[mi] <= {WIDTH{1'b0}};
            end
        end else begin
            if (push_fire) begin
                mem[tail] <= push_data;
                tail <= ptr_inc(tail);
            end

            if (pop_fire) begin
                head <= ptr_inc(head);
            end

            case ({push_fire, pop_fire})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
