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
    parameter [3:0] TC_DATA_DEFAULT = `TC_DATA_FP16,  // Extended to 4-bit for FP6
    parameter TC_USE_OP_TYPE = 1,
    parameter [1:0] TC_FP4_FORMAT = `TC_FP4_E2M1,
    parameter [1:0] TC_FP6_FORMAT = `TC_FP6_E3M2,     // 5th-gen Tensor Core FP6
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
    localparam VIDEO_WBQ_DEPTH = 4;
    localparam ALU_WBQ_COUNT_W = $clog2(ALU_WBQ_DEPTH + 1);
    localparam MUL_WBQ_COUNT_W = $clog2(MUL_WBQ_DEPTH + 1);
    localparam FPU32_WBQ_COUNT_W = $clog2(FPU32_WBQ_DEPTH + 1);
    localparam FPU64_WBQ_COUNT_W = $clog2(FPU64_WBQ_DEPTH + 1);
    localparam FP16_WBQ_COUNT_W = $clog2(FP16_WBQ_DEPTH + 1);
    localparam SFU_WBQ_COUNT_W = $clog2(SFU_WBQ_DEPTH + 1);
    localparam SHFL_WBQ_COUNT_W = $clog2(SHFL_WBQ_DEPTH + 1);
    localparam VIDEO_WBQ_COUNT_W = $clog2(VIDEO_WBQ_DEPTH + 1);
    localparam [ALU_WBQ_COUNT_W-1:0] ALU_WBQ_DEPTH_VAL = ALU_WBQ_DEPTH;
    localparam [MUL_WBQ_COUNT_W-1:0] MUL_WBQ_DEPTH_VAL = MUL_WBQ_DEPTH;
    localparam [FPU32_WBQ_COUNT_W-1:0] FPU32_WBQ_DEPTH_VAL = FPU32_WBQ_DEPTH;
    localparam [FPU64_WBQ_COUNT_W-1:0] FPU64_WBQ_DEPTH_VAL = FPU64_WBQ_DEPTH;
    localparam [FP16_WBQ_COUNT_W-1:0] FP16_WBQ_DEPTH_VAL = FP16_WBQ_DEPTH;
    localparam [SFU_WBQ_COUNT_W-1:0] SFU_WBQ_DEPTH_VAL = SFU_WBQ_DEPTH;
    localparam [SHFL_WBQ_COUNT_W-1:0] SHFL_WBQ_DEPTH_VAL = SHFL_WBQ_DEPTH;
    localparam [VIDEO_WBQ_COUNT_W-1:0] VIDEO_WBQ_DEPTH_VAL = VIDEO_WBQ_DEPTH;

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
    localparam TENSOR_ISSUE_OP_MSB = TENSOR_ISSUE_OP_LSB + 4 - 1;  // Extended to 4-bit for FP6
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
    reg  [NUM_WARPS-1:0] warp_stalled_async;  // Waiting on cp.async wait_group/all
    reg  [NUM_WARPS-1:0] warp_stalled_branch; // Branch in pipeline, wait for resolution
    reg  [NUM_WARPS-1:0] warp_exit_pending;  // EXIT issued, waiting for drain
    reg  [31:0]          warp_pc [0:NUM_WARPS-1];
    reg  [31:0]          warp_fetch_pc [0:NUM_WARPS-1];
    reg  [NUM_LANES-1:0] warp_mask [0:NUM_WARPS-1];  // Active thread mask
    reg  [3:0]           cp_async_pending [0:NUM_WARPS-1];  // Outstanding cp.async copies
    reg  [3:0]           cp_async_wait_threshold [0:NUM_WARPS-1];
    reg  [NUM_WARPS-1:0] cp_async_wait_all;
    reg                  barrier_pending;  // simple global barrier flag

    // Warp-level synchronization (bar.warp.sync) state
    reg  [NUM_WARPS-1:0] warp_sync_pending;           // Warp has bar.warp.sync in progress
    reg  [31:0]          warp_sync_arrived [0:NUM_WARPS-1];  // Per-lane arrival bits
    reg  [31:0]          warp_sync_mask [0:NUM_WARPS-1];     // Expected thread mask (membermask)

    // Cluster barrier state (barrier.cluster - Hopper+ Thread Block Cluster sync)
    // Note: Full cluster interconnect is external; SM tracks local state
    reg  [NUM_WARPS-1:0] cluster_barrier_pending;     // Warp waiting on cluster barrier
    reg  [NUM_WARPS-1:0] cluster_barrier_arrived;     // Warp has arrived at barrier
    reg  [7:0]           cluster_barrier_id [0:NUM_WARPS-1]; // Barrier ID per warp
    reg  [15:0]          cluster_barrier_thread_count;  // Expected thread count (shared)
    reg  [15:0]          cluster_local_arrive_count;    // Local SM arrive count
    reg                  cluster_barrier_complete;      // Set when all have arrived (stub for single SM)

    wire [NUM_WARPS-1:0] warp_ready = warp_valid & ~warp_stalled_mem &
                                       ~warp_stalled_fu & ~warp_stalled_sync &
                                       ~warp_stalled_async & ~warp_stalled_branch &
                                       ~warp_exit_pending & ~mbarrier_warp_blocked &
                                       ~warp_stalled_wgmma & ~cluster_barrier_pending;

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
    reg                  issue_video_op;       // Video SIMD operation
    reg                  issue_reg_write;
    reg                  issue_bar_warp_sync;  // bar.warp.sync flag
    reg                  issue_tex_op;         // Texture operation (tex/txq/suld/sust/sured)
    reg                  issue_cache_policy_op; // Cache policy operation
    reg                  issue_stack_op;       // Stack operation (alloca/stacksave/stackrestore)
    reg                  issue_debug_op;       // Debug operation (brkpt/trap/pmevent)
    reg                  issue_misc_op;        // Misc operation (nanosleep/setmaxnreg)
    reg                  issue_st_async_op;    // st.async operation
    reg                  issue_multimem_op;    // multimem operation
    reg                  issue_barrier_cluster_op;  // barrier.cluster operation

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
    reg                  issue1_video_op;       // Video SIMD operation
    reg                  issue1_reg_write;
    reg                  issue1_bar_warp_sync;  // bar.warp.sync flag
    reg                  issue1_tex_op;         // Texture operation (tex/txq/suld/sust/sured)
    reg                  issue1_cache_policy_op; // Cache policy operation
    reg                  issue1_stack_op;       // Stack operation (alloca/stacksave/stackrestore)
    reg                  issue1_debug_op;       // Debug operation (brkpt/trap/pmevent)
    reg                  issue1_misc_op;        // Misc operation (nanosleep/setmaxnreg)
    reg                  issue1_st_async_op;    // st.async operation
    reg                  issue1_multimem_op;    // multimem operation
    reg                  issue1_barrier_cluster_op;  // barrier.cluster operation

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
    wire                 dec_cvt_op;  // CVT (type conversion) operation
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
    wire                 dec1_cvt_op;  // CVT (type conversion) operation
    wire                 dec1_sfu_op, dec1_tensor_op;
    wire                 dec1_mem_read, dec1_mem_write, dec1_mem_shared;
    wire                 dec1_branch_op, dec1_sync_op;
    wire                 dec1_special_reg, dec1_exit_op;
    wire                 dec1_reduce_op;
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
    wire [3:0] tensor_issue_op_type;  // Extended to 4-bit for FP6
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

    // Video SIMD Unit Signals
    wire [SIMD_WIDTH-1:0] video_result;
    wire                  video_valid_in, video_valid_out;

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

    // Async Copy Engine Signals (cp.async support)
    wire                  ace_ready;
    wire                  ace_done;
    wire [3:0]            ace_pending_count;
    wire                  ace_gmem_req_valid;
    wire [31:0]           ace_gmem_req_addr;
    wire [4:0]            ace_gmem_req_size;
    wire [2:0]            ace_gmem_req_cache;
    wire                  ace_smem_wr_en;
    wire [13:0]           ace_smem_wr_addr;
    wire [127:0]          ace_smem_wr_data;
    wire [4:0]            ace_smem_wr_size;  // 5 bits to hold values up to 16

    // mbarrier Unit Signals (Hopper+ memory barriers)
    wire                  mbarrier_ready;
    wire                  mbarrier_done;
    wire [31:0]           mbarrier_result;
    wire                  mbarrier_result_valid;
    wire [NUM_WARPS-1:0]  mbarrier_warp_blocked;

    // WGMMA Unit Signals (Hopper+ Warpgroup MMA)
    wire                  wgmma_ready;
    wire                  wgmma_done;
    wire [3:0]            wgmma_pending_ops;
    wire [1023:0]         wgmma_accum_out;
    reg  [NUM_WARPS-1:0]  warp_stalled_wgmma;         // Warps waiting on WGMMA wait_group
    reg  [3:0]            wgmma_wait_threshold [0:NUM_WARPS-1];  // Per-warp wait threshold

    // WGMMA-SMEM Interface Signals
    wire                  wgmma_smem_rd_en;           // WGMMA shared memory read enable
    wire [13:0]           wgmma_smem_addr_a;          // Base address for matrix A tile
    wire [13:0]           wgmma_smem_addr_b;          // Base address for matrix B tile
    wire [511:0]          wgmma_smem_data_a;          // 512-bit data from SMEM for matrix A
    wire [511:0]          wgmma_smem_data_b;          // 512-bit data from SMEM for matrix B
    wire                  wgmma_smem_rd_valid;        // SMEM read data valid

    // WGMMA Accumulator Interface
    reg  [1023:0]         wgmma_accum_reg [0:7];      // 8 accumulator registers per warpgroup
    wire [2:0]            wgmma_accum_idx;            // Accumulator index from descriptor

    // Texture Unit Signals (tex/txq/suld/sust/sured)
    wire                  tex_valid_in;
    wire                  tex_valid_out;
    wire [127:0]          tex_result;
    wire                  tex_busy;
    wire                  tex_mem_req;
    wire                  tex_mem_write;
    wire [31:0]           tex_mem_addr;
    wire [127:0]          tex_mem_wdata;
    reg                   tex_mem_pending;  // Track pending texture memory request
    wire                  tex_mem_ready;    // Ready signal to texture unit
    wire [127:0]          tex_mem_rdata;    // Read data to texture unit
    wire                  tex_mem_valid;    // Response valid to texture unit

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

    // Pipeline flush signals for branch taken (uses branch_taken_combined defined later)
    // Flush decode/issue stages for the branching warp
    wire                  branch_flush_dec0;
    wire                  branch_flush_dec1;

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
    // Note: Unlike RISC-V, CUDA/PTX R0 is a normal register, not hardwired to 0
    wire lane0_ra_busy = dec0_valid && scoreboard_busy[dec0_warp_id][dec_ra];
    wire lane0_rb_busy = dec0_valid && scoreboard_busy[dec0_warp_id][dec_rb];
    wire lane0_rc_busy = dec0_valid && scoreboard_busy[dec0_warp_id][dec_rc];
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
    // WGMMA stall: stall if WGMMA instruction and WGMMA unit not ready
    wire dec_wgmma_op = dec_wgmma_load || dec_wgmma_store || dec_wgmma_mma;
    wire lane0_stall_wgmma = dec0_valid && dec_wgmma_op && !wgmma_ready;
    wire lane0_stall_wbq = dec0_valid && (
                           (dec_alu_op && (alu_inflight == ALU_WBQ_DEPTH_VAL)) ||
                           (dec_mul_op && (mul_inflight == MUL_WBQ_DEPTH_VAL)) ||
                           (dec_fp32_op && (fpu32_inflight == FPU32_WBQ_DEPTH_VAL)) ||
                           (dec_fp64_op && (fpu64_inflight == FPU64_WBQ_DEPTH_VAL)) ||
                           (dec_fp16_op && (fp16_inflight == FP16_WBQ_DEPTH_VAL)) ||
                           (dec_sfu_op && (sfu_inflight == SFU_WBQ_DEPTH_VAL)) ||
                           (dec_shuffle_op && (shfl_inflight == SHFL_WBQ_DEPTH_VAL)) ||
                           (dec_video_op && (video_inflight == VIDEO_WBQ_DEPTH_VAL))
                           );
    wire lane0_ready = dec0_valid && !lane0_stall_raw && !lane0_stall_fu &&
                       !lane0_stall_mem && !lane0_stall_atomic &&
                       !lane0_stall_tensor && !lane0_stall_wgmma && !lane0_stall_wbq;

    // Lane 1 dependency checks (compute-only)
    // Note: Unlike RISC-V, CUDA/PTX R0 is a normal register, not hardwired to 0
    wire lane1_ra_busy = dec1_valid && scoreboard_busy[dec1_warp_id][dec1_ra];
    wire lane1_rb_busy = dec1_valid && scoreboard_busy[dec1_warp_id][dec1_rb];
    wire lane1_rc_busy = dec1_valid && scoreboard_busy[dec1_warp_id][dec1_rc];
    wire lane1_stall_raw = dec1_valid && (lane1_ra_busy || lane1_rb_busy || lane1_rc_busy);
    wire lane1_stall_fu = dec1_valid && (pending_fu_count[dec1_warp_id] >= 8);
    wire lane1_stall_wbq = dec1_valid && (
                           (dec1_alu_op && (alu_inflight == ALU_WBQ_DEPTH_VAL)) ||
                           (dec1_mul_op && (mul_inflight == MUL_WBQ_DEPTH_VAL)) ||
                           (dec1_fp32_op && (fpu32_inflight == FPU32_WBQ_DEPTH_VAL)) ||
                           (dec1_fp64_op && (fpu64_inflight == FPU64_WBQ_DEPTH_VAL)) ||
                           (dec1_fp16_op && (fp16_inflight == FP16_WBQ_DEPTH_VAL)) ||
                           (dec1_sfu_op && (sfu_inflight == SFU_WBQ_DEPTH_VAL)) ||
                           (dec1_shuffle_op && (shfl_inflight == SHFL_WBQ_DEPTH_VAL)) ||
                           (dec1_video_op && (video_inflight == VIDEO_WBQ_DEPTH_VAL))
                           );
    wire lane1_is_compute = dec1_alu_op || dec1_mul_op || dec1_div_op ||
                            dec1_fp32_op || dec1_fp64_op || dec1_fp16_op ||
                            dec1_sfu_op || dec1_shuffle_op || dec1_video_op;
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

    wire lane0_alu = lane0_ready && (dec_alu_op || dec_branch_op || dec_cvt_op);  // CVT routed through ALU
    wire lane0_mul = lane0_ready && (dec_mul_op || dec_div_op);
    wire lane0_fp32 = lane0_ready && dec_fp32_op;
    wire lane0_fp64 = lane0_ready && dec_fp64_op;
    wire lane0_fp16 = lane0_ready && dec_fp16_op;
    wire lane0_sfu = lane0_ready && dec_sfu_op;
    wire lane0_shfl = lane0_ready && dec_shuffle_op;
    wire lane0_video = lane0_ready && dec_video_op;  // Video SIMD unit
    wire lane1_alu = lane1_ready && (dec1_alu_op || dec1_cvt_op);  // CVT routed through ALU
    wire lane1_mul = lane1_ready && (dec1_mul_op || dec1_div_op);
    wire lane1_fp32 = lane1_ready && dec1_fp32_op;
    wire lane1_fp64 = lane1_ready && dec1_fp64_op;
    wire lane1_fp16 = lane1_ready && dec1_fp16_op;
    wire lane1_sfu = lane1_ready && dec1_sfu_op;
    wire lane1_shfl = lane1_ready && dec1_shuffle_op;
    wire lane1_video = lane1_ready && dec1_video_op;  // Video SIMD unit
    wire lane_unit_conflict = (lane0_alu && lane1_alu) ||
                              (lane0_mul && lane1_mul) ||
                              (lane0_fp32 && lane1_fp32) ||
                              (lane0_fp64 && lane1_fp64) ||
                              (lane0_fp16 && lane1_fp16) ||
                              (lane0_sfu && lane1_sfu) ||
                              (lane0_shfl && lane1_shfl) ||
                              (lane0_video && lane1_video);
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
    wire issue_cpasync = issue_valid && (issue_opcode == `OP_CPASYNC);
    wire issue1_cpasync = issue1_valid && (issue1_opcode == `OP_CPASYNC);
    wire issue_cpasync_copy = issue_cpasync &&
                              ((issue_func == `CPASYNC_CA) ||
                               (issue_func == `CPASYNC_CG) ||
                               (issue_func == `CPASYNC_BULK));
    wire issue1_cpasync_copy = issue1_cpasync &&
                               ((issue1_func == `CPASYNC_CA) ||
                                (issue1_func == `CPASYNC_CG) ||
                                (issue1_func == `CPASYNC_BULK));
    wire issue_cpasync_wait = issue_cpasync && (issue_func == `CPASYNC_WAIT);
    wire issue_cpasync_wait_all = issue_cpasync && (issue_func == `CPASYNC_WAIT_ALL);
    wire issue1_cpasync_wait = issue1_cpasync && (issue1_func == `CPASYNC_WAIT);
    wire issue1_cpasync_wait_all = issue1_cpasync && (issue1_func == `CPASYNC_WAIT_ALL);

    // Aliases for compatibility with later code sections
    wire issue_stall_mem = lane0_stall_mem;
    wire issue_accept = issue0_fire;

    // cp.async backpressure: stall issue if ACE not ready for cp.async instructions
    wire cpasync_stall = (issue_cpasync_copy || issue1_cpasync_copy ||
                          issue_cpasync_wait || issue1_cpasync_wait ||
                          issue_cpasync_wait_all || issue1_cpasync_wait_all) && !ace_ready;

    // mbarrier instruction detection
    wire issue_mbarrier = issue_valid && (issue_opcode == `OP_MBARRIER);
    wire issue1_mbarrier = issue1_valid && (issue1_opcode == `OP_MBARRIER);

    // mbarrier backpressure: stall if mbarrier unit not ready
    wire mbarrier_stall = (issue_mbarrier || issue1_mbarrier) && !mbarrier_ready;

    // WGMMA instruction detection (Hopper+ Warpgroup MMA)
    wire issue_wgmma = issue_valid && (issue_opcode == `OP_WGMMA_MMA ||
                                       issue_opcode == `OP_WGMMA_LOAD ||
                                       issue_opcode == `OP_WGMMA_STORE);
    wire issue1_wgmma = issue1_valid && (issue1_opcode == `OP_WGMMA_MMA ||
                                         issue1_opcode == `OP_WGMMA_LOAD ||
                                         issue1_opcode == `OP_WGMMA_STORE);
    wire issue_wgmma_mma = issue_valid && (issue_opcode == `OP_WGMMA_MMA);
    wire issue1_wgmma_mma = issue1_valid && (issue1_opcode == `OP_WGMMA_MMA);
    wire issue_wgmma_wait = issue_wgmma_mma && (issue_func == `WGMMA_WAIT_GROUP);
    wire issue1_wgmma_wait = issue1_wgmma_mma && (issue1_func == `WGMMA_WAIT_GROUP);

    // WGMMA backpressure: stall if WGMMA unit not ready
    wire wgmma_stall = (issue_wgmma || issue1_wgmma) && !wgmma_ready;

    assign frontend_flush = (issue_valid && issue_exit_op && (INIT_WARPS == 1));

    // Branches also use ALU to compute zero flag for condition check
    wire alu_issue0 = issue_valid && (issue_alu_op || issue_branch_op);
    wire alu_issue1 = issue1_valid && (issue1_alu_op || issue1_branch_op);
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
    wire video_issue0 = issue_valid && issue_video_op;
    wire video_issue1 = issue1_valid && issue1_video_op;
    wire special_reg_issue0 = issue_valid && issue_special_reg;
    wire special_reg_issue1 = issue1_valid && issue1_special_reg;
    wire tex_issue0 = issue_valid && issue_tex_op;
    wire tex_issue1 = issue1_valid && issue1_tex_op;

    wire alu_issue = alu_issue0 || alu_issue1;
    wire mul_issue = mul_issue0 || mul_issue1;
    wire fpu32_issue = fpu32_issue0 || fpu32_issue1;
    wire fpu64_issue = fpu64_issue0 || fpu64_issue1;
    wire fp16_issue = fp16_issue0 || fp16_issue1;
    wire sfu_issue = sfu_issue0 || sfu_issue1;
    wire shfl_issue = shfl_issue0 || shfl_issue1;
    wire video_issue = video_issue0 || video_issue1;
    wire special_reg_issue = special_reg_issue0 || special_reg_issue1;
    wire tex_issue = tex_issue0 || tex_issue1;

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

    // FPU32 pipeline tracking (1 stage for 1-cycle simd_fpu latency)
    reg [WARP_ID_W-1:0] fpu32_warp_pipe [0:0];
    reg [4:0]           fpu32_rd_pipe [0:0];
    reg [NUM_LANES-1:0] fpu32_mask_pipe [0:0];

    // FPU64 pipeline tracking (4 stages)
    reg [WARP_ID_W-1:0] fpu64_warp_pipe [0:4];
    reg [4:0]           fpu64_rd_pipe [0:4];
    reg [NUM_LANES-1:0] fpu64_mask_pipe [0:4];

    // FP16 pipeline tracking (2 stages)
    reg [WARP_ID_W-1:0] fp16_warp_pipe [0:2];  // Extended to 3 stages for 3-cycle FP16 latency
    reg [4:0]           fp16_rd_pipe [0:2];
    reg [NUM_LANES-1:0] fp16_mask_pipe [0:2];

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

    // WGMMA operation tracking (for scoreboard)
    reg [WARP_ID_W-1:0] wgmma_pending_warp;
    reg [4:0]           wgmma_pending_rd;
    reg                 wgmma_pending_valid;

    // Shuffle pipeline tracking (1 stage delay for proper writeback timing)
    reg [WARP_ID_W-1:0] shuffle_warp_pipe;
    reg [4:0]           shuffle_rd_pipe;
    reg [NUM_LANES-1:0] shuffle_mask_pipe;
    reg [SIMD_WIDTH-1:0] shuffle_result_pipe;
    reg                 shuffle_valid_pipe;

    // Video SIMD pipeline tracking (2 stages for video_unit latency)
    reg [WARP_ID_W-1:0] video_warp_pipe [0:1];
    reg [4:0]           video_rd_pipe [0:1];
    reg [NUM_LANES-1:0] video_mask_pipe [0:1];

    // Texture unit pipeline tracking (variable latency - use pending registers)
    reg [WARP_ID_W-1:0] tex_warp_pending;
    reg [4:0]           tex_rd_pending;
    reg [NUM_LANES-1:0] tex_mask_pending;
    reg                 tex_pending_valid;
    reg [127:0]         tex_result_latched;
    reg                 tex_result_valid_latched;

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
    wire [WB_PKT_W-1:0] video_wbq_in, video_wbq_out;
    wire [WB_PKT_W-1:0] special_wbq_out;
    wire                 alu_wbq_push, alu_wbq_pop, alu_wbq_full, alu_wbq_empty;
    wire                 mul_wbq_push, mul_wbq_pop, mul_wbq_full, mul_wbq_empty;
    wire                 fpu32_wbq_push, fpu32_wbq_pop, fpu32_wbq_full, fpu32_wbq_empty;
    wire                 fpu64_wbq_push, fpu64_wbq_pop, fpu64_wbq_full, fpu64_wbq_empty;
    wire                 fp16_wbq_push, fp16_wbq_pop, fp16_wbq_full, fp16_wbq_empty;
    wire                 sfu_wbq_push, sfu_wbq_pop, sfu_wbq_full, sfu_wbq_empty;
    wire                 shfl_wbq_push, shfl_wbq_pop, shfl_wbq_full, shfl_wbq_empty;
    wire                 video_wbq_push, video_wbq_pop, video_wbq_full, video_wbq_empty;
    wire                 special_wbq_pop, special_wbq_full, special_wbq_empty;

    wire [WARP_ID_W-1:0] alu_wbq_warp;
    wire [WARP_ID_W-1:0] mul_wbq_warp;
    wire [WARP_ID_W-1:0] fpu32_wbq_warp;
    wire [WARP_ID_W-1:0] fpu64_wbq_warp;
    wire [WARP_ID_W-1:0] fp16_wbq_warp;
    wire [WARP_ID_W-1:0] sfu_wbq_warp;
    wire [WARP_ID_W-1:0] shfl_wbq_warp;
    wire [WARP_ID_W-1:0] video_wbq_warp;
    wire [WARP_ID_W-1:0] special_wbq_warp;
    wire [4:0]           alu_wbq_rd;
    wire [4:0]           mul_wbq_rd;
    wire [4:0]           fpu32_wbq_rd;
    wire [4:0]           fpu64_wbq_rd;
    wire [4:0]           fp16_wbq_rd;
    wire [4:0]           sfu_wbq_rd;
    wire [4:0]           shfl_wbq_rd;
    wire [4:0]           video_wbq_rd;
    wire [4:0]           special_wbq_rd;
    wire [NUM_LANES-1:0] alu_wbq_mask;
    wire [NUM_LANES-1:0] mul_wbq_mask;
    wire [NUM_LANES-1:0] fpu32_wbq_mask;
    wire [NUM_LANES-1:0] fpu64_wbq_mask;
    wire [NUM_LANES-1:0] fp16_wbq_mask;
    wire [NUM_LANES-1:0] sfu_wbq_mask;
    wire [NUM_LANES-1:0] shfl_wbq_mask;
    wire [NUM_LANES-1:0] video_wbq_mask;
    wire [NUM_LANES-1:0] special_wbq_mask;
    wire [SIMD_WIDTH-1:0] alu_wbq_data;
    wire [SIMD_WIDTH-1:0] mul_wbq_data;
    wire [SIMD_WIDTH-1:0] fpu32_wbq_data;
    wire [SIMD_WIDTH-1:0] fpu64_wbq_data;
    wire [SIMD_WIDTH-1:0] fp16_wbq_data;
    wire [SIMD_WIDTH-1:0] sfu_wbq_data;
    wire [SIMD_WIDTH-1:0] shfl_wbq_data;
    wire [SIMD_WIDTH-1:0] video_wbq_data;
    wire [SIMD_WIDTH-1:0] special_wbq_data;

    reg [ALU_WBQ_COUNT_W-1:0] alu_inflight;
    reg [MUL_WBQ_COUNT_W-1:0] mul_inflight;
    reg [FPU32_WBQ_COUNT_W-1:0] fpu32_inflight;
    reg [FPU64_WBQ_COUNT_W-1:0] fpu64_inflight;
    reg [FP16_WBQ_COUNT_W-1:0] fp16_inflight;
    reg [SFU_WBQ_COUNT_W-1:0] sfu_inflight;
    reg [SHFL_WBQ_COUNT_W-1:0] shfl_inflight;
    reg [VIDEO_WBQ_COUNT_W-1:0] video_inflight;

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
    localparam FETCH_PIPE_DEPTH = 2;  // Support 2 in-flight fetches (same-cycle response works with icache bypass)
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

    reg [31:0] fetch_debug_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fetch_arb_ptr <= 0;
            // warp_inst_buf_valid is managed in the buffer always block
            fetch_debug_cnt <= 0;
        end else begin
            // Debug first few fetch cycles
            if (fetch_debug_cnt < 20) begin
                `ifdef SIMULATION
                $display("[SM%0d FETCH] req=%b ready=%b fire=%b warp_valid=%04b needs_fetch=%04b pending=%04b buf_valid=%04b",
                         SM_ID, fetch_req, icache_ready, fetch_fire,
                         warp_valid, warp_needs_fetch, warp_fetch_pending, warp_inst_buf_valid);
                `endif
                fetch_debug_cnt <= fetch_debug_cnt + 1;
            end
            // Round-robin update on successful fetch request
            if (fetch_fire) begin
                fetch_arb_ptr <= (fetch_warp_id + 1) % NUM_WARPS;
            end
        end
    end

    // Combinatorial same-cycle hit detection
    // When icache_valid comes back in the same cycle as fetch_fire, the fetch pipeline
    // isn't updated yet (non-blocking assigns), so we need to detect this case.
    // same_cycle_hit is true when the response comes back before the pipeline is filled.
    wire same_cycle_hit = fetch_fire && icache_valid && (fetch_pipe_valid == 0);

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
        end else if (kernel_start) begin
            // Clear fetch pipeline state on kernel start
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
            // Handle both early response (1-cycle latency) and normal delayed response
            if (icache_valid && fetch_pipe_valid[0] && !fetch_pipe_valid[FETCH_PIPE_DEPTH-1]) begin
                // Early response: request is in stage 0
                warp_fetch_pending[fetch_pipe_warp[0]] <= 1'b0;
            end else if (icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1]) begin
                // Delayed response: request reached final stage
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
    // Handle responses at various pipeline depths
    // delayed_response_valid: response comes after FETCH_PIPE_DEPTH cycles (normal case)
    wire delayed_response_valid = icache_valid && fetch_pipe_valid[FETCH_PIPE_DEPTH-1];
    // early_response_valid: response comes after just 1 cycle (testbench with 1-cycle latency)
    // In this case, the request is in pipe stage 0 but not yet in the final stage
    wire early_response_valid = icache_valid && fetch_pipe_valid[0] && !fetch_pipe_valid[FETCH_PIPE_DEPTH-1];
    // Choose the right warp ID based on which stage has the response
    wire [WARP_ID_W-1:0] fill_warp_id = same_cycle_hit ? fetch_warp_id :
                                         early_response_valid ? fetch_pipe_warp[0] :
                                         fetch_pipe_warp[FETCH_PIPE_DEPTH-1];
    wire fill_valid = same_cycle_hit || delayed_response_valid || early_response_valid;

    // DEBUG: Track fetch pipeline state - disabled for faster simulation
    `ifdef DEBUG_FETCH
    always @(posedge clk) begin
        if (fetch_fire)
            `ifdef SIMULATION
            $display("[%0t SM%0d FETCH] fetch_fire: warp=%0d pc=0x%08h imem_req=%b", $time, SM_ID, fetch_warp_id, warp_fetch_pc[fetch_warp_id], imem_req);
            `endif
        if (icache_valid)
            `ifdef SIMULATION
            $display("[%0t SM%0d FETCH] icache_valid: pipe_valid=%b imem_data=0x%016h icache_data=0x%08h", $time, SM_ID, fetch_pipe_valid, imem_data, icache_data);
            `endif
        if (fill_valid)
            `ifdef SIMULATION
            $display("[%0t SM%0d FETCH] fill_valid: warp=%0d early=%b delayed=%b", $time, SM_ID, fill_warp_id, early_response_valid, delayed_response_valid);
            `endif
    end
    `endif

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
            // Clear instruction buffer on new kernel start (prevents stale instructions)
            if (kernel_start) begin
                warp_inst_buf_valid <= 0;
            end else begin
            // Debug: trace consume/fill
            `ifdef SIMULATION
            if (|warp_inst_consume || |warp_fill)
                $display("[%0t SM%0d BUF] consume=%b fill=%b buf_valid=%b",
                         $time, SM_ID, warp_inst_consume, warp_fill, warp_inst_buf_valid);
            `endif
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
            end else if (early_response_valid) begin
                // Early response (1-cycle latency): use warp ID from pipeline stage 0
                warp_inst_buf[fetch_pipe_warp[0]] <= icache_data;
            end else if (delayed_response_valid) begin
                // Delayed response: use warp ID from final pipeline stage
                warp_inst_buf[fetch_pipe_warp[FETCH_PIPE_DEPTH-1]] <= icache_data;
            end

            // PC advances when fetch request is sent (not on response)
            // This ensures PC points to the NEXT instruction to fetch
            if (fetch_fire) begin
                warp_fetch_pc[fetch_warp_id] <= warp_fetch_pc[fetch_warp_id] + 4;
            end
            end // end of else (not kernel_start)
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
            assign pd_is_compute[pd_i] = (op == `OP_ALU) || (op == `OP_ALU_IMM) || (op == `OP_MUL) ||
                                         (op == `OP_DIV) ||  // Added DIV to compute path
                                         (op == `OP_FP32_ARITH) || (op == `OP_FP16_ARITH) ||
                                         (op == `OP_SFU) || (op == `OP_MOV_SPECIAL) ||
                                         (op == `OP_MOV_IMM) || (op == `OP_SETP) ||
                                         (op == `OP_CVT) || (op == `OP_NOP) || (op == `OP_VIDEO);
            assign pd_is_tensor[pd_i]  = (op == `OP_WMMA_MMA) ||
                                         (op == `OP_WGMMA_MMA) || (op == `OP_WGMMA_LOAD) ||
                                         (op == `OP_WGMMA_STORE);
            assign pd_is_memory[pd_i]  = (op == `OP_LD_GLOBAL) || (op == `OP_ST_GLOBAL) ||
                                         (op == `OP_LD_SHARED) || (op == `OP_ST_SHARED) ||
                                         (op == `OP_LD_LOCAL) || (op == `OP_LD_PARAM) ||
                                         (op == `OP_ATOM) || (op == `OP_RED) ||
                                         (op == `OP_PREFETCH) || (op == `OP_CPASYNC);
            assign pd_is_branch[pd_i]  = (op == `OP_BRANCH) || (op == `OP_EXIT) ||
                                         (op == `OP_BAR_SYNC) || (op == `OP_MEMBAR);
            assign pd_writes_reg[pd_i] = (op != `OP_ST_GLOBAL) && (op != `OP_ST_SHARED) &&
                                         (op != `OP_BRANCH) && (op != `OP_EXIT) &&
                                         (op != `OP_NOP) && (op != `OP_BAR_SYNC) &&
                                         (op != `OP_MEMBAR);
        end
    endgenerate

    // Scheduler Instantiation
    // Note: Use `SCHED_LANES from gpu_defines.vh (2 default, 4 with GPU_PROFILE_HPC)
    localparam SCHED_LANES = `SCHED_LANES;  // Pipeline width (configurable)
    wire [SCHED_LANES-1:0] sched_issue_valid_mask;
    wire [WARP_ID_W-1:0] sched_issue_warp_id [0:SCHED_LANES-1];
    wire [31:0] sched_issue_inst [0:SCHED_LANES-1];
    wire [2:0] sched_issue_pipe [0:SCHED_LANES-1];

    //------------------------------------------------------------------------
    // Memory Pipeline In-Flight Counter
    // Track memory instructions between scheduler and memory interface
    // Prevent issuing new memory ops when one is already in the pipeline
    //------------------------------------------------------------------------
    reg [2:0] mem_pipe_inflight;  // Count of memory ops in scheduler->mem pipeline
    wire sched_issues_memory = sched_issue_valid_mask[0] && pd_is_memory[sched_issue_warp_id[0]];
    wire mem_response_complete = gmem_resp_valid || smem_resp_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_pipe_inflight <= 3'd0;
        end else begin
            case ({sched_issues_memory, mem_response_complete})
                2'b10: mem_pipe_inflight <= mem_pipe_inflight + 3'd1;  // Issue, no response
                2'b01: mem_pipe_inflight <= (mem_pipe_inflight > 0) ? mem_pipe_inflight - 3'd1 : 3'd0;  // Response, no issue
                // 2'b11: no change (issue and response same cycle)
                // 2'b00: no change
                default: ;
            endcase
        end
    end

    // Pipeline readiness signals (simplified)
    wire pipe_compute0_ready = 1'b1; // Pipeline always accepts unless stall logic says otherwise
    wire pipe_compute1_ready = 1'b1;
    // Tensor pipeline ready: both WMMA and WGMMA units must be ready
    // WGMMA shares the tensor pipeline, so stall if WGMMA unit is not ready
    wire pipe_tensor_ready   = !tensor_issue_full && wgmma_ready;
    // Memory pipeline ready only if no memory ops in flight AND not stalled
    // Also gate with ace_ready for cp.async backpressure (OP_CPASYNC is classified as memory)
    wire pipe_memory_ready   = (mem_pipe_inflight == 0) && !issue_stall_mem && ace_ready;
    // Branch pipeline is always ready - per-warp stall (warp_stalled_branch) handles flow control
    wire pipe_branch_ready   = 1'b1;

    //------------------------------------------------------------------------
    // Scheduler Selection: Blackwell (configurable) or Advanced (dual-issue)
    //------------------------------------------------------------------------
`ifdef USE_BLACKWELL_SCHEDULER
    blackwell_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_SCHEDULERS(SCHED_LANES)  // Match pipeline width
    ) u_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_diverged({NUM_WARPS{1'b0}}), // Todo: connect to CFU
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
`else
    advanced_warp_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_ISSUE(SCHED_LANES)
    ) u_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_diverged({NUM_WARPS{1'b0}}), // Todo: connect to CFU
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
`endif

    // Map Scheduler Output to Pipeline Signals
    // Replaces dec0_fire / dec1_fire logic
    assign issue0_fire = sched_issue_valid_mask[0];
    assign issue1_fire = sched_issue_valid_mask[1];

    // DEBUG: Scheduler output
    always @(posedge clk) begin
        `ifdef SIMULATION
        if (|warp_inst_buf_valid)
            $display("[%0t SM%0d SCHED] issue0_fire=%b buf_valid=%b sched_mask=%b",
                     $time, SM_ID, issue0_fire, warp_inst_buf_valid, sched_issue_valid_mask);
        `endif
    end

    // We reuse the 'dec0' pipeline registers to hold the scheduled instructions
    // effectively merging Decode/Issue stages into one logical flow handled by scheduler+decoder
    // Note: This overrides the previous 'dec0_warp_id <= ifq_warp_head' logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec0_valid <= 0;
            dec1_valid <= 0;
        end else begin
            // Debug: trace issue0_fire
            `ifdef SIMULATION
            if (issue0_fire)
                $display("[%0t SM%0d] SCHED: warp=%0d pc=0x%04x inst=0x%08x branch_flush=%b",
                         $time, SM_ID, sched_issue_warp_id[0], warp_pc[sched_issue_warp_id[0]],
                         sched_issue_inst[0], branch_flush_dec0);
            `endif
            // Flush decode stage if branch taken for same warp
            if (branch_flush_dec0) begin
                dec0_valid <= 0;
            end else begin
                dec0_valid <= issue0_fire;
                if (issue0_fire) begin
                    dec0_warp_id <= sched_issue_warp_id[0];
                    dec0_instruction <= sched_issue_inst[0];
                    dec0_pc <= warp_pc[sched_issue_warp_id[0]]; // Capture current PC
                    // Advance PC at scheduling time for non-branch instructions
                    // (branches will override PC when they resolve)
                    if (!pd_is_branch[sched_issue_warp_id[0]]) begin
                        warp_pc[sched_issue_warp_id[0]] <= warp_pc[sched_issue_warp_id[0]] + 4;
                    end
                    // Note: warp_stalled_branch is set in main always block for branch scheduling
                end
            end

            // Flush decode stage 1 if branch taken for same warp
            if (branch_flush_dec1) begin
                dec1_valid <= 0;
            end else begin
                dec1_valid <= issue1_fire;
                if (issue1_fire) begin
                    dec1_warp_id <= sched_issue_warp_id[1];
                    dec1_instruction <= sched_issue_inst[1];
                    dec1_pc <= warp_pc[sched_issue_warp_id[1]];
                    `ifdef SIMULATION
                    $display("[%0t SM%0d] ISSUE1_FIRE: warp=%0d pc=0x%08x inst=0x%08x",
                             $time, SM_ID, sched_issue_warp_id[1], warp_pc[sched_issue_warp_id[1]], sched_issue_inst[1]);
                    `endif
                    if (!pd_is_branch[sched_issue_warp_id[1]]) begin
                        warp_pc[sched_issue_warp_id[1]] <= warp_pc[sched_issue_warp_id[1]] + 4;
                    end
                    // Note: warp_stalled_branch is set in main always block for branch scheduling
                end
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
    
    // Extended decoder signals (Lane 0)
    wire                 dec_mem_param, dec_mem_const, dec_mem_local, dec_mem_vector;
    wire [1:0]           dec_vec_size;
    wire                 dec_reduce_op;
    wire                 dec_vote_op, dec_redux_op;
    wire                 dec_wmma_load, dec_wmma_store, dec_mma_op;
    wire                 dec_call_op, dec_membar_op;
    wire                 dec_video_op;
    wire                 dec_tex_op, dec_txq_op, dec_surf_ld, dec_surf_st, dec_surf_red;
    wire                 dec_cpasync_op, dec_prefetch_op, dec_wgmma_load, dec_wgmma_store, dec_wgmma_mma;
    wire [2:0]           dec_cache_hint;
    wire                 dec_mbarrier_op;
    wire                 dec_bar_warp_sync;
    wire                 dec_cache_policy_op;
    wire                 dec_stack_op;
    wire                 dec_debug_op;
    wire                 dec_misc_op;
    wire                 dec_st_async_op;
    wire                 dec_multimem_op;
    wire                 dec_barrier_cluster_op;

    // Extended decoder signals (Lane 1)
    wire dec1_fp32_special;
    wire dec1_wmma_mma;
    wire dec1_shfl_op;
    wire                 dec1_mem_param, dec1_mem_const, dec1_mem_local, dec1_mem_vector;
    wire [1:0]           dec1_vec_size;
    wire                 dec1_vote_op, dec1_redux_op;
    wire                 dec1_wmma_load, dec1_wmma_store, dec1_mma_op;
    wire                 dec1_call_op, dec1_membar_op;
    wire                 dec1_video_op;
    wire                 dec1_tex_op, dec1_txq_op, dec1_surf_ld, dec1_surf_st, dec1_surf_red;
    wire                 dec1_cpasync_op, dec1_prefetch_op, dec1_wgmma_load, dec1_wgmma_store, dec1_wgmma_mma;
    wire [2:0]           dec1_cache_hint;
    wire                 dec1_mbarrier_op;
    wire                 dec1_bar_warp_sync;
    wire                 dec1_cache_policy_op;
    wire                 dec1_stack_op;
    wire                 dec1_debug_op;
    wire                 dec1_misc_op;
    wire                 dec1_st_async_op;
    wire                 dec1_multimem_op;
    wire                 dec1_barrier_cluster_op;

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
        .cvt_op      (dec_cvt_op),
        .fp32_special(dec_fp32_special),
        .wmma_mma    (dec_wmma_mma),
        .mem_read    (dec_mem_read),
        .mem_write   (dec_mem_write),
        .mem_shared  (dec_mem_shared),
        .branch_op   (dec_branch_op),
        .sync_op     (dec_sync_op),
        .special_reg (dec_special_reg),
        .exit_op     (dec_exit_op),
        .atomic_op   (dec_atomic_op),
        .shfl_op     (dec_shfl_op),
        .reg_write   (dec_reg_write),
        .pred_write  (),
        .pred_addr   (),
        .mem_param   (dec_mem_param),
        .mem_const   (dec_mem_const),
        .mem_local   (dec_mem_local),
        .mem_vector  (dec_mem_vector),
        .vec_size    (dec_vec_size),
        .reduce_op   (dec_reduce_op),
        .vote_op     (dec_vote_op),
        .redux_op    (dec_redux_op),
        .wmma_load   (dec_wmma_load),
        .wmma_store  (dec_wmma_store),
        .mma_op      (dec_mma_op),
        .call_op     (dec_call_op),
        .membar_op   (dec_membar_op),
        .video_op    (dec_video_op),
        .tex_op      (dec_tex_op),
        .txq_op      (dec_txq_op),
        .surf_ld     (dec_surf_ld),
        .surf_st     (dec_surf_st),
        .surf_red    (dec_surf_red),
        .cpasync_op  (dec_cpasync_op),
        .prefetch_op (dec_prefetch_op),
        .wgmma_load  (dec_wgmma_load),
        .wgmma_store (dec_wgmma_store),
        .wgmma_mma   (dec_wgmma_mma),
        .cache_hint  (dec_cache_hint),
        .mbarrier_op (dec_mbarrier_op),
        .bar_warp_sync (dec_bar_warp_sync),
        .cache_policy_op (dec_cache_policy_op),
        .stack_op (dec_stack_op),
        .debug_op (dec_debug_op),
        .misc_op (dec_misc_op),
        .st_async_op (dec_st_async_op),
        .multimem_op (dec_multimem_op),
        .barrier_cluster_op (dec_barrier_cluster_op)
    );

    //------------------------------------------------------------------------
    // Instruction Decoder (lane 1)
    //------------------------------------------------------------------------
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
        .cvt_op      (dec1_cvt_op),
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
        .pred_addr   (),
        .mem_param   (dec1_mem_param),
        .mem_const   (dec1_mem_const),
        .mem_local   (dec1_mem_local),
        .mem_vector  (dec1_mem_vector),
        .vec_size    (dec1_vec_size),
        .reduce_op   (dec1_reduce_op),
        .vote_op     (dec1_vote_op),
        .redux_op    (dec1_redux_op),
        .wmma_load   (dec1_wmma_load),
        .wmma_store  (dec1_wmma_store),
        .mma_op      (dec1_mma_op),
        .call_op     (dec1_call_op),
        .membar_op   (dec1_membar_op),
        .video_op    (dec1_video_op),
        .tex_op      (dec1_tex_op),
        .txq_op      (dec1_txq_op),
        .surf_ld     (dec1_surf_ld),
        .surf_st     (dec1_surf_st),
        .surf_red    (dec1_surf_red),
        .cpasync_op  (dec1_cpasync_op),
        .prefetch_op (dec1_prefetch_op),
        .wgmma_load  (dec1_wgmma_load),
        .wgmma_store (dec1_wgmma_store),
        .wgmma_mma   (dec1_wgmma_mma),
        .cache_hint  (dec1_cache_hint),
        .mbarrier_op (dec1_mbarrier_op),
        .bar_warp_sync (dec1_bar_warp_sync),
        .cache_policy_op (dec1_cache_policy_op),
        .stack_op (dec1_stack_op),
        .debug_op (dec1_debug_op),
        .misc_op (dec1_misc_op),
        .st_async_op (dec1_st_async_op),
        .multimem_op (dec1_multimem_op),
        .barrier_cluster_op (dec1_barrier_cluster_op)
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
            issue_bar_warp_sync <= 1'b0;
            issue_tex_op <= 1'b0;
            issue_cache_policy_op <= 1'b0;
            issue_stack_op <= 1'b0;
            issue_debug_op <= 1'b0;
            issue_misc_op <= 1'b0;
            issue_st_async_op <= 1'b0;
            issue_multimem_op <= 1'b0;
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
            issue1_bar_warp_sync <= 1'b0;
            issue1_tex_op <= 1'b0;
            issue1_cache_policy_op <= 1'b0;
            issue1_stack_op <= 1'b0;
            issue1_debug_op <= 1'b0;
            issue1_misc_op <= 1'b0;
            issue1_st_async_op <= 1'b0;
            issue1_multimem_op <= 1'b0;
        end else begin
            // Debug: always trace - unconditional
            if (dec_valid || issue_valid)
                `ifdef SIMULATION
                $display("[%0t SM%0d] ISSUE_STAGE: dec_valid=%b dec0_valid=%b issue_valid=%b rst_n=%b",
                         $time, SM_ID, dec_valid, dec0_valid, issue_valid, rst_n);
                `endif
            // Issue stage triggers when decoder output is valid (dec_valid)
            // This ensures decoder has finished processing before we latch its outputs
            // Suppress issue if branch flush is active for this warp
            issue_valid <= dec_valid && !branch_flush_dec0;
            issue1_valid <= dec1_dec_valid && !branch_flush_dec1;  // Set from decoder output
            if (dec_valid && !branch_flush_dec0) begin
                // With the new scheduler flow, always use lane 0's decoder output
                // (the old lane0_ready-based selection doesn't apply here)
                `ifdef SIMULATION
                $display("[%0t SM%0d] DECODE->ISSUE: dec0_pc=0x%04x opcode=0x%02x warp_pc=0x%04x",
                         $time, SM_ID, dec0_pc, dec_opcode, warp_pc[dec0_warp_id]);
                `endif
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
                    // Use merged mask when reconverging, normal mask otherwise
                    issue_mask <= dec0_at_reconverge ? dec0_merged_mask : warp_mask[dec0_warp_id];
                    issue_alu_op <= dec_alu_op || dec_cvt_op;  // CVT routed through ALU
                    issue_mul_op <= dec_mul_op;
                    issue_div_op <= dec_div_op;
                    issue_fp32_op <= dec_fp32_op;
                    issue_fp64_op <= dec_fp64_op;
                    issue_fp16_op <= dec_fp16_op;
                    issue_sfu_op <= dec_sfu_op;
                    issue_tensor_op <= dec_tensor_op;
                    issue_video_op <= dec_video_op;  // Video SIMD unit (separate path)
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
                    issue_bar_warp_sync <= dec_bar_warp_sync;
                    issue_tex_op <= dec_tex_op || dec_txq_op || dec_surf_ld || dec_surf_st || dec_surf_red;
                    issue_cache_policy_op <= dec_cache_policy_op;
                    issue_stack_op <= dec_stack_op;
                    issue_debug_op <= dec_debug_op;
                    issue_misc_op <= dec_misc_op;
                    issue_st_async_op <= dec_st_async_op;
                    issue_multimem_op <= dec_multimem_op;
                    issue_barrier_cluster_op <= dec_barrier_cluster_op;

                    // Handle decode-stage reconvergence: update mask and pop stack
                    if (dec0_at_reconverge) begin
                        `ifdef SIMULATION
                        $display("[SM%0d] DECODE RECONVERGENCE at PC=0x%04x: merging mask=0x%08x with waiting=0x%08x -> 0x%08x",
                                 SM_ID, dec0_pc, warp_mask[dec0_warp_id],
                                 sm_div_stack_mask[dec0_warp_id][dec0_top_idx], dec0_merged_mask);
                        `endif
                        warp_mask[dec0_warp_id] <= dec0_merged_mask;
                        sm_div_stack_ptr[dec0_warp_id] <= dec0_div_ptr - 2'd1;
                    end
                end
            end

            // Slot 1: capture decoder outputs when decoder output is valid (NOT at issue1_fire time)
            // This matches the timing of issue1_valid which is set from dec1_dec_valid
            if (dec1_dec_valid && !branch_flush_dec1) begin
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
                issue1_alu_op <= dec1_alu_op || dec1_cvt_op;  // CVT routed through ALU
                issue1_mul_op <= dec1_mul_op;
                issue1_div_op <= dec1_div_op;
                issue1_fp32_op <= dec1_fp32_op;
                issue1_fp64_op <= dec1_fp64_op;
                issue1_fp16_op <= dec1_fp16_op;
                issue1_sfu_op <= dec1_sfu_op;
                issue1_tensor_op <= dec1_tensor_op;
                issue1_video_op <= dec1_video_op;  // Video SIMD unit (separate path)
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
                issue1_bar_warp_sync <= dec1_bar_warp_sync;
                issue1_tex_op <= dec1_tex_op || dec1_txq_op || dec1_surf_ld || dec1_surf_st || dec1_surf_red;
                issue1_cache_policy_op <= dec1_cache_policy_op;
                issue1_stack_op <= dec1_stack_op;
                issue1_debug_op <= dec1_debug_op;
                issue1_misc_op <= dec1_misc_op;
                issue1_st_async_op <= dec1_st_async_op;
                issue1_multimem_op <= dec1_multimem_op;
                issue1_barrier_cluster_op <= dec1_barrier_cluster_op;
            end
        end
    end

    wire all_at_barrier = (&(warp_stalled_sync | ~warp_valid)) && barrier_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize scoreboards
            for (sb_init = 0; sb_init < NUM_WARPS; sb_init = sb_init + 1) begin
                scoreboard_busy[sb_init] <= 32'b0;
                pending_fu_count[sb_init] <= 4'b0;
            end
            barrier_pending <= 1'b0;
            // Initialize cluster barrier shared state
            cluster_barrier_thread_count <= 16'b0;
            cluster_local_arrive_count <= 16'b0;
            cluster_barrier_complete <= 1'b0;
            wgmma_pending_valid <= 1'b0;
            wgmma_pending_warp <= {WARP_ID_W{1'b0}};
            wgmma_pending_rd <= 5'b0;
        end else begin
            // Mark destination register as busy on issue
            // Note: Unlike RISC-V, CUDA/PTX R0 is a normal register, not hardwired to 0
            if (issue0_fire && issue0_reg_write_sel) begin
                scoreboard_busy[issue0_warp_sel][issue0_rd_sel] <= 1'b1;
            end
            if (issue1_fire && dec1_reg_write) begin
                scoreboard_busy[dec1_warp_id][dec1_rd] <= 1'b1;
            end

            // WGMMA MMA tracking: mark destination busy and track pending operation
            if (issue_wgmma_mma) begin
                wgmma_pending_valid <= 1'b1;
                wgmma_pending_warp <= issue_warp_id;
                wgmma_pending_rd <= issue_rd;
            end else if (issue1_wgmma_mma) begin
                wgmma_pending_valid <= 1'b1;
                wgmma_pending_warp <= issue1_warp_id;
                wgmma_pending_rd <= issue1_rd;
            end

            // Clear on writeback
            if (wb_valid) begin
                scoreboard_busy[wb_warp_id][wb_rd] <= 1'b0;
            end

            // Clear WGMMA pending when operation completes
            if (wgmma_done && wgmma_pending_valid) begin
                scoreboard_busy[wgmma_pending_warp][wgmma_pending_rd] <= 1'b0;
                wgmma_pending_valid <= 1'b0;
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

            // Barrier tracking: set when any sync_op issues; release when all warps reach it
            if (issue_valid && issue_sync_op) begin
                barrier_pending <= 1'b1;
                `ifdef SIMULATION
                $display("[SM%0d] barrier_pending set by issue0: sync_op=%b", SM_ID, issue_sync_op);
                `endif
            end
            if (issue1_valid && issue1_sync_op) begin
                barrier_pending <= 1'b1;
                `ifdef SIMULATION
                $display("[SM%0d] barrier_pending set by issue1: sync_op=%b", SM_ID, issue1_sync_op);
                `endif
            end
            if (all_at_barrier) begin
                barrier_pending <= 1'b0;
                `ifdef SIMULATION
                $display("[SM%0d] barrier_pending cleared by all_at_barrier", SM_ID);
                `endif
            end
            if (kernel_start)
                barrier_pending <= 1'b0;
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
            video_inflight <= {VIDEO_WBQ_COUNT_W{1'b0}};
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
            case ({video_issue, video_wbq_pop})
                2'b10: video_inflight <= video_inflight + 1'b1;
                2'b01: video_inflight <= video_inflight - 1'b1;
                default: video_inflight <= video_inflight;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Async Copy Tracking (cp.async) - Using Real async_copy_engine
    // The async_copy_engine handles actual memory transactions.
    // This tracking manages per-warp pending counts and wait stalls.
    //------------------------------------------------------------------------
    integer cp_w;
    reg [3:0] pending_val;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_stalled_async <= {NUM_WARPS{1'b0}};
            cp_async_wait_all <= {NUM_WARPS{1'b0}};
            for (cp_w = 0; cp_w < NUM_WARPS; cp_w = cp_w + 1) begin
                cp_async_pending[cp_w] <= 4'd0;
                cp_async_wait_threshold[cp_w] <= 4'd0;
            end
        end else if (kernel_start) begin
            warp_stalled_async <= {NUM_WARPS{1'b0}};
            cp_async_wait_all <= {NUM_WARPS{1'b0}};
            for (cp_w = 0; cp_w < NUM_WARPS; cp_w = cp_w + 1) begin
                cp_async_pending[cp_w] <= 4'd0;
                cp_async_wait_threshold[cp_w] <= 4'd0;
            end
        end else begin
            // Track which warp to decrement when ace completes (first warp with pending > 0)
            // This is a simplification; a full implementation would track warp IDs in queue
            for (cp_w = 0; cp_w < NUM_WARPS; cp_w = cp_w + 1) begin
                pending_val = cp_async_pending[cp_w];

                // New async copies (CA/CG/BULK variants) - increment pending
                if (issue_cpasync_copy && (issue_warp_id == cp_w[WARP_ID_W-1:0])) begin
                    pending_val = pending_val + 1'b1;
                end
                if (issue1_cpasync_copy && (issue1_warp_id == cp_w[WARP_ID_W-1:0])) begin
                    pending_val = pending_val + 1'b1;
                end

                // Decrement pending when async_copy_engine completes a copy
                // ace_smem_wr_en indicates a write to shared memory (completion)
                // Only decrement first warp with pending copies (warp 0 has priority)
                if (ace_smem_wr_en && pending_val > 0 && cp_w == 0) begin
                    // For single-warp or first-warp-priority, decrement warp 0
                    // A full implementation would use a FIFO to track warp ordering
                    pending_val = pending_val - 1'b1;
                end else if (ace_smem_wr_en && pending_val > 0 && cp_async_pending[0] == 0 && cp_w == 1) begin
                    // If warp 0 has no pending, try warp 1
                    pending_val = pending_val - 1'b1;
                end else if (ace_smem_wr_en && pending_val > 0 && cp_async_pending[0] == 0 && cp_async_pending[1] == 0 && cp_w == 2) begin
                    // If warps 0,1 have no pending, try warp 2
                    pending_val = pending_val - 1'b1;
                end else if (ace_smem_wr_en && pending_val > 0 && cp_async_pending[0] == 0 && cp_async_pending[1] == 0 && cp_async_pending[2] == 0 && cp_w == 3) begin
                    // If warps 0,1,2 have no pending, try warp 3
                    pending_val = pending_val - 1'b1;
                end

                cp_async_pending[cp_w] <= pending_val;

                // Release warps waiting on wait_group/wait_all
                // Use the real pending count from tracking
                if (warp_stalled_async[cp_w]) begin
                    if (cp_async_wait_all[cp_w]) begin
                        // wait_all: release when all copies complete
                        if (pending_val == 0) begin
                            warp_stalled_async[cp_w] <= 1'b0;
                            cp_async_wait_all[cp_w] <= 1'b0;
                        end
                    end else if (pending_val <= cp_async_wait_threshold[cp_w]) begin
                        // wait_group N: release when <= N groups pending
                        warp_stalled_async[cp_w] <= 1'b0;
                    end
                end
            end

            // wait_group / wait_all handling - stall the requesting warp
            if (issue_cpasync_wait) begin
                warp_stalled_async[issue_warp_id] <= 1'b1;
                cp_async_wait_threshold[issue_warp_id] <= issue_imm16[3:0];
                cp_async_wait_all[issue_warp_id] <= 1'b0;
            end else if (issue_cpasync_wait_all) begin
                warp_stalled_async[issue_warp_id] <= 1'b1;
                cp_async_wait_threshold[issue_warp_id] <= 4'd0;
                cp_async_wait_all[issue_warp_id] <= 1'b1;
            end
            if (issue1_cpasync_wait) begin
                warp_stalled_async[issue1_warp_id] <= 1'b1;
                cp_async_wait_threshold[issue1_warp_id] <= issue1_imm16[3:0];
                cp_async_wait_all[issue1_warp_id] <= 1'b0;
            end else if (issue1_cpasync_wait_all) begin
                warp_stalled_async[issue1_warp_id] <= 1'b1;
                cp_async_wait_threshold[issue1_warp_id] <= 4'd0;
                cp_async_wait_all[issue1_warp_id] <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // WGMMA Wait Tracking (wgmma.wait_group)
    // Manages warp stalls for WGMMA wait_group operations.
    //------------------------------------------------------------------------
    integer wgmma_w;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warp_stalled_wgmma <= {NUM_WARPS{1'b0}};
            for (wgmma_w = 0; wgmma_w < NUM_WARPS; wgmma_w = wgmma_w + 1) begin
                wgmma_wait_threshold[wgmma_w] <= 4'd0;
            end
        end else if (kernel_start) begin
            warp_stalled_wgmma <= {NUM_WARPS{1'b0}};
            for (wgmma_w = 0; wgmma_w < NUM_WARPS; wgmma_w = wgmma_w + 1) begin
                wgmma_wait_threshold[wgmma_w] <= 4'd0;
            end
        end else begin
            // Release warps waiting on wgmma.wait_group when pending_ops condition met
            for (wgmma_w = 0; wgmma_w < NUM_WARPS; wgmma_w = wgmma_w + 1) begin
                if (warp_stalled_wgmma[wgmma_w]) begin
                    // Release when wgmma_pending_ops <= wait_threshold
                    if (wgmma_pending_ops <= wgmma_wait_threshold[wgmma_w]) begin
                        warp_stalled_wgmma[wgmma_w] <= 1'b0;
                    end
                end
            end

            // wgmma.wait_group handling - stall the requesting warp
            if (issue_wgmma_wait) begin
                warp_stalled_wgmma[issue_warp_id] <= 1'b1;
                wgmma_wait_threshold[issue_warp_id] <= issue_imm16[3:0];
            end
            if (issue1_wgmma_wait) begin
                warp_stalled_wgmma[issue1_warp_id] <= 1'b1;
                wgmma_wait_threshold[issue1_warp_id] <= issue1_imm16[3:0];
            end
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
    wire [5:0] alu_issue_opcode = alu_use_slot0 ? issue_opcode : issue1_opcode;
    wire [15:0] alu_issue_imm16 = alu_use_slot0 ? issue_imm16 : issue1_imm16;
    wire alu_issue_use_imm = alu_use_slot0 ? issue_use_imm : issue1_use_imm;
    wire [SIMD_WIDTH-1:0] alu_op_a = alu_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] alu_op_b = alu_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;

    // For MOV_IMM, use 0 as operand_a so result = 0 + imm16 = imm16
    wire alu_is_mov_imm = (alu_issue_opcode == `OP_MOV_IMM);
    // For BRANCH, use OR with 0 to pass through register and set zero flag
    wire alu_is_branch = (alu_issue_opcode == `OP_BRANCH);
    wire [SIMD_WIDTH-1:0] alu_operand_a = alu_is_mov_imm ? {SIMD_WIDTH{1'b0}} : alu_op_a;
    wire [SIMD_WIDTH-1:0] alu_op_c = alu_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;
    wire [SIMD_WIDTH-1:0] alu_result_hi;
    wire [NUM_LANES-1:0]  alu_ovf;
    wire [NUM_LANES-1:0]  alu_cout;

    simd_alu u_simd_alu (
        .func       (alu_is_mov_imm ? `FUNC_ADD : (alu_is_branch ? `FUNC_OR : alu_issue_func)),
        .operand_a  (alu_operand_a),
        // For branch, use 0 as operand_b so result = ra | 0 = ra
        .operand_b  (alu_is_branch ? {SIMD_WIDTH{1'b0}} :
                     (alu_issue_use_imm ? {NUM_LANES{{16'b0, alu_issue_imm16}}} : alu_op_b)),
        .operand_c  (alu_op_c),
        .pred_in    ({NUM_LANES{1'b0}}), // TODO: Connect predicate register file
        .carry_in   ({NUM_LANES{1'b0}}), // TODO: Connect carry flag
        .lane_mask  (alu_issue_mask),
        .result     (alu_result),
        .result_hi  (alu_result_hi),
        .zero_flags (alu_zero),
        .neg_flags  (alu_neg),
        .ovf_flags  (alu_ovf),
        .carry_out  (alu_cout)
    );

    // ALU pipeline tracking (1 stage delay to avoid issue-stage race)
    reg [3:0] alu_debug_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alu_valid_pipe <= 1'b0;
            alu_warp_pipe <= 0;
            alu_rd_pipe <= 0;
            alu_mask_pipe <= 0;
            alu_result_pipe <= 0;
            alu_debug_cnt <= 0;
        end else begin
            alu_valid_pipe <= alu_issue;
            // DEBUG: trace every cycle for first 20 cycles of activity
            `ifdef SIMULATION
            $display("[%0t SM%0d] ALU_PIPE_DBG: alu_issue=%b alu_valid_pipe=%b issue_valid=%b issue_alu_op=%b",
                     $time, SM_ID, alu_issue, alu_valid_pipe, issue_valid, issue_alu_op);
            `endif
            if (alu_issue) begin
                alu_warp_pipe <= alu_issue_warp;
                alu_rd_pipe <= alu_issue_rd;
                alu_mask_pipe <= alu_issue_mask;
                alu_result_pipe <= alu_result;
                // DEBUG: trace ALU ops (increased limit for loop debugging)
                if (alu_debug_cnt < 50) begin
                    `ifdef SIMULATION
                    $display("[SM%0d] ALU: rd=R%0d ra=R%0d func=%0d imm=%b",
                        SM_ID, alu_issue_rd, issue_ra, alu_issue_func, alu_issue_use_imm);
                    `endif
                    `ifdef SIMULATION
                    $display("        op_a: lane0=0x%08x lane1=0x%08x lane15=0x%08x",
                        alu_op_a[31:0], alu_op_a[63:32], alu_op_a[511:480]);
                    `endif
                    `ifdef SIMULATION
                    $display("        result: lane0=0x%08x lane1=0x%08x lane15=0x%08x",
                        alu_result[31:0], alu_result[63:32], alu_result[511:480]);
                    `endif
                    alu_debug_cnt <= alu_debug_cnt + 1;
                end
            end
        end
    end

    assign alu_valid_out = alu_valid_pipe;

    //------------------------------------------------------------------------
    // Special Register Execution (MOV_SPECIAL - 1 cycle)
    // Reads special registers like tid.x, ctaid.x, ntid.x, nctaid.x, etc.
    //------------------------------------------------------------------------
    wire special_use_slot0 = special_reg_issue0;
    wire [WARP_ID_W-1:0] special_issue_warp = special_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] special_issue_rd = special_use_slot0 ? issue_rd : issue1_rd;
    wire [4:0] special_issue_ra = special_use_slot0 ? issue_ra : issue1_ra;  // Special reg code
    wire [NUM_LANES-1:0] special_issue_mask = special_use_slot0 ? issue_mask : issue1_mask;

    // Generate special register value
    // For %tid.x: each lane gets its lane index (0-31)
    // For other special registers: same value replicated to all lanes
    reg [31:0] special_reg_scalar;  // Scalar value for non-per-lane registers
    wire special_is_tid_x = (special_issue_ra == `SREG_TID_X);
    wire special_is_laneid = (special_issue_ra == `SREG_LANEID);

    always @(*) begin
        case (special_issue_ra)
            `SREG_TID_X:    special_reg_scalar = 32'd0;  // Not used - per-lane below
            `SREG_TID_Y:    special_reg_scalar = 32'd0;
            `SREG_TID_Z:    special_reg_scalar = 32'd0;
            `SREG_CTAID_X:  special_reg_scalar = block_id_regs[0];
            `SREG_CTAID_Y:  special_reg_scalar = block_id_regs[1];
            `SREG_CTAID_Z:  special_reg_scalar = block_id_regs[2];
            `SREG_NTID_X:   special_reg_scalar = block_dim_regs[0];
            `SREG_NTID_Y:   special_reg_scalar = block_dim_regs[1];
            `SREG_NTID_Z:   special_reg_scalar = block_dim_regs[2];
            `SREG_NCTAID_X: special_reg_scalar = grid_dim_regs[0];
            `SREG_NCTAID_Y: special_reg_scalar = grid_dim_regs[1];
            `SREG_NCTAID_Z: special_reg_scalar = grid_dim_regs[2];
            `SREG_WARPID:   special_reg_scalar = special_issue_warp;
            `SREG_SMID:     special_reg_scalar = SM_ID;
            `SREG_ACTIVEMASK: special_reg_scalar = {{(32-NUM_LANES){1'b0}}, warp_mask[special_issue_warp]};
            default:        special_reg_scalar = 32'd0;
        endcase
    end

    // Generate per-lane thread IDs for %tid.x, or replicate scalar for others
    // Lane IDs: lane 0=0, lane 1=1, ..., lane 31=31
    wire [SIMD_WIDTH-1:0] special_result;
    genvar sr_i;
    generate
        for (sr_i = 0; sr_i < NUM_LANES; sr_i = sr_i + 1) begin : gen_special_reg
            assign special_result[sr_i*32 +: 32] =
                special_is_tid_x ? (special_issue_warp * NUM_LANES + sr_i) :
                special_is_laneid ? sr_i[31:0] :
                special_reg_scalar;
        end
    endgenerate

    // Special register pipeline tracking (1 stage)
    reg                  special_valid_pipe;
    reg [WARP_ID_W-1:0]  special_warp_pipe;
    reg [4:0]            special_rd_pipe;
    reg [NUM_LANES-1:0]  special_mask_pipe;
    reg [SIMD_WIDTH-1:0] special_result_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            special_valid_pipe <= 1'b0;
            special_warp_pipe <= 0;
            special_rd_pipe <= 0;
            special_mask_pipe <= 0;
            special_result_pipe <= 0;
        end else begin
            special_valid_pipe <= special_reg_issue;
            if (special_reg_issue) begin
                special_warp_pipe <= special_issue_warp;
                special_rd_pipe <= special_issue_rd;
                special_mask_pipe <= special_issue_mask;
                special_result_pipe <= special_result;
                // DEBUG - show per-lane values to verify %tid.x
                `ifdef SIMULATION
                $display("[SM%0d] MOV_SPECIAL issued: warp=%0d rd=R%0d ra=%0d is_tid_x=%b",
                    SM_ID, special_issue_warp, special_issue_rd, special_issue_ra, special_is_tid_x);
                `endif
                `ifdef SIMULATION
                $display("        lane0=0x%08h lane1=0x%08h lane31=0x%08h",
                    special_result[31:0], special_result[63:32], special_result[1023:992]);
                `endif
            end
        end
    end

    wire special_valid_out = special_valid_pipe;

    // DEBUG: special writeback
    always @(posedge clk) begin
        if (special_valid_out) begin
            `ifdef SIMULATION
            $display("[SM%0d] MOV_SPECIAL result ready: warp=%0d rd=R%0d value=0x%08h",
                SM_ID, special_warp_pipe, special_rd_pipe, special_result_pipe[31:0]);
            `endif
        end
    end

    //------------------------------------------------------------------------
    // SIMD Multiplier
    //------------------------------------------------------------------------
    wire mul_use_slot0 = mul_issue0;
    wire mul_use_slot1 = mul_issue1;
    wire [WARP_ID_W-1:0] mul_issue_warp = mul_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] mul_issue_rd = mul_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] mul_issue_mask = mul_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] mul_issue_func = mul_use_slot0 ? issue_func : issue1_func;
    wire        mul_issue_is_div = mul_use_slot0 ? issue_div_op : issue1_div_op;
    wire [SIMD_WIDTH-1:0] mul_op_a = mul_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] mul_op_b = mul_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] mul_op_c = mul_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;

    simd_mul_unit u_simd_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (mul_issue),
        .is_div    (mul_issue_is_div),
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
        .func      (fpu32_issue_func),
        .rnd_mode  (2'b00),
        .ftz       (1'b0),
        .operand_a (fpu32_op_a),
        .operand_b (fpu32_op_b),
        .operand_c (fpu32_op_c),
        .valid_in  (fpu32_valid_in),
        .lane_mask (fpu32_issue_mask),
        .result    (fpu32_result),
        .valid_out (fpu32_valid_out),
        .overflow_flags (),
        .invalid_flags  ()
    );
    assign fpu32_ready = 1'b1;  // Always ready (pipelined)

    // FPU32 pipeline tracking (1 stage for 1-cycle simd_fpu latency)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fpu32_warp_pipe[0] <= 0;
            fpu32_rd_pipe[0] <= 0;
            fpu32_mask_pipe[0] <= 0;
        end else begin
            fpu32_warp_pipe[0] <= fpu32_valid_in ? fpu32_issue_warp : {WARP_ID_W{1'b0}};
            fpu32_rd_pipe[0] <= fpu32_valid_in ? fpu32_issue_rd : 5'b0;
            fpu32_mask_pipe[0] <= fpu32_valid_in ? fpu32_issue_mask : {NUM_LANES{1'b0}};
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

    // Extend 32-bit operands to 64-bit for FP64 unit
    wire [NUM_LANES*64-1:0] fpu64_op_a_ext;
    wire [NUM_LANES*64-1:0] fpu64_op_b_ext;
    wire [NUM_LANES*64-1:0] fpu64_op_c_ext;
    genvar fpu64_ext_i;
    generate
        for (fpu64_ext_i = 0; fpu64_ext_i < NUM_LANES; fpu64_ext_i = fpu64_ext_i + 1) begin : fpu64_ext
            assign fpu64_op_a_ext[fpu64_ext_i*64 +: 64] = {32'b0, fpu64_op_a[fpu64_ext_i*32 +: 32]};
            assign fpu64_op_b_ext[fpu64_ext_i*64 +: 64] = {32'b0, fpu64_op_b[fpu64_ext_i*32 +: 32]};
            assign fpu64_op_c_ext[fpu64_ext_i*64 +: 64] = {32'b0, fpu64_op_c[fpu64_ext_i*32 +: 32]};
        end
    endgenerate

    simd_fpu64 u_simd_fpu64 (
        .clk       (clk),
        .rst_n     (rst_n),
        .func      (fpu64_issue_func),
        .rnd_mode  (2'b00),
        .ftz       (1'b0),
        .operand_a (fpu64_op_a_ext),
        .operand_b (fpu64_op_b_ext),
        .operand_c (fpu64_op_c_ext),
        .valid_in  (fpu64_valid_in),
        .lane_mask (fpu64_issue_mask),
        .result    (fpu64_result),
        .valid_out (fpu64_valid_out),
        .overflow_flags (),
        .invalid_flags  ()
    );
    assign fpu64_ready = 1'b1;  // Always ready (pipelined)

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

    `ifdef SIMULATION
    reg [7:0] fp16_dbg_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) fp16_dbg_cnt <= 0;
        else if (fp16_issue && fp16_dbg_cnt < 10) begin
            `ifdef SIMULATION
            $display("[%0t SM%0d FP16_ISSUE] func=%0d rd=R%0d ra=R%0d rb=R%0d mask=0x%08x",
                     $time, SM_ID, fp16_issue_func, fp16_issue_rd, issue_ra, issue_rb, fp16_issue_mask);
            `endif
            `ifdef SIMULATION
            $display("  op_a[0]=0x%08x op_b[0]=0x%08x", fp16_op_a[31:0], fp16_op_b[31:0]);
            `endif
            fp16_dbg_cnt <= fp16_dbg_cnt + 1;
        end
    end
    `endif

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
            for (fp16_i = 0; fp16_i < 3; fp16_i = fp16_i + 1) begin
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
            fp16_warp_pipe[2] <= fp16_warp_pipe[1];
            fp16_rd_pipe[2] <= fp16_rd_pipe[1];
            fp16_mask_pipe[2] <= fp16_mask_pipe[1];
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
        .result    (sfu_result),
        .invalid_flags ()
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
    `ifdef SIMULATION
    reg [7:0] mem_pend_dbg_cnt;
    `endif
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_pending_valid <= 1'b0;
            `ifdef SIMULATION
            mem_pend_dbg_cnt <= 0;
            `endif
        end else if (gmem_req_valid && gmem_req_ready && issue_mem_read && !mem_pending_valid) begin
            mem_pending_valid <= 1'b1;
            mem_warp_pending <= issue_warp_id;
            mem_rd_pending <= issue_rd;
            mem_mask_pending <= issue_mask;
            `ifdef SIMULATION
            if (mem_pend_dbg_cnt < 10) begin
                `ifdef SIMULATION
                $display("[%0t SM%0d MEM_PEND] SET rd=R%0d warp=%0d mask=0x%08x",
                         $time, SM_ID, issue_rd, issue_warp_id, issue_mask);
                `endif
                mem_pend_dbg_cnt <= mem_pend_dbg_cnt + 1;
            end
            `endif
        end else if (gmem_resp_valid) begin
            mem_pending_valid <= 1'b0;
            `ifdef SIMULATION
            if (mem_pend_dbg_cnt < 10) begin
                `ifdef SIMULATION
                $display("[%0t SM%0d MEM_PEND] CLEAR (resp_valid) pending_was=%b latched=%b",
                         $time, SM_ID, mem_pending_valid, gmem_resp_latched);
                `endif
                mem_pend_dbg_cnt <= mem_pend_dbg_cnt + 1;
            end
            `endif
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

    // DEBUG: track store issuance
    reg [3:0] store_issue_debug_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            store_pending_valid <= 1'b0;
            store_warp_pending <= 0;
            store_mask_pending <= 0;
            store_issue_debug_cnt <= 0;
        end else begin
            if (issue_valid && issue_mem_write && !issue_mem_read &&
                !issue_atomic_op && !store_pending_valid) begin
                store_pending_valid <= 1'b1;
                store_warp_pending <= issue_warp_id;
                store_mask_pending <= issue_mask;
                if (store_issue_debug_cnt < 4) begin
                    `ifdef SIMULATION
                    $display("[SM%0d] Store registered: warp=%0d mask=0x%04x", SM_ID, issue_warp_id, issue_mask[15:0]);
                    `endif
                    store_issue_debug_cnt <= store_issue_debug_cnt + 1;
                end
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
        .TC_FP6_FORMAT   (TC_FP6_FORMAT),
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
    // Video SIMD Unit (VADD4/VSUB4/VABSDIFF4/DP4A/DP2A)
    //------------------------------------------------------------------------
    wire video_use_slot0 = video_issue0;
    wire video_use_slot1 = video_issue1;
    wire [WARP_ID_W-1:0] video_issue_warp = video_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] video_issue_rd = video_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] video_issue_mask = video_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] video_issue_func = video_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] video_op_a = video_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] video_op_b = video_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] video_op_c = video_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;
    // Extract signed flag from func[5] for video operations
    wire video_is_signed = video_issue_func[5];

    assign video_valid_in = video_issue;

    video_simd_unit #(
        .NUM_LANES(NUM_LANES)
    ) u_video_simd (
        .clk       (clk),
        .rst_n     (rst_n),
        .func      (video_issue_func),
        .valid_in  (video_valid_in),
        .is_signed (video_is_signed),
        .operand_a (video_op_a),
        .operand_b (video_op_b),
        .operand_c (video_op_c),
        .lane_mask (video_issue_mask),
        .result    (video_result),
        .valid_out (video_valid_out)
    );

    // Video pipeline tracking (2 stages for video_unit latency)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            video_warp_pipe[0] <= 0;
            video_rd_pipe[0] <= 0;
            video_mask_pipe[0] <= 0;
            video_warp_pipe[1] <= 0;
            video_rd_pipe[1] <= 0;
            video_mask_pipe[1] <= 0;
        end else begin
            // Stage 0: capture issue info
            video_warp_pipe[0] <= video_valid_in ? video_issue_warp : {WARP_ID_W{1'b0}};
            video_rd_pipe[0] <= video_valid_in ? video_issue_rd : 5'b0;
            video_mask_pipe[0] <= video_valid_in ? video_issue_mask : {NUM_LANES{1'b0}};
            // Stage 1: shift pipeline
            video_warp_pipe[1] <= video_warp_pipe[0];
            video_rd_pipe[1] <= video_rd_pipe[0];
            video_mask_pipe[1] <= video_mask_pipe[0];
        end
    end

    //------------------------------------------------------------------------
    // Texture Unit (tex/txq/suld/sust/sured)
    //------------------------------------------------------------------------
    wire tex_use_slot0 = tex_issue0;
    wire tex_use_slot1 = tex_issue1;
    wire [WARP_ID_W-1:0] tex_issue_warp = tex_use_slot0 ? issue_warp_id : issue1_warp_id;
    wire [4:0] tex_issue_rd = tex_use_slot0 ? issue_rd : issue1_rd;
    wire [NUM_LANES-1:0] tex_issue_mask = tex_use_slot0 ? issue_mask : issue1_mask;
    wire [5:0] tex_issue_opcode = tex_use_slot0 ? issue_opcode : issue1_opcode;
    wire [5:0] tex_issue_func = tex_use_slot0 ? issue_func : issue1_func;
    wire [SIMD_WIDTH-1:0] tex_coord_s = tex_use_slot0 ? rf_rd_data_a : rf1_rd_data_a;
    wire [SIMD_WIDTH-1:0] tex_coord_t = tex_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;
    wire [SIMD_WIDTH-1:0] tex_coord_r = tex_use_slot0 ? rf_rd_data_c : rf1_rd_data_c;
    wire [SIMD_WIDTH-1:0] tex_store_data = tex_use_slot0 ? rf_rd_data_b : rf1_rd_data_b;

    assign tex_valid_in = tex_issue && !tex_busy;

    texture_unit #(
        .CACHE_SIZE_KB(16),
        .MAX_ANISO    (16)
    ) u_texture (
        .clk            (clk),
        .rst_n          (rst_n),
        .opcode         (tex_issue_opcode),
        .func           (tex_issue_func),
        .valid_in       (tex_valid_in),

        // Coordinates from registers (using first lane for simplicity)
        .coord_s        (tex_coord_s[31:0]),
        .coord_t        (tex_coord_t[31:0]),
        .coord_r        (tex_coord_r[31:0]),
        .coord_q        (32'b0),

        // LOD - simplified (explicit LOD not used in basic implementation)
        .lod            (32'b0),
        .dsdx           (32'b0),
        .dsdy           (32'b0),
        .dtdx           (32'b0),
        .dtdy           (32'b0),

        // Texture descriptor - placeholder (would come from descriptor table in full impl)
        .tex_base_addr  (32'h0),
        .tex_width      (16'd256),
        .tex_height     (16'd256),
        .tex_depth      (16'd1),
        .tex_format     (4'h0),
        .tex_filter     (4'h0),
        .tex_wrap_s     (4'h0),
        .tex_wrap_t     (4'h0),
        .tex_wrap_r     (4'h0),
        .num_mip_levels (4'd1),

        // Store data for surface writes
        .store_data     ({tex_store_data[127:0]}),

        // Memory interface - connected to global memory arbiter
        .mem_req        (tex_mem_req),
        .mem_write      (tex_mem_write),
        .mem_addr       (tex_mem_addr),
        .mem_wdata      (tex_mem_wdata),
        .mem_ready      (tex_mem_ready),
        .mem_rdata      (tex_mem_rdata),
        .mem_valid      (tex_mem_valid),

        .result         (tex_result),
        .valid_out      (tex_valid_out),
        .busy           (tex_busy)
    );

    // Texture pipeline tracking (variable latency)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tex_pending_valid <= 1'b0;
            tex_warp_pending <= 0;
            tex_rd_pending <= 0;
            tex_mask_pending <= 0;
            tex_result_latched <= 128'b0;
            tex_result_valid_latched <= 1'b0;
        end else begin
            // Track pending texture operation
            if (tex_valid_in) begin
                tex_pending_valid <= 1'b1;
                tex_warp_pending <= tex_issue_warp;
                tex_rd_pending <= tex_issue_rd;
                tex_mask_pending <= tex_issue_mask;
            end

            // Latch result when valid
            if (tex_valid_out && tex_pending_valid && !tex_result_valid_latched) begin
                tex_result_latched <= tex_result;
                tex_result_valid_latched <= 1'b1;
            end

            // Clear pending and latch when writeback consumes it
            if (tex_result_valid_latched && wb_found && wb_sel == 4'd12) begin
                tex_pending_valid <= 1'b0;
                tex_result_valid_latched <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Writeback queues (capture FU outputs for arbitration)
    //------------------------------------------------------------------------
    assign alu_wbq_in = pack_wb(alu_warp_pipe, alu_rd_pipe, alu_mask_pipe, alu_result_pipe);
    assign mul_wbq_in = pack_wb(mul_warp_pipe, mul_rd_pipe, mul_mask_pipe, mul_result);
    assign fpu32_wbq_in = pack_wb(fpu32_warp_pipe[0], fpu32_rd_pipe[0], fpu32_mask_pipe[0], fpu32_result);
    assign fpu64_wbq_in = pack_wb(fpu64_warp_pipe[4], fpu64_rd_pipe[4], fpu64_mask_pipe[4], fpu64_result_trunc);
    assign fp16_wbq_in = pack_wb(fp16_warp_pipe[2], fp16_rd_pipe[2], fp16_mask_pipe[2], fp16_result);
    assign sfu_wbq_in = pack_wb(sfu_warp_pipe[7], sfu_rd_pipe[7], sfu_mask_pipe[7], sfu_result);
    assign shfl_wbq_in = pack_wb(shuffle_warp_pipe, shuffle_rd_pipe, shuffle_mask_pipe, shuffle_result_pipe);
    assign video_wbq_in = pack_wb(video_warp_pipe[1], video_rd_pipe[1], video_mask_pipe[1], video_result);
    wire [WB_PKT_W-1:0] special_wbq_in = pack_wb(special_warp_pipe, special_rd_pipe, special_mask_pipe, special_result_pipe);

    assign alu_wbq_push = alu_valid_out;

    // DEBUG: track ALU WBQ pushes
    always @(posedge clk) begin
        if (alu_wbq_push) begin
            `ifdef SIMULATION
            $display("[SM%0d] ALU_WBQ_PUSH: rd=R%0d mask=0x%08x data[0]=0x%08x empty=%b full=%b",
                     SM_ID, alu_rd_pipe, alu_mask_pipe, alu_result_pipe[31:0], alu_wbq_empty, alu_wbq_full);
            `endif
        end
    end
    assign mul_wbq_push = mul_valid_out;
    assign fpu32_wbq_push = fpu32_valid_out;
    assign fpu64_wbq_push = fpu64_valid_out;
    assign fp16_wbq_push = fp16_valid_out;
    assign sfu_wbq_push = sfu_valid_out;
    assign shfl_wbq_push = shuffle_valid_out;
    assign video_wbq_push = video_valid_out;
    wire special_wbq_push = special_valid_out;

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

    // Video SIMD WBQ (2-cycle latency)
    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(VIDEO_WBQ_DEPTH)
    ) u_video_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (video_wbq_push),
        .push_data(video_wbq_in),
        .pop      (video_wbq_pop),
        .pop_data (video_wbq_out),
        .full     (video_wbq_full),
        .empty    (video_wbq_empty)
    );

    // Special register WBQ (reuse ALU depth since it's also 1-cycle)
    wb_fifo #(
        .WIDTH(WB_PKT_W),
        .DEPTH(ALU_WBQ_DEPTH)
    ) u_special_wbq (
        .clk      (clk),
        .rst_n    (rst_n),
        .push     (special_wbq_push),
        .push_data(special_wbq_in),
        .pop      (special_wbq_pop),
        .pop_data (special_wbq_out),
        .full     (special_wbq_full),
        .empty    (special_wbq_empty)
    );

    assign alu_wbq_warp = alu_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign mul_wbq_warp = mul_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu32_wbq_warp = fpu32_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fpu64_wbq_warp = fpu64_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign fp16_wbq_warp = fp16_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign sfu_wbq_warp = sfu_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign shfl_wbq_warp = shfl_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign video_wbq_warp = video_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign alu_wbq_rd = alu_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign mul_wbq_rd = mul_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu32_wbq_rd = fpu32_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fpu64_wbq_rd = fpu64_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign fp16_wbq_rd = fp16_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign sfu_wbq_rd = sfu_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign shfl_wbq_rd = shfl_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign video_wbq_rd = video_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign alu_wbq_mask = alu_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign mul_wbq_mask = mul_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu32_wbq_mask = fpu32_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fpu64_wbq_mask = fpu64_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign fp16_wbq_mask = fp16_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign sfu_wbq_mask = sfu_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign shfl_wbq_mask = shfl_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign video_wbq_mask = video_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign alu_wbq_data = alu_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign mul_wbq_data = mul_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fpu32_wbq_data = fpu32_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fpu64_wbq_data = fpu64_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign fp16_wbq_data = fp16_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign sfu_wbq_data = sfu_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign shfl_wbq_data = shfl_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign video_wbq_data = video_wbq_out[WB_DATA_MSB:WB_DATA_LSB];
    assign special_wbq_warp = special_wbq_out[WB_WARP_MSB:WB_WARP_LSB];
    assign special_wbq_rd = special_wbq_out[WB_RD_MSB:WB_RD_LSB];
    assign special_wbq_mask = special_wbq_out[WB_MASK_MSB:WB_MASK_LSB];
    assign special_wbq_data = special_wbq_out[WB_DATA_MSB:WB_DATA_LSB];

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
    // Async Copy Engine (cp.async support)
    // Handles asynchronous global->shared memory copies
    //------------------------------------------------------------------------
    // Input signals for async_copy_engine
    // For cp.async: src_addr (global) from ra, dst_addr (shared) from imm16
    // Size encoded in func field
    // Gate with ace_ready to implement backpressure - prevents dropped requests

    // TMA instruction detection (cp.async.bulk.tensor)
    wire issue_cpasync_tma = issue_cpasync && (issue_func == `CPASYNC_BULK_TENSOR);
    wire issue1_cpasync_tma = issue1_cpasync && (issue1_func == `CPASYNC_BULK_TENSOR);

    wire        ace_valid_in = ace_ready && (issue_cpasync_copy || issue1_cpasync_copy ||
                               issue_cpasync_tma || issue1_cpasync_tma ||
                               (issue_cpasync && (issue_func == `CPASYNC_COMMIT ||
                                                  issue_func == `CPASYNC_WAIT ||
                                                  issue_func == `CPASYNC_WAIT_ALL)) ||
                               (issue1_cpasync && (issue1_func == `CPASYNC_COMMIT ||
                                                   issue1_func == `CPASYNC_WAIT ||
                                                   issue1_func == `CPASYNC_WAIT_ALL)) ||
                               // st.async operations
                               issue_st_async || issue1_st_async);
    wire [5:0]  ace_func = issue_cpasync ? issue_func :
                           issue1_cpasync ? issue1_func :
                           issue_st_async ? st_async_func :
                           issue1_st_async ? issue1_func : 6'b0;
    // Source address from register (lane 0 for simplicity - real impl would be per-lane)
    wire [31:0] ace_src_addr = issue_cpasync ? rf_rd_data_a[31:0] :
                               issue1_cpasync ? rf1_rd_data_a[31:0] : 32'b0;
    // Destination address in shared memory from imm16
    wire [13:0] ace_dst_addr = issue_cpasync ? issue_imm16[13:0] :
                               issue1_cpasync ? issue1_imm16[13:0] : 14'b0;
    // Size from rb register (bits [3:0] encode 4/8/16 bytes)
    wire [3:0]  ace_size = issue_cpasync ? issue_rb[3:0] :
                           issue1_cpasync ? issue1_rb[3:0] : 4'd4;
    // Wait count for wait_group
    wire [3:0]  ace_wait_count = issue_cpasync ? issue_imm16[3:0] :
                                 issue1_cpasync ? issue1_imm16[3:0] : 4'b0;

    // TMA Interface signals
    // For cp.async.bulk.tensor: tensor_desc from {rb, ra} (64-bit), coords from rc and imm16
    // Tensor descriptor: [31:0] base addr in ra, [63:32] metadata in rb
    wire [63:0] ace_tensor_desc = issue_cpasync_tma ? {rf_rd_data_b[31:0], rf_rd_data_a[31:0]} :
                                  issue1_cpasync_tma ? {rf1_rd_data_b[31:0], rf1_rd_data_a[31:0]} : 64'b0;
    // X coordinate (byte offset) from rc register
    wire [31:0] ace_tensor_coord_x = issue_cpasync_tma ? rf_rd_data_c[31:0] :
                                     issue1_cpasync_tma ? rf1_rd_data_c[31:0] : 32'b0;
    // Y coordinate (row offset) from imm16 field (16-bit is sufficient for most tile sizes)
    wire [31:0] ace_tensor_coord_y = issue_cpasync_tma ? {16'b0, issue_imm16} :
                                     issue1_cpasync_tma ? {16'b0, issue1_imm16} : 32'b0;

    // TMA busy status
    wire ace_tma_busy;

    // st.async Interface signals
    // For st.async: global address from ra, shared memory address from imm16, data from rb
    wire ace_is_store = (issue_st_async && (st_async_func == `ST_ASYNC_GLOBAL)) ||
                        (issue1_st_async && (issue1_func == `ST_ASYNC_GLOBAL));
    wire [31:0] ace_store_gmem_addr = issue_st_async ? st_async_addr :
                                      issue1_st_async ? rf1_rd_data_a[0] : 32'b0;
    wire [127:0] ace_store_data = issue_st_async ? {96'b0, st_async_data} :
                                  issue1_st_async ? {96'b0, rf1_rd_data_b[0]} : 128'b0;

    // Global memory response for async_copy_engine
    // We create a simple arbiter: ACE gets priority when it has pending requests
    wire        ace_gmem_resp_valid;
    wire [127:0] ace_gmem_resp_data;

    // Global memory write interface for st.async
    wire        ace_gmem_wr_valid;
    wire [31:0] ace_gmem_wr_addr;
    wire [127:0] ace_gmem_wr_data;
    wire [4:0]  ace_gmem_wr_size;
    wire        ace_gmem_wr_done;  // Tie high for now (instant completion)

    // Shared memory read interface for st.async.global (if reading from SMEM)
    wire        ace_smem_rd_en;
    wire [13:0] ace_smem_rd_addr;
    wire [127:0] ace_smem_rd_data = 128'b0;  // TODO: Connect to shared memory read
    wire        ace_smem_rd_valid = ace_smem_rd_en;  // Instant read for now

    // Simple global memory write done signal (instant completion for simulation)
    assign ace_gmem_wr_done = ace_gmem_wr_valid;

    // Determine the opcode for ACE (cp.async or st.async)
    wire [5:0] ace_opcode = (issue_cpasync || issue1_cpasync) ? `OP_CPASYNC :
                            (issue_st_async || issue1_st_async) ? `OP_ST_ASYNC : 6'b0;

    async_copy_engine #(
        .MAX_GROUPS(8),
        .MAX_PENDING(16),
        .SHARED_MEM_ADDR_W(14),
        .GLOBAL_ADDR_W(32)
    ) u_async_copy_engine (
        .clk            (clk),
        .rst_n          (rst_n),
        // Control interface
        .opcode         (ace_opcode),
        .func           (ace_func),
        .valid_in       (ace_valid_in),
        .src_addr       (ace_src_addr),
        .dst_addr       (ace_dst_addr),
        .size           (ace_size),
        .cache_hint     (3'b0),  // Default cache hint
        .wait_count     (ace_wait_count),
        // TMA Interface (for cp.async.bulk.tensor)
        .tensor_desc    (ace_tensor_desc),
        .tensor_coord_x (ace_tensor_coord_x),
        .tensor_coord_y (ace_tensor_coord_y),
        // st.async Interface
        .is_store       (ace_is_store),
        .store_gmem_addr(ace_store_gmem_addr),
        .store_data     (ace_store_data),
        // Status outputs
        .ready          (ace_ready),
        .done           (ace_done),
        .pending_count  (ace_pending_count),
        .tma_busy       (ace_tma_busy),
        // Global memory read interface
        .gmem_req_valid (ace_gmem_req_valid),
        .gmem_req_addr  (ace_gmem_req_addr),
        .gmem_req_size  (ace_gmem_req_size),
        .gmem_req_cache (ace_gmem_req_cache),
        .gmem_resp_valid(ace_gmem_resp_valid),
        .gmem_resp_data (ace_gmem_resp_data),
        // Global memory write interface (st.async)
        .gmem_wr_valid  (ace_gmem_wr_valid),
        .gmem_wr_addr   (ace_gmem_wr_addr),
        .gmem_wr_data   (ace_gmem_wr_data),
        .gmem_wr_size   (ace_gmem_wr_size),
        .gmem_wr_done   (ace_gmem_wr_done),
        // Shared memory write interface (cp.async)
        .smem_wr_en     (ace_smem_wr_en),
        .smem_wr_addr   (ace_smem_wr_addr),
        .smem_wr_data   (ace_smem_wr_data),
        .smem_wr_size   (ace_smem_wr_size),
        // Shared memory read interface (st.async.global)
        .smem_rd_en     (ace_smem_rd_en),
        .smem_rd_addr   (ace_smem_rd_addr),
        .smem_rd_data   (ace_smem_rd_data),
        .smem_rd_valid  (ace_smem_rd_valid)
    );

    //------------------------------------------------------------------------
    // mbarrier Unit (Hopper+ Memory Barriers)
    //------------------------------------------------------------------------
    // Determine which lane is issuing an mbarrier instruction
    wire        mbarrier_valid_in = mbarrier_ready && (issue_mbarrier || issue1_mbarrier);
    wire [5:0]  mbarrier_func = issue_mbarrier ? issue_func : issue1_func;
    wire [13:0] mbarrier_addr = issue_mbarrier ? rf_rd_data_a[13:0] : rf1_rd_data_a[13:0];
    wire [31:0] mbarrier_count = issue_mbarrier ? rf_rd_data_b[31:0] : rf1_rd_data_b[31:0];
    wire [WARP_ID_W-1:0] mbarrier_warp = issue_mbarrier ? issue_warp_id : issue1_warp_id;
    wire [31:0] mbarrier_mask = issue_mbarrier ? issue_mask : issue1_mask;

    `ifdef SIMULATION
    always @(posedge clk) begin
        if (mbarrier_valid_in) begin
            $display("[SM%0d] MBARRIER: func=%0d addr=0x%h count=%0d warp=%0d mask=0x%08x",
                     SM_ID, mbarrier_func, mbarrier_addr, mbarrier_count, mbarrier_warp, mbarrier_mask);
        end
        if (issue_mbarrier || issue1_mbarrier) begin
            $display("[SM%0d] mbarrier detected: issue_mbarrier=%b issue1_mbarrier=%b ready=%b",
                     SM_ID, issue_mbarrier, issue1_mbarrier, mbarrier_ready);
        end
    end
    `endif

    // Async arrival from cp.async: signal when transaction completes
    wire        mbarrier_async_arrive = ace_smem_wr_en;  // cp.async completion
    wire [13:0] mbarrier_async_addr = ace_smem_wr_addr;
    wire [31:0] mbarrier_async_bytes = {27'b0, ace_smem_wr_size};

    mbarrier_unit #(
        .NUM_BARRIERS(8),
        .NUM_WARPS(NUM_WARPS),
        .SMEM_ADDR_W(14),
        .WARP_ID_W(WARP_ID_W)
    ) u_mbarrier (
        .clk                (clk),
        .rst_n              (rst_n),
        // Control interface
        .valid_in           (mbarrier_valid_in),
        .func               (mbarrier_func),
        .barrier_addr       (mbarrier_addr),
        .count              (mbarrier_count),
        .warp_id            (mbarrier_warp),
        .thread_mask        (mbarrier_mask),
        // Status outputs
        .ready              (mbarrier_ready),
        .done               (mbarrier_done),
        .result             (mbarrier_result),
        .result_valid       (mbarrier_result_valid),
        // Async arrival (from cp.async)
        .async_arrive_valid (mbarrier_async_arrive),
        .async_barrier_addr (mbarrier_async_addr),
        .async_tx_bytes     (mbarrier_async_bytes),
        // Warp stall interface
        .warp_blocked       (mbarrier_warp_blocked),
        // Shared memory interface (unused for now - barriers cached internally)
        .smem_rd_en         (),
        .smem_rd_addr       (),
        .smem_rd_data       (128'b0),
        .smem_rd_valid      (1'b0),
        .smem_wr_en         (),
        .smem_wr_addr       (),
        .smem_wr_data       (),
        .smem_wr_mask       ()
    );

    //------------------------------------------------------------------------
    // Cache Policy Control (Hopper+)
    // Handles createpolicy, applypriority, and discard instructions
    //------------------------------------------------------------------------
    // Cache policy instruction detection
    wire issue_cache_policy = issue_valid && issue_cache_policy_op;
    wire issue1_cache_policy = issue1_valid && issue1_cache_policy_op;

    // Determine which lane is issuing a cache policy instruction
    wire cache_policy_active = issue_cache_policy || issue1_cache_policy;
    wire [5:0] cache_policy_func = issue_cache_policy ? issue_func : issue1_func;

    // createpolicy: policy_id from imm16[2:0], priority from rb register
    wire cache_policy_create = cache_policy_active && (cache_policy_func == `CACHE_CREATEPOLICY);
    wire [2:0] cache_policy_create_id = issue_cache_policy ? issue_imm16[2:0] : issue1_imm16[2:0];
    wire [7:0] cache_policy_create_priority = issue_cache_policy ? rf_rd_data_b[7:0] : rf1_rd_data_b[7:0];

    // applypriority: address from ra, policy_id from rb[2:0]
    wire cache_policy_apply = cache_policy_active && (cache_policy_func == `CACHE_APPLYPRIORITY);
    wire [31:0] cache_policy_apply_addr = issue_cache_policy ? rf_rd_data_a[31:0] : rf1_rd_data_a[31:0];
    wire [2:0] cache_policy_apply_id = issue_cache_policy ? rf_rd_data_b[2:0] : rf1_rd_data_b[2:0];

    // discard: address from ra register
    wire cache_policy_discard = cache_policy_active && (cache_policy_func == `CACHE_DISCARD);
    wire [31:0] cache_policy_discard_addr = issue_cache_policy ? rf_rd_data_a[31:0] : rf1_rd_data_a[31:0];

    // Cache policy result (from createpolicy) - writeback handling
    reg cache_policy_token_valid_r;
    reg [31:0] cache_policy_token_r;
    reg [WARP_ID_W-1:0] cache_policy_wb_warp;
    reg [4:0] cache_policy_wb_rd;
    reg [NUM_LANES-1:0] cache_policy_wb_mask;

    //------------------------------------------------------------------------
    // Stack Operations (Phase 6.2)
    // Handles alloca, stacksave, stackrestore
    //------------------------------------------------------------------------
    // Per-warp stack pointers (local memory stack)
    reg [31:0] warp_stack_ptr [0:NUM_WARPS-1];

    // Stack operation detection
    wire issue_stack = issue_valid && issue_stack_op;
    wire issue1_stack = issue1_valid && issue1_stack_op;
    wire stack_active = issue_stack || issue1_stack;
    wire [5:0] stack_func = issue_stack ? issue_func : issue1_func;
    wire [WARP_ID_W-1:0] stack_warp_id = issue_stack ? issue_warp_id : issue1_warp_id;

    // alloca: size from ra register
    wire stack_alloca = stack_active && (stack_func == `STACK_ALLOCA);
    wire [31:0] stack_alloca_size = issue_stack ? rf_rd_data_a[0] : rf1_rd_data_a[0];

    // stacksave: no operand (just read stack pointer)
    wire stack_save = stack_active && (stack_func == `STACK_SAVE);

    // stackrestore: new stack pointer from ra register
    wire stack_restore = stack_active && (stack_func == `STACK_RESTORE);
    wire [31:0] stack_restore_ptr = issue_stack ? rf_rd_data_a[0] : rf1_rd_data_a[0];

    // Stack result for writeback
    reg stack_result_valid_r;
    reg [31:0] stack_result_r;
    reg [WARP_ID_W-1:0] stack_wb_warp;
    reg [4:0] stack_wb_rd;
    reg [NUM_LANES-1:0] stack_wb_mask;

    //------------------------------------------------------------------------
    // Debug Operations (Phase 6.2)
    // Handles brkpt, trap, pmevent
    //------------------------------------------------------------------------
    wire issue_debug = issue_valid && issue_debug_op;
    wire issue1_debug = issue1_valid && issue1_debug_op;
    wire debug_active = issue_debug || issue1_debug;
    wire [5:0] debug_func = issue_debug ? issue_func : issue1_func;

    // Debug event outputs (active-high pulses for external monitoring)
    reg debug_brkpt_event;    // Breakpoint hit
    reg debug_trap_event;     // Trap triggered
    reg [31:0] debug_pmevent_id; // Performance event ID
    reg debug_pmevent_valid;

    //------------------------------------------------------------------------
    // Misc Operations (Phase 6.2)
    // Handles nanosleep, setmaxnreg
    //------------------------------------------------------------------------
    wire issue_misc = issue_valid && issue_misc_op;
    wire issue1_misc = issue1_valid && issue1_misc_op;
    wire misc_active = issue_misc || issue1_misc;
    wire [5:0] misc_func = issue_misc ? issue_func : issue1_func;
    wire [WARP_ID_W-1:0] misc_warp_id = issue_misc ? issue_warp_id : issue1_warp_id;

    // nanosleep: delay value from ra register (in cycles for simulation)
    wire misc_nanosleep = misc_active && (misc_func == `MISC_NANOSLEEP);
    wire [31:0] nanosleep_cycles = issue_misc ? rf_rd_data_a[0] : rf1_rd_data_a[0];

    // Per-warp nanosleep counter
    reg [31:0] warp_nanosleep_counter [0:NUM_WARPS-1];
    wire [NUM_WARPS-1:0] warp_sleeping;
    genvar ns_i;
    generate
        for (ns_i = 0; ns_i < NUM_WARPS; ns_i = ns_i + 1) begin : gen_sleep_check
            assign warp_sleeping[ns_i] = (warp_nanosleep_counter[ns_i] > 0);
        end
    endgenerate

    // setmaxnreg: max register count from imm16
    wire misc_setmaxnreg = misc_active && (misc_func == `MISC_SETMAXNREG);
    reg [15:0] warp_maxnreg [0:NUM_WARPS-1];

    //------------------------------------------------------------------------
    // Async Store Operations (Phase 1.2)
    // Handles st.async.global, st.async.shared, and async commit/wait
    //------------------------------------------------------------------------
    wire issue_st_async = issue_valid && issue_st_async_op;
    wire issue1_st_async = issue1_valid && issue1_st_async_op;
    wire st_async_active = issue_st_async || issue1_st_async;
    wire [5:0] st_async_func = issue_st_async ? issue_func : issue1_func;
    wire [WARP_ID_W-1:0] st_async_warp_id = issue_st_async ? issue_warp_id : issue1_warp_id;

    // st.async store data and address
    wire [31:0] st_async_addr = issue_st_async ? rf_rd_data_a[0] : rf1_rd_data_a[0];
    wire [31:0] st_async_data = issue_st_async ? rf_rd_data_b[0] : rf1_rd_data_b[0];
    wire st_async_to_shared = (st_async_func == `ST_ASYNC_SHARED);
    wire st_async_commit = (st_async_func == `ST_ASYNC_COMMIT);
    wire st_async_wait = (st_async_func == `ST_ASYNC_WAIT);
    wire [3:0] st_async_wait_count = issue_st_async ? issue_imm16[3:0] : issue1_imm16[3:0];

    // Per-warp async store group tracking (similar to cp.async groups)
    reg [7:0] warp_st_async_pending [0:NUM_WARPS-1];  // Pending stores per group
    reg [3:0] warp_st_async_groups [0:NUM_WARPS-1];   // Active group count
    wire [NUM_WARPS-1:0] warp_st_async_stalled;       // Warps waiting on st.async completion

    genvar sta_i;
    generate
        for (sta_i = 0; sta_i < NUM_WARPS; sta_i = sta_i + 1) begin : gen_st_async_stall
            assign warp_st_async_stalled[sta_i] = (warp_st_async_groups[sta_i] > 0);
        end
    endgenerate

    //------------------------------------------------------------------------
    // Multimem Operations (Phase 1.2 - Distributed Shared Memory)
    // Enables cross-SM shared memory access for Thread Block Clusters
    //------------------------------------------------------------------------
    wire issue_multimem = issue_valid && issue_multimem_op;
    wire issue1_multimem = issue1_valid && issue1_multimem_op;
    wire multimem_active = issue_multimem || issue1_multimem;
    wire [5:0] multimem_func = issue_multimem ? issue_func : issue1_func;
    wire [WARP_ID_W-1:0] multimem_warp_id = issue_multimem ? issue_warp_id : issue1_warp_id;

    // Multimem address includes target SM ID in upper bits: [31:24] = target_sm_mask, [23:0] = smem_addr
    wire [31:0] multimem_addr = issue_multimem ? rf_rd_data_a[0] : rf1_rd_data_a[0];
    wire [31:0] multimem_data = issue_multimem ? rf_rd_data_b[0] : rf1_rd_data_b[0];
    wire [7:0] multimem_target_mask = multimem_addr[31:24];  // Which SMs to target
    wire [23:0] multimem_smem_addr = multimem_addr[23:0];    // Shared memory address

    // Multimem state for result tracking
    reg multimem_result_valid_r;
    reg [31:0] multimem_result_r;
    reg [WARP_ID_W-1:0] multimem_wb_warp;
    reg [4:0] multimem_wb_rd;
    reg [NUM_LANES-1:0] multimem_wb_mask;

    // Multicast interface outputs (for external routing to other SMs)
    // Note: Actual cross-SM routing requires cluster-level interconnect
    wire multimem_req_valid = multimem_active && (multimem_func == `MULTIMEM_ST || multimem_func == `MULTIMEM_RED);
    wire multimem_req_write = (multimem_func == `MULTIMEM_ST);
    wire multimem_req_reduce = (multimem_func == `MULTIMEM_RED);

    //------------------------------------------------------------------------
    // WGMMA Unit (Hopper+ Warpgroup MMA)
    // Handles wgmma.mma_async, wgmma.fence, wgmma.commit_group, wgmma.wait_group
    //------------------------------------------------------------------------
    // Input signals for WGMMA
    wire        wgmma_valid_in = wgmma_ready && (issue_wgmma || issue1_wgmma);
    wire [5:0]  wgmma_func = issue_wgmma ? issue_func : issue1_func;
    wire [WARP_ID_W-1:0] wgmma_issue_warp = issue_wgmma ? issue_warp_id : issue1_warp_id;
    // Warpgroup = warp_id / 4 (right shift by 2)
    // Use explicit division to avoid bit-select issues with small WARP_ID_W
    wire [2:0]  wgmma_warpgroup = wgmma_issue_warp >> 2;

    // Matrix descriptors from registers (64-bit descriptors from 2 consecutive 32-bit regs)
    wire [63:0] wgmma_desc_a = issue_wgmma ? {rf_rd_data_a[63:0]} : {rf1_rd_data_a[63:0]};
    wire [63:0] wgmma_desc_b = issue_wgmma ? {rf_rd_data_b[63:0]} : {rf1_rd_data_b[63:0]};
    wire [31:0] wgmma_scale_d = issue_wgmma ? rf_rd_data_c[31:0] : rf1_rd_data_c[31:0];
    wire [3:0]  wgmma_wait_count = issue_wgmma ? issue_imm16[3:0] : issue1_imm16[3:0];

    // Extract shared memory base addresses from descriptors
    // Descriptor format: [31:0] = base_addr (word address in shared memory)
    // The descriptor's base_addr field contains the SMEM address for the matrix tile
    assign wgmma_smem_addr_a = wgmma_desc_a[13:0];  // 14-bit SMEM address for matrix A
    assign wgmma_smem_addr_b = wgmma_desc_b[13:0];  // 14-bit SMEM address for matrix B

    // WGMMA reads from SMEM when a MMA operation is starting
    // Only enable SMEM read for actual MMA operations (not fence/commit/wait)
    wire wgmma_is_mma_op = (wgmma_func == `WGMMA_M64N8K16)  || (wgmma_func == `WGMMA_M64N16K16) ||
                           (wgmma_func == `WGMMA_M64N32K16) || (wgmma_func == `WGMMA_M64N64K16) ||
                           (wgmma_func == `WGMMA_M64N128K16) || (wgmma_func == `WGMMA_M64N256K16);
    assign wgmma_smem_rd_en = wgmma_valid_in && wgmma_is_mma_op;

    // Accumulator index from descriptor (use bits from scale_d or imm16)
    // Typically the accumulator index is encoded in the instruction immediate
    assign wgmma_accum_idx = issue_wgmma ? issue_imm16[6:4] : issue1_imm16[6:4];

    // WGMMA Accumulator register file management
    integer wgmma_acc_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (wgmma_acc_i = 0; wgmma_acc_i < 8; wgmma_acc_i = wgmma_acc_i + 1) begin
                wgmma_accum_reg[wgmma_acc_i] <= 1024'b0;
            end
        end else if (wgmma_done) begin
            // Write back WGMMA result to accumulator register
            wgmma_accum_reg[wgmma_accum_idx] <= wgmma_accum_out;
        end
    end

    // Select current accumulator value for WGMMA input
    wire [1023:0] wgmma_accum_in = wgmma_accum_reg[wgmma_accum_idx];

    wgmma #(
        .WARPGROUP_SIZE(4),
        .THREADS_PER_WARP(NUM_LANES),
        .MAX_PENDING_OPS(8)
    ) u_wgmma (
        .clk            (clk),
        .rst_n          (rst_n),
        // Control interface
        .func           (wgmma_func),
        .valid_in       (wgmma_valid_in),
        .warpgroup_id   (wgmma_warpgroup),
        .wait_count     (wgmma_wait_count),
        // Matrix descriptors
        .desc_a         (wgmma_desc_a),
        .desc_b         (wgmma_desc_b),
        .scale_d        (wgmma_scale_d),
        // Data interface - now wired to shared memory!
        .data_a         (wgmma_smem_data_a),        // From shared memory read port
        .data_b         (wgmma_smem_data_b),        // From shared memory read port
        .accum_in       (wgmma_accum_in),           // From accumulator register file
        .accum_out      (wgmma_accum_out),
        // Status outputs
        .ready          (wgmma_ready),
        .done           (wgmma_done),
        .pending_ops    (wgmma_pending_ops)
    );

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
        .bank_conflict(smem_bank_conflict),
        // Async copy write port (from async_copy_engine)
        .async_wr_en  (ace_smem_wr_en),
        .async_wr_addr(ace_smem_wr_addr),
        .async_wr_data(ace_smem_wr_data),
        .async_wr_size(ace_smem_wr_size),
        // WGMMA wide read ports (512-bit for tensor operations)
        .wgmma_rd_en    (wgmma_smem_rd_en),
        .wgmma_rd_addr_a(wgmma_smem_addr_a),
        .wgmma_rd_addr_b(wgmma_smem_addr_b),
        .wgmma_rd_data_a(wgmma_smem_data_a),
        .wgmma_rd_data_b(wgmma_smem_data_b),
        .wgmma_rd_valid (wgmma_smem_rd_valid)
    );

    //------------------------------------------------------------------------
    // Global Memory Interface with Async Copy Arbitration
    //------------------------------------------------------------------------
    // Memory arbiter: normal instructions have priority, ACE uses idle cycles
    // ACE state machine for tracking requests
    reg         ace_mem_pending;
    reg [31:0]  ace_mem_addr_saved;
    reg [4:0]   ace_mem_size_saved;

    // Normal memory request signals
    wire gmem_normal_req_valid = issue_valid && !issue_mem_shared && !issue_atomic_op &&
                                 (issue_mem_read || issue_mem_write);
    wire gmem_normal_req_write = issue_mem_write;

    // Priority: Normal > ACE > Texture
    // ACE requests only when no normal request and ACE has pending work
    wire gmem_use_ace = ace_gmem_req_valid && !gmem_normal_req_valid && !ace_mem_pending;
    // Texture requests only when no normal or ACE request
    wire gmem_use_tex = tex_mem_req && !gmem_normal_req_valid && !gmem_use_ace && !tex_mem_pending;

    assign gmem_req_valid = gmem_normal_req_valid || gmem_use_ace || gmem_use_tex;
    assign gmem_req_write = gmem_use_ace ? 1'b0 :
                            gmem_use_tex ? tex_mem_write :
                            gmem_normal_req_write;

    // For ACE: replicate single address across all lanes (only lane 0 matters)
    wire [NUM_LANES*32-1:0] ace_replicated_addr = {NUM_LANES{ace_gmem_req_addr}};
    wire [NUM_LANES*32-1:0] tex_replicated_addr = {NUM_LANES{tex_mem_addr}};
    assign gmem_req_addr = gmem_use_ace ? ace_replicated_addr :
                           gmem_use_tex ? tex_replicated_addr :
                           rf_rd_data_a;
    // Write data: texture uses 128-bit width replicated to lanes 0-3
    wire [SIMD_WIDTH-1:0] tex_wdata_extended = {{(SIMD_WIDTH-128){1'b0}}, tex_mem_wdata};
    assign gmem_req_wdata = gmem_use_tex ? tex_wdata_extended : rf_rd_data_b;

    // Track ACE memory request state
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ace_mem_pending <= 1'b0;
            ace_mem_addr_saved <= 32'b0;
            ace_mem_size_saved <= 5'b0;
        end else begin
            if (gmem_use_ace && gmem_req_ready) begin
                // ACE request accepted
                ace_mem_pending <= 1'b1;
                ace_mem_addr_saved <= ace_gmem_req_addr;
                ace_mem_size_saved <= ace_gmem_req_size;
            end else if (ace_mem_pending && gmem_resp_valid && !tex_mem_pending) begin
                // ACE response received (only when not waiting for tex)
                ace_mem_pending <= 1'b0;
            end
        end
    end

    // Track Texture memory request state
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tex_mem_pending <= 1'b0;
        end else begin
            if (gmem_use_tex && gmem_req_ready) begin
                // Texture request accepted
                tex_mem_pending <= 1'b1;
            end else if (tex_mem_pending && gmem_resp_valid && !ace_mem_pending) begin
                // Texture response received (only when not waiting for ACE)
                tex_mem_pending <= 1'b0;
            end
        end
    end

    // Route response to ACE when ACE request is pending
    // Extract 128-bit data from lane 0-3 of the response
    assign ace_gmem_resp_valid = ace_mem_pending && gmem_resp_valid && !tex_mem_pending;
    assign ace_gmem_resp_data = {gmem_resp_rdata[127:0]};  // First 128 bits (4 lanes)

    // Route response to Texture unit
    assign tex_mem_ready = !tex_mem_pending && gmem_req_ready;  // Ready when not pending
    assign tex_mem_valid = tex_mem_pending && gmem_resp_valid && !ace_mem_pending;
    assign tex_mem_rdata = gmem_resp_rdata[127:0];  // First 128 bits

    // DEBUG: trace first few load/store operations
    reg [3:0] gmem_debug_cnt;
    reg [3:0] gmem_store_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gmem_debug_cnt <= 0;
            gmem_store_cnt <= 0;
        end else if (gmem_req_valid && issue_mem_read && gmem_debug_cnt < 4) begin
            `ifdef SIMULATION
            $display("[SM%0d] LOAD issued: ra=R%0d mask=0x%04x", SM_ID, issue_ra, issue_mask[15:0]);
            `endif
            `ifdef SIMULATION
            $display("        lane0=0x%08x lane1=0x%08x lane15=0x%08x",
                rf_rd_data_a[31:0], rf_rd_data_a[63:32], rf_rd_data_a[511:480]);
            `endif
            gmem_debug_cnt <= gmem_debug_cnt + 1;
        end else if (gmem_req_valid && issue_mem_write && gmem_store_cnt < 4) begin
            `ifdef SIMULATION
            $display("[SM%0d] STORE issued: ra=R%0d rb=R%0d mask=0x%04x", SM_ID, issue_ra, issue_rb, issue_mask[15:0]);
            `endif
            `ifdef SIMULATION
            $display("        addr lane0=0x%08x data lane0=0x%08x",
                rf_rd_data_a[31:0], rf_rd_data_b[31:0]);
            `endif
            gmem_store_cnt <= gmem_store_cnt + 1;
        end
    end

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
    // Combinational Branch Logic with Divergence Detection
    //------------------------------------------------------------------------
    wire branch_is_unconditional = (issue_rd[4:3] == 2'b00) || (issue_rd[4:3] == 2'b11);
    wire branch_is_if_zero       = (issue_rd[4:3] == 2'b01);  // BR_IF_TRUE
    wire branch_is_if_not_zero   = (issue_rd[4:3] == 2'b10);  // BR_IF_FALSE

    // Compute per-lane branch condition (which lanes satisfy the condition)
    // For BR_IF_ZERO: lanes with alu_zero=1 (register==0) should take branch
    // For BR_IF_NOT_ZERO: lanes with alu_zero=0 (register!=0) should take branch
    wire [NUM_LANES-1:0] branch_cond_lanes = branch_is_if_zero ? alu_zero : ~alu_zero;

    // Compute which active lanes want to take vs not take the branch
    wire [NUM_LANES-1:0] taken_lanes = branch_cond_lanes & issue_mask;
    wire [NUM_LANES-1:0] not_taken_lanes = ~branch_cond_lanes & issue_mask;

    // Divergence occurs when some active lanes want to branch and some don't
    wire threads_diverge = (taken_lanes != 0) && (not_taken_lanes != 0) &&
                          !branch_is_unconditional && issue_valid && issue_branch_op;

    // For non-divergent case, determine if all active threads take the branch
    wire all_lanes_take_branch = (taken_lanes != 0) && (not_taken_lanes == 0);
    wire no_lanes_take_branch = (taken_lanes == 0);

    // Simple (non-divergent) branch taken: unconditional OR all active lanes agree
    wire simple_branch_taken = issue_valid && issue_branch_op && (
        branch_is_unconditional ||
        all_lanes_take_branch
    );
    wire [31:0] simple_branch_target = issue_pc + {{16{issue_imm16[15]}}, issue_imm16};

    // For divergent branches: use "not-taken-first" execution strategy
    // Strategy for forward branches (like if-then):
    // 1. Execute not-taken threads at fall-through (PC+4)
    // 2. Push {branch_target, taken_threads} to SM's divergence stack
    // 3. When not-taken threads reach branch_target, pop stack and add taken threads
    // 4. Reconverge with full mask
    wire divergent_branch = issue_valid && issue_branch_op && threads_diverge;

    // Combined branch taken signal:
    // - Non-divergent: taken if all lanes agree to branch
    // - Divergent: NOT taken (not-taken threads execute fall-through first)
    wire branch_taken_combined = simple_branch_taken;  // Don't take for divergent
    wire [31:0] branch_target_combined = simple_branch_target;

    // For divergent branches, mask changes to not-taken threads (they execute first)
    wire [NUM_LANES-1:0] divergent_new_mask = not_taken_lanes;

    //------------------------------------------------------------------------
    // Lane 1 Branch Logic (for dual-issue support)
    // When branch is issued on lane 1, we need separate calculations
    //------------------------------------------------------------------------
    wire branch1_is_unconditional = (issue1_rd[4:3] == 2'b00) || (issue1_rd[4:3] == 2'b11);
    wire branch1_is_if_zero       = (issue1_rd[4:3] == 2'b01);
    wire branch1_is_if_not_zero   = (issue1_rd[4:3] == 2'b10);

    // When lane 1 has a branch and ALU is processing it, alu_zero reflects lane 1's condition
    // Note: ALU multiplexes - when alu_use_slot0=0 and alu_use_slot1=1, alu_zero is for lane 1
    wire [NUM_LANES-1:0] branch1_cond_lanes = branch1_is_if_zero ? alu_zero : ~alu_zero;
    wire [NUM_LANES-1:0] taken1_lanes = branch1_cond_lanes & issue1_mask;
    wire [NUM_LANES-1:0] not_taken1_lanes = ~branch1_cond_lanes & issue1_mask;

    wire threads1_diverge = (taken1_lanes != 0) && (not_taken1_lanes != 0) &&
                           !branch1_is_unconditional && issue1_valid && issue1_branch_op;

    wire all1_lanes_take_branch = (taken1_lanes != 0) && (not_taken1_lanes == 0);

    wire simple_branch1_taken = issue1_valid && issue1_branch_op && (
        branch1_is_unconditional ||
        all1_lanes_take_branch
    );
    wire [31:0] simple_branch1_target = issue1_pc + {{16{issue1_imm16[15]}}, issue1_imm16};

    wire divergent_branch1 = issue1_valid && issue1_branch_op && threads1_diverge;
    wire branch1_taken_combined = simple_branch1_taken;
    wire [31:0] branch1_target_combined = simple_branch1_target;
    wire [NUM_LANES-1:0] divergent1_new_mask = not_taken1_lanes;

    // Divergence stack per warp for SM-level reconvergence tracking
    reg [31:0] sm_div_stack_pc   [0:NUM_WARPS-1][0:3];  // PC to reconverge at
    reg [NUM_LANES-1:0] sm_div_stack_mask [0:NUM_WARPS-1][0:3];  // Threads waiting
    reg [1:0] sm_div_stack_ptr [0:NUM_WARPS-1];  // Stack pointer per warp

    // Check if current PC matches a reconvergence point (issue stage)
    wire [1:0] curr_div_ptr = sm_div_stack_ptr[issue_warp_id];
    wire [1:0] div_top_idx = (curr_div_ptr > 0) ? (curr_div_ptr - 2'd1) : 2'd0;
    wire sm_at_reconverge = (curr_div_ptr > 0) &&
                           (sm_div_stack_pc[issue_warp_id][div_top_idx] == issue_pc) &&
                           issue_valid && !issue_branch_op;

    // Lane 1 divergence stack pointer
    wire [1:0] curr1_div_ptr = sm_div_stack_ptr[issue1_warp_id];

    // Decode-stage reconvergence detection (for early mask merge)
    wire [1:0] dec0_div_ptr = sm_div_stack_ptr[dec0_warp_id];
    wire [1:0] dec0_top_idx = (dec0_div_ptr > 0) ? (dec0_div_ptr - 2'd1) : 2'd0;
    wire dec0_at_reconverge = (dec0_div_ptr > 0) &&
                             (sm_div_stack_pc[dec0_warp_id][dec0_top_idx] == dec0_pc) &&
                             dec_valid && !dec_branch_op;
    wire [NUM_LANES-1:0] dec0_merged_mask = warp_mask[dec0_warp_id] |
                                            sm_div_stack_mask[dec0_warp_id][dec0_top_idx];

    // Assign flush signals - flush on taken branch OR divergent branch (mask changes)
    // For divergent branch, we flush to ensure next instruction uses updated mask
    // Include both lane 0 and lane 1 branches
    wire any_branch_flush = branch_taken_combined || divergent_branch ||
                           branch1_taken_combined || divergent_branch1;
    assign branch_flush_dec0 = any_branch_flush && ((dec0_warp_id == issue_warp_id) ||
                                                    (dec0_warp_id == issue1_warp_id)) && dec0_valid;
    assign branch_flush_dec1 = any_branch_flush && ((dec1_warp_id == issue_warp_id) ||
                                                    (dec1_warp_id == issue1_warp_id)) && dec1_valid;

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
        // Branch type from bits [25:24] of instruction (rd[4:3])
        // 00=unconditional, 01=if_zero (BR_IF_TRUE), 10=if_not_zero (BR_IF_FALSE), 11=uniform
        .branch_type     ({4'b0, issue_rd[4:3]}),
        // Branch offset from imm16, sign-extended
        .branch_target   (issue_pc + {{16{issue_imm16[15]}}, issue_imm16}),
        // Branch condition: zero flag from ALU (ra | 0)
        .branch_cond     (alu_zero),
        .is_uniform      (issue_rd[4:3] == 2'b11),
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

    //------------------------------------------------------------------------
    // Memory Response Latch
    // Capture memory response data when it arrives, since gmem_resp_valid
    // is only high for one cycle but writeback arbiter runs one cycle later
    //------------------------------------------------------------------------
    reg gmem_resp_latched;
    reg [WARP_ID_W-1:0] gmem_resp_warp;
    reg [4:0] gmem_resp_rd;
    reg [NUM_LANES*DATA_WIDTH-1:0] gmem_resp_data;
    reg [NUM_LANES-1:0] gmem_resp_mask;

    reg smem_resp_latched;
    reg [WARP_ID_W-1:0] smem_resp_warp;
    reg [4:0] smem_resp_rd;
    reg [NUM_LANES*DATA_WIDTH-1:0] smem_resp_data;
    reg [NUM_LANES-1:0] smem_resp_mask;

    // mbarrier result latch (for test_wait/try_wait writeback)
    reg mbarrier_result_latched;
    reg [WARP_ID_W-1:0] mbarrier_wb_warp;
    reg [4:0] mbarrier_wb_rd;
    reg [31:0] mbarrier_wb_result;
    reg [NUM_LANES-1:0] mbarrier_wb_mask;

    `ifdef SIMULATION
    reg [7:0] latch_dbg_cnt;
    `endif
    always @(posedge clk or negedge rst_n) begin : latch_block
        integer init_idx;
        if (!rst_n) begin
            gmem_resp_latched <= 1'b0;
            smem_resp_latched <= 1'b0;
            mbarrier_result_latched <= 1'b0;
            cache_policy_token_valid_r <= 1'b0;
            cache_policy_token_r <= 32'b0;
            cache_policy_wb_warp <= 0;
            cache_policy_wb_rd <= 5'b0;
            cache_policy_wb_mask <= 0;
            // Stack state initialization
            stack_result_valid_r <= 1'b0;
            stack_result_r <= 32'b0;
            stack_wb_warp <= 0;
            stack_wb_rd <= 5'b0;
            stack_wb_mask <= 0;
            // Debug state initialization
            debug_brkpt_event <= 1'b0;
            debug_trap_event <= 1'b0;
            debug_pmevent_id <= 32'b0;
            debug_pmevent_valid <= 1'b0;
            // Initialize per-warp state
            for (init_idx = 0; init_idx < NUM_WARPS; init_idx = init_idx + 1) begin
                warp_stack_ptr[init_idx] <= 32'h80000000;  // Default stack base in local memory
                warp_nanosleep_counter[init_idx] <= 32'b0;
                warp_maxnreg[init_idx] <= 16'd32;  // Default max 32 registers
                warp_st_async_pending[init_idx] <= 8'b0;
                warp_st_async_groups[init_idx] <= 4'b0;
            end
            // Multimem state initialization
            multimem_result_valid_r <= 1'b0;
            multimem_result_r <= 32'b0;
            multimem_wb_warp <= 0;
            multimem_wb_rd <= 5'b0;
            multimem_wb_mask <= 0;
            `ifdef SIMULATION
            latch_dbg_cnt <= 0;
            `endif
        end else begin
            // Latch global memory response
            if (gmem_resp_valid && mem_pending_valid && !gmem_resp_latched) begin
                gmem_resp_latched <= 1'b1;
                gmem_resp_warp <= mem_warp_pending;
                gmem_resp_rd <= mem_rd_pending;
                gmem_resp_data <= gmem_resp_rdata;
                gmem_resp_mask <= mem_mask_pending;
                `ifdef SIMULATION
                if (latch_dbg_cnt < 10) begin
                    `ifdef SIMULATION
                    $display("[%0t SM%0d GMEM_LATCH] CAPTURED rd=R%0d warp=%0d data[0]=0x%08x",
                             $time, SM_ID, mem_rd_pending, mem_warp_pending, gmem_resp_rdata[31:0]);
                    `endif
                    latch_dbg_cnt <= latch_dbg_cnt + 1;
                end
                `endif
            end else if (gmem_resp_latched && wb_found && wb_sel == 4'd7 && !smem_resp_latched) begin
                gmem_resp_latched <= 1'b0;  // Clear latch when writeback consumes it
                `ifdef SIMULATION
                if (latch_dbg_cnt < 10) begin
                    `ifdef SIMULATION
                    $display("[%0t SM%0d GMEM_LATCH] CONSUMED rd=R%0d", $time, SM_ID, gmem_resp_rd);
                    `endif
                    latch_dbg_cnt <= latch_dbg_cnt + 1;
                end
                `endif
            end

            // Latch shared memory response
            if (smem_resp_valid && smem_pending_valid && !smem_resp_latched) begin
                smem_resp_latched <= 1'b1;
                smem_resp_warp <= smem_warp_pending;
                smem_resp_rd <= smem_rd_pending;
                smem_resp_data <= smem_resp_rdata;
                smem_resp_mask <= smem_mask_pending;
            end else if (smem_resp_latched && wb_found && wb_sel == 4'd7) begin
                smem_resp_latched <= 1'b0;  // Clear latch when writeback consumes it
            end

            // Latch mbarrier result (for test_wait/try_wait)
            if (mbarrier_result_valid && !mbarrier_result_latched) begin
                mbarrier_result_latched <= 1'b1;
                mbarrier_wb_warp <= mbarrier_warp;
                mbarrier_wb_rd <= issue_mbarrier ? issue_rd : issue1_rd;
                mbarrier_wb_result <= mbarrier_result;
                mbarrier_wb_mask <= mbarrier_mask;
            end else if (mbarrier_result_latched && wb_found && wb_sel == 4'd11) begin
                mbarrier_result_latched <= 1'b0;  // Clear latch when writeback consumes it
            end

            // Latch cache policy token (for createpolicy writeback)
            // createpolicy generates a token synchronously, so latch when instruction issues
            if (cache_policy_create && !cache_policy_token_valid_r) begin
                cache_policy_token_valid_r <= 1'b1;
                cache_policy_wb_warp <= issue_cache_policy ? issue_warp_id : issue1_warp_id;
                cache_policy_wb_rd <= issue_cache_policy ? issue_rd : issue1_rd;
                cache_policy_wb_mask <= issue_cache_policy ? issue_mask : issue1_mask;
                // Generate policy token: [31:24]=magic(0xCA), [23:16]=priority, [15:8]=reserved, [7:0]=policy_id
                cache_policy_token_r <= {8'hCA, cache_policy_create_priority, 8'h00, 5'b0, cache_policy_create_id};
            end else if (cache_policy_token_valid_r && wb_found && wb_sel == 4'd14) begin
                cache_policy_token_valid_r <= 1'b0;  // Clear latch when writeback consumes it
            end

            // Stack operations (alloca/stacksave write results, stackrestore updates internal state)
            if (stack_alloca && !stack_result_valid_r) begin
                stack_result_valid_r <= 1'b1;
                stack_wb_warp <= stack_warp_id;
                stack_wb_rd <= issue_stack ? issue_rd : issue1_rd;
                stack_wb_mask <= issue_stack ? issue_mask : issue1_mask;
                // Return current stack pointer (before allocation), then bump pointer
                stack_result_r <= warp_stack_ptr[stack_warp_id];
                warp_stack_ptr[stack_warp_id] <= warp_stack_ptr[stack_warp_id] + stack_alloca_size;
            end else if (stack_save && !stack_result_valid_r) begin
                stack_result_valid_r <= 1'b1;
                stack_wb_warp <= stack_warp_id;
                stack_wb_rd <= issue_stack ? issue_rd : issue1_rd;
                stack_wb_mask <= issue_stack ? issue_mask : issue1_mask;
                stack_result_r <= warp_stack_ptr[stack_warp_id];
            end else if (stack_restore) begin
                // stackrestore doesn't write to register, just updates stack pointer
                warp_stack_ptr[stack_warp_id] <= stack_restore_ptr;
            end else if (stack_result_valid_r && wb_found && wb_sel == 4'd15) begin
                stack_result_valid_r <= 1'b0;  // Clear latch when writeback consumes it
            end

            // Debug operations (generate events)
            debug_brkpt_event <= 1'b0;
            debug_trap_event <= 1'b0;
            debug_pmevent_valid <= 1'b0;
            if (debug_active) begin
                case (debug_func)
                    `DEBUG_BRKPT: debug_brkpt_event <= 1'b1;
                    `DEBUG_TRAP: debug_trap_event <= 1'b1;
                    `DEBUG_PMEVENT: begin
                        debug_pmevent_valid <= 1'b1;
                        debug_pmevent_id <= issue_debug ? rf_rd_data_a[0] : rf1_rd_data_a[0];
                    end
                endcase
            end

            // Misc operations (nanosleep, setmaxnreg)
            if (misc_nanosleep) begin
                warp_nanosleep_counter[misc_warp_id] <= nanosleep_cycles;
            end
            if (misc_setmaxnreg) begin
                warp_maxnreg[misc_warp_id] <= issue_misc ? issue_imm16 : issue1_imm16;
            end

            // Decrement nanosleep counters for all sleeping warps
            begin : nanosleep_decrement
                integer ns_idx;
                for (ns_idx = 0; ns_idx < NUM_WARPS; ns_idx = ns_idx + 1) begin
                    if (warp_nanosleep_counter[ns_idx] > 0) begin
                        warp_nanosleep_counter[ns_idx] <= warp_nanosleep_counter[ns_idx] - 1;
                    end
                end
            end

            // st.async operations
            if (st_async_active) begin
                if (st_async_commit) begin
                    // Commit current async store group - increment group count
                    warp_st_async_groups[st_async_warp_id] <= warp_st_async_groups[st_async_warp_id] + 1;
                    warp_st_async_pending[st_async_warp_id] <= 8'b0;  // Reset pending count for new group
                end else if (st_async_wait) begin
                    // Wait handled in warp scheduler - decrement when stores complete
                    // For simulation, assume instant completion
                    if (warp_st_async_groups[st_async_warp_id] >= st_async_wait_count) begin
                        warp_st_async_groups[st_async_warp_id] <= warp_st_async_groups[st_async_warp_id] - st_async_wait_count;
                    end else begin
                        warp_st_async_groups[st_async_warp_id] <= 4'b0;
                    end
                end else begin
                    // Actual async store - increment pending count
                    warp_st_async_pending[st_async_warp_id] <= warp_st_async_pending[st_async_warp_id] + 1;
                    // Note: Actual store routing happens through existing memory paths
                    // The store is marked as async and goes into the async queue
                end
            end

            // multimem.ld result writeback handling
            if ((multimem_func == `MULTIMEM_LD) && multimem_active && !multimem_result_valid_r) begin
                // For distributed shared memory load - currently reads local shared memory
                // Full implementation needs cluster interconnect
                multimem_result_valid_r <= 1'b1;
                multimem_wb_warp <= multimem_warp_id;
                multimem_wb_rd <= issue_multimem ? issue_rd : issue1_rd;
                multimem_wb_mask <= issue_multimem ? issue_mask : issue1_mask;
                // Result comes from local shared memory for now (placeholder)
                multimem_result_r <= 32'h0;  // TODO: Wire to actual distributed smem result
            end else if (multimem_result_valid_r && wb_found && wb_sel == 5'd16) begin
                multimem_result_valid_r <= 1'b0;  // Clear latch when writeback consumes it
            end
        end
    end

    // Collect all ready FU outputs for round-robin arbitration
    wire [16:0] fu_ready;
    assign fu_ready[0] = !alu_wbq_empty;                     // ALU (queued)
    assign fu_ready[1] = !mul_wbq_empty;                     // MUL (queued)
    assign fu_ready[2] = !fpu32_wbq_empty;                   // FPU32 (queued)
    assign fu_ready[3] = !fpu64_wbq_empty;                   // FPU64 (queued)
    assign fu_ready[4] = !fp16_wbq_empty;                    // FP16 (queued)
    assign fu_ready[5] = !sfu_wbq_empty;                     // SFU (queued)
    assign fu_ready[6] = !tensor_wbq_empty;                  // Tensor (queued)
    // Memory is ready if we have a latched response (from previous cycle) or store pending
    // Note: We latch on cycle N when resp_valid arrives, then fu_ready[7]=1 on cycle N+1
    assign fu_ready[7] = gmem_resp_latched || smem_resp_latched || store_pending_valid;
    assign fu_ready[8] = !shfl_wbq_empty;                    // Shuffle (queued)
    assign fu_ready[9] = atomic_valid_out;                   // Atomic
    assign fu_ready[10] = !special_wbq_empty;                // Special registers (queued)
    assign fu_ready[11] = mbarrier_result_latched;           // mbarrier (test_wait/try_wait results)
    assign fu_ready[12] = tex_result_valid_latched;          // Texture (latched results)
    assign fu_ready[13] = !video_wbq_empty;                  // Video SIMD (queued)
    assign fu_ready[14] = cache_policy_token_valid_r;        // Cache policy (createpolicy results)
    assign fu_ready[15] = stack_result_valid_r;              // Stack (alloca/stacksave results)
    assign fu_ready[16] = multimem_result_valid_r;           // Multimem (distributed smem load results)

    // Round-robin selection for writeback
    reg [4:0] wb_sel;  // 5 bits for 17 FUs
    reg       wb_found;
    integer   wb_i;

    always @(*) begin
        wb_found = 1'b0;
        wb_sel = 0;
        // Start from last priority + 1 for fairness (17 FU sources)
        for (wb_i = 0; wb_i < 17; wb_i = wb_i + 1) begin
            if (!wb_found && fu_ready[(wb_arb_priority + wb_i) % 17]) begin
                wb_sel = (wb_arb_priority + wb_i) % 17;
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
    assign special_wbq_pop = wb_found && (wb_sel == 4'd10);
    wire tex_wbq_pop = wb_found && (wb_sel == 4'd12);
    assign video_wbq_pop = wb_found && (wb_sel == 4'd13);

    // Writeback arbiter with proper warp/rd tracking from FU pipelines
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wb_valid <= 1'b0;
            wb_arb_priority <= 0;
        end else begin
            if (wb_found) begin
                wb_valid <= 1'b1;
                wb_arb_priority <= (wb_sel + 1) % 17;  // Advance for fairness (17 FU sources)

                // DEBUG: trace which FU is causing writebacks
                `ifdef SIMULATION
                $display("[SM%0d] WB_SEL: wb_sel=%0d fu_ready=%014b", SM_ID, wb_sel, fu_ready);
                `endif

                case (wb_sel)
                    4'd0: begin  // ALU (1-cycle pipeline for proper timing)
                        wb_warp_id <= alu_wbq_warp;
                        wb_rd <= alu_wbq_rd;
                        wb_data <= alu_wbq_data;
                        wb_mask <= alu_wbq_mask;
                        `ifdef SIMULATION
                        $display("[SM%0d] ALU_WB: rd=R%0d warp=%0d mask=0x%08x data[0]=0x%08x",
                                 SM_ID, alu_wbq_rd, alu_wbq_warp, alu_wbq_mask, alu_wbq_data[31:0]);
                        `endif
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
                        `ifdef SIMULATION
                        $display("[SM%0d] FPU32_WB: rd=R%0d warp=%0d mask=0x%08x data[0]=0x%08x",
                                 SM_ID, fpu32_wbq_rd, fpu32_wbq_warp, fpu32_wbq_mask, fpu32_wbq_data[31:0]);
                        `endif
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
                        `ifdef SIMULATION
                        $display("[SM%0d] SFU_WB: rd=R%0d warp=%0d mask=0x%08x data[0]=0x%08x",
                                 SM_ID, sfu_wbq_rd, sfu_wbq_warp, sfu_wbq_mask, sfu_wbq_data[31:0]);
                        `endif
                    end
                    4'd6: begin  // Tensor (variable latency)
                        wb_warp_id <= tensor_wbq_warp;
                        wb_rd <= tensor_wbq_rd;
                        wb_data <= tensor_wbq_data;
                        wb_mask <= tensor_wbq_mask;
                    end
                    4'd7: begin  // Memory - use latched values
                        `ifdef SIMULATION
                        // $display("[%0t SM%0d CASE7] ...", $time, SM_ID, ...); // Debug disabled
                        `endif
                        if (smem_resp_latched) begin
                            wb_warp_id <= smem_resp_warp;
                            wb_rd <= smem_resp_rd;
                            wb_data <= smem_resp_data;
                            wb_mask <= smem_resp_mask;
                        end else if (gmem_resp_latched) begin
                            wb_warp_id <= gmem_resp_warp;
                            wb_rd <= gmem_resp_rd;
                            wb_data <= gmem_resp_data;
                            wb_mask <= gmem_resp_mask;
                        end else begin
                            // Store completion (no register writeback)
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
                    4'd10: begin  // Special registers (MOV_SPECIAL)
                        wb_warp_id <= special_wbq_warp;
                        wb_rd <= special_wbq_rd;
                        wb_data <= special_wbq_data;
                        wb_mask <= special_wbq_mask;
                    end
                    4'd11: begin  // mbarrier (test_wait/try_wait results)
                        wb_warp_id <= mbarrier_wb_warp;
                        wb_rd <= mbarrier_wb_rd;
                        wb_data <= {NUM_LANES{mbarrier_wb_result}};  // Replicate result to all lanes
                        wb_mask <= mbarrier_wb_mask;
                    end
                    4'd12: begin  // Texture unit (tex/txq/suld/sust/sured)
                        wb_warp_id <= tex_warp_pending;
                        wb_rd <= tex_rd_pending;
                        // Texture returns 128-bit result (RGBA 4x32-bit), replicate across lanes
                        wb_data <= {(NUM_LANES/4){tex_result_latched}};
                        wb_mask <= tex_mask_pending;
                        `ifdef SIMULATION
                        $display("[SM%0d] TEX_WB: rd=R%0d warp=%0d mask=0x%08x result=0x%032x",
                                 SM_ID, tex_rd_pending, tex_warp_pending, tex_mask_pending, tex_result_latched);
                        `endif
                    end
                    4'd13: begin  // Video SIMD unit (VADD4/VSUB4/DP4A/DP2A/etc)
                        wb_warp_id <= video_wbq_warp;
                        wb_rd <= video_wbq_rd;
                        wb_data <= video_wbq_data;
                        wb_mask <= video_wbq_mask;
                        `ifdef SIMULATION
                        $display("[SM%0d] VIDEO_WB: rd=R%0d warp=%0d mask=0x%08x data[0]=0x%08x",
                                 SM_ID, video_wbq_rd, video_wbq_warp, video_wbq_mask, video_wbq_data[31:0]);
                        `endif
                    end
                    4'd14: begin  // Cache policy (createpolicy token result)
                        wb_warp_id <= cache_policy_wb_warp;
                        wb_rd <= cache_policy_wb_rd;
                        wb_data <= {NUM_LANES{cache_policy_token_r}};  // Replicate token to all lanes
                        wb_mask <= cache_policy_wb_mask;
                        `ifdef SIMULATION
                        $display("[SM%0d] CACHE_POLICY_WB: rd=R%0d warp=%0d mask=0x%08x token=0x%08x",
                                 SM_ID, cache_policy_wb_rd, cache_policy_wb_warp, cache_policy_wb_mask, cache_policy_token_r);
                        `endif
                    end
                    4'd15: begin  // Stack (alloca/stacksave result)
                        wb_warp_id <= stack_wb_warp;
                        wb_rd <= stack_wb_rd;
                        wb_data <= {NUM_LANES{stack_result_r}};  // Replicate stack pointer to all lanes
                        wb_mask <= stack_wb_mask;
                        `ifdef SIMULATION
                        $display("[SM%0d] STACK_WB: rd=R%0d warp=%0d mask=0x%08x ptr=0x%08x",
                                 SM_ID, stack_wb_rd, stack_wb_warp, stack_wb_mask, stack_result_r);
                        `endif
                    end
                    5'd16: begin  // Multimem (distributed shared memory load result)
                        wb_warp_id <= multimem_wb_warp;
                        wb_rd <= multimem_wb_rd;
                        wb_data <= {NUM_LANES{multimem_result_r}};  // Replicate result to all lanes
                        wb_mask <= multimem_wb_mask;
                        `ifdef SIMULATION
                        $display("[SM%0d] MULTIMEM_WB: rd=R%0d warp=%0d mask=0x%08x data=0x%08x",
                                 SM_ID, multimem_wb_rd, multimem_wb_warp, multimem_wb_mask, multimem_result_r);
                        `endif
                    end
                endcase
            end else begin
                wb_valid <= 1'b0;
            end
        end
    end

    // Register file write
    // Note: PTX/CUDA allows writes to R0 (unlike RISC-V where R0 is hardwired to 0)
    assign rf_wr_en = wb_valid;
    assign rf_wr_data = wb_data;
    assign rf_wr_mask = wb_mask;

    // DEBUG: Track memory writeback - disabled for faster simulation
    `ifdef DEBUG_MEMWB
    always @(posedge clk) begin
        if (gmem_resp_valid)
            `ifdef SIMULATION
            $display("[%0t SM%0d MEM_RESP] valid, gmem_latched=%b smem_latched=%b store_pend=%b, mem_rd=%0d, mem_warp=%0d",
                     $time, SM_ID, gmem_resp_latched, smem_resp_latched, store_pending_valid, mem_rd_pending, mem_warp_pending);
            `endif
        if (gmem_resp_latched || smem_resp_latched)
            `ifdef SIMULATION
            $display("[%0t SM%0d LATCH] gmem_latched=%b (rd=%0d warp=%0d) smem_latched=%b wb_found=%b wb_sel=%0d",
                     $time, SM_ID, gmem_resp_latched, gmem_resp_rd, gmem_resp_warp, smem_resp_latched, wb_found, wb_sel);
            `endif
        if (wb_valid && wb_sel == 4'd7)
            `ifdef SIMULATION
            $display("[%0t SM%0d WB_MEM] warp=%0d rd=R%0d gmem_latched=%b smem_latched=%b data=0x%08h",
                     $time, SM_ID, wb_warp_id, wb_rd, gmem_resp_latched, smem_resp_latched, wb_data[31:0]);
            `endif
    end
    `endif

    // DEBUG: Track R23 register writes and CVT operations
    `ifdef SIMULATION
    always @(posedge clk) begin
        // Track all writes to R1 (loop counter)
        if (rf_wr_en && wb_rd == 5'd1) begin
            `ifdef SIMULATION
            $display("[%0t R1_WRITE] warp=%0d data=0x%08h from %s",
                     $time, wb_warp_id, wb_data[31:0],
                     wb_sel == 4'd0 ? "ALU" : "OTHER");
            `endif
        end
        // Debug: trace all writebacks
        if (rf_wr_en) begin
            `ifdef SIMULATION
            $display("[%0t WB] warp=%0d rd=R%0d data=0x%08h sel=%0d", $time, wb_warp_id, wb_rd, wb_data[31:0], wb_sel);
            `endif
        end
        // Track all writes to R23
        if (rf_wr_en && wb_rd == 5'd23) begin
            `ifdef SIMULATION
            $display("[%0t R23_WRITE] warp=%0d wb_sel=%0d data=0x%08h from %s",
                     $time, wb_warp_id, wb_sel, wb_data[31:0],
                     wb_sel == 4'd0 ? "ALU" :
                     wb_sel == 4'd1 ? "MUL" :
                     wb_sel == 4'd2 ? "FPU32" :
                     wb_sel == 4'd4 ? "FP16" : "OTHER");
            `endif
        end

        // Track CVT instructions
        if (alu_issue && issue_func[5:0] == 6'd40) begin  // CVT_F32_F16 = 40
            `ifdef SIMULATION
            $display("[%0t CVT_ISSUE] rd=R%0d ra=R%0d operand_a[15:0]=0x%04h",
                     $time, alu_issue_rd, dec_ra, alu_op_a[15:0]);
            `endif
        end

        // Track ALU result for CVT
        if (alu_valid_pipe && alu_rd_pipe == 5'd23) begin
            `ifdef SIMULATION
            $display("[%0t ALU_R23_RESULT] alu_result_pipe=0x%08h", $time, alu_result_pipe[31:0]);
            `endif
        end
    end
    `endif

    //========================================================================
    // Warp State Management
    //========================================================================
    integer w, rst_j;
    reg [31:0] init_total_threads;
    reg [31:0] init_threads_in_warp;
    reg [NUM_LANES-1:0] init_computed_mask;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                warp_valid[w] <= 1'b0;
                warp_active[w] <= 1'b0;
                warp_stalled_mem[w] <= 1'b0;
                warp_stalled_fu[w] <= 1'b0;
                warp_stalled_sync[w] <= 1'b0;
                warp_stalled_branch[w] <= 1'b0;
                warp_exit_pending[w] <= 1'b0;
                warp_pc[w] <= 32'b0;
                warp_fetch_pc[w] <= 32'b0;
                warp_mask[w] <= {NUM_LANES{1'b1}};
                // Initialize warp-level sync state (bar.warp.sync)
                warp_sync_pending[w] <= 1'b0;
                warp_sync_arrived[w] <= 32'b0;
                warp_sync_mask[w] <= 32'b0;
                // Initialize cluster barrier state (barrier.cluster)
                cluster_barrier_pending[w] <= 1'b0;
                cluster_barrier_arrived[w] <= 1'b0;
                cluster_barrier_id[w] <= 8'b0;
                // Initialize SM divergence stack
                sm_div_stack_ptr[w] <= 2'b0;
                for (rst_j = 0; rst_j < 4; rst_j = rst_j + 1) begin
                    sm_div_stack_pc[w][rst_j] <= 32'b0;
                    sm_div_stack_mask[w][rst_j] <= {NUM_LANES{1'b0}};
                end
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
            // Kernel start - allocate initial warps with proper masks
            if (kernel_start) begin
                `ifdef SIMULATION
                $display("[SM%0d] kernel_start: block_dim=(%0d,%0d,%0d) kernel_pc=0x%08h",
                         SM_ID, block_dim_x, block_dim_y, block_dim_z, kernel_pc);
                `endif

                init_total_threads = block_dim_x * block_dim_y * block_dim_z;

                for (w = 0; w < NUM_WARPS; w = w + 1) begin
                    if (w < INIT_WARPS) begin
                        // Calculate how many threads belong to this warp
                        // Warp w covers thread IDs [w*32, (w+1)*32-1]
                        if (init_total_threads >= ((w + 1) * NUM_LANES)) begin
                            // Full warp - all 32 threads active
                            init_computed_mask = {NUM_LANES{1'b1}};
                        end else if (init_total_threads > (w * NUM_LANES)) begin
                            // Partial warp - only some threads active
                            init_threads_in_warp = init_total_threads - (w * NUM_LANES);
                            init_computed_mask = (1'b1 << init_threads_in_warp) - 1'b1;
                        end else begin
                            // No threads in this warp
                            init_computed_mask = {NUM_LANES{1'b0}};
                        end

                        warp_valid[w] <= (init_computed_mask != 0);
                        warp_active[w] <= (init_computed_mask != 0);
                        warp_exit_pending[w] <= 1'b0;
                        warp_pc[w] <= kernel_pc;
                        warp_fetch_pc[w] <= kernel_pc;
                        warp_mask[w] <= init_computed_mask;

                        `ifdef SIMULATION
                        $display("[SM%0d] Warp %0d initialized: mask=0x%08h (total_threads=%0d)",
                                 SM_ID, w, init_computed_mask, init_total_threads);
                        `endif
                    end else begin
                        warp_valid[w] <= 1'b0;
                        warp_active[w] <= 1'b0;
                        warp_exit_pending[w] <= 1'b0;
                        warp_pc[w] <= 32'b0;
                        warp_fetch_pc[w] <= 32'b0;
                        warp_mask[w] <= {NUM_LANES{1'b0}};
                    end
                    // Reset warp-level sync state (bar.warp.sync)
                    warp_sync_pending[w] <= 1'b0;
                    warp_sync_arrived[w] <= 32'b0;
                    warp_sync_mask[w] <= 32'b0;
                    // Reset cluster barrier state (barrier.cluster)
                    cluster_barrier_pending[w] <= 1'b0;
                    cluster_barrier_arrived[w] <= 1'b0;
                    cluster_barrier_id[w] <= 8'b0;
                    // Reset all stall signals (prevents stale state from previous kernel)
                    warp_stalled_mem[w] <= 1'b0;
                    warp_stalled_fu[w] <= 1'b0;
                    warp_stalled_sync[w] <= 1'b0;
                    warp_stalled_async[w] <= 1'b0;
                    warp_stalled_branch[w] <= 1'b0;
                end
                // Reset cluster barrier shared state on kernel start
                cluster_barrier_thread_count <= 16'b0;
                cluster_local_arrive_count <= 16'b0;
                cluster_barrier_complete <= 1'b0;
                `ifdef SIMULATION
                $display("[%0t SM%0d] KERNEL_START_DONE: warp_valid=0x%04b fetch_valid_arb=%b imem_ready=%b",
                         $time, SM_ID, warp_valid, fetch_valid_arb, imem_ready);
                `endif
            end

            // NOTE: Fetch PC is advanced in the instruction buffer fill logic (line 1083)
            // when icache returns valid data. Don't advance here on fetch_fire to avoid
            // double-counting. The fetch_fire signal is used for other purposes (arbitration).

            // NOTE: PC is now updated at scheduling time (in decode stage block above)
            // to ensure correct PC tracking for branches. Branches override PC when taken.

            // Branch stall management: set stall when branch is scheduled, clear when it resolves
            // Set stall when branch is issued from scheduler
            if (issue0_fire && pd_is_branch[sched_issue_warp_id[0]] && !branch_flush_dec0) begin
                warp_stalled_branch[sched_issue_warp_id[0]] <= 1'b1;
            end
            if (issue1_fire && pd_is_branch[sched_issue_warp_id[1]] && !branch_flush_dec1) begin
                warp_stalled_branch[sched_issue_warp_id[1]] <= 1'b1;
            end

            // Branch handling (uses combinational simple_branch_taken for immediate response)
            if (issue_valid && issue_branch_op) begin
                `ifdef SIMULATION
                $display("[%0t SM%0d] BRANCH: pc=0x%04x opcode=0x%02x type=%0d ra=R%0d imm16=0x%04x target=0x%04x zero=%b taken=%b diverge=%b rf_rd_a=0x%08h",
                         $time, SM_ID, issue_pc, issue_opcode, issue_rd[4:3], issue_ra, issue_imm16,
                         issue_pc + {{16{issue_imm16[15]}}, issue_imm16},
                         alu_zero[0], branch_taken_combined, threads_diverge, rf_rd_data_a[31:0]);
                `endif
                // For divergent branches, keep stall set for one more cycle to let pipeline flush
                // For non-divergent branches, clear stall immediately
                if (!threads_diverge) begin
                    warp_stalled_branch[issue_warp_id] <= 1'b0;
                end
                // For divergent branches, stall is cleared in the divergent_branch handler below
            end
            // Handle non-divergent branch taken
            if (branch_taken_combined) begin
                `ifdef SIMULATION
                $display("[SM%0d] BRANCH TAKEN: new_pc=0x%04x (flushing pipeline)", SM_ID, branch_target_combined);
                `endif
                `ifdef SIMULATION
                $display("[SM%0d]   flush_dec0=%b (dec0_warp=%0d dec0_valid=%b) flush_dec1=%b issue_warp=%0d",
                         SM_ID, branch_flush_dec0, dec0_warp_id, dec0_valid, branch_flush_dec1, issue_warp_id);
                `endif
                warp_pc[issue_warp_id] <= branch_target_combined;
                warp_fetch_pc[issue_warp_id] <= branch_target_combined;
                warp_mask[issue_warp_id] <= issue_mask;
                // Flush instruction buffer and pipeline for this warp
                warp_inst_buf_valid[issue_warp_id] <= 1'b0;
                warp_fetch_pending[issue_warp_id] <= 1'b0;
            end

            // Handle divergent branch (not-taken-first strategy)
            // Not-taken threads execute fall-through, taken threads pushed to stack
            if (divergent_branch) begin
                `ifdef SIMULATION
                $display("[SM%0d] DIVERGENT BRANCH: pc=0x%04x target=0x%04x",
                         SM_ID, issue_pc, simple_branch_target);
                `endif
                `ifdef SIMULATION
                $display("[SM%0d]   taken_mask=0x%08x not_taken_mask=0x%08x",
                         SM_ID, taken_lanes, not_taken_lanes);
                `endif
                `ifdef SIMULATION
                $display("[SM%0d]   pushing: reconverge_pc=0x%04x waiting_mask=0x%08x",
                         SM_ID, simple_branch_target, taken_lanes);
                `endif

                // Push {branch_target, taken_lanes} to divergence stack
                sm_div_stack_pc[issue_warp_id][curr_div_ptr] <= simple_branch_target;
                sm_div_stack_mask[issue_warp_id][curr_div_ptr] <= taken_lanes;
                sm_div_stack_ptr[issue_warp_id] <= curr_div_ptr + 2'd1;

                // Execute not-taken threads first (fall-through)
                // PC stays at fall-through address (issue_pc + 4)
                warp_mask[issue_warp_id] <= not_taken_lanes;
                warp_pc[issue_warp_id] <= issue_pc + 32'd4;
                warp_fetch_pc[issue_warp_id] <= issue_pc + 32'd4;

                // Flush instruction buffer and pipeline to ensure next fetch uses new mask
                warp_inst_buf_valid[issue_warp_id] <= 1'b0;
                warp_fetch_pending[issue_warp_id] <= 1'b0;

                // Clear branch stall - divergent branch is handled, warp can continue with new mask
                warp_stalled_branch[issue_warp_id] <= 1'b0;
            end

            //------------------------------------------------------------------------
            // Lane 1 Branch Handling (mirrors lane 0 handling above)
            //------------------------------------------------------------------------
            // Branch handling for lane 1 (uses combinational simple_branch1_taken)
            if (issue1_valid && issue1_branch_op) begin
                `ifdef SIMULATION
                $display("[%0t SM%0d] BRANCH1: pc=0x%04x opcode=0x%02x type=%0d ra=R%0d imm16=0x%04x target=0x%04x zero=%b taken=%b diverge=%b",
                         $time, SM_ID, issue1_pc, issue1_opcode, issue1_rd[4:3], issue1_ra, issue1_imm16,
                         issue1_pc + {{16{issue1_imm16[15]}}, issue1_imm16},
                         alu_zero[0], branch1_taken_combined, threads1_diverge);
                `endif
                // For non-divergent branches, clear stall immediately
                if (!threads1_diverge) begin
                    warp_stalled_branch[issue1_warp_id] <= 1'b0;
                end
            end

            // Handle non-divergent branch taken on lane 1
            if (branch1_taken_combined) begin
                `ifdef SIMULATION
                $display("[SM%0d] BRANCH1 TAKEN: new_pc=0x%04x (flushing pipeline)", SM_ID, branch1_target_combined);
                `endif
                warp_pc[issue1_warp_id] <= branch1_target_combined;
                warp_fetch_pc[issue1_warp_id] <= branch1_target_combined;
                warp_mask[issue1_warp_id] <= issue1_mask;
                // Flush instruction buffer and pipeline for this warp
                warp_inst_buf_valid[issue1_warp_id] <= 1'b0;
                warp_fetch_pending[issue1_warp_id] <= 1'b0;
            end

            // Handle divergent branch on lane 1
            if (divergent_branch1) begin
                `ifdef SIMULATION
                $display("[SM%0d] DIVERGENT BRANCH1: pc=0x%04x target=0x%04x",
                         SM_ID, issue1_pc, simple_branch1_target);
                $display("[SM%0d]   taken_mask=0x%08x not_taken_mask=0x%08x",
                         SM_ID, taken1_lanes, not_taken1_lanes);
                `endif

                // Push {branch_target, taken_lanes} to divergence stack
                sm_div_stack_pc[issue1_warp_id][curr1_div_ptr] <= simple_branch1_target;
                sm_div_stack_mask[issue1_warp_id][curr1_div_ptr] <= taken1_lanes;
                sm_div_stack_ptr[issue1_warp_id] <= curr1_div_ptr + 2'd1;

                // Execute not-taken threads first (fall-through)
                warp_mask[issue1_warp_id] <= not_taken1_lanes;
                warp_pc[issue1_warp_id] <= issue1_pc + 32'd4;
                warp_fetch_pc[issue1_warp_id] <= issue1_pc + 32'd4;

                // Flush instruction buffer and pipeline
                warp_inst_buf_valid[issue1_warp_id] <= 1'b0;
                warp_fetch_pending[issue1_warp_id] <= 1'b0;

                // Clear branch stall
                warp_stalled_branch[issue1_warp_id] <= 1'b0;
            end

            // Handle SM-level reconvergence (when PC matches stacked reconverge_pc)
            // Note: Reconvergence is now handled at decode stage (dec0_at_reconverge)
            // to ensure the instruction uses the merged mask
            if (sm_at_reconverge) begin
                `ifdef SIMULATION
                $display("[SM%0d] ISSUE RECONVERGE (verify): PC=0x%04x issue_mask=0x%08x",
                         SM_ID, issue_pc, issue_mask);
                `endif
            end

            // Branch predictor update (when branch resolves)
            // Priority: lane 0, then lane 1 (only one update per cycle)
            if (issue_valid && issue_branch_op) begin
                bp_update_valid <= 1'b1;
                bp_update_warp_id <= issue_warp_id;
                bp_update_pc <= issue_pc;
                bp_update_taken <= branch_taken_combined;
                bp_update_target <= branch_target_combined;
                bp_update_is_call <= (issue_func == 6'h01);  // JAL-like
                bp_update_is_return <= (issue_func == 6'h02); // RET-like
                bp_update_mispredicted <= 1'b0;  // TODO: Compare with prediction
            end else if (issue1_valid && issue1_branch_op) begin
                bp_update_valid <= 1'b1;
                bp_update_warp_id <= issue1_warp_id;
                bp_update_pc <= issue1_pc;
                bp_update_taken <= branch1_taken_combined;
                bp_update_target <= branch1_target_combined;
                bp_update_is_call <= (issue1_func == 6'h01);
                bp_update_is_return <= (issue1_func == 6'h02);
                bp_update_mispredicted <= 1'b0;
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

            // Block-level sync barrier (bar.sync)
            // Only stall for bar.sync (not bar.warp.sync)
            if (issue_valid && issue_sync_op && !issue_bar_warp_sync) begin
                warp_stalled_sync[issue_warp_id] <= 1'b1;
                // Clear branch stall (bar.sync is in pd_is_branch but not an actual branch)
                warp_stalled_branch[issue_warp_id] <= 1'b0;
                // Advance PC past the barrier instruction (since pd_is_branch prevents auto-advance)
                warp_pc[issue_warp_id] <= issue_pc + 32'd4;
                warp_fetch_pc[issue_warp_id] <= issue_pc + 32'd4;
                // Flush instruction buffer to force re-fetch at new PC after barrier releases
                warp_inst_buf_valid[issue_warp_id] <= 1'b0;
                warp_fetch_pending[issue_warp_id] <= 1'b0;
                `ifdef SIMULATION
                $display("[SM%0d] bar.sync issued: warp=%0d, setting warp_stalled_sync, advancing PC to 0x%08x",
                         SM_ID, issue_warp_id, issue_pc + 32'd4);
                `endif
            end
            if (issue1_valid && issue1_sync_op && !issue1_bar_warp_sync) begin
                warp_stalled_sync[issue1_warp_id] <= 1'b1;
                // Clear branch stall (bar.sync is in pd_is_branch but not an actual branch)
                warp_stalled_branch[issue1_warp_id] <= 1'b0;
                // Advance PC past the barrier instruction
                warp_pc[issue1_warp_id] <= issue1_pc + 32'd4;
                warp_fetch_pc[issue1_warp_id] <= issue1_pc + 32'd4;
                // Flush instruction buffer to force re-fetch at new PC after barrier releases
                warp_inst_buf_valid[issue1_warp_id] <= 1'b0;
                warp_fetch_pending[issue1_warp_id] <= 1'b0;
                `ifdef SIMULATION
                $display("[SM%0d] bar.sync issued (slot1): warp=%0d, setting warp_stalled_sync, advancing PC to 0x%08x",
                         SM_ID, issue1_warp_id, issue1_pc + 32'd4);
                `endif
            end
            // Check all warps at barrier and release
            if (all_at_barrier) begin
                `ifdef SIMULATION
                $display("[SM%0d] all_at_barrier=1, releasing all warps! stalled=%04b valid=%04b pending=%b",
                         SM_ID, warp_stalled_sync, warp_valid, barrier_pending);
                `endif
                for (w = 0; w < NUM_WARPS; w = w + 1) begin
                    warp_stalled_sync[w] <= 1'b0;
                end
            end

            //================================================================
            // Warp-level sync barrier (bar.warp.sync)
            //================================================================
            // When bar.warp.sync issues:
            // - Set the expected membermask from register file (rf_rd_data_a[31:0])
            // - Mark arriving threads based on issue_mask
            // - Stall the warp (using warp_stalled_sync)
            //
            // When all specified threads arrive:
            // - Release the warp by clearing warp_stalled_sync
            // - Reset arrival tracking for next barrier

            // Handle bar.warp.sync from issue slot 0
            if (issue_valid && issue_bar_warp_sync) begin
                // Set expected mask from ra register (first lane's value as membermask)
                warp_sync_mask[issue_warp_id] <= rf_rd_data_a[31:0];
                // Mark arriving threads (issue_mask is NUM_LANES bits = 32 bits)
                warp_sync_arrived[issue_warp_id] <= warp_sync_arrived[issue_warp_id] | issue_mask;
                // Mark sync pending and stall the warp
                warp_sync_pending[issue_warp_id] <= 1'b1;
                warp_stalled_sync[issue_warp_id] <= 1'b1;
                `ifdef SIMULATION
                $display("[SM%0d] bar.warp.sync: warp=%0d membermask=0x%08x arriving=0x%08x",
                         SM_ID, issue_warp_id, rf_rd_data_a[31:0], issue_mask);
                `endif
            end

            // Handle bar.warp.sync from issue slot 1
            if (issue1_valid && issue1_bar_warp_sync) begin
                warp_sync_mask[issue1_warp_id] <= rf1_rd_data_a[31:0];
                warp_sync_arrived[issue1_warp_id] <= warp_sync_arrived[issue1_warp_id] | issue1_mask;
                warp_sync_pending[issue1_warp_id] <= 1'b1;
                warp_stalled_sync[issue1_warp_id] <= 1'b1;
                `ifdef SIMULATION
                $display("[SM%0d] bar.warp.sync: warp=%0d membermask=0x%08x arriving=0x%08x",
                         SM_ID, issue1_warp_id, rf1_rd_data_a[31:0], issue1_mask);
                `endif
            end

            // Check for warp-level sync completion and release
            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                if (warp_sync_pending[w]) begin
                    // Check if all participating threads have arrived
                    // (arrived & mask) == mask means all required threads are here
                    if ((warp_sync_arrived[w] & warp_sync_mask[w]) == warp_sync_mask[w]) begin
                        // All threads arrived - release the warp
                        warp_stalled_sync[w] <= 1'b0;
                        warp_sync_pending[w] <= 1'b0;
                        warp_sync_arrived[w] <= 32'b0;  // Reset for next barrier
                        `ifdef SIMULATION
                        $display("[SM%0d] bar.warp.sync COMPLETE: warp=%0d", SM_ID, w);
                        `endif
                    end
                end
            end

            //================================================================
            // Cluster Barrier Operations (barrier.cluster - Hopper+)
            //================================================================
            // Handle barrier.cluster from issue slot 0
            if (issue_valid && issue_barrier_cluster_op) begin
                case (issue_func)
                    `CLUSTER_BARRIER_INIT: begin
                        // Initialize cluster barrier with expected thread count
                        // ra = expected thread count across all SMs in cluster
                        cluster_barrier_thread_count <= rf_rd_data_a[15:0];
                        cluster_local_arrive_count <= 16'b0;
                        cluster_barrier_complete <= 1'b0;
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.init: warp=%0d thread_count=%0d",
                                 SM_ID, issue_warp_id, rf_rd_data_a[15:0]);
                        `endif
                    end
                    `CLUSTER_BARRIER_ARRIVE: begin
                        // Signal arrival at cluster barrier (non-blocking)
                        cluster_barrier_arrived[issue_warp_id] <= 1'b1;
                        cluster_barrier_id[issue_warp_id] <= issue_imm16[7:0];
                        // Increment local arrive count by active thread count
                        cluster_local_arrive_count <= cluster_local_arrive_count + $countones(issue_mask);
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.arrive: warp=%0d barrier_id=%0d",
                                 SM_ID, issue_warp_id, issue_imm16[7:0]);
                        `endif
                    end
                    `CLUSTER_BARRIER_WAIT: begin
                        // Wait for all threads to arrive (blocking)
                        if (!cluster_barrier_complete) begin
                            cluster_barrier_pending[issue_warp_id] <= 1'b1;
                            `ifdef SIMULATION
                            $display("[SM%0d] barrier.cluster.wait: warp=%0d STALLED",
                                     SM_ID, issue_warp_id);
                            `endif
                        end
                    end
                    `CLUSTER_BARRIER_SYNC: begin
                        // Combined arrive and wait (most common)
                        cluster_barrier_arrived[issue_warp_id] <= 1'b1;
                        cluster_barrier_id[issue_warp_id] <= issue_imm16[7:0];
                        cluster_local_arrive_count <= cluster_local_arrive_count + $countones(issue_mask);
                        if (!cluster_barrier_complete) begin
                            cluster_barrier_pending[issue_warp_id] <= 1'b1;
                        end
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.sync: warp=%0d barrier_id=%0d",
                                 SM_ID, issue_warp_id, issue_imm16[7:0]);
                        `endif
                    end
                    default: ;  // Unknown barrier.cluster operation
                endcase
            end

            // Handle barrier.cluster from issue slot 1
            if (issue1_valid && issue1_barrier_cluster_op) begin
                case (issue1_func)
                    `CLUSTER_BARRIER_INIT: begin
                        cluster_barrier_thread_count <= rf1_rd_data_a[15:0];
                        cluster_local_arrive_count <= 16'b0;
                        cluster_barrier_complete <= 1'b0;
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.init: warp=%0d thread_count=%0d",
                                 SM_ID, issue1_warp_id, rf1_rd_data_a[15:0]);
                        `endif
                    end
                    `CLUSTER_BARRIER_ARRIVE: begin
                        cluster_barrier_arrived[issue1_warp_id] <= 1'b1;
                        cluster_barrier_id[issue1_warp_id] <= issue1_imm16[7:0];
                        cluster_local_arrive_count <= cluster_local_arrive_count + $countones(issue1_mask);
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.arrive: warp=%0d barrier_id=%0d",
                                 SM_ID, issue1_warp_id, issue1_imm16[7:0]);
                        `endif
                    end
                    `CLUSTER_BARRIER_WAIT: begin
                        if (!cluster_barrier_complete) begin
                            cluster_barrier_pending[issue1_warp_id] <= 1'b1;
                            `ifdef SIMULATION
                            $display("[SM%0d] barrier.cluster.wait: warp=%0d STALLED",
                                     SM_ID, issue1_warp_id);
                            `endif
                        end
                    end
                    `CLUSTER_BARRIER_SYNC: begin
                        cluster_barrier_arrived[issue1_warp_id] <= 1'b1;
                        cluster_barrier_id[issue1_warp_id] <= issue1_imm16[7:0];
                        cluster_local_arrive_count <= cluster_local_arrive_count + $countones(issue1_mask);
                        if (!cluster_barrier_complete) begin
                            cluster_barrier_pending[issue1_warp_id] <= 1'b1;
                        end
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster.sync: warp=%0d barrier_id=%0d",
                                 SM_ID, issue1_warp_id, issue1_imm16[7:0]);
                        `endif
                    end
                    default: ;
                endcase
            end

            // Check for cluster barrier completion
            // Note: For single-SM operation, completion is when local count reaches expected
            // For multi-SM clusters, this would be signaled by cluster interconnect
            if (!cluster_barrier_complete && (cluster_local_arrive_count >= cluster_barrier_thread_count) && (cluster_barrier_thread_count > 0)) begin
                cluster_barrier_complete <= 1'b1;
                // Release all waiting warps
                for (w = 0; w < NUM_WARPS; w = w + 1) begin
                    if (cluster_barrier_pending[w]) begin
                        cluster_barrier_pending[w] <= 1'b0;
                        cluster_barrier_arrived[w] <= 1'b0;
                        `ifdef SIMULATION
                        $display("[SM%0d] barrier.cluster COMPLETE: releasing warp=%0d", SM_ID, w);
                        `endif
                    end
                end
                // Reset for next barrier
                cluster_local_arrive_count <= 16'b0;
            end

            // Exit instruction (defer warp teardown until in-flight ops drain)
            if (issue_valid && issue_exit_op) begin
                warp_exit_pending[issue_warp_id] <= 1'b1;
            end

            for (w = 0; w < NUM_WARPS; w = w + 1) begin
                if (warp_exit_pending[w] &&
                    (pending_fu_count[w] == 0) &&
                    (cp_async_pending[w] == 0) &&
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

    //========================================================================
    // Debug: Track warp_valid and fetch state after kernel_start
    //========================================================================
    `ifdef SIMULATION
    reg [7:0] post_kernel_debug_cnt;
    reg kernel_started_dbg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            post_kernel_debug_cnt <= 0;
            kernel_started_dbg <= 0;
        end else begin
            if (kernel_start) begin
                kernel_started_dbg <= 1;
                post_kernel_debug_cnt <= 0;
            end
            if (kernel_started_dbg && post_kernel_debug_cnt < 30) begin
                $display("[%0t SM%0d POST_START] cycle=%0d warp_valid=%04b warp_ready=%04b fetch_req=%b imem_ready=%b buf_valid=%04b",
                         $time, SM_ID, post_kernel_debug_cnt,
                         warp_valid, warp_ready, fetch_req, imem_ready, warp_inst_buf_valid);
                $display("  stalls: mem=%04b fu=%04b sync=%04b async=%04b branch=%04b exit=%04b mbar=%04b wgmma=%04b cluster=%04b",
                         warp_stalled_mem, warp_stalled_fu, warp_stalled_sync, warp_stalled_async,
                         warp_stalled_branch, warp_exit_pending, mbarrier_warp_blocked,
                         warp_stalled_wgmma, cluster_barrier_pending);
                post_kernel_debug_cnt <= post_kernel_debug_cnt + 1;
            end
        end
    end
    `endif

endmodule


//============================================================================
// SIMD FP16 Wrapper - instantiates fp16_unit for each lane
// fp16_unit has 3-cycle latency, simd_fp16 should NOT add more
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
    output wire                     valid_out,
    output wire [NUM_LANES*DATA_WIDTH-1:0] result
);
    // FP16 packed operations (2x FP16 per 32-bit lane)
    // Instantiate fp16_unit for each lane

    wire [DATA_WIDTH-1:0] lane_result [0:NUM_LANES-1];
    wire [NUM_LANES-1:0] lane_valid_out;

    // Instantiate fp16_unit for each lane
    genvar lane;
    generate
        for (lane = 0; lane < NUM_LANES; lane = lane + 1) begin : fp16_lanes
            fp16_unit u_fp16 (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .valid_in   (valid_in && lane_mask[lane]),
                .packed_mode(1'b0),  // Single FP16 mode
                .operand_a  (operand_a[lane*DATA_WIDTH +: DATA_WIDTH]),
                .operand_b  (operand_b[lane*DATA_WIDTH +: DATA_WIDTH]),
                .operand_c  (32'b0),  // No FMA accumulator in simple mul
                .result     (lane_result[lane]),
                .valid_out  (lane_valid_out[lane]),
                .overflow   (),
                .underflow  (),
                .inexact    (),
                .invalid    ()
            );
        end
    endgenerate

    // Use lane 0's valid_out as the overall valid signal
    // (all lanes should produce valid at the same time)
    assign valid_out = lane_valid_out[0];

    // Collect results from all lanes combinatorially
    genvar j;
    generate
        for (j = 0; j < NUM_LANES; j = j + 1) begin : result_collect
            assign result[j*DATA_WIDTH +: DATA_WIDTH] = lane_result[j];
        end
    endgenerate

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
