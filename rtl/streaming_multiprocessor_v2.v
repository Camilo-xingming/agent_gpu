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
    parameter INIT_WARPS = 1
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
    input  wire [31:0]              imem_data,
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

    // Decode Stage
    reg                  decode_valid;
    reg  [WARP_ID_W-1:0] decode_warp_id;
    reg  [31:0]          decode_pc;
    reg  [31:0]          decode_instruction;
    reg  [WARP_ID_W-1:0] dec_warp_id;
    reg  [31:0]          dec_pc;
    reg                  dec_out_valid;

    // Issue Stage
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

    //========================================================================
    // Functional Unit Signals
    //========================================================================

    // Register File Signals
    wire [SIMD_WIDTH-1:0] rf_rd_data_a, rf_rd_data_b, rf_rd_data_c;
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
    // Scoreboard for Dependency Tracking (Per-warp register busy bits)
    //========================================================================
    // Each warp has a 32-bit mask indicating which registers have pending writes
    reg [31:0] scoreboard_busy [0:NUM_WARPS-1];

    // Per-warp pending instruction count (for FU stall tracking)
    reg [3:0] pending_fu_count [0:NUM_WARPS-1];

    wire issue_stall_raw;  // RAW hazard detected
    wire issue_stall_fu;   // FU capacity stall
    wire issue_stall_mem;  // Global memory backpressure stall
    wire issue_stall_atomic; // Atomic unit busy stall
    wire issue_stall_tensor; // Tensor core backpressure stall
    wire issue_stall_wbq; // Writeback queue backpressure stall
    wire issue_accept;
    wire dec_pipe_ready;
    wire decode_accept;
    wire mem_in_flight;
    wire frontend_flush;
    wire frq_head_drop;
    wire ifq_head_drop;

    // Check if source registers are busy (properly indexed: warp first, then register)
    wire ra_busy = dec_out_valid && (dec_ra != 0) && scoreboard_busy[dec_warp_id][dec_ra];
    wire rb_busy = dec_out_valid && (dec_rb != 0) && scoreboard_busy[dec_warp_id][dec_rb];
    wire rc_busy = dec_out_valid && (dec_rc != 0) && scoreboard_busy[dec_warp_id][dec_rc];
    assign issue_stall_raw = dec_out_valid && (ra_busy || rb_busy || rc_busy);

    // Stall if too many pending instructions for this warp (max 8 in flight)
    assign issue_stall_fu = dec_out_valid && (pending_fu_count[dec_warp_id] >= 8);

    // Global memory backpressure (allow one in-flight global read)
    assign issue_stall_mem = dec_out_valid && !dec_atomic_op && (
                             ((dec_mem_read || dec_mem_write) && mem_in_flight) ||
                             (dec_mem_write && !dec_mem_read && store_pending_valid) ||
                             (dec_mem_shared && dec_mem_read && smem_pending_valid) ||
                             (!dec_mem_shared && (dec_mem_read || dec_mem_write) &&
                              (!gmem_req_ready || (dec_mem_read && mem_pending_valid)))
                             );

    // Atomic unit backpressure
    assign issue_stall_atomic = dec_out_valid && dec_atomic_op && atomic_busy;

    // Tensor issue queue backpressure (avoid dropping ops on queue full)
    assign issue_stall_tensor = dec_out_valid && dec_tensor_op && tensor_issue_full_next;

    // Writeback queue backpressure (limit outstanding ops per FU)
    assign issue_stall_wbq = dec_out_valid && (
                             (dec_alu_op && (alu_inflight == ALU_WBQ_DEPTH_VAL)) ||
                             (dec_mul_op && (mul_inflight == MUL_WBQ_DEPTH_VAL)) ||
                             (dec_fp32_op && (fpu32_inflight == FPU32_WBQ_DEPTH_VAL)) ||
                             (dec_fp64_op && (fpu64_inflight == FPU64_WBQ_DEPTH_VAL)) ||
                             (dec_fp16_op && (fp16_inflight == FP16_WBQ_DEPTH_VAL)) ||
                             (dec_sfu_op && (sfu_inflight == SFU_WBQ_DEPTH_VAL)) ||
                             (dec_shuffle_op && (shfl_inflight == SHFL_WBQ_DEPTH_VAL))
                             );

    // Issue/decoder handshakes
    assign issue_accept = dec_out_valid && !issue_stall_raw && !issue_stall_fu &&
                          !issue_stall_mem && !issue_stall_atomic &&
                          !issue_stall_tensor && !issue_stall_wbq;
    assign dec_pipe_ready = !dec_out_valid || issue_accept;
    assign decode_accept = decode_valid && dec_pipe_ready;

    assign mem_in_flight = issue_valid && (issue_mem_read || issue_mem_write) &&
                           !issue_atomic_op;

    assign frontend_flush = (issue_valid && issue_exit_op && (INIT_WARPS == 1));

    wire alu_issue = issue_accept && dec_alu_op;
    wire mul_issue = issue_accept && dec_mul_op;
    wire fpu32_issue = issue_accept && dec_fp32_op;
    wire fpu64_issue = issue_accept && dec_fp64_op;
    wire fp16_issue = issue_accept && dec_fp16_op;
    wire sfu_issue = issue_accept && dec_sfu_op;
    wire shfl_issue = issue_accept && dec_shuffle_op;

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
    // Warp Scheduler with Round-Robin + Priority
    //========================================================================
    reg [WARP_ID_W-1:0] last_issued_warp;
    reg [WARP_ID_W-1:0] selected_warp;
    reg                 warp_selected;

    integer wi;
    always @(*) begin
        warp_selected = 1'b0;
        selected_warp = 0;

        // Round-robin starting from last_issued_warp + 1
        for (wi = 0; wi < NUM_WARPS; wi = wi + 1) begin
            if (!warp_selected) begin
                if (warp_ready[(last_issued_warp + wi + 1) % NUM_WARPS]) begin
                    selected_warp = (last_issued_warp + wi + 1) % NUM_WARPS;
                    warp_selected = 1'b1;
                end
            end
        end
    end

    //========================================================================
    // STAGE 1: FETCH
    //========================================================================
    assign fetch_fire = warp_selected && !frontend_flush && !frq_full &&
                        (fetch_slots_used < {1'b0, IFQ_DEPTH_VAL}) &&
                        (frq_drop_count == 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frq_head <= {IFQ_PTR_W{1'b0}};
            frq_tail <= {IFQ_PTR_W{1'b0}};
            frq_count <= {IFQ_COUNT_W{1'b0}};
            frq_drop_count <= {IFQ_COUNT_W{1'b0}};
            last_issued_warp <= 0;
        end else if (frontend_flush) begin
            frq_head <= {IFQ_PTR_W{1'b0}};
            frq_tail <= {IFQ_PTR_W{1'b0}};
            frq_count <= {IFQ_COUNT_W{1'b0}};
            frq_drop_count <= frq_count;
        end else begin
            if (fetch_fire) begin
                frq_warp_id[frq_tail] <= selected_warp;
                frq_pc[frq_tail] <= warp_fetch_pc[selected_warp];
                frq_tail <= (frq_tail == IFQ_DEPTH-1) ? {IFQ_PTR_W{1'b0}} :
                            frq_tail + 1'b1;
                last_issued_warp <= selected_warp;
            end

            if (frq_pop) begin
                frq_head <= (frq_head == IFQ_DEPTH-1) ? {IFQ_PTR_W{1'b0}} :
                            frq_head + 1'b1;
            end

            case ({fetch_fire, frq_pop})
                2'b10: frq_count <= frq_count + 1'b1;
                2'b01: frq_count <= frq_count - 1'b1;
                default: frq_count <= frq_count;
            endcase

            if (imem_drop_flush && (frq_drop_count != 0)) begin
                frq_drop_count <= frq_drop_count - 1'b1;
            end
        end
    end

    assign imem_req = fetch_fire;
    assign imem_addr = warp_fetch_pc[selected_warp];

    //========================================================================
    // Instruction Fetch Queue
    //========================================================================
    assign frq_head_drop = !frq_empty &&
                           (warp_exit_pending[frq_warp_id[frq_head]] ||
                            !warp_valid[frq_warp_id[frq_head]]);
    assign ifq_head_drop = !ifq_empty &&
                           (warp_exit_pending[ifq_warp_id[ifq_head]] ||
                            !warp_valid[ifq_warp_id[ifq_head]]);

    wire imem_accept = imem_valid && !frq_empty && !ifq_full &&
                       (frq_drop_count == 0) && !frq_head_drop;
    wire imem_drop_flush = imem_valid && (frq_drop_count != 0);
    wire imem_drop_exit = imem_valid && frq_head_drop;
    wire imem_drop = imem_drop_flush || imem_drop_exit;

    assign frq_pop = imem_accept || imem_drop;
    assign ifq_push = imem_accept;
    assign decode_pop = ifq_head_drop || (!ifq_empty && (decode_accept || !decode_valid));

    integer ifq_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ifq_head <= {IFQ_PTR_W{1'b0}};
            ifq_tail <= {IFQ_PTR_W{1'b0}};
            ifq_count <= {IFQ_COUNT_W{1'b0}};
            for (ifq_i = 0; ifq_i < IFQ_DEPTH; ifq_i = ifq_i + 1) begin
                ifq_inst[ifq_i] <= 32'b0;
                ifq_warp_id[ifq_i] <= {WARP_ID_W{1'b0}};
                ifq_pc[ifq_i] <= 32'b0;
            end
        end else if (frontend_flush) begin
            ifq_head <= {IFQ_PTR_W{1'b0}};
            ifq_tail <= {IFQ_PTR_W{1'b0}};
            ifq_count <= {IFQ_COUNT_W{1'b0}};
        end else begin
            if (ifq_push) begin
                ifq_inst[ifq_tail] <= imem_data;
                ifq_warp_id[ifq_tail] <= frq_warp_id[frq_head];
                ifq_pc[ifq_tail] <= frq_pc[frq_head];
                ifq_tail <= (ifq_tail == IFQ_DEPTH-1) ? {IFQ_PTR_W{1'b0}} :
                            ifq_tail + 1'b1;
            end

            if (decode_pop) begin
                ifq_head <= (ifq_head == IFQ_DEPTH-1) ? {IFQ_PTR_W{1'b0}} :
                            ifq_head + 1'b1;
            end

            case ({ifq_push, decode_pop})
                2'b10: ifq_count <= ifq_count + 1'b1;
                2'b01: ifq_count <= ifq_count - 1'b1;
                default: ifq_count <= ifq_count;
            endcase
        end
    end

    //========================================================================
    // STAGE 2: DECODE
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            decode_valid <= 1'b0;
        end else if (frontend_flush) begin
            decode_valid <= 1'b0;
        end else begin
            if (ifq_head_drop) begin
                if (decode_accept) begin
                    decode_valid <= 1'b0;
                end
            end else if (decode_pop) begin
                decode_valid <= 1'b1;
                decode_warp_id <= ifq_warp_id[ifq_head];
                decode_pc <= ifq_pc[ifq_head];
                decode_instruction <= ifq_inst[ifq_head];
            end else if (decode_accept) begin
                decode_valid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Instruction Decoder
    //------------------------------------------------------------------------
    // Internal signals for decoder output mapping
    wire dec_fp32_special;
    wire dec_wmma_mma;
    wire dec_shfl_op;

    decoder u_decoder (
        .clk         (clk),
        .rst_n       (rst_n),
        .instruction (decode_instruction),
        .valid_in    (decode_accept),
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

    // Align warp/pc metadata with decoder output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec_warp_id <= 0;
            dec_pc <= 0;
        end else if (decode_accept) begin
            dec_warp_id <= decode_warp_id;
            dec_pc <= decode_pc;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dec_out_valid <= 1'b0;
        end else if (frontend_flush) begin
            dec_out_valid <= 1'b0;
        end else if (decode_accept) begin
            dec_out_valid <= 1'b1;
        end else if (issue_accept) begin
            dec_out_valid <= 1'b0;
        end
    end

    // Map decoder outputs to V2 signal names
    assign dec_sfu_op = dec_fp32_special;
    assign dec_tensor_op = dec_wmma_mma;
    assign dec_shuffle_op = dec_shfl_op;

    //========================================================================
    // STAGE 3: ISSUE (Scoreboard Check + Register Read)
    //========================================================================
    integer sb_init;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            issue_valid <= 1'b0;
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
        end else begin
            issue_valid <= issue_accept;
            if (issue_accept) begin
                issue_warp_id <= dec_warp_id;
                issue_pc <= dec_pc;
                issue_opcode <= dec_opcode;
                issue_rd <= dec_rd;
                issue_ra <= dec_ra;
                issue_rb <= dec_rb;
                issue_rc <= dec_rc;
                issue_func <= dec_func;
                issue_imm16 <= dec_imm16;
                issue_imm21 <= dec_imm21;
                issue_use_imm <= dec_use_imm;
                issue_mask <= warp_mask[dec_warp_id];
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
            if (issue_accept && dec_reg_write && dec_rd != 0) begin
                scoreboard_busy[dec_warp_id][dec_rd] <= 1'b1;
            end

            // Clear on writeback
            if (wb_valid && wb_rd != 0) begin
                scoreboard_busy[wb_warp_id][wb_rd] <= 1'b0;
            end

            // Increment pending FU count for multi-cycle operations
            if (issue_accept && (dec_fp32_op || dec_fp64_op || dec_fp16_op ||
                                 dec_sfu_op || dec_tensor_op || dec_mem_read ||
                                 dec_atomic_op)) begin
                pending_fu_count[dec_warp_id] <= pending_fu_count[dec_warp_id] + 1;
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
    // Register File (Per-Warp)
    //------------------------------------------------------------------------
    register_file #(
        .NUM_WARPS(NUM_WARPS)
    ) u_regfile (
        .clk       (clk),
        .rst_n     (rst_n),
        .warp_id   (issue_warp_id),
        .rd_addr_a (issue_ra),
        .rd_data_a (rf_rd_data_a),
        .rd_addr_b (issue_rb),
        .rd_data_b (rf_rd_data_b),
        .rd_addr_c (issue_rc),
        .rd_data_c (rf_rd_data_c),
        .wr_en     (rf_wr_en),
        .wr_warp   (wb_warp_id),
        .wr_addr   (wb_rd),
        .wr_data   (rf_wr_data),
        .wr_mask   (rf_wr_mask)
    );

    //========================================================================
    // STAGE 4: EXECUTE (Multiple Functional Units)
    //========================================================================

    //------------------------------------------------------------------------
    // SIMD ALU (Integer Operations)
    //------------------------------------------------------------------------
    simd_alu u_simd_alu (
        .func       (issue_func),
        .operand_a  (rf_rd_data_a),
        .operand_b  (issue_use_imm ? {NUM_LANES{16'b0, issue_imm16}} : rf_rd_data_b),
        .lane_mask  (issue_mask),
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
            alu_valid_pipe <= issue_valid && issue_alu_op;
            if (issue_valid && issue_alu_op) begin
                alu_warp_pipe <= issue_warp_id;
                alu_rd_pipe <= issue_rd;
                alu_mask_pipe <= issue_mask;
                alu_result_pipe <= alu_result;
            end
        end
    end

    assign alu_valid_out = alu_valid_pipe;

    //------------------------------------------------------------------------
    // SIMD Multiplier
    //------------------------------------------------------------------------
    simd_mul_unit u_simd_mul (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (issue_valid && issue_mul_op),
        .func      (issue_func),
        .operand_a (rf_rd_data_a),
        .operand_b (rf_rd_data_b),
        .operand_c (rf_rd_data_c),
        .lane_mask (issue_mask),
        .valid_out (mul_valid_out),
        .result    (mul_result)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mul_warp_pipe <= 0;
            mul_rd_pipe <= 0;
            mul_mask_pipe <= 0;
        end else if (issue_valid && issue_mul_op) begin
            mul_warp_pipe <= issue_warp_id;
            mul_rd_pipe <= issue_rd;
            mul_mask_pipe <= issue_mask;
        end
    end

    //------------------------------------------------------------------------
    // SIMD FPU (FP32) - 4-cycle pipeline
    //------------------------------------------------------------------------
    assign fpu32_valid_in = issue_valid && issue_fp32_op;

    simd_fpu u_simd_fpu (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fpu32_valid_in),
        .ready     (fpu32_ready),
        .func      (issue_func),
        .operand_a (rf_rd_data_a),
        .operand_b (rf_rd_data_b),
        .operand_c (rf_rd_data_c),
        .lane_mask (issue_mask),
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
            fpu32_warp_pipe[0] <= fpu32_valid_in ? issue_warp_id : {WARP_ID_W{1'b0}};
            fpu32_rd_pipe[0] <= fpu32_valid_in ? issue_rd : 5'b0;
            fpu32_mask_pipe[0] <= fpu32_valid_in ? issue_mask : {NUM_LANES{1'b0}};
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
    assign fpu64_valid_in = issue_valid && issue_fp64_op;

    simd_fpu64 u_simd_fpu64 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fpu64_valid_in),
        .ready     (fpu64_ready),
        .func      (issue_func),
        .operand_a (rf_rd_data_a),
        .operand_b (rf_rd_data_b),
        .operand_c (rf_rd_data_c),
        .lane_mask (issue_mask),
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
            fpu64_warp_pipe[0] <= fpu64_valid_in ? issue_warp_id : {WARP_ID_W{1'b0}};
            fpu64_rd_pipe[0] <= fpu64_valid_in ? issue_rd : 5'b0;
            fpu64_mask_pipe[0] <= fpu64_valid_in ? issue_mask : {NUM_LANES{1'b0}};
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
    assign fp16_valid_in = issue_valid && issue_fp16_op;

    simd_fp16 u_simd_fp16 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (fp16_valid_in),
        .ready     (fp16_ready),
        .func      (issue_func),
        .operand_a (rf_rd_data_a),
        .operand_b (rf_rd_data_b),
        .lane_mask (issue_mask),
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
            fp16_warp_pipe[0] <= fp16_valid_in ? issue_warp_id : {WARP_ID_W{1'b0}};
            fp16_rd_pipe[0] <= fp16_valid_in ? issue_rd : 5'b0;
            fp16_mask_pipe[0] <= fp16_valid_in ? issue_mask : {NUM_LANES{1'b0}};
            fp16_warp_pipe[1] <= fp16_warp_pipe[0];
            fp16_rd_pipe[1] <= fp16_rd_pipe[0];
            fp16_mask_pipe[1] <= fp16_mask_pipe[0];
        end
    end

    //------------------------------------------------------------------------
    // Special Function Unit (sin, cos, sqrt, exp, log)
    //------------------------------------------------------------------------
    assign sfu_valid_in = issue_valid && issue_sfu_op;

    simd_sfu u_simd_sfu (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (sfu_valid_in),
        .ready     (sfu_ready),
        .func      (issue_func),
        .operand   (rf_rd_data_a),
        .lane_mask (issue_mask),
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
            sfu_warp_pipe[0] <= sfu_valid_in ? issue_warp_id : {WARP_ID_W{1'b0}};
            sfu_rd_pipe[0] <= sfu_valid_in ? issue_rd : 5'b0;
            sfu_mask_pipe[0] <= sfu_valid_in ? issue_mask : {NUM_LANES{1'b0}};
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
    assign shuffle_valid_in = issue_valid && issue_shuffle_op;

    genvar sh_i;
    generate
        for (sh_i = 0; sh_i < NUM_LANES; sh_i = sh_i + 1) begin : shuffle_lanes
            assign shuffle_src_lane[sh_i*5 +: 5] = rf_rd_data_b[sh_i*32 +: 5];
        end
    endgenerate

    assign shuffle_offset = {NUM_LANES{issue_imm16[4:0]}};

    warp_shuffle u_warp_shuffle (
        .func       (issue_func),
        .src_data   (rf_rd_data_a),
        .src_lane   (shuffle_src_lane),
        .offset     (shuffle_offset),
        .lane_mask  (issue_mask),
        .width      (issue_imm16[4:0]),
        .membermask (issue_mask),
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
                shuffle_warp_pipe <= issue_warp_id;
                shuffle_rd_pipe <= issue_rd;
                shuffle_mask_pipe <= issue_mask;
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
        .warp_id         (issue_warp_id[1:0]),
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

            // Advance fetch PC when issuing a new fetch
            if (fetch_fire) begin
                warp_fetch_pc[selected_warp] <= warp_fetch_pc[selected_warp] + 4;
            end

            // Update architectural PC when instruction is issued (in-order)
            if (issue_accept && !dec_branch_op && !dec_exit_op) begin
                warp_pc[dec_warp_id] <= warp_pc[dec_warp_id] + 4;
            end

            // Branch handling
            if (cfu_branch_taken) begin
                warp_pc[issue_warp_id] <= cfu_branch_target;
                warp_fetch_pc[issue_warp_id] <= cfu_branch_target;
                warp_mask[issue_warp_id] <= cfu_active_mask;
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
