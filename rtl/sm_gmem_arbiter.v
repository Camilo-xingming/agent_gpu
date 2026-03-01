//============================================================================
// sm_gmem_arbiter.v — Global Memory Request Arbiter
//
// Priority mux for 4 request sources → single memory_interface port:
//   Normal (pipeline LD/ST) > Atomic > ACE (async copy) > Texture
//
// Includes per-source request tracking FSMs and response routing.
// Part of RALPH-6 God Module Refactor.
//============================================================================
`include "gpu_defines.vh"

module sm_gmem_arbiter #(
    parameter NUM_LANES  = `THREADS_PER_WARP,
    parameter SIMD_WIDTH = NUM_LANES * 32
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // --- Normal pipeline LD/ST ---
    input  wire                     normal_req_valid,   // issue_valid && !shared && !atomic && (read||write)
    input  wire                     normal_req_write,   // issue_mem_write
    input  wire [NUM_LANES*32-1:0]  normal_req_addr,    // rf_rd_data_a
    input  wire [SIMD_WIDTH-1:0]    normal_req_wdata,   // rf_rd_data_b
    input  wire [NUM_LANES-1:0]     normal_req_mask,    // issue_mask

    // --- Atomic unit ---
    input  wire                     atomic_req,         // atomic_mem_req
    input  wire                     atomic_write,       // atomic_mem_write
    input  wire [5:0]               atomic_lane,        // atomic_mem_lane
    input  wire [31:0]              atomic_addr,        // atomic_mem_addr
    input  wire [31:0]              atomic_wdata,       // atomic_mem_wdata
    input  wire                     atomic_shared_pending, // atomic_mem_shared_pending
    // Shared-memory atomic response (bypass path)
    input  wire                     smem_atomic_resp_valid,
    input  wire [31:0]              smem_atomic_resp_rdata,

    output wire                     atomic_ready,       // atomic_mem_ready
    output wire                     atomic_resp_read_valid,
    output wire                     atomic_resp_write_valid,
    output wire [SIMD_WIDTH-1:0]    atomic_rdata,       // atomic_mem_rdata
    output wire                     atomic_pending,     // atomic_mem_pending

    // --- ACE (Async Copy Engine) ---
    input  wire                     ace_req_valid,      // ace_gmem_req_valid
    input  wire [31:0]              ace_req_addr,       // ace_gmem_req_addr
    input  wire [4:0]               ace_req_size,       // ace_gmem_req_size

    output wire                     ace_resp_valid,     // ace_gmem_resp_valid
    output wire [127:0]             ace_resp_data,      // ace_gmem_resp_data
    output wire                     ace_pending,        // ace_mem_pending (for external tracking)

    // --- Texture unit ---
    input  wire                     tex_req,            // tex_mem_req
    input  wire [31:0]              tex_addr,           // tex_mem_addr
    input  wire                     tex_write,          // tex_mem_write
    input  wire [127:0]             tex_wdata,          // tex_mem_wdata

    output wire                     tex_ready,          // tex_mem_ready
    output wire                     tex_resp_valid,     // tex_mem_valid
    output wire [127:0]             tex_resp_data,      // tex_mem_rdata
    output wire                     tex_pending,        // tex_mem_pending

    // --- Merged output to memory_interface ---
    output wire                     gmem_req_valid,
    output wire                     gmem_req_write,
    output wire [NUM_LANES*32-1:0]  gmem_req_addr,
    output wire [SIMD_WIDTH-1:0]    gmem_req_wdata,
    output wire [NUM_LANES-1:0]     gmem_req_mask,

    // --- From memory_interface ---
    input  wire                     gmem_req_ready,
    input  wire                     gmem_resp_valid,
    input  wire [SIMD_WIDTH-1:0]    gmem_resp_rdata,

    // --- Routed normal-path response (for SM LD/ST path) ---
    output wire                     normal_resp_valid,
    output wire [SIMD_WIDTH-1:0]    normal_resp_rdata
);

    //------------------------------------------------------------------------
    // Priority arbitration: Normal > Atomic > ACE > Texture
    //------------------------------------------------------------------------
    wire use_atomic = atomic_req && !normal_req_valid &&
                      !atomic_pending_r && !atomic_shared_pending;

    wire use_ace = ace_req_valid && !normal_req_valid && !use_atomic && !ace_pending_r;

    wire use_tex = tex_req && !normal_req_valid && !use_atomic && !use_ace && !tex_pending_r;

    assign gmem_req_valid = normal_req_valid || use_atomic || use_ace || use_tex;

    assign gmem_req_write = use_atomic ? atomic_write :
                            use_ace    ? 1'b0 :
                            use_tex    ? tex_write :
                            normal_req_write;

    //------------------------------------------------------------------------
    // Atomic: expand single-lane to per-lane address/data vectors
    //------------------------------------------------------------------------
    reg  [NUM_LANES-1:0]     atomic_req_mask_r;
    reg  [NUM_LANES*32-1:0]  atomic_req_addr_vec;
    reg  [SIMD_WIDTH-1:0]    atomic_req_wdata_vec;
    integer lane_i;

    always @(*) begin
        atomic_req_mask_r    = {NUM_LANES{1'b0}};
        atomic_req_addr_vec  = {NUM_LANES{32'b0}};
        atomic_req_wdata_vec = {SIMD_WIDTH{1'b0}};
        if (atomic_req) begin
            atomic_req_mask_r[atomic_lane[4:0]] = 1'b1;
            for (lane_i = 0; lane_i < NUM_LANES; lane_i = lane_i + 1) begin
                if (atomic_lane == lane_i[5:0]) begin
                    atomic_req_addr_vec[lane_i*32 +: 32]  = atomic_addr;
                    atomic_req_wdata_vec[lane_i*32 +: 32] = atomic_wdata;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Address / data / mask mux
    //------------------------------------------------------------------------
    wire [NUM_LANES*32-1:0] ace_replicated_addr = {NUM_LANES{ace_req_addr}};
    wire [NUM_LANES*32-1:0] tex_replicated_addr = {NUM_LANES{tex_addr}};
    wire [SIMD_WIDTH-1:0]   tex_wdata_extended  = {{(SIMD_WIDTH-128){1'b0}}, tex_wdata};

    assign gmem_req_addr  = use_atomic ? atomic_req_addr_vec :
                            use_ace    ? ace_replicated_addr :
                            use_tex    ? tex_replicated_addr :
                            normal_req_addr;

    assign gmem_req_wdata = use_atomic ? atomic_req_wdata_vec :
                            use_tex    ? tex_wdata_extended :
                            normal_req_wdata;

    assign gmem_req_mask  = use_atomic ? atomic_req_mask_r : normal_req_mask;

    //------------------------------------------------------------------------
    // Atomic request tracking FSM
    //------------------------------------------------------------------------
    reg              atomic_pending_r;
    reg              atomic_ready_gmem;
    reg              atomic_resp_read_valid_gmem;
    reg              atomic_resp_write_valid_gmem;
    reg              atomic_pending_is_write;
    reg [SIMD_WIDTH-1:0] atomic_rdata_gmem;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            atomic_pending_r  <= 1'b0;
            atomic_ready_gmem <= 1'b0;
            atomic_resp_read_valid_gmem  <= 1'b0;
            atomic_resp_write_valid_gmem <= 1'b0;
            atomic_pending_is_write <= 1'b0;
            atomic_rdata_gmem <= {SIMD_WIDTH{1'b0}};
        end else begin
            atomic_ready_gmem <= 1'b0;
            atomic_resp_read_valid_gmem  <= 1'b0;
            atomic_resp_write_valid_gmem <= 1'b0;
            if (atomic_shared_pending) begin
                atomic_pending_r <= 1'b0;
            end else if (use_atomic && gmem_req_ready) begin
                atomic_pending_r <= 1'b1;
                atomic_pending_is_write <= atomic_write;
            end else if (atomic_pending_r && gmem_resp_valid) begin
                atomic_pending_r  <= 1'b0;
                atomic_ready_gmem <= 1'b1;
                atomic_resp_read_valid_gmem  <= !atomic_pending_is_write;
                atomic_resp_write_valid_gmem <= atomic_pending_is_write;
                atomic_rdata_gmem <= gmem_resp_rdata;
            end
        end
    end

    assign atomic_ready   = atomic_shared_pending ? smem_atomic_resp_valid : atomic_ready_gmem;
    assign atomic_resp_read_valid =
        atomic_shared_pending ? (smem_atomic_resp_valid && !atomic_write) : atomic_resp_read_valid_gmem;
    assign atomic_resp_write_valid =
        atomic_shared_pending ? (smem_atomic_resp_valid && atomic_write) : atomic_resp_write_valid_gmem;
    assign atomic_rdata   = atomic_shared_pending ? {NUM_LANES{smem_atomic_resp_rdata}} : atomic_rdata_gmem;
    assign atomic_pending = atomic_pending_r;

    //------------------------------------------------------------------------
    // ACE request tracking FSM
    //------------------------------------------------------------------------
    reg        ace_pending_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ace_pending_r <= 1'b0;
        end else begin
            if (use_ace && gmem_req_ready && !atomic_pending_r) begin
                ace_pending_r <= 1'b1;
            end else if (ace_pending_r && gmem_resp_valid && !tex_pending_r && !atomic_pending_r) begin
                ace_pending_r <= 1'b0;
            end
        end
    end

    assign ace_resp_valid = ace_pending_r && gmem_resp_valid && !tex_pending_r && !atomic_pending_r;
    assign ace_resp_data  = gmem_resp_rdata[127:0];
    assign ace_pending    = ace_pending_r;

    //------------------------------------------------------------------------
    // Texture request tracking FSM
    //------------------------------------------------------------------------
    reg        tex_pending_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tex_pending_r <= 1'b0;
        end else begin
            if (use_tex && gmem_req_ready) begin
                tex_pending_r <= 1'b1;
            end else if (tex_pending_r && gmem_resp_valid && !ace_pending_r && !atomic_pending_r) begin
                tex_pending_r <= 1'b0;
            end
        end
    end

    assign tex_ready     = !tex_pending_r && gmem_req_ready;
    assign tex_resp_valid = tex_pending_r && gmem_resp_valid && !ace_pending_r && !atomic_pending_r;
    assign tex_resp_data = gmem_resp_rdata[127:0];
    assign tex_pending   = tex_pending_r;

    // Response belongs to normal path when no side-unit request is pending.
    assign normal_resp_valid = gmem_resp_valid && !atomic_pending_r && !ace_pending_r && !tex_pending_r;
    assign normal_resp_rdata = gmem_resp_rdata;

endmodule
