//============================================================================
// RalphGPU - Command Processor
// Issue #151: Kernel launch interface with command queue
//
// Two modes:
//   1. Legacy mode (CP_ENABLE=0): CSR writes directly launch kernel
//   2. Queue mode  (CP_ENABLE=1): Host pushes descriptors to queue
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module command_processor #(
    parameter NUM_SM         = `NUM_SM,
    parameter QUEUE_DEPTH    = 8,
    parameter DESC_WORDS     = 16
)(
    input  wire        clk,
    input  wire        rst_n,

    // CSR Interface
    input  wire        csr_wr_en,
    input  wire [11:0] csr_addr,
    input  wire [31:0] csr_wr_data,
    output reg  [31:0] csr_rd_data,
    output wire        csr_rd_valid,

    // Legacy kernel launch inputs
    input  wire        legacy_kernel_start,
    input  wire [31:0] legacy_kernel_pc,
    input  wire [31:0] legacy_grid_dim_x,
    input  wire [31:0] legacy_grid_dim_y,
    input  wire [31:0] legacy_grid_dim_z,
    input  wire [31:0] legacy_block_dim_x,
    input  wire [31:0] legacy_block_dim_y,
    input  wire [31:0] legacy_block_dim_z,

    // SM Control Interface
    output reg  [NUM_SM-1:0] sm_kernel_start,
    output reg  [31:0]       sm_kernel_pc,
    output reg  [31:0]       sm_block_id_x  [0:NUM_SM-1],
    output reg  [31:0]       sm_block_id_y  [0:NUM_SM-1],
    output reg  [31:0]       sm_block_id_z  [0:NUM_SM-1],
    output reg  [31:0]       sm_block_dim_x,
    output reg  [31:0]       sm_block_dim_y,
    output reg  [31:0]       sm_block_dim_z,
    output reg  [31:0]       sm_grid_dim_x,
    output reg  [31:0]       sm_grid_dim_y,
    output reg  [31:0]       sm_grid_dim_z,

    input  wire [NUM_SM-1:0] sm_done,

    // Status
    output wire        gpu_busy,
    output wire        irq_kernel_done,
    output wire [31:0] fence_value,

    // AXI Master for Command Queue (Phase 2 stubs)
    output wire        m_axi_arvalid,
    output wire [31:0] m_axi_araddr,
    input  wire        m_axi_arready,
    input  wire [31:0] m_axi_rdata,
    input  wire [1:0]  m_axi_rresp,
    input  wire        m_axi_rvalid,
    output wire        m_axi_rready
);

    localparam Q_PTR_W = $clog2(QUEUE_DEPTH);

    // CSR address map
    localparam CSR_CMD_QUEUE_BASE_LO = 12'h030;
    localparam CSR_CMD_QUEUE_BASE_HI = 12'h034;
    localparam CSR_CMD_QUEUE_SIZE    = 12'h038;
    localparam CSR_CMD_QUEUE_HEAD    = 12'h03C;
    localparam CSR_CMD_QUEUE_TAIL    = 12'h040;
    localparam CSR_CMD_FENCE_VALUE   = 12'h044;
    localparam CSR_CMD_FENCE_SIGNAL  = 12'h048;
    localparam CSR_CP_STATUS         = 12'h04C;
    localparam CSR_DESC_BASE         = 12'h050;

    // Descriptor word definitions
    localparam DESC_IDX_KERNEL_PC          = 0;
    localparam DESC_IDX_GRID_DIM_X         = 1;
    localparam DESC_IDX_GRID_DIM_Y         = 2;
    localparam DESC_IDX_GRID_DIM_Z         = 3;
    localparam DESC_IDX_BLOCK_DIM_X        = 4;
    localparam DESC_IDX_BLOCK_DIM_Y        = 5;
    localparam DESC_IDX_BLOCK_DIM_Z        = 6;
    localparam DESC_IDX_SHARED_MEM_SIZE    = 7;
    localparam DESC_IDX_KERNEL_PARAMS_BASE = 8;
    localparam DESC_IDX_FENCE_ID           = 10;
    localparam DESC_IDX_FLAGS              = 11;

    // FSM states
    localparam CP_IDLE      = 3'd0;
    localparam CP_LOAD_DESC = 3'd1;
    localparam CP_DISPATCH  = 3'd2;
    localparam CP_WAIT      = 3'd3;
    localparam CP_COMPLETE  = 3'd4;

    //========================================================================
    // Registers
    //========================================================================
    reg        cp_enable;

    // Descriptor staging (CSR-written)
    reg [31:0] desc_staging [0:DESC_WORDS-1];

    // Queue push interface (from CSR doorbell write)
    reg        queue_push_valid;
    wire       queue_push_ready;
    reg [31:0] queue_push_kernel_pc;
    reg [31:0] queue_push_grid_dim_x;
    reg [31:0] queue_push_grid_dim_y;
    reg [31:0] queue_push_grid_dim_z;
    reg [31:0] queue_push_block_dim_x;
    reg [31:0] queue_push_block_dim_y;
    reg [31:0] queue_push_block_dim_z;
    reg [31:0] queue_push_shared_mem_size;
    reg [31:0] queue_push_kernel_params_base;
    reg [31:0] queue_push_fence_id;
    reg [31:0] queue_push_flags;

    // Queue pop interface (to scheduler FSM)
    reg        queue_pop_ready;
    wire       queue_pop_valid;
    wire [31:0] queue_pop_kernel_pc;
    wire [31:0] queue_pop_grid_dim_x;
    wire [31:0] queue_pop_grid_dim_y;
    wire [31:0] queue_pop_grid_dim_z;
    wire [31:0] queue_pop_block_dim_x;
    wire [31:0] queue_pop_block_dim_y;
    wire [31:0] queue_pop_block_dim_z;
    wire [31:0] queue_pop_shared_mem_size;
    wire [31:0] queue_pop_kernel_params_base;
    wire [31:0] queue_pop_fence_id;
    wire [31:0] queue_pop_flags;
    wire [Q_PTR_W-1:0] queue_head;
    wire [Q_PTR_W-1:0] queue_tail;
    wire [Q_PTR_W:0]   queue_count;

    // Active kernel state
    reg [31:0] active_kernel_pc;
    reg [31:0] active_grid_dim_x;
    reg [31:0] active_grid_dim_y;
    reg [31:0] active_grid_dim_z;
    reg [31:0] active_block_dim_x;
    reg [31:0] active_block_dim_y;
    reg [31:0] active_block_dim_z;
    reg [31:0] active_shared_mem_size;
    reg [31:0] active_kernel_params_base;
    reg [31:0] active_fence_id;

    // CTA dispatch state
    reg [31:0] total_blocks;
    reg [31:0] dispatched_blocks;
    reg [NUM_SM-1:0] sm_busy;

    // Completion tracking
    reg [31:0] fence_value_reg;
    reg [31:0] fence_signal_reg;
    reg        kernel_done_irq;

    // Main FSM
    reg [2:0] cp_state;

    //========================================================================
    // Queue
    //========================================================================
    command_queue #(
        .DEPTH     (QUEUE_DEPTH),
        .PTR_WIDTH (Q_PTR_W)
    ) u_command_queue (
        .clk                     (clk),
        .rst_n                   (rst_n),
        .push_valid              (queue_push_valid),
        .push_ready              (queue_push_ready),
        .push_kernel_pc          (queue_push_kernel_pc),
        .push_grid_dim_x         (queue_push_grid_dim_x),
        .push_grid_dim_y         (queue_push_grid_dim_y),
        .push_grid_dim_z         (queue_push_grid_dim_z),
        .push_block_dim_x        (queue_push_block_dim_x),
        .push_block_dim_y        (queue_push_block_dim_y),
        .push_block_dim_z        (queue_push_block_dim_z),
        .push_shared_mem_size    (queue_push_shared_mem_size),
        .push_kernel_params_base (queue_push_kernel_params_base),
        .push_fence_id           (queue_push_fence_id),
        .push_flags              (queue_push_flags),
        .pop_valid               (queue_pop_valid),
        .pop_ready               (queue_pop_ready),
        .pop_kernel_pc           (queue_pop_kernel_pc),
        .pop_grid_dim_x          (queue_pop_grid_dim_x),
        .pop_grid_dim_y          (queue_pop_grid_dim_y),
        .pop_grid_dim_z          (queue_pop_grid_dim_z),
        .pop_block_dim_x         (queue_pop_block_dim_x),
        .pop_block_dim_y         (queue_pop_block_dim_y),
        .pop_block_dim_z         (queue_pop_block_dim_z),
        .pop_shared_mem_size     (queue_pop_shared_mem_size),
        .pop_kernel_params_base  (queue_pop_kernel_params_base),
        .pop_fence_id            (queue_pop_fence_id),
        .pop_flags               (queue_pop_flags),
        .head                    (queue_head),
        .tail                    (queue_tail),
        .count                   (queue_count)
    );

    //========================================================================
    // CSR Read
    //========================================================================
    wire cp_addr_hit = (csr_addr >= CSR_CMD_QUEUE_BASE_LO && csr_addr <= CSR_CP_STATUS) ||
                       (csr_addr >= CSR_DESC_BASE && csr_addr < (CSR_DESC_BASE + DESC_WORDS * 4));
    assign csr_rd_valid = cp_addr_hit;

    always @(*) begin
        csr_rd_data = 32'b0;
        case (csr_addr)
            CSR_CMD_QUEUE_BASE_LO: csr_rd_data = 32'b0;
            CSR_CMD_QUEUE_BASE_HI: csr_rd_data = 32'b0;
            CSR_CMD_QUEUE_SIZE:    csr_rd_data = QUEUE_DEPTH;
            CSR_CMD_QUEUE_HEAD:    csr_rd_data = {{(32-Q_PTR_W){1'b0}}, queue_head};
            CSR_CMD_QUEUE_TAIL:    csr_rd_data = {{(32-Q_PTR_W){1'b0}}, queue_tail};
            CSR_CMD_FENCE_VALUE:   csr_rd_data = fence_value_reg;
            CSR_CMD_FENCE_SIGNAL:  csr_rd_data = fence_signal_reg;
            CSR_CP_STATUS:         csr_rd_data = {24'b0, cp_enable, cp_state, queue_count[3:0]};
            default: begin
                if (csr_addr >= CSR_DESC_BASE && csr_addr < (CSR_DESC_BASE + DESC_WORDS * 4))
                    csr_rd_data = desc_staging[(csr_addr - CSR_DESC_BASE) >> 2];
            end
        endcase
    end

    //========================================================================
    // CSR Write
    //========================================================================
    integer desc_wr_idx;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cp_enable <= 1'b0;
            fence_signal_reg <= 32'hFFFF_FFFF;
            for (desc_wr_idx = 0; desc_wr_idx < DESC_WORDS; desc_wr_idx = desc_wr_idx + 1)
                desc_staging[desc_wr_idx] <= 32'b0;

            queue_push_valid              <= 1'b0;
            queue_push_kernel_pc          <= 32'b0;
            queue_push_grid_dim_x         <= 32'b0;
            queue_push_grid_dim_y         <= 32'b0;
            queue_push_grid_dim_z         <= 32'b0;
            queue_push_block_dim_x        <= 32'b0;
            queue_push_block_dim_y        <= 32'b0;
            queue_push_block_dim_z        <= 32'b0;
            queue_push_shared_mem_size    <= 32'b0;
            queue_push_kernel_params_base <= 32'b0;
            queue_push_fence_id           <= 32'b0;
            queue_push_flags              <= 32'b0;
        end else begin
            queue_push_valid <= 1'b0;
            if (csr_wr_en) begin
                case (csr_addr)
                    12'h004: cp_enable <= csr_wr_data[1];
                    CSR_CMD_FENCE_SIGNAL: fence_signal_reg <= csr_wr_data;
                    CSR_CMD_QUEUE_TAIL: begin
                        if (cp_enable && queue_push_ready) begin
                            queue_push_kernel_pc          <= desc_staging[DESC_IDX_KERNEL_PC];
                            queue_push_grid_dim_x         <= desc_staging[DESC_IDX_GRID_DIM_X];
                            queue_push_grid_dim_y         <= desc_staging[DESC_IDX_GRID_DIM_Y];
                            queue_push_grid_dim_z         <= desc_staging[DESC_IDX_GRID_DIM_Z];
                            queue_push_block_dim_x        <= desc_staging[DESC_IDX_BLOCK_DIM_X];
                            queue_push_block_dim_y        <= desc_staging[DESC_IDX_BLOCK_DIM_Y];
                            queue_push_block_dim_z        <= desc_staging[DESC_IDX_BLOCK_DIM_Z];
                            queue_push_shared_mem_size    <= desc_staging[DESC_IDX_SHARED_MEM_SIZE];
                            queue_push_kernel_params_base <= desc_staging[DESC_IDX_KERNEL_PARAMS_BASE];
                            queue_push_fence_id           <= desc_staging[DESC_IDX_FENCE_ID];
                            queue_push_flags              <= desc_staging[DESC_IDX_FLAGS];
                            queue_push_valid              <= 1'b1;
                        end
                    end
                    default: begin
                        if (csr_addr >= CSR_DESC_BASE && csr_addr < (CSR_DESC_BASE + DESC_WORDS * 4))
                            desc_staging[(csr_addr - CSR_DESC_BASE) >> 2] <= csr_wr_data;
                    end
                endcase
            end
        end
    end

    //========================================================================
    // Main FSM
    //========================================================================
    integer sm_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cp_state          <= CP_IDLE;
            total_blocks      <= 32'b0;
            dispatched_blocks <= 32'b0;
            sm_busy           <= {NUM_SM{1'b0}};
            sm_kernel_start   <= {NUM_SM{1'b0}};
            sm_kernel_pc      <= 32'b0;
            sm_block_dim_x    <= 32'd32;
            sm_block_dim_y    <= 32'd1;
            sm_block_dim_z    <= 32'd1;
            sm_grid_dim_x     <= 32'd1;
            sm_grid_dim_y     <= 32'd1;
            sm_grid_dim_z     <= 32'd1;
            for (sm_i = 0; sm_i < NUM_SM; sm_i = sm_i + 1) begin
                sm_block_id_x[sm_i] <= 32'b0;
                sm_block_id_y[sm_i] <= 32'b0;
                sm_block_id_z[sm_i] <= 32'b0;
            end
            active_kernel_pc         <= 32'b0;
            active_grid_dim_x        <= 32'd1;
            active_grid_dim_y        <= 32'd1;
            active_grid_dim_z        <= 32'd1;
            active_block_dim_x       <= 32'd32;
            active_block_dim_y       <= 32'd1;
            active_block_dim_z       <= 32'd1;
            active_shared_mem_size   <= 32'b0;
            active_kernel_params_base<= 32'b0;
            active_fence_id          <= 32'b0;
            fence_value_reg          <= 32'b0;
            kernel_done_irq          <= 1'b0;
            queue_pop_ready          <= 1'b0;
        end else begin
            sm_kernel_start <= {NUM_SM{1'b0}};
            queue_pop_ready <= 1'b0;

            if (csr_wr_en && csr_addr == 12'h000)
                kernel_done_irq <= 1'b0;

            for (sm_i = 0; sm_i < NUM_SM; sm_i = sm_i + 1) begin
                if (sm_busy[sm_i] && sm_done[sm_i])
                    sm_busy[sm_i] <= 1'b0;
            end

            case (cp_state)
                CP_IDLE: begin
                    if (cp_enable && queue_pop_valid) begin
                        cp_state <= CP_LOAD_DESC;
                    end else if (!cp_enable && legacy_kernel_start) begin
                        active_kernel_pc   <= legacy_kernel_pc;
                        active_grid_dim_x  <= legacy_grid_dim_x;
                        active_grid_dim_y  <= legacy_grid_dim_y;
                        active_grid_dim_z  <= legacy_grid_dim_z;
                        active_block_dim_x <= legacy_block_dim_x;
                        active_block_dim_y <= legacy_block_dim_y;
                        active_block_dim_z <= legacy_block_dim_z;
                        active_fence_id    <= 32'b0;
                        total_blocks <= legacy_grid_dim_x * legacy_grid_dim_y * legacy_grid_dim_z;
                        dispatched_blocks <= 32'b0;
                        sm_busy <= {NUM_SM{1'b0}};
                        cp_state <= CP_DISPATCH;
                    end
                end

                CP_LOAD_DESC: begin
                    active_kernel_pc          <= queue_pop_kernel_pc;
                    active_grid_dim_x         <= queue_pop_grid_dim_x;
                    active_grid_dim_y         <= queue_pop_grid_dim_y;
                    active_grid_dim_z         <= queue_pop_grid_dim_z;
                    active_block_dim_x        <= queue_pop_block_dim_x;
                    active_block_dim_y        <= queue_pop_block_dim_y;
                    active_block_dim_z        <= queue_pop_block_dim_z;
                    active_shared_mem_size    <= queue_pop_shared_mem_size;
                    active_kernel_params_base <= queue_pop_kernel_params_base;
                    active_fence_id           <= queue_pop_fence_id;

                    total_blocks <= queue_pop_grid_dim_x * queue_pop_grid_dim_y * queue_pop_grid_dim_z;
                    dispatched_blocks <= 32'b0;
                    sm_busy <= {NUM_SM{1'b0}};

                    queue_pop_ready <= 1'b1;
                    cp_state <= CP_DISPATCH;
                end

                CP_DISPATCH: begin
                    sm_kernel_pc   <= active_kernel_pc;
                    sm_block_dim_x <= active_block_dim_x;
                    sm_block_dim_y <= active_block_dim_y;
                    sm_block_dim_z <= active_block_dim_z;
                    sm_grid_dim_x  <= active_grid_dim_x;
                    sm_grid_dim_y  <= active_grid_dim_y;
                    sm_grid_dim_z  <= active_grid_dim_z;

                    begin : dispatch_block
                        integer next_block;
                        next_block = dispatched_blocks;
                        for (sm_i = 0; sm_i < NUM_SM; sm_i = sm_i + 1) begin
                            if (!sm_busy[sm_i] && (next_block < total_blocks)) begin
                                sm_busy[sm_i] <= 1'b1;
                                sm_kernel_start[sm_i] <= 1'b1;
                                sm_block_id_x[sm_i] <= next_block % active_grid_dim_x;
                                sm_block_id_y[sm_i] <= (next_block / active_grid_dim_x) % active_grid_dim_y;
                                sm_block_id_z[sm_i] <= next_block / (active_grid_dim_x * active_grid_dim_y);
                                next_block = next_block + 1;
                            end
                        end
                        dispatched_blocks <= next_block;
                    end

                    cp_state <= CP_WAIT;
                end

                CP_WAIT: begin
                    if (dispatched_blocks < total_blocks) begin
                        cp_state <= CP_DISPATCH;
                    end else if (sm_busy == {NUM_SM{1'b0}}) begin
                        cp_state <= CP_COMPLETE;
                    end
                end

                CP_COMPLETE: begin
                    fence_value_reg <= active_fence_id;
                    if (active_fence_id == fence_signal_reg ||
                        fence_signal_reg == 32'hFFFF_FFFF) begin
                        kernel_done_irq <= 1'b1;
                    end
                    if (cp_enable && queue_pop_valid)
                        cp_state <= CP_LOAD_DESC;
                    else
                        cp_state <= CP_IDLE;
                end

                default: cp_state <= CP_IDLE;
            endcase
        end
    end

    assign gpu_busy = (cp_state != CP_IDLE);
    assign irq_kernel_done = kernel_done_irq;
    assign fence_value = fence_value_reg;
    assign m_axi_arvalid = 1'b0;
    assign m_axi_araddr  = 32'b0;
    assign m_axi_rready  = 1'b0;

endmodule
