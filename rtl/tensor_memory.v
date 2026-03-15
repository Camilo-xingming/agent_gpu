//============================================================================
// RalphGPU - Tensor Memory (TMEM) Module
// Blackwell-style 256KB on-chip Tensor Memory per SM
//
// Features:
// - 256KB storage (512 columns x 128 rows x 32-bit cells)
// - Column-based allocation for tcgen05.alloc/dealloc
// - High-bandwidth read/write ports (16 TB/s read, 8 TB/s write per SM)
// - Row/column address encoding (row[6:0], column[8:0])
// - Integration with Tensor Core for async MMA accumulator storage
//
// Architecture based on NVIDIA Blackwell Tensor Memory:
// - Replaces register-based accumulator storage
// - Operands reside in shared memory and TMEM
// - Supports per-thread tensor operations via tcgen05 instructions
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tensor_memory #(
    parameter NUM_ROWS    = 128,          // 128 rows (lanes)
    parameter NUM_COLS    = 512,          // 512 columns
    parameter DATA_WIDTH  = 32,           // 32-bit cells
    parameter ADDR_WIDTH  = 16            // Address width (9-bit col + 7-bit row)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Column Allocation Interface (tcgen05.alloc/dealloc)
    //------------------------------------------------------------------------
    input  wire                     alloc_valid,      // tcgen05.alloc request
    input  wire [8:0]               alloc_num_cols,   // Number of columns to allocate (1-512)
    output reg                      alloc_ready,      // Allocation complete
    output reg  [8:0]               alloc_col_base,   // Allocated column base address
    output reg                      alloc_fail,       // Allocation failed (no space)

    input  wire                     dealloc_valid,    // tcgen05.dealloc request
    input  wire [8:0]               dealloc_col_base, // Column base to deallocate
    input  wire [8:0]               dealloc_num_cols, // Number of columns to deallocate

    //------------------------------------------------------------------------
    // Load Interface (tcgen05.ld - TMEM to registers)
    // High-bandwidth read: 512-bit (16 words) per cycle per port
    //------------------------------------------------------------------------
    input  wire                     ld_valid,         // Load request valid
    input  wire [6:0]               ld_row,           // Row address (0-127)
    input  wire [8:0]               ld_col_base,      // Column base address
    input  wire [3:0]               ld_num_cols,      // Number of columns to read (1-16)
    output reg                      ld_ready,         // Load data ready
    output reg  [511:0]             ld_data,          // Load data (up to 16 x 32-bit)

    //------------------------------------------------------------------------
    // Store Interface (tcgen05.st - registers to TMEM)
    // High-bandwidth write: 256-bit (8 words) per cycle per port
    //------------------------------------------------------------------------
    input  wire                     st_valid,         // Store request valid
    input  wire [6:0]               st_row,           // Row address (0-127)
    input  wire [8:0]               st_col_base,      // Column base address
    input  wire [3:0]               st_num_cols,      // Number of columns to write (1-8)
    input  wire [255:0]             st_data,          // Store data (up to 8 x 32-bit)
    input  wire [7:0]               st_mask,          // Write mask (1 bit per column)
    output reg                      st_ready,         // Store complete

    //------------------------------------------------------------------------
    // MMA Accumulator Interface (for tcgen05.mma integration)
    // Dedicated high-bandwidth port for tensor core accumulator access
    //------------------------------------------------------------------------
    input  wire                     mma_wr_valid,     // MMA write accumulator
    input  wire [6:0]               mma_row,          // Row address
    input  wire [8:0]               mma_col_base,     // Column base
    input  wire [511:0]             mma_wr_data,      // Accumulator data (16 x 32-bit)
    input  wire [15:0]              mma_wr_mask,      // Write mask (1 bit per column)

    input  wire                     mma_rd_valid,     // MMA read accumulator
    output reg  [511:0]             mma_rd_data,      // Accumulator data output
    output reg                      mma_rd_ready,     // Read data ready

    //------------------------------------------------------------------------
    // Copy Interface (tcgen05.cp - async tensor data transfers)
    //------------------------------------------------------------------------
    input  wire                     cp_valid,         // Copy request valid
    input  wire                     cp_direction,     // 0: SMEM->TMEM, 1: TMEM->SMEM
    input  wire [6:0]               cp_row,           // TMEM row
    input  wire [8:0]               cp_col_base,      // TMEM column base
    input  wire [511:0]             cp_wr_data,       // Data to write (SMEM->TMEM)
    output reg  [511:0]             cp_rd_data,       // Data read (TMEM->SMEM)
    output reg                      cp_ready,         // Copy complete

    //------------------------------------------------------------------------
    // Status and Debug
    //------------------------------------------------------------------------
    output wire [9:0]               cols_allocated,   // Number of columns currently allocated
    output wire [9:0]               cols_free,        // Number of free columns
    output wire                     tmem_full,        // TMEM fully allocated
    output wire                     tmem_empty        // TMEM completely free
);

    //------------------------------------------------------------------------
    // Parameters and Derived Constants
    //------------------------------------------------------------------------
    localparam TOTAL_SIZE_BITS = NUM_ROWS * NUM_COLS * DATA_WIDTH; // 256KB = 2097152 bits
    localparam COL_ADDR_WIDTH = 9;  // log2(512)
    localparam ROW_ADDR_WIDTH = 7;  // log2(128)

    // Allocation granularity (minimum allocation unit)
    localparam ALLOC_GRANULARITY = 8;  // Allocate in units of 8 columns

    //------------------------------------------------------------------------
    // Tensor Memory Storage
    // Organized as 128 rows x 512 columns of 32-bit cells
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] tmem [0:NUM_ROWS-1][0:NUM_COLS-1];

    //------------------------------------------------------------------------
    // Column Allocation Tracking
    // Bitmap: 1 = allocated, 0 = free
    // Allocation is contiguous and starts from column 0
    //------------------------------------------------------------------------
    reg [NUM_COLS-1:0] col_alloc_bitmap;
    reg [9:0] alloc_watermark;  // High-water mark of allocation (next free column)

    // Status signals
    assign cols_allocated = alloc_watermark;
    assign cols_free = NUM_COLS - alloc_watermark;
    assign tmem_full = (alloc_watermark >= NUM_COLS);
    assign tmem_empty = (alloc_watermark == 0);

    always @(posedge clk or negedge rst_n) begin
        integer i;
        integer j;
        reg [9:0] aligned_cols;
        if (!rst_n) begin
            col_alloc_bitmap <= {NUM_COLS{1'b0}};
            alloc_watermark <= 10'd0;
            alloc_ready <= 1'b0;
            alloc_col_base <= 9'd0;
            alloc_fail <= 1'b0;
        end else begin
            alloc_ready <= 1'b0;
            alloc_fail <= 1'b0;

            if (alloc_valid) begin
                // Round up to allocation granularity
                aligned_cols = ((alloc_num_cols + ALLOC_GRANULARITY - 1) / ALLOC_GRANULARITY) * ALLOC_GRANULARITY;

                if (alloc_watermark + aligned_cols <= NUM_COLS) begin
                    // Allocation succeeds
                    alloc_col_base <= alloc_watermark[8:0];
                    alloc_watermark <= alloc_watermark + aligned_cols;

                    // Mark columns as allocated
                    for (i = 0; i < NUM_COLS; i = i + 1) begin
                        if (i >= alloc_watermark && i < alloc_watermark + aligned_cols) begin
                            col_alloc_bitmap[i] <= 1'b1;
                        end
                    end

                    alloc_ready <= 1'b1;
                end else begin
                    // Allocation fails
                    alloc_fail <= 1'b1;
                    alloc_ready <= 1'b1;
                end
            end

            // Deallocation (tcgen05.dealloc)
            // Note: Blackwell requires dealloc before kernel exit
            if (dealloc_valid) begin
                // Simple deallocation: just clear the bitmap
                // In practice, should validate ownership
                for (j = 0; j < NUM_COLS; j = j + 1) begin
                    if (j >= dealloc_col_base && j < dealloc_col_base + dealloc_num_cols) begin
                        col_alloc_bitmap[j] <= 1'b0;
                    end
                end

                // Update watermark if deallocating from the end
                if (dealloc_col_base + dealloc_num_cols >= alloc_watermark) begin
                    alloc_watermark <= dealloc_col_base;
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Load Interface (tcgen05.ld)
    // Read up to 16 columns (512 bits) per cycle from a single row
    //------------------------------------------------------------------------
    integer ld_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ld_ready <= 1'b0;
            ld_data <= 512'b0;
        end else begin
            ld_ready <= 1'b0;

            if (ld_valid) begin
                // Read up to 16 columns starting from ld_col_base
                for (ld_i = 0; ld_i < 16; ld_i = ld_i + 1) begin
                    if (ld_i < ld_num_cols && (ld_col_base + ld_i) < NUM_COLS) begin
                        ld_data[ld_i*32 +: 32] <= tmem[ld_row][ld_col_base + ld_i];
                    end else begin
                        ld_data[ld_i*32 +: 32] <= 32'b0;
                    end
                end
                ld_ready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Store Interface (tcgen05.st)
    // Write up to 8 columns (256 bits) per cycle to a single row
    //------------------------------------------------------------------------
    integer st_i;
    always @(posedge clk) begin
        if (st_valid) begin
            for (st_i = 0; st_i < 8; st_i = st_i + 1) begin
                if (st_mask[st_i] && st_i < st_num_cols && (st_col_base + st_i) < NUM_COLS) begin
                    tmem[st_row][st_col_base + st_i] <= st_data[st_i*32 +: 32];
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st_ready <= 1'b0;
        end else begin
            st_ready <= st_valid;
        end
    end

    //------------------------------------------------------------------------
    // MMA Accumulator Interface
    // Dedicated port for tensor core with full 512-bit bandwidth
    //------------------------------------------------------------------------
    integer mma_wr_i;
    always @(posedge clk) begin
        if (mma_wr_valid) begin
            for (mma_wr_i = 0; mma_wr_i < 16; mma_wr_i = mma_wr_i + 1) begin
                if (mma_wr_mask[mma_wr_i] && (mma_col_base + mma_wr_i) < NUM_COLS) begin
                    tmem[mma_row][mma_col_base + mma_wr_i] <= mma_wr_data[mma_wr_i*32 +: 32];
                end
            end
        end
    end

    integer mma_rd_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mma_rd_ready <= 1'b0;
            mma_rd_data <= 512'b0;
        end else begin
            mma_rd_ready <= 1'b0;

            if (mma_rd_valid) begin
                for (mma_rd_i = 0; mma_rd_i < 16; mma_rd_i = mma_rd_i + 1) begin
                    if ((mma_col_base + mma_rd_i) < NUM_COLS) begin
                        mma_rd_data[mma_rd_i*32 +: 32] <= tmem[mma_row][mma_col_base + mma_rd_i];
                    end else begin
                        mma_rd_data[mma_rd_i*32 +: 32] <= 32'b0;
                    end
                end
                mma_rd_ready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Copy Interface (tcgen05.cp)
    // Async tensor data transfers between SMEM and TMEM
    //------------------------------------------------------------------------
    integer cp_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cp_ready <= 1'b0;
            cp_rd_data <= 512'b0;
        end else begin
            cp_ready <= 1'b0;

            if (cp_valid) begin
                if (cp_direction == 1'b0) begin
                    // SMEM -> TMEM (write)
                    for (cp_i = 0; cp_i < 16; cp_i = cp_i + 1) begin
                        if ((cp_col_base + cp_i) < NUM_COLS) begin
                            tmem[cp_row][cp_col_base + cp_i] <= cp_wr_data[cp_i*32 +: 32];
                        end
                    end
                end else begin
                    // TMEM -> SMEM (read)
                    for (cp_i = 0; cp_i < 16; cp_i = cp_i + 1) begin
                        if ((cp_col_base + cp_i) < NUM_COLS) begin
                            cp_rd_data[cp_i*32 +: 32] <= tmem[cp_row][cp_col_base + cp_i];
                        end else begin
                            cp_rd_data[cp_i*32 +: 32] <= 32'b0;
                        end
                    end
                end
                cp_ready <= 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Initialization (Simulation)
    //------------------------------------------------------------------------
    integer init_row, init_col;
    initial begin
        for (init_row = 0; init_row < NUM_ROWS; init_row = init_row + 1) begin
            for (init_col = 0; init_col < NUM_COLS; init_col = init_col + 1) begin
                tmem[init_row][init_col] = {DATA_WIDTH{1'b0}};
            end
        end
    end

    //------------------------------------------------------------------------
    // Debug Assertions (Simulation Only)
    //------------------------------------------------------------------------
`ifdef SIMULATION
    always @(posedge clk) begin
        // Check for access to unallocated columns
        if (st_valid && !col_alloc_bitmap[st_col_base]) begin
            $display("WARNING: [%0t] TMEM write to unallocated column %0d", $time, st_col_base);
        end
        if (ld_valid && !col_alloc_bitmap[ld_col_base]) begin
            $display("WARNING: [%0t] TMEM read from unallocated column %0d", $time, ld_col_base);
        end

        // Check row bounds
        if (ld_valid && ld_row >= NUM_ROWS) begin
            $display("ERROR: [%0t] TMEM load row %0d out of bounds", $time, ld_row);
        end
        if (st_valid && st_row >= NUM_ROWS) begin
            $display("ERROR: [%0t] TMEM store row %0d out of bounds", $time, st_row);
        end
    end
`endif

endmodule
