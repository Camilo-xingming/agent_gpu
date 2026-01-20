//============================================================================
// RalphGPU - TMA Unit (Tensor Memory Accelerator)
// Handles address generation for 2D/3D tiled copies (cp.async.bulk.tensor)
//
// Features:
// - 2D tiled copy with configurable box dimensions
// - Stride-based address generation
// - Integration with async_copy_engine for memory transfers
// - Support for tensor descriptors (simplified 64-bit format)
//
// Descriptor format (64-bit):
// [31:0]   - Base address in global memory
// [47:32]  - Stride in bytes (row pitch)
// [55:48]  - Box width in bytes
// [63:56]  - Box height in rows
//============================================================================

`include "gpu_defines.vh"

module tma_unit #(
    parameter ADDR_W = 32,
    parameter SMEM_ADDR_W = 14,
    parameter TRANSFER_SIZE = 16    // 16 bytes per transfer (128-bit)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control Interface
    input  wire                     start,
    input  wire [63:0]              tensor_desc,    // Tensor descriptor
    input  wire [31:0]              coord_x,        // Byte offset X in tensor
    input  wire [31:0]              coord_y,        // Row offset Y in tensor
    input  wire [SMEM_ADDR_W-1:0]   dst_base,       // Shared memory base address

    // Output Request Stream (to async_copy_engine)
    output reg                      req_valid,
    output reg  [ADDR_W-1:0]        req_src_addr,
    output reg  [SMEM_ADDR_W-1:0]   req_dst_addr,
    output reg  [4:0]               req_size,       // Size in bytes (4, 8, 16)
    input  wire                     req_ready,      // ACE ready to accept

    // Status
    output reg                      done,
    output reg                      busy,
    output reg  [15:0]              bytes_copied    // Total bytes copied
);

    //------------------------------------------------------------------------
    // Descriptor Unpacking
    //------------------------------------------------------------------------
    wire [31:0] desc_base_addr  = tensor_desc[31:0];
    wire [15:0] desc_stride     = tensor_desc[47:32];
    wire [7:0]  desc_box_width  = tensor_desc[55:48];   // Width in bytes
    wire [7:0]  desc_box_height = tensor_desc[63:56];   // Height in rows

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE     = 3'd0;
    localparam ST_START    = 3'd1;
    localparam ST_GEN_REQ  = 3'd2;
    localparam ST_WAIT_ACK = 3'd3;
    localparam ST_NEXT_ROW = 3'd4;
    localparam ST_DONE     = 3'd5;

    reg [2:0]   state;
    reg [7:0]   curr_row;       // Current row being processed
    reg [7:0]   curr_col;       // Current column offset in bytes
    reg [31:0]  row_base_addr;  // Base address for current row
    reg [SMEM_ADDR_W-1:0] smem_offset;  // Current SMEM offset

    //------------------------------------------------------------------------
    // Address Calculation
    //------------------------------------------------------------------------
    wire [31:0] computed_src_addr = row_base_addr + {24'b0, curr_col};
    wire [SMEM_ADDR_W-1:0] computed_dst_addr = dst_base + smem_offset;

    // Determine transfer size for current request
    // Use 16 bytes unless near end of row
    wire [7:0] remaining_in_row = desc_box_width - curr_col;
    wire [4:0] transfer_size_sel = (remaining_in_row >= 8'd16) ? 5'd16 :
                                   (remaining_in_row >= 8'd8)  ? 5'd8  :
                                   (remaining_in_row >= 8'd4)  ? 5'd4  : 5'd4;

    //------------------------------------------------------------------------
    // Main State Machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            req_valid <= 1'b0;
            req_src_addr <= 32'b0;
            req_dst_addr <= {SMEM_ADDR_W{1'b0}};
            req_size <= 5'd16;
            done <= 1'b0;
            busy <= 1'b0;
            bytes_copied <= 16'b0;
            curr_row <= 8'b0;
            curr_col <= 8'b0;
            row_base_addr <= 32'b0;
            smem_offset <= {SMEM_ADDR_W{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    req_valid <= 1'b0;

                    if (start && desc_box_width > 0 && desc_box_height > 0) begin
                        state <= ST_START;
                        busy <= 1'b1;
                        bytes_copied <= 16'b0;
                        curr_row <= 8'b0;
                        curr_col <= 8'b0;
                        smem_offset <= {SMEM_ADDR_W{1'b0}};

                        // Calculate initial row base address
                        // Base + (coord_y + 0) * stride + coord_x
                        row_base_addr <= desc_base_addr + (coord_y * {16'b0, desc_stride}) + coord_x;

                        `ifdef SIMULATION
                        $display("[TMA] Start: base=0x%08x stride=%0d width=%0d height=%0d coord=(%0d,%0d) dst=0x%04x",
                                 desc_base_addr, desc_stride, desc_box_width, desc_box_height,
                                 coord_x, coord_y, dst_base);
                        `endif
                    end
                end

                ST_START: begin
                    // Generate first request
                    state <= ST_GEN_REQ;
                end

                ST_GEN_REQ: begin
                    // Generate memory request
                    req_valid <= 1'b1;
                    req_src_addr <= computed_src_addr;
                    req_dst_addr <= computed_dst_addr;
                    req_size <= transfer_size_sel;
                    state <= ST_WAIT_ACK;

                    `ifdef SIMULATION
                    $display("[TMA] Req: row=%0d col=%0d src=0x%08x dst=0x%04x size=%0d",
                             curr_row, curr_col, computed_src_addr, computed_dst_addr, transfer_size_sel);
                    `endif
                end

                ST_WAIT_ACK: begin
                    if (req_ready) begin
                        // Request accepted, advance position
                        req_valid <= 1'b0;
                        bytes_copied <= bytes_copied + {11'b0, req_size};

                        // Advance column (req_size is 5 bits, pad to 8 bits)
                        curr_col <= curr_col + {3'b0, req_size};
                        smem_offset <= smem_offset + {{(SMEM_ADDR_W-5){1'b0}}, req_size};

                        // Check if row complete
                        if (curr_col + {3'b0, req_size} >= desc_box_width) begin
                            state <= ST_NEXT_ROW;
                        end else begin
                            state <= ST_GEN_REQ;
                        end
                    end
                end

                ST_NEXT_ROW: begin
                    curr_col <= 8'b0;
                    curr_row <= curr_row + 1'b1;

                    // Check if all rows complete
                    if (curr_row + 1'b1 >= desc_box_height) begin
                        state <= ST_DONE;
                    end else begin
                        // Calculate next row base address
                        row_base_addr <= row_base_addr + {16'b0, desc_stride};
                        state <= ST_GEN_REQ;
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    req_valid <= 1'b0;
                    state <= ST_IDLE;

                    `ifdef SIMULATION
                    $display("[TMA] Done: total_bytes=%0d", bytes_copied);
                    `endif
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// TMA Descriptor Builder
// Helper module to construct tensor descriptors
//============================================================================
module tma_descriptor_builder (
    input  wire [31:0]  base_addr,      // Base address in global memory
    input  wire [15:0]  stride,         // Row stride in bytes
    input  wire [7:0]   box_width,      // Box width in bytes
    input  wire [7:0]   box_height,     // Box height in rows

    output wire [63:0]  descriptor      // Packed descriptor
);

    assign descriptor = {box_height, box_width, stride, base_addr};

endmodule


//============================================================================
// TMA 3D Extension (for future 3D tensor copies)
//============================================================================
module tma_unit_3d #(
    parameter ADDR_W = 32,
    parameter SMEM_ADDR_W = 14,
    parameter TRANSFER_SIZE = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control Interface
    input  wire                     start,
    input  wire [95:0]              tensor_desc,    // Extended 96-bit descriptor
                                                    // [31:0]   Base address
                                                    // [47:32]  Row stride (Y)
                                                    // [63:48]  Slice stride (Z)
                                                    // [71:64]  Box width (X)
                                                    // [79:72]  Box height (Y)
                                                    // [87:80]  Box depth (Z)
                                                    // [95:88]  Reserved
    input  wire [31:0]              coord_x,
    input  wire [31:0]              coord_y,
    input  wire [31:0]              coord_z,
    input  wire [SMEM_ADDR_W-1:0]   dst_base,

    // Output Request Stream
    output reg                      req_valid,
    output reg  [ADDR_W-1:0]        req_src_addr,
    output reg  [SMEM_ADDR_W-1:0]   req_dst_addr,
    output reg  [4:0]               req_size,
    input  wire                     req_ready,

    // Status
    output reg                      done,
    output reg                      busy,
    output reg  [23:0]              bytes_copied
);

    //------------------------------------------------------------------------
    // Descriptor Unpacking
    //------------------------------------------------------------------------
    wire [31:0] desc_base        = tensor_desc[31:0];
    wire [15:0] desc_row_stride  = tensor_desc[47:32];
    wire [15:0] desc_slice_stride = tensor_desc[63:48];
    wire [7:0]  desc_box_x       = tensor_desc[71:64];
    wire [7:0]  desc_box_y       = tensor_desc[79:72];
    wire [7:0]  desc_box_z       = tensor_desc[87:80];

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE      = 4'd0;
    localparam ST_START     = 4'd1;
    localparam ST_GEN_REQ   = 4'd2;
    localparam ST_WAIT_ACK  = 4'd3;
    localparam ST_NEXT_ROW  = 4'd4;
    localparam ST_NEXT_SLICE = 4'd5;
    localparam ST_DONE      = 4'd6;

    reg [3:0]   state;
    reg [7:0]   curr_x, curr_y, curr_z;
    reg [31:0]  slice_base_addr;
    reg [31:0]  row_base_addr;
    reg [SMEM_ADDR_W-1:0] smem_offset;

    wire [31:0] computed_src = row_base_addr + {24'b0, curr_x};
    wire [SMEM_ADDR_W-1:0] computed_dst = dst_base + smem_offset;
    wire [7:0] remaining = desc_box_x - curr_x;
    wire [4:0] xfer_size = (remaining >= 8'd16) ? 5'd16 :
                           (remaining >= 8'd8)  ? 5'd8  : 5'd4;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            req_valid <= 1'b0;
            done <= 1'b0;
            busy <= 1'b0;
            bytes_copied <= 24'b0;
            curr_x <= 8'b0;
            curr_y <= 8'b0;
            curr_z <= 8'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done <= 1'b0;
                    req_valid <= 1'b0;
                    if (start && desc_box_x > 0 && desc_box_y > 0 && desc_box_z > 0) begin
                        state <= ST_START;
                        busy <= 1'b1;
                        bytes_copied <= 24'b0;
                        curr_x <= 8'b0;
                        curr_y <= 8'b0;
                        curr_z <= 8'b0;
                        smem_offset <= {SMEM_ADDR_W{1'b0}};
                        slice_base_addr <= desc_base + coord_z * {16'b0, desc_slice_stride};
                        row_base_addr <= desc_base + coord_z * {16'b0, desc_slice_stride} +
                                        coord_y * {16'b0, desc_row_stride} + coord_x;
                    end
                end

                ST_START: state <= ST_GEN_REQ;

                ST_GEN_REQ: begin
                    req_valid <= 1'b1;
                    req_src_addr <= computed_src;
                    req_dst_addr <= computed_dst;
                    req_size <= xfer_size;
                    state <= ST_WAIT_ACK;
                end

                ST_WAIT_ACK: begin
                    if (req_ready) begin
                        req_valid <= 1'b0;
                        bytes_copied <= bytes_copied + {19'b0, req_size};
                        curr_x <= curr_x + {3'b0, req_size};
                        smem_offset <= smem_offset + {{(SMEM_ADDR_W-5){1'b0}}, req_size};

                        if (curr_x + {3'b0, req_size} >= desc_box_x) begin
                            state <= ST_NEXT_ROW;
                        end else begin
                            state <= ST_GEN_REQ;
                        end
                    end
                end

                ST_NEXT_ROW: begin
                    curr_x <= 8'b0;
                    curr_y <= curr_y + 1'b1;

                    if (curr_y + 1'b1 >= desc_box_y) begin
                        state <= ST_NEXT_SLICE;
                    end else begin
                        row_base_addr <= row_base_addr + {16'b0, desc_row_stride};
                        state <= ST_GEN_REQ;
                    end
                end

                ST_NEXT_SLICE: begin
                    curr_x <= 8'b0;
                    curr_y <= 8'b0;
                    curr_z <= curr_z + 1'b1;

                    if (curr_z + 1'b1 >= desc_box_z) begin
                        state <= ST_DONE;
                    end else begin
                        slice_base_addr <= slice_base_addr + {16'b0, desc_slice_stride};
                        row_base_addr <= slice_base_addr + {16'b0, desc_slice_stride} +
                                        coord_y * {16'b0, desc_row_stride} + coord_x;
                        state <= ST_GEN_REQ;
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    req_valid <= 1'b0;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
