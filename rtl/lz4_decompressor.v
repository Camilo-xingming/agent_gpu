//============================================================================
// RalphGPU - LZ4 Hardware Decompressor
// High-throughput LZ4 decompression for FP4 memory bandwidth doubling
//
// LZ4 Format:
// - Token: [literal_len:4][match_len:4]
// - If literal_len == 15: additional bytes until byte < 255
// - Literal bytes
// - Offset: 2 bytes (little-endian)
// - If match_len == 15: additional bytes until byte < 255
//
// Design Features:
// - 64-bit input bus for HBM/GDDR bandwidth matching
// - Pipelined architecture for sustained throughput
// - History buffer for match copy operations
// - Support for up to 64KB history window
//============================================================================

`timescale 1ns / 1ps

module lz4_decompressor #(
    parameter INPUT_WIDTH   = 64,           // Input data width (bits)
    parameter OUTPUT_WIDTH  = 256,          // Output data width (bits) - 8x32-bit words
    parameter HISTORY_DEPTH = 16384,        // History buffer depth (bytes) - 64KB
    parameter HISTORY_ADDR_W = $clog2(HISTORY_DEPTH)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Control Interface
    //------------------------------------------------------------------------
    input  wire                     start,              // Start decompression
    input  wire [31:0]              compressed_size,    // Size of compressed data (bytes)
    input  wire [31:0]              uncompressed_size,  // Expected uncompressed size
    output reg                      done,               // Decompression complete
    output reg                      error,              // Error flag

    //------------------------------------------------------------------------
    // Compressed Data Input (AXI-Stream style)
    //------------------------------------------------------------------------
    input  wire [INPUT_WIDTH-1:0]   s_axis_tdata,
    input  wire                     s_axis_tvalid,
    output reg                      s_axis_tready,
    input  wire                     s_axis_tlast,

    //------------------------------------------------------------------------
    // Decompressed Data Output (AXI-Stream style)
    //------------------------------------------------------------------------
    output reg  [OUTPUT_WIDTH-1:0]  m_axis_tdata,
    output reg                      m_axis_tvalid,
    input  wire                     m_axis_tready,
    output reg                      m_axis_tlast,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output reg  [31:0]              stat_bytes_in,
    output reg  [31:0]              stat_bytes_out,
    output reg  [31:0]              stat_literal_count,
    output reg  [31:0]              stat_match_count
);

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE          = 4'd0;
    localparam ST_READ_TOKEN    = 4'd1;
    localparam ST_LITERAL_LEN   = 4'd2;
    localparam ST_COPY_LITERAL  = 4'd3;
    localparam ST_READ_OFFSET   = 4'd4;
    localparam ST_MATCH_LEN     = 4'd5;
    localparam ST_COPY_MATCH    = 4'd6;
    localparam ST_OUTPUT        = 4'd7;
    localparam ST_DONE          = 4'd8;
    localparam ST_ERROR         = 4'd9;

    reg [3:0] state, next_state;

    //------------------------------------------------------------------------
    // Input Buffer (shift register for byte-level access)
    //------------------------------------------------------------------------
    localparam INPUT_BUF_SIZE = 16;  // Bytes
    reg [7:0] input_buf [0:INPUT_BUF_SIZE-1];
    reg [3:0] input_buf_count;
    reg [3:0] input_buf_rd_ptr;

    //------------------------------------------------------------------------
    // History Buffer (circular buffer for match lookback)
    //------------------------------------------------------------------------
    reg [7:0] history [0:HISTORY_DEPTH-1];
    reg [HISTORY_ADDR_W-1:0] history_wr_ptr;
    reg [HISTORY_ADDR_W-1:0] history_rd_ptr;

    //------------------------------------------------------------------------
    // Output Buffer
    //------------------------------------------------------------------------
    localparam OUTPUT_BUF_SIZE = 32;  // Bytes
    reg [7:0] output_buf [0:OUTPUT_BUF_SIZE-1];
    reg [4:0] output_buf_count;

    //------------------------------------------------------------------------
    // Parsing State
    //------------------------------------------------------------------------
    reg [3:0]  token_literal_len;   // From token byte (0-15)
    reg [3:0]  token_match_len;     // From token byte (0-15)
    reg [31:0] literal_len;         // Extended literal length
    reg [31:0] match_len;           // Extended match length (add 4 for MINMATCH)
    reg [15:0] match_offset;        // Offset into history buffer
    reg [31:0] copy_count;          // Bytes remaining to copy

    //------------------------------------------------------------------------
    // Counters
    //------------------------------------------------------------------------
    reg [31:0] bytes_consumed;
    reg [31:0] bytes_produced;

    //------------------------------------------------------------------------
    // Helper: Get byte from input buffer
    //------------------------------------------------------------------------
    wire [7:0] input_byte = input_buf[input_buf_rd_ptr];
    wire input_buf_empty = (input_buf_count == 0);
    wire input_buf_has_data = !input_buf_empty;

    //------------------------------------------------------------------------
    // State Machine - Sequential
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    //------------------------------------------------------------------------
    // State Machine - Combinational
    //------------------------------------------------------------------------
    always @(*) begin
        next_state = state;

        case (state)
            ST_IDLE: begin
                if (start) begin
                    next_state = ST_READ_TOKEN;
                end
            end

            ST_READ_TOKEN: begin
                if (bytes_consumed >= compressed_size) begin
                    next_state = ST_DONE;
                end else if (input_buf_has_data) begin
                    // Check if literal length needs extension
                    if (input_byte[7:4] == 4'hF) begin
                        next_state = ST_LITERAL_LEN;
                    end else if (input_byte[7:4] > 0) begin
                        next_state = ST_COPY_LITERAL;
                    end else begin
                        next_state = ST_READ_OFFSET;
                    end
                end
            end

            ST_LITERAL_LEN: begin
                if (input_buf_has_data) begin
                    if (input_byte == 8'hFF) begin
                        next_state = ST_LITERAL_LEN;  // Continue reading
                    end else begin
                        next_state = ST_COPY_LITERAL;
                    end
                end
            end

            ST_COPY_LITERAL: begin
                if (copy_count == 0) begin
                    // Check if we're at end of block
                    if (bytes_consumed >= compressed_size) begin
                        next_state = ST_DONE;
                    end else begin
                        next_state = ST_READ_OFFSET;
                    end
                end
            end

            ST_READ_OFFSET: begin
                if (input_buf_count >= 2) begin
                    // Check if match length needs extension
                    if (token_match_len == 4'hF) begin
                        next_state = ST_MATCH_LEN;
                    end else begin
                        next_state = ST_COPY_MATCH;
                    end
                end
            end

            ST_MATCH_LEN: begin
                if (input_buf_has_data) begin
                    if (input_byte == 8'hFF) begin
                        next_state = ST_MATCH_LEN;  // Continue reading
                    end else begin
                        next_state = ST_COPY_MATCH;
                    end
                end
            end

            ST_COPY_MATCH: begin
                if (copy_count == 0) begin
                    next_state = ST_OUTPUT;
                end
            end

            ST_OUTPUT: begin
                if (output_buf_count == 0 || (m_axis_tvalid && m_axis_tready)) begin
                    if (bytes_produced >= uncompressed_size) begin
                        next_state = ST_DONE;
                    end else begin
                        next_state = ST_READ_TOKEN;
                    end
                end
            end

            ST_DONE: begin
                next_state = ST_IDLE;
            end

            ST_ERROR: begin
                // Stay in error until reset
                next_state = ST_ERROR;
            end

            default: next_state = ST_ERROR;
        endcase
    end

    //------------------------------------------------------------------------
    // Main Datapath
    //------------------------------------------------------------------------
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all state
            done <= 1'b0;
            error <= 1'b0;
            s_axis_tready <= 1'b0;
            m_axis_tdata <= {OUTPUT_WIDTH{1'b0}};
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;

            input_buf_count <= 4'd0;
            input_buf_rd_ptr <= 4'd0;
            output_buf_count <= 5'd0;

            history_wr_ptr <= {HISTORY_ADDR_W{1'b0}};
            history_rd_ptr <= {HISTORY_ADDR_W{1'b0}};

            token_literal_len <= 4'd0;
            token_match_len <= 4'd0;
            literal_len <= 32'd0;
            match_len <= 32'd0;
            match_offset <= 16'd0;
            copy_count <= 32'd0;

            bytes_consumed <= 32'd0;
            bytes_produced <= 32'd0;

            stat_bytes_in <= 32'd0;
            stat_bytes_out <= 32'd0;
            stat_literal_count <= 32'd0;
            stat_match_count <= 32'd0;

            for (i = 0; i < INPUT_BUF_SIZE; i = i + 1) begin
                input_buf[i] <= 8'd0;
            end
            for (i = 0; i < OUTPUT_BUF_SIZE; i = i + 1) begin
                output_buf[i] <= 8'd0;
            end
        end else begin
            // Default outputs
            done <= 1'b0;
            m_axis_tvalid <= 1'b0;

            // Fill input buffer from AXI stream
            s_axis_tready <= (input_buf_count < INPUT_BUF_SIZE - 8);
            if (s_axis_tvalid && s_axis_tready) begin
                for (i = 0; i < 8; i = i + 1) begin
                    input_buf[(input_buf_count + i) % INPUT_BUF_SIZE] <=
                        s_axis_tdata[i*8 +: 8];
                end
                input_buf_count <= input_buf_count + 8;
            end

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        // Reset counters
                        bytes_consumed <= 32'd0;
                        bytes_produced <= 32'd0;
                        history_wr_ptr <= {HISTORY_ADDR_W{1'b0}};
                        stat_bytes_in <= 32'd0;
                        stat_bytes_out <= 32'd0;
                        stat_literal_count <= 32'd0;
                        stat_match_count <= 32'd0;
                        error <= 1'b0;
                    end
                end

                ST_READ_TOKEN: begin
                    if (input_buf_has_data && bytes_consumed < compressed_size) begin
                        token_literal_len <= input_byte[7:4];
                        token_match_len <= input_byte[3:0];
                        literal_len <= {28'd0, input_byte[7:4]};
                        match_len <= {28'd0, input_byte[3:0]} + 4;  // MINMATCH=4

                        // Consume token byte
                        input_buf_rd_ptr <= (input_buf_rd_ptr + 1) % INPUT_BUF_SIZE;
                        input_buf_count <= input_buf_count - 1;
                        bytes_consumed <= bytes_consumed + 1;
                        stat_bytes_in <= stat_bytes_in + 1;
                    end
                end

                ST_LITERAL_LEN: begin
                    if (input_buf_has_data) begin
                        literal_len <= literal_len + {24'd0, input_byte};

                        input_buf_rd_ptr <= (input_buf_rd_ptr + 1) % INPUT_BUF_SIZE;
                        input_buf_count <= input_buf_count - 1;
                        bytes_consumed <= bytes_consumed + 1;
                        stat_bytes_in <= stat_bytes_in + 1;

                        if (input_byte != 8'hFF) begin
                            copy_count <= literal_len + {24'd0, input_byte};
                        end
                    end
                end

                ST_COPY_LITERAL: begin
                    if (state == ST_LITERAL_LEN && next_state == ST_COPY_LITERAL) begin
                        copy_count <= literal_len;
                    end

                    if (input_buf_has_data && copy_count > 0) begin
                        // Copy literal byte to output and history
                        output_buf[output_buf_count] <= input_byte;
                        history[history_wr_ptr] <= input_byte;
                        history_wr_ptr <= (history_wr_ptr + 1) % HISTORY_DEPTH;

                        output_buf_count <= output_buf_count + 1;
                        copy_count <= copy_count - 1;

                        input_buf_rd_ptr <= (input_buf_rd_ptr + 1) % INPUT_BUF_SIZE;
                        input_buf_count <= input_buf_count - 1;
                        bytes_consumed <= bytes_consumed + 1;
                        bytes_produced <= bytes_produced + 1;

                        stat_bytes_in <= stat_bytes_in + 1;
                        stat_bytes_out <= stat_bytes_out + 1;
                        stat_literal_count <= stat_literal_count + 1;
                    end
                end

                ST_READ_OFFSET: begin
                    if (input_buf_count >= 2) begin
                        // Read 16-bit offset (little-endian)
                        match_offset <= {input_buf[(input_buf_rd_ptr + 1) % INPUT_BUF_SIZE],
                                        input_buf[input_buf_rd_ptr]};

                        input_buf_rd_ptr <= (input_buf_rd_ptr + 2) % INPUT_BUF_SIZE;
                        input_buf_count <= input_buf_count - 2;
                        bytes_consumed <= bytes_consumed + 2;
                        stat_bytes_in <= stat_bytes_in + 2;

                        // Prepare for match copy
                        copy_count <= match_len;
                    end
                end

                ST_MATCH_LEN: begin
                    if (input_buf_has_data) begin
                        match_len <= match_len + {24'd0, input_byte};

                        input_buf_rd_ptr <= (input_buf_rd_ptr + 1) % INPUT_BUF_SIZE;
                        input_buf_count <= input_buf_count - 1;
                        bytes_consumed <= bytes_consumed + 1;
                        stat_bytes_in <= stat_bytes_in + 1;

                        if (input_byte != 8'hFF) begin
                            copy_count <= match_len + {24'd0, input_byte};
                        end
                    end
                end

                ST_COPY_MATCH: begin
                    if (copy_count > 0) begin
                        // Calculate source address in history
                        history_rd_ptr <= (history_wr_ptr - match_offset) % HISTORY_DEPTH;

                        // Copy from history
                        output_buf[output_buf_count] <= history[history_rd_ptr];
                        history[history_wr_ptr] <= history[history_rd_ptr];
                        history_wr_ptr <= (history_wr_ptr + 1) % HISTORY_DEPTH;

                        output_buf_count <= output_buf_count + 1;
                        copy_count <= copy_count - 1;
                        bytes_produced <= bytes_produced + 1;

                        stat_bytes_out <= stat_bytes_out + 1;
                        stat_match_count <= stat_match_count + 1;
                    end
                end

                ST_OUTPUT: begin
                    // Output accumulated data
                    if (output_buf_count >= (OUTPUT_WIDTH/8) ||
                        bytes_produced >= uncompressed_size) begin
                        // Pack output buffer to output data
                        for (i = 0; i < OUTPUT_WIDTH/8; i = i + 1) begin
                            if (i < output_buf_count) begin
                                m_axis_tdata[i*8 +: 8] <= output_buf[i];
                            end else begin
                                m_axis_tdata[i*8 +: 8] <= 8'd0;
                            end
                        end
                        m_axis_tvalid <= 1'b1;
                        m_axis_tlast <= (bytes_produced >= uncompressed_size);

                        if (m_axis_tready) begin
                            output_buf_count <= 5'd0;
                        end
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                end

                ST_ERROR: begin
                    error <= 1'b1;
                end
                default: ; // lint: CASEINCOMPLETE
            endcase
        end
    end

endmodule
