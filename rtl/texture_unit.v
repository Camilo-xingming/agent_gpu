//============================================================================
// RalphGPU - Texture/Surface Unit
// PTX texture and surface memory instructions
// Supports 1D/2D/3D/Cube textures with filtering and LOD
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module texture_unit #(
    parameter CACHE_SIZE_KB = 16,  // Texture cache size
    parameter MAX_ANISO     = 16   // Maximum anisotropic filtering level
)(
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire [5:0]  opcode,     // TEX, TXQ, SULD, SUST, SURED
    input  wire [5:0]  func,       // Texture type/query type
    input  wire        valid_in,

    // Texture coordinates (normalized or unnormalized)
    input  wire [31:0] coord_s,    // S coordinate / X
    input  wire [31:0] coord_t,    // T coordinate / Y
    input  wire [31:0] coord_r,    // R coordinate / Z (for 3D/cube)
    input  wire [31:0] coord_q,    // Q coordinate / array index

    // LOD control
    input  wire [31:0] lod,        // Explicit LOD (for tex.level)
    input  wire [31:0] dsdx,       // Gradient for LOD calculation
    input  wire [31:0] dsdy,
    input  wire [31:0] dtdx,
    input  wire [31:0] dtdy,

    // Texture descriptor (from descriptor table)
    input  wire [31:0] tex_base_addr,
    input  wire [15:0] tex_width,
    input  wire [15:0] tex_height,
    input  wire [15:0] tex_depth,
    input  wire [3:0]  tex_format,     // RGBA8, RGBA16F, etc.
    input  wire [3:0]  tex_filter,     // Point, bilinear, trilinear
    input  wire [3:0]  tex_wrap_s,     // Wrap/clamp/mirror
    input  wire [3:0]  tex_wrap_t,
    input  wire [3:0]  tex_wrap_r,
    input  wire [3:0]  num_mip_levels,

    // Surface store data
    input  wire [127:0] store_data,

    // Memory interface (to L2 cache/memory)
    output reg         mem_req,
    output reg         mem_write,
    output reg  [31:0] mem_addr,
    output reg  [127:0] mem_wdata,
    input  wire        mem_ready,
    input  wire [127:0] mem_rdata,
    input  wire        mem_valid,

    // Result (RGBA as 4x32-bit or single 32-bit for TXQ)
    output reg  [127:0] result,
    output reg          valid_out,

    // Status
    output reg          busy
);

    //------------------------------------------------------------------------
    // Texture format definitions
    //------------------------------------------------------------------------
    localparam FMT_RGBA8_UNORM    = 4'h0;
    localparam FMT_RGBA8_SNORM    = 4'h1;
    localparam FMT_RGBA16_FLOAT   = 4'h2;
    localparam FMT_RGBA32_FLOAT   = 4'h3;
    localparam FMT_R32_FLOAT      = 4'h4;
    localparam FMT_RG32_FLOAT     = 4'h5;
    localparam FMT_R8_UNORM       = 4'h6;
    localparam FMT_RG8_UNORM      = 4'h7;
    localparam FMT_R16_FLOAT      = 4'h8;
    localparam FMT_RG16_FLOAT     = 4'h9;

    // Filter modes
    localparam FILTER_POINT       = 4'h0;
    localparam FILTER_LINEAR      = 4'h1;
    localparam FILTER_TRILINEAR   = 4'h2;
    localparam FILTER_ANISO       = 4'h3;

    // Wrap modes
    localparam WRAP_REPEAT        = 4'h0;
    localparam WRAP_CLAMP         = 4'h1;
    localparam WRAP_MIRROR        = 4'h2;
    localparam WRAP_BORDER        = 4'h3;

    //------------------------------------------------------------------------
    // State machine
    //------------------------------------------------------------------------
    localparam IDLE         = 3'b000;
    localparam CALC_ADDR    = 3'b001;
    localparam FETCH_TEX    = 3'b010;
    localparam FILTER       = 3'b011;
    localparam OUTPUT       = 3'b100;
    localparam TXQ_EXEC     = 3'b101;
    localparam SURF_ACCESS  = 3'b110;

    reg [2:0] state;
    reg [2:0] next_state;

    //------------------------------------------------------------------------
    // Pipeline registers
    //------------------------------------------------------------------------
    reg [5:0]  opcode_r, func_r;
    reg [31:0] coord_s_r, coord_t_r, coord_r_r;
    reg [31:0] lod_r;
    reg [15:0] tex_w_r, tex_h_r, tex_d_r;
    reg [3:0]  filter_r, wrap_s_r, wrap_t_r;
    reg [3:0]  format_r;
    reg [4:0]  texel_size_r;

    // Calculated addresses and weights
    reg [31:0] texel_addr [0:7];  // Up to 8 texels for trilinear
    reg [7:0]  blend_weight [0:7];
    reg [2:0]  num_texels;

    // Fetched texel data
    reg [31:0] texel_r [0:7];
    reg [31:0] texel_g [0:7];
    reg [31:0] texel_b [0:7];
    reg [31:0] texel_a [0:7];
    reg [2:0]  fetch_idx;

    // Bilinear blend fractional weights
    reg [7:0]  blend_frac_s, blend_frac_t;

    // Memory request pending flag for multi-texel fetch
    reg         req_pending;

    //------------------------------------------------------------------------
    // Coordinate wrapping
    //------------------------------------------------------------------------
    function [31:0] wrap_coord;
        input [31:0] coord;
        input [15:0] size;
        input [3:0]  mode;
        reg [31:0] wrapped;
        begin
            case (mode)
                WRAP_REPEAT: begin
                    // coord mod size
                    wrapped = coord % {16'b0, size};
                end
                WRAP_CLAMP: begin
                    if (coord[31])  // negative
                        wrapped = 32'b0;
                    else if (coord >= {16'b0, size})
                        wrapped = {16'b0, size} - 1;
                    else
                        wrapped = coord;
                end
                WRAP_MIRROR: begin
                    wrapped = coord % ({16'b0, size} * 2);
                    if (wrapped >= {16'b0, size})
                        wrapped = ({16'b0, size} * 2) - wrapped - 1;
                end
                default: begin
                    wrapped = coord;
                end
            endcase
            wrap_coord = wrapped;
        end
    endfunction

    //------------------------------------------------------------------------
    // LOD calculation (simplified)
    //------------------------------------------------------------------------
    function [31:0] calc_lod;
        input [31:0] dsdx_in, dsdy_in, dtdx_in, dtdy_in;
        input [15:0] w, h;
        reg [31:0] rho_x, rho_y, rho;
        begin
            // Simplified LOD calculation
            // rho = max(|dsdx|*w + |dtdx|*h, |dsdy|*w + |dtdy|*h)
            rho_x = (dsdx_in[31] ? (~dsdx_in + 1) : dsdx_in) * {16'b0, w};
            rho_y = (dsdy_in[31] ? (~dsdy_in + 1) : dsdy_in) * {16'b0, w};
            rho = (rho_x > rho_y) ? rho_x : rho_y;
            // log2(rho) approximation
            calc_lod = (rho > 32'h10000) ? 32'h40000 :  // LOD 4
                      (rho > 32'h1000)  ? 32'h30000 :  // LOD 3
                      (rho > 32'h100)   ? 32'h20000 :  // LOD 2
                      (rho > 32'h10)    ? 32'h10000 :  // LOD 1
                                          32'h00000;   // LOD 0
        end
    endfunction

    //------------------------------------------------------------------------
    // Address calculation for 2D texture
    //------------------------------------------------------------------------
    function [31:0] calc_2d_addr;
        input [31:0] base;
        input [15:0] x, y;
        input [15:0] pitch;  // Bytes per row
        input [4:0]  bpp;    // Bytes per pixel
        begin
            calc_2d_addr = base + ({16'b0, y} * {16'b0, pitch}) +
                          ({16'b0, x} * {27'b0, bpp});
        end
    endfunction

    //------------------------------------------------------------------------
    // Bilinear interpolation (8-bit UNORM)
    //------------------------------------------------------------------------
    function [7:0] bilinear_interp;
        input [7:0] c00, c10, c01, c11;
        input [7:0] fx, fy;  // Fractional weights (0-255)
        reg [15:0] tmp1, tmp2, tmp3;
        begin
            // Linear interpolation in X
            tmp1 = (16'd255 - {8'b0, fx}) * c00 + {8'b0, fx} * c10;
            tmp2 = (16'd255 - {8'b0, fx}) * c01 + {8'b0, fx} * c11;
            // Linear interpolation in Y
            tmp3 = (16'd255 - {8'b0, fy}) * tmp1[15:8] + {8'b0, fy} * tmp2[15:8];
            bilinear_interp = tmp3[15:8];
        end
    endfunction

    // Truncated wrap coordinates for calc_2d_addr (iverilog can't part-select function returns)
    /* verilator lint_off WIDTHTRUNC */
    wire [31:0] int_s = { {8{coord_s_r[31]}}, coord_s_r[31:8] };
    wire [31:0] int_t = { {8{coord_t_r[31]}}, coord_t_r[31:8] };
    wire [31:0] int_r = { {8{coord_r_r[31]}}, coord_r_r[31:8] };
    wire [15:0] wrap_s_trunc = wrap_coord(int_s, tex_w_r, wrap_s_r);
    wire [15:0] wrap_t_trunc = wrap_coord(int_t, tex_h_r, wrap_t_r);
    wire [15:0] wrap_s1_trunc = wrap_coord(int_s + 1, tex_w_r, wrap_s_r);
    wire [15:0] wrap_t1_trunc = wrap_coord(int_t + 1, tex_h_r, wrap_t_r);
    /* verilator lint_on WIDTHTRUNC */

    //------------------------------------------------------------------------
    // Main state machine
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 1'b0;
            busy <= 1'b0;
            mem_req <= 1'b0;
            mem_write <= 1'b0;
            result <= 128'b0;
            fetch_idx <= 3'b0;
            req_pending <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            mem_req <= 1'b0;
            mem_write <= 1'b0;

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        busy <= 1'b1;
                        opcode_r <= opcode;
                        func_r <= func;
                        coord_s_r <= coord_s;
                        coord_t_r <= coord_t;
                        coord_r_r <= coord_r;
                        lod_r <= lod;
                        tex_w_r <= tex_width;
                        tex_h_r <= tex_height;
                        tex_d_r <= tex_depth;
                        filter_r <= tex_filter;
                        wrap_s_r <= tex_wrap_s;
                        format_r <= tex_format;
                        case (tex_format)
                            FMT_R8_UNORM: texel_size_r <= 5'd1;
                            FMT_RG8_UNORM, FMT_R16_FLOAT: texel_size_r <= 5'd2;
                            FMT_RGBA8_UNORM, FMT_RGBA8_SNORM, FMT_R32_FLOAT, FMT_RG16_FLOAT: texel_size_r <= 5'd4;
                            FMT_RGBA16_FLOAT, FMT_RG32_FLOAT: texel_size_r <= 5'd8;
                            FMT_RGBA32_FLOAT: texel_size_r <= 5'd16;
                            default: texel_size_r <= 5'd4;
                        endcase
                        wrap_t_r <= tex_wrap_t;

                        case (opcode)
                            `OP_TEX: state <= CALC_ADDR;
                            `OP_TXQ: state <= TXQ_EXEC;
                            `OP_SULD, `OP_SUST, `OP_SURED: state <= SURF_ACCESS;
                            default: state <= IDLE;
                        endcase
                    end
                end

                CALC_ADDR: begin
                    // Calculate texel addresses based on texture type and filter
                    case (func_r)
                        `TEX_1D: begin
                            texel_addr[0] <= tex_base_addr +
                                wrap_coord(int_s, tex_w_r, wrap_s_r) * {27'b0, texel_size_r};
                            num_texels <= 3'd1;
                        end

                        `TEX_2D: begin
                            if (filter_r == FILTER_POINT) begin
                                // Single texel
                                texel_addr[0] <= calc_2d_addr(
                                    tex_base_addr,
                                    wrap_s_trunc,
                                    wrap_t_trunc,
                                    tex_w_r * {11'b0, texel_size_r},  // Assuming RGBA8
                                    texel_size_r[4:0]
                                );
                                num_texels <= 3'd1;
                            end else begin
                                // Bilinear: 4 neighbor texels
                                // Integer coords for corners
                                // s0,t0 = floor(coord), s1,t1 = floor(coord)+1
                                texel_addr[0] <= calc_2d_addr(tex_base_addr,
                                    wrap_s_trunc,
                                    wrap_t_trunc,
                                    tex_w_r * {11'b0, texel_size_r}, texel_size_r);
                                texel_addr[1] <= calc_2d_addr(tex_base_addr,
                                    wrap_s1_trunc,
                                    wrap_t_trunc,
                                    tex_w_r * {11'b0, texel_size_r}, texel_size_r);
                                texel_addr[2] <= calc_2d_addr(tex_base_addr,
                                    wrap_s_trunc,
                                    wrap_t1_trunc,
                                    tex_w_r * {11'b0, texel_size_r}, texel_size_r);
                                texel_addr[3] <= calc_2d_addr(tex_base_addr,
                                    wrap_s1_trunc,
                                    wrap_t1_trunc,
                                    tex_w_r * {11'b0, texel_size_r}, texel_size_r);
                                // Fractional weights from low bits of coord
                                blend_frac_s <= coord_s_r[7:0];
                                blend_frac_t <= coord_t_r[7:0];
                                num_texels <= 3'd4;
                            end
                        end

                        `TEX_3D: begin
                            // 3D texture addressing
                            texel_addr[0] <= tex_base_addr +
                                wrap_coord(int_r, tex_d_r, tex_wrap_r) * (tex_w_r * tex_h_r * {11'b0, texel_size_r}) +
                                wrap_coord(int_t, tex_h_r, wrap_t_r) * (tex_w_r * {11'b0, texel_size_r}) +
                                wrap_coord(int_s, tex_w_r, wrap_s_r) * {27'b0, texel_size_r};
                            num_texels <= 3'd1;
                        end

                        default: begin
                            texel_addr[0] <= tex_base_addr;
                            num_texels <= 3'd1;
                        end
                    endcase
                    fetch_idx <= 3'b0;
                    state <= FETCH_TEX;
                end

                FETCH_TEX: begin
                    if (fetch_idx < num_texels) begin
                        if (!req_pending && mem_ready) begin
                            mem_req <= 1'b1;
                            mem_addr <= texel_addr[fetch_idx];
                            req_pending <= 1'b1;
                        end else if (mem_valid && req_pending) begin
                            case (format_r)
                                FMT_R8_UNORM: begin
                                    texel_r[fetch_idx] <= {24'b0, mem_rdata[7:0]};
                                    texel_g[fetch_idx] <= 32'b0;
                                    texel_b[fetch_idx] <= 32'b0;
                                    texel_a[fetch_idx] <= 32'h000000FF;
                                end
                                FMT_RGBA16_FLOAT: begin
                                    texel_r[fetch_idx] <= {16'b0, mem_rdata[15:0]};
                                    texel_g[fetch_idx] <= {16'b0, mem_rdata[31:16]};
                                    texel_b[fetch_idx] <= {16'b0, mem_rdata[47:32]};
                                    texel_a[fetch_idx] <= {16'b0, mem_rdata[63:48]};
                                end
                                FMT_RGBA32_FLOAT: begin
                                    texel_r[fetch_idx] <= mem_rdata[31:0];
                                    texel_g[fetch_idx] <= mem_rdata[63:32];
                                    texel_b[fetch_idx] <= mem_rdata[95:64];
                                    texel_a[fetch_idx] <= mem_rdata[127:96];
                                end
                                default: begin
                                    texel_r[fetch_idx] <= {24'b0, mem_rdata[7:0]};
                                    texel_g[fetch_idx] <= {24'b0, mem_rdata[15:8]};
                                    texel_b[fetch_idx] <= {24'b0, mem_rdata[23:16]};
                                    texel_a[fetch_idx] <= {24'b0, mem_rdata[31:24]};
                                end
                            endcase
                            fetch_idx <= fetch_idx + 1;
                            req_pending <= 1'b0;
                        end
                    end else begin
                        state <= FILTER;
                    end
                end

                FILTER: begin
                    if (filter_r == FILTER_POINT || num_texels == 1) begin
                        // Point sampling - return first texel
                        result <= {texel_a[0], texel_b[0], texel_g[0], texel_r[0]};
                    end else if (num_texels == 3'd4) begin
                        // Bilinear interpolation of 4 texels
                        result[31:0]   <= {24'b0, bilinear_interp(
                            texel_r[0][7:0], texel_r[1][7:0],
                            texel_r[2][7:0], texel_r[3][7:0],
                            blend_frac_s, blend_frac_t)};
                        result[63:32]  <= {24'b0, bilinear_interp(
                            texel_g[0][7:0], texel_g[1][7:0],
                            texel_g[2][7:0], texel_g[3][7:0],
                            blend_frac_s, blend_frac_t)};
                        result[95:64]  <= {24'b0, bilinear_interp(
                            texel_b[0][7:0], texel_b[1][7:0],
                            texel_b[2][7:0], texel_b[3][7:0],
                            blend_frac_s, blend_frac_t)};
                        result[127:96] <= {24'b0, bilinear_interp(
                            texel_a[0][7:0], texel_a[1][7:0],
                            texel_a[2][7:0], texel_a[3][7:0],
                            blend_frac_s, blend_frac_t)};
                    end else begin
                        result <= {texel_a[0], texel_b[0], texel_g[0], texel_r[0]};
                    end
                    state <= OUTPUT;
                end

                OUTPUT: begin
                    valid_out <= 1'b1;
                    busy <= 1'b0;
                    state <= IDLE;
                end

                TXQ_EXEC: begin
                    // Texture query - return requested property
                    case (func_r)
                        `TXQ_WIDTH:  result <= {96'b0, 16'b0, tex_w_r};
                        `TXQ_HEIGHT: result <= {96'b0, 16'b0, tex_h_r};
                        `TXQ_DEPTH:  result <= {96'b0, 16'b0, tex_d_r};
                        `TXQ_LEVELS: result <= {124'b0, num_mip_levels};
                        default:     result <= 128'b0;
                    endcase
                    state <= OUTPUT;
                end

                SURF_ACCESS: begin
                    // Surface load/store
                    if (opcode_r == `OP_SULD) begin
                        // Surface load
                        if (!mem_req && mem_ready) begin
                            mem_req <= 1'b1;
                            mem_addr <= tex_base_addr +
                                       int_t * (tex_w_r * {11'b0, texel_size_r}) +
                                       int_s * {27'b0, texel_size_r};
                        end else if (mem_valid) begin
                            result <= mem_rdata;
                            state <= OUTPUT;
                        end
                    end else if (opcode_r == `OP_SUST) begin
                        // Surface store
                        if (!mem_req && mem_ready) begin
                            mem_req <= 1'b1;
                            mem_write <= 1'b1;
                            mem_addr <= tex_base_addr +
                                       int_t * (tex_w_r * {11'b0, texel_size_r}) +
                                       int_s * {27'b0, texel_size_r};
                            mem_wdata <= store_data;
                        end else if (mem_valid) begin
                            state <= OUTPUT;
                        end
                    end else begin
                        // SURED - surface reduction (atomic on surface)
                        state <= OUTPUT;
                    end
                end
                default: ; // lint: unreachable states
            endcase
        end
    end

endmodule

//============================================================================
// Texture Cache - Small L1 cache for texture data
//============================================================================
module texture_cache #(
    parameter CACHE_SIZE_KB = 16,
    parameter LINE_SIZE     = 64,   // 64 bytes per cache line
    parameter NUM_WAYS      = 4
)(
    input  wire        clk,
    input  wire        rst_n,

    // Request interface
    input  wire        req_valid,
    input  wire [31:0] req_addr,
    output reg         req_ready,
    output reg  [127:0] req_data,
    output reg         req_hit,

    // Memory interface (to L2)
    output reg         mem_req,
    output reg  [31:0] mem_addr,
    input  wire        mem_valid,
    input  wire [511:0] mem_data  // Full cache line
);

    localparam NUM_SETS = (CACHE_SIZE_KB * 1024) / (LINE_SIZE * NUM_WAYS);
    localparam SET_BITS = $clog2(NUM_SETS);
    localparam TAG_BITS = 32 - SET_BITS - 6;  // 6 bits for 64-byte offset

    // Cache storage
    reg [TAG_BITS-1:0] tag_array [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg                valid_array [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [511:0]        data_array [0:NUM_SETS-1][0:NUM_WAYS-1];
    reg [1:0]          lru_array [0:NUM_SETS-1];  // LRU per set

    wire [SET_BITS-1:0] req_set = req_addr[SET_BITS+5:6];
    wire [TAG_BITS-1:0] req_tag = req_addr[31:SET_BITS+6];
    wire [5:0]          req_offset = req_addr[5:0];

    // Hit detection
    wire [NUM_WAYS-1:0] way_hit;
    genvar w;
    generate
        for (w = 0; w < NUM_WAYS; w = w + 1) begin : hit_check
            assign way_hit[w] = valid_array[req_set][w] &&
                               (tag_array[req_set][w] == req_tag);
        end
    endgenerate

    wire cache_hit = |way_hit;

    // State machine
    localparam IDLE = 2'b00;
    localparam MISS = 2'b01;
    localparam FILL = 2'b10;

    reg [1:0] state;
    reg [1:0] replace_way;

    // Reset indices
    integer rst_i, rst_j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            req_ready <= 1'b1;
            req_hit <= 1'b0;
            mem_req <= 1'b0;
            for (rst_i = 0; rst_i < NUM_SETS; rst_i = rst_i + 1) begin
                for (rst_j = 0; rst_j < NUM_WAYS; rst_j = rst_j + 1) begin
                    valid_array[rst_i][rst_j] <= 1'b0;
                end
                lru_array[rst_i] <= 2'b0;
            end
        end else begin
            mem_req <= 1'b0;
            req_hit <= 1'b0;

            case (state)
                IDLE: begin
                    if (req_valid) begin
                        if (cache_hit) begin
                            // Cache hit
                            req_hit <= 1'b1;
                            // Extract 128 bits from cache line
                            case (req_offset[5:4])
                                2'b00: req_data <= data_array[req_set][way_hit[0] ? 0 :
                                                                      way_hit[1] ? 1 :
                                                                      way_hit[2] ? 2 : 3][127:0];
                                2'b01: req_data <= data_array[req_set][way_hit[0] ? 0 :
                                                                      way_hit[1] ? 1 :
                                                                      way_hit[2] ? 2 : 3][255:128];
                                2'b10: req_data <= data_array[req_set][way_hit[0] ? 0 :
                                                                      way_hit[1] ? 1 :
                                                                      way_hit[2] ? 2 : 3][383:256];
                                2'b11: req_data <= data_array[req_set][way_hit[0] ? 0 :
                                                                      way_hit[1] ? 1 :
                                                                      way_hit[2] ? 2 : 3][511:384];
                            endcase
                        end else begin
                            // Cache miss
                            req_ready <= 1'b0;
                            mem_req <= 1'b1;
                            mem_addr <= {req_addr[31:6], 6'b0};  // Align to line
                            replace_way <= lru_array[req_set];
                            state <= MISS;
                        end
                    end
                end

                MISS: begin
                    if (mem_valid) begin
                        // Fill cache line
                        data_array[req_set][replace_way] <= mem_data;
                        tag_array[req_set][replace_way] <= req_tag;
                        valid_array[req_set][replace_way] <= 1'b1;
                        lru_array[req_set] <= replace_way + 1;
                        state <= FILL;
                    end
                end

                FILL: begin
                    // Return data
                    req_hit <= 1'b1;
                    case (req_offset[5:4])
                        2'b00: req_data <= data_array[req_set][replace_way][127:0];
                        2'b01: req_data <= data_array[req_set][replace_way][255:128];
                        2'b10: req_data <= data_array[req_set][replace_way][383:256];
                        2'b11: req_data <= data_array[req_set][replace_way][511:384];
                    endcase
                    req_ready <= 1'b1;
                    state <= IDLE;
                end
                default: ; // lint: CASEINCOMPLETE
            endcase
        end
    end

endmodule
