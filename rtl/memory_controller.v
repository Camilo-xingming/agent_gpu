//============================================================================
// RalphGPU - Memory Controller Interface
// Simplified CDC-safe controller with async FIFOs and a basic memory model.
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"
`include "memory_config.vh"

module memory_controller #(
    parameter DATA_WIDTH        = `MEM_DATA_WIDTH,
    parameter NUM_CHANNELS      = `MEM_NUM_CHANNELS,
    parameter BURST_LENGTH      = `MEM_BURST_LENGTH,
    parameter ADDR_WIDTH        = 32,
    parameter REQ_QUEUE_DEPTH   = `MEM_REQ_QUEUE_DEPTH
)(
    input  wire                         clk,
    input  wire                         mem_clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // L2 Cache Interface (GPU side)
    //------------------------------------------------------------------------
    input  wire                         l2_req_valid,
    input  wire                         l2_req_write,
    input  wire [ADDR_WIDTH-1:0]        l2_req_addr,
    input  wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_req_wdata,
    input  wire [DATA_WIDTH*BURST_LENGTH/8-1:0] l2_req_wmask,
    output wire                         l2_req_ready,
    output wire                         l2_resp_valid,
    output wire [DATA_WIDTH*BURST_LENGTH-1:0] l2_resp_rdata,

    //------------------------------------------------------------------------
    // DDR/HBM Physical Interface (Memory side)
    //------------------------------------------------------------------------
    output wire [NUM_CHANNELS-1:0]      mem_cs_n,
    output wire [NUM_CHANNELS-1:0]      mem_ras_n,
    output wire [NUM_CHANNELS-1:0]      mem_cas_n,
    output wire [NUM_CHANNELS-1:0]      mem_we_n,
    output wire [NUM_CHANNELS*17-1:0]   mem_addr,
    output wire [NUM_CHANNELS*3-1:0]    mem_ba,
    output wire [NUM_CHANNELS*2-1:0]    mem_bg,

    output wire [NUM_CHANNELS*DATA_WIDTH-1:0]   mem_dq_out,
    input  wire [NUM_CHANNELS*DATA_WIDTH-1:0]   mem_dq_in,
    output wire [NUM_CHANNELS-1:0]              mem_dq_oe,
    output wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_out,
    input  wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dqs_in,
    output wire [NUM_CHANNELS*DATA_WIDTH/8-1:0] mem_dm,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_read_count,
    output wire [31:0]                  stat_write_count,
    output wire [31:0]                  stat_row_hits,
    output wire [31:0]                  stat_row_misses
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam BURST_BITS  = DATA_WIDTH * BURST_LENGTH;
    localparam BURST_BYTES = BURST_BITS / 8;
    localparam FIFO_DEPTH  = (REQ_QUEUE_DEPTH < 2) ? 2 : REQ_QUEUE_DEPTH;
    localparam PTR_W       = $clog2(FIFO_DEPTH);
    localparam REQ_FIFO_W  = 1 + ADDR_WIDTH + BURST_BITS + BURST_BYTES;
    localparam RESP_FIFO_W = BURST_BITS;

    localparam MEM_DEPTH       = 1024;
    localparam MEM_INDEX_W     = $clog2(MEM_DEPTH);
    localparam BURST_OFFSET_W  = $clog2(BURST_BYTES);

    function [PTR_W:0] bin2gray;
        input [PTR_W:0] bin;
        begin
            bin2gray = (bin >> 1) ^ bin;
        end
    endfunction

    //------------------------------------------------------------------------
    // Request FIFO (clk -> mem_clk)
    //------------------------------------------------------------------------
    reg [REQ_FIFO_W-1:0] req_fifo_mem [0:FIFO_DEPTH-1];
    reg [PTR_W:0] req_wr_ptr_bin;
    reg [PTR_W:0] req_wr_ptr_gray;
    reg [PTR_W:0] req_rd_ptr_bin;
    reg [PTR_W:0] req_rd_ptr_gray;
    reg [PTR_W:0] req_rd_ptr_gray_sync1;
    reg [PTR_W:0] req_rd_ptr_gray_sync2;
    reg [PTR_W:0] req_wr_ptr_gray_sync1;
    reg [PTR_W:0] req_wr_ptr_gray_sync2;

    wire [PTR_W:0] req_wr_ptr_gray_next = bin2gray(req_wr_ptr_bin + 1'b1);
    wire req_fifo_full = (req_wr_ptr_gray_next ==
                          {~req_rd_ptr_gray_sync2[PTR_W:PTR_W-1],
                           req_rd_ptr_gray_sync2[PTR_W-2:0]});
    wire req_fifo_empty_mem = (req_rd_ptr_gray == req_wr_ptr_gray_sync2);

    assign l2_req_ready = !req_fifo_full;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_wr_ptr_bin <= {PTR_W+1{1'b0}};
            req_wr_ptr_gray <= {PTR_W+1{1'b0}};
        end else if (l2_req_valid && l2_req_ready) begin
            req_fifo_mem[req_wr_ptr_bin[PTR_W-1:0]] <=
                {l2_req_write, l2_req_addr, l2_req_wdata, l2_req_wmask};
            req_wr_ptr_bin <= req_wr_ptr_bin + 1'b1;
            req_wr_ptr_gray <= req_wr_ptr_gray_next;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_rd_ptr_gray_sync1 <= {PTR_W+1{1'b0}};
            req_rd_ptr_gray_sync2 <= {PTR_W+1{1'b0}};
        end else begin
            req_rd_ptr_gray_sync1 <= req_rd_ptr_gray;
            req_rd_ptr_gray_sync2 <= req_rd_ptr_gray_sync1;
        end
    end

    always @(posedge mem_clk or negedge rst_n) begin
        if (!rst_n) begin
            req_wr_ptr_gray_sync1 <= {PTR_W+1{1'b0}};
            req_wr_ptr_gray_sync2 <= {PTR_W+1{1'b0}};
        end else begin
            req_wr_ptr_gray_sync1 <= req_wr_ptr_gray;
            req_wr_ptr_gray_sync2 <= req_wr_ptr_gray_sync1;
        end
    end

    //------------------------------------------------------------------------
    // Response FIFO (mem_clk -> clk)
    //------------------------------------------------------------------------
    reg [RESP_FIFO_W-1:0] resp_fifo_mem [0:FIFO_DEPTH-1];
    reg [PTR_W:0] resp_wr_ptr_bin;
    reg [PTR_W:0] resp_wr_ptr_gray;
    reg [PTR_W:0] resp_rd_ptr_bin;
    reg [PTR_W:0] resp_rd_ptr_gray;
    reg [PTR_W:0] resp_rd_ptr_gray_sync1;
    reg [PTR_W:0] resp_rd_ptr_gray_sync2;
    reg [PTR_W:0] resp_wr_ptr_gray_sync1;
    reg [PTR_W:0] resp_wr_ptr_gray_sync2;

    wire [PTR_W:0] resp_wr_ptr_gray_next = bin2gray(resp_wr_ptr_bin + 1'b1);
    wire resp_fifo_full = (resp_wr_ptr_gray_next ==
                           {~resp_rd_ptr_gray_sync2[PTR_W:PTR_W-1],
                            resp_rd_ptr_gray_sync2[PTR_W-2:0]});
    wire resp_fifo_empty = (resp_rd_ptr_gray == resp_wr_ptr_gray_sync2);

    reg l2_resp_valid_reg;
    reg [RESP_FIFO_W-1:0] l2_resp_rdata_reg;
    assign l2_resp_valid = l2_resp_valid_reg;
    assign l2_resp_rdata = l2_resp_rdata_reg;

    always @(posedge mem_clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_rd_ptr_gray_sync1 <= {PTR_W+1{1'b0}};
            resp_rd_ptr_gray_sync2 <= {PTR_W+1{1'b0}};
        end else begin
            resp_rd_ptr_gray_sync1 <= resp_rd_ptr_gray;
            resp_rd_ptr_gray_sync2 <= resp_rd_ptr_gray_sync1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_wr_ptr_gray_sync1 <= {PTR_W+1{1'b0}};
            resp_wr_ptr_gray_sync2 <= {PTR_W+1{1'b0}};
        end else begin
            resp_wr_ptr_gray_sync1 <= resp_wr_ptr_gray;
            resp_wr_ptr_gray_sync2 <= resp_wr_ptr_gray_sync1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_rd_ptr_bin <= {PTR_W+1{1'b0}};
            resp_rd_ptr_gray <= {PTR_W+1{1'b0}};
            l2_resp_valid_reg <= 1'b0;
            l2_resp_rdata_reg <= {RESP_FIFO_W{1'b0}};
        end else begin
            l2_resp_valid_reg <= 1'b0;
            if (!resp_fifo_empty) begin
                l2_resp_rdata_reg <= resp_fifo_mem[resp_rd_ptr_bin[PTR_W-1:0]];
                l2_resp_valid_reg <= 1'b1;
                resp_rd_ptr_bin <= resp_rd_ptr_bin + 1'b1;
                resp_rd_ptr_gray <= bin2gray(resp_rd_ptr_bin + 1'b1);
            end
        end
    end

    //------------------------------------------------------------------------
    // Memory model and request handling (mem_clk domain)
    //------------------------------------------------------------------------
    `ifndef SYNTHESIS
    reg [BURST_BITS-1:0] mem_array [0:MEM_DEPTH-1];
`endif

    wire [REQ_FIFO_W-1:0] req_fifo_rdata = req_fifo_mem[req_rd_ptr_bin[PTR_W-1:0]];
    wire req_fifo_write = req_fifo_rdata[REQ_FIFO_W-1];
    wire [ADDR_WIDTH-1:0] req_fifo_addr = req_fifo_rdata[REQ_FIFO_W-2 -: ADDR_WIDTH];
    wire [BURST_BITS-1:0] req_fifo_wdata =
        req_fifo_rdata[REQ_FIFO_W-2-ADDR_WIDTH -: BURST_BITS];
    wire [BURST_BYTES-1:0] req_fifo_wmask = req_fifo_rdata[BURST_BYTES-1:0];

    wire [MEM_INDEX_W-1:0] mem_index =
        req_fifo_addr[BURST_OFFSET_W +: MEM_INDEX_W];

    wire req_pop = !req_fifo_empty_mem && (req_fifo_write || !resp_fifo_full);

    integer mem_i;
    integer mem_b;
    always @(posedge mem_clk or negedge rst_n) begin
        if (!rst_n) begin
            req_rd_ptr_bin <= {PTR_W+1{1'b0}};
            req_rd_ptr_gray <= {PTR_W+1{1'b0}};
            resp_wr_ptr_bin <= {PTR_W+1{1'b0}};
            resp_wr_ptr_gray <= {PTR_W+1{1'b0}};
`ifndef SYNTHESIS
            for (mem_i = 0; mem_i < MEM_DEPTH; mem_i = mem_i + 1) begin
                mem_array[mem_i] <= {BURST_BITS{1'b0}};
            end
`endif
        end else begin
            if (req_pop) begin
                if (req_fifo_write) begin
                    for (mem_b = 0; mem_b < BURST_BYTES; mem_b = mem_b + 1) begin
                        if (req_fifo_wmask[mem_b]) begin
`ifndef SYNTHESIS
                            mem_array[mem_index][mem_b*8 +: 8] <= req_fifo_wdata[mem_b*8 +: 8];
`endif
                        end
                    end
                end else begin
`ifndef SYNTHESIS
                    resp_fifo_mem[resp_wr_ptr_bin[PTR_W-1:0]] <= mem_array[mem_index];
`else
                    resp_fifo_mem[resp_wr_ptr_bin[PTR_W-1:0]] <= {BURST_BITS{1'b0}};
`endif
                    resp_wr_ptr_bin <= resp_wr_ptr_bin + 1'b1;
                    resp_wr_ptr_gray <= bin2gray(resp_wr_ptr_bin + 1'b1);
                end
                req_rd_ptr_bin <= req_rd_ptr_bin + 1'b1;
                req_rd_ptr_gray <= bin2gray(req_rd_ptr_bin + 1'b1);
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] read_cnt;
    reg [31:0] write_cnt;
    reg [31:0] row_hit_cnt;
    reg [31:0] row_miss_cnt;

    assign stat_read_count  = read_cnt;
    assign stat_write_count = write_cnt;
    assign stat_row_hits    = row_hit_cnt;
    assign stat_row_misses  = row_miss_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_cnt <= 0;
            write_cnt <= 0;
            row_hit_cnt <= 0;
            row_miss_cnt <= 0;
        end else begin
            if (l2_req_valid && l2_req_ready) begin
                if (l2_req_write)
                    write_cnt <= write_cnt + 1'b1;
                else
                    read_cnt <= read_cnt + 1'b1;
            end
        end
    end

    //------------------------------------------------------------------------
    // Physical interface idle (not modeled)
    //------------------------------------------------------------------------
    assign mem_cs_n  = {NUM_CHANNELS{1'b1}};
    assign mem_ras_n = {NUM_CHANNELS{1'b1}};
    assign mem_cas_n = {NUM_CHANNELS{1'b1}};
    assign mem_we_n  = {NUM_CHANNELS{1'b1}};
    assign mem_addr  = {NUM_CHANNELS*17{1'b0}};
    assign mem_ba    = {NUM_CHANNELS*3{1'b0}};
    assign mem_bg    = {NUM_CHANNELS*2{1'b0}};
    assign mem_dq_out = {NUM_CHANNELS*DATA_WIDTH{1'b0}};
    assign mem_dq_oe  = {NUM_CHANNELS{1'b0}};
    assign mem_dqs_out = {NUM_CHANNELS*DATA_WIDTH/8{1'b0}};
    assign mem_dm = {NUM_CHANNELS*DATA_WIDTH/8{1'b0}};

endmodule
