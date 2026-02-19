//============================================================================
// RalphGPU - Reusable AXI4 Slave Memory Model for Testbenches
//============================================================================
// Usage:
//   1. Instantiate in your testbench
//   2. Connect AXI signals to DUT's master port
//   3. Pre-initialize memory via direct array access: axi_slave.memory[idx]
//   4. Read results via direct array access
//
// Features:
//   - Single-beat reads/writes (len=0, burst signals ignored)
//   - Handles simultaneous AW+W (common in simple masters)
//   - Configurable memory size, data width, address width
//   - Optional verbose logging (MEM_VERBOSE parameter)
//
// Limitations:
//   - No burst support (awlen/arlen ignored, single-beat only)
//   - No byte-lane strobes (wstrb ignored, full-word writes only)
//============================================================================

`timescale 1ns / 1ps

module tb_axi_slave #(
    parameter AXI_DATA_WIDTH = 32,
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_ID_WIDTH   = 4,
    parameter MEM_DEPTH      = 8192,
    parameter MEM_VERBOSE    = 1,
    parameter ADDR_LSB       = $clog2(AXI_DATA_WIDTH/8)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // AXI Write Address Channel
    input  wire [AXI_ID_WIDTH-1:0]      s_axi_awid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_awaddr,
    input  wire [7:0]                   s_axi_awlen,
    input  wire [2:0]                   s_axi_awsize,
    input  wire [1:0]                   s_axi_awburst,
    input  wire                         s_axi_awvalid,
    output reg                          s_axi_awready,

    // AXI Write Data Channel
    input  wire [AXI_DATA_WIDTH-1:0]    s_axi_wdata,
    input  wire [AXI_DATA_WIDTH/8-1:0]  s_axi_wstrb,
    input  wire                         s_axi_wlast,
    input  wire                         s_axi_wvalid,
    output reg                          s_axi_wready,

    // AXI Write Response Channel
    output reg  [AXI_ID_WIDTH-1:0]      s_axi_bid,
    output reg  [1:0]                   s_axi_bresp,
    output reg                          s_axi_bvalid,
    input  wire                         s_axi_bready,

    // AXI Read Address Channel
    input  wire [AXI_ID_WIDTH-1:0]      s_axi_arid,
    input  wire [AXI_ADDR_WIDTH-1:0]    s_axi_araddr,
    input  wire [7:0]                   s_axi_arlen,
    input  wire [2:0]                   s_axi_arsize,
    input  wire [1:0]                   s_axi_arburst,
    input  wire                         s_axi_arvalid,
    output reg                          s_axi_arready,

    // AXI Read Data Channel
    output reg  [AXI_ID_WIDTH-1:0]      s_axi_rid,
    output reg  [AXI_DATA_WIDTH-1:0]    s_axi_rdata,
    output reg  [1:0]                   s_axi_rresp,
    output reg                          s_axi_rlast,
    output reg                          s_axi_rvalid,
    input  wire                         s_axi_rready
);

    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused_ok = &{s_axi_awlen, s_axi_awsize, s_axi_awburst,
                        s_axi_wstrb, s_axi_wlast,
                        s_axi_arlen, s_axi_arsize, s_axi_arburst};
    /* verilator lint_on UNUSEDSIGNAL */

    //------------------------------------------------------------------------
    // Memory Array
    //------------------------------------------------------------------------
    reg [AXI_DATA_WIDTH-1:0] memory [0:MEM_DEPTH-1];

    //------------------------------------------------------------------------
    // Internal State
    //------------------------------------------------------------------------
    reg [AXI_ADDR_WIDTH-1:0] pending_write_addr;
    reg [AXI_ID_WIDTH-1:0]   pending_write_id;

    //------------------------------------------------------------------------
    // Address to Word Index
    //------------------------------------------------------------------------
    localparam IDX_WIDTH = $clog2(MEM_DEPTH);

    function automatic [IDX_WIDTH-1:0] addr_to_idx;
        input [AXI_ADDR_WIDTH-1:0] addr;
        begin
            addr_to_idx = addr[ADDR_LSB +: IDX_WIDTH];
        end
    endfunction

    //------------------------------------------------------------------------
    // AXI Logic - Matches original tb_vector_add.v behavior
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            s_axi_wready  <= 1'b1;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            s_axi_bid     <= '0;
            s_axi_arready <= 1'b1;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rlast   <= 1'b0;
            s_axi_rid     <= '0;
            s_axi_rdata   <= '0;
            pending_write_addr <= '0;
            pending_write_id   <= '0;
        end else begin
            // Capture write address (can happen same cycle as data)
            if (s_axi_awvalid && s_axi_awready) begin
                pending_write_addr <= s_axi_awaddr;
                pending_write_id   <= s_axi_awid;
            end

            // Write data - use captured address or current address if same cycle
            if (s_axi_wvalid && s_axi_wready) begin
                // If AW fires same cycle, use s_axi_awaddr directly
                // Otherwise use pending_write_addr from previous cycle
                if (s_axi_awvalid && s_axi_awready) begin
                    memory[addr_to_idx(s_axi_awaddr)] <= s_axi_wdata;
                    s_axi_bid <= s_axi_awid;
                    if (MEM_VERBOSE) begin
                        $display("[AXI_SLAVE] Write: addr=0x%08X idx=%0d data=0x%08X",
                                 s_axi_awaddr, addr_to_idx(s_axi_awaddr), s_axi_wdata);
                    end
                end else begin
                    memory[addr_to_idx(pending_write_addr)] <= s_axi_wdata;
                    s_axi_bid <= pending_write_id;
                    if (MEM_VERBOSE) begin
                        $display("[AXI_SLAVE] Write: addr=0x%08X idx=%0d data=0x%08X",
                                 pending_write_addr, addr_to_idx(pending_write_addr), s_axi_wdata);
                    end
                end
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            // Read
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rid    <= s_axi_arid;
                s_axi_rdata  <= memory[addr_to_idx(s_axi_araddr)];
                s_axi_rlast  <= 1'b1;
                s_axi_rresp  <= 2'b00;
                if (MEM_VERBOSE) begin
                    $display("[AXI_SLAVE] Read:  addr=0x%08X idx=%0d data=0x%08X",
                             s_axi_araddr, addr_to_idx(s_axi_araddr),
                             memory[addr_to_idx(s_axi_araddr)]);
                end
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                s_axi_rlast  <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Helper Tasks
    //------------------------------------------------------------------------
    task mem_init;
        input [31:0] start_idx;
        input [31:0] count;
        input [AXI_DATA_WIDTH-1:0] pattern;
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                memory[start_idx + i] = pattern + i;
            end
        end
    endtask

    task mem_clear;
        input [AXI_DATA_WIDTH-1:0] value;
        integer i;
        begin
            for (i = 0; i < MEM_DEPTH; i = i + 1) begin
                memory[i] = value;
            end
        end
    endtask

endmodule
