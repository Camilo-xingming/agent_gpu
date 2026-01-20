//============================================================================
// RalphGPU - CHI (Coherent Hub Interface) Controller
// AMBA CHI-based interconnect for multi-GPU chiplet communication
//
// CHI Features:
// - Cache coherency protocol (MOESI-based)
// - Request/Response/Snoop/Data channels
// - Multi-chiplet scalability
// - Hardware cache line tracking
//
// Supported Operations:
// - ReadShared, ReadUnique, ReadNoSnp
// - WriteUniquePtl, WriteUniqueFull, WriteNoSnp
// - CleanShared, CleanInvalid, MakeInvalid
// - Snoop filtering for efficiency
//============================================================================

`timescale 1ns / 1ps

module chi_controller #(
    parameter NUM_CHIPLETS     = 4,           // Number of chiplets in system
    parameter CHIPLET_ID       = 0,           // This chiplet's ID
    parameter ADDR_WIDTH       = 48,          // Address width
    parameter DATA_WIDTH       = 512,         // Data bus width (cache line)
    parameter TXN_ID_WIDTH     = 12,          // Transaction ID width
    parameter NODE_ID_WIDTH    = 7,           // CHI Node ID width
    parameter NUM_SN_ENTRIES   = 256,         // Snoop filter entries
    parameter NUM_OUTSTANDING  = 64           // Max outstanding transactions
)(
    input  wire                     clk,
    input  wire                     rst_n,

    //------------------------------------------------------------------------
    // Local Request Interface (from L2 Cache/Memory Controller)
    //------------------------------------------------------------------------
    input  wire                     local_req_valid,
    output reg                      local_req_ready,
    input  wire [4:0]               local_req_opcode,  // CHI opcode
    input  wire [ADDR_WIDTH-1:0]    local_req_addr,
    input  wire [TXN_ID_WIDTH-1:0]  local_req_txn_id,
    input  wire [DATA_WIDTH-1:0]    local_req_data,
    input  wire                     local_req_excl,    // Exclusive access

    output reg                      local_resp_valid,
    input  wire                     local_resp_ready,
    output reg  [4:0]               local_resp_opcode,
    output reg  [TXN_ID_WIDTH-1:0]  local_resp_txn_id,
    output reg  [DATA_WIDTH-1:0]    local_resp_data,
    output reg  [1:0]               local_resp_result, // 00=OK, 01=EXOK, 10=NACK, 11=ERR

    //------------------------------------------------------------------------
    // CHI Request Channel (TX)
    //------------------------------------------------------------------------
    output reg                      chi_req_valid,
    input  wire                     chi_req_ready,
    output reg  [4:0]               chi_req_opcode,
    output reg  [ADDR_WIDTH-1:0]    chi_req_addr,
    output reg  [TXN_ID_WIDTH-1:0]  chi_req_txn_id,
    output reg  [NODE_ID_WIDTH-1:0] chi_req_src_id,
    output reg  [NODE_ID_WIDTH-1:0] chi_req_tgt_id,
    output reg                      chi_req_excl,

    //------------------------------------------------------------------------
    // CHI Response Channel (RX)
    //------------------------------------------------------------------------
    input  wire                     chi_resp_valid,
    output reg                      chi_resp_ready,
    input  wire [4:0]               chi_resp_opcode,
    input  wire [TXN_ID_WIDTH-1:0]  chi_resp_txn_id,
    input  wire [NODE_ID_WIDTH-1:0] chi_resp_src_id,
    input  wire [1:0]               chi_resp_result,

    //------------------------------------------------------------------------
    // CHI Snoop Channel (RX)
    //------------------------------------------------------------------------
    input  wire                     chi_snp_valid,
    output reg                      chi_snp_ready,
    input  wire [4:0]               chi_snp_opcode,
    input  wire [ADDR_WIDTH-1:0]    chi_snp_addr,
    input  wire [TXN_ID_WIDTH-1:0]  chi_snp_txn_id,
    input  wire [NODE_ID_WIDTH-1:0] chi_snp_src_id,

    //------------------------------------------------------------------------
    // CHI Snoop Response Channel (TX)
    //------------------------------------------------------------------------
    output reg                      chi_snp_resp_valid,
    input  wire                     chi_snp_resp_ready,
    output reg  [4:0]               chi_snp_resp_opcode,
    output reg  [TXN_ID_WIDTH-1:0]  chi_snp_resp_txn_id,
    output reg  [NODE_ID_WIDTH-1:0] chi_snp_resp_tgt_id,
    output reg  [DATA_WIDTH-1:0]    chi_snp_resp_data,
    output reg                      chi_snp_resp_has_data,

    //------------------------------------------------------------------------
    // CHI Data Channel (bidirectional)
    //------------------------------------------------------------------------
    output reg                      chi_data_valid,
    input  wire                     chi_data_ready,
    output reg  [DATA_WIDTH-1:0]    chi_data_data,
    output reg  [TXN_ID_WIDTH-1:0]  chi_data_txn_id,
    output reg  [NODE_ID_WIDTH-1:0] chi_data_tgt_id,

    input  wire                     chi_rxdata_valid,
    output reg                      chi_rxdata_ready,
    input  wire [DATA_WIDTH-1:0]    chi_rxdata_data,
    input  wire [TXN_ID_WIDTH-1:0]  chi_rxdata_txn_id,

    //------------------------------------------------------------------------
    // Snoop Interface to Local Cache
    //------------------------------------------------------------------------
    output reg                      snp_req_valid,
    input  wire                     snp_req_ready,
    output reg  [4:0]               snp_req_opcode,
    output reg  [ADDR_WIDTH-1:0]    snp_req_addr,

    input  wire                     snp_resp_valid,
    output reg                      snp_resp_ready,
    input  wire [2:0]               snp_resp_state,    // Cache line state
    input  wire [DATA_WIDTH-1:0]    snp_resp_data,
    input  wire                     snp_resp_has_data,

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    output reg  [31:0]              stat_req_sent,
    output reg  [31:0]              stat_req_rcvd,
    output reg  [31:0]              stat_snp_sent,
    output reg  [31:0]              stat_snp_rcvd,
    output reg  [31:0]              stat_data_transfers
);

    //------------------------------------------------------------------------
    // CHI Opcodes
    //------------------------------------------------------------------------
    localparam OP_READ_SHARED      = 5'h01;
    localparam OP_READ_UNIQUE      = 5'h02;
    localparam OP_READ_NO_SNP      = 5'h03;
    localparam OP_WRITE_UNIQUE_PTL = 5'h04;
    localparam OP_WRITE_UNIQUE_FULL= 5'h05;
    localparam OP_WRITE_NO_SNP     = 5'h06;
    localparam OP_CLEAN_SHARED     = 5'h07;
    localparam OP_CLEAN_INVALID    = 5'h08;
    localparam OP_MAKE_INVALID     = 5'h09;
    localparam OP_EVICT            = 5'h0A;

    // Snoop Opcodes
    localparam SNP_READ            = 5'h10;
    localparam SNP_CLEAN           = 5'h11;
    localparam SNP_CLEAN_INVALID   = 5'h12;
    localparam SNP_MAKE_INVALID    = 5'h13;
    localparam SNP_UNIQUE          = 5'h14;

    // Response Opcodes
    localparam RSP_COMP            = 5'h18;
    localparam RSP_COMP_DATA       = 5'h19;
    localparam RSP_COMP_ACK        = 5'h1A;
    localparam RSP_SNP_RESP        = 5'h1B;
    localparam RSP_SNP_RESP_DATA   = 5'h1C;

    //------------------------------------------------------------------------
    // Cache Line States (MOESI)
    //------------------------------------------------------------------------
    localparam STATE_INVALID   = 3'd0;
    localparam STATE_SHARED    = 3'd1;
    localparam STATE_EXCLUSIVE = 3'd2;
    localparam STATE_MODIFIED  = 3'd3;
    localparam STATE_OWNED     = 3'd4;

    //------------------------------------------------------------------------
    // Outstanding Transaction Tracker
    //------------------------------------------------------------------------
    localparam TXN_ENTRY_W = ADDR_WIDTH + TXN_ID_WIDTH + 5 + 1;  // addr + id + op + valid

    reg [TXN_ENTRY_W-1:0] txn_table [0:NUM_OUTSTANDING-1];
    reg [5:0] txn_head, txn_tail;
    reg [6:0] txn_count;

    wire txn_full = (txn_count >= NUM_OUTSTANDING);
    wire txn_empty = (txn_count == 0);

    //------------------------------------------------------------------------
    // Snoop Filter (tracks remote cache states)
    //------------------------------------------------------------------------
    localparam FILTER_TAG_W = ADDR_WIDTH - 6;  // Exclude offset bits
    localparam FILTER_IDX_W = $clog2(NUM_SN_ENTRIES);

    reg [FILTER_TAG_W-1:0] snp_filter_tag [0:NUM_SN_ENTRIES-1];
    reg [NUM_CHIPLETS-1:0] snp_filter_presence [0:NUM_SN_ENTRIES-1];
    reg [2:0]              snp_filter_state [0:NUM_SN_ENTRIES-1];
    reg                    snp_filter_valid [0:NUM_SN_ENTRIES-1];

    //------------------------------------------------------------------------
    // State Machine
    //------------------------------------------------------------------------
    localparam ST_IDLE           = 4'd0;
    localparam ST_SEND_REQ       = 4'd1;
    localparam ST_WAIT_RESP      = 4'd2;
    localparam ST_WAIT_DATA      = 4'd3;
    localparam ST_PROCESS_SNP    = 4'd4;
    localparam ST_SEND_SNP_RESP  = 4'd5;
    localparam ST_SEND_DATA      = 4'd6;
    localparam ST_LOCAL_RESP     = 4'd7;

    reg [3:0] state, next_state;

    //------------------------------------------------------------------------
    // Request Processing State
    //------------------------------------------------------------------------
    reg [4:0]               pending_req_opcode;
    reg [ADDR_WIDTH-1:0]    pending_req_addr;
    reg [TXN_ID_WIDTH-1:0]  pending_req_txn_id;
    reg [DATA_WIDTH-1:0]    pending_req_data;
    reg                     pending_req_excl;
    reg                     pending_req_valid;

    //------------------------------------------------------------------------
    // Snoop Processing State
    //------------------------------------------------------------------------
    reg [4:0]               pending_snp_opcode;
    reg [ADDR_WIDTH-1:0]    pending_snp_addr;
    reg [TXN_ID_WIDTH-1:0]  pending_snp_txn_id;
    reg [NODE_ID_WIDTH-1:0] pending_snp_src_id;
    reg                     pending_snp_valid;

    //------------------------------------------------------------------------
    // Response/Data State
    //------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0]    response_data;
    reg [1:0]               response_result;
    reg                     response_has_data;

    //------------------------------------------------------------------------
    // Helper: Calculate target node from address
    //------------------------------------------------------------------------
    function [NODE_ID_WIDTH-1:0] addr_to_node;
        input [ADDR_WIDTH-1:0] addr;
        begin
            // Simple address interleaving: use address bits to determine home node
            addr_to_node = addr[11:5] % NUM_CHIPLETS;
        end
    endfunction

    //------------------------------------------------------------------------
    // Helper: Snoop filter lookup
    //------------------------------------------------------------------------
    wire [FILTER_IDX_W-1:0] snp_filter_idx = pending_snp_addr[6 +: FILTER_IDX_W];
    wire snp_filter_hit = snp_filter_valid[snp_filter_idx] &&
                          (snp_filter_tag[snp_filter_idx] == pending_snp_addr[ADDR_WIDTH-1:6]);

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
                if (pending_snp_valid) begin
                    next_state = ST_PROCESS_SNP;
                end else if (pending_req_valid && !txn_full) begin
                    next_state = ST_SEND_REQ;
                end
            end

            ST_SEND_REQ: begin
                if (chi_req_valid && chi_req_ready) begin
                    // Determine if we need data back
                    case (pending_req_opcode)
                        OP_READ_SHARED, OP_READ_UNIQUE, OP_READ_NO_SNP:
                            next_state = ST_WAIT_DATA;
                        default:
                            next_state = ST_WAIT_RESP;
                    endcase
                end
            end

            ST_WAIT_RESP: begin
                if (chi_resp_valid) begin
                    if (chi_resp_opcode == RSP_COMP_DATA) begin
                        next_state = ST_WAIT_DATA;
                    end else begin
                        next_state = ST_LOCAL_RESP;
                    end
                end
            end

            ST_WAIT_DATA: begin
                if (chi_rxdata_valid) begin
                    next_state = ST_LOCAL_RESP;
                end
            end

            ST_PROCESS_SNP: begin
                if (snp_req_ready) begin
                    next_state = ST_SEND_SNP_RESP;
                end
            end

            ST_SEND_SNP_RESP: begin
                if (snp_resp_valid) begin
                    if (snp_resp_has_data) begin
                        next_state = ST_SEND_DATA;
                    end else if (chi_snp_resp_ready) begin
                        next_state = ST_IDLE;
                    end
                end
            end

            ST_SEND_DATA: begin
                if (chi_data_ready) begin
                    next_state = ST_IDLE;
                end
            end

            ST_LOCAL_RESP: begin
                if (local_resp_ready) begin
                    next_state = ST_IDLE;
                end
            end
        endcase
    end

    //------------------------------------------------------------------------
    // Main Datapath
    //------------------------------------------------------------------------
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset outputs
            local_req_ready <= 1'b1;
            local_resp_valid <= 1'b0;
            local_resp_opcode <= 5'd0;
            local_resp_txn_id <= {TXN_ID_WIDTH{1'b0}};
            local_resp_data <= {DATA_WIDTH{1'b0}};
            local_resp_result <= 2'b00;

            chi_req_valid <= 1'b0;
            chi_req_opcode <= 5'd0;
            chi_req_addr <= {ADDR_WIDTH{1'b0}};
            chi_req_txn_id <= {TXN_ID_WIDTH{1'b0}};
            chi_req_src_id <= CHIPLET_ID[NODE_ID_WIDTH-1:0];
            chi_req_tgt_id <= {NODE_ID_WIDTH{1'b0}};
            chi_req_excl <= 1'b0;

            chi_resp_ready <= 1'b1;
            chi_snp_ready <= 1'b1;
            chi_rxdata_ready <= 1'b1;

            chi_snp_resp_valid <= 1'b0;
            chi_snp_resp_opcode <= 5'd0;
            chi_snp_resp_txn_id <= {TXN_ID_WIDTH{1'b0}};
            chi_snp_resp_tgt_id <= {NODE_ID_WIDTH{1'b0}};
            chi_snp_resp_data <= {DATA_WIDTH{1'b0}};
            chi_snp_resp_has_data <= 1'b0;

            chi_data_valid <= 1'b0;
            chi_data_data <= {DATA_WIDTH{1'b0}};
            chi_data_txn_id <= {TXN_ID_WIDTH{1'b0}};
            chi_data_tgt_id <= {NODE_ID_WIDTH{1'b0}};

            snp_req_valid <= 1'b0;
            snp_req_opcode <= 5'd0;
            snp_req_addr <= {ADDR_WIDTH{1'b0}};
            snp_resp_ready <= 1'b1;

            // Reset internal state
            pending_req_valid <= 1'b0;
            pending_snp_valid <= 1'b0;

            txn_head <= 6'd0;
            txn_tail <= 6'd0;
            txn_count <= 7'd0;

            response_data <= {DATA_WIDTH{1'b0}};
            response_result <= 2'b00;
            response_has_data <= 1'b0;

            // Reset statistics
            stat_req_sent <= 32'd0;
            stat_req_rcvd <= 32'd0;
            stat_snp_sent <= 32'd0;
            stat_snp_rcvd <= 32'd0;
            stat_data_transfers <= 32'd0;

            // Reset snoop filter
            for (i = 0; i < NUM_SN_ENTRIES; i = i + 1) begin
                snp_filter_valid[i] <= 1'b0;
            end
        end else begin
            // Default: deassert valid signals after handshake
            if (chi_req_valid && chi_req_ready) chi_req_valid <= 1'b0;
            if (chi_snp_resp_valid && chi_snp_resp_ready) chi_snp_resp_valid <= 1'b0;
            if (chi_data_valid && chi_data_ready) chi_data_valid <= 1'b0;
            if (local_resp_valid && local_resp_ready) local_resp_valid <= 1'b0;
            if (snp_req_valid && snp_req_ready) snp_req_valid <= 1'b0;

            // Accept local requests
            if (local_req_valid && local_req_ready && !pending_req_valid) begin
                pending_req_opcode <= local_req_opcode;
                pending_req_addr <= local_req_addr;
                pending_req_txn_id <= local_req_txn_id;
                pending_req_data <= local_req_data;
                pending_req_excl <= local_req_excl;
                pending_req_valid <= 1'b1;
                local_req_ready <= 1'b0;
            end

            // Accept incoming snoops
            if (chi_snp_valid && chi_snp_ready && !pending_snp_valid) begin
                pending_snp_opcode <= chi_snp_opcode;
                pending_snp_addr <= chi_snp_addr;
                pending_snp_txn_id <= chi_snp_txn_id;
                pending_snp_src_id <= chi_snp_src_id;
                pending_snp_valid <= 1'b1;
                chi_snp_ready <= 1'b0;
                stat_snp_rcvd <= stat_snp_rcvd + 1;
            end

            case (state)
                ST_IDLE: begin
                    local_req_ready <= !pending_req_valid && !txn_full;
                    chi_snp_ready <= !pending_snp_valid;
                end

                ST_SEND_REQ: begin
                    chi_req_valid <= 1'b1;
                    chi_req_opcode <= pending_req_opcode;
                    chi_req_addr <= pending_req_addr;
                    chi_req_txn_id <= pending_req_txn_id;
                    chi_req_src_id <= CHIPLET_ID[NODE_ID_WIDTH-1:0];
                    chi_req_tgt_id <= addr_to_node(pending_req_addr);
                    chi_req_excl <= pending_req_excl;

                    if (chi_req_ready) begin
                        // Record in transaction table
                        txn_table[txn_tail] <= {pending_req_addr, pending_req_txn_id,
                                                pending_req_opcode, 1'b1};
                        txn_tail <= (txn_tail + 1) % NUM_OUTSTANDING;
                        txn_count <= txn_count + 1;
                        stat_req_sent <= stat_req_sent + 1;
                    end
                end

                ST_WAIT_RESP: begin
                    if (chi_resp_valid && chi_resp_txn_id == pending_req_txn_id) begin
                        response_result <= chi_resp_result;
                        chi_resp_ready <= 1'b1;
                    end
                end

                ST_WAIT_DATA: begin
                    if (chi_rxdata_valid && chi_rxdata_txn_id == pending_req_txn_id) begin
                        response_data <= chi_rxdata_data;
                        response_has_data <= 1'b1;
                        stat_data_transfers <= stat_data_transfers + 1;
                    end
                end

                ST_PROCESS_SNP: begin
                    // Forward snoop to local cache
                    snp_req_valid <= 1'b1;
                    snp_req_opcode <= pending_snp_opcode;
                    snp_req_addr <= pending_snp_addr;
                end

                ST_SEND_SNP_RESP: begin
                    if (snp_resp_valid) begin
                        chi_snp_resp_valid <= 1'b1;
                        chi_snp_resp_opcode <= snp_resp_has_data ? RSP_SNP_RESP_DATA : RSP_SNP_RESP;
                        chi_snp_resp_txn_id <= pending_snp_txn_id;
                        chi_snp_resp_tgt_id <= pending_snp_src_id;
                        chi_snp_resp_data <= snp_resp_data;
                        chi_snp_resp_has_data <= snp_resp_has_data;
                        stat_snp_sent <= stat_snp_sent + 1;
                    end
                end

                ST_SEND_DATA: begin
                    chi_data_valid <= 1'b1;
                    chi_data_data <= snp_resp_data;
                    chi_data_txn_id <= pending_snp_txn_id;
                    chi_data_tgt_id <= pending_snp_src_id;

                    if (chi_data_ready) begin
                        pending_snp_valid <= 1'b0;
                        chi_snp_ready <= 1'b1;
                        stat_data_transfers <= stat_data_transfers + 1;
                    end
                end

                ST_LOCAL_RESP: begin
                    local_resp_valid <= 1'b1;
                    local_resp_opcode <= RSP_COMP_DATA;
                    local_resp_txn_id <= pending_req_txn_id;
                    local_resp_data <= response_data;
                    local_resp_result <= response_result;

                    if (local_resp_ready) begin
                        // Complete transaction
                        pending_req_valid <= 1'b0;
                        local_req_ready <= 1'b1;

                        // Update snoop filter
                        snp_filter_tag[pending_req_addr[6 +: FILTER_IDX_W]] <= pending_req_addr[ADDR_WIDTH-1:6];
                        snp_filter_valid[pending_req_addr[6 +: FILTER_IDX_W]] <= 1'b1;
                        snp_filter_presence[pending_req_addr[6 +: FILTER_IDX_W]][CHIPLET_ID] <= 1'b1;

                        // Remove from transaction table
                        txn_head <= (txn_head + 1) % NUM_OUTSTANDING;
                        txn_count <= txn_count - 1;
                    end
                end
            endcase
        end
    end

endmodule
