//============================================================================
// RalphGPU - Enhanced TLB with Page Walker and Fault Handling
// Features:
//   - Two-level TLB (L1 per-SM, L2 shared)
//   - Hardware page table walker
//   - Page fault handling and reporting
//   - Multiple page sizes (4KB, 2MB, 1GB)
//   - ASID/VMID support
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tlb_enhanced #(
    parameter NUM_SMS           = 4,
    parameter VADDR_WIDTH       = 48,           // Virtual address width
    parameter PADDR_WIDTH       = 40,           // Physical address width
    parameter L1_ENTRIES        = 32,           // Per-SM L1 TLB entries
    parameter L1_WAYS           = 4,
    parameter L2_ENTRIES        = 512,          // Shared L2 TLB entries
    parameter L2_WAYS           = 8,
    parameter ASID_WIDTH        = 16,           // Address Space ID
    parameter PAGE_LEVELS       = 4             // Page table levels (like x86-64)
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Translation Request Interface (per SM)
    //------------------------------------------------------------------------
    input  wire [NUM_SMS-1:0]           req_valid,
    input  wire [VADDR_WIDTH*NUM_SMS-1:0] req_vaddr,
    input  wire [NUM_SMS-1:0]           req_write,
    input  wire [ASID_WIDTH*NUM_SMS-1:0] req_asid,
    output wire [NUM_SMS-1:0]           req_ready,

    output wire [NUM_SMS-1:0]           resp_valid,
    output wire [PADDR_WIDTH*NUM_SMS-1:0] resp_paddr,
    output wire [NUM_SMS-1:0]           resp_fault,
    output wire [3:0]                   resp_fault_code [0:NUM_SMS-1],

    //------------------------------------------------------------------------
    // Page Table Walk Interface (to Memory)
    //------------------------------------------------------------------------
    output wire                         ptw_req_valid,
    output wire [PADDR_WIDTH-1:0]       ptw_req_addr,
    input  wire                         ptw_req_ready,
    input  wire                         ptw_resp_valid,
    input  wire [63:0]                  ptw_resp_data,

    //------------------------------------------------------------------------
    // Page Table Base Register
    //------------------------------------------------------------------------
    input  wire [PADDR_WIDTH-1:0]       page_table_base,
    input  wire [ASID_WIDTH-1:0]        current_asid,

    //------------------------------------------------------------------------
    // TLB Management
    //------------------------------------------------------------------------
    input  wire                         invalidate_all,
    input  wire                         invalidate_asid,
    input  wire [ASID_WIDTH-1:0]        invalidate_asid_val,
    input  wire                         invalidate_page,
    input  wire [VADDR_WIDTH-1:0]       invalidate_vaddr,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_l1_hits,
    output wire [31:0]                  stat_l1_misses,
    output wire [31:0]                  stat_l2_hits,
    output wire [31:0]                  stat_l2_misses,
    output wire [31:0]                  stat_page_walks,
    output wire [31:0]                  stat_page_faults
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam SM_WIDTH = $clog2(NUM_SMS);
    localparam L1_IDX_WIDTH = $clog2(L1_ENTRIES / L1_WAYS);
    localparam L2_IDX_WIDTH = $clog2(L2_ENTRIES / L2_WAYS);
    localparam PAGE_OFFSET = 12;               // 4KB page offset
    localparam VPN_WIDTH = VADDR_WIDTH - PAGE_OFFSET;
    localparam PPN_WIDTH = PADDR_WIDTH - PAGE_OFFSET;
    localparam L1_TAG_WIDTH = VPN_WIDTH - L1_IDX_WIDTH;
    localparam L2_TAG_WIDTH = VPN_WIDTH - L2_IDX_WIDTH;

    // Page sizes
    localparam PAGE_4K  = 2'b00;
    localparam PAGE_2M  = 2'b01;
    localparam PAGE_1G  = 2'b10;

    // Fault codes
    localparam FAULT_NONE           = 4'h0;
    localparam FAULT_NOT_PRESENT    = 4'h1;
    localparam FAULT_WRITE_PROTECT  = 4'h2;
    localparam FAULT_USER_ACCESS    = 4'h3;
    localparam FAULT_RESERVED       = 4'h4;

    //------------------------------------------------------------------------
    // TLB Entry Format
    //------------------------------------------------------------------------
    // [valid, asid, vpn_tag, ppn, page_size, permissions]
    localparam L1_ENTRY_WIDTH = 1 + ASID_WIDTH + L1_TAG_WIDTH + PPN_WIDTH + 2 + 4;
    localparam L2_ENTRY_WIDTH = 1 + ASID_WIDTH + L2_TAG_WIDTH + PPN_WIDTH + 2 + 4;

    //------------------------------------------------------------------------
    // L1 TLB (per SM)
    //------------------------------------------------------------------------
    reg [L1_ENTRY_WIDTH-1:0] l1_tlb [0:NUM_SMS-1][0:L1_ENTRIES/L1_WAYS-1][0:L1_WAYS-1];
    reg [$clog2(L1_WAYS)-1:0] l1_lru [0:NUM_SMS-1][0:L1_ENTRIES/L1_WAYS-1];

    //------------------------------------------------------------------------
    // L2 TLB (shared)
    //------------------------------------------------------------------------
    reg [L2_ENTRY_WIDTH-1:0] l2_tlb [0:L2_ENTRIES/L2_WAYS-1][0:L2_WAYS-1];
    reg [$clog2(L2_WAYS)-1:0] l2_lru [0:L2_ENTRIES/L2_WAYS-1];

    //------------------------------------------------------------------------
    // Page Walker State
    //------------------------------------------------------------------------
    localparam PTW_IDLE     = 3'd0;
    localparam PTW_L4       = 3'd1;      // PML4
    localparam PTW_L3       = 3'd2;      // PDPT
    localparam PTW_L2       = 3'd3;      // PD
    localparam PTW_L1       = 3'd4;      // PT
    localparam PTW_DONE     = 3'd5;
    localparam PTW_FAULT    = 3'd6;

    reg [2:0] ptw_state;
    reg [SM_WIDTH-1:0] ptw_sm;
    reg [VADDR_WIDTH-1:0] ptw_vaddr;
    reg [ASID_WIDTH-1:0] ptw_asid;
    reg ptw_write;
    reg [PADDR_WIDTH-1:0] ptw_next_addr;
    reg [PPN_WIDTH-1:0] ptw_ppn;
    reg [1:0] ptw_page_size;
    reg [3:0] ptw_permissions;
    reg [3:0] ptw_fault_code;

    //------------------------------------------------------------------------
    // Request Arbitration
    //------------------------------------------------------------------------
    reg [NUM_SMS-1:0] pending_requests;
    reg [SM_WIDTH-1:0] current_sm;
    reg processing;

    // VPN extraction
    function [VPN_WIDTH-1:0] get_vpn;
        input [VADDR_WIDTH-1:0] vaddr;
        begin
            get_vpn = vaddr[VADDR_WIDTH-1:PAGE_OFFSET];
        end
    endfunction

    // L1 index
    function [L1_IDX_WIDTH-1:0] l1_index;
        input [VADDR_WIDTH-1:0] vaddr;
        begin
            l1_index = vaddr[PAGE_OFFSET +: L1_IDX_WIDTH];
        end
    endfunction

    // L2 index
    function [L2_IDX_WIDTH-1:0] l2_index;
        input [VADDR_WIDTH-1:0] vaddr;
        begin
            l2_index = vaddr[PAGE_OFFSET +: L2_IDX_WIDTH];
        end
    endfunction

    //------------------------------------------------------------------------
    // L1 TLB Lookup
    //------------------------------------------------------------------------
    reg [NUM_SMS-1:0] l1_hit;
    reg [PPN_WIDTH-1:0] l1_ppn [0:NUM_SMS-1];
    reg [1:0] l1_page_size [0:NUM_SMS-1];
    reg [3:0] l1_perms [0:NUM_SMS-1];
    reg [$clog2(L1_WAYS)-1:0] l1_hit_way [0:NUM_SMS-1];

    integer l1_s, l1_w;
    /* verilator lint_off LATCH */
    always @(*) begin
        for (l1_s = 0; l1_s < NUM_SMS; l1_s = l1_s + 1) begin
            l1_hit[l1_s] = 0;
            l1_ppn[l1_s] = 0;
            l1_page_size[l1_s] = 0;
            l1_perms[l1_s] = 0;
            l1_hit_way[l1_s] = 0;

            if (req_valid[l1_s]) begin
                begin
                    reg [VADDR_WIDTH-1:0] vaddr;
                    reg [L1_IDX_WIDTH-1:0] idx;
                    reg [L1_TAG_WIDTH-1:0] tag;
                    reg [ASID_WIDTH-1:0] asid;

                    vaddr = req_vaddr[l1_s*VADDR_WIDTH +: VADDR_WIDTH];
                    idx = l1_index(vaddr);
                    tag = vaddr[VADDR_WIDTH-1 -: L1_TAG_WIDTH];
                    asid = req_asid[l1_s*ASID_WIDTH +: ASID_WIDTH];

                    for (l1_w = 0; l1_w < L1_WAYS; l1_w = l1_w + 1) begin
                        begin
                            reg [L1_ENTRY_WIDTH-1:0] entry;
                            reg valid;
                            reg [ASID_WIDTH-1:0] e_asid;
                            reg [L1_TAG_WIDTH-1:0] e_tag;

                            entry = l1_tlb[l1_s][idx][l1_w];
                            valid = entry[L1_ENTRY_WIDTH-1];
                            e_asid = entry[L1_ENTRY_WIDTH-2 -: ASID_WIDTH];
                            e_tag = entry[L1_ENTRY_WIDTH-2-ASID_WIDTH -: L1_TAG_WIDTH];

                            if (valid && e_asid == asid && e_tag == tag) begin
                                l1_hit[l1_s] = 1;
                                l1_hit_way[l1_s] = l1_w;
                                l1_ppn[l1_s] = entry[PPN_WIDTH+6-1:6];
                                l1_page_size[l1_s] = entry[5:4];
                                l1_perms[l1_s] = entry[3:0];
                            end
                        end
                    end
                end
            end
        end
    end
    /* verilator lint_on LATCH */

    //------------------------------------------------------------------------
    // L2 TLB Lookup
    //------------------------------------------------------------------------
    reg l2_hit;
    reg [PPN_WIDTH-1:0] l2_ppn_r;
    reg [1:0] l2_page_size_r;
    reg [3:0] l2_perms_r;
    reg [$clog2(L2_WAYS)-1:0] l2_hit_way_r;

    integer l2_w;
    /* verilator lint_off LATCH */
    always @(*) begin
        l2_hit = 0;
        l2_ppn_r = 0;
        l2_page_size_r = 0;
        l2_perms_r = 0;
        l2_hit_way_r = 0;

        if (processing && !l1_hit[current_sm]) begin
            begin
                reg [VADDR_WIDTH-1:0] vaddr;
                reg [L2_IDX_WIDTH-1:0] idx;
                reg [L2_TAG_WIDTH-1:0] tag;
                reg [ASID_WIDTH-1:0] asid;

                vaddr = req_vaddr[current_sm*VADDR_WIDTH +: VADDR_WIDTH];
                idx = l2_index(vaddr);
                tag = vaddr[VADDR_WIDTH-1 -: L2_TAG_WIDTH];
                asid = req_asid[current_sm*ASID_WIDTH +: ASID_WIDTH];

                for (l2_w = 0; l2_w < L2_WAYS; l2_w = l2_w + 1) begin
                    begin
                        reg [L2_ENTRY_WIDTH-1:0] entry;
                        reg valid;
                        reg [ASID_WIDTH-1:0] e_asid;
                        reg [L2_TAG_WIDTH-1:0] e_tag;

                        entry = l2_tlb[idx][l2_w];
                        valid = entry[L2_ENTRY_WIDTH-1];
                        e_asid = entry[L2_ENTRY_WIDTH-2 -: ASID_WIDTH];
                        e_tag = entry[L2_ENTRY_WIDTH-2-ASID_WIDTH -: L2_TAG_WIDTH];

                        if (valid && e_asid == asid && e_tag == tag) begin
                            l2_hit = 1;
                            l2_hit_way_r = l2_w;
                            l2_ppn_r = entry[PPN_WIDTH+6-1:6];
                            l2_page_size_r = entry[5:4];
                            l2_perms_r = entry[3:0];
                        end
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Request Ready Logic
    //------------------------------------------------------------------------
    assign req_ready = ~pending_requests | l1_hit;

    //------------------------------------------------------------------------
    // Main Control Logic
    //------------------------------------------------------------------------
    reg [NUM_SMS-1:0] resp_valid_r;
    reg [PADDR_WIDTH-1:0] resp_paddr_r [0:NUM_SMS-1];
    reg [NUM_SMS-1:0] resp_fault_r;
    reg [3:0] resp_fault_code_r [0:NUM_SMS-1];

    integer ctrl_s, ctrl_w, ctrl_i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_requests <= 0;
            processing <= 0;
            current_sm <= 0;
            ptw_state <= PTW_IDLE;
            resp_valid_r <= 0;
            resp_fault_r <= 0;

            // Initialize TLBs
            for (ctrl_s = 0; ctrl_s < NUM_SMS; ctrl_s = ctrl_s + 1) begin
                resp_paddr_r[ctrl_s] <= 0;
                resp_fault_code_r[ctrl_s] <= 0;
                for (ctrl_i = 0; ctrl_i < L1_ENTRIES/L1_WAYS; ctrl_i = ctrl_i + 1) begin
                    l1_lru[ctrl_s][ctrl_i] <= 0;
                    for (ctrl_w = 0; ctrl_w < L1_WAYS; ctrl_w = ctrl_w + 1) begin
                        l1_tlb[ctrl_s][ctrl_i][ctrl_w] <= 0;
                    end
                end
            end
            for (ctrl_i = 0; ctrl_i < L2_ENTRIES/L2_WAYS; ctrl_i = ctrl_i + 1) begin
                l2_lru[ctrl_i] <= 0;
                for (ctrl_w = 0; ctrl_w < L2_WAYS; ctrl_w = ctrl_w + 1) begin
                    l2_tlb[ctrl_i][ctrl_w] <= 0;
                end
            end
        end else begin
            resp_valid_r <= 0;
            resp_fault_r <= 0;

            // Handle L1 hits immediately
            for (ctrl_s = 0; ctrl_s < NUM_SMS; ctrl_s = ctrl_s + 1) begin
                if (req_valid[ctrl_s] && l1_hit[ctrl_s]) begin
                    begin
                        reg [VADDR_WIDTH-1:0] vaddr;
                        reg [PADDR_WIDTH-1:0] paddr;

                        vaddr = req_vaddr[ctrl_s*VADDR_WIDTH +: VADDR_WIDTH];

                        // Construct physical address based on page size
                        case (l1_page_size[ctrl_s])
                            PAGE_4K: paddr = {l1_ppn[ctrl_s], vaddr[PAGE_OFFSET-1:0]};
                            PAGE_2M: paddr = {l1_ppn[ctrl_s][PPN_WIDTH-1:9],
                                             vaddr[PAGE_OFFSET+9-1:0]};
                            PAGE_1G: paddr = {l1_ppn[ctrl_s][PPN_WIDTH-1:18],
                                             vaddr[PAGE_OFFSET+18-1:0]};
                            default: paddr = {l1_ppn[ctrl_s], vaddr[PAGE_OFFSET-1:0]};
                        endcase

                        // Check permissions
                        if (req_write[ctrl_s] && !l1_perms[ctrl_s][1]) begin
                            resp_valid_r[ctrl_s] <= 1;
                            resp_fault_r[ctrl_s] <= 1;
                            resp_fault_code_r[ctrl_s] <= FAULT_WRITE_PROTECT;
                        end else begin
                            resp_valid_r[ctrl_s] <= 1;
                            resp_paddr_r[ctrl_s] <= paddr;
                        end

                        // Update LRU
                        l1_lru[ctrl_s][l1_index(vaddr)] <=
                            (l1_hit_way[ctrl_s] + 1) % L1_WAYS;
                    end
                end else if (req_valid[ctrl_s] && !pending_requests[ctrl_s]) begin
                    pending_requests[ctrl_s] <= 1;
                end
            end

            // Process pending requests (L2 lookup and page walk)
            if (!processing && |pending_requests) begin
                // Select SM with pending request
                for (ctrl_s = 0; ctrl_s < NUM_SMS; ctrl_s = ctrl_s + 1) begin
                    if (pending_requests[ctrl_s] && !processing) begin
                        processing <= 1;
                        current_sm <= ctrl_s;
                    end
                end
            end else if (processing) begin
                // Check L2
                if (l2_hit && ptw_state == PTW_IDLE) begin
                    begin
                        reg [VADDR_WIDTH-1:0] vaddr;
                        reg [PADDR_WIDTH-1:0] paddr;
                        reg [L1_IDX_WIDTH-1:0] l1_idx;

                        vaddr = req_vaddr[current_sm*VADDR_WIDTH +: VADDR_WIDTH];
                        l1_idx = l1_index(vaddr);

                        // Construct physical address
                        case (l2_page_size_r)
                            PAGE_4K: paddr = {l2_ppn_r, vaddr[PAGE_OFFSET-1:0]};
                            PAGE_2M: paddr = {l2_ppn_r[PPN_WIDTH-1:9],
                                             vaddr[PAGE_OFFSET+9-1:0]};
                            PAGE_1G: paddr = {l2_ppn_r[PPN_WIDTH-1:18],
                                             vaddr[PAGE_OFFSET+18-1:0]};
                            default: paddr = {l2_ppn_r, vaddr[PAGE_OFFSET-1:0]};
                        endcase

                        // Install in L1
                        l1_tlb[current_sm][l1_idx][l1_lru[current_sm][l1_idx]] <= {
                            1'b1,
                            req_asid[current_sm*ASID_WIDTH +: ASID_WIDTH],
                            vaddr[VADDR_WIDTH-1 -: L1_TAG_WIDTH],
                            l2_ppn_r,
                            l2_page_size_r,
                            l2_perms_r
                        };
                        l1_lru[current_sm][l1_idx] <=
                            (l1_lru[current_sm][l1_idx] + 1) % L1_WAYS;

                        // Return translation
                        resp_valid_r[current_sm] <= 1;
                        resp_paddr_r[current_sm] <= paddr;

                        pending_requests[current_sm] <= 0;
                        processing <= 0;
                    end
                end else begin
                    // Need page walk
                    case (ptw_state)
                        PTW_IDLE: begin
                            ptw_vaddr <= req_vaddr[current_sm*VADDR_WIDTH +: VADDR_WIDTH];
                            ptw_asid <= req_asid[current_sm*ASID_WIDTH +: ASID_WIDTH];
                            ptw_write <= req_write[current_sm];
                            // Start at PML4 (level 4)
                            ptw_next_addr <= page_table_base +
                                {ptw_vaddr[47:39], 3'b000};  // PML4 index
                            ptw_state <= PTW_L4;
                            ptw_page_size <= PAGE_4K;
                        end

                        PTW_L4, PTW_L3, PTW_L2, PTW_L1: begin
                            if (ptw_resp_valid) begin
                                // Parse page table entry
                                if (!ptw_resp_data[0]) begin
                                    // Not present
                                    ptw_fault_code <= FAULT_NOT_PRESENT;
                                    ptw_state <= PTW_FAULT;
                                end else if (ptw_write && !ptw_resp_data[1]) begin
                                    // Write protect
                                    ptw_fault_code <= FAULT_WRITE_PROTECT;
                                    ptw_state <= PTW_FAULT;
                                end else if (ptw_resp_data[7] || ptw_state == PTW_L1) begin
                                    // Large page or final level
                                    ptw_ppn <= ptw_resp_data[51:12];
                                    ptw_permissions <= ptw_resp_data[3:0];
                                    if (ptw_state == PTW_L3)
                                        ptw_page_size <= PAGE_1G;
                                    else if (ptw_state == PTW_L2)
                                        ptw_page_size <= PAGE_2M;
                                    else
                                        ptw_page_size <= PAGE_4K;
                                    ptw_state <= PTW_DONE;
                                end else begin
                                    // Continue walking
                                    case (ptw_state)
                                        PTW_L4: begin
                                            ptw_next_addr <= {ptw_resp_data[51:12], 12'b0} +
                                                {ptw_vaddr[38:30], 3'b000};
                                            ptw_state <= PTW_L3;
                                        end
                                        PTW_L3: begin
                                            ptw_next_addr <= {ptw_resp_data[51:12], 12'b0} +
                                                {ptw_vaddr[29:21], 3'b000};
                                            ptw_state <= PTW_L2;
                                        end
                                        PTW_L2: begin
                                            ptw_next_addr <= {ptw_resp_data[51:12], 12'b0} +
                                                {ptw_vaddr[20:12], 3'b000};
                                            ptw_state <= PTW_L1;
                                        end
                                        default: ptw_state <= PTW_FAULT;
                                    endcase
                                end
                            end
                        end

                        PTW_DONE: begin
                            begin
                                reg [PADDR_WIDTH-1:0] paddr;
                                reg [L1_IDX_WIDTH-1:0] l1_idx;
                                reg [L2_IDX_WIDTH-1:0] l2_idx;

                                l1_idx = l1_index(ptw_vaddr);
                                l2_idx = l2_index(ptw_vaddr);

                                // Construct physical address
                                case (ptw_page_size)
                                    PAGE_4K: paddr = {ptw_ppn, ptw_vaddr[PAGE_OFFSET-1:0]};
                                    PAGE_2M: paddr = {ptw_ppn[PPN_WIDTH-1:9],
                                                     ptw_vaddr[PAGE_OFFSET+9-1:0]};
                                    PAGE_1G: paddr = {ptw_ppn[PPN_WIDTH-1:18],
                                                     ptw_vaddr[PAGE_OFFSET+18-1:0]};
                                    default: paddr = {ptw_ppn, ptw_vaddr[PAGE_OFFSET-1:0]};
                                endcase

                                // Install in L2
                                l2_tlb[l2_idx][l2_lru[l2_idx]] <= {
                                    1'b1,
                                    ptw_asid,
                                    ptw_vaddr[VADDR_WIDTH-1 -: L2_TAG_WIDTH],
                                    ptw_ppn,
                                    ptw_page_size,
                                    ptw_permissions
                                };
                                l2_lru[l2_idx] <= (l2_lru[l2_idx] + 1) % L2_WAYS;

                                // Install in L1
                                l1_tlb[current_sm][l1_idx][l1_lru[current_sm][l1_idx]] <= {
                                    1'b1,
                                    ptw_asid,
                                    ptw_vaddr[VADDR_WIDTH-1 -: L1_TAG_WIDTH],
                                    ptw_ppn,
                                    ptw_page_size,
                                    ptw_permissions
                                };
                                l1_lru[current_sm][l1_idx] <=
                                    (l1_lru[current_sm][l1_idx] + 1) % L1_WAYS;

                                // Return translation
                                resp_valid_r[current_sm] <= 1;
                                resp_paddr_r[current_sm] <= paddr;

                                pending_requests[current_sm] <= 0;
                                processing <= 0;
                                ptw_state <= PTW_IDLE;
                            end
                        end

                        PTW_FAULT: begin
                            resp_valid_r[current_sm] <= 1;
                            resp_fault_r[current_sm] <= 1;
                            resp_fault_code_r[current_sm] <= ptw_fault_code;

                            pending_requests[current_sm] <= 0;
                            processing <= 0;
                            ptw_state <= PTW_IDLE;
                        end

                        default: ptw_state <= PTW_IDLE;
                    endcase
                end
            end

            // Handle invalidations
            if (invalidate_all) begin
                for (ctrl_s = 0; ctrl_s < NUM_SMS; ctrl_s = ctrl_s + 1) begin
                    for (ctrl_i = 0; ctrl_i < L1_ENTRIES/L1_WAYS; ctrl_i = ctrl_i + 1) begin
                        for (ctrl_w = 0; ctrl_w < L1_WAYS; ctrl_w = ctrl_w + 1) begin
                            l1_tlb[ctrl_s][ctrl_i][ctrl_w][L1_ENTRY_WIDTH-1] <= 0;
                        end
                    end
                end
                for (ctrl_i = 0; ctrl_i < L2_ENTRIES/L2_WAYS; ctrl_i = ctrl_i + 1) begin
                    for (ctrl_w = 0; ctrl_w < L2_WAYS; ctrl_w = ctrl_w + 1) begin
                        l2_tlb[ctrl_i][ctrl_w][L2_ENTRY_WIDTH-1] <= 0;
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Page Table Walk Memory Interface
    //------------------------------------------------------------------------
    assign ptw_req_valid = (ptw_state == PTW_L4 || ptw_state == PTW_L3 ||
                           ptw_state == PTW_L2 || ptw_state == PTW_L1);
    assign ptw_req_addr = ptw_next_addr;

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------
    assign resp_valid = resp_valid_r;
    assign resp_fault = resp_fault_r;

    genvar out_s;
    generate
        for (out_s = 0; out_s < NUM_SMS; out_s = out_s + 1) begin : gen_out
            assign resp_paddr[out_s*PADDR_WIDTH +: PADDR_WIDTH] = resp_paddr_r[out_s];
            assign resp_fault_code[out_s] = resp_fault_code_r[out_s];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] l1_hit_count;
    reg [31:0] l1_miss_count;
    reg [31:0] l2_hit_count;
    reg [31:0] l2_miss_count;
    reg [31:0] page_walk_count;
    reg [31:0] page_fault_count;

    assign stat_l1_hits = l1_hit_count;
    assign stat_l1_misses = l1_miss_count;
    assign stat_l2_hits = l2_hit_count;
    assign stat_l2_misses = l2_miss_count;
    assign stat_page_walks = page_walk_count;
    assign stat_page_faults = page_fault_count;

    integer stat_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l1_hit_count <= 0;
            l1_miss_count <= 0;
            l2_hit_count <= 0;
            l2_miss_count <= 0;
            page_walk_count <= 0;
            page_fault_count <= 0;
        end else begin
            for (stat_s = 0; stat_s < NUM_SMS; stat_s = stat_s + 1) begin
                if (req_valid[stat_s] && l1_hit[stat_s])
                    l1_hit_count <= l1_hit_count + 1;
            end
            if (processing && l2_hit)
                l2_hit_count <= l2_hit_count + 1;
            if (ptw_state == PTW_L4 && ptw_req_ready)
                page_walk_count <= page_walk_count + 1;
            if (ptw_state == PTW_FAULT)
                page_fault_count <= page_fault_count + 1;
        end
    end

endmodule
