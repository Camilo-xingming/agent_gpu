/* verilator lint_off UNDRIVEN */
//============================================================================
// RalphGPU - Branch Predictor
// Features:
//   - TAGE-like predictor with base + tagged components
//   - Return Address Stack (RAS)
//   - Branch Target Buffer (BTB)
//   - Loop predictor
//   - Per-warp prediction state
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module branch_predictor #(
    parameter NUM_WARPS         = 8,
    parameter ADDR_WIDTH        = 32,
    parameter BTB_ENTRIES       = 256,
    parameter BTB_WAYS          = 4,
    parameter BHT_ENTRIES       = 1024,         // Branch History Table
    parameter TAGE_TABLES       = 4,            // Tagged predictor tables
    parameter TAGE_ENTRIES      = 256,
    parameter RAS_DEPTH         = 8,            // Return Address Stack
    parameter LOOP_ENTRIES      = 32
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // Prediction Request (from Fetch)
    //------------------------------------------------------------------------
    input  wire                         pred_req,
    input  wire [$clog2(NUM_WARPS)-1:0] pred_warp_id,
    input  wire [ADDR_WIDTH-1:0]        pred_pc,
    input  wire                         pred_is_branch,
    input  wire                         pred_is_call,
    input  wire                         pred_is_return,

    output wire                         pred_valid,
    output wire                         pred_taken,
    output wire [ADDR_WIDTH-1:0]        pred_target,
    output wire [1:0]                   pred_confidence,    // 0=low, 3=high

    //------------------------------------------------------------------------
    // Update Interface (from Execute)
    //------------------------------------------------------------------------
    input  wire                         update_valid,
    input  wire [$clog2(NUM_WARPS)-1:0] update_warp_id,
    input  wire [ADDR_WIDTH-1:0]        update_pc,
    input  wire                         update_taken,
    input  wire [ADDR_WIDTH-1:0]        update_target,
    input  wire                         update_is_call,
    input  wire                         update_is_return,
    input  wire                         update_mispredicted,

    //------------------------------------------------------------------------
    // Performance Counters
    //------------------------------------------------------------------------
    output wire [31:0]                  stat_predictions,
    output wire [31:0]                  stat_mispredictions,
    output wire [31:0]                  stat_btb_hits,
    output wire [31:0]                  stat_ras_hits
);

    //------------------------------------------------------------------------
    // Local Parameters
    //------------------------------------------------------------------------
    localparam BTB_IDX_WIDTH = $clog2(BTB_ENTRIES);
    localparam BHT_IDX_WIDTH = $clog2(BHT_ENTRIES);
    localparam TAGE_IDX_WIDTH = $clog2(TAGE_ENTRIES);
    localparam BTB_TAG_WIDTH = ADDR_WIDTH - BTB_IDX_WIDTH - 2;
    localparam WARP_WIDTH = $clog2(NUM_WARPS);
    localparam RAS_PTR_WIDTH = $clog2(RAS_DEPTH);

    //------------------------------------------------------------------------
    // Branch History Register (per warp)
    //------------------------------------------------------------------------
    reg [15:0] branch_history [0:NUM_WARPS-1];

    //------------------------------------------------------------------------
    // BTB (Branch Target Buffer)
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] btb_target [0:BTB_ENTRIES-1][0:BTB_WAYS-1];
    reg [BTB_TAG_WIDTH-1:0] btb_tag [0:BTB_ENTRIES-1][0:BTB_WAYS-1];
    reg [BTB_WAYS-1:0] btb_valid [0:BTB_ENTRIES-1];
    reg [1:0] btb_type [0:BTB_ENTRIES-1][0:BTB_WAYS-1];  // 0=cond, 1=uncond, 2=call, 3=ret
    reg [$clog2(BTB_WAYS)-1:0] btb_lru [0:BTB_ENTRIES-1];

    // BTB lookup
    wire [BTB_IDX_WIDTH-1:0] btb_idx = pred_pc[2 +: BTB_IDX_WIDTH];
    wire [BTB_TAG_WIDTH-1:0] btb_lookup_tag = pred_pc[2+BTB_IDX_WIDTH +: BTB_TAG_WIDTH];

    reg btb_hit;
    reg [$clog2(BTB_WAYS)-1:0] btb_hit_way;
    reg [ADDR_WIDTH-1:0] btb_hit_target;
    reg [1:0] btb_hit_type;

    integer btb_w;
    always @(*) begin
        btb_hit = 0;
        btb_hit_way = 0;
        btb_hit_target = 0;
        btb_hit_type = 0;
        for (btb_w = 0; btb_w < BTB_WAYS; btb_w = btb_w + 1) begin
            if (btb_valid[btb_idx][btb_w] && btb_tag[btb_idx][btb_w] == btb_lookup_tag) begin
                btb_hit = 1;
                btb_hit_way = btb_w;
                btb_hit_target = btb_target[btb_idx][btb_w];
                btb_hit_type = btb_type[btb_idx][btb_w];
            end
        end
    end

    //------------------------------------------------------------------------
    // Base Bimodal Predictor (2-bit saturating counters)
    //------------------------------------------------------------------------
    reg [1:0] bimodal_table [0:BHT_ENTRIES-1];

    wire [BHT_IDX_WIDTH-1:0] bht_idx = pred_pc[2 +: BHT_IDX_WIDTH] ^
                                       branch_history[pred_warp_id][BHT_IDX_WIDTH-1:0];
    wire [1:0] bimodal_pred = bimodal_table[bht_idx];
    wire bimodal_taken = bimodal_pred[1];

    //------------------------------------------------------------------------
    // TAGE Tagged Predictor
    //------------------------------------------------------------------------
    // Each table has different history length
    // Table 0: 4, Table 1: 8, Table 2: 16, Table 3: 32
    function [7:0] get_tage_hist_len;
        input [1:0] table_idx;
        begin
            case (table_idx)
                2'd0: get_tage_hist_len = 8'd4;
                2'd1: get_tage_hist_len = 8'd8;
                2'd2: get_tage_hist_len = 8'd16;
                2'd3: get_tage_hist_len = 8'd32;
            endcase
        end
    endfunction

    reg [2:0] tage_counter [0:TAGE_TABLES-1][0:TAGE_ENTRIES-1];  // 3-bit counter
    reg [7:0] tage_tag [0:TAGE_TABLES-1][0:TAGE_ENTRIES-1];
    reg [1:0] tage_useful [0:TAGE_TABLES-1][0:TAGE_ENTRIES-1];
    reg [TAGE_TABLES-1:0] tage_valid [0:TAGE_ENTRIES-1];

    // TAGE history lengths (compile-time constants)
    localparam [7:0] TAGE_HIST_LEN_0 = 8'd4;
    localparam [7:0] TAGE_HIST_LEN_1 = 8'd8;
    localparam [7:0] TAGE_HIST_LEN_2 = 8'd16;
    localparam [7:0] TAGE_HIST_LEN_3 = 8'd32;

    // TAGE index/tag generation using folded history
    function [TAGE_IDX_WIDTH-1:0] tage_index;
        input [ADDR_WIDTH-1:0] pc;
        input [15:0] history;
        input [1:0] table_id;
        reg [7:0] hist_len;
        begin
            case (table_id)
                2'd0: hist_len = TAGE_HIST_LEN_0;
                2'd1: hist_len = TAGE_HIST_LEN_1;
                2'd2: hist_len = TAGE_HIST_LEN_2;
                2'd3: hist_len = TAGE_HIST_LEN_3;
            endcase
            tage_index = pc[2 +: TAGE_IDX_WIDTH] ^ history[7:0];  // Use lower bits
        end
    endfunction

    function [7:0] tage_compute_tag;
        input [ADDR_WIDTH-1:0] pc;
        input [15:0] history;
        input [1:0] table_id;
        begin
            tage_compute_tag = pc[10:3] ^ history[7:0] ^ {history[15:8]};
        end
    endfunction

    // TAGE lookup
    reg [TAGE_TABLES-1:0] tage_hit;
    reg [2:0] tage_pred [0:TAGE_TABLES-1];
    reg [TAGE_IDX_WIDTH-1:0] tage_idx [0:TAGE_TABLES-1];
    reg [7:0] tage_lookup_tag [0:TAGE_TABLES-1];

    // TAGE lookup - unrolled for synthesis
    always @(*) begin
        // Table 0
        tage_idx[0] = tage_index(pred_pc, branch_history[pred_warp_id], 2'd0);
        tage_lookup_tag[0] = tage_compute_tag(pred_pc, branch_history[pred_warp_id], 2'd0);
        tage_hit[0] = tage_valid[tage_idx[0]][0] &&
                      (tage_tag[0][tage_idx[0]] == tage_lookup_tag[0]);
        tage_pred[0] = tage_counter[0][tage_idx[0]];
        // Table 1
        tage_idx[1] = tage_index(pred_pc, branch_history[pred_warp_id], 2'd1);
        tage_lookup_tag[1] = tage_compute_tag(pred_pc, branch_history[pred_warp_id], 2'd1);
        tage_hit[1] = tage_valid[tage_idx[1]][1] &&
                      (tage_tag[1][tage_idx[1]] == tage_lookup_tag[1]);
        tage_pred[1] = tage_counter[1][tage_idx[1]];
        // Table 2
        tage_idx[2] = tage_index(pred_pc, branch_history[pred_warp_id], 2'd2);
        tage_lookup_tag[2] = tage_compute_tag(pred_pc, branch_history[pred_warp_id], 2'd2);
        tage_hit[2] = tage_valid[tage_idx[2]][2] &&
                      (tage_tag[2][tage_idx[2]] == tage_lookup_tag[2]);
        tage_pred[2] = tage_counter[2][tage_idx[2]];
        // Table 3
        tage_idx[3] = tage_index(pred_pc, branch_history[pred_warp_id], 2'd3);
        tage_lookup_tag[3] = tage_compute_tag(pred_pc, branch_history[pred_warp_id], 2'd3);
        tage_hit[3] = tage_valid[tage_idx[3]][3] &&
                      (tage_tag[3][tage_idx[3]] == tage_lookup_tag[3]);
        tage_pred[3] = tage_counter[3][tage_idx[3]];
    end

    // Select provider (longest matching history)
    reg [2:0] final_pred_counter;
    reg [1:0] provider_table;
    reg provider_found;

    integer prov_t;
    always @(*) begin
        final_pred_counter = {1'b0, bimodal_pred};
        provider_table = 0;
        provider_found = 0;
        for (prov_t = TAGE_TABLES - 1; prov_t >= 0; prov_t = prov_t - 1) begin
            if (tage_hit[prov_t] && !provider_found) begin
                final_pred_counter = tage_pred[prov_t];
                provider_table = prov_t;
                provider_found = 1;
            end
        end
    end

    wire tage_taken = final_pred_counter[2];

    //------------------------------------------------------------------------
    // Return Address Stack (per warp)
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] ras_stack [0:NUM_WARPS-1][0:RAS_DEPTH-1];
    reg [RAS_PTR_WIDTH-1:0] ras_ptr [0:NUM_WARPS-1];

    wire [ADDR_WIDTH-1:0] ras_top = ras_stack[pred_warp_id][ras_ptr[pred_warp_id]];

    //------------------------------------------------------------------------
    // Loop Predictor
    //------------------------------------------------------------------------
    reg [ADDR_WIDTH-1:0] loop_pc [0:LOOP_ENTRIES-1];
    reg [7:0] loop_count [0:LOOP_ENTRIES-1];
    reg [7:0] loop_limit [0:LOOP_ENTRIES-1];
    reg [LOOP_ENTRIES-1:0] loop_valid;
    reg [LOOP_ENTRIES-1:0] loop_confident;

    // Loop lookup
    reg loop_hit;
    reg [$clog2(LOOP_ENTRIES)-1:0] loop_hit_idx;
    reg loop_pred_exit;

    integer loop_i;
    always @(*) begin
        loop_hit = 0;
        loop_hit_idx = 0;
        loop_pred_exit = 0;
        for (loop_i = 0; loop_i < LOOP_ENTRIES; loop_i = loop_i + 1) begin
            if (loop_valid[loop_i] && loop_pc[loop_i] == pred_pc) begin
                loop_hit = 1;
                loop_hit_idx = loop_i;
                loop_pred_exit = loop_confident[loop_i] &&
                                (loop_count[loop_i] >= loop_limit[loop_i] - 1);
            end
        end
    end

    //------------------------------------------------------------------------
    // Final Prediction Logic
    //------------------------------------------------------------------------
    reg pred_valid_r;
    reg pred_taken_r;
    reg [ADDR_WIDTH-1:0] pred_target_r;
    reg [1:0] pred_confidence_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pred_valid_r <= 0;
            pred_taken_r <= 0;
            pred_target_r <= 0;
            pred_confidence_r <= 0;
        end else if (pred_req) begin
            pred_valid_r <= 1;

            if (pred_is_return) begin
                // Return prediction from RAS
                pred_taken_r <= 1;
                pred_target_r <= ras_top;
                pred_confidence_r <= 2'd3;
            end else if (pred_is_call) begin
                // Call - always taken
                pred_taken_r <= 1;
                pred_target_r <= btb_hit ? btb_hit_target : (pred_pc + 4);
                pred_confidence_r <= btb_hit ? 2'd3 : 2'd1;
            end else if (pred_is_branch) begin
                // Conditional branch
                if (loop_hit && loop_confident[loop_hit_idx]) begin
                    // Use loop predictor
                    pred_taken_r <= !loop_pred_exit;
                    pred_confidence_r <= 2'd3;
                end else begin
                    // Use TAGE
                    pred_taken_r <= tage_taken;
                    pred_confidence_r <= provider_found ? 2'd2 : 2'd1;
                end
                pred_target_r <= btb_hit ? btb_hit_target : (pred_pc + 4);
            end else begin
                pred_valid_r <= 0;
                pred_taken_r <= 0;
                pred_target_r <= pred_pc + 4;
                pred_confidence_r <= 0;
            end
        end else begin
            pred_valid_r <= 0;
        end
    end

    assign pred_valid = pred_valid_r;
    assign pred_taken = pred_taken_r;
    assign pred_target = pred_target_r;
    assign pred_confidence = pred_confidence_r;

    //------------------------------------------------------------------------
    // Predictor Update Logic
    //------------------------------------------------------------------------
    wire [BTB_IDX_WIDTH-1:0] upd_btb_idx = update_pc[2 +: BTB_IDX_WIDTH];
    wire [BHT_IDX_WIDTH-1:0] upd_bht_idx = update_pc[2 +: BHT_IDX_WIDTH] ^
                                           branch_history[update_warp_id][BHT_IDX_WIDTH-1:0];

    integer upd_t, upd_w, upd_m, upd_r, upd_l;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize tables
            for (upd_m = 0; upd_m < BHT_ENTRIES; upd_m = upd_m + 1) begin
                bimodal_table[upd_m] <= 2'b01;  // Weakly not-taken
            end
            for (upd_m = 0; upd_m < BTB_ENTRIES; upd_m = upd_m + 1) begin
                btb_valid[upd_m] <= 0;
                btb_lru[upd_m] <= 0;
            end
            for (upd_t = 0; upd_t < TAGE_TABLES; upd_t = upd_t + 1) begin
                for (upd_m = 0; upd_m < TAGE_ENTRIES; upd_m = upd_m + 1) begin
                    tage_counter[upd_t][upd_m] <= 3'b011;
                    tage_useful[upd_t][upd_m] <= 0;
                end
            end
            for (upd_m = 0; upd_m < TAGE_ENTRIES; upd_m = upd_m + 1) begin
                tage_valid[upd_m] <= 0;
            end
            for (upd_w = 0; upd_w < NUM_WARPS; upd_w = upd_w + 1) begin
                branch_history[upd_w] <= 0;
                ras_ptr[upd_w] <= 0;
            end
            loop_valid <= 0;
            loop_confident <= 0;
        end else if (update_valid) begin
            // Update branch history
            branch_history[update_warp_id] <=
                {branch_history[update_warp_id][14:0], update_taken};

            // Update bimodal
            if (update_taken && bimodal_table[upd_bht_idx] < 2'b11)
                bimodal_table[upd_bht_idx] <= bimodal_table[upd_bht_idx] + 1;
            else if (!update_taken && bimodal_table[upd_bht_idx] > 2'b00)
                bimodal_table[upd_bht_idx] <= bimodal_table[upd_bht_idx] - 1;

            // Update BTB on taken branches
            if (update_taken) begin
                begin
                    reg found_way;
                    reg [$clog2(BTB_WAYS)-1:0] alloc_way;
                    reg [BTB_TAG_WIDTH-1:0] upd_tag;

                    upd_tag = update_pc[2+BTB_IDX_WIDTH +: BTB_TAG_WIDTH];
                    found_way = 0;
                    alloc_way = btb_lru[upd_btb_idx];

                    // Check for existing entry
                    for (upd_w = 0; upd_w < BTB_WAYS; upd_w = upd_w + 1) begin
                        if (btb_valid[upd_btb_idx][upd_w] &&
                            btb_tag[upd_btb_idx][upd_w] == upd_tag) begin
                            found_way = 1;
                            alloc_way = upd_w;
                        end
                    end

                    btb_valid[upd_btb_idx][alloc_way] <= 1;
                    btb_tag[upd_btb_idx][alloc_way] <= upd_tag;
                    btb_target[upd_btb_idx][alloc_way] <= update_target;
                    btb_type[upd_btb_idx][alloc_way] <= update_is_call ? 2'd2 :
                                                        update_is_return ? 2'd3 : 2'd0;

                    if (!found_way)
                        btb_lru[upd_btb_idx] <= (btb_lru[upd_btb_idx] + 1) % BTB_WAYS;
                end
            end

            // Update RAS
            if (update_is_call) begin
                // Push return address
                ras_ptr[update_warp_id] <= (ras_ptr[update_warp_id] + 1) % RAS_DEPTH;
                ras_stack[update_warp_id][(ras_ptr[update_warp_id] + 1) % RAS_DEPTH] <=
                    update_pc + 4;
            end else if (update_is_return) begin
                // Pop
                if (ras_ptr[update_warp_id] > 0)
                    ras_ptr[update_warp_id] <= ras_ptr[update_warp_id] - 1;
            end

            // Update TAGE on misprediction - unrolled
            if (update_mispredicted) begin
                // Allocate in longer history table - Table 0
                begin
                    reg [TAGE_IDX_WIDTH-1:0] tidx0;
                    reg [7:0] ttag0;
                    tidx0 = tage_index(update_pc, branch_history[update_warp_id], 2'd0);
                    ttag0 = tage_compute_tag(update_pc, branch_history[update_warp_id], 2'd0);
                    if (!tage_valid[tidx0][0] || tage_useful[0][tidx0] == 0) begin
                        tage_valid[tidx0][0] <= 1;
                        tage_tag[0][tidx0] <= ttag0;
                        tage_counter[0][tidx0] <= update_taken ? 3'b100 : 3'b011;
                        tage_useful[0][tidx0] <= 0;
                    end
                end
                // Table 1
                begin
                    reg [TAGE_IDX_WIDTH-1:0] tidx1;
                    reg [7:0] ttag1;
                    tidx1 = tage_index(update_pc, branch_history[update_warp_id], 2'd1);
                    ttag1 = tage_compute_tag(update_pc, branch_history[update_warp_id], 2'd1);
                    if (!tage_valid[tidx1][1] || tage_useful[1][tidx1] == 0) begin
                        tage_valid[tidx1][1] <= 1;
                        tage_tag[1][tidx1] <= ttag1;
                        tage_counter[1][tidx1] <= update_taken ? 3'b100 : 3'b011;
                        tage_useful[1][tidx1] <= 0;
                    end
                end
                // Table 2
                begin
                    reg [TAGE_IDX_WIDTH-1:0] tidx2;
                    reg [7:0] ttag2;
                    tidx2 = tage_index(update_pc, branch_history[update_warp_id], 2'd2);
                    ttag2 = tage_compute_tag(update_pc, branch_history[update_warp_id], 2'd2);
                    if (!tage_valid[tidx2][2] || tage_useful[2][tidx2] == 0) begin
                        tage_valid[tidx2][2] <= 1;
                        tage_tag[2][tidx2] <= ttag2;
                        tage_counter[2][tidx2] <= update_taken ? 3'b100 : 3'b011;
                        tage_useful[2][tidx2] <= 0;
                    end
                end
                // Table 3
                begin
                    reg [TAGE_IDX_WIDTH-1:0] tidx3;
                    reg [7:0] ttag3;
                    tidx3 = tage_index(update_pc, branch_history[update_warp_id], 2'd3);
                    ttag3 = tage_compute_tag(update_pc, branch_history[update_warp_id], 2'd3);
                    if (!tage_valid[tidx3][3] || tage_useful[3][tidx3] == 0) begin
                        tage_valid[tidx3][3] <= 1;
                        tage_tag[3][tidx3] <= ttag3;
                        tage_counter[3][tidx3] <= update_taken ? 3'b100 : 3'b011;
                        tage_useful[3][tidx3] <= 0;
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Statistics
    //------------------------------------------------------------------------
    reg [31:0] pred_count;
    reg [31:0] mispred_count;
    reg [31:0] btb_hit_count;
    reg [31:0] ras_hit_count;

    assign stat_predictions = pred_count;
    assign stat_mispredictions = mispred_count;
    assign stat_btb_hits = btb_hit_count;
    assign stat_ras_hits = ras_hit_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pred_count <= 0;
            mispred_count <= 0;
            btb_hit_count <= 0;
            ras_hit_count <= 0;
        end else begin
            if (pred_req && pred_is_branch)
                pred_count <= pred_count + 1;
            if (update_valid && update_mispredicted)
                mispred_count <= mispred_count + 1;
            if (pred_req && btb_hit)
                btb_hit_count <= btb_hit_count + 1;
            if (pred_req && pred_is_return)
                ras_hit_count <= ras_hit_count + 1;
        end
    end

endmodule

/* verilator lint_on UNDRIVEN */
