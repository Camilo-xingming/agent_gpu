`timescale 1ns / 1ps

module tb_sm_writeback_arbiter;
    localparam NUM_WARPS  = 4;
    localparam NUM_LANES  = 4;
    localparam DATA_WIDTH = 32;
    localparam WARP_ID_W  = (NUM_WARPS > 1) ? $clog2(NUM_WARPS) : 1;
    localparam SIMD_WIDTH = NUM_LANES * DATA_WIDTH;

    reg clk;
    reg rst_n;

    // Queued FU sources
    reg                        alu_wbq_empty;
    reg [WARP_ID_W-1:0]        alu_wbq_warp;
    reg [4:0]                  alu_wbq_rd;
    reg [NUM_LANES-1:0]        alu_wbq_mask;
    reg [SIMD_WIDTH-1:0]       alu_wbq_data;

    reg                        mul_wbq_empty;
    reg [WARP_ID_W-1:0]        mul_wbq_warp;
    reg [4:0]                  mul_wbq_rd;
    reg [NUM_LANES-1:0]        mul_wbq_mask;
    reg [SIMD_WIDTH-1:0]       mul_wbq_data;

    reg                        fpu32_wbq_empty;
    reg [WARP_ID_W-1:0]        fpu32_wbq_warp;
    reg [4:0]                  fpu32_wbq_rd;
    reg [NUM_LANES-1:0]        fpu32_wbq_mask;
    reg [SIMD_WIDTH-1:0]       fpu32_wbq_data;

    reg                        fpu64_wbq_empty;
    reg [WARP_ID_W-1:0]        fpu64_wbq_warp;
    reg [4:0]                  fpu64_wbq_rd;
    reg [NUM_LANES-1:0]        fpu64_wbq_mask;
    reg [SIMD_WIDTH-1:0]       fpu64_wbq_data;

    reg                        fp16_wbq_empty;
    reg [WARP_ID_W-1:0]        fp16_wbq_warp;
    reg [4:0]                  fp16_wbq_rd;
    reg [NUM_LANES-1:0]        fp16_wbq_mask;
    reg [SIMD_WIDTH-1:0]       fp16_wbq_data;

    reg                        sfu_wbq_empty;
    reg [WARP_ID_W-1:0]        sfu_wbq_warp;
    reg [4:0]                  sfu_wbq_rd;
    reg [NUM_LANES-1:0]        sfu_wbq_mask;
    reg [SIMD_WIDTH-1:0]       sfu_wbq_data;

    reg                        tensor_wbq_empty;
    reg [WARP_ID_W-1:0]        tensor_wbq_warp;
    reg [4:0]                  tensor_wbq_rd;
    reg [NUM_LANES-1:0]        tensor_wbq_mask;
    reg [SIMD_WIDTH-1:0]       tensor_wbq_data;

    reg                        shfl_wbq_empty;
    reg [WARP_ID_W-1:0]        shfl_wbq_warp;
    reg [4:0]                  shfl_wbq_rd;
    reg [NUM_LANES-1:0]        shfl_wbq_mask;
    reg [SIMD_WIDTH-1:0]       shfl_wbq_data;

    reg                        special_wbq_empty;
    reg [WARP_ID_W-1:0]        special_wbq_warp;
    reg [4:0]                  special_wbq_rd;
    reg [NUM_LANES-1:0]        special_wbq_mask;
    reg [SIMD_WIDTH-1:0]       special_wbq_data;

    reg                        video_wbq_empty;
    reg [WARP_ID_W-1:0]        video_wbq_warp;
    reg [4:0]                  video_wbq_rd;
    reg [NUM_LANES-1:0]        video_wbq_mask;
    reg [SIMD_WIDTH-1:0]       video_wbq_data;

    // Non-queued sources
    reg                        gmem_resp_latched;
    reg [WARP_ID_W-1:0]        gmem_resp_warp;
    reg [4:0]                  gmem_resp_rd;
    reg [SIMD_WIDTH-1:0]       gmem_resp_data;
    reg [NUM_LANES-1:0]        gmem_resp_mask;

    reg                        smem_resp_latched;
    reg [WARP_ID_W-1:0]        smem_resp_warp;
    reg [4:0]                  smem_resp_rd;
    reg [SIMD_WIDTH-1:0]       smem_resp_data;
    reg [NUM_LANES-1:0]        smem_resp_mask;

    reg                        store_pending_valid;
    reg [WARP_ID_W-1:0]        store_warp_pending;
    reg [NUM_LANES-1:0]        store_mask_pending;

    reg                        atomic_valid_out_latched;
    reg [WARP_ID_W-1:0]        atomic_warp_pending;
    reg [4:0]                  atomic_rd_pending;
    reg [SIMD_WIDTH-1:0]       atomic_result;
    reg [NUM_LANES-1:0]        atomic_mask_pending;

    reg                        mbarrier_result_latched;
    reg [WARP_ID_W-1:0]        mbarrier_wb_warp;
    reg [4:0]                  mbarrier_wb_rd;
    reg [31:0]                 mbarrier_wb_result;
    reg [NUM_LANES-1:0]        mbarrier_wb_mask;

    reg                        tex_result_valid_latched;
    reg [WARP_ID_W-1:0]        tex_warp_pending;
    reg [4:0]                  tex_rd_pending;
    reg [127:0]                tex_result_latched;
    reg [NUM_LANES-1:0]        tex_mask_pending;

    reg                        cache_policy_token_valid_r;
    reg [WARP_ID_W-1:0]        cache_policy_wb_warp;
    reg [4:0]                  cache_policy_wb_rd;
    reg [NUM_LANES-1:0]        cache_policy_wb_mask;
    reg [31:0]                 cache_policy_token_r;

    reg                        stack_result_valid_r;
    reg [WARP_ID_W-1:0]        stack_wb_warp;
    reg [4:0]                  stack_wb_rd;
    reg [NUM_LANES-1:0]        stack_wb_mask;
    reg [31:0]                 stack_result_r;

    reg                        multimem_result_valid_r;
    reg [WARP_ID_W-1:0]        multimem_wb_warp;
    reg [4:0]                  multimem_wb_rd;
    reg [NUM_LANES-1:0]        multimem_wb_mask;
    reg [31:0]                 multimem_result_r;

    wire                       wb_valid;
    wire [WARP_ID_W-1:0]       wb_warp_id;
    wire [4:0]                 wb_rd;
    wire [SIMD_WIDTH-1:0]      wb_data;
    wire [NUM_LANES-1:0]       wb_mask;
    wire                       wb_found;
    wire [4:0]                 wb_sel;

    wire                       alu_wbq_pop;
    wire                       mul_wbq_pop;
    wire                       fpu32_wbq_pop;
    wire                       fpu64_wbq_pop;
    wire                       fp16_wbq_pop;
    wire                       sfu_wbq_pop;
    wire                       tensor_wbq_pop;
    wire                       shfl_wbq_pop;
    wire                       special_wbq_pop;
    wire                       video_wbq_pop;
    wire                       tex_wbq_pop;

    wire [10:0] pop_vec = {
        tex_wbq_pop,
        video_wbq_pop,
        special_wbq_pop,
        shfl_wbq_pop,
        tensor_wbq_pop,
        sfu_wbq_pop,
        fp16_wbq_pop,
        fpu64_wbq_pop,
        fpu32_wbq_pop,
        mul_wbq_pop,
        alu_wbq_pop
    };

    sm_writeback_arbiter #(
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .SM_ID(0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),

        .alu_wbq_empty(alu_wbq_empty),
        .alu_wbq_warp(alu_wbq_warp),
        .alu_wbq_rd(alu_wbq_rd),
        .alu_wbq_mask(alu_wbq_mask),
        .alu_wbq_data(alu_wbq_data),

        .mul_wbq_empty(mul_wbq_empty),
        .mul_wbq_warp(mul_wbq_warp),
        .mul_wbq_rd(mul_wbq_rd),
        .mul_wbq_mask(mul_wbq_mask),
        .mul_wbq_data(mul_wbq_data),

        .fpu32_wbq_empty(fpu32_wbq_empty),
        .fpu32_wbq_warp(fpu32_wbq_warp),
        .fpu32_wbq_rd(fpu32_wbq_rd),
        .fpu32_wbq_mask(fpu32_wbq_mask),
        .fpu32_wbq_data(fpu32_wbq_data),

        .fpu64_wbq_empty(fpu64_wbq_empty),
        .fpu64_wbq_warp(fpu64_wbq_warp),
        .fpu64_wbq_rd(fpu64_wbq_rd),
        .fpu64_wbq_mask(fpu64_wbq_mask),
        .fpu64_wbq_data(fpu64_wbq_data),

        .fp16_wbq_empty(fp16_wbq_empty),
        .fp16_wbq_warp(fp16_wbq_warp),
        .fp16_wbq_rd(fp16_wbq_rd),
        .fp16_wbq_mask(fp16_wbq_mask),
        .fp16_wbq_data(fp16_wbq_data),

        .sfu_wbq_empty(sfu_wbq_empty),
        .sfu_wbq_warp(sfu_wbq_warp),
        .sfu_wbq_rd(sfu_wbq_rd),
        .sfu_wbq_mask(sfu_wbq_mask),
        .sfu_wbq_data(sfu_wbq_data),

        .tensor_wbq_empty(tensor_wbq_empty),
        .tensor_wbq_warp(tensor_wbq_warp),
        .tensor_wbq_rd(tensor_wbq_rd),
        .tensor_wbq_mask(tensor_wbq_mask),
        .tensor_wbq_data(tensor_wbq_data),

        .shfl_wbq_empty(shfl_wbq_empty),
        .shfl_wbq_warp(shfl_wbq_warp),
        .shfl_wbq_rd(shfl_wbq_rd),
        .shfl_wbq_mask(shfl_wbq_mask),
        .shfl_wbq_data(shfl_wbq_data),

        .special_wbq_empty(special_wbq_empty),
        .special_wbq_warp(special_wbq_warp),
        .special_wbq_rd(special_wbq_rd),
        .special_wbq_mask(special_wbq_mask),
        .special_wbq_data(special_wbq_data),

        .video_wbq_empty(video_wbq_empty),
        .video_wbq_warp(video_wbq_warp),
        .video_wbq_rd(video_wbq_rd),
        .video_wbq_mask(video_wbq_mask),
        .video_wbq_data(video_wbq_data),

        .gmem_resp_latched(gmem_resp_latched),
        .gmem_resp_warp(gmem_resp_warp),
        .gmem_resp_rd(gmem_resp_rd),
        .gmem_resp_data(gmem_resp_data),
        .gmem_resp_mask(gmem_resp_mask),
        .smem_resp_latched(smem_resp_latched),
        .smem_resp_warp(smem_resp_warp),
        .smem_resp_rd(smem_resp_rd),
        .smem_resp_data(smem_resp_data),
        .smem_resp_mask(smem_resp_mask),
        .store_pending_valid(store_pending_valid),
        .store_warp_pending(store_warp_pending),
        .store_mask_pending(store_mask_pending),

        .atomic_valid_out_latched(atomic_valid_out_latched),
        .atomic_warp_pending(atomic_warp_pending),
        .atomic_rd_pending(atomic_rd_pending),
        .atomic_result(atomic_result),
        .atomic_mask_pending(atomic_mask_pending),

        .mbarrier_result_latched(mbarrier_result_latched),
        .mbarrier_wb_warp(mbarrier_wb_warp),
        .mbarrier_wb_rd(mbarrier_wb_rd),
        .mbarrier_wb_result(mbarrier_wb_result),
        .mbarrier_wb_mask(mbarrier_wb_mask),

        .tex_result_valid_latched(tex_result_valid_latched),
        .tex_warp_pending(tex_warp_pending),
        .tex_rd_pending(tex_rd_pending),
        .tex_result_latched(tex_result_latched),
        .tex_mask_pending(tex_mask_pending),

        .cache_policy_token_valid_r(cache_policy_token_valid_r),
        .cache_policy_wb_warp(cache_policy_wb_warp),
        .cache_policy_wb_rd(cache_policy_wb_rd),
        .cache_policy_wb_mask(cache_policy_wb_mask),
        .cache_policy_token_r(cache_policy_token_r),

        .stack_result_valid_r(stack_result_valid_r),
        .stack_wb_warp(stack_wb_warp),
        .stack_wb_rd(stack_wb_rd),
        .stack_wb_mask(stack_wb_mask),
        .stack_result_r(stack_result_r),

        .multimem_result_valid_r(multimem_result_valid_r),
        .multimem_wb_warp(multimem_wb_warp),
        .multimem_wb_rd(multimem_wb_rd),
        .multimem_wb_mask(multimem_wb_mask),
        .multimem_result_r(multimem_result_r),

        .wb_valid(wb_valid),
        .wb_warp_id(wb_warp_id),
        .wb_rd(wb_rd),
        .wb_data(wb_data),
        .wb_mask(wb_mask),
        .wb_found(wb_found),
        .wb_sel(wb_sel),

        .alu_wbq_pop(alu_wbq_pop),
        .mul_wbq_pop(mul_wbq_pop),
        .fpu32_wbq_pop(fpu32_wbq_pop),
        .fpu64_wbq_pop(fpu64_wbq_pop),
        .fp16_wbq_pop(fp16_wbq_pop),
        .sfu_wbq_pop(sfu_wbq_pop),
        .tensor_wbq_pop(tensor_wbq_pop),
        .shfl_wbq_pop(shfl_wbq_pop),
        .special_wbq_pop(special_wbq_pop),
        .video_wbq_pop(video_wbq_pop),
        .tex_wbq_pop(tex_wbq_pop)
    );

    always #5 clk = ~clk;

    integer pass_count;
    integer fail_count;

    task check;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s", msg);
            end
        end
    endtask

    task tick;
        begin
            @(posedge clk);
            #1;
        end
    endtask

    task clear_inputs;
        begin
            alu_wbq_empty = 1'b1; alu_wbq_warp = 0; alu_wbq_rd = 0; alu_wbq_mask = 0; alu_wbq_data = 0;
            mul_wbq_empty = 1'b1; mul_wbq_warp = 0; mul_wbq_rd = 0; mul_wbq_mask = 0; mul_wbq_data = 0;
            fpu32_wbq_empty = 1'b1; fpu32_wbq_warp = 0; fpu32_wbq_rd = 0; fpu32_wbq_mask = 0; fpu32_wbq_data = 0;
            fpu64_wbq_empty = 1'b1; fpu64_wbq_warp = 0; fpu64_wbq_rd = 0; fpu64_wbq_mask = 0; fpu64_wbq_data = 0;
            fp16_wbq_empty = 1'b1; fp16_wbq_warp = 0; fp16_wbq_rd = 0; fp16_wbq_mask = 0; fp16_wbq_data = 0;
            sfu_wbq_empty = 1'b1; sfu_wbq_warp = 0; sfu_wbq_rd = 0; sfu_wbq_mask = 0; sfu_wbq_data = 0;
            tensor_wbq_empty = 1'b1; tensor_wbq_warp = 0; tensor_wbq_rd = 0; tensor_wbq_mask = 0; tensor_wbq_data = 0;
            shfl_wbq_empty = 1'b1; shfl_wbq_warp = 0; shfl_wbq_rd = 0; shfl_wbq_mask = 0; shfl_wbq_data = 0;
            special_wbq_empty = 1'b1; special_wbq_warp = 0; special_wbq_rd = 0; special_wbq_mask = 0; special_wbq_data = 0;
            video_wbq_empty = 1'b1; video_wbq_warp = 0; video_wbq_rd = 0; video_wbq_mask = 0; video_wbq_data = 0;

            gmem_resp_latched = 1'b0; gmem_resp_warp = 0; gmem_resp_rd = 0; gmem_resp_data = 0; gmem_resp_mask = 0;
            smem_resp_latched = 1'b0; smem_resp_warp = 0; smem_resp_rd = 0; smem_resp_data = 0; smem_resp_mask = 0;
            store_pending_valid = 1'b0; store_warp_pending = 0; store_mask_pending = 0;

            atomic_valid_out_latched = 1'b0; atomic_warp_pending = 0; atomic_rd_pending = 0;
            atomic_result = 0; atomic_mask_pending = 0;

            mbarrier_result_latched = 1'b0; mbarrier_wb_warp = 0; mbarrier_wb_rd = 0;
            mbarrier_wb_result = 0; mbarrier_wb_mask = 0;

            tex_result_valid_latched = 1'b0; tex_warp_pending = 0; tex_rd_pending = 0;
            tex_result_latched = 0; tex_mask_pending = 0;

            cache_policy_token_valid_r = 1'b0; cache_policy_wb_warp = 0; cache_policy_wb_rd = 0;
            cache_policy_wb_mask = 0; cache_policy_token_r = 0;

            stack_result_valid_r = 1'b0; stack_wb_warp = 0; stack_wb_rd = 0;
            stack_wb_mask = 0; stack_result_r = 0;

            multimem_result_valid_r = 1'b0; multimem_wb_warp = 0; multimem_wb_rd = 0;
            multimem_wb_mask = 0; multimem_result_r = 0;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        pass_count = 0;
        fail_count = 0;

        clear_inputs();

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        // Case 0: idle state
        clear_inputs();
        #1;
        check(!wb_found, "case0: wb_found should be low when no source ready");
        check(pop_vec == 11'b0, "case0: no pop on idle");
        tick();
        check(!wb_valid, "case0: wb_valid should be low after idle cycle");

        // Case 1: ALU selected at reset priority
        clear_inputs();
        alu_wbq_empty = 1'b0;
        alu_wbq_warp  = 2'd1;
        alu_wbq_rd    = 5'd5;
        alu_wbq_mask  = 4'b1011;
        alu_wbq_data  = 128'h00000004_00000003_00000002_00000001;
        #1;
        check(wb_found && wb_sel == 5'd0, "case1: expected ALU grant");
        check(pop_vec == 11'b00000000001, "case1: expected ALU pop only");
        tick();
        check(wb_valid, "case1: wb_valid should assert");
        check(wb_warp_id == 2'd1, "case1: ALU warp id");
        check(wb_rd == 5'd5, "case1: ALU rd");
        check(wb_mask == 4'b1011, "case1: ALU mask");
        check(wb_data == 128'h00000004_00000003_00000002_00000001, "case1: ALU data");

        // Case 2: round-robin prefers MUL next (priority advanced to 1)
        clear_inputs();
        alu_wbq_empty = 1'b0;
        alu_wbq_warp  = 2'd2;
        alu_wbq_rd    = 5'd7;
        alu_wbq_mask  = 4'b1111;
        alu_wbq_data  = 128'hA0A0A0A0_A0A0A0A0_A0A0A0A0_A0A0A0A0;

        mul_wbq_empty = 1'b0;
        mul_wbq_warp  = 2'd3;
        mul_wbq_rd    = 5'd9;
        mul_wbq_mask  = 4'b0111;
        mul_wbq_data  = 128'h00000013_00000012_00000011_00000010;
        #1;
        check(wb_found && wb_sel == 5'd1, "case2: expected MUL grant via round-robin");
        check(pop_vec == 11'b00000000010, "case2: expected MUL pop only");
        tick();
        check(wb_valid, "case2: wb_valid should assert");
        check(wb_warp_id == 2'd3, "case2: MUL warp id");
        check(wb_rd == 5'd9, "case2: MUL rd");
        check(wb_mask == 4'b0111, "case2: MUL mask");
        check(wb_data == 128'h00000013_00000012_00000011_00000010, "case2: MUL data");

        // Case 3: with same contenders, arbitration wraps and picks ALU
        #1;
        check(wb_found && wb_sel == 5'd0, "case3: expected ALU after wrap-around");
        check(pop_vec == 11'b00000000001, "case3: expected ALU pop only");
        tick();
        check(wb_valid, "case3: wb_valid should assert");
        check(wb_warp_id == 2'd2, "case3: ALU warp id");
        check(wb_rd == 5'd7, "case3: ALU rd");

        // Case 4: SHFL queue path (index 8)
        clear_inputs();
        shfl_wbq_empty = 1'b0;
        shfl_wbq_warp  = 2'd1;
        shfl_wbq_rd    = 5'd18;
        shfl_wbq_mask  = 4'b0101;
        shfl_wbq_data  = 128'h11111114_11111113_11111112_11111111;
        #1;
        check(wb_found && wb_sel == 5'd8, "case4: expected SHFL grant");
        check(pop_vec == 11'b00010000000, "case4: expected SHFL pop only");
        tick();
        check(wb_warp_id == 2'd1, "case4: SHFL warp id");
        check(wb_rd == 5'd18, "case4: SHFL rd");
        check(wb_data == 128'h11111114_11111113_11111112_11111111, "case4: SHFL data");

        // Case 5: SPECIAL queue path (index 10)
        clear_inputs();
        special_wbq_empty = 1'b0;
        special_wbq_warp  = 2'd0;
        special_wbq_rd    = 5'd3;
        special_wbq_mask  = 4'b0011;
        special_wbq_data  = 128'h22222224_22222223_22222222_22222221;
        #1;
        check(wb_found && wb_sel == 5'd10, "case5: expected SPECIAL grant");
        check(pop_vec == 11'b00100000000, "case5: expected SPECIAL pop only");
        tick();
        check(wb_warp_id == 2'd0, "case5: SPECIAL warp id");
        check(wb_rd == 5'd3, "case5: SPECIAL rd");

        // Case 6: VIDEO queue path (index 13)
        clear_inputs();
        video_wbq_empty = 1'b0;
        video_wbq_warp  = 2'd2;
        video_wbq_rd    = 5'd11;
        video_wbq_mask  = 4'b1110;
        video_wbq_data  = 128'h33333334_33333333_33333332_33333331;
        #1;
        check(wb_found && wb_sel == 5'd13, "case6: expected VIDEO grant");
        check(pop_vec == 11'b01000000000, "case6: expected VIDEO pop only");
        tick();
        check(wb_warp_id == 2'd2, "case6: VIDEO warp id");
        check(wb_rd == 5'd11, "case6: VIDEO rd");

        // Case 7: memory path uses SMEM over GMEM when both latched
        clear_inputs();
        gmem_resp_latched = 1'b1;
        gmem_resp_warp    = 2'd1;
        gmem_resp_rd      = 5'd4;
        gmem_resp_data    = 128'h44444444_44444444_44444444_44444444;
        gmem_resp_mask    = 4'b1100;

        smem_resp_latched = 1'b1;
        smem_resp_warp    = 2'd3;
        smem_resp_rd      = 5'd6;
        smem_resp_data    = 128'h55555554_55555553_55555552_55555551;
        smem_resp_mask    = 4'b1111;
        #1;
        check(wb_found && wb_sel == 5'd7, "case7: expected memory grant");
        check(pop_vec == 11'b0, "case7: non-queued source should not pop WBQ");
        tick();
        check(wb_warp_id == 2'd3, "case7: SMEM warp should win over GMEM");
        check(wb_rd == 5'd6, "case7: SMEM rd should win over GMEM");
        check(wb_data == 128'h55555554_55555553_55555552_55555551, "case7: SMEM data should win over GMEM");
        check(wb_mask == 4'b1111, "case7: SMEM mask should win over GMEM");

        // Case 8: memory store completion path
        clear_inputs();
        store_pending_valid = 1'b1;
        store_warp_pending  = 2'd2;
        store_mask_pending  = 4'b0101;
        #1;
        check(wb_found && wb_sel == 5'd7, "case8: expected memory store grant");
        tick();
        check(wb_warp_id == 2'd2, "case8: store warp should propagate");
        check(wb_rd == 5'd0, "case8: store completion rd should be zero");
        check(wb_data == {SIMD_WIDTH{1'b0}}, "case8: store completion data should be zero");
        check(wb_mask == 4'b0101, "case8: store mask should propagate");

        // Case 9: atomic path
        clear_inputs();
        atomic_valid_out_latched = 1'b1;
        atomic_warp_pending      = 2'd1;
        atomic_rd_pending        = 5'd14;
        atomic_result            = 128'h66666664_66666663_66666662_66666661;
        atomic_mask_pending      = 4'b1010;
        #1;
        check(wb_found && wb_sel == 5'd9, "case9: expected atomic grant");
        tick();
        check(wb_warp_id == 2'd1, "case9: atomic warp");
        check(wb_rd == 5'd14, "case9: atomic rd");
        check(wb_data == 128'h66666664_66666663_66666662_66666661, "case9: atomic data");
        check(wb_mask == 4'b1010, "case9: atomic mask");

        // Case 10: mbarrier scalar replication path
        clear_inputs();
        mbarrier_result_latched = 1'b1;
        mbarrier_wb_warp        = 2'd0;
        mbarrier_wb_rd          = 5'd20;
        mbarrier_wb_result      = 32'hDEAD_BEEF;
        mbarrier_wb_mask        = 4'b0110;
        #1;
        check(wb_found && wb_sel == 5'd11, "case10: expected mbarrier grant");
        tick();
        check(wb_warp_id == 2'd0, "case10: mbarrier warp");
        check(wb_rd == 5'd20, "case10: mbarrier rd");
        check(wb_data == {NUM_LANES{32'hDEAD_BEEF}}, "case10: mbarrier result replication");
        check(wb_mask == 4'b0110, "case10: mbarrier mask");

        // Case 11: texture 2x2 result path
        clear_inputs();
        tex_result_valid_latched = 1'b1;
        tex_warp_pending         = 2'd3;
        tex_rd_pending           = 5'd21;
        tex_result_latched       = 128'h77777774_77777773_77777772_77777771;
        tex_mask_pending         = 4'b1111;
        #1;
        check(wb_found && wb_sel == 5'd12, "case11: expected texture grant");
        check(tex_wbq_pop, "case11: texture grant should assert tex_wbq_pop");
        tick();
        check(wb_warp_id == 2'd3, "case11: texture warp");
        check(wb_rd == 5'd21, "case11: texture rd");
        check(wb_data == 128'h77777774_77777773_77777772_77777771, "case11: texture data mapping");
        check(wb_mask == 4'b1111, "case11: texture mask");

        // Case 12: cache policy token replication path
        clear_inputs();
        cache_policy_token_valid_r = 1'b1;
        cache_policy_wb_warp       = 2'd2;
        cache_policy_wb_rd         = 5'd22;
        cache_policy_wb_mask       = 4'b0011;
        cache_policy_token_r       = 32'hCAFEBABE;
        #1;
        check(wb_found && wb_sel == 5'd14, "case12: expected cache policy grant");
        tick();
        check(wb_warp_id == 2'd2, "case12: cache policy warp");
        check(wb_rd == 5'd22, "case12: cache policy rd");
        check(wb_data == {NUM_LANES{32'hCAFEBABE}}, "case12: cache token replication");
        check(wb_mask == 4'b0011, "case12: cache policy mask");

        // Case 13: stack result replication path
        clear_inputs();
        stack_result_valid_r = 1'b1;
        stack_wb_warp        = 2'd1;
        stack_wb_rd          = 5'd23;
        stack_wb_mask        = 4'b1001;
        stack_result_r       = 32'h01020304;
        #1;
        check(wb_found && wb_sel == 5'd15, "case13: expected stack grant");
        tick();
        check(wb_warp_id == 2'd1, "case13: stack warp");
        check(wb_rd == 5'd23, "case13: stack rd");
        check(wb_data == {NUM_LANES{32'h01020304}}, "case13: stack result replication");
        check(wb_mask == 4'b1001, "case13: stack mask");

        // Case 14: multimem result replication path
        clear_inputs();
        multimem_result_valid_r = 1'b1;
        multimem_wb_warp        = 2'd0;
        multimem_wb_rd          = 5'd24;
        multimem_wb_mask        = 4'b1110;
        multimem_result_r       = 32'hB16B00B5;
        #1;
        check(wb_found && wb_sel == 5'd16, "case14: expected multimem grant");
        tick();
        check(wb_warp_id == 2'd0, "case14: multimem warp");
        check(wb_rd == 5'd24, "case14: multimem rd");
        check(wb_data == {NUM_LANES{32'hB16B00B5}}, "case14: multimem result replication");
        check(wb_mask == 4'b1110, "case14: multimem mask");

        // Case 15: tensor queue path (index 6)
        clear_inputs();
        tensor_wbq_empty = 1'b0;
        tensor_wbq_warp  = 2'd3;
        tensor_wbq_rd    = 5'd25;
        tensor_wbq_mask  = 4'b1101;
        tensor_wbq_data  = 128'h88888884_88888883_88888882_88888881;
        #1;
        check(wb_found && wb_sel == 5'd6, "case15: expected tensor grant");
        check(pop_vec == 11'b00001000000, "case15: expected tensor pop only");
        tick();
        check(wb_warp_id == 2'd3, "case15: tensor warp");
        check(wb_rd == 5'd25, "case15: tensor rd");
        check(wb_data == 128'h88888884_88888883_88888882_88888881, "case15: tensor data");

        // Case 16: return to idle and ensure wb_valid drops
        clear_inputs();
        #1;
        check(!wb_found, "case16: wb_found low after clearing all sources");
        check(pop_vec == 11'b0, "case16: no pop after clearing all sources");
        tick();
        check(!wb_valid, "case16: wb_valid should deassert when no source ready");

        if (fail_count == 0) begin
            $display("========================================");
            $display("[PASS] tb_sm_writeback_arbiter: %0d checks", pass_count);
            $display("========================================");
            $finish;
        end else begin
            $display("========================================");
            $display("[FAIL] tb_sm_writeback_arbiter: %0d passed, %0d failed", pass_count, fail_count);
            $display("========================================");
            $fatal(1, "tb_sm_writeback_arbiter failed");
        end
    end

endmodule
