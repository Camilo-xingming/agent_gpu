`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_sm_special_reg;
    localparam NUM_LANES  = 4;
    localparam SIMD_WIDTH = NUM_LANES * 32;
    localparam WARP_ID_W  = (`WARPS_PER_SM > 1) ? $clog2(`WARPS_PER_SM) : 1;

    reg                      clk;
    reg                      rst_n;
    reg                      issue_valid;
    reg                      use_slot0;
    reg  [WARP_ID_W-1:0]     slot0_warp_id;
    reg  [WARP_ID_W-1:0]     slot1_warp_id;
    reg  [4:0]               slot0_rd;
    reg  [4:0]               slot1_rd;
    reg  [4:0]               slot0_ra;
    reg  [4:0]               slot1_ra;
    reg  [NUM_LANES-1:0]     slot0_mask;
    reg  [NUM_LANES-1:0]     slot1_mask;
    reg  [NUM_LANES-1:0]     warp_active_mask;

    reg                      kernel_start;
    reg  [31:0]              block_id_x;
    reg  [31:0]              block_id_y;
    reg  [31:0]              block_id_z;
    reg  [31:0]              block_dim_x;
    reg  [31:0]              block_dim_y;
    reg  [31:0]              block_dim_z;
    reg  [31:0]              grid_dim_x;
    reg  [31:0]              grid_dim_y;
    reg  [31:0]              grid_dim_z;

    wire                     valid_out;
    wire [WARP_ID_W-1:0]     warp_out;
    wire [4:0]               rd_out;
    wire [NUM_LANES-1:0]     mask_out;
    wire [SIMD_WIDTH-1:0]    result_out;

    integer pass_count;
    integer fail_count;
    integer i;

    sm_special_reg #(
        .NUM_LANES(NUM_LANES),
        .SIMD_WIDTH(SIMD_WIDTH),
        .SM_ID(7)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .issue_valid(issue_valid),
        .use_slot0(use_slot0),
        .slot0_warp_id(slot0_warp_id),
        .slot1_warp_id(slot1_warp_id),
        .slot0_rd(slot0_rd),
        .slot1_rd(slot1_rd),
        .slot0_ra(slot0_ra),
        .slot1_ra(slot1_ra),
        .slot0_mask(slot0_mask),
        .slot1_mask(slot1_mask),
        .warp_active_mask(warp_active_mask),
        .kernel_start(kernel_start),
        .block_id_x(block_id_x),
        .block_id_y(block_id_y),
        .block_id_z(block_id_z),
        .block_dim_x(block_dim_x),
        .block_dim_y(block_dim_y),
        .block_dim_z(block_dim_z),
        .grid_dim_x(grid_dim_x),
        .grid_dim_y(grid_dim_y),
        .grid_dim_z(grid_dim_z),
        .valid_out(valid_out),
        .warp_out(warp_out),
        .rd_out(rd_out),
        .mask_out(mask_out),
        .result_out(result_out)
    );

    task tick;
    begin
        #5 clk = 1'b1;
        #5 clk = 1'b0;
    end
    endtask

    task check_bool;
        input cond;
        input [255:0] label;
    begin
        if (cond) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] %0s", label);
        end
    end
    endtask

    task check_u32;
        input [31:0] got;
        input [31:0] exp;
        input [255:0] label;
    begin
        if (got === exp) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] %0s got=0x%08x exp=0x%08x", label, got, exp);
        end
    end
    endtask

    task check_header;
        input [WARP_ID_W-1:0] exp_warp;
        input [4:0]           exp_rd;
        input [NUM_LANES-1:0] exp_mask;
        input [255:0]         label;
    begin
        check_bool(valid_out === 1'b1, {label, " valid_out"});
        check_bool(warp_out === exp_warp, {label, " warp_out"});
        check_bool(rd_out === exp_rd, {label, " rd_out"});
        check_bool(mask_out === exp_mask, {label, " mask_out"});
    end
    endtask

    task issue_cycle;
        input                  sel_slot0;
        input [WARP_ID_W-1:0]  warp0;
        input [WARP_ID_W-1:0]  warp1;
        input [4:0]            rd0;
        input [4:0]            rd1;
        input [4:0]            ra0;
        input [4:0]            ra1;
        input [NUM_LANES-1:0]  mask0;
        input [NUM_LANES-1:0]  mask1;
        input [NUM_LANES-1:0]  active_mask_in;
    begin
        use_slot0        = sel_slot0;
        slot0_warp_id    = warp0;
        slot1_warp_id    = warp1;
        slot0_rd         = rd0;
        slot1_rd         = rd1;
        slot0_ra         = ra0;
        slot1_ra         = ra1;
        slot0_mask       = mask0;
        slot1_mask       = mask1;
        warp_active_mask = active_mask_in;
        issue_valid      = 1'b1;
        tick();
        issue_valid      = 1'b0;
    end
    endtask

    initial begin
        clk              = 1'b0;
        rst_n            = 1'b0;
        issue_valid      = 1'b0;
        use_slot0        = 1'b1;
        slot0_warp_id    = {WARP_ID_W{1'b0}};
        slot1_warp_id    = {WARP_ID_W{1'b0}};
        slot0_rd         = 5'd0;
        slot1_rd         = 5'd0;
        slot0_ra         = 5'd0;
        slot1_ra         = 5'd0;
        slot0_mask       = {NUM_LANES{1'b0}};
        slot1_mask       = {NUM_LANES{1'b0}};
        warp_active_mask = {NUM_LANES{1'b0}};
        kernel_start     = 1'b0;
        block_id_x       = 32'd0;
        block_id_y       = 32'd0;
        block_id_z       = 32'd0;
        block_dim_x      = 32'd0;
        block_dim_y      = 32'd0;
        block_dim_z      = 32'd0;
        grid_dim_x       = 32'd0;
        grid_dim_y       = 32'd0;
        grid_dim_z       = 32'd0;

        pass_count = 0;
        fail_count = 0;

        // Reset
        tick();
        tick();
        check_bool(valid_out === 1'b0, "reset valid_out low");
        rst_n = 1'b1;

        // Capture kernel context #1
        block_id_x   = 32'd5;
        block_id_y   = 32'd6;
        block_id_z   = 32'd7;
        block_dim_x  = 32'd64;
        block_dim_y  = 32'd8;
        block_dim_z  = 32'd2;
        grid_dim_x   = 32'd16;
        grid_dim_y   = 32'd17;
        grid_dim_z   = 32'd18;
        kernel_start = 1'b1;
        tick();
        kernel_start = 1'b0;

        // Change inputs without kernel_start: latched value must remain old
        block_id_x = 32'd77;
        issue_cycle(1'b1, 3'd2, 3'd1, 5'd3, 5'd0, `SREG_CTAID_X, 5'd0, 4'b1111, 4'b0000, 4'b1010);
        check_header(3'd2, 5'd3, 4'b1111, "ctaid_x old latch");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd5, "ctaid_x lane scalar");
        tick();
        check_bool(valid_out === 1'b0, "ctaid_x pulse clear");

        // Capture kernel context #2
        block_id_x   = 32'd9;
        block_id_y   = 32'd21;
        block_id_z   = 32'd31;
        block_dim_x  = 32'd16;
        block_dim_y  = 32'd12;
        block_dim_z  = 32'd4;
        grid_dim_x   = 32'd32;
        grid_dim_y   = 32'd33;
        grid_dim_z   = 32'd42;
        kernel_start = 1'b1;
        tick();
        kernel_start = 1'b0;

        issue_cycle(1'b1, 3'd1, 3'd0, 5'd4, 5'd0, `SREG_NCTAID_Z, 5'd0, 4'b1110, 4'b0000, 4'b1111);
        check_header(3'd1, 5'd4, 4'b1110, "nctaid_z update");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd42, "nctaid_z lane scalar");
        tick();

        // %tid.x per-warp isolation
        issue_cycle(1'b1, 3'd0, 3'd0, 5'd6, 5'd0, `SREG_TID_X, 5'd0, 4'b1111, 4'b0000, 4'b1111);
        check_header(3'd0, 5'd6, 4'b1111, "tid.x warp0");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], i[31:0], "tid.x warp0 lane");
        tick();

        issue_cycle(1'b1, 3'd3, 3'd0, 5'd6, 5'd0, `SREG_TID_X, 5'd0, 4'b1111, 4'b0000, 4'b1111);
        check_header(3'd3, 5'd6, 4'b1111, "tid.x warp3");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], (3*NUM_LANES+i), "tid.x warp3 lane");
        tick();

        // %laneid
        issue_cycle(1'b1, 3'd2, 3'd0, 5'd7, 5'd0, `SREG_LANEID, 5'd0, 4'b0101, 4'b0000, 4'b1111);
        check_header(3'd2, 5'd7, 4'b0101, "laneid");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], i[31:0], "laneid lane");
        tick();

        // %warpid, %smid, %activemask
        issue_cycle(1'b1, 3'd3, 3'd0, 5'd8, 5'd0, `SREG_WARPID, 5'd0, 4'b0011, 4'b0000, 4'b1111);
        check_header(3'd3, 5'd8, 4'b0011, "warpid");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd3, "warpid scalar");
        tick();

        issue_cycle(1'b1, 3'd1, 3'd0, 5'd9, 5'd0, `SREG_SMID, 5'd0, 4'b1111, 4'b0000, 4'b1111);
        check_header(3'd1, 5'd9, 4'b1111, "smid");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd7, "smid scalar");
        tick();

        issue_cycle(1'b1, 3'd1, 3'd0, 5'd10, 5'd0, `SREG_ACTIVEMASK, 5'd0, 4'b1111, 4'b0000, 4'b0101);
        check_header(3'd1, 5'd10, 4'b1111, "activemask");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'h0000_0005, "activemask scalar");
        tick();

        // Slot arbitration edge: both slots populated, use_slot0 selects source
        issue_cycle(1'b0, 3'd1, 3'd2, 5'd11, 5'd12, `SREG_CTAID_Y, `SREG_NTID_X, 4'b0011, 4'b1100, 4'b1111);
        check_header(3'd2, 5'd12, 4'b1100, "slot1 selected");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd16, "slot1 ntid_x");
        tick();

        issue_cycle(1'b1, 3'd1, 3'd2, 5'd11, 5'd12, `SREG_CTAID_Y, `SREG_NTID_X, 4'b0011, 4'b1100, 4'b1111);
        check_header(3'd1, 5'd11, 4'b0011, "slot0 selected");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd21, "slot0 ctaid_y");
        tick();

        // Invalid register ID => zero vector
        issue_cycle(1'b1, 3'd1, 3'd0, 5'd13, 5'd0, 5'd31, 5'd0, 4'b1111, 4'b0000, 4'b1111);
        check_header(3'd1, 5'd13, 4'b1111, "invalid sreg id");
        for (i = 0; i < NUM_LANES; i = i + 1)
            check_u32(result_out[i*32 +: 32], 32'd0, "invalid sreg returns zero");
        tick();

        if (fail_count == 0) begin
            $display("========================================");
            $display("[PASS] tb_sm_special_reg: %0d checks", pass_count);
            $display("========================================");
        end else begin
            $display("========================================");
            $display("[FAIL] tb_sm_special_reg: %0d failures / %0d checks", fail_count, pass_count + fail_count);
            $display("========================================");
            $fatal(1);
        end

        $finish;
    end

endmodule
