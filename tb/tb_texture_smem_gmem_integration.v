`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_texture_smem_gmem_integration;
    localparam NUM_LANES   = 8;
    localparam SIMD_WIDTH  = NUM_LANES * 32;
    localparam SMEM_ADDR_W = 8;

    reg clk;
    reg rst_n;

    reg [5:0]   opcode;
    reg [5:0]   func;
    reg         valid_in;
    reg [31:0]  coord_s;
    reg [31:0]  coord_t;
    reg [31:0]  coord_r;
    reg [31:0]  coord_q;
    reg [31:0]  lod;
    reg [31:0]  dsdx;
    reg [31:0]  dsdy;
    reg [31:0]  dtdx;
    reg [31:0]  dtdy;
    reg [31:0]  tex_base_addr;
    reg [15:0]  tex_width;
    reg [15:0]  tex_height;
    reg [15:0]  tex_depth;
    reg [3:0]   tex_format;
    reg [3:0]   tex_filter;
    reg [3:0]   tex_wrap_s;
    reg [3:0]   tex_wrap_t;
    reg [3:0]   tex_wrap_r;
    reg [3:0]   num_mip_levels;
    reg [127:0] store_data;

    wire         tex_mem_req;
    wire         tex_mem_write;
    wire [31:0]  tex_mem_addr;
    wire [127:0] tex_mem_wdata;
    wire [127:0] tex_result;
    wire         tex_valid_out;
    wire         tex_busy;

    wire         tex_mem_ready;
    wire [127:0] tex_mem_rdata;
    wire         tex_mem_valid;

    reg                     normal_req_valid;
    reg                     normal_req_write;
    reg [NUM_LANES*32-1:0]  normal_req_addr;
    reg [SIMD_WIDTH-1:0]    normal_req_wdata;
    reg [NUM_LANES-1:0]     normal_req_mask;

    reg                     atomic_req;
    reg                     atomic_write;
    reg [5:0]               atomic_lane;
    reg [31:0]              atomic_addr;
    reg [31:0]              atomic_wdata;
    reg                     atomic_shared_pending;
    reg                     smem_atomic_resp_valid;
    reg [31:0]              smem_atomic_resp_rdata;

    wire                    atomic_ready;
    wire                    atomic_resp_read_valid;
    wire                    atomic_resp_write_valid;
    wire [SIMD_WIDTH-1:0]   atomic_rdata;
    wire                    atomic_pending;

    reg                     ace_req_valid;
    reg [31:0]              ace_req_addr;
    reg [4:0]               ace_req_size;

    wire                    ace_resp_valid;
    wire [127:0]            ace_resp_data;
    wire                    ace_pending;

    wire                    tex_ready;
    wire                    tex_resp_valid;
    wire [127:0]            tex_resp_data;
    wire                    tex_pending;

    wire                    gmem_req_valid;
    wire                    gmem_req_write;
    wire [NUM_LANES*32-1:0] gmem_req_addr;
    wire [SIMD_WIDTH-1:0]   gmem_req_wdata;
    wire [NUM_LANES-1:0]    gmem_req_mask;

    reg                     gmem_req_ready;
    reg                     gmem_resp_valid;
    reg [SIMD_WIDTH-1:0]    gmem_resp_rdata;

    wire                    normal_resp_valid;
    wire [SIMD_WIDTH-1:0]   normal_resp_rdata;

    reg                             smem_req_valid;
    reg                             smem_req_write;
    reg [NUM_LANES*SMEM_ADDR_W-1:0] smem_req_addr;
    reg [NUM_LANES*32-1:0]          smem_req_wdata;
    reg [NUM_LANES-1:0]             smem_req_mask;
    wire                            smem_resp_valid;
    wire [NUM_LANES*32-1:0]         smem_resp_rdata;
    wire                            smem_bank_conflict;

    integer pass_count;
    integer fail_count;
    integer timeout;

    reg        gmem_req_d;
    reg [31:0] gmem_addr_d;
    reg        saw_gmem_req;
    reg [31:0] observed_tex_addr;

    texture_unit dut_texture (
        .clk(clk),
        .rst_n(rst_n),
        .opcode(opcode),
        .func(func),
        .valid_in(valid_in),
        .coord_s(coord_s),
        .coord_t(coord_t),
        .coord_r(coord_r),
        .coord_q(coord_q),
        .lod(lod),
        .dsdx(dsdx),
        .dsdy(dsdy),
        .dtdx(dtdx),
        .dtdy(dtdy),
        .tex_base_addr(tex_base_addr),
        .tex_width(tex_width),
        .tex_height(tex_height),
        .tex_depth(tex_depth),
        .tex_format(tex_format),
        .tex_filter(tex_filter),
        .tex_wrap_s(tex_wrap_s),
        .tex_wrap_t(tex_wrap_t),
        .tex_wrap_r(tex_wrap_r),
        .num_mip_levels(num_mip_levels),
        .store_data(store_data),
        .mem_req(tex_mem_req),
        .mem_write(tex_mem_write),
        .mem_addr(tex_mem_addr),
        .mem_wdata(tex_mem_wdata),
        .mem_ready(tex_mem_ready),
        .mem_rdata(tex_mem_rdata),
        .mem_valid(tex_mem_valid),
        .result(tex_result),
        .valid_out(tex_valid_out),
        .busy(tex_busy)
    );

    sm_gmem_arbiter #(
        .NUM_LANES(NUM_LANES),
        .SIMD_WIDTH(SIMD_WIDTH)
    ) dut_arbiter (
        .clk(clk),
        .rst_n(rst_n),
        .normal_req_valid(normal_req_valid),
        .normal_req_write(normal_req_write),
        .normal_req_addr(normal_req_addr),
        .normal_req_wdata(normal_req_wdata),
        .normal_req_mask(normal_req_mask),
        .atomic_req(atomic_req),
        .atomic_write(atomic_write),
        .atomic_lane(atomic_lane),
        .atomic_addr(atomic_addr),
        .atomic_wdata(atomic_wdata),
        .atomic_shared_pending(atomic_shared_pending),
        .smem_atomic_resp_valid(smem_atomic_resp_valid),
        .smem_atomic_resp_rdata(smem_atomic_resp_rdata),
        .atomic_ready(atomic_ready),
        .atomic_resp_read_valid(atomic_resp_read_valid),
        .atomic_resp_write_valid(atomic_resp_write_valid),
        .atomic_rdata(atomic_rdata),
        .atomic_pending(atomic_pending),
        .ace_req_valid(ace_req_valid),
        .ace_req_addr(ace_req_addr),
        .ace_req_size(ace_req_size),
        .ace_resp_valid(ace_resp_valid),
        .ace_resp_data(ace_resp_data),
        .ace_pending(ace_pending),
        .tex_req(tex_mem_req),
        .tex_addr(tex_mem_addr),
        .tex_write(tex_mem_write),
        .tex_wdata(tex_mem_wdata),
        .tex_ready(tex_ready),
        .tex_resp_valid(tex_resp_valid),
        .tex_resp_data(tex_resp_data),
        .tex_pending(tex_pending),
        .gmem_req_valid(gmem_req_valid),
        .gmem_req_write(gmem_req_write),
        .gmem_req_addr(gmem_req_addr),
        .gmem_req_wdata(gmem_req_wdata),
        .gmem_req_mask(gmem_req_mask),
        .gmem_req_ready(gmem_req_ready),
        .gmem_resp_valid(gmem_resp_valid),
        .gmem_resp_rdata(gmem_resp_rdata),
        .normal_resp_valid(normal_resp_valid),
        .normal_resp_rdata(normal_resp_rdata)
    );

    shared_memory #(
        .SIZE_KB(1),
        .NUM_BANKS(NUM_LANES),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(SMEM_ADDR_W)
    ) dut_smem (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(smem_req_valid),
        .req_write(smem_req_write),
        .req_addr(smem_req_addr),
        .req_wdata(smem_req_wdata),
        .req_mask(smem_req_mask),
        .resp_valid(smem_resp_valid),
        .resp_rdata(smem_resp_rdata),
        .bank_conflict(smem_bank_conflict),
        .atomic_req_valid(1'b0),
        .atomic_req_write(1'b0),
        .atomic_req_addr({SMEM_ADDR_W{1'b0}}),
        .atomic_req_wdata(32'b0),
        .atomic_req_mask(1'b0),
        .atomic_resp_valid(),
        .atomic_resp_rdata(),
        .async_wr_en(1'b0),
        .async_wr_addr({SMEM_ADDR_W{1'b0}}),
        .async_wr_data(128'b0),
        .async_wr_size(5'd0),
        .wgmma_rd_en(1'b0),
        .wgmma_rd_addr_a({SMEM_ADDR_W{1'b0}}),
        .wgmma_rd_addr_b({SMEM_ADDR_W{1'b0}}),
        .wgmma_rd_data_a(),
        .wgmma_rd_data_b(),
        .wgmma_rd_valid(),
        .ace_rd_en(1'b0),
        .ace_rd_addr({SMEM_ADDR_W{1'b0}}),
        .ace_rd_data(),
        .ace_rd_valid()
    );

    assign tex_mem_ready = tex_ready;
    assign tex_mem_valid = tex_resp_valid;
    assign tex_mem_rdata = tex_resp_data;

    always #5 clk = ~clk;

    function [NUM_LANES*SMEM_ADDR_W-1:0] smem_addr_lane0;
        input [SMEM_ADDR_W-1:0] addr;
        begin
            smem_addr_lane0 = {NUM_LANES*SMEM_ADDR_W{1'b0}};
            smem_addr_lane0[SMEM_ADDR_W-1:0] = addr;
        end
    endfunction

    function [NUM_LANES*32-1:0] smem_wdata_lane0;
        input [31:0] data;
        begin
            smem_wdata_lane0 = {NUM_LANES*32{1'b0}};
            smem_wdata_lane0[31:0] = data;
        end
    endfunction

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

    task clear_side_inputs;
        begin
            normal_req_valid = 1'b0;
            normal_req_write = 1'b0;
            normal_req_addr  = {NUM_LANES*32{1'b0}};
            normal_req_wdata = {SIMD_WIDTH{1'b0}};
            normal_req_mask  = {NUM_LANES{1'b0}};

            atomic_req = 1'b0;
            atomic_write = 1'b0;
            atomic_lane = 6'd0;
            atomic_addr = 32'b0;
            atomic_wdata = 32'b0;
            atomic_shared_pending = 1'b0;
            smem_atomic_resp_valid = 1'b0;
            smem_atomic_resp_rdata = 32'b0;

            ace_req_valid = 1'b0;
            ace_req_addr = 32'b0;
            ace_req_size = 5'd0;

            smem_req_valid = 1'b0;
            smem_req_write = 1'b0;
            smem_req_addr  = {NUM_LANES*SMEM_ADDR_W{1'b0}};
            smem_req_wdata = {NUM_LANES*32{1'b0}};
            smem_req_mask  = {NUM_LANES{1'b0}};
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gmem_req_d <= 1'b0;
            gmem_addr_d <= 32'b0;
            gmem_resp_valid <= 1'b0;
            gmem_resp_rdata <= {SIMD_WIDTH{1'b0}};
            saw_gmem_req <= 1'b0;
            observed_tex_addr <= 32'b0;
        end else begin
            gmem_resp_valid <= 1'b0;
            gmem_req_d <= gmem_req_valid && !gmem_req_write && gmem_req_ready;
            gmem_addr_d <= gmem_req_addr[31:0];

            if (gmem_req_valid && !gmem_req_write && gmem_req_ready) begin
                saw_gmem_req <= 1'b1;
                observed_tex_addr <= gmem_req_addr[31:0];
            end

            if (gmem_req_d) begin
                gmem_resp_valid <= 1'b1;
                gmem_resp_rdata <= {{(SIMD_WIDTH-128){1'b0}},
                    {96'b0, 8'hFF, 8'hAA, gmem_addr_d[15:8], gmem_addr_d[7:0]}};
            end
        end
    end

    initial begin
        reg [127:0] expected_tex_result;
        reg [7:0] smem_addr;

        clk = 1'b0;
        rst_n = 1'b0;
        pass_count = 0;
        fail_count = 0;

        opcode = 6'b0;
        func = 6'b0;
        valid_in = 1'b0;
        coord_s = 32'b0;
        coord_t = 32'b0;
        coord_r = 32'b0;
        coord_q = 32'b0;
        lod = 32'b0;
        dsdx = 32'b0;
        dsdy = 32'b0;
        dtdx = 32'b0;
        dtdy = 32'b0;
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd64;
        tex_height = 16'd64;
        tex_depth = 16'd1;
        tex_format = 4'h0;
        tex_filter = 4'h0;
        tex_wrap_s = 4'h1;
        tex_wrap_t = 4'h1;
        tex_wrap_r = 4'h1;
        num_mip_levels = 4'd1;
        store_data = 128'b0;

        gmem_req_ready = 1'b1;

        clear_side_inputs();

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        tick();

        // Step 1: issue texture fetch request.
        @(posedge clk);
        opcode <= `OP_TEX;
        func <= `TEX_2D;
        coord_s <= (32'd7 << 8);
        coord_t <= (32'd3 << 8);
        coord_r <= 32'b0;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;

        // Step 2: verify request reached GMEM through arbiter (cache miss path).
        timeout = 0;
        while (!saw_gmem_req && timeout < 80) begin
            tick();
            timeout = timeout + 1;
        end
        check(saw_gmem_req, "texture request reached sm_gmem_arbiter/gmem");
        check(tex_pending, "arbiter tracks outstanding texture request");

        // Step 3: wait texture result and verify data came back from GMEM model.
        timeout = 0;
        while (!tex_valid_out && timeout < 120) begin
            tick();
            timeout = timeout + 1;
        end
        check(tex_valid_out, "texture unit returned valid_out after gmem response");

        expected_tex_result = {32'h000000FF, 32'h000000AA, {24'd0, observed_tex_addr[15:8]}, {24'd0, observed_tex_addr[7:0]}};
        check(tex_result == expected_tex_result, "texture payload matches gmem response pattern");

        // Step 4: shared memory writeback and readback.
        smem_addr = 8'h13;

        smem_req_addr = smem_addr_lane0(smem_addr);
        smem_req_wdata = smem_wdata_lane0(tex_result[31:0]);
        smem_req_mask = 8'b0000_0001;
        smem_req_write = 1'b1;
        smem_req_valid = 1'b1;
        tick();
        check(smem_resp_valid, "shared memory writeback acknowledged");

        smem_req_valid = 1'b0;
        smem_req_write = 1'b0;
        tick();

        smem_req_addr = smem_addr_lane0(smem_addr);
        smem_req_mask = 8'b0000_0001;
        smem_req_write = 1'b0;
        smem_req_valid = 1'b1;
        tick();
        check(smem_resp_valid, "shared memory readback acknowledged");
        check(!smem_bank_conflict, "no bank conflict for single-lane writeback/readback");
        check(smem_resp_rdata[31:0] == tex_result[31:0], "shared memory readback equals texture writeback payload");

        smem_req_valid = 1'b0;
        tick();

        if (fail_count == 0) begin
            $display("[PASS] tb_texture_smem_gmem_integration: %0d checks", pass_count);
        end else begin
            $display("[FAIL] tb_texture_smem_gmem_integration: %0d passed, %0d failed", pass_count, fail_count);
            $fatal(1, "tb_texture_smem_gmem_integration failed");
        end

        $finish;
    end
endmodule
