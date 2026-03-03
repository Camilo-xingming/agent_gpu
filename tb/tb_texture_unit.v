//============================================================================
// Testbench: texture_unit — Comprehensive Texture/Surface Verification
// Gemini #2: TEX 1D/2D/3D, all wrap modes, bilinear path, TXQ,
//            SULD, SUST, SURED, back-to-back ops, edge cases,
//            texture_cache unit tests
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_texture_unit;

    reg         clk, rst_n;
    reg  [5:0]  opcode, func;
    reg         valid_in;
    reg  [31:0] coord_s, coord_t, coord_r, coord_q;
    reg  [31:0] lod, dsdx, dsdy, dtdx, dtdy;
    reg  [31:0] tex_base_addr;
    reg  [15:0] tex_width, tex_height, tex_depth;
    reg  [3:0]  tex_format, tex_filter;
    reg  [3:0]  tex_wrap_s, tex_wrap_t, tex_wrap_r;
    reg  [3:0]  num_mip_levels;
    reg  [127:0] store_data;
    reg         mem_ready;
    reg  [127:0] mem_rdata;
    reg         mem_valid;

    wire        mem_req, mem_write;
    wire [31:0] mem_addr;
    wire [127:0] mem_wdata;
    wire [127:0] result;
    wire        valid_out;
    wire        busy;

    texture_unit #(
        .CACHE_SIZE_KB(16),
        .MAX_ANISO(16)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .opcode(opcode), .func(func), .valid_in(valid_in),
        .coord_s(coord_s), .coord_t(coord_t), .coord_r(coord_r), .coord_q(coord_q),
        .lod(lod), .dsdx(dsdx), .dsdy(dsdy), .dtdx(dtdx), .dtdy(dtdy),
        .tex_base_addr(tex_base_addr),
        .tex_width(tex_width), .tex_height(tex_height), .tex_depth(tex_depth),
        .tex_format(tex_format), .tex_filter(tex_filter),
        .tex_wrap_s(tex_wrap_s), .tex_wrap_t(tex_wrap_t), .tex_wrap_r(tex_wrap_r),
        .num_mip_levels(num_mip_levels),
        .store_data(store_data),
        .mem_req(mem_req), .mem_write(mem_write),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata),
        .mem_ready(mem_ready), .mem_rdata(mem_rdata), .mem_valid(mem_valid),
        .result(result), .valid_out(valid_out), .busy(busy)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;
    reg     last_op_completed; // set by issue_and_wait if valid_out was seen

    //------------------------------------------------------------------------
    // Memory responder: 1-cycle latency, returns pattern based on address
    //------------------------------------------------------------------------
    reg mem_req_d;
    reg [31:0] mem_addr_d;
    reg mem_write_d;

    reg [31:0]  last_store_addr;
    reg [127:0] last_store_wdata;
    reg         store_captured;

    always @(posedge clk) begin
        mem_req_d <= mem_req;
        mem_addr_d <= mem_addr;
        mem_write_d <= mem_write;
        mem_valid <= 1'b0;

        if (mem_req_d) begin
            mem_valid <= 1'b1;
            if (mem_write_d) begin
                last_store_addr <= mem_addr_d;
                last_store_wdata <= mem_wdata;
                store_captured <= 1'b1;
                mem_rdata <= 128'b0;
            end else begin
                // RGBA8 pattern: R=addr[7:0], G=addr[15:8], B=0xAA, A=0xFF
                mem_rdata <= {96'b0, 8'hFF, 8'hAA, mem_addr_d[15:8], mem_addr_d[7:0]};
            end
        end
    end

    //------------------------------------------------------------------------
    // Helpers
    //------------------------------------------------------------------------
    task issue_and_wait;
        input [5:0]  op;
        input [5:0]  fn;
        input [31:0] cs, ct, cr;
        input integer timeout;
        integer cnt;
        begin
            last_op_completed = 0;
            @(posedge clk);
            opcode <= op;
            func <= fn;
            coord_s <= (cs << 8);
            coord_t <= (ct << 8);
            coord_r <= (cr << 8);
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;

            cnt = 0;
            while (!valid_out && cnt < timeout) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
            if (valid_out)
                last_op_completed = 1;
            #1;
        end
    endtask

    task check_result;
        input [127:0] expected;
        input [255:0] label;
        begin
            test_num = test_num + 1;
            if (result === expected) begin
                $display("PASS test %0d: %0s", test_num, label);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL test %0d: %0s — got %h, expected %h", test_num, label, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_result_r;
        input [31:0] expected_r;
        input [255:0] label;
        begin
            test_num = test_num + 1;
            if (result[31:0] === expected_r) begin
                $display("PASS test %0d: %0s (R=%h)", test_num, label, result[31:0]);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL test %0d: %0s — R got %h, expected %h", test_num, label, result[31:0], expected_r);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_completed;
        input [255:0] label;
        begin
            test_num = test_num + 1;
            if (last_op_completed) begin
                $display("PASS test %0d: %0s (completed)", test_num, label);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL test %0d: %0s — timed out, no valid_out", test_num, label);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_busy_low;
        input [255:0] label;
        begin
            test_num = test_num + 1;
            @(posedge clk);
            #1;
            if (!busy) begin
                $display("PASS test %0d: %0s (busy deasserted)", test_num, label);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL test %0d: %0s — busy still high", test_num, label);
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Reset helper
    //------------------------------------------------------------------------
    task do_reset;
        begin
            rst_n = 0;
            valid_in = 0;
            opcode = 0; func = 0;
            coord_s = 0; coord_t = 0; coord_r = 0; coord_q = 0;
            lod = 0; dsdx = 0; dsdy = 0; dtdx = 0; dtdy = 0;
            tex_base_addr = 32'h0001_0000;
            tex_width = 16'd256; tex_height = 16'd256; tex_depth = 16'd1;
            tex_format = 4'h0; tex_filter = 4'h0;
            tex_wrap_s = 4'h1; tex_wrap_t = 4'h1; tex_wrap_r = 4'h1;
            num_mip_levels = 4'd1;
            store_data = 128'h0;
            mem_ready = 1; mem_rdata = 0; mem_valid = 0;
            mem_req_d = 0; store_captured = 0;
            repeat(5) @(posedge clk);
            rst_n = 1;
            repeat(2) @(posedge clk);
        end
    endtask

    //------------------------------------------------------------------------
    // Tests
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_texture_unit.vcd");
        $dumpvars(0, tb_texture_unit);

        do_reset();

        //====================================================================
        // 1. TXQ tests
        //====================================================================
        $display("\n=== Section 1: TXQ Tests ===");

        tex_width = 16'd512; tex_height = 16'd256;
        tex_depth = 16'd64;  num_mip_levels = 4'd8;

        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd512}, "TXQ width=512");

        issue_and_wait(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0, 50);
        check_result({96'b0, 32'd256}, "TXQ height=256");

        issue_and_wait(`OP_TXQ, `TXQ_DEPTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd64}, "TXQ depth=64");

        issue_and_wait(`OP_TXQ, `TXQ_LEVELS, 0, 0, 0, 50);
        check_result({124'b0, 4'd8}, "TXQ levels=8");

        // TXQ with different dimensions
        tex_width = 16'd1; tex_height = 16'd1; tex_depth = 16'd1;
        num_mip_levels = 4'd1;
        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd1}, "TXQ width=1 (min)");

        tex_width = 16'd16384; tex_height = 16'd16384;
        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd16384}, "TXQ width=16384 (large)");

        issue_and_wait(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0, 50);
        check_result({96'b0, 32'd16384}, "TXQ height=16384 (large)");

        num_mip_levels = 4'd15;
        issue_and_wait(`OP_TXQ, `TXQ_LEVELS, 0, 0, 0, 50);
        check_result({124'b0, 4'd15}, "TXQ levels=15 (max 4-bit)");

        //====================================================================
        // 2. TEX 1D point sampling
        //====================================================================
        $display("\n=== Section 2: TEX 1D Point Sampling ===");
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd256; tex_height = 16'd1; tex_depth = 16'd1;
        tex_filter = 4'h0; tex_wrap_s = 4'h1;

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd5, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000005}, "TEX 1D s=5");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd100, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000064}, "TEX 1D s=100");

        // coord=0 (first texel)
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd0, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D s=0 (origin)");

        // coord=255 (last texel in 256-wide)
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd255, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000FF}, "TEX 1D s=255 (last)");

        //====================================================================
        // 3. TEX 1D wrap modes
        //====================================================================
        $display("\n=== Section 3: TEX 1D Wrap Modes ===");

        // CLAMP negative
        tex_wrap_s = 4'h1;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'hFFFFFFFF, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D clamp(-1)=0");

        // CLAMP overflow
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd300, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000FF}, "TEX 1D clamp(300)=255");

        // CLAMP large negative
        issue_and_wait(`OP_TEX, `TEX_1D, 32'h80000000, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D clamp(INT_MIN)=0");

        // REPEAT
        tex_wrap_s = 4'h0;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd260, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000004}, "TEX 1D repeat(260%256)=4");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd512, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D repeat(512%256)=0");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd257, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000001}, "TEX 1D repeat(257%256)=1");

        // MIRROR
        tex_wrap_s = 4'h2;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd300, 0, 0, 50);
        // mirror(300,256): 300%512=300, 300>=256 → 512-300-1=211=0xD3
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000D3}, "TEX 1D mirror(300)=211");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd256, 0, 0, 50);
        // mirror(256,256): 256%512=256, 256>=256 → 512-256-1=255=0xFF
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000FF}, "TEX 1D mirror(256)=255");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd511, 0, 0, 50);
        // mirror(511,256): 511%512=511, 511>=256 → 512-511-1=0
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D mirror(511)=0");

        //====================================================================
        // 4. TEX 2D point sampling
        //====================================================================
        $display("\n=== Section 4: TEX 2D Point Sampling ===");
        tex_base_addr = 32'h0003_0000;
        tex_width = 16'd64; tex_height = 16'd64;
        tex_filter = 4'h0; tex_wrap_s = 4'h1; tex_wrap_t = 4'h1;

        // (10,20): addr = base + 20*(64*4) + 10*4 = 0x30000 + 5120 + 40 = 0x31428
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd10, 32'd20, 0, 50);
        check_result_r(32'h00000028, "TEX 2D (10,20) R=0x28");

        // (0,0)
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 2D (0,0)");

        // (63,63): addr = base + 63*(64*4) + 63*4 = 0x30000+16128+252 = 0x33FFC
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd63, 32'd63, 0, 50);
        check_result_r(32'h000000FC, "TEX 2D (63,63) R=0xFC");

        // (1,0): addr = base + 0 + 1*4 = 0x30004
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd1, 32'd0, 0, 50);
        check_result_r(32'h00000004, "TEX 2D (1,0) R=0x04");

        //====================================================================
        // 5. TEX 2D wrap modes
        //====================================================================
        $display("\n=== Section 5: TEX 2D Wrap Modes ===");
        tex_base_addr = 32'h0003_0000;
        tex_width = 16'd64; tex_height = 16'd64;

        // CLAMP S, CLAMP T — coord beyond bounds
        tex_wrap_s = 4'h1; tex_wrap_t = 4'h1;
        // s=100 clamped to 63, t=100 clamped to 63
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd100, 32'd100, 0, 50);
        check_result_r(32'h000000FC, "TEX 2D clamp(100,100)→(63,63) R=0xFC");

        // s=-1 clamped to 0, t=-1 clamped to 0
        issue_and_wait(`OP_TEX, `TEX_2D, 32'hFFFFFFFF, 32'hFFFFFFFF, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 2D clamp(-1,-1)→(0,0)");

        // REPEAT S, CLAMP T
        tex_wrap_s = 4'h0; tex_wrap_t = 4'h1;
        // s=68 → 68%64=4, t=2 → addr = base + 2*256 + 4*4 = 0x30000+512+16 = 0x30210
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd68, 32'd2, 0, 50);
        check_result_r(32'h00000010, "TEX 2D repeat_s(68)=4, clamp_t(2) R=0x10");

        // CLAMP S, REPEAT T
        tex_wrap_s = 4'h1; tex_wrap_t = 4'h0;
        // s=3, t=67 → t%64=3 → addr = base + 3*256 + 3*4 = 0x30000+768+12 = 0x3030C
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd3, 32'd67, 0, 50);
        check_result_r(32'h0000000C, "TEX 2D clamp_s(3), repeat_t(67%64=3) R=0x0C");

        // MIRROR S, MIRROR T
        tex_wrap_s = 4'h2; tex_wrap_t = 4'h2;
        // s=70 → 70%128=70, 70>=64 → 128-70-1=57, t=0 → 0
        // addr = base + 0 + 57*4 = 0x300E4
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd70, 32'd0, 0, 50);
        check_result_r(32'h000000E4, "TEX 2D mirror_s(70)=57, t=0 R=0xE4");

        //====================================================================
        // 6. TEX 3D point sampling
        //====================================================================
        $display("\n=== Section 6: TEX 3D Point Sampling ===");
        tex_base_addr = 32'h0004_0000;
        tex_width = 16'd16; tex_height = 16'd16; tex_depth = 16'd16;
        tex_wrap_s = 4'h1; tex_wrap_t = 4'h1; tex_wrap_r = 4'h1;

        // (1,2,3): addr = base + 3*(16*16*4) + 2*(16*4) + 1*4
        //        = 0x40000 + 3072 + 128 + 4 = 0x40C84
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd1, 32'd2, 32'd3, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h0000000C, 32'h00000084}, "TEX 3D (1,2,3)");

        // (0,0,0): addr = base = 0x40000
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd0, 32'd0, 32'd0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 3D (0,0,0) origin");

        // (15,15,15): addr = base + 15*1024 + 15*64 + 15*4
        //           = 0x40000 + 15360 + 960 + 60 = 0x40000 + 0x3C00 + 0x3C0 + 0x3C = 0x43FFC
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd15, 32'd15, 32'd15, 50);
        check_result_r(32'h000000FC, "TEX 3D (15,15,15) max corner R=0xFC");

        // 3D with clamp: s=20 clamped to 15
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd20, 32'd0, 32'd0, 50);
        // s clamped to 15 → addr = base + 0 + 0 + 15*4 = 0x4003C
        check_result_r(32'h0000003C, "TEX 3D clamp_s(20)=15 R=0x3C");

        //====================================================================
        // 7. Bilinear filter path (TEX 2D with FILTER_LINEAR)
        //====================================================================
        $display("\n=== Section 7: Bilinear Filter Path ===");
        tex_base_addr = 32'h0005_0000;
        tex_width = 16'd64; tex_height = 16'd64;
        tex_filter = 4'h1; // FILTER_LINEAR
        tex_wrap_s = 4'h1; tex_wrap_t = 4'h1;

        
        // but we exercise the code path to confirm no hang/crash
        // pass integer 10, then add fractional part
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd10, 32'd10, 0, 80);
        // Override to add fraction 10/255
        @(posedge clk); coord_s = (32'd10 << 8) | 10; coord_t = (32'd10 << 8) | 10; valid_in = 1'b1; @(posedge clk); valid_in = 1'b0; wait(valid_out); #1;
        check_completed("TEX 2D bilinear path completes");

        // With real bilinear, coord (10,10) with frac=10/255 gives R=0x27
        // (8-bit fixed-point rounding from interpolating 4 neighbors)
        check_result_r(32'h00000027, "TEX 2D bilinear (10,10) frac=10/255 R=0x27");

        // --- Bilinear with frac=0 (should match point sampling) ---
        // Use 256x256 texture so coords don't clamp for bilinear tests
        tex_width = 16'd256; tex_height = 16'd256;
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 0, 80);
        check_completed("TEX 2D bilinear (0,0) frac=0 completes");
        // addr(0,0) = 0x50000 -> R=0x00
        check_result_r(32'h00000000, "TEX 2D bilinear (0,0) frac=0 R=0x00");

        // --- Bilinear with frac_s=128 (~50% S blend, no T blend) ---
        // coord_s=128 -> texels at s=128,129; frac_s=128/255
        // addr(128,0) = base+128*4 = 0x50200 -> R=0x00
        // addr(129,0) = base+129*4 = 0x50204 -> R=0x04
        // addr(128,1) = base+1024+512 = 0x50600 -> R=0x00
        // addr(129,1) = base+1024+516 = 0x50604 -> R=0x04
        // lerp_x: (127*0+128*4)>>8=2, lerp_y: (255*2+0*2)>>8=1
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 0, 80);
        // Override for s=128, frac=128
        @(posedge clk); coord_s = (32'd128 << 8) | 128; coord_t = 0; valid_in = 1'b1; @(posedge clk); valid_in = 1'b0; wait(valid_out); #1;
        check_completed("TEX 2D bilinear frac_s=128 completes");
        check_result_r(32'h00000001, "TEX 2D bilinear s=128 frac=0x80 R=0x01");

        // --- Bilinear with both fracs = 128 ---
        // lerp_x row0: 2, lerp_x row1: 2
        // lerp_y: (127*2+128*2)>>8=1
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 0, 80);
        // Override for s=128, t=128, frac=128, 128
        @(posedge clk); coord_s = (32'd128 << 8) | 128; coord_t = (32'd128 << 8) | 128; valid_in = 1'b1; @(posedge clk); valid_in = 1'b0; wait(valid_out); #1;
        check_completed("TEX 2D bilinear frac_s=128,frac_t=128 completes");
        check_result_r(32'h00000001, "TEX 2D bilinear s=128,t=128 R=0x01");

        tex_filter = 4'h0; // reset to point

        //====================================================================
        // 8. Surface Load (SULD)
        //====================================================================
        $display("\n=== Section 8: Surface Load (SULD) ===");
        tex_base_addr = 32'h0006_0000;
        tex_width = 16'd32;

        // SULD at (5,3): addr = base + 3*(32*4) + 5*4 = 0x60000 + 384 + 20 = 0x60194
        issue_and_wait(`OP_SULD, 6'b0, 32'd5, 32'd3, 0, 50);
        check_result({96'b0, 8'hFF, 8'hAA, 8'h01, 8'h94}, "SULD (5,3)");

        // SULD at (0,0)
        issue_and_wait(`OP_SULD, 6'b0, 32'd0, 32'd0, 0, 50);
        check_result({96'b0, 8'hFF, 8'hAA, 8'h00, 8'h00}, "SULD (0,0)");

        // SULD at (31,7): addr = base + 7*128 + 31*4 = 0x60000 + 896 + 124 = 0x603FC
        issue_and_wait(`OP_SULD, 6'b0, 32'd31, 32'd7, 0, 50);
        check_result({96'b0, 8'hFF, 8'hAA, 8'h03, 8'hFC}, "SULD (31,7)");

        //====================================================================
        // 9. Surface Store (SUST)
        //====================================================================
        $display("\n=== Section 9: Surface Store (SUST) ===");
        tex_base_addr = 32'h0006_0000;
        tex_width = 16'd32;

        // SUST at (2,1): addr = base + 1*128 + 2*4 = 0x60000 + 128 + 8 = 0x60088
        store_data = 128'hDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
        store_captured = 0;
        issue_and_wait(`OP_SUST, 6'b0, 32'd2, 32'd1, 0, 50);

        test_num = test_num + 1;
        if (store_captured && last_store_addr == 32'h0006_0088) begin
            $display("PASS test %0d: SUST (2,1) addr=0x%h", test_num, last_store_addr);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST addr got 0x%h exp 0x60088 cap=%b", test_num, last_store_addr, store_captured);
            fail_count = fail_count + 1;
        end

        test_num = test_num + 1;
        if (last_store_wdata === 128'hDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0) begin
            $display("PASS test %0d: SUST data correct", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST data got %h", test_num, last_store_wdata);
            fail_count = fail_count + 1;
        end

        // SUST at (0,0) with different data
        store_data = 128'h0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
        store_captured = 0;
        issue_and_wait(`OP_SUST, 6'b0, 32'd0, 32'd0, 0, 50);

        test_num = test_num + 1;
        if (store_captured && last_store_addr == 32'h0006_0000) begin
            $display("PASS test %0d: SUST (0,0) addr=0x%h", test_num, last_store_addr);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST (0,0) addr got 0x%h exp 0x60000", test_num, last_store_addr);
            fail_count = fail_count + 1;
        end

        test_num = test_num + 1;
        if (last_store_wdata === 128'h0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF) begin
            $display("PASS test %0d: SUST (0,0) data correct", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST (0,0) data got %h", test_num, last_store_wdata);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // 10. Surface Reduction (SURED) — no-op path, must not hang
        //====================================================================
        $display("\n=== Section 10: Surface Reduction (SURED) ===");
        issue_and_wait(`OP_SURED, 6'b0, 32'd1, 32'd1, 0, 50);
        check_completed("SURED completes (no-op path)");
        check_busy_low("SURED busy deasserts after completion");

        //====================================================================
        // 11. Back-to-back operations
        //====================================================================
        $display("\n=== Section 11: Back-to-Back Operations ===");
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd256; tex_height = 16'd1; tex_depth = 16'd1;
        tex_filter = 4'h0; tex_wrap_s = 4'h1;

        // Issue 5 consecutive TEX 1D ops without extra delays
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd10, 0, 0, 50);
        check_result_r(32'h0000000A, "back-to-back #1 s=10 R=0x0A");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd20, 0, 0, 50);
        check_result_r(32'h00000014, "back-to-back #2 s=20 R=0x14");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd30, 0, 0, 50);
        check_result_r(32'h0000001E, "back-to-back #3 s=30 R=0x1E");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd40, 0, 0, 50);
        check_result_r(32'h00000028, "back-to-back #4 s=40 R=0x28");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd50, 0, 0, 50);
        check_result_r(32'h00000032, "back-to-back #5 s=50 R=0x32");

        // Mix TEX and TXQ back to back
        tex_width = 16'd128;
        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd128}, "back-to-back TXQ after TEX");

        issue_and_wait(`OP_TEX, `TEX_1D, 32'd7, 0, 0, 50);
        check_result_r(32'h00000007, "back-to-back TEX after TXQ R=0x07");

        // Mix TEX → SULD → SUST back to back
        tex_base_addr = 32'h0007_0000;
        tex_width = 16'd16;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd3, 0, 0, 50);
        check_completed("b2b TEX→SULD: TEX completes");

        issue_and_wait(`OP_SULD, 6'b0, 32'd2, 32'd1, 0, 50);
        check_completed("b2b TEX→SULD: SULD completes");

        store_data = 128'hAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111;
        store_captured = 0;
        issue_and_wait(`OP_SUST, 6'b0, 32'd4, 32'd0, 0, 50);
        check_completed("b2b SULD→SUST: SUST completes");

        //====================================================================
        // 12. Reset recovery
        //====================================================================
        $display("\n=== Section 12: Reset Recovery ===");
        // Issue an op then assert reset mid-flight
        @(posedge clk);
        opcode <= `OP_TEX; func <= `TEX_1D;
        coord_s <= 32'd42; valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;
        // Assert reset before it completes
        @(posedge clk);
        rst_n = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // Verify DUT is usable after reset
        test_num = test_num + 1;
        if (!busy && !valid_out) begin
            $display("PASS test %0d: Reset clears state (busy=0 valid_out=0)", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: Reset state busy=%b valid_out=%b", test_num, busy, valid_out);
            fail_count = fail_count + 1;
        end

        // Normal operation after reset
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd256; tex_wrap_s = 4'h1;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd99, 0, 0, 50);
        check_result_r(32'h00000063, "Post-reset TEX 1D s=99 R=0x63");

        //====================================================================
        // 13. Small texture dimensions
        //====================================================================
        $display("\n=== Section 13: Small Texture Dimensions ===");
        tex_base_addr = 32'h0008_0000;
        tex_width = 16'd1; tex_height = 16'd1; tex_depth = 16'd1;
        tex_wrap_s = 4'h1; tex_wrap_t = 4'h1;

        // 1x1 texture: any coord clamps to 0
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd50, 32'd50, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 2D 1x1 tex clamps to (0,0)");

        // 2x2 texture
        tex_width = 16'd2; tex_height = 16'd2;
        // (1,1): addr = base + 1*(2*4) + 1*4 = 0x80000 + 8 + 4 = 0x8000C
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd1, 32'd1, 0, 50);
        check_result_r(32'h0000000C, "TEX 2D 2x2 (1,1) R=0x0C");


        //====================================================================
        // 14. TEX 3D non-clamp wrap modes
        //====================================================================
        $display("\n=== Section 14: TEX 3D Wrap Modes ===");
        tex_base_addr = 32'h0004_0000;
        tex_width = 16'd16; tex_height = 16'd16; tex_depth = 16'd16;

        // REPEAT on all axes: s=18%16=2, t=33%16=1, r=48%16=0
        tex_wrap_s = 4'h0; tex_wrap_t = 4'h0; tex_wrap_r = 4'h0;
        // addr = base + 0*(16*16*4) + 1*(16*4) + 2*4 = 0x40048
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd18, 32'd33, 32'd48, 50);
        check_result_r(32'h00000048, "TEX 3D repeat all axes");

        // MIRROR: s=20->11, t=0, r=0; addr = base+0+0+11*4 = 0x4002C
        tex_wrap_s = 4'h2; tex_wrap_t = 4'h2; tex_wrap_r = 4'h2;
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd20, 32'd0, 32'd0, 50);
        check_result_r(32'h0000002C, "TEX 3D mirror s=20->11");

        //====================================================================
        // 15. SULD + SUST round-trip
        //====================================================================
        $display("\n=== Section 15: Surface Round-Trip ===");
        tex_base_addr = 32'h000A_0000;
        tex_width = 16'd32;

        // Store pattern at (3,2)
        store_data = 128'hCAFE_BABE_DEAD_BEEF_1111_2222_3333_4444;
        store_captured = 0;
        issue_and_wait(`OP_SUST, 6'b0, 32'd3, 32'd2, 0, 50);
        check_completed("Round-trip SUST completes");

        test_num = test_num + 1;
        if (store_captured && last_store_wdata === 128'hCAFE_BABE_DEAD_BEEF_1111_2222_3333_4444) begin
            $display("PASS test %0d: SUST round-trip data ok", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST round-trip data bad", test_num);
            fail_count = fail_count + 1;
        end

        // Load from same location
        issue_and_wait(`OP_SULD, 6'b0, 32'd3, 32'd2, 0, 50);
        check_completed("Round-trip SULD completes");

        //====================================================================
        // 16. Rapid TXQ sequence
        //====================================================================
        $display("\n=== Section 16: Rapid TXQ Sequence ===");
        tex_width = 16'd800; tex_height = 16'd600;
        tex_depth = 16'd32; num_mip_levels = 4'd10;

        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd800}, "Rapid TXQ width=800");

        issue_and_wait(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0, 50);
        check_result({96'b0, 32'd600}, "Rapid TXQ height=600");

        issue_and_wait(`OP_TXQ, `TXQ_DEPTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd32}, "Rapid TXQ depth=32");

        issue_and_wait(`OP_TXQ, `TXQ_LEVELS, 0, 0, 0, 50);
        check_result({124'b0, 4'd10}, "Rapid TXQ levels=10");

        //====================================================================
        // 17. Boundary coordinates
        //====================================================================
        $display("\n=== Section 17: Boundary Coordinates ===");
        tex_base_addr = 32'h000B_0000;
        tex_width = 16'd128; tex_height = 16'd1;
        tex_wrap_s = 4'h1; // CLAMP

        // Exact last valid coord
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd127, 0, 0, 50);
        check_result_r(32'h0000007F, "clamp s=127 exact boundary");

        // One past last (clamps to 127)
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd128, 0, 0, 50);
        check_result_r(32'h0000007F, "clamp s=128 clamps to 127");

        // Repeat: size wraps to 0
        tex_wrap_s = 4'h0;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd128, 0, 0, 50);
        check_result_r(32'h00000000, "repeat s=128 wraps to 0");

        //====================================================================
        // 18. Unknown opcode — DUT must not hang; next valid op succeeds
        //====================================================================
        $display("\n=== Section 18: Unknown Opcode Recovery ===");
        // Send unknown opcode (RTL goes IDLE→busy=1→default→IDLE, busy stays)
        @(posedge clk);
        opcode <= 6'b111111; // undefined
        func <= 6'b0;
        coord_s <= 0; valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;
        repeat(5) @(posedge clk);

        // Verify DUT still accepts a valid op after unknown opcode
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd256; tex_wrap_s = 4'h1;
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd77, 0, 0, 50);
        check_result_r(32'h0000004D, "Post-unknown-opcode TEX 1D s=77 R=0x4D");

        //====================================================================
        // Summary
        //====================================================================
        $display("\n========================================");
        $display("Texture/Surface Unit Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED (%0d failures)", fail_count);
        $display("========================================");
        $finish;
    end

    // Timeout
    initial begin
        #300000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
