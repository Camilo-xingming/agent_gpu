//============================================================================
// Testbench: texture_unit — Gemini #2 Texture/Surface Verification
// Tests: TEX 1D/2D/3D point sampling, wrap modes, TXQ, SULD, SUST
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

    // Memory responder: 1-cycle latency response to mem_req
    // Returns a predictable pattern based on address
    reg mem_req_d;
    reg [31:0] mem_addr_d;
    reg mem_write_d;

    // Captured store for verification
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
                // Store — capture and ack
                last_store_addr <= mem_addr_d;
                last_store_wdata <= mem_wdata;
                store_captured <= 1'b1;
                mem_rdata <= 128'b0;
            end else begin
                // Load — return RGBA8 pattern based on addr low bits
                // Addr[7:0] → R, Addr[15:8] → G, const B=0xAA, A=0xFF
                mem_rdata <= {96'b0, 8'hFF, 8'hAA, mem_addr_d[15:8], mem_addr_d[7:0]};
            end
        end
    end

    //------------------------------------------------------------------------
    // Helper: issue a texture op and wait for valid_out
    //------------------------------------------------------------------------
    task issue_and_wait;
        input [5:0]  op;
        input [5:0]  fn;
        input [31:0] cs, ct, cr;
        input integer timeout;
        integer cnt;
        begin
            @(posedge clk);
            opcode <= op;
            func <= fn;
            coord_s <= cs;
            coord_t <= ct;
            coord_r <= cr;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;

            cnt = 0;
            while (!valid_out && cnt < timeout) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
            #1; // NBA settle
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

    //------------------------------------------------------------------------
    // Tests
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_texture_unit.vcd");
        $dumpvars(0, tb_texture_unit);

        // Defaults
        rst_n = 0;
        valid_in = 0;
        opcode = 0; func = 0;
        coord_s = 0; coord_t = 0; coord_r = 0; coord_q = 0;
        lod = 0; dsdx = 0; dsdy = 0; dtdx = 0; dtdy = 0;
        tex_base_addr = 32'h0001_0000;
        tex_width = 16'd256;
        tex_height = 16'd256;
        tex_depth = 16'd1;
        tex_format = 4'h0; // RGBA8_UNORM
        tex_filter = 4'h0; // FILTER_POINT
        tex_wrap_s = 4'h1; // WRAP_CLAMP
        tex_wrap_t = 4'h1;
        tex_wrap_r = 4'h1;
        num_mip_levels = 4'd1;
        store_data = 128'h0;
        mem_ready = 1;
        mem_rdata = 0;
        mem_valid = 0;
        mem_req_d = 0;
        store_captured = 0;

        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        //====================================================================
        // TXQ tests (no memory access needed)
        //====================================================================
        $display("\n--- TXQ Tests ---");

        // Test: TXQ width
        tex_width = 16'd512;
        tex_height = 16'd256;
        tex_depth = 16'd64;
        num_mip_levels = 4'd8;

        issue_and_wait(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd512}, "TXQ width=512");

        // Test: TXQ height
        issue_and_wait(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0, 50);
        check_result({96'b0, 32'd256}, "TXQ height=256");

        // Test: TXQ depth
        issue_and_wait(`OP_TXQ, `TXQ_DEPTH, 0, 0, 0, 50);
        check_result({96'b0, 32'd64}, "TXQ depth=64");

        // Test: TXQ levels
        issue_and_wait(`OP_TXQ, `TXQ_LEVELS, 0, 0, 0, 50);
        check_result({124'b0, 4'd8}, "TXQ levels=8");

        //====================================================================
        // TEX 1D point sampling
        //====================================================================
        $display("\n--- TEX 1D Tests ---");
        tex_base_addr = 32'h0002_0000;
        tex_width = 16'd256;
        tex_height = 16'd1;
        tex_depth = 16'd1;
        tex_filter = 4'h0; // POINT
        tex_wrap_s = 4'h1; // CLAMP

        // TEX 1D at coord_s=5 (clamp, within range)
        // addr = base + wrap(5, 256, clamp) = 0x20000 + 5 = 0x20005
        // mem returns RGBA8: R=addr[7:0]=0x05, G=addr[15:8]=0x00, B=0xAA, A=0xFF
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd5, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000005}, "TEX 1D s=5 point");

        // TEX 1D at coord_s=100
        // addr = base + 100 = 0x20064
        // R=0x64, G=0x00
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd100, 0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000064}, "TEX 1D s=100 point");

        //====================================================================
        // TEX 1D wrap modes
        //====================================================================
        $display("\n--- TEX 1D Wrap Mode Tests ---");

        // CLAMP: coord=-1 (negative) should clamp to 0
        tex_wrap_s = 4'h1; // CLAMP
        issue_and_wait(`OP_TEX, `TEX_1D, 32'hFFFFFFFF, 0, 0, 50);
        // clamp(-1, 256) = 0 (negative check). addr = base + 0 = 0x20000
        // R=0x00, G=0x00
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 1D clamp negative");

        // CLAMP: coord=300 (> size=256) should clamp to 255
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd300, 0, 0, 50);
        // clamp(300, 256) = 255. addr = base + 255 = 0x200FF
        // R=0xFF, G=0x00
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000FF}, "TEX 1D clamp overflow");

        // REPEAT: coord=260 with size=256 → 260 % 256 = 4
        tex_wrap_s = 4'h0; // REPEAT
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd260, 0, 0, 50);
        // repeat(260, 256) = 4. addr = base + 4 = 0x20004
        // R=0x04, G=0x00
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000004}, "TEX 1D repeat 260%%256=4");

        // MIRROR: coord=300 with size=256 → 300 % 512 = 300, 300 >= 256 → 512-300-1 = 211
        tex_wrap_s = 4'h2; // MIRROR
        issue_and_wait(`OP_TEX, `TEX_1D, 32'd300, 0, 0, 50);
        // mirror(300, 256) = 512-300-1 = 211 = 0xD3. addr = base + 0xD3 = 0x200D3
        // R=0xD3, G=0x00
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h000000D3}, "TEX 1D mirror 300→211");

        //====================================================================
        // TEX 2D point sampling
        //====================================================================
        $display("\n--- TEX 2D Tests ---");
        tex_base_addr = 32'h0003_0000;
        tex_width = 16'd64;
        tex_height = 16'd64;
        tex_filter = 4'h0; // POINT
        tex_wrap_s = 4'h1; // CLAMP
        tex_wrap_t = 4'h1; // CLAMP

        // TEX 2D at (10, 20) with 64-wide RGBA8 (pitch = 64*4 = 256, bpp=4)
        // addr = base + 20*256 + 10*4 = 0x30000 + 5120 + 40 = 0x30000 + 0x1400 + 0x28 = 0x31428
        // R = addr[7:0] = 0x28, G = addr[15:8] = 0x01
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd10, 32'd20, 0, 50);
        check_result_r(32'h00000028, "TEX 2D (10,20) R channel");

        // TEX 2D at (0, 0)
        // addr = base + 0 + 0 = 0x30000
        // R = 0x00, G = 0x00
        issue_and_wait(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 0, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h00000000, 32'h00000000}, "TEX 2D (0,0) all channels");

        //====================================================================
        // TEX 3D point sampling
        //====================================================================
        $display("\n--- TEX 3D Tests ---");
        tex_base_addr = 32'h0004_0000;
        tex_width = 16'd16;
        tex_height = 16'd16;
        tex_depth = 16'd16;
        tex_wrap_s = 4'h1; // CLAMP
        tex_wrap_t = 4'h1;
        tex_wrap_r = 4'h1;

        // TEX 3D at (1, 2, 3)
        // addr = base + wrap(3,16,clamp)*(16*16*4) + wrap(2,16,clamp)*(16*4) + wrap(1,16,clamp)*4
        //      = 0x40000 + 3*1024 + 2*64 + 1*4 = 0x40000 + 0xC00 + 0x80 + 0x4 = 0x40C84
        // R = 0x84, G = 0x0C
        issue_and_wait(`OP_TEX, `TEX_3D, 32'd1, 32'd2, 32'd3, 50);
        check_result({32'h000000FF, 32'h000000AA, 32'h0000000C, 32'h00000084}, "TEX 3D (1,2,3)");

        //====================================================================
        // Surface Load (SULD)
        //====================================================================
        $display("\n--- Surface Load/Store Tests ---");
        tex_base_addr = 32'h0005_0000;
        tex_width = 16'd32;

        // SULD at (5, 3): addr = base + 3*(32*4) + 5*4 = 0x50000 + 384 + 20 = 0x50194
        // R = 0x94, G = 0x01
        issue_and_wait(`OP_SULD, 6'b0, 32'd5, 32'd3, 0, 50);
        check_result({96'b0, 8'hFF, 8'hAA, 8'h01, 8'h94}, "SULD (5,3)");

        //====================================================================
        // Surface Store (SUST)
        //====================================================================
        store_data = 128'hDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0;
        store_captured = 0;

        issue_and_wait(`OP_SUST, 6'b0, 32'd2, 32'd1, 0, 50);
        // addr = base + 1*(32*4) + 2*4 = 0x50000 + 128 + 8 = 0x50088
        test_num = test_num + 1;
        if (store_captured && last_store_addr == 32'h0005_0088) begin
            $display("PASS test %0d: SUST addr=0x%h correct", test_num, last_store_addr);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST addr got 0x%h, expected 0x50088, captured=%b", test_num, last_store_addr, store_captured);
            fail_count = fail_count + 1;
        end

        // Check stored data
        test_num = test_num + 1;
        if (last_store_wdata === 128'hDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0) begin
            $display("PASS test %0d: SUST data correct", test_num);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL test %0d: SUST data got %h", test_num, last_store_wdata);
            fail_count = fail_count + 1;
        end

        //====================================================================
        // Summary
        //====================================================================
        $display("\n========================================");
        $display("Texture Unit Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $display("========================================");
        $finish;
    end

    // Timeout
    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
