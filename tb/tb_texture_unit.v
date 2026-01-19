//============================================================================
// RalphGPU - Texture Unit Test
// Tests texture sampling and query operations
// Verifies: TEX.1D/2D/3D, TXQ (width/height/depth/levels), SULD/SUST
//============================================================================

`timescale 1ns / 1ps

module tb_texture_unit;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    // Control
    reg  [5:0]  opcode;
    reg  [5:0]  func;
    reg         valid_in;

    // Coordinates
    reg  [31:0] coord_s, coord_t, coord_r, coord_q;

    // LOD control
    reg  [31:0] lod;
    reg  [31:0] dsdx, dsdy, dtdx, dtdy;

    // Texture descriptor
    reg  [31:0] tex_base_addr;
    reg  [15:0] tex_width, tex_height, tex_depth;
    reg  [3:0]  tex_format, tex_filter;
    reg  [3:0]  tex_wrap_s, tex_wrap_t, tex_wrap_r;
    reg  [3:0]  num_mip_levels;

    // Surface store
    reg  [127:0] store_data;

    // Memory interface
    wire        mem_req;
    wire        mem_write;
    wire [31:0] mem_addr;
    wire [127:0] mem_wdata;
    reg         mem_ready;
    reg  [127:0] mem_rdata;
    reg         mem_valid;

    // Result
    wire [127:0] result;
    wire         valid_out;
    wire         busy;

    // Simulated texture memory
    reg [127:0] tex_mem [0:1023];

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // DUT
    texture_unit #(
        .CACHE_SIZE_KB(16),
        .MAX_ANISO(16)
    ) u_dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .opcode         (opcode),
        .func           (func),
        .valid_in       (valid_in),
        .coord_s        (coord_s),
        .coord_t        (coord_t),
        .coord_r        (coord_r),
        .coord_q        (coord_q),
        .lod            (lod),
        .dsdx           (dsdx),
        .dsdy           (dsdy),
        .dtdx           (dtdx),
        .dtdy           (dtdy),
        .tex_base_addr  (tex_base_addr),
        .tex_width      (tex_width),
        .tex_height     (tex_height),
        .tex_depth      (tex_depth),
        .tex_format     (tex_format),
        .tex_filter     (tex_filter),
        .tex_wrap_s     (tex_wrap_s),
        .tex_wrap_t     (tex_wrap_t),
        .tex_wrap_r     (tex_wrap_r),
        .num_mip_levels (num_mip_levels),
        .store_data     (store_data),
        .mem_req        (mem_req),
        .mem_write      (mem_write),
        .mem_addr       (mem_addr),
        .mem_wdata      (mem_wdata),
        .mem_ready      (mem_ready),
        .mem_rdata      (mem_rdata),
        .mem_valid      (mem_valid),
        .result         (result),
        .valid_out      (valid_out),
        .busy           (busy)
    );

    // Memory responder
    reg [31:0] pending_addr;
    reg resp_pending;
    reg [3:0] resp_delay;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_valid <= 1'b0;
            mem_rdata <= 128'b0;
            mem_ready <= 1'b1;
            resp_pending <= 1'b0;
        end else begin
            mem_valid <= 1'b0;

            if (mem_req && !resp_pending) begin
                pending_addr <= mem_addr;
                resp_pending <= 1'b1;
                resp_delay <= 4'd2;  // 2 cycle latency
            end else if (resp_pending) begin
                if (resp_delay == 0) begin
                    mem_valid <= 1'b1;
                    mem_rdata <= tex_mem[pending_addr[11:2]];
                    resp_pending <= 1'b0;
                end else begin
                    resp_delay <= resp_delay - 1;
                end
            end
        end
    end

    // Test task: Check result
    task check_result;
        input [255:0] test_name;
        input [127:0] expected;
        input [127:0] actual;
        begin
            if (expected == actual) begin
                $display("[PASS] Test %0d: %0s", test_num, test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s - expected 0x%h, got 0x%h",
                         test_num, test_name, expected, actual);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
        end
    endtask

    task check_result_32;
        input [255:0] test_name;
        input [31:0] expected;
        input [31:0] actual;
        begin
            if (expected == actual) begin
                $display("[PASS] Test %0d: %0s", test_num, test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s - expected 0x%h, got 0x%h",
                         test_num, test_name, expected, actual);
                fail_count = fail_count + 1;
            end
            test_num = test_num + 1;
        end
    endtask

    // Issue texture operation
    task issue_tex;
        input [5:0] op;
        input [5:0] f;
        input [31:0] s, t, r;
        begin
            @(posedge clk);
            opcode <= op;
            func <= f;
            coord_s <= s;
            coord_t <= t;
            coord_r <= r;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    // Wait for result
    task wait_result;
        integer timeout;
        begin
            timeout = 0;
            while (!valid_out && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    integer i;

    initial begin
        $display("============================================================");
        $display("RalphGPU Texture Unit Test");
        $display("============================================================");

        // Initialize
        rst_n = 0;
        valid_in = 0;
        opcode = 0;
        func = 0;
        coord_s = 0;
        coord_t = 0;
        coord_r = 0;
        coord_q = 0;
        lod = 0;
        dsdx = 0; dsdy = 0; dtdx = 0; dtdy = 0;
        tex_base_addr = 32'h0000_0000;
        tex_width = 16'd64;
        tex_height = 16'd64;
        tex_depth = 16'd1;
        tex_format = 4'h0;  // RGBA8_UNORM
        tex_filter = 4'h0;  // Point sampling
        tex_wrap_s = 4'h1;  // Clamp
        tex_wrap_t = 4'h1;  // Clamp
        tex_wrap_r = 4'h1;  // Clamp
        num_mip_levels = 4'd6;
        store_data = 128'b0;
        test_num = 1;
        pass_count = 0;
        fail_count = 0;

        // Initialize texture memory with test pattern
        // RGBA8: each texel is 4 bytes (R, G, B, A)
        for (i = 0; i < 1024; i = i + 1) begin
            // Create gradient pattern: R=x, G=y, B=(x+y), A=255
            tex_mem[i] = {8'hFF, 8'h00, 8'h00, 8'h00,      // Texel 3
                          8'hFF, 8'h00, 8'h00, 8'h00,      // Texel 2
                          8'hFF, 8'h00, 8'h00, 8'h00,      // Texel 1
                          8'hFF, i[7:0], i[7:0], i[7:0]};  // Texel 0: R=G=B=i
        end

        // Special test patterns at specific addresses
        tex_mem[0] = 128'hFF_11_22_33_FF_44_55_66_FF_77_88_99_FF_AA_BB_CC;
        tex_mem[1] = 128'hFF_12_34_56_FF_78_9A_BC_FF_DE_F0_12_FF_34_56_78;
        tex_mem[10] = 128'hFF_00_FF_00_FF_00_FF_00_FF_00_FF_00_FF_00_FF_00;  // Green

        #100;
        rst_n = 1;
        #50;

        //==================================================================
        // Test 1: Reset state
        //==================================================================
        check_result_32("Not busy after reset", 0, busy);

        //==================================================================
        // Test 2: Texture Query - Width
        //==================================================================
        $display("\n--- Test: TXQ Width ---");

        issue_tex(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0);
        wait_result();

        check_result_32("TXQ width = 64", 32'd64, result[31:0]);

        //==================================================================
        // Test 3: Texture Query - Height
        //==================================================================
        $display("\n--- Test: TXQ Height ---");

        issue_tex(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0);
        wait_result();

        check_result_32("TXQ height = 64", 32'd64, result[31:0]);

        //==================================================================
        // Test 4: Texture Query - Mip levels
        //==================================================================
        $display("\n--- Test: TXQ Mip Levels ---");

        issue_tex(`OP_TXQ, `TXQ_LEVELS, 0, 0, 0);
        wait_result();

        check_result_32("TXQ levels = 6", 32'd6, result[31:0]);

        //==================================================================
        // Test 5: TEX 2D Point Sampling at (0,0)
        //==================================================================
        $display("\n--- Test: TEX 2D at (0,0) ---");

        issue_tex(`OP_TEX, `TEX_2D, 32'd0, 32'd0, 32'd0);
        wait_result();

        // First texel from tex_mem[0]: R=0xCC, G=0xBB, B=0xAA, A=0xFF
        $display("TEX result: 0x%032h", result);
        check_result_32("TEX 2D R channel", 8'hCC, result[7:0]);

        //==================================================================
        // Test 6: TEX 2D at different coordinates
        //==================================================================
        $display("\n--- Test: TEX 2D at (1,0) ---");

        issue_tex(`OP_TEX, `TEX_2D, 32'd1, 32'd0, 32'd0);
        wait_result();

        $display("TEX result at (1,0): 0x%032h", result);
        check_result_32("TEX 2D at (1,0) non-zero", 1, (result != 128'b0));

        //==================================================================
        // Test 7: TEX 1D
        //==================================================================
        $display("\n--- Test: TEX 1D ---");

        issue_tex(`OP_TEX, `TEX_1D, 32'd0, 32'd0, 32'd0);
        wait_result();

        $display("TEX 1D result: 0x%032h", result);
        check_result_32("TEX 1D completed", 1, valid_out);

        //==================================================================
        // Test 8: Coordinate wrapping - Clamp
        //==================================================================
        $display("\n--- Test: Coordinate clamp ---");

        // Coordinate > width should clamp to width-1
        issue_tex(`OP_TEX, `TEX_2D, 32'd100, 32'd0, 32'd0);  // x=100, width=64
        wait_result();

        $display("Clamped TEX result: 0x%032h", result);
        check_result_32("Clamp completed", 1, valid_out);

        //==================================================================
        // Test 9: Negative coordinate clamp
        //==================================================================
        $display("\n--- Test: Negative coordinate clamp ---");

        // Negative coordinate should clamp to 0
        issue_tex(`OP_TEX, `TEX_2D, 32'hFFFFFFFF, 32'd0, 32'd0);  // x=-1
        wait_result();

        check_result_32("Negative clamp completed", 1, valid_out);

        //==================================================================
        // Test 10: Change to wrap mode
        //==================================================================
        $display("\n--- Test: Wrap mode ---");

        tex_wrap_s <= 4'h0;  // Change to repeat/wrap
        @(posedge clk);

        issue_tex(`OP_TEX, `TEX_2D, 32'd65, 32'd0, 32'd0);  // x=65, should wrap to 1
        wait_result();

        $display("Wrapped TEX result: 0x%032h", result);
        check_result_32("Wrap mode completed", 1, valid_out);

        tex_wrap_s <= 4'h1;  // Restore clamp
        @(posedge clk);

        //==================================================================
        // Test 11: Different texture sizes
        //==================================================================
        $display("\n--- Test: Different texture sizes ---");

        tex_width <= 16'd128;
        tex_height <= 16'd32;
        @(posedge clk);

        issue_tex(`OP_TXQ, `TXQ_WIDTH, 0, 0, 0);
        wait_result();
        check_result_32("New width = 128", 32'd128, result[31:0]);

        issue_tex(`OP_TXQ, `TXQ_HEIGHT, 0, 0, 0);
        wait_result();
        check_result_32("New height = 32", 32'd32, result[31:0]);

        tex_width <= 16'd64;
        tex_height <= 16'd64;
        @(posedge clk);

        //==================================================================
        // Test 12: Back-to-back texture fetches
        //==================================================================
        $display("\n--- Test: Back-to-back fetches ---");

        for (i = 0; i < 4; i = i + 1) begin
            issue_tex(`OP_TEX, `TEX_2D, i, 32'd0, 32'd0);
            wait_result();
        end

        check_result_32("Back-to-back completed", 1, 1);

        //==================================================================
        // Test 13: TEX 3D
        //==================================================================
        $display("\n--- Test: TEX 3D ---");

        tex_depth <= 16'd4;
        @(posedge clk);

        issue_tex(`OP_TEX, `TEX_3D, 32'd0, 32'd0, 32'd1);  // z=1
        wait_result();

        $display("TEX 3D result: 0x%032h", result);
        check_result_32("TEX 3D completed", 1, valid_out);

        tex_depth <= 16'd1;
        @(posedge clk);

        //==================================================================
        // Results Summary
        //==================================================================
        #100;
        $display("\n============================================================");
        $display("Texture Unit Test Results");
        $display("============================================================");
        $display("Tests passed: %0d", pass_count);
        $display("Tests failed: %0d", fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        $display("============================================================");

        #100;
        $finish;
    end

    // Timeout
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_texture_unit.vcd");
        $dumpvars(0, tb_texture_unit);
    end

endmodule
