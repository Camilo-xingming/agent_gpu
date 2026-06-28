//============================================================================
// RalphGPU - TMA Unit Testbench
// Coverage focus:
// - 1D/2D/3D descriptor decode and address generation
// - Box-copy request pattern validation
// - Completion pulse checks (mbarrier surrogate via done edge)
//============================================================================

`timescale 1ns/1ps
`include "gpu_defines.vh"

module tb_tma_unit;

    //------------------------------------------------------------------------
    // Parameters
    //------------------------------------------------------------------------
    parameter ADDR_W = 32;
    parameter SMEM_ADDR_W = 14;
    parameter TRANSFER_SIZE = 16;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz
    end

    //------------------------------------------------------------------------
    // DUT Signals - TMA Unit
    //------------------------------------------------------------------------
    reg                     tma_start;
    reg  [63:0]             tma_tensor_desc;
    reg  [31:0]             tma_coord_x;
    reg  [31:0]             tma_coord_y;
    reg  [SMEM_ADDR_W-1:0]  tma_dst_base;

    wire                    tma_req_valid;
    wire [ADDR_W-1:0]       tma_req_src_addr;
    wire [SMEM_ADDR_W-1:0]  tma_req_dst_addr;
    wire [4:0]              tma_req_size;
    reg                     tma_req_ready;

    wire                    tma_done;
    wire                    tma_busy;
    wire [15:0]             tma_bytes_copied;

    //------------------------------------------------------------------------
    // DUT Signals - TMA 3D Unit
    //------------------------------------------------------------------------
    reg                     tma3d_start;
    reg  [95:0]             tma3d_tensor_desc;
    reg  [31:0]             tma3d_coord_x;
    reg  [31:0]             tma3d_coord_y;
    reg  [31:0]             tma3d_coord_z;
    reg  [SMEM_ADDR_W-1:0]  tma3d_dst_base;

    wire                    tma3d_req_valid;
    wire [ADDR_W-1:0]       tma3d_req_src_addr;
    wire [SMEM_ADDR_W-1:0]  tma3d_req_dst_addr;
    wire [4:0]              tma3d_req_size;
    reg                     tma3d_req_ready;

    wire                    tma3d_done;
    wire                    tma3d_busy;
    wire [23:0]             tma3d_bytes_copied;

    //------------------------------------------------------------------------
    // DUT Signals - Descriptor Builder
    //------------------------------------------------------------------------
    reg  [31:0]             desc_base_addr;
    reg  [15:0]             desc_stride;
    reg  [7:0]              desc_box_width;
    reg  [7:0]              desc_box_height;
    wire [63:0]             built_descriptor;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    tma_unit #(
        .ADDR_W(ADDR_W),
        .SMEM_ADDR_W(SMEM_ADDR_W),
        .TRANSFER_SIZE(TRANSFER_SIZE)
    ) u_tma (
        .clk(clk),
        .rst_n(rst_n),
        .start(tma_start),
        .tensor_desc(tma_tensor_desc),
        .coord_x(tma_coord_x),
        .coord_y(tma_coord_y),
        .dst_base(tma_dst_base),
        .req_valid(tma_req_valid),
        .req_src_addr(tma_req_src_addr),
        .req_dst_addr(tma_req_dst_addr),
        .req_size(tma_req_size),
        .req_ready(tma_req_ready),
        .done(tma_done),
        .busy(tma_busy),
        .bytes_copied(tma_bytes_copied)
    );

    tma_unit_3d #(
        .ADDR_W(ADDR_W),
        .SMEM_ADDR_W(SMEM_ADDR_W),
        .TRANSFER_SIZE(TRANSFER_SIZE)
    ) u_tma3d (
        .clk(clk),
        .rst_n(rst_n),
        .start(tma3d_start),
        .tensor_desc(tma3d_tensor_desc),
        .coord_x(tma3d_coord_x),
        .coord_y(tma3d_coord_y),
        .coord_z(tma3d_coord_z),
        .dst_base(tma3d_dst_base),
        .req_valid(tma3d_req_valid),
        .req_src_addr(tma3d_req_src_addr),
        .req_dst_addr(tma3d_req_dst_addr),
        .req_size(tma3d_req_size),
        .req_ready(tma3d_req_ready),
        .done(tma3d_done),
        .busy(tma3d_busy),
        .bytes_copied(tma3d_bytes_copied)
    );

    tma_descriptor_builder u_desc_builder (
        .base_addr(desc_base_addr),
        .stride(desc_stride),
        .box_width(desc_box_width),
        .box_height(desc_box_height),
        .descriptor(built_descriptor)
    );

    //------------------------------------------------------------------------
    // Test Variables
    //------------------------------------------------------------------------
    integer test_count;
    integer pass_count;
    integer fail_count;
    integer req_count;
    integer total_bytes;

    integer cycle_count;
    integer tma_last_req_cycle;
    integer tma3d_last_req_cycle;
    integer tma_done_cycle;
    integer tma3d_done_cycle;
    integer mbarrier_2d_pulse_count;
    integer mbarrier_3d_pulse_count;

    reg tma_done_q;
    reg tma3d_done_q;

    reg [ADDR_W-1:0] observed_src_addrs [0:255];
    reg [SMEM_ADDR_W-1:0] observed_dst_addrs [0:255];
    reg [4:0] observed_sizes [0:255];

    //------------------------------------------------------------------------
    // Completion Pulse Monitor (mbarrier surrogate)
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            tma_last_req_cycle <= -1;
            tma3d_last_req_cycle <= -1;
            tma_done_cycle <= -1;
            tma3d_done_cycle <= -1;
            mbarrier_2d_pulse_count <= 0;
            mbarrier_3d_pulse_count <= 0;
            tma_done_q <= 1'b0;
            tma3d_done_q <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;

            if (tma_req_valid && tma_req_ready) begin
                tma_last_req_cycle <= cycle_count;
            end
            if (tma3d_req_valid && tma3d_req_ready) begin
                tma3d_last_req_cycle <= cycle_count;
            end

            if (tma_done && !tma_done_q) begin
                mbarrier_2d_pulse_count <= mbarrier_2d_pulse_count + 1;
                tma_done_cycle <= cycle_count;
            end
            if (tma3d_done && !tma3d_done_q) begin
                mbarrier_3d_pulse_count <= mbarrier_3d_pulse_count + 1;
                tma3d_done_cycle <= cycle_count;
            end

            tma_done_q <= tma_done;
            tma3d_done_q <= tma3d_done;
        end
    end

    //------------------------------------------------------------------------
    // Test Tasks
    //------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n = 0;
            tma_start = 0;
            tma_tensor_desc = 64'b0;
            tma_coord_x = 32'b0;
            tma_coord_y = 32'b0;
            tma_dst_base = {SMEM_ADDR_W{1'b0}};
            tma_req_ready = 0;

            tma3d_start = 0;
            tma3d_tensor_desc = 96'b0;
            tma3d_coord_x = 32'b0;
            tma3d_coord_y = 32'b0;
            tma3d_coord_z = 32'b0;
            tma3d_dst_base = {SMEM_ADDR_W{1'b0}};
            tma3d_req_ready = 0;

            desc_base_addr = 32'b0;
            desc_stride = 16'b0;
            desc_box_width = 8'b0;
            desc_box_height = 8'b0;

            repeat (5) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);
        end
    endtask

    task run_2d_copy_capture;
        input [31:0] base_addr;
        input [15:0] stride;
        input [7:0] box_width;
        input [7:0] box_height;
        input [31:0] coord_x;
        input [31:0] coord_y;
        input [SMEM_ADDR_W-1:0] dst_base;
        begin
            tma_tensor_desc = {box_height, box_width, stride, base_addr};
            tma_coord_x = coord_x;
            tma_coord_y = coord_y;
            tma_dst_base = dst_base;
            tma_req_ready = 1'b1;

            @(posedge clk);
            tma_start = 1'b1;
            @(posedge clk);
            tma_start = 1'b0;

            wait (tma_busy);

            req_count = 0;
            total_bytes = 0;
            while (!tma_done) begin
                @(posedge clk);
                if (tma_req_valid && tma_req_ready) begin
                    observed_src_addrs[req_count] = tma_req_src_addr;
                    observed_dst_addrs[req_count] = tma_req_dst_addr;
                    observed_sizes[req_count] = tma_req_size;
                    total_bytes = total_bytes + tma_req_size;
                    req_count = req_count + 1;
                end
            end

            @(posedge clk);
        end
    endtask

    task run_3d_copy_capture;
        input [31:0] base_addr;
        input [15:0] row_stride;
        input [15:0] slice_stride;
        input [7:0] box_x;
        input [7:0] box_y;
        input [7:0] box_z;
        input [31:0] coord_x;
        input [31:0] coord_y;
        input [31:0] coord_z;
        input [SMEM_ADDR_W-1:0] dst_base;
        begin
            tma3d_tensor_desc = {8'b0, box_z, box_y, box_x, slice_stride, row_stride, base_addr};
            tma3d_coord_x = coord_x;
            tma3d_coord_y = coord_y;
            tma3d_coord_z = coord_z;
            tma3d_dst_base = dst_base;
            tma3d_req_ready = 1'b1;

            @(posedge clk);
            tma3d_start = 1'b1;
            @(posedge clk);
            tma3d_start = 1'b0;

            wait (tma3d_busy);

            req_count = 0;
            total_bytes = 0;
            while (!tma3d_done) begin
                @(posedge clk);
                if (tma3d_req_valid && tma3d_req_ready) begin
                    observed_src_addrs[req_count] = tma3d_req_src_addr;
                    observed_dst_addrs[req_count] = tma3d_req_dst_addr;
                    observed_sizes[req_count] = tma3d_req_size;
                    total_bytes = total_bytes + tma3d_req_size;
                    req_count = req_count + 1;
                end
            end

            @(posedge clk);
        end
    endtask

    task test_descriptor_builder;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            $display("\n--- Test %0d: Descriptor Builder ---", test_count);

            desc_base_addr = 32'h1000_0000;
            desc_stride = 16'd256;
            desc_box_width = 8'd64;
            desc_box_height = 8'd16;
            @(posedge clk);

            if (built_descriptor[31:0] != 32'h1000_0000 ||
                built_descriptor[47:32] != 16'd256 ||
                built_descriptor[55:48] != 8'd64 ||
                built_descriptor[63:56] != 8'd16) begin
                test_ok = 1'b0;
                $display("FAIL: descriptor mismatch expected=0x%016x got=0x%016x",
                         {8'd16, 8'd64, 16'd256, 32'h1000_0000}, built_descriptor);
            end

            if (test_ok) begin
                $display("PASS: Descriptor correctly packed");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_1d_descriptor_copy;
        integer before_mbarrier;
        reg [ADDR_W-1:0] expected_base;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            before_mbarrier = mbarrier_2d_pulse_count;

            $display("\n--- Test %0d: 1D Descriptor (height=1) ---", test_count);

            run_2d_copy_capture(32'h0100_0000, 16'd512, 8'd36, 8'd1, 32'd32, 32'd5, 14'h120);
            expected_base = 32'h0100_0000 + (32'd5 * 16'd512) + 32'd32;

            if (req_count != 3) begin
                test_ok = 1'b0;
                $display("FAIL: expected 3 requests, got %0d", req_count);
            end
            if (tma_bytes_copied != 16'd36) begin
                test_ok = 1'b0;
                $display("FAIL: expected bytes=36, got %0d", tma_bytes_copied);
            end

            if (observed_src_addrs[0] != expected_base ||
                observed_src_addrs[1] != expected_base + 16 ||
                observed_src_addrs[2] != expected_base + 32) begin
                test_ok = 1'b0;
                $display("FAIL: unexpected 1D src pattern");
            end

            if (observed_dst_addrs[0] != 14'h120 ||
                observed_dst_addrs[1] != 14'h130 ||
                observed_dst_addrs[2] != 14'h140) begin
                test_ok = 1'b0;
                $display("FAIL: unexpected 1D dst pattern");
            end

            if (observed_sizes[0] != 16 || observed_sizes[1] != 16 || observed_sizes[2] != 4) begin
                test_ok = 1'b0;
                $display("FAIL: unexpected 1D transfer sizes %0d/%0d/%0d",
                         observed_sizes[0], observed_sizes[1], observed_sizes[2]);
            end

            if (mbarrier_2d_pulse_count != before_mbarrier + 1) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse missing for 1D copy");
            end
            if (tma_done_cycle <= tma_last_req_cycle) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse before final request ack");
            end

            if (test_ok) begin
                $display("PASS: 1D descriptor/addressing and completion pulse verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_2d_box_copy_pattern;
        integer before_mbarrier;
        integer row;
        integer col;
        integer idx;
        integer expected_size;
        integer expected_dst_offset;
        reg [ADDR_W-1:0] expected_src;
        reg [SMEM_ADDR_W-1:0] expected_dst;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            before_mbarrier = mbarrier_2d_pulse_count;

            $display("\n--- Test %0d: 2D Box Copy Pattern ---", test_count);

            // 3 rows * (16B + 4B) each row
            run_2d_copy_capture(32'h0200_0000, 16'd80, 8'd20, 8'd3, 32'd12, 32'd2, 14'h400);

            if (req_count != 6) begin
                test_ok = 1'b0;
                $display("FAIL: expected 6 requests, got %0d", req_count);
            end
            if (tma_bytes_copied != 16'd60) begin
                test_ok = 1'b0;
                $display("FAIL: expected bytes=60, got %0d", tma_bytes_copied);
            end

            idx = 0;
            expected_dst_offset = 0;
            for (row = 0; row < 3; row = row + 1) begin
                col = 0;
                while (col < 20) begin
                    expected_size = ((20 - col) >= 16) ? 16 : (((20 - col) >= 8) ? 8 : 4);
                    expected_src = 32'h0200_0000 + ((32'd2 + row) * 16'd80) + 32'd12 + col;
                    expected_dst = 14'h400 + expected_dst_offset;

                    if (observed_src_addrs[idx] != expected_src) begin
                        test_ok = 1'b0;
                        $display("FAIL: src mismatch idx=%0d exp=0x%08x got=0x%08x",
                                 idx, expected_src, observed_src_addrs[idx]);
                    end
                    if (observed_dst_addrs[idx] != expected_dst) begin
                        test_ok = 1'b0;
                        $display("FAIL: dst mismatch idx=%0d exp=0x%04x got=0x%04x",
                                 idx, expected_dst, observed_dst_addrs[idx]);
                    end
                    if (observed_sizes[idx] != expected_size) begin
                        test_ok = 1'b0;
                        $display("FAIL: size mismatch idx=%0d exp=%0d got=%0d",
                                 idx, expected_size, observed_sizes[idx]);
                    end

                    expected_dst_offset = expected_dst_offset + expected_size;
                    col = col + expected_size;
                    idx = idx + 1;
                end
            end

            if (idx != req_count) begin
                test_ok = 1'b0;
                $display("FAIL: request count accounting mismatch idx=%0d req_count=%0d", idx, req_count);
            end

            if (mbarrier_2d_pulse_count != before_mbarrier + 1) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse missing for 2D box copy");
            end
            if (tma_done_cycle <= tma_last_req_cycle) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse before final request ack (2D)");
            end

            if (test_ok) begin
                $display("PASS: 2D box copy request pattern verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_3d_descriptor_box_copy;
        integer before_mbarrier;
        integer z;
        integer y;
        integer x;
        integer idx;
        integer expected_size;
        integer expected_dst_offset;
        reg [ADDR_W-1:0] expected_src;
        reg [SMEM_ADDR_W-1:0] expected_dst;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            before_mbarrier = mbarrier_3d_pulse_count;

            $display("\n--- Test %0d: 3D Descriptor + Box Copy Pattern ---", test_count);

            // 2 slices * 2 rows * (16B + 4B) each row = 8 requests, 80 bytes
            run_3d_copy_capture(32'h0300_0000, 16'd64, 16'd256, 8'd20, 8'd2, 8'd2,
                                32'd8, 32'd1, 32'd2, 14'h600);

            if (req_count != 8) begin
                test_ok = 1'b0;
                $display("FAIL: expected 8 requests, got %0d", req_count);
            end
            if (tma3d_bytes_copied != 24'd80) begin
                test_ok = 1'b0;
                $display("FAIL: expected bytes=80, got %0d", tma3d_bytes_copied);
            end

            idx = 0;
            expected_dst_offset = 0;
            for (z = 0; z < 2; z = z + 1) begin
                for (y = 0; y < 2; y = y + 1) begin
                    x = 0;
                    while (x < 20) begin
                        expected_size = ((20 - x) >= 16) ? 16 : (((20 - x) >= 8) ? 8 : 4);
                        expected_src = 32'h0300_0000 + ((32'd2 + z) * 16'd256) +
                                       ((32'd1 + y) * 16'd64) + 32'd8 + x;
                        expected_dst = 14'h600 + expected_dst_offset;

                        if (observed_src_addrs[idx] != expected_src) begin
                            test_ok = 1'b0;
                            $display("FAIL: 3D src mismatch idx=%0d exp=0x%08x got=0x%08x",
                                     idx, expected_src, observed_src_addrs[idx]);
                        end
                        if (observed_dst_addrs[idx] != expected_dst) begin
                            test_ok = 1'b0;
                            $display("FAIL: 3D dst mismatch idx=%0d exp=0x%04x got=0x%04x",
                                     idx, expected_dst, observed_dst_addrs[idx]);
                        end
                        if (observed_sizes[idx] != expected_size) begin
                            test_ok = 1'b0;
                            $display("FAIL: 3D size mismatch idx=%0d exp=%0d got=%0d",
                                     idx, expected_size, observed_sizes[idx]);
                        end

                        expected_dst_offset = expected_dst_offset + expected_size;
                        x = x + expected_size;
                        idx = idx + 1;
                    end
                end
            end

            if (idx != req_count) begin
                test_ok = 1'b0;
                $display("FAIL: 3D request count accounting mismatch idx=%0d req_count=%0d", idx, req_count);
            end

            if (mbarrier_3d_pulse_count != before_mbarrier + 1) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse missing for 3D copy");
            end
            if (tma3d_done_cycle <= tma3d_last_req_cycle) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse before final request ack (3D)");
            end

            if (test_ok) begin
                $display("PASS: 3D descriptor/addressing and completion pulse verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_backpressure;
        integer before_mbarrier;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            before_mbarrier = mbarrier_2d_pulse_count;
            $display("\n--- Test %0d: TMA Backpressure Handling ---", test_count);

            tma_tensor_desc = {8'd1, 8'd64, 16'd64, 32'h2000_0000};
            tma_coord_x = 0;
            tma_coord_y = 0;
            tma_dst_base = 14'h800;
            tma_req_ready = 0;

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            wait (tma_busy);
            repeat (5) @(posedge clk);

            if (!(tma_req_valid && !tma_req_ready)) begin
                test_ok = 1'b0;
                $display("FAIL: expected pending request under backpressure");
            end

            tma_req_ready = 1;
            wait (tma_done);
            @(posedge clk);

            if (tma_bytes_copied != 64) begin
                test_ok = 1'b0;
                $display("FAIL: expected bytes=64, got %0d", tma_bytes_copied);
            end
            if (mbarrier_2d_pulse_count != before_mbarrier + 1) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse missing after backpressure flow");
            end

            if (test_ok) begin
                $display("PASS: Backpressure flow verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_coord_offset;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            $display("\n--- Test %0d: Coordinate Offset Calculation ---", test_count);

            // Base addr + (coord_y * stride) + coord_x
            // 0x1000 + (2 * 256) + 16 = 0x1210
            tma_tensor_desc = {8'd1, 8'd32, 16'd256, 32'h0000_1000};
            tma_coord_x = 32'd16;
            tma_coord_y = 32'd2;
            tma_dst_base = 14'h0;
            tma_req_ready = 1;

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            wait (tma_busy);
            wait (tma_req_valid);
            @(posedge clk);

            if (tma_req_src_addr != 32'h0000_1210) begin
                test_ok = 1'b0;
                $display("FAIL: expected first src=0x00001210, got 0x%08x", tma_req_src_addr);
            end

            wait (tma_done);
            @(posedge clk);

            if (test_ok) begin
                $display("PASS: Coordinate offset verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_zero_size_no_completion;
        reg test_ok;
        reg saw_2d_done;
        reg saw_3d_done;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            saw_2d_done = 1'b0;
            saw_3d_done = 1'b0;

            $display("\n--- Test %0d: Zero-Size Guard + No Completion Pulse ---", test_count);

            // Drain any lingering done pulse from the prior test
            repeat (2) @(posedge clk);

            // 2D width=0 should not start or generate completion pulse
            tma_tensor_desc = {8'd4, 8'd0, 16'd64, 32'h3000_0000};
            tma_coord_x = 0;
            tma_coord_y = 0;
            tma_dst_base = 14'h0;
            tma_req_ready = 1;
            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;
            repeat (10) begin
                @(posedge clk);
                if (tma_done) saw_2d_done = 1'b1;
            end
            if (tma_busy || saw_2d_done) begin
                test_ok = 1'b0;
                $display("FAIL: zero-width 2D copy should stay idle and not pulse done");
            end

            // 3D depth=0 should not start or generate completion pulse
            tma3d_tensor_desc = {8'b0, 8'd0, 8'd2, 8'd16, 16'd256, 16'd64, 32'h3100_0000};
            tma3d_coord_x = 0;
            tma3d_coord_y = 0;
            tma3d_coord_z = 0;
            tma3d_dst_base = 14'h20;
            tma3d_req_ready = 1;
            @(posedge clk);
            tma3d_start = 1;
            @(posedge clk);
            tma3d_start = 0;
            repeat (10) begin
                @(posedge clk);
                if (tma3d_done) saw_3d_done = 1'b1;
            end
            if (tma3d_busy || saw_3d_done) begin
                test_ok = 1'b0;
                $display("FAIL: zero-depth 3D copy should stay idle and not pulse done");
            end

            if (test_ok) begin
                $display("PASS: zero-size guard and completion suppression verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask
    task test_large_tile;
        integer before_mbarrier;
        reg test_ok;
        begin
            test_count = test_count + 1;
            test_ok = 1'b1;
            before_mbarrier = mbarrier_2d_pulse_count;
            $display("\n--- Test %0d: Large Tile Copy (128x8) ---", test_count);

            run_2d_copy_capture(32'h4000_0000, 16'd256, 8'd128, 8'd8, 32'd0, 32'd0, 14'h0);

            if (tma_bytes_copied != 16'd1024) begin
                test_ok = 1'b0;
                $display("FAIL: expected bytes=1024, got %0d", tma_bytes_copied);
            end
            if (req_count != 64) begin
                test_ok = 1'b0;
                $display("FAIL: expected 64 requests, got %0d", req_count);
            end
            if (mbarrier_2d_pulse_count != before_mbarrier + 1) begin
                test_ok = 1'b0;
                $display("FAIL: completion pulse missing for large tile");
            end

            if (test_ok) begin
                $display("PASS: large tile copy verified");
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU TMA Unit Testbench");
        $display("============================================================");

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        reset_dut();

        test_descriptor_builder();
        test_1d_descriptor_copy();
        test_2d_box_copy_pattern();
        test_coord_offset();
        test_backpressure();
        test_zero_size_no_completion();
        test_large_tile();
        test_3d_descriptor_box_copy();

        // Summary
        $display("\n============================================================");
        $display("Test Summary: %0d/%0d tests passed", pass_count, test_count);
        if (fail_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("FAILURES: %0d", fail_count);
        end
        $display("============================================================");

        #100;
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        if (fail_count > 0) $fatal(1, "Test Failed");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_tma_unit.vcd");
        $dumpvars(0, tb_tma_unit);
    end

endmodule
