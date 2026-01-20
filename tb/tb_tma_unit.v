//============================================================================
// RalphGPU - TMA Unit Testbench
// Tests Tensor Memory Accelerator functionality for 2D/3D tensor copies
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
    integer i;
    integer req_count;
    integer total_bytes;

    // Expected request tracking
    reg [ADDR_W-1:0] expected_src_addrs [0:255];
    reg [SMEM_ADDR_W-1:0] expected_dst_addrs [0:255];
    integer expected_req_count;

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

    task test_descriptor_builder;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Descriptor Builder ---", test_count);

            desc_base_addr = 32'h1000_0000;
            desc_stride = 16'd256;
            desc_box_width = 8'd64;
            desc_box_height = 8'd16;
            @(posedge clk);

            if (built_descriptor[31:0] == 32'h1000_0000 &&
                built_descriptor[47:32] == 16'd256 &&
                built_descriptor[55:48] == 8'd64 &&
                built_descriptor[63:56] == 8'd16) begin
                $display("PASS: Descriptor correctly packed");
                $display("  Base=0x%08x, Stride=%0d, Width=%0d, Height=%0d",
                         built_descriptor[31:0], built_descriptor[47:32],
                         built_descriptor[55:48], built_descriptor[63:56]);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Descriptor mismatch");
                $display("  Expected: 0x%016x", {8'd16, 8'd64, 16'd256, 32'h1000_0000});
                $display("  Got:      0x%016x", built_descriptor);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_simple_2d_copy;
        input [31:0] base_addr;
        input [15:0] stride;
        input [7:0]  box_width;
        input [7:0]  box_height;
        input [31:0] coord_x;
        input [31:0] coord_y;
        input [SMEM_ADDR_W-1:0] dst_base;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: 2D TMA Copy ---", test_count);
            $display("  base=0x%08x stride=%0d width=%0d height=%0d",
                     base_addr, stride, box_width, box_height);
            $display("  coord=(%0d,%0d) dst=0x%04x", coord_x, coord_y, dst_base);

            // Build descriptor and start TMA
            tma_tensor_desc = {box_height, box_width, stride, base_addr};
            tma_coord_x = coord_x;
            tma_coord_y = coord_y;
            tma_dst_base = dst_base;
            tma_req_ready = 1;

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            // Wait for TMA to become busy
            wait (tma_busy);
            $display("  TMA started, busy=1");

            // Count requests
            req_count = 0;
            total_bytes = 0;
            while (!tma_done) begin
                @(posedge clk);
                if (tma_req_valid && tma_req_ready) begin
                    req_count = req_count + 1;
                    total_bytes = total_bytes + tma_req_size;
                    $display("  Request %0d: src=0x%08x dst=0x%04x size=%0d",
                             req_count, tma_req_src_addr, tma_req_dst_addr, tma_req_size);
                end
            end

            @(posedge clk);
            $display("  TMA completed: requests=%0d, bytes_copied=%0d", req_count, tma_bytes_copied);

            // Verify total bytes
            if (tma_bytes_copied == box_width * box_height) begin
                $display("PASS: Correct byte count");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Expected %0d bytes, got %0d", box_width * box_height, tma_bytes_copied);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_3d_copy;
        input [31:0] base_addr;
        input [15:0] row_stride;
        input [15:0] slice_stride;
        input [7:0]  box_x;
        input [7:0]  box_y;
        input [7:0]  box_z;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: 3D TMA Copy ---", test_count);
            $display("  base=0x%08x row_stride=%0d slice_stride=%0d",
                     base_addr, row_stride, slice_stride);
            $display("  box=(%0d,%0d,%0d)", box_x, box_y, box_z);

            // Build 96-bit descriptor
            tma3d_tensor_desc = {8'b0, box_z, box_y, box_x, slice_stride, row_stride, base_addr};
            tma3d_coord_x = 0;
            tma3d_coord_y = 0;
            tma3d_coord_z = 0;
            tma3d_dst_base = 14'h0;
            tma3d_req_ready = 1;

            @(posedge clk);
            tma3d_start = 1;
            @(posedge clk);
            tma3d_start = 0;

            // Wait for TMA to become busy
            wait (tma3d_busy);
            $display("  TMA3D started, busy=1");

            // Count requests
            req_count = 0;
            while (!tma3d_done) begin
                @(posedge clk);
                if (tma3d_req_valid && tma3d_req_ready) begin
                    req_count = req_count + 1;
                end
            end

            @(posedge clk);
            $display("  TMA3D completed: requests=%0d, bytes_copied=%0d", req_count, tma3d_bytes_copied);

            // Verify total bytes
            if (tma3d_bytes_copied == box_x * box_y * box_z) begin
                $display("PASS: Correct 3D byte count");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Expected %0d bytes, got %0d", box_x * box_y * box_z, tma3d_bytes_copied);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_backpressure;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: TMA Backpressure Handling ---", test_count);

            // Setup a 64-byte row copy
            tma_tensor_desc = {8'd1, 8'd64, 16'd64, 32'h2000_0000};
            tma_coord_x = 0;
            tma_coord_y = 0;
            tma_dst_base = 14'h800;
            tma_req_ready = 0;  // Start with backpressure

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            wait (tma_busy);

            // Wait for first request
            repeat (5) @(posedge clk);
            if (tma_req_valid && !tma_req_ready) begin
                $display("  Request pending with backpressure");
            end

            // Release backpressure
            tma_req_ready = 1;
            @(posedge clk);

            // Wait for completion
            wait (tma_done);
            @(posedge clk);

            if (tma_bytes_copied == 64) begin
                $display("PASS: Backpressure handled correctly, bytes=%0d", tma_bytes_copied);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Unexpected byte count after backpressure");
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_coord_offset;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Coordinate Offset Calculation ---", test_count);

            // Base addr + (coord_y * stride) + coord_x
            // 0x1000 + (2 * 256) + 16 = 0x1000 + 0x200 + 0x10 = 0x1210
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

            // Check first request address
            wait (tma_req_valid);
            @(posedge clk);

            if (tma_req_src_addr == 32'h0000_1210) begin
                $display("PASS: First address correct: 0x%08x", tma_req_src_addr);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Expected 0x00001210, got 0x%08x", tma_req_src_addr);
                fail_count = fail_count + 1;
            end

            wait (tma_done);
            @(posedge clk);
        end
    endtask

    task test_zero_size;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Zero Size Protection ---", test_count);

            // Test with zero width
            tma_tensor_desc = {8'd4, 8'd0, 16'd64, 32'h3000_0000};  // width=0
            tma_coord_x = 0;
            tma_coord_y = 0;
            tma_dst_base = 14'h0;
            tma_req_ready = 1;

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            // Should not become busy with zero size
            repeat (10) @(posedge clk);

            if (!tma_busy && !tma_done) begin
                $display("PASS: Zero width copy not started");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Zero width copy should not start");
                fail_count = fail_count + 1;
            end

            // Test with zero height
            tma_tensor_desc = {8'd0, 8'd64, 16'd64, 32'h3000_0000};  // height=0
            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            repeat (10) @(posedge clk);

            if (!tma_busy && !tma_done) begin
                $display("PASS: Zero height copy not started");
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Zero height copy should not start");
                fail_count = fail_count + 1;
            end
        end
    endtask

    task test_large_tile;
        begin
            test_count = test_count + 1;
            $display("\n--- Test %0d: Large Tile Copy (128x8) ---", test_count);

            // 128 bytes wide x 8 rows = 1024 bytes total
            tma_tensor_desc = {8'd8, 8'd128, 16'd256, 32'h4000_0000};
            tma_coord_x = 0;
            tma_coord_y = 0;
            tma_dst_base = 14'h0;
            tma_req_ready = 1;

            @(posedge clk);
            tma_start = 1;
            @(posedge clk);
            tma_start = 0;

            wait (tma_done);
            @(posedge clk);

            if (tma_bytes_copied == 16'd1024) begin
                $display("PASS: Large tile copied correctly, bytes=%0d", tma_bytes_copied);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: Expected 1024 bytes, got %0d", tma_bytes_copied);
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

        // Test 1: Descriptor Builder
        test_descriptor_builder();

        // Test 2: Simple 16x1 copy (single row, single transfer)
        test_simple_2d_copy(32'h0000_1000, 16'd64, 8'd16, 8'd1, 32'd0, 32'd0, 14'h0);

        // Test 3: 64x4 copy (4 rows, 4 transfers each)
        test_simple_2d_copy(32'h0000_2000, 16'd128, 8'd64, 8'd4, 32'd0, 32'd0, 14'h100);

        // Test 4: 32x2 copy with offset
        test_simple_2d_copy(32'h0000_3000, 16'd64, 8'd32, 8'd2, 32'd16, 32'd4, 14'h200);

        // Test 5: Coordinate offset verification
        test_coord_offset();

        // Test 6: Backpressure handling
        test_backpressure();

        // Test 7: Zero size protection
        test_zero_size();

        // Test 8: Large tile copy
        test_large_tile();

        // Test 9: 3D TMA copy
        test_3d_copy(32'h5000_0000, 16'd64, 16'd256, 8'd32, 8'd2, 8'd2);

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
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_tma_unit.vcd");
        $dumpvars(0, tb_tma_unit);
    end

endmodule
