//============================================================================
// Testbench for Command Processor (Issue #151)
//
// Tests:
// 1. Legacy mode: CSR-direct kernel launch (backward compat)
// 2. Queue mode: single kernel via command queue
// 3. Queue mode: multi-kernel sequential dispatch
// 4. Fence and interrupt mechanism
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_command_processor;

    localparam CLK_PERIOD = 10;
    localparam NUM_SM = 2;
    localparam QUEUE_DEPTH = 8;
    localparam DESC_WORDS = 16;

    // CSR addresses
    localparam CSR_GPU_STATUS        = 12'h000;
    localparam CSR_GPU_CONTROL       = 12'h004;
    localparam CSR_CMD_QUEUE_TAIL    = 12'h040;
    localparam CSR_CMD_FENCE_VALUE   = 12'h044;
    localparam CSR_CMD_FENCE_SIGNAL  = 12'h048;
    localparam CSR_CP_STATUS         = 12'h04C;
    localparam CSR_DESC_BASE         = 12'h050;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk, rst_n;
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        csr_rd_valid;

    reg         legacy_kernel_start;
    reg  [31:0] legacy_kernel_pc;
    reg  [31:0] legacy_grid_dim_x, legacy_grid_dim_y, legacy_grid_dim_z;
    reg  [31:0] legacy_block_dim_x, legacy_block_dim_y, legacy_block_dim_z;

    wire [NUM_SM-1:0] sm_kernel_start;
    wire [31:0]       sm_kernel_pc;
    wire [31:0]       sm_block_dim_x, sm_block_dim_y, sm_block_dim_z;
    wire [31:0]       sm_grid_dim_x, sm_grid_dim_y, sm_grid_dim_z;
    wire        gpu_busy;
    wire        irq_kernel_done;
    wire [31:0] fence_value;

    // SM done simulation
    reg [NUM_SM-1:0] sm_done_reg;

    // sm_block_id wires
    wire [31:0] sm_block_id_x [0:NUM_SM-1];
    wire [31:0] sm_block_id_y [0:NUM_SM-1];
    wire [31:0] sm_block_id_z [0:NUM_SM-1];

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    command_processor #(
        .NUM_SM      (NUM_SM),
        .QUEUE_DEPTH (QUEUE_DEPTH),
        .DESC_WORDS  (DESC_WORDS)
    ) u_cp (
        .clk              (clk),
        .rst_n            (rst_n),
        .csr_wr_en        (csr_wr_en),
        .csr_addr         (csr_addr),
        .csr_wr_data      (csr_wr_data),
        .csr_rd_data      (csr_rd_data),
        .csr_rd_valid     (csr_rd_valid),
        .legacy_kernel_start (legacy_kernel_start),
        .legacy_kernel_pc    (legacy_kernel_pc),
        .legacy_grid_dim_x   (legacy_grid_dim_x),
        .legacy_grid_dim_y   (legacy_grid_dim_y),
        .legacy_grid_dim_z   (legacy_grid_dim_z),
        .legacy_block_dim_x  (legacy_block_dim_x),
        .legacy_block_dim_y  (legacy_block_dim_y),
        .legacy_block_dim_z  (legacy_block_dim_z),
        .sm_kernel_start  (sm_kernel_start),
        .sm_kernel_pc     (sm_kernel_pc),
        .sm_block_id_x    (sm_block_id_x),
        .sm_block_id_y    (sm_block_id_y),
        .sm_block_id_z    (sm_block_id_z),
        .sm_block_dim_x   (sm_block_dim_x),
        .sm_block_dim_y   (sm_block_dim_y),
        .sm_block_dim_z   (sm_block_dim_z),
        .sm_grid_dim_x    (sm_grid_dim_x),
        .sm_grid_dim_y    (sm_grid_dim_y),
        .sm_grid_dim_z    (sm_grid_dim_z),
        .sm_done          (sm_done_reg),
        .gpu_busy         (gpu_busy),
        .irq_kernel_done  (irq_kernel_done),
        .fence_value      (fence_value)
    );

    //------------------------------------------------------------------------
    // Test infrastructure
    //------------------------------------------------------------------------
    integer total_tests, passed_tests, failed_tests;

    task write_csr(input [11:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            csr_wr_en <= 1'b1;
            csr_addr <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    task write_desc_word(input [3:0] word_idx, input [31:0] data);
        begin
            write_csr(CSR_DESC_BASE + {8'b0, word_idx} * 4, data);
        end
    endtask

    task automatic sim_sm_done(input integer sm_id, input integer delay_cycles);
        begin
            repeat (delay_cycles) @(posedge clk);
            sm_done_reg[sm_id] <= 1'b1;
            @(posedge clk);
            sm_done_reg[sm_id] <= 1'b0;
        end
    endtask

    task wait_idle(input integer timeout);
        integer cnt;
        begin
            cnt = 0;
            while (gpu_busy && cnt < timeout) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
        end
    endtask

    task check_val(input [255:0] name, input [31:0] actual, input [31:0] expected);
        begin
            total_tests = total_tests + 1;
            if (actual === expected) begin
                passed_tests = passed_tests + 1;
                $display("[PASS] %0s: 0x%08h", name, actual);
            end else begin
                failed_tests = failed_tests + 1;
                $display("[FAIL] %0s: expected 0x%08h got 0x%08h", name, expected, actual);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_command_processor.vcd");
        $dumpvars(0, tb_command_processor);

        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 12'b0;
        csr_wr_data = 32'b0;
        legacy_kernel_start = 0;
        legacy_kernel_pc = 32'b0;
        legacy_grid_dim_x = 32'd1;
        legacy_grid_dim_y = 32'd1;
        legacy_grid_dim_z = 32'd1;
        legacy_block_dim_x = 32'd32;
        legacy_block_dim_y = 32'd1;
        legacy_block_dim_z = 32'd1;
        sm_done_reg = {NUM_SM{1'b0}};

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 1: Legacy mode — single kernel, 1 block
        //====================================================================
        $display("\n=== Test 1: Legacy mode single kernel ===");
        legacy_kernel_pc = 32'h0000_1000;
        legacy_grid_dim_x = 32'd1;
        legacy_grid_dim_y = 32'd1;
        legacy_grid_dim_z = 32'd1;

        @(posedge clk);
        legacy_kernel_start = 1'b1;
        @(posedge clk);
        legacy_kernel_start = 1'b0;

        repeat (5) @(posedge clk);
        check_val("T1 gpu_busy", {31'b0, gpu_busy}, 32'd1);
        check_val("T1 kernel_pc", sm_kernel_pc, 32'h0000_1000);

        // Simulate SM0 done
        fork
            sim_sm_done(0, 10);
        join
        repeat (5) @(posedge clk);

        check_val("T1 irq", {31'b0, irq_kernel_done}, 32'd1);
        wait_idle(100);
        check_val("T1 idle", {31'b0, gpu_busy}, 32'd0);

        //====================================================================
        // Test 2: Legacy mode — 4 blocks across 2 SMs
        //====================================================================
        $display("\n=== Test 2: Legacy mode 4 blocks ===");
        write_csr(CSR_GPU_STATUS, 32'h1);
        repeat (2) @(posedge clk);

        legacy_kernel_pc = 32'h0000_2000;
        legacy_grid_dim_x = 32'd2;
        legacy_grid_dim_y = 32'd2;
        legacy_grid_dim_z = 32'd1;

        @(posedge clk);
        legacy_kernel_start = 1'b1;
        @(posedge clk);
        legacy_kernel_start = 1'b0;

        repeat (5) @(posedge clk);
        check_val("T2 gpu_busy", {31'b0, gpu_busy}, 32'd1);

        // Wave 1: SM0 and SM1 complete
        fork
            sim_sm_done(0, 8);
            sim_sm_done(1, 10);
        join
        repeat (5) @(posedge clk);

        // Wave 2: SM0 and SM1 complete
        fork
            sim_sm_done(0, 8);
            sim_sm_done(1, 10);
        join

        wait_idle(200);
        check_val("T2 idle", {31'b0, gpu_busy}, 32'd0);

        //====================================================================
        // Test 3: Queue mode — single kernel via command queue
        //====================================================================
        $display("\n=== Test 3: Queue mode single kernel ===");
        write_csr(CSR_GPU_STATUS, 32'h1);

        // Enable CP queue mode: bit[1]=1
        write_csr(CSR_GPU_CONTROL, 32'h2);

        // Write descriptor words
        write_desc_word(0, 32'h0000_3000); // kernel_pc
        write_desc_word(1, 32'd1);          // grid_dim_x
        write_desc_word(2, 32'd1);          // grid_dim_y
        write_desc_word(3, 32'd1);          // grid_dim_z
        write_desc_word(4, 32'd32);         // block_dim_x
        write_desc_word(5, 32'd1);          // block_dim_y
        write_desc_word(6, 32'd1);          // block_dim_z
        write_desc_word(7, 32'd0);          // shared_mem
        write_desc_word(8, 32'd0);          // param_addr
        write_desc_word(9, 32'd0);          // param_size
        write_desc_word(10, 32'd42);        // fence_id

        // Doorbell: bump tail
        write_csr(CSR_CMD_QUEUE_TAIL, 32'd1);

        repeat (10) @(posedge clk);
        check_val("T3 gpu_busy", {31'b0, gpu_busy}, 32'd1);
        check_val("T3 kernel_pc", sm_kernel_pc, 32'h0000_3000);

        // SM0 done
        fork
            sim_sm_done(0, 10);
        join
        repeat (5) @(posedge clk);

        wait_idle(100);
        check_val("T3 fence", fence_value, 32'd42);
        check_val("T3 irq", {31'b0, irq_kernel_done}, 32'd1);

        //====================================================================
        // Test 4: Queue mode — 2 kernels back-to-back
        //====================================================================
        $display("\n=== Test 4: Queue mode 2 kernels ===");
        write_csr(CSR_GPU_STATUS, 32'h1);

        // Kernel A: PC=0x4000, grid 2x1x1, fence=100
        write_desc_word(0, 32'h0000_4000);
        write_desc_word(1, 32'd2);
        write_desc_word(2, 32'd1);
        write_desc_word(3, 32'd1);
        write_desc_word(4, 32'd32);
        write_desc_word(5, 32'd1);
        write_desc_word(6, 32'd1);
        write_desc_word(10, 32'd100);
        write_csr(CSR_CMD_QUEUE_TAIL, 32'd2);

        // Kernel B: PC=0x5000, grid 1x1x1, fence=200
        write_desc_word(0, 32'h0000_5000);
        write_desc_word(1, 32'd1);
        write_desc_word(2, 32'd1);
        write_desc_word(3, 32'd1);
        write_desc_word(10, 32'd200);
        write_csr(CSR_CMD_QUEUE_TAIL, 32'd3);

        // Wait for kernel A dispatch
        repeat (10) @(posedge clk);
        check_val("T4 kernelA_pc", sm_kernel_pc, 32'h0000_4000);

        // Complete kernel A
        fork
            sim_sm_done(0, 8);
            sim_sm_done(1, 10);
        join

        // Wait for kernel B
        repeat (15) @(posedge clk);
        check_val("T4 fence_after_A", fence_value, 32'd100);

        // Complete kernel B
        fork
            sim_sm_done(0, 8);
        join

        wait_idle(200);
        check_val("T4 fence_after_B", fence_value, 32'd200);

        //====================================================================
        $display("\n====================================");
        $display("Total: %0d  Passed: %0d  Failed: %0d",
                 total_tests, passed_tests, failed_tests);
        $display("====================================");
        if (failed_tests == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end


    // Timeout
    initial begin
        #500000;
        $display("[TIMEOUT] Simulation exceeded max time");
        $finish;
    end

endmodule
