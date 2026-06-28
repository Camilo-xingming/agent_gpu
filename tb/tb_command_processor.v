//============================================================================
// Testbench for Command Processor (Issue #151/#266/#522)
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
    localparam CSR_KERNEL_PC         = 12'h008;
    localparam CSR_GRID_DIM_X        = 12'h00C;
    localparam CSR_GRID_DIM_Y        = 12'h010;
    localparam CSR_GRID_DIM_Z        = 12'h014;
    localparam CSR_BLOCK_DIM_X       = 12'h018;
    localparam CSR_BLOCK_DIM_Y       = 12'h01C;
    localparam CSR_BLOCK_DIM_Z       = 12'h020;
    localparam CSR_CMD_QUEUE_SIZE    = 12'h038;
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

    wire [NUM_SM-1:0] sm_kernel_start;
    wire [31:0]       sm_kernel_pc;
    wire [31:0]       sm_block_dim_x, sm_block_dim_y, sm_block_dim_z;
    wire [31:0]       sm_grid_dim_x, sm_grid_dim_y, sm_grid_dim_z;
    wire              gpu_busy;
    wire              irq_kernel_done;
    wire [31:0]       fence_value;
    wire              kernel_launch_pulse;

    // SM done simulation
    reg [NUM_SM-1:0] sm_done_reg;

    // sm_block_id flattened buses
    wire [NUM_SM*32-1:0] sm_block_id_x;
    wire [NUM_SM*32-1:0] sm_block_id_y;
    wire [NUM_SM*32-1:0] sm_block_id_z;

    // Phase 2 AXI stubs
    wire        m_axi_arvalid;
    wire [31:0] m_axi_araddr;
    reg         m_axi_arready;
    reg  [31:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rvalid;
    wire        m_axi_rready;

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    command_processor #(
        .NUM_SM      (NUM_SM),
        .QUEUE_DEPTH (QUEUE_DEPTH),
        .DESC_WORDS  (DESC_WORDS)
    ) u_cp (
        .clk               (clk),
        .rst_n             (rst_n),
        .csr_wr_en         (csr_wr_en),
        .csr_addr          (csr_addr),
        .csr_wr_data       (csr_wr_data),
        .csr_rd_data       (csr_rd_data),
        .csr_rd_valid      (csr_rd_valid),
        .sm_kernel_start   (sm_kernel_start),
        .sm_kernel_pc      (sm_kernel_pc),
        .sm_block_id_x     (sm_block_id_x),
        .sm_block_id_y     (sm_block_id_y),
        .sm_block_id_z     (sm_block_id_z),
        .sm_block_dim_x    (sm_block_dim_x),
        .sm_block_dim_y    (sm_block_dim_y),
        .sm_block_dim_z    (sm_block_dim_z),
        .sm_grid_dim_x     (sm_grid_dim_x),
        .sm_grid_dim_y     (sm_grid_dim_y),
        .sm_grid_dim_z     (sm_grid_dim_z),
        .sm_done           (sm_done_reg),
        .gpu_busy          (gpu_busy),
        .irq_kernel_done   (irq_kernel_done),
        .fence_value       (fence_value),
        .kernel_launch_pulse(kernel_launch_pulse),
        .m_axi_arvalid     (m_axi_arvalid),
        .m_axi_araddr      (m_axi_araddr),
        .m_axi_arready     (m_axi_arready),
        .m_axi_rdata       (m_axi_rdata),
        .m_axi_rresp       (m_axi_rresp),
        .m_axi_rvalid      (m_axi_rvalid),
        .m_axi_rready      (m_axi_rready)
    );

    //------------------------------------------------------------------------
    // Test infrastructure
    //------------------------------------------------------------------------
    integer total_tests, passed_tests, failed_tests;
    integer doorbell_seq;

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

    task push_kernel_desc(
        input [31:0] kernel_pc,
        input [31:0] grid_x,
        input [31:0] grid_y,
        input [31:0] grid_z,
        input [31:0] block_x,
        input [31:0] block_y,
        input [31:0] block_z,
        input [31:0] fence_id
    );
        begin
            write_desc_word(0,  kernel_pc);
            write_desc_word(1,  grid_x);
            write_desc_word(2,  grid_y);
            write_desc_word(3,  grid_z);
            write_desc_word(4,  block_x);
            write_desc_word(5,  block_y);
            write_desc_word(6,  block_z);
            write_desc_word(10, fence_id);
            doorbell_seq = doorbell_seq + 1;
            write_csr(CSR_CMD_QUEUE_TAIL, doorbell_seq[31:0]);
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
                $fatal(1, "[FAIL] %0s: expected 0x%08h got 0x%08h", name, expected, actual);
            end
        end
    endtask

    task check_csr_mask(
        input [255:0] name,
        input [11:0] addr,
        input [31:0] mask,
        input [31:0] expected
    );
        begin
            csr_addr = addr;
            @(posedge clk);
            check_val(name, csr_rd_data & mask, expected);
        end
    endtask

    //------------------------------------------------------------------------
    // Main test sequence
    //------------------------------------------------------------------------
    initial begin
        integer i;

        $dumpfile("tb_command_processor.vcd");
        $dumpvars(0, tb_command_processor);

        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        doorbell_seq = 0;

        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 12'b0;
        csr_wr_data = 32'b0;
        sm_done_reg = {NUM_SM{1'b0}};
        m_axi_arready = 1'b0;
        m_axi_rdata = 32'b0;
        m_axi_rresp = 2'b0;
        m_axi_rvalid = 1'b0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        //====================================================================
        // Test 1: Reset state
        //====================================================================
        $display("\n=== Test 1: Reset state ===");
        check_csr_mask("T1 cp_enable reset", CSR_CP_STATUS, 32'h0000_0080, 32'h0000_0000);

        //====================================================================
        // Test 2: Enable queue mode reflected in CSR
        //====================================================================
        $display("\n=== Test 2: Enable queue mode ===");
        write_csr(CSR_GPU_CONTROL, 32'h2);
        repeat (2) @(posedge clk);
        check_csr_mask("T2 cp_enable set", CSR_CP_STATUS, 32'h0000_0080, 32'h0000_0080);

        //====================================================================
        // Test 3: Single queued dispatch starts kernel
        //====================================================================
        $display("\n=== Test 3: Single queued dispatch ===");
        write_csr(CSR_GPU_STATUS, 32'h1); // clear irq
        push_kernel_desc(32'h0000_1000, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd11);
        repeat (6) @(posedge clk);
        check_val("T3 gpu_busy", {31'b0, gpu_busy}, 32'd1);
        check_val("T3 kernel_pc", sm_kernel_pc, 32'h0000_1000);

        //====================================================================
        // Test 4: Completion raises IRQ with default fence signal
        //====================================================================
        $display("\n=== Test 4: Completion IRQ default fence ===");
        fork
            sim_sm_done(0, 6);
        join
        wait_idle(100);
        check_val("T4 irq", {31'b0, irq_kernel_done}, 32'd1);

        //====================================================================
        // Test 5: IRQ clear on GPU_STATUS write
        //====================================================================
        $display("\n=== Test 5: IRQ clear ===");
        write_csr(CSR_GPU_STATUS, 32'h1);
        repeat (2) @(posedge clk);
        check_val("T5 irq cleared", {31'b0, irq_kernel_done}, 32'd0);

        //====================================================================
        // Test 6: Fence mismatch suppresses IRQ
        //====================================================================
        $display("\n=== Test 6: Fence mismatch no IRQ ===");
        write_csr(CSR_CMD_FENCE_SIGNAL, 32'd999);
        push_kernel_desc(32'h0000_1100, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd123);
        fork
            sim_sm_done(0, 8);
        join
        wait_idle(120);
        check_val("T6 fence_value", fence_value, 32'd123);
        check_val("T6 irq low", {31'b0, irq_kernel_done}, 32'd0);

        //====================================================================
        // Test 7: Fence match raises IRQ
        //====================================================================
        $display("\n=== Test 7: Fence match IRQ ===");
        write_csr(CSR_GPU_STATUS, 32'h1);
        write_csr(CSR_CMD_FENCE_SIGNAL, 32'd555);
        push_kernel_desc(32'h0000_1200, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd555);
        fork
            sim_sm_done(0, 8);
        join
        wait_idle(120);
        check_val("T7 irq high", {31'b0, irq_kernel_done}, 32'd1);
        write_csr(CSR_GPU_STATUS, 32'h1);
        write_csr(CSR_CMD_FENCE_SIGNAL, 32'hFFFF_FFFF);

        //====================================================================
        // Test 8: Invalid queued descriptor sets invalid flag
        //====================================================================
        $display("\n=== Test 8: Invalid queued descriptor ===");
        write_csr(CSR_CP_STATUS, 32'h0000_0200); // clear invalid bit
        push_kernel_desc(32'h0000_1300, 32'd0, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd1);
        repeat (8) @(posedge clk);
        check_csr_mask("T8 invalid flag", CSR_CP_STATUS, 32'h0000_0200, 32'h0000_0200);
        check_val("T8 still idle", {31'b0, gpu_busy}, 32'd0);

        //====================================================================
        // Test 9: Invalid flag clear path
        //====================================================================
        $display("\n=== Test 9: Clear invalid flag ===");
        write_csr(CSR_CP_STATUS, 32'h0000_0200);
        repeat (2) @(posedge clk);
        check_csr_mask("T9 invalid cleared", CSR_CP_STATUS, 32'h0000_0200, 32'h0000_0000);

        //====================================================================
        // Test 10: Queue overflow set when full while active kernel runs
        //====================================================================
        $display("\n=== Test 10: Queue overflow set ===");
        write_csr(CSR_CP_STATUS, 32'h0000_0100); // clear overflow bit
        push_kernel_desc(32'h0000_2000, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd200);
        repeat (10) @(posedge clk);
        check_val("T10 active kernel", sm_kernel_pc, 32'h0000_2000);
        for (i = 0; i < 8; i = i + 1) begin
            push_kernel_desc(32'h0000_2100 + i, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd201 + i);
            repeat (2) @(posedge clk);
        end
        push_kernel_desc(32'h0000_22FF, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd255);
        repeat (6) @(posedge clk);
        check_csr_mask("T10 overflow flag", CSR_CP_STATUS, 32'h0000_0100, 32'h0000_0100);

        //====================================================================
        // Test 11: Overflow clear path + drain queued kernels
        //====================================================================
        $display("\n=== Test 11: Clear overflow + drain ===");
        write_csr(CSR_CP_STATUS, 32'h0000_0100);
        repeat (2) @(posedge clk);
        check_csr_mask("T11 overflow cleared", CSR_CP_STATUS, 32'h0000_0100, 32'h0000_0000);
        for (i = 0; i < 9; i = i + 1) begin
            fork
                sim_sm_done(0, 4);
            join
            repeat (12) @(posedge clk);
        end
        wait_idle(800);
        check_val("T11 idle after drain", {31'b0, gpu_busy}, 32'd0);

        //====================================================================
        // Test 12: Queued dispatch replaces legacy launch path
        //====================================================================
        $display("\n=== Test 12: Queued dispatch (modern path) ===");
        write_csr(CSR_GPU_CONTROL, 32'h2); // force queue mode enabled
        write_csr(CSR_GPU_STATUS, 32'h1);  // clear irq
        push_kernel_desc(32'h0000_3000, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd300);
        repeat (6) @(posedge clk);
        check_val("T12 gpu_busy", {31'b0, gpu_busy}, 32'd1);
        check_val("T12 kernel_pc", sm_kernel_pc, 32'h0000_3000);
        fork
            sim_sm_done(0, 6);
        join
        wait_idle(120);
        write_csr(CSR_GPU_STATUS, 32'h1);

        //====================================================================
        // Test 13: Queued invalid dimensions set invalid flag
        //====================================================================
        $display("\n=== Test 13: Queued invalid dimensions ===");
        write_csr(CSR_CP_STATUS, 32'h0000_0200);
        push_kernel_desc(32'h0000_3010, 32'd0, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd301);
        repeat (8) @(posedge clk);
        check_csr_mask("T13 invalid flag", CSR_CP_STATUS, 32'h0000_0200, 32'h0000_0200);
        check_val("T13 still idle", {31'b0, gpu_busy}, 32'd0);
        write_csr(CSR_CP_STATUS, 32'h0000_0200);

        //====================================================================
        // Test 14: 1D dispatch metadata load
        //====================================================================
        $display("\n=== Test 14: 1D dispatch metadata ===");
        rst_n = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        repeat (3) @(posedge clk);
        doorbell_seq = 0;
        sm_done_reg = {NUM_SM{1'b0}};
        write_csr(CSR_GPU_CONTROL, 32'h2);
        repeat (2) @(posedge clk);
        push_kernel_desc(32'h0000_4000, 32'd4, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd400);
        repeat (20) @(posedge clk);
        check_val("T14 kernel_pc", sm_kernel_pc, 32'h0000_4000);
        check_val("T14 grid_x", sm_grid_dim_x, 32'd4);
        check_val("T14 sm0 block_x", sm_block_id_x[31:0], 32'd0);

        //====================================================================
        // Test 15: 1D completion sequencing drains kernel
        //====================================================================
        $display("\n=== Test 15: 1D completion sequencing ===");
        for (i = 0; i < 4; i = i + 1) begin
            fork
                sim_sm_done(0, 4);
                sim_sm_done(1, 4);
            join
            repeat (8) @(posedge clk);
        end
        wait_idle(300);
        check_val("T15 idle", {31'b0, gpu_busy}, 32'd0);
        check_val("T15 fence", fence_value, 32'd400);

        //====================================================================
        // Test 16: 2D descriptor metadata load
        //====================================================================
        $display("\n=== Test 16: 2D descriptor metadata ===");
        push_kernel_desc(32'h0000_4100, 32'd2, 32'd2, 32'd1, 32'd32, 32'd1, 32'd1, 32'd410);
        repeat (20) @(posedge clk);
        check_val("T16 grid_x", sm_grid_dim_x, 32'd2);
        check_val("T16 grid_y", sm_grid_dim_y, 32'd2);
        check_val("T16 sm0 x", sm_block_id_x[31:0], 32'd0);
        check_val("T16 sm0 y", sm_block_id_y[31:0], 32'd0);
        for (i = 0; i < 4; i = i + 1) begin
            fork
                sim_sm_done(0, 4);
                sim_sm_done(1, 4);
            join
            repeat (8) @(posedge clk);
        end
        wait_idle(300);
        check_val("T16 fence", fence_value, 32'd410);

        //====================================================================
        // Test 17: 3D descriptor metadata load
        //====================================================================
        $display("\n=== Test 17: 3D descriptor metadata ===");
        push_kernel_desc(32'h0000_4200, 32'd1, 32'd1, 32'd2, 32'd32, 32'd1, 32'd1, 32'd420);
        repeat (20) @(posedge clk);
        check_val("T17 grid_z", sm_grid_dim_z, 32'd2);
        check_val("T17 sm0 z", sm_block_id_z[31:0], 32'd0);
        for (i = 0; i < 2; i = i + 1) begin
            fork
                sim_sm_done(0, 4);
                sim_sm_done(1, 4);
            join
            repeat (8) @(posedge clk);
        end
        wait_idle(220);
        check_val("T17 fence", fence_value, 32'd420);

        //====================================================================
        // Test 18: Queue mode remains enabled across kernels
        //====================================================================
        $display("\n=== Test 18: Queue mode sticky enable ===");
        check_csr_mask("T18 cp_enable", CSR_CP_STATUS, 32'h0000_0080, 32'h0000_0080);

        //====================================================================
        // Test 19: FIFO order preserved for back-to-back queued kernels
        //====================================================================
        $display("\n=== Test 19: Back-to-back FIFO order ===");
        push_kernel_desc(32'h0000_5000, 32'd2, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd500);
        repeat (20) @(posedge clk);
        check_val("T19 active A", sm_kernel_pc, 32'h0000_5000);
        push_kernel_desc(32'h0000_5100, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd510);
        repeat (4) @(posedge clk);
        check_csr_mask("T19 queued count", CSR_CP_STATUS, 32'h0000_000F, 32'h0000_0001);
        fork
            sim_sm_done(0, 4);
            sim_sm_done(1, 6);
        join
        repeat (12) @(posedge clk);
        check_val("T19 active B", sm_kernel_pc, 32'h0000_5100);
        fork
            sim_sm_done(0, 4);
        join
        wait_idle(200);
        check_val("T19 fence B", fence_value, 32'd510);

        //====================================================================
        // Test 20: CP state reflects WAIT and IDLE transitions
        //====================================================================
        $display("\n=== Test 20: CP state WAIT/IDLE ===");
        push_kernel_desc(32'h0000_5200, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd520);
        repeat (6) @(posedge clk);
        check_csr_mask("T20 state wait", CSR_CP_STATUS, 32'h0000_0070, 32'h0000_0030);
        fork
            sim_sm_done(0, 4);
        join
        wait_idle(120);
        check_csr_mask("T20 state idle", CSR_CP_STATUS, 32'h0000_0070, 32'h0000_0000);

        //====================================================================
        // Test 21: Busy path remains active until all completions retire
        //====================================================================
        $display("\n=== Test 21: Busy path retires correctly ===");
        push_kernel_desc(32'h0000_5300, 32'd3, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1, 32'd530);
        repeat (10) @(posedge clk);
        check_val("T21 no start while busy", {30'b0, sm_kernel_start}, 32'd0);
        fork
            sim_sm_done(0, 4);
        join
        repeat (10) @(posedge clk);
        check_val("T21 still busy", {31'b0, gpu_busy}, 32'd1);
        fork
            sim_sm_done(1, 4);
            sim_sm_done(0, 6);
        join
        wait_idle(240);
        check_val("T21 idle", {31'b0, gpu_busy}, 32'd0);

        //====================================================================
        // Test 22: Queue size CSR remains stable
        //====================================================================
        $display("\n=== Test 22: Queue size CSR ===");
        check_csr_mask("T22 queue size", CSR_CMD_QUEUE_SIZE, 32'hFFFF_FFFF, QUEUE_DEPTH);

        //====================================================================
        $display("\n====================================");
        $display("Total: %0d  Passed: %0d  Failed: %0d", total_tests, passed_tests, failed_tests);
        $display("====================================");
        if (failed_tests == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    // Timeout
    initial begin
        #800000;
        $fatal(1, "[TIMEOUT] Simulation exceeded max time");
    end

endmodule
