`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_ralph_gpu_top_cp_integration;

    localparam CLK_PERIOD = 10;
    localparam NUM_SM = 2;

    localparam CSR_GPU_STATUS   = 12'h000;
    localparam CSR_GPU_CONTROL  = 12'h004;
    localparam CSR_KERNEL_PC    = 12'h008;
    localparam CSR_GRID_DIM_X   = 12'h00C;
    localparam CSR_GRID_DIM_Y   = 12'h010;
    localparam CSR_GRID_DIM_Z   = 12'h014;
    localparam CSR_BLOCK_DIM_X  = 12'h018;
    localparam CSR_BLOCK_DIM_Y  = 12'h01C;
    localparam CSR_BLOCK_DIM_Z  = 12'h020;
    localparam CSR_ERROR_STATUS = 12'h024;
    localparam CSR_PERF_BASE    = 12'h100;

    integer total_tests;
    integer passed_tests;
    integer failed_tests;

    integer launch_count;
    integer sm0_launches;
    integer sm1_launches;
    reg [31:0] seen_block_mask;
    reg kernel_launch_seen;

    reg clk;
    reg rst_n;

    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    wire [3:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready;

    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready;

    reg  [3:0]  m_axi_bid;
    reg  [1:0]  m_axi_bresp;
    reg         m_axi_bvalid;
    wire        m_axi_bready;

    wire [3:0]  m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready;

    reg  [3:0]  m_axi_rid;
    reg  [31:0] m_axi_rdata;
    reg  [1:0]  m_axi_rresp;
    reg         m_axi_rlast;
    reg         m_axi_rvalid;
    wire        m_axi_rready;

    reg [31:0] instruction_mem [0:255];
    reg        pending_imem_req;
    reg [31:0] pending_imem_addr;

    integer i;

    function [31:0] encode_nop;
        begin
            encode_nop = {`OP_NOP, 26'b0};
        end
    endfunction

    function [31:0] encode_exit;
        begin
            encode_exit = {`OP_EXIT, 26'b0};
        end
    endfunction

    ralph_gpu_top #(
        .NUM_SM(NUM_SM)
    ) u_gpu (
        .clk            (clk),
        .rst_n          (rst_n),
        .csr_wr_en      (csr_wr_en),
        .csr_addr       (csr_addr),
        .csr_wr_data    (csr_wr_data),
        .csr_rd_data    (csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req       (imem_req),
        .imem_addr      (imem_addr),
        .imem_data      (imem_data),
        .imem_valid     (imem_valid),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_imem_req  <= 1'b0;
            pending_imem_addr <= 32'b0;
            imem_valid        <= 1'b0;
            imem_data         <= 64'b0;
        end else begin
            imem_valid <= pending_imem_req;
            if (pending_imem_req) begin
                imem_data <= {
                    instruction_mem[(pending_imem_addr >> 2) + 1],
                    instruction_mem[pending_imem_addr >> 2]
                };
            end
            pending_imem_req  <= imem_req;
            pending_imem_addr <= imem_addr;
        end
    end

    task record_block;
        input integer sm_id;
        input [31:0] block_x;
        begin
            launch_count = launch_count + 1;
            if (sm_id == 0)
                sm0_launches = sm0_launches + 1;
            else
                sm1_launches = sm1_launches + 1;

            if (block_x < 32)
                seen_block_mask[block_x] = 1'b1;
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (rst_n) begin
            if (u_gpu.kernel_launch_pulse)
                kernel_launch_seen = 1'b1;
            if (u_gpu.sm_kernel_start[0])
                record_block(0, u_gpu.sm_block_id_x[31:0]);
            if (u_gpu.sm_kernel_start[1])
                record_block(1, u_gpu.sm_block_id_x[63:32]);
        end
    end

    task reset_dispatch_trace;
        begin
            launch_count = 0;
            sm0_launches = 0;
            sm1_launches = 0;
            seen_block_mask = 32'b0;
            kernel_launch_seen = 1'b0;
        end
    endtask

    task check_true;
        input [255:0] name;
        input cond;
        begin
            total_tests = total_tests + 1;
            if (cond) begin
                passed_tests = passed_tests + 1;
                $display("[PASS] %0s", name);
            end else begin
                failed_tests = failed_tests + 1;
                $display("[FAIL] %0s", name);
            end
        end
    endtask

    task check_eq;
        input [255:0] name;
        input [31:0] actual;
        input [31:0] expected;
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

    task write_csr;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_wr_en   <= 1'b1;
            csr_addr    <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en   <= 1'b0;
        end
    endtask

    task read_csr;
        input [11:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            csr_addr <= addr;
            #1 data = csr_rd_data;
        end
    endtask

    task clear_irq;
        begin
            write_csr(CSR_GPU_STATUS, 32'h1);
        end
    endtask

    task launch_legacy_kernel;
        input [31:0] kernel_pc;
        input [31:0] grid_x;
        input [31:0] grid_y;
        input [31:0] grid_z;
        input [31:0] block_x;
        input [31:0] block_y;
        input [31:0] block_z;
        begin
            write_csr(CSR_KERNEL_PC, kernel_pc);
            write_csr(CSR_GRID_DIM_X, grid_x);
            write_csr(CSR_GRID_DIM_Y, grid_y);
            write_csr(CSR_GRID_DIM_Z, grid_z);
            write_csr(CSR_BLOCK_DIM_X, block_x);
            write_csr(CSR_BLOCK_DIM_Y, block_y);
            write_csr(CSR_BLOCK_DIM_Z, block_z);
            write_csr(CSR_GPU_CONTROL, 32'h1);
        end
    endtask

    task wait_irq;
        input integer timeout_cycles;
        output integer timed_out;
        integer cnt;
        begin
            cnt = 0;
            while (!irq_kernel_done && cnt < timeout_cycles) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
            timed_out = (irq_kernel_done) ? 0 : 1;
        end
    endtask

    task wait_idle;
        input integer timeout_cycles;
        output integer timed_out;
        integer cnt;
        begin
            cnt = 0;
            while (u_gpu.gpu_busy && cnt < timeout_cycles) begin
                @(posedge clk);
                cnt = cnt + 1;
            end
            timed_out = (u_gpu.gpu_busy) ? 1 : 0;
        end
    endtask

    reg [31:0] rd_data;
    integer timeout_flag;

    initial begin
        $dumpfile("tb_ralph_gpu_top_cp_integration.vcd");
        $dumpvars(0, tb_ralph_gpu_top_cp_integration);

        total_tests  = 0;
        passed_tests = 0;
        failed_tests = 0;

        csr_wr_en   = 1'b0;
        csr_addr    = 12'b0;
        csr_wr_data = 32'b0;

        m_axi_awready = 1'b1;
        m_axi_wready  = 1'b1;
        m_axi_bid     = 4'b0;
        m_axi_bresp   = 2'b0;
        m_axi_bvalid  = 1'b0;
        m_axi_arready = 1'b1;
        m_axi_rid     = 4'b0;
        m_axi_rdata   = 32'b0;
        m_axi_rresp   = 2'b0;
        m_axi_rlast   = 1'b0;
        m_axi_rvalid  = 1'b0;

        for (i = 0; i < 256; i = i + 1)
            instruction_mem[i] = encode_nop();
        instruction_mem[0] = encode_nop();
        instruction_mem[1] = encode_exit();

        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // Test 1: Legacy CSR launch path works end-to-end.
        reset_dispatch_trace();
        clear_irq();
        launch_legacy_kernel(32'd0, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1);
        wait_irq(3000, timeout_flag);
        check_true("T1 legacy launch finishes", !timeout_flag);
        check_true("T1 launch pulse observed", kernel_launch_seen);
        check_true("T1 at least one block dispatched", launch_count >= 1);

        // Test 2: Multi-block dispatch (grid > NUM_SM) rotates block IDs.
        reset_dispatch_trace();
        clear_irq();
        launch_legacy_kernel(32'd0, 32'd5, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1);
        wait_irq(8000, timeout_flag);
        check_true("T2 multi-block launch finishes", !timeout_flag);
        check_eq("T2 exactly five dispatches", launch_count, 32'd5);
        check_true("T2 saw block IDs 0..4", (seen_block_mask[4:0] == 5'b11111));
        check_true("T2 both SMs received work", (sm0_launches > 0) && (sm1_launches > 0));

        // Test 3: Error status is cleared on launch pulse.
        reset_dispatch_trace();
        clear_irq();
        u_gpu.error_pending = 1'b1;
        u_gpu.error_code = 4'hA;
        u_gpu.error_sm_id = 8'h01;
        u_gpu.error_warp_id = 8'h02;
        read_csr(CSR_ERROR_STATUS, rd_data);
        check_true("T3 error status starts asserted", rd_data[0]);

        launch_legacy_kernel(32'd0, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1);
        repeat (3) @(posedge clk);
        read_csr(CSR_ERROR_STATUS, rd_data);
        check_true("T3 launch pulse observed", kernel_launch_seen);
        check_eq("T3 error status cleared after launch", rd_data, 32'd0);
        wait_irq(3000, timeout_flag);
        check_true("T3 launch still completes", !timeout_flag);

        // Test 4: Performance counters clear on launch pulse.
        reset_dispatch_trace();
        clear_irq();
        wait_idle(500, timeout_flag);
        check_true("T4 GPU idle before counter seed", !timeout_flag);

        @(posedge clk);
        u_gpu.u_perf_counters.counters[0] = 48'd1234;
        read_csr(CSR_PERF_BASE, rd_data);
        check_eq("T4 seeded perf counter visible", rd_data, 32'd1234);

        launch_legacy_kernel(32'd0, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1);
        repeat (3) @(posedge clk);
        read_csr(CSR_PERF_BASE, rd_data);
        check_true("T4 launch pulse observed", kernel_launch_seen);
        check_true("T4 perf counter dropped after launch clear", rd_data < 32'd1234);
        wait_irq(3000, timeout_flag);
        check_true("T4 launch completes", !timeout_flag);

        $display("\n========================================");
        $display("Command Processor Top Integration Summary");
        $display("========================================");
        $display("Total tests : %0d", total_tests);
        $display("Passed      : %0d", passed_tests);
        $display("Failed      : %0d", failed_tests);

        if (failed_tests == 0) begin
            $display("[SUCCESS] All integration checks passed.");
            $finish;
        end else begin
            $fatal(1, "[FAIL] %0d checks failed.", failed_tests);
        end
    end

endmodule
