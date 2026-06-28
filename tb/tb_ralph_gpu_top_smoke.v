`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_ralph_gpu_top_smoke;

    localparam CLK_PERIOD = 10;
    localparam NUM_SM = 1;

    localparam CSR_GPU_STATUS  = 12'h000;
    localparam CSR_GPU_CONTROL = 12'h004;
    localparam CSR_KERNEL_PC   = 12'h008;
    localparam CSR_GRID_DIM_X  = 12'h00C;
    localparam CSR_GRID_DIM_Y  = 12'h010;
    localparam CSR_GRID_DIM_Z  = 12'h014;
    localparam CSR_BLOCK_DIM_X = 12'h018;
    localparam CSR_BLOCK_DIM_Y = 12'h01C;
    localparam CSR_BLOCK_DIM_Z = 12'h020;
    localparam CSR_CP_STATUS   = 12'h04C;

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
    reg [31:0] global_mem [0:4095];

    integer i;
    integer total_tests;
    integer passed_tests;
    integer failed_tests;

    integer axi_read_count;
    integer axi_write_count;
    reg [31:0] last_read_addr;
    reg [31:0] last_write_addr;
    reg [31:0] last_write_data;
    reg        kernel_launch_seen;
    reg        sm_start_seen;
    integer    kernel_launch_count;
    integer    sim_cycle;
    integer    last_launch_cycle;
    integer    last_irq_rise_cycle;
    reg        irq_prev;
    integer    wb_count;
    reg        wb_rd1_seen;
    reg [31:0] wb_rd1_data_lane0;

    reg        pending_imem_req;
    reg [31:0] pending_imem_addr;

    reg        pending_axi_read;
    reg [31:0] pending_axi_addr;
    reg [1:0]  axi_read_delay;

    reg        aw_pending;
    reg [31:0] pending_awaddr;

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

    function [31:0] encode_mov_imm;
        input [4:0]  rd;
        input [15:0] imm;
        begin
            encode_mov_imm = {`OP_MOV_IMM, rd, 5'b0, imm};
        end
    endfunction

    function [31:0] encode_ld_global;
        input [4:0] rd;
        input [4:0] ra;
        begin
            encode_ld_global = {`OP_LD_GLOBAL, rd, ra, 16'b0};
        end
    endfunction

    function [31:0] encode_st_global;
        input [4:0] ra;
        input [4:0] rb;
        begin
            encode_st_global = {`OP_ST_GLOBAL, 5'b0, ra, rb, 11'b0};
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

    // Instruction memory model: one-cycle delayed 64-bit fetch.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pending_imem_req <= 1'b0;
            pending_imem_addr <= 32'b0;
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            imem_valid <= pending_imem_req;
            if (pending_imem_req) begin
                imem_data <= {
                    instruction_mem[(pending_imem_addr >> 2) + 1],
                    instruction_mem[pending_imem_addr >> 2]
                };
            end
            pending_imem_req <= imem_req;
            pending_imem_addr <= imem_addr;
        end
    end

    // AXI memory model for LD/ST smoke path.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 32'b0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            m_axi_rid <= 4'b0;
            pending_axi_read <= 1'b0;
            pending_axi_addr <= 32'b0;
            axi_read_delay <= 2'b0;

            m_axi_awready <= 1'b1;
            m_axi_wready <= 1'b1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            m_axi_bid <= 4'b0;
            aw_pending <= 1'b0;
            pending_awaddr <= 32'b0;

            axi_read_count <= 0;
            axi_write_count <= 0;
            last_read_addr <= 32'b0;
            last_write_addr <= 32'b0;
            last_write_data <= 32'b0;
            wb_count <= 0;
            wb_rd1_seen <= 1'b0;
            wb_rd1_data_lane0 <= 32'b0;
        end else begin
            // Read address handshake -> delayed read response.
            if (m_axi_arvalid && m_axi_arready) begin
                pending_axi_read <= 1'b1;
                pending_axi_addr <= m_axi_araddr;
                last_read_addr <= m_axi_araddr;
                axi_read_count <= axi_read_count + 1;
                axi_read_delay <= 2'd1;
                m_axi_arready <= 1'b0;
            end else if (pending_axi_read && axi_read_delay != 0) begin
                axi_read_delay <= axi_read_delay - 1'b1;
            end else if (pending_axi_read && axi_read_delay == 0) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= global_mem[pending_axi_addr[15:2]];
                m_axi_rid <= m_axi_arid;
                m_axi_rlast <= 1'b1;
                pending_axi_read <= 1'b0;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
                m_axi_rlast <= 1'b0;
                m_axi_arready <= 1'b1;
            end

            // Write address/data + response.
            if (m_axi_awvalid && m_axi_awready) begin
                aw_pending <= 1'b1;
                pending_awaddr <= m_axi_awaddr;
            end

            if (m_axi_wvalid && m_axi_wready && aw_pending) begin
                global_mem[pending_awaddr[15:2]] <= m_axi_wdata;
                last_write_addr <= pending_awaddr;
                last_write_data <= m_axi_wdata;
                axi_write_count <= axi_write_count + 1;
                aw_pending <= 1'b0;
                m_axi_bvalid <= 1'b1;
                m_axi_bid <= m_axi_awid;
            end else if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
            end

            if (u_gpu.sm_gen[0].u_sm.wb_valid) begin
                wb_count <= wb_count + 1;
                if (u_gpu.sm_gen[0].u_sm.wb_warp_id == 0 &&
                    u_gpu.sm_gen[0].u_sm.wb_rd == 5'd1) begin
                    wb_rd1_seen <= 1'b1;
                    wb_rd1_data_lane0 <= u_gpu.sm_gen[0].u_sm.wb_data[31:0];
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_launch_seen <= 1'b0;
            sm_start_seen <= 1'b0;
            kernel_launch_count <= 0;
            sim_cycle <= 0;
            last_launch_cycle <= -1;
            last_irq_rise_cycle <= -1;
            irq_prev <= 1'b0;
        end else begin
            sim_cycle <= sim_cycle + 1;
            if (u_gpu.kernel_launch_pulse) begin
                kernel_launch_seen <= 1'b1;
                kernel_launch_count <= kernel_launch_count + 1;
                last_launch_cycle <= sim_cycle;
            end
            if (u_gpu.sm_kernel_start[0])
                sm_start_seen <= 1'b1;
            if (!irq_prev && irq_kernel_done)
                last_irq_rise_cycle <= sim_cycle;
            irq_prev <= irq_kernel_done;
        end
    end

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
            csr_wr_en <= 1'b1;
            csr_addr <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en <= 1'b0;
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

    task launch_kernel_cfg;
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

    task launch_kernel;
        begin
            launch_kernel_cfg(32'd0, 32'd1, 32'd1, 32'd1, 32'd32, 32'd1, 32'd1);
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
            timed_out = irq_kernel_done ? 0 : 1;
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
            timed_out = u_gpu.gpu_busy ? 1 : 0;
        end
    endtask

    integer timeout_flag;
    integer prev_irq_cycle;
    integer reads_after_first;
    integer writes_after_first;
    reg [31:0] rd_data;

    initial begin
        $dumpfile("tb_ralph_gpu_top_smoke.vcd");
        $dumpvars(0, tb_ralph_gpu_top_smoke);

        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;

        csr_wr_en = 1'b0;
        csr_addr = 12'b0;
        csr_wr_data = 32'b0;

        for (i = 0; i < 256; i = i + 1)
            instruction_mem[i] = encode_nop();
        for (i = 0; i < 4096; i = i + 1)
            global_mem[i] = 32'h0;

        // Program: MOV r10,0x1000 ; LD r1,[r10] ; MOV r11,0x1004 ; ST [r11],r1 ; EXIT
        instruction_mem[0] = encode_mov_imm(5'd10, 16'h1000);
        instruction_mem[1] = encode_ld_global(5'd1, 5'd10);
        instruction_mem[2] = encode_mov_imm(5'd11, 16'h1004);
        instruction_mem[3] = encode_st_global(5'd11, 5'd1);
        instruction_mem[4] = encode_exit();

        global_mem[16'h1000 >> 2] = 32'hDEAD_BEEF;

        rst_n = 1'b0;
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        // Test 1: First kernel run + IRQ timing/clear behavior.
        clear_irq();
        launch_kernel();
        wait_irq(12000, timeout_flag);

        check_true("T1 kernel launch pulse observed", kernel_launch_seen);
        check_true("T1 SM start observed", sm_start_seen);
        check_true("T1 kernel completed (irq)", !timeout_flag);
        check_true("T1 irq asserted at completion", irq_kernel_done);
        check_true("T1 axi read happened", axi_read_count > 0);
        check_eq("T1 read addr is source 0x1000", last_read_addr, 32'h0000_1000);
        check_true("T1 SM writeback observed", wb_count > 0);
        check_true("T1 LD writeback to R1 observed", wb_rd1_seen);
        check_eq("T1 R1 lane0 writeback equals source word", wb_rd1_data_lane0, 32'hDEAD_BEEF);

        reads_after_first = axi_read_count;
        writes_after_first = axi_write_count;
        prev_irq_cycle = last_irq_rise_cycle;

        clear_irq();
        @(posedge clk);
        check_true("T1 irq clears after status write", !irq_kernel_done);

        // Test 2: Error injection via invalid kernel config (block_dim_x=0).
        launch_kernel_cfg(32'd0, 32'd1, 32'd1, 32'd1, 32'd0, 32'd1, 32'd1);
        repeat (8) @(posedge clk);
        check_true("T2 invalid launch does not raise irq", !irq_kernel_done);
        read_csr(CSR_CP_STATUS, rd_data);
        check_true("T2 invalid command bit set", rd_data[9]);
        wait_idle(200, timeout_flag);
        check_true("T2 gpu returns idle after invalid launch", !timeout_flag);

        write_csr(CSR_GPU_CONTROL, 32'h0);
        write_csr(CSR_CP_STATUS, 32'h0000_0200);
        repeat (2) @(posedge clk);
        read_csr(CSR_CP_STATUS, rd_data);
        check_true("T2 invalid command bit clears", !rd_data[9]);

        // Test 3: Second kernel run proves sequential multi-kernel execution.
        global_mem[16'h1000 >> 2] = 32'hCAFE_1234;
        global_mem[16'h1004 >> 2] = 32'h0000_0000;
        wb_rd1_seen = 1'b0;
        wb_rd1_data_lane0 = 32'b0;

        launch_kernel();
        wait_irq(12000, timeout_flag);
        check_true("T3 second kernel completed", !timeout_flag);
        check_true("T3 second irq asserted", irq_kernel_done);
        check_true("T3 second irq edge observed", last_irq_rise_cycle > prev_irq_cycle);
        check_true("T3 multi-kernel launch count >= 2", kernel_launch_count >= 2);
        check_true("T3 axi reads increased after second launch", axi_read_count > reads_after_first);
        check_true("T3 second LD writeback observed", wb_rd1_seen);
        check_eq("T3 second lane0 writeback value", wb_rd1_data_lane0, 32'hCAFE_1234);

        clear_irq();
        @(posedge clk);
        check_true("T3 irq clears after final status write", !irq_kernel_done);

        $display("\n========================================");
        $display("GPU Top Smoke Summary");
        $display("========================================");
        $display("Total tests : %0d", total_tests);
        $display("Passed      : %0d", passed_tests);
        $display("Failed      : %0d", failed_tests);

        if (failed_tests == 0) begin
            $display("[SUCCESS] GPU top-level smoke test passed.");
            $finish;
        end else begin
            $fatal(1, "[FAIL] %0d checks failed.", failed_tests);
        end
    end

    initial begin
        #2000000;
        $fatal(1, "TIMEOUT: tb_ralph_gpu_top_smoke");
    end

endmodule
