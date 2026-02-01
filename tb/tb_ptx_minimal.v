//============================================================================
// Minimal PTX Test - Debug version
//============================================================================
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_ptx_minimal;
    localparam CLK_PERIOD = 10;
    localparam TIMEOUT = 5000;

    reg clk;
    reg rst_n;

    initial begin
        $display("=== PTX Minimal Debug Test ===");
        $display("Time: %0t - Starting", $time);
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // CSR Interface
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;

    // Instruction Memory Interface
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;

    // AXI4 Memory Interface (stubbed)
    wire [3:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire        m_axi_awvalid;
    reg         m_axi_awready = 1'b1;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast;
    wire        m_axi_wvalid;
    reg         m_axi_wready = 1'b1;
    reg  [3:0]  m_axi_bid = 4'b0;
    reg  [1:0]  m_axi_bresp = 2'b0;
    reg         m_axi_bvalid = 1'b0;
    wire        m_axi_bready;
    wire [3:0]  m_axi_arid;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire        m_axi_arvalid;
    reg         m_axi_arready = 1'b1;
    reg  [3:0]  m_axi_rid = 4'b0;
    reg  [31:0] m_axi_rdata = 32'b0;
    reg  [1:0]  m_axi_rresp = 2'b0;
    reg         m_axi_rlast = 1'b0;
    reg         m_axi_rvalid = 1'b0;
    wire        m_axi_rready;

    // DUT
    ralph_gpu_top #(.NUM_SM(1)) u_gpu (
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

    // Instruction memory - 1-cycle latency to allow GPU top queue to fill
    reg [31:0] imem [0:255];
    reg [31:0] imem_req_addr_d;
    reg        imem_req_pending;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_req_pending <= 1'b0;
            imem_req_addr_d <= 32'b0;
            imem_valid <= 1'b0;
            imem_data <= 64'b0;
        end else begin
            if (imem_req) begin
                imem_req_addr_d <= imem_addr;
                imem_req_pending <= 1'b1;
            end
            if (imem_req_pending) begin
                imem_data <= {imem[(imem_req_addr_d >> 2) + 1], imem[imem_req_addr_d >> 2]};
                imem_valid <= 1'b1;
                imem_req_pending <= 1'b0;
            end else begin
                imem_valid <= 1'b0;
            end
        end
    end

    // Simple AXI write response
    always @(posedge clk) begin
        if (m_axi_awvalid && m_axi_awready) begin
            m_axi_awready <= 1'b0;
        end
        if (m_axi_wvalid && m_axi_wready) begin
            m_axi_bvalid <= 1'b1;
            m_axi_bid <= m_axi_awid;
            m_axi_awready <= 1'b1;
        end
        if (m_axi_bvalid && m_axi_bready) begin
            m_axi_bvalid <= 1'b0;
        end
    end

    // Test sequence
    integer i;
    integer cycles;

    initial begin
        $display("Time: %0t - Init block start", $time);

        // Clear memory
        for (i = 0; i < 256; i = i + 1) imem[i] = 32'hFFFF_FFFF;  // NOP

        // Simple program: MOV r0, #0xCAFE; EXIT
        // MOV_IMM format: [31:26]=opcode(0x30), [25:21]=rd, [15:0]=imm16
        imem[0] = {6'b110000, 5'd0, 5'd0, 16'hCAFE};  // MOV r0, #0xCAFE
        imem[1] = {6'b001011, 26'd0};                  // EXIT

        $display("Time: %0t - Program loaded", $time);
        $display("  imem[0] = 0x%08h (MOV r0, #0xCAFE)", imem[0]);
        $display("  imem[1] = 0x%08h (EXIT)", imem[1]);

        // Reset
        rst_n = 1'b0;
        csr_wr_en = 1'b0;
        csr_addr = 12'b0;
        csr_wr_data = 32'b0;
        #100;
        rst_n = 1'b1;
        #50;
        $display("Time: %0t - Reset complete", $time);

        // Configure kernel - 1 warp (32 threads)
        @(posedge clk);
        csr_addr = 12'h00C; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h010; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h014; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h018; csr_wr_data = 32'd32; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h01C; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h020; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h008; csr_wr_data = 32'd0; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;
        csr_addr = 12'h004; csr_wr_data = 32'd1; csr_wr_en = 1'b1; @(posedge clk); csr_wr_en = 0;

        $display("Time: %0t - Kernel launched", $time);

        // Wait for completion
        cycles = 0;
        while (!irq_kernel_done && cycles < TIMEOUT) begin
            @(posedge clk);
            cycles = cycles + 1;
            if (cycles % 100 == 0) begin
                $display("Time: %0t - Cycle %0d, kernel_done=%b, imem_req=%b, imem_addr=0x%h",
                         $time, cycles, irq_kernel_done, imem_req, imem_addr);
            end
        end

        if (irq_kernel_done) begin
            $display("PASS: Kernel completed in %0d cycles", cycles);
        end else begin
            $display("FAIL: Timeout after %0d cycles", cycles);
        end

        #100;
        $finish;
    end

    // Debug: monitor fetch activity
    always @(posedge clk) begin
        if (imem_req) begin
            $display("Time: %0t - FETCH addr=0x%08h data=0x%016h valid=%b",
                     $time, imem_addr, imem_data, imem_valid);
        end
    end

    // Debug: monitor imem response path
    always @(posedge clk) begin
        if (imem_valid || imem_req_pending) begin
            $display("Time: %0t - IMEM_RESP valid=%b pending=%b addr_d=0x%08h data=0x%016h",
                     $time, imem_valid, imem_req_pending, imem_req_addr_d, imem_data);
        end
    end

    // Debug: monitor SM-side imem handshake (hierarchical taps)
    always @(posedge clk) begin
        if (u_gpu.sm_imem_req[0] || u_gpu.sm_imem_ready[0] || u_gpu.sm_imem_valids[0]) begin
            $display("Time: %0t - SM_IMEM req=%b ready=%b valid=%b data=0x%016h q_empty=%b",
                     $time, u_gpu.sm_imem_req[0], u_gpu.sm_imem_ready[0], u_gpu.sm_imem_valids[0],
                     u_gpu.sm_imem_datas[0], u_gpu.imem_q_empty);
        end
    end

    // Debug: SM internal fetch/buffer/issue signals
    always @(posedge clk) begin
        if (u_gpu.sm_gen[0].u_sm.fetch_fire ||
            u_gpu.sm_gen[0].u_sm.icache_valid ||
            (|u_gpu.sm_gen[0].u_sm.warp_inst_buf_valid) ||
            u_gpu.sm_gen[0].u_sm.issue0_fire ||
            u_gpu.sm_gen[0].u_sm.issue_valid) begin
            $display("Time: %0t - SM_INT fetch_fire=%b icache_valid=%b fetch_pipe_valid=%b buf_valid=%b issue0_fire=%b issue_valid=%b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.fetch_fire,
                     u_gpu.sm_gen[0].u_sm.icache_valid,
                     u_gpu.sm_gen[0].u_sm.fetch_pipe_valid,
                     u_gpu.sm_gen[0].u_sm.warp_inst_buf_valid,
                     u_gpu.sm_gen[0].u_sm.issue0_fire,
                     u_gpu.sm_gen[0].u_sm.issue_valid);
        end
    end

    // Debug: SM scheduler state
    always @(posedge clk) begin
        if ((|u_gpu.sm_gen[0].u_sm.warp_valid) ||
            (|u_gpu.sm_gen[0].u_sm.warp_ready) ||
            (|u_gpu.sm_gen[0].u_sm.sched_issue_valid_mask)) begin
            $display("Time: %0t - SM_SCHED warp_valid=%04b warp_ready=%04b exit_pending=%04b issue_mask=%b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.warp_valid,
                     u_gpu.sm_gen[0].u_sm.warp_ready,
                     u_gpu.sm_gen[0].u_sm.warp_exit_pending,
                     u_gpu.sm_gen[0].u_sm.sched_issue_valid_mask);
        end
    end

    // Debug: decode classification for warp 0
    always @(posedge clk) begin
        if (u_gpu.sm_gen[0].u_sm.warp_inst_buf_valid[0]) begin
            $display("Time: %0t - SM_DEC warp0 inst=0x%08h compute=%b memory=%b tensor=%b branch=%b writes_reg=%b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.warp_inst_buf[0],
                     u_gpu.sm_gen[0].u_sm.pd_is_compute[0],
                     u_gpu.sm_gen[0].u_sm.pd_is_memory[0],
                     u_gpu.sm_gen[0].u_sm.pd_is_tensor[0],
                     u_gpu.sm_gen[0].u_sm.pd_is_branch[0],
                     u_gpu.sm_gen[0].u_sm.pd_writes_reg[0]);
        end
    end

    // Debug: scheduler hazard inputs for warp 0
    always @(posedge clk) begin
        if (u_gpu.sm_gen[0].u_sm.warp_inst_buf_valid[0]) begin
            $display("Time: %0t - SM_HAZ warp0 rs1=%0d rs2=%0d rs3=%0d rd=%0d has_hazard=%b scoreboard0=%032b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.pd_rs1[0],
                     u_gpu.sm_gen[0].u_sm.pd_rs2[0],
                     u_gpu.sm_gen[0].u_sm.pd_rs3[0],
                     u_gpu.sm_gen[0].u_sm.pd_rd[0],
                     u_gpu.sm_gen[0].u_sm.u_scheduler.warp_has_hazard[0],
                     u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0]);
            $display("Time: %0t - SM_HAZ_BITS rs1=%b rs2=%b rs3=%b rd=%b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.pd_rs1[0],
                     u_gpu.sm_gen[0].u_sm.pd_rs2[0],
                     u_gpu.sm_gen[0].u_sm.pd_rs3[0],
                     u_gpu.sm_gen[0].u_sm.pd_rd[0]);
            $display("Time: %0t - SM_HAZ_SB sb[rs1]=%b sb[rs2]=%b sb[rs3]=%b sb[rd]=%b",
                     $time,
                     u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs1[0]],
                     u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs2[0]],
                     u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs3[0]],
                     u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rd[0]]);
            $display("Time: %0t - SM_HAZ_CALC raw=%b waw=%b",
                     $time,
                     (u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs1[0]] ||
                      u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs2[0]] ||
                      u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rs3[0]]),
                     (u_gpu.sm_gen[0].u_sm.pd_writes_reg[0] &&
                      u_gpu.sm_gen[0].u_sm.u_scheduler.scoreboard[0][u_gpu.sm_gen[0].u_sm.pd_rd[0]]));
        end
    end

endmodule
