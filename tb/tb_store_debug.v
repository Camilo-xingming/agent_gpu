//============================================================================
// Minimal store debug: MOV_IMM R2, 42 → ST R2 → gmem[0x100]
//============================================================================
`timescale 1ns / 1ps

module tb_store_debug;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;

    reg clk, rst_n;
    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end

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

    ralph_gpu_top #(.NUM_SM(1)) u_dut (
        .clk(clk), .rst_n(rst_n),
        .csr_wr_en(csr_wr_en), .csr_addr(csr_addr),
        .csr_wr_data(csr_wr_data), .csr_rd_data(csr_rd_data),
        .irq_kernel_done(irq_kernel_done),
        .imem_req(imem_req), .imem_addr(imem_addr),
        .imem_data(imem_data), .imem_valid(imem_valid),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );

    // ---- Instruction Memory ----
    reg [31:0] imem_storage [0:31];

    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem_storage[imem_addr[10:2] + 1], imem_storage[imem_addr[10:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    // ---- Data Memory (simple AXI slave) ----
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_wr_addr;
    wire [31:0] wr_addr_eff = (m_axi_awvalid && m_axi_awready) ? m_axi_awaddr : pending_wr_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1; m_axi_wready <= 1'b1; m_axi_bvalid <= 1'b0;
            pending_wr_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                pending_wr_addr <= m_axi_awaddr;
                $display("[AXI_AW] addr=0x%08h", m_axi_awaddr);
            end
            if (m_axi_wvalid && m_axi_wready) begin
                gmem[wr_addr_eff[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1; m_axi_bid <= m_axi_awid;
                $display("[AXI_W]  data=0x%08h -> gmem[%0d] (addr=0x%08h)", m_axi_wdata, wr_addr_eff[13:2], wr_addr_eff);
            end else if (m_axi_bvalid && m_axi_bready) m_axi_bvalid <= 1'b0;
        end
    end

    // No reads needed for this test
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b1; m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                m_axi_rdata <= gmem[m_axi_araddr[13:2]];
                m_axi_rvalid <= 1'b1; m_axi_rid <= m_axi_arid; m_axi_rlast <= 1'b1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;
            end
        end
    end

    // ---- Pipeline trace probes ----
    wire        sm_issue_valid;
    wire        sm_issue_memwr;
    wire [4:0]  sm_issue_ra, sm_issue_rb, sm_issue_rd;

    // Probe into the SM using correct hierarchy: sm_gen[0].u_sm
    assign sm_issue_valid = u_dut.sm_gen[0].u_sm.issue_valid;
    assign sm_issue_memwr = u_dut.sm_gen[0].u_sm.issue_mem_write;
    assign sm_issue_ra    = u_dut.sm_gen[0].u_sm.issue_ra;
    assign sm_issue_rb    = u_dut.sm_gen[0].u_sm.issue_rb;
    assign sm_issue_rd    = u_dut.sm_gen[0].u_sm.issue_rd;

    // Probe register file data (full SIMD width, extract lane 0 in display)
    wire [`THREADS_PER_WARP*`DATA_WIDTH-1:0] rf_data_a_full = u_dut.sm_gen[0].u_sm.rf_rd_data_a;
    wire [`THREADS_PER_WARP*`DATA_WIDTH-1:0] rf_data_b_full = u_dut.sm_gen[0].u_sm.rf_rd_data_b;

    // Probe writeback
    wire wb_en = u_dut.sm_gen[0].u_sm.wb_valid;
    wire [4:0] wb_rd_addr = u_dut.sm_gen[0].u_sm.wb_rd;
    wire [`THREADS_PER_WARP*`DATA_WIDTH-1:0] wb_data_full = u_dut.sm_gen[0].u_sm.wb_data;

    // Probe gmem arbiter
    wire gmem_nrv = u_dut.sm_gen[0].u_sm.gmem_normal_req_valid;

    // Probe warp IDs
    wire [1:0] issue_warp = u_dut.sm_gen[0].u_sm.issue_warp_id;
    wire [1:0] wb_warp = u_dut.sm_gen[0].u_sm.wb_warp_id;

    // Probe regfile write signals
    wire rf_we = u_dut.sm_gen[0].u_sm.rf_wr_en;
    wire [4:0] rf_wa = u_dut.sm_gen[0].u_sm.wb_rd;
    wire [31:0] rf_wm = u_dut.sm_gen[0].u_sm.rf_wr_mask;

    // Extract lane 0 explicitly as separate 32-bit wires
    wire [31:0] rf_a_l0 = rf_data_a_full[31:0];
    wire [31:0] rf_b_l0 = rf_data_b_full[31:0];
    wire [31:0] wb_d_l0 = wb_data_full[31:0];

    // Note: cannot probe bank_regs multi-dim array cross-hierarchy in iverilog

    // Probe memory interface internals
    wire [31:0] memif_addr_l0 = u_dut.sm_gen[0].u_sm.u_mem_if.addr_buf[31:0];
    wire [31:0] memif_wdata_l0 = u_dut.sm_gen[0].u_sm.u_mem_if.wdata_buf[31:0];
    wire [31:0] memif_mask = u_dut.sm_gen[0].u_sm.u_mem_if.mask_buf;
    wire [5:0]  memif_lane = u_dut.sm_gen[0].u_sm.u_mem_if.current_lane;
    wire [2:0]  memif_state = u_dut.sm_gen[0].u_sm.u_mem_if.state;

    always @(posedge clk) begin
        if (sm_issue_valid) begin
            $display("[ISSUE @%0t] warp=%0d memwr=%b ra=R%0d rb=R%0d rd=R%0d rf_a_l0=0x%08h rf_b_l0=0x%08h gmem_nrv=%b",
                     $time, issue_warp, sm_issue_memwr, sm_issue_ra, sm_issue_rb, sm_issue_rd,
                     rf_a_l0, rf_b_l0, gmem_nrv);
        end
        if (wb_en) begin
            $display("[WB @%0t] warp=%0d rd=R%0d data_l0=0x%08h wr_en=%b wr_mask=0x%08h", $time, wb_warp, wb_rd_addr, wb_d_l0, rf_we, rf_wm);
        end
        if (rf_we) begin
            $display("[RF_WR @%0t] warp=%0d addr=R%0d mask=0x%08h", $time, wb_warp, rf_wa, rf_wm);
        end
        // Trace memory interface state transitions
        if (memif_state != 0) begin
            $display("[MEMIF @%0t] state=%0d lane=%0d mask=0x%08h addr_l0=0x%08h wdata_l0=0x%08h",
                     $time, memif_state, memif_lane, memif_mask, memif_addr_l0, memif_wdata_l0);
        end
    end

    // ---- CSR write helper ----
    task csr_write(input [11:0] a, input [31:0] d);
        begin @(posedge clk); csr_addr <= a; csr_wr_data <= d; csr_wr_en <= 1'b1;
              @(posedge clk); csr_wr_en <= 1'b0; end
    endtask

    integer i, cycles;

    initial begin
        $dumpfile("store_debug.vcd");
        $dumpvars(0, tb_store_debug);

        rst_n = 0; csr_wr_en = 0; csr_addr = 0; csr_wr_data = 0;
        m_axi_bresp = 0; m_axi_rresp = 0;

        // Init memories
        for (i = 0; i < 32; i = i+1) imem_storage[i] = {`OP_EXIT, 26'd0};
        for (i = 0; i < 4096; i = i+1) gmem[i] = 32'hDEADBEEF;

        $display("=== Store Debug Test ===");

        #100; rst_n = 1; #50;

        // Program:
        //   MOV_IMM R2, 42       ; R2 = 42
        //   MOV_IMM R10, 0x100   ; R10 = 0x100 (store address)
        //   ST_GLOBAL [R10], R2  ; gmem[0x100] = R2 = 42
        //   EXIT
        imem_storage[0] = {`OP_MOV_IMM, 5'd2,  5'd0, 16'd42};     // R2 = 42
        imem_storage[1] = {`OP_MOV_IMM, 5'd10, 5'd0, 16'h0100};   // R10 = 0x100
        imem_storage[2] = {`OP_ST_GLOBAL, 5'd0, 5'd10, 5'd2, 5'd0, 6'd0};  // ST [R10+0], R2
        imem_storage[3] = {`OP_EXIT, 26'd0};

        // Launch kernel: 1 warp, 1 thread per block
        csr_write(12'h008, 32'h0);    // PC = 0
        csr_write(12'h00C, 32'h1);    // grid_dim = 1
        csr_write(12'h018, 32'h1);    // block_dim = 1
        csr_write(12'h004, 32'h1);    // launch

        cycles = 0;
        while (!irq_kernel_done && cycles < 500) begin
            @(posedge clk);
            cycles = cycles + 1;
        end

        #200;

        if (cycles >= 500) begin
            $display("*** TIMEOUT after %0d cycles ***", cycles);
        end else begin
            $display("Kernel done in %0d cycles", cycles);
        end

        $display("gmem[0x100>>2] = gmem[%0d] = 0x%08h (expected 0x0000002A = 42)",
                 32'h100 >> 2, gmem[32'h100 >> 2]);
        $display("gmem[0] = 0x%08h", gmem[0]);

        $finish;
    end

endmodule
