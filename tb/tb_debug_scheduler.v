// Simple debug testbench for scheduler stall investigation
`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_debug_scheduler;
    localparam CLK_PERIOD = 10;
    
    reg clk, rst_n;
    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
    
    // CSR Interface
    reg         csr_wr_en;
    reg  [11:0] csr_addr;
    reg  [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire        irq_kernel_done;
    
    // Instruction Memory
    wire        imem_req;
    wire [31:0] imem_addr;
    reg  [63:0] imem_data;
    reg         imem_valid;
    
    // AXI (stub)
    wire [3:0]  m_axi_awid, m_axi_arid;
    wire [31:0] m_axi_awaddr, m_axi_araddr, m_axi_wdata;
    wire [7:0]  m_axi_awlen, m_axi_arlen;
    wire [2:0]  m_axi_awsize, m_axi_arsize;
    wire [1:0]  m_axi_awburst, m_axi_arburst;
    wire        m_axi_awvalid, m_axi_wvalid, m_axi_wlast, m_axi_bready, m_axi_arvalid, m_axi_rready;
    wire [3:0]  m_axi_wstrb;
    
    ralph_gpu_top #(.NUM_SM(1)) u_gpu (
        .clk(clk), .rst_n(rst_n),
        .csr_wr_en(csr_wr_en), .csr_addr(csr_addr), .csr_wr_data(csr_wr_data),
        .csr_rd_data(csr_rd_data), .irq_kernel_done(irq_kernel_done),
        .imem_req(imem_req), .imem_addr(imem_addr), .imem_data(imem_data), .imem_valid(imem_valid),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(1'b1),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(1'b1),
        .m_axi_bid(4'b0), .m_axi_bresp(2'b0), .m_axi_bvalid(1'b0), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(1'b1),
        .m_axi_rid(4'b0), .m_axi_rdata(32'b0), .m_axi_rresp(2'b0), .m_axi_rlast(1'b0),
        .m_axi_rvalid(1'b0), .m_axi_rready(m_axi_rready)
    );
    
    // Instruction memory - NOP + EXIT
    reg [31:0] imem [0:15];
    reg imem_pending;
    reg [31:0] imem_pending_addr;
    
    initial begin
        // Simple program: MOV r0, 42; EXIT
        // Instruction format: [31:26]=OPCODE, [25:21]=RD, [15:0]=IMM16
        // OP_MOV_IMM = 6'b110000 = 0x30, OP_EXIT = 6'b001011 = 0x0B
        imem[0] = {6'b110000, 5'd0, 5'd0, 16'd42};  // MOV_IMM r0, #42 (opcode=0x30, rd=0, imm=42)
        imem[1] = 32'h00000000;  // NOP padding
        imem[2] = {6'b001011, 26'd0};  // EXIT (opcode=0x0B)
        imem[3] = 32'h00000000;
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 0;
            imem_data <= 0;
            imem_pending <= 0;
        end else begin
            if (imem_req) begin
                imem_pending <= 1;
                imem_pending_addr <= imem_addr;
                imem_valid <= 0;
                $display("[IMEM] Req at addr=0x%08h", imem_addr);
            end else if (imem_pending) begin
                imem_data <= {imem[(imem_pending_addr >> 2) + 1], imem[imem_pending_addr >> 2]};
                imem_valid <= 1;
                imem_pending <= 0;
                $display("[IMEM] Resp: data=0x%016h", {imem[(imem_pending_addr >> 2) + 1], imem[imem_pending_addr >> 2]});
            end else begin
                imem_valid <= 0;
            end
        end
    end
    
    integer cycle_count;
    initial begin
        $display("=== Debug Scheduler Test ===");
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        cycle_count = 0;
        
        #100;
        rst_n = 1;
        #50;
        
        // Configure kernel: 1 block, 32 threads
        @(posedge clk);
        csr_addr = 12'h00C; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // grid_dim_x
        csr_addr = 12'h010; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // grid_dim_y
        csr_addr = 12'h014; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // grid_dim_z
        csr_addr = 12'h018; csr_wr_data = 32; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0; // block_dim_x (1 warp)
        csr_addr = 12'h01C; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // block_dim_y
        csr_addr = 12'h020; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // block_dim_z
        csr_addr = 12'h008; csr_wr_data = 0; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // kernel_pc = 0
        csr_addr = 12'h004; csr_wr_data = 1; csr_wr_en = 1; @(posedge clk); csr_wr_en = 0;  // launch
        
        $display("[TB] Kernel launched, waiting for completion...");
        
        while (!irq_kernel_done && cycle_count < 500) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
            if (cycle_count % 50 == 0)
                $display("[TB] Cycle %0d: irq_done=%b", cycle_count, irq_kernel_done);
        end
        
        if (irq_kernel_done)
            $display("[TB] PASS: Kernel completed in %0d cycles", cycle_count);
        else
            $display("[TB] FAIL: Timeout after %0d cycles", cycle_count);
        
        #100;
        $finish;
    end
endmodule
