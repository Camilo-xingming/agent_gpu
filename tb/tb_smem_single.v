//============================================================================
// Single Shared Memory Test
//============================================================================
`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_smem_single;
    // Clock and reset
    reg clk, rst_n;
    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin rst_n = 0; #100 rst_n = 1; end

    // Memory
    reg [31:0] instruction_mem [0:1023];
    reg [31:0] global_mem [0:16383];
    
    // DUT interface
    reg csr_wr_en;
    reg [11:0] csr_addr;
    reg [31:0] csr_wr_data;
    wire [31:0] csr_rd_data;
    wire irq_kernel_done;
    
    wire imem_req;
    wire [31:0] imem_addr;
    reg [63:0] imem_data;
    reg imem_valid;
    
    // AXI simplified
    wire [31:0] m_axi_awaddr, m_axi_araddr;
    wire m_axi_awvalid, m_axi_wvalid, m_axi_arvalid;
    reg m_axi_awready, m_axi_wready, m_axi_arready;
    reg [31:0] m_axi_rdata;
    reg m_axi_rvalid, m_axi_rlast;
    wire m_axi_rready, m_axi_bready;
    reg m_axi_bvalid;
    wire [31:0] m_axi_wdata;
    wire m_axi_wlast;
    
    // Unused AXI signals
    wire [3:0] m_axi_awid, m_axi_arid;
    wire [7:0] m_axi_awlen, m_axi_arlen;
    wire [2:0] m_axi_awsize, m_axi_arsize;
    wire [1:0] m_axi_awburst, m_axi_arburst;
    wire [3:0] m_axi_wstrb;
    
    // DUT
    ralph_gpu_top dut (
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
        .m_axi_bid(4'b0), .m_axi_bresp(2'b0), .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rid(4'b0), .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(2'b0), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );
    
    // CSR write task
    task csr_write(input [11:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            csr_wr_en <= 1;
            csr_addr <= addr;
            csr_wr_data <= data;
            @(posedge clk);
            csr_wr_en <= 0;
        end
    endtask
    
    // Instruction memory model
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {instruction_mem[imem_addr[12:2] + 1], instruction_mem[imem_addr[12:2]]};
            imem_valid <= 1;
        end else begin
            imem_valid <= 0;
        end
    end
    
    // Global memory model
    localparam GMEM_BASE = 32'h80002000;
    reg [31:0] ar_addr_saved;
    reg [7:0] ar_len_saved;
    reg [7:0] burst_cnt;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1;
            m_axi_wready <= 1;
            m_axi_arready <= 1;
            m_axi_rvalid <= 0;
            m_axi_bvalid <= 0;
            burst_cnt <= 0;
        end else begin
            // Write handling
            if (m_axi_awvalid && m_axi_awready) begin
                ar_addr_saved <= m_axi_awaddr;
            end
            if (m_axi_wvalid && m_axi_wready) begin
                global_mem[(ar_addr_saved - GMEM_BASE) >> 2] <= m_axi_wdata;
                ar_addr_saved <= ar_addr_saved + 4;
                if (m_axi_wlast) m_axi_bvalid <= 1;
            end
            if (m_axi_bvalid && m_axi_bready) m_axi_bvalid <= 0;
            
            // Read handling
            if (m_axi_arvalid && m_axi_arready) begin
                ar_addr_saved <= m_axi_araddr;
                ar_len_saved <= m_axi_arlen;
                burst_cnt <= 0;
                m_axi_rvalid <= 1;
                m_axi_rdata <= global_mem[(m_axi_araddr - GMEM_BASE) >> 2];
                m_axi_rlast <= (m_axi_arlen == 0);
            end else if (m_axi_rvalid && m_axi_rready) begin
                if (m_axi_rlast) begin
                    m_axi_rvalid <= 0;
                    m_axi_rlast <= 0;
                end else begin
                    burst_cnt <= burst_cnt + 1;
                    ar_addr_saved <= ar_addr_saved + 4;
                    m_axi_rdata <= global_mem[(ar_addr_saved + 4 - GMEM_BASE) >> 2];
                    m_axi_rlast <= (burst_cnt + 1 == ar_len_saved);
                end
            end
        end
    end
    
    // Main test
    integer i, cycles;
    initial begin
        $dumpfile("tb_smem_single.vcd");
        $dumpvars(0, tb_smem_single);
        
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;
        
        // Clear memories
        for (i = 0; i < 1024; i = i + 1) instruction_mem[i] = 32'h2c000000; // NOP/EXIT
        for (i = 0; i < 16384; i = i + 1) global_mem[i] = 0;
        
        // Load test hex
        $readmemh("sim/test_08_memory_shared.hex", instruction_mem);
        
        wait(rst_n);
        #200;
        
        $display("========================================");
        $display("Shared Memory Test Start");
        $display("========================================");
        
        // Configure kernel
        csr_write(12'h008, 32'h0000_0000);  // KERNEL_PC
        csr_write(12'h00C, 32'h0000_0001);  // GRID_DIM_X
        csr_write(12'h010, 32'h0000_0001);  // GRID_DIM_Y
        csr_write(12'h014, 32'h0000_0001);  // GRID_DIM_Z
        csr_write(12'h018, 32'h0000_0080);  // BLOCK_DIM_X = 128
        csr_write(12'h01C, 32'h0000_0001);  // BLOCK_DIM_Y
        csr_write(12'h020, 32'h0000_0001);  // BLOCK_DIM_Z
        
        // Start kernel
        $display("Starting Kernel...");
        csr_write(12'h004, 32'h0000_0001);
        
        // Wait for completion
        cycles = 0;
        while (!irq_kernel_done && cycles < 200000) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        
        $display("Kernel done after %0d cycles", cycles);
        
        // Check result
        $display("Result at 0x80002000: 0x%08x (expect 0xCAFECAFE for PASS)", global_mem[0]);
        if (global_mem[0] == 32'hCAFECAFE) begin
            $display("[PASS] Shared Memory Test");
        end else begin
            $fatal(1, "[FAIL] Shared Memory Test - got 0x%08x", global_mem[0]);
        end
        
        #1000;
        $finish;
    end
    
    // Timeout
    initial begin
        #2000000;
        $display("TIMEOUT!");
        $finish;
    end
endmodule
