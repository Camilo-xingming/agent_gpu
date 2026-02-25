//============================================================================
// RalphGPU - Command Queue Integration Test
// Verifies: Command Processor fetching from Global Memory, Parsing and Dispatching
//============================================================================

`timescale 1ns / 1ps
`include "../rtl/gpu_defines.vh"

module tb_command_queue;

    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // DUT Interface
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

    // DUT
    ralph_gpu_top #(
        .L1D_BYPASS(0)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .csr_wr_en       (csr_wr_en),
        .csr_addr        (csr_addr),
        .csr_wr_data     (csr_wr_data),
        .csr_rd_data     (csr_rd_data),
        .irq_kernel_done (irq_kernel_done),
        .imem_req        (imem_req),
        .imem_addr       (imem_addr),
        .imem_data       (imem_data),
        .imem_valid      (imem_valid),
        .m_axi_awid      (m_axi_awid),
        .m_axi_awaddr    (m_axi_awaddr),
        .m_axi_awlen     (m_axi_awlen),
        .m_axi_awsize    (m_axi_awsize),
        .m_axi_awburst   (m_axi_awburst),
        .m_axi_awvalid   (m_axi_awvalid),
        .m_axi_awready   (m_axi_awready),
        .m_axi_wdata     (m_axi_wdata),
        .m_axi_wstrb     (m_axi_wstrb),
        .m_axi_wlast     (m_axi_wlast),
        .m_axi_wvalid    (m_axi_wvalid),
        .m_axi_wready    (m_axi_wready),
        .m_axi_bid       (m_axi_bid),
        .m_axi_bresp     (m_axi_bresp),
        .m_axi_bvalid    (m_axi_bvalid),
        .m_axi_bready    (m_axi_bready),
        .m_axi_arid      (m_axi_arid),
        .m_axi_araddr    (m_axi_araddr),
        .m_axi_arlen     (m_axi_arlen),
        .m_axi_arsize    (m_axi_arsize),
        .m_axi_arburst   (m_axi_arburst),
        .m_axi_arvalid   (m_axi_arvalid),
        .m_axi_arready   (m_axi_arready),
        .m_axi_rid       (m_axi_rid),
        .m_axi_rdata     (m_axi_rdata),
        .m_axi_rresp     (m_axi_rresp),
        .m_axi_rlast     (m_axi_rlast),
        .m_axi_rvalid    (m_axi_rvalid),
        .m_axi_rready    (m_axi_rready)
    );

    // IMEM
    reg [31:0] imem [0:255];
    initial begin
        for (integer j = 0; j < 256; j = j + 1) imem[j] = 32'hFC000000;
        $readmemh("vector_add.hex", imem);
    end
    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem[imem_addr[9:2] + 1], imem[imem_addr[9:2]]};
            imem_valid <= 1'b1;
        end else imem_valid <= 1'b0;
    end

    // Data Memory
    reg [31:0] data_memory [0:16383]; // 64KB
    reg [31:0] pending_write_addr;
    reg [31:0] pending_read_addr;
    reg [7:0]  pending_read_beats;
    reg [3:0]  pending_read_id;
    reg        read_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1; m_axi_wready <= 1; m_axi_bvalid <= 0;
            m_axi_arready <= 1; m_axi_rvalid <= 0; read_active <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) pending_write_addr <= m_axi_awaddr;
            if (m_axi_wvalid && m_axi_wready) begin
                m_axi_bvalid <= 1; m_axi_bid <= m_axi_awid;
                data_memory[pending_write_addr[15:2]] <= m_axi_wdata;
            end else if (m_axi_bvalid && m_axi_bready) m_axi_bvalid <= 0;

            if (!read_active && m_axi_arvalid && m_axi_arready) begin
                read_active <= 1; m_axi_arready <= 0;
                pending_read_addr <= m_axi_araddr;
                pending_read_beats <= m_axi_arlen + 1;
                pending_read_id <= m_axi_arid;
                m_axi_rvalid <= 1; m_axi_rid <= m_axi_arid;
                m_axi_rdata <= data_memory[m_axi_araddr[15:2]];
                m_axi_rlast <= (m_axi_arlen == 0);
            end else if (read_active && m_axi_rvalid && m_axi_rready) begin
                if (pending_read_beats <= 1) begin
                    m_axi_rvalid <= 0; read_active <= 0; m_axi_arready <= 1;
                end else begin
                    pending_read_addr <= pending_read_addr + 4;
                    pending_read_beats <= pending_read_beats - 1;
                    m_axi_rvalid <= 1; m_axi_rid <= pending_read_id;
                    m_axi_rdata <= data_memory[(pending_read_addr + 4) >> 2];
                    m_axi_rlast <= (pending_read_beats == 2);
                end
            end
        end
    end

    task csr_write(input [11:0] addr, input [31:0] data);
        begin @(posedge clk); csr_addr <= addr; csr_wr_data <= data; csr_wr_en <= 1; @(posedge clk); csr_wr_en <= 0; end
    endtask

    initial begin
        rst_n = 0; csr_wr_en = 0; #100; rst_n = 1; #200;

        // Initialize Vector Add Data
        for (integer i = 0; i < 32; i = i + 1) begin
            data_memory[i] = i * 10;
            data_memory[1024 + i] = i * 5;
            data_memory[2048 + i] = 0;
        end

        // Setup Command Queue Packet at 0x4000
        data_memory[16'h4000 >> 2] = 32'h0000_0000; // PC
        data_memory[16'h4004 >> 2] = 32'h0000_0001; // GRID X
        data_memory[16'h4008 >> 2] = 32'h0000_0001; // GRID Y
        data_memory[16'h400C >> 2] = 32'h0000_0001; // GRID Z
        data_memory[16'h4010 >> 2] = 32'h0000_0020; // BLOCK X
        data_memory[16'h4014 >> 2] = 32'h0000_0001; // BLOCK Y
        data_memory[16'h4018 >> 2] = 32'h0000_0001; // BLOCK Z
        data_memory[16'h401C >> 2] = 32'h0000_0000; // Flags

        // Configure CP via CSRs
        csr_write(12'h030, 32'h0000_4000); // QUEUE_BASE
        csr_write(12'h034, 32'h0000_0010); // QUEUE_SIZE = 16
        csr_write(12'h03C, 32'h0000_0001); // QUEUE_TAIL = 1 (1 command pending)
        csr_write(12'h004, 32'h0000_0002); // GPU_CONTROL: Enable Queue (bit 1)

        $display("--- Command Queue Enabled, Waiting for Kernel ---");
        
        fork
            begin
                wait(irq_kernel_done);
                $display("--- Kernel Finished via Command Queue ---");
            end
            begin
                #200000;
                $display("--- Timeout waiting for Queue Kernel ---");
                $finish;
            end
        join

        // Verify result
$display("C[31] = %0d (expected %0d)", data_memory[2048+31], 31*15); if (data_memory[2048+31] == 31*15) $display("*** COMMAND QUEUE TEST PASSED ***");
        else $display("*** COMMAND QUEUE TEST FAILED: C[31]=%0d ***", data_memory[2048+31]);

        $finish;
    end

    initial begin
        $dumpfile("tb_command_queue.vcd");
        $dumpvars(0, tb_command_queue);
    end

endmodule
