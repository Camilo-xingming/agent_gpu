`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_interrupt_exceptions;
    localparam CLK_PERIOD = 10;
    localparam TIMEOUT_CYCLES = 2000;

    localparam CSR_GPU_STATUS      = 12'h000;
    localparam CSR_GPU_CONTROL     = 12'h004;
    localparam CSR_KERNEL_PC       = 12'h008;
    localparam CSR_GRID_DIM_X      = 12'h00C;
    localparam CSR_GRID_DIM_Y      = 12'h010;
    localparam CSR_GRID_DIM_Z      = 12'h014;
    localparam CSR_BLOCK_DIM_X     = 12'h018;
    localparam CSR_BLOCK_DIM_Y     = 12'h01C;
    localparam CSR_BLOCK_DIM_Z     = 12'h020;
    localparam CSR_ERROR_STATUS    = 12'h024;
    localparam CSR_ERROR_WARP_MASK = 12'h028;
    localparam CSR_ERROR_INFO      = 12'h02C;

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

    reg [31:0] imem [0:255];
    reg [31:0] imem_req_addr_d;
    reg        imem_req_pending;

    function [31:0] enc_r;
        input [5:0] op;
        input [4:0] rd;
        input [4:0] ra;
        input [4:0] rb;
        input [4:0] rc;
        input [5:0] func;
        begin
            enc_r = {op, rd, ra, rb, rc, func};
        end
    endfunction

    function [31:0] enc_mov_imm;
        input [4:0] rd;
        input [15:0] imm;
        begin
            enc_mov_imm = {`OP_MOV_IMM, rd, 5'd0, imm};
        end
    endfunction

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

    always #(CLK_PERIOD/2) clk = ~clk;

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

    task csr_write;
        input [11:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            csr_addr <= addr;
            csr_wr_data <= data;
            csr_wr_en <= 1'b1;
            @(posedge clk);
            csr_wr_en <= 1'b0;
        end
    endtask

    task csr_read;
        input [11:0] addr;
        output [31:0] data;
        begin
            @(posedge clk);
            csr_addr <= addr;
            @(posedge clk);
            data = csr_rd_data;
        end
    endtask

    task run_case;
        input [31:0] start_pc;
        input [3:0] expected_code;
        input [255:0] name;
        integer cycles;
        reg [31:0] err_status;
        reg [31:0] err_info;
        reg [31:0] err_warp_mask;
        begin
            csr_write(CSR_ERROR_STATUS, 32'h1);
            csr_write(CSR_GPU_STATUS, 32'h1);
            csr_write(CSR_KERNEL_PC, start_pc);
            csr_write(CSR_GPU_CONTROL, 32'h1);

            cycles = 0;
            while (!irq_kernel_done && cycles < TIMEOUT_CYCLES) begin
                @(posedge clk);
                cycles = cycles + 1;
            end

            if (!irq_kernel_done) begin
                $fatal(1, "[FAIL] %0s timeout", name);
                $finish;
            end

            csr_read(CSR_ERROR_STATUS, err_status);
            csr_read(CSR_ERROR_INFO, err_info);
            csr_read(CSR_ERROR_WARP_MASK, err_warp_mask);

            if (!err_status[0]) begin
                $fatal(1, "[FAIL] %0s missing error pending", name);
                $finish;
            end
            if (err_status[4:1] != expected_code) begin
                $fatal(1, "[FAIL] %0s wrong code got=%0d exp=%0d status=0x%08h", name, err_status[4:1], expected_code, err_status);
                $finish;
            end
            if (!err_warp_mask[0]) begin
                $fatal(1, "[FAIL] %0s warp mask bit0 not set: 0x%08h", name, err_warp_mask);
                $finish;
            end

            $display("[PASS] %0s code=%0d status=0x%08h info=0x%08h", name, err_status[4:1], err_status, err_info);
        end
    endtask

    integer i;
    initial begin
        clk = 0;
        rst_n = 0;
        csr_wr_en = 0;
        csr_addr = 0;
        csr_wr_data = 0;

        for (i = 0; i < 256; i = i + 1) begin
            imem[i] = {`OP_NOP, 26'b0};
        end

        // Case 1: illegal instruction (OP_MISC with unsupported func)
        imem[0] = enc_r(`OP_MISC, 5'd0, 5'd0, 5'd0, 5'd0, 6'h3F);

        // Case 2: divide by zero
        imem[16] = enc_r(`OP_DIV, 5'd1, 5'd0, 5'd0, 5'd0, 6'h00);

        // Case 3: shared memory OOB (addr 0x5000 > default shared size)
        imem[32] = enc_mov_imm(5'd1, 16'h5000);
        imem[33] = enc_r(`OP_LD_SHARED, 5'd2, 5'd1, 5'd0, 5'd0, 6'h00);

        #100;
        rst_n = 1'b1;

        csr_write(CSR_GRID_DIM_X, 32'd1);
        csr_write(CSR_GRID_DIM_Y, 32'd1);
        csr_write(CSR_GRID_DIM_Z, 32'd1);
        csr_write(CSR_BLOCK_DIM_X, 32'd1);
        csr_write(CSR_BLOCK_DIM_Y, 32'd1);
        csr_write(CSR_BLOCK_DIM_Z, 32'd1);

        run_case(32'd0, 4'h1, "illegal instruction");
        run_case(32'd64, 4'h3, "divide by zero");
        run_case(32'd128, 4'h2, "address OOB");

        $display("[PASS] interrupt/exception handling smoke tests complete");
        #50;
        $finish;
    end
endmodule
