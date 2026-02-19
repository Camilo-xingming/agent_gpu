//============================================================================
// RalphGPU — RALPH-10 Performance Profiling Benchmark v3
//
// Correct instruction encoding matching ptx_assembler.py output format:
//   Register: {opcode[31:26], rd[25:21], ra[20:16], rb[15:11], func[10:5], rc[4:0]}
//   MOV_IMM:  {OP_MOV_IMM[31:26], rd[25:21], 5'd0[20:16], imm16[15:0]}
//   LD:       {OP_LD[31:26], rd[25:21], ra_base[20:16], rb=off_hi[15:11], func=off_lo[10:5], rc[4:0]}
//   ST:       {OP_ST[31:26], 5'd0[25:21], ra_base[20:16], rb_data[15:11], func=off_lo[10:5], rc=off_hi[4:0]}
//============================================================================
`timescale 1ns / 1ps

module tb_perf_benchmark;

    `include "../rtl/gpu_defines.vh"

    parameter CLK_PERIOD = 10;
    parameter TIMEOUT = 5000;

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

    // ---- Instruction + Data Memory ----
    reg [31:0] imem_storage [0:511];
    reg [31:0] gmem [0:4095];
    reg [31:0] pending_wr_addr;

    always @(posedge clk) begin
        if (imem_req) begin
            imem_data <= {imem_storage[imem_addr[10:2] + 1], imem_storage[imem_addr[10:2]]};
            imem_valid <= 1'b1;
        end else begin
            imem_valid <= 1'b0;
        end
    end

    // AXI write: AW and W may arrive on the same cycle from memory_interface,
    // so use m_axi_awaddr directly when both fire simultaneously.
    wire [31:0] wr_addr_effective = (m_axi_awvalid && m_axi_awready) ? m_axi_awaddr : pending_wr_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1'b1; m_axi_wready <= 1'b1; m_axi_bvalid <= 1'b0;
            pending_wr_addr <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) pending_wr_addr <= m_axi_awaddr;
            if (m_axi_wvalid && m_axi_wready) begin
                gmem[wr_addr_effective[13:2]] <= m_axi_wdata;
                m_axi_bvalid <= 1'b1; m_axi_bid <= m_axi_awid;
            end else if (m_axi_bvalid && m_axi_bready) m_axi_bvalid <= 1'b0;
        end
    end

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

    // ---- AXI Write Monitor ----
    always @(posedge clk) begin
        if (m_axi_awvalid && m_axi_awready)
            $display("[AXI_AW] addr=0x%08h", m_axi_awaddr);
        if (m_axi_wvalid && m_axi_wready)
            $display("[AXI_W]  data=0x%08h -> gmem[0x%04h] (eff_addr=0x%08h)", m_axi_wdata, wr_addr_effective[13:2], wr_addr_effective);
    end

    // ---- Performance Counters (hierarchical probe) ----
    wire perf_issue   = u_dut.sm_perf_issue_valid[0];
    wire perf_dual    = u_dut.sm_perf_dual_issue[0];
    wire perf_sb      = u_dut.sm_perf_stall_scoreboard[0];
    wire perf_mem     = u_dut.sm_perf_stall_mem[0];
    wire perf_ifetch  = u_dut.sm_perf_stall_ifetch[0];
    wire perf_alu_act = u_dut.sm_perf_fu_alu_active[0];
    wire perf_fpu_act = u_dut.sm_perf_fu_fpu_active[0];
    wire perf_ldst    = u_dut.sm_perf_fu_ldst_active[0];
    wire perf_tensor  = u_dut.sm_perf_fu_tensor_active[0];

    integer cnt_issue, cnt_dual, cnt_sb, cnt_mem, cnt_ifetch;
    integer cnt_alu, cnt_fpu, cnt_ldst, cnt_tensor;

    task csr_write(input [11:0] a, input [31:0] d);
        begin @(posedge clk); csr_addr <= a; csr_wr_data <= d; csr_wr_en <= 1'b1;
              @(posedge clk); csr_wr_en <= 1'b0; end
    endtask

    task init_mem;
        integer i;
        begin
            for (i = 0; i < 512; i = i+1) imem_storage[i] = {`OP_EXIT, 26'd0};
            for (i = 0; i < 4096; i = i+1) gmem[i] = 32'h0;
        end
    endtask

    task reset_perf;
        begin
            cnt_issue=0; cnt_dual=0; cnt_sb=0; cnt_mem=0; cnt_ifetch=0;
            cnt_alu=0; cnt_fpu=0; cnt_ldst=0; cnt_tensor=0;
        end
    endtask

    task run_kernel_profiled(input integer timeout, output integer cycles);
        begin
            reset_perf;
            csr_write(12'h008, 32'h0);
            csr_write(12'h00C, 32'h1);
            csr_write(12'h018, 32'h1);
            csr_write(12'h004, 32'h1);
            cycles = 0;
            while (!irq_kernel_done && cycles < timeout) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (perf_issue)   cnt_issue   = cnt_issue   + 1;
                if (perf_dual)    cnt_dual    = cnt_dual    + 1;
                if (perf_sb)      cnt_sb      = cnt_sb      + 1;
                if (perf_mem)     cnt_mem     = cnt_mem     + 1;
                if (perf_ifetch)  cnt_ifetch  = cnt_ifetch  + 1;
                if (perf_alu_act) cnt_alu     = cnt_alu     + 1;
                if (perf_fpu_act) cnt_fpu     = cnt_fpu     + 1;
                if (perf_ldst)    cnt_ldst    = cnt_ldst    + 1;
                if (perf_tensor)  cnt_tensor  = cnt_tensor  + 1;
            end
            #200;
        end
    endtask

    task print_profile(input integer cycles, input integer num_inst);
        begin
            $display("  Cycles:        %0d", cycles);
            $display("  Instructions:  %0d issued (%0d expected)", cnt_issue, num_inst);
            if (cycles > 0)
                $display("  IPC:           %0d.%02d", cnt_issue/cycles, ((cnt_issue*100)/cycles)%100);
            $display("  Dual-issue:    %0d", cnt_dual);
            $display("  --- Stall Breakdown ---");
            $display("  Scoreboard:    %0d (%0d%%)", cnt_sb, cycles>0?(cnt_sb*100)/cycles:0);
            $display("  Memory:        %0d (%0d%%)", cnt_mem, cycles>0?(cnt_mem*100)/cycles:0);
            $display("  IFetch:        %0d (%0d%%)", cnt_ifetch, cycles>0?(cnt_ifetch*100)/cycles:0);
            $display("  --- FU Utilization ---");
            $display("  ALU:           %0d (%0d%%)", cnt_alu, cycles>0?(cnt_alu*100)/cycles:0);
            $display("  FPU:           %0d (%0d%%)", cnt_fpu, cycles>0?(cnt_fpu*100)/cycles:0);
            $display("  LDST:          %0d (%0d%%)", cnt_ldst, cycles>0?(cnt_ldst*100)/cycles:0);
        end
    endtask

    // ---- Encoding helpers matching ptx_assembler.py ----
    // Bit layout: {opcode[31:26], rd[25:21], ra[20:16], rb[15:11], rc[10:6], func[5:0]}
    function [31:0] enc_reg;
        input [5:0] op; input [4:0] rd, ra, rb; input [5:0] func; input [4:0] rc;
        enc_reg = {op, rd, ra, rb, rc, func};
    endfunction

    function [31:0] enc_mov_imm;
        input [4:0] rd; input [15:0] imm;
        enc_mov_imm = {`OP_MOV_IMM, rd, 5'd0, imm};
    endfunction

    // LD_GLOBAL: rd = mem[ra + offset11]
    // Assembler: rb=off[10:6], func=off[5:0], rc=0
    function [31:0] enc_ld;
        input [4:0] rd, ra; input [10:0] offset;
        enc_ld = {`OP_LD_GLOBAL, rd, ra, offset[10:6], 5'd0, offset[5:0]};
    endfunction

    // ST_GLOBAL: mem[ra + offset11] = rb(data)
    // Assembler: rb=data_reg, rc=off[10:6], func=off[5:0]
    function [31:0] enc_st;
        input [4:0] data_rb, addr_ra; input [10:0] offset;
        enc_st = {`OP_ST_GLOBAL, 5'd0, addr_ra, data_rb, offset[10:6], offset[5:0]};
    endfunction

    function [31:0] enc_exit;
        enc_exit = {`OP_EXIT, 26'd0};
    endfunction

    integer cycle_count;

    initial begin
        rst_n = 0; csr_wr_en = 0; csr_addr = 0; csr_wr_data = 0;
        m_axi_bresp = 0; m_axi_rresp = 0;

        $display("============================================================");
        $display("  RALPH-10: Performance Benchmark v3 (correct encoding)");
        $display("  Config: 1 SM, %0d warps, %0d lanes, issue_width=%0d",
                 `WARPS_PER_SM, `THREADS_PER_WARP, `SM_ISSUE_WIDTH);
        $display("============================================================");

        #100; rst_n = 1; #50;

        // ==================================================================
        // Bench 1: ALU Throughput — 20 independent MOV_IMM
        // ==================================================================
        $display("\n[BENCH 1] ALU Throughput — 20 independent MOV_IMM");
        init_mem;
        imem_storage[0]  = enc_mov_imm(5'd1,  16'd1);
        imem_storage[1]  = enc_mov_imm(5'd2,  16'd2);
        imem_storage[2]  = enc_mov_imm(5'd3,  16'd3);
        imem_storage[3]  = enc_mov_imm(5'd4,  16'd4);
        imem_storage[4]  = enc_mov_imm(5'd5,  16'd5);
        imem_storage[5]  = enc_mov_imm(5'd6,  16'd6);
        imem_storage[6]  = enc_mov_imm(5'd7,  16'd7);
        imem_storage[7]  = enc_mov_imm(5'd8,  16'd8);
        imem_storage[8]  = enc_mov_imm(5'd9,  16'd9);
        imem_storage[9]  = enc_mov_imm(5'd10, 16'd10);
        imem_storage[10] = enc_mov_imm(5'd11, 16'd11);
        imem_storage[11] = enc_mov_imm(5'd12, 16'd12);
        imem_storage[12] = enc_mov_imm(5'd13, 16'd13);
        imem_storage[13] = enc_mov_imm(5'd14, 16'd14);
        imem_storage[14] = enc_mov_imm(5'd15, 16'd15);
        imem_storage[15] = enc_mov_imm(5'd16, 16'd16);
        imem_storage[16] = enc_mov_imm(5'd17, 16'd17);
        imem_storage[17] = enc_mov_imm(5'd18, 16'd18);
        imem_storage[18] = enc_mov_imm(5'd19, 16'd19);
        imem_storage[19] = enc_mov_imm(5'd20, 16'd20);
        imem_storage[20] = enc_mov_imm(5'd30, 16'h1000);
        imem_storage[21] = enc_st(5'd20, 5'd30, 11'd0);
        imem_storage[22] = enc_exit();

        run_kernel_profiled(TIMEOUT, cycle_count);
        if (cycle_count >= TIMEOUT) $display("  *** TIMEOUT ***");
        else begin
            print_profile(cycle_count, 23);
            $display("  Result: gmem[0x1000]=%0d (expected 20)  gmem[0]=%0d", gmem[32'h1000 >> 2], gmem[0]);
        end

        rst_n = 0; #50; rst_n = 1; #50;

        // ==================================================================
        // Bench 2: ALU RAW Chain — r1=1, r2=r1+r1, r3=r2+r2, ...
        // ==================================================================
        $display("\n[BENCH 2] ALU RAW Chain — 20 dependent ADDs");
        init_mem;
        imem_storage[0]  = enc_mov_imm(5'd1, 16'd1);
        imem_storage[1]  = enc_reg(`OP_ALU, 5'd2,  5'd1,  5'd1,  `FUNC_ADD, 5'd0);
        imem_storage[2]  = enc_reg(`OP_ALU, 5'd3,  5'd2,  5'd2,  `FUNC_ADD, 5'd0);
        imem_storage[3]  = enc_reg(`OP_ALU, 5'd4,  5'd3,  5'd3,  `FUNC_ADD, 5'd0);
        imem_storage[4]  = enc_reg(`OP_ALU, 5'd5,  5'd4,  5'd4,  `FUNC_ADD, 5'd0);
        imem_storage[5]  = enc_reg(`OP_ALU, 5'd6,  5'd5,  5'd5,  `FUNC_ADD, 5'd0);
        imem_storage[6]  = enc_reg(`OP_ALU, 5'd7,  5'd6,  5'd6,  `FUNC_ADD, 5'd0);
        imem_storage[7]  = enc_reg(`OP_ALU, 5'd8,  5'd7,  5'd7,  `FUNC_ADD, 5'd0);
        imem_storage[8]  = enc_reg(`OP_ALU, 5'd9,  5'd8,  5'd8,  `FUNC_ADD, 5'd0);
        imem_storage[9]  = enc_reg(`OP_ALU, 5'd10, 5'd9,  5'd9,  `FUNC_ADD, 5'd0);
        imem_storage[10] = enc_reg(`OP_ALU, 5'd11, 5'd10, 5'd10, `FUNC_ADD, 5'd0);
        imem_storage[11] = enc_reg(`OP_ALU, 5'd12, 5'd11, 5'd11, `FUNC_ADD, 5'd0);
        imem_storage[12] = enc_reg(`OP_ALU, 5'd13, 5'd12, 5'd12, `FUNC_ADD, 5'd0);
        imem_storage[13] = enc_reg(`OP_ALU, 5'd14, 5'd13, 5'd13, `FUNC_ADD, 5'd0);
        imem_storage[14] = enc_reg(`OP_ALU, 5'd15, 5'd14, 5'd14, `FUNC_ADD, 5'd0);
        imem_storage[15] = enc_reg(`OP_ALU, 5'd16, 5'd15, 5'd15, `FUNC_ADD, 5'd0);
        imem_storage[16] = enc_reg(`OP_ALU, 5'd17, 5'd16, 5'd16, `FUNC_ADD, 5'd0);
        imem_storage[17] = enc_reg(`OP_ALU, 5'd18, 5'd17, 5'd17, `FUNC_ADD, 5'd0);
        imem_storage[18] = enc_reg(`OP_ALU, 5'd19, 5'd18, 5'd18, `FUNC_ADD, 5'd0);
        imem_storage[19] = enc_reg(`OP_ALU, 5'd20, 5'd19, 5'd19, `FUNC_ADD, 5'd0);
        imem_storage[20] = enc_mov_imm(5'd30, 16'h1000);
        imem_storage[21] = enc_st(5'd20, 5'd30, 11'd0);
        imem_storage[22] = enc_exit();

        run_kernel_profiled(TIMEOUT, cycle_count);
        if (cycle_count >= TIMEOUT) $display("  *** TIMEOUT ***");
        else begin
            print_profile(cycle_count, 23);
            $display("  Result: gmem[0x1000]=%0d (expected 524288)", gmem[32'h1000 >> 2]);
        end

        rst_n = 0; #50; rst_n = 1; #50;

        // ==================================================================
        // Bench 3: FP32 Throughput — 10 independent add.f32
        // ==================================================================
        $display("\n[BENCH 3] FP32 Throughput — 10 independent FP32 ADDs");
        init_mem;
        imem_storage[0]  = enc_mov_imm(5'd31, 16'd16);
        imem_storage[1]  = enc_mov_imm(5'd1, 16'h3F80);
        imem_storage[2]  = enc_reg(`OP_ALU, 5'd1, 5'd1, 5'd31, `FUNC_SHL, 5'd0);
        imem_storage[3]  = enc_mov_imm(5'd2, 16'h4000);
        imem_storage[4]  = enc_reg(`OP_ALU, 5'd2, 5'd2, 5'd31, `FUNC_SHL, 5'd0);
        imem_storage[5]  = enc_reg(`OP_FP32_ARITH, 5'd3,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[6]  = enc_reg(`OP_FP32_ARITH, 5'd4,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[7]  = enc_reg(`OP_FP32_ARITH, 5'd5,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[8]  = enc_reg(`OP_FP32_ARITH, 5'd6,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[9]  = enc_reg(`OP_FP32_ARITH, 5'd7,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[10] = enc_reg(`OP_FP32_ARITH, 5'd8,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[11] = enc_reg(`OP_FP32_ARITH, 5'd9,  5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[12] = enc_reg(`OP_FP32_ARITH, 5'd10, 5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[13] = enc_reg(`OP_FP32_ARITH, 5'd11, 5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[14] = enc_reg(`OP_FP32_ARITH, 5'd12, 5'd1, 5'd2, 6'd0, 5'd0);
        imem_storage[15] = enc_mov_imm(5'd30, 16'h1000);
        imem_storage[16] = enc_st(5'd3, 5'd30, 11'd0);
        imem_storage[17] = enc_exit();

        run_kernel_profiled(TIMEOUT, cycle_count);
        if (cycle_count >= TIMEOUT) $display("  *** TIMEOUT ***");
        else begin
            print_profile(cycle_count, 18);
            $display("  Result: gmem[0x1000]=0x%08x (expected 0x40400000=3.0f)", gmem[32'h1000>>2]);
        end

        rst_n = 0; #50; rst_n = 1; #50;

        // ==================================================================
        // Bench 4: Memory Latency — 5 global loads + sum
        // ==================================================================
        $display("\n[BENCH 4] Memory Latency — 5 loads + sum");
        init_mem;
        gmem[32'h0000 >> 2] = 32'd10;
        gmem[32'h0004 >> 2] = 32'd20;
        gmem[32'h0008 >> 2] = 32'd30;
        gmem[32'h000C >> 2] = 32'd40;
        gmem[32'h0010 >> 2] = 32'd50;

        imem_storage[0]  = enc_mov_imm(5'd29, 16'h0000);
        imem_storage[1]  = enc_ld(5'd1, 5'd29, 11'd0);
        imem_storage[2]  = enc_ld(5'd2, 5'd29, 11'd4);
        imem_storage[3]  = enc_ld(5'd3, 5'd29, 11'd8);
        imem_storage[4]  = enc_ld(5'd4, 5'd29, 11'd12);
        imem_storage[5]  = enc_ld(5'd5, 5'd29, 11'd16);
        imem_storage[6]  = enc_reg(`OP_ALU, 5'd10, 5'd1, 5'd2, `FUNC_ADD, 5'd0);
        imem_storage[7]  = enc_reg(`OP_ALU, 5'd11, 5'd3, 5'd4, `FUNC_ADD, 5'd0);
        imem_storage[8]  = enc_reg(`OP_ALU, 5'd12, 5'd10, 5'd5, `FUNC_ADD, 5'd0);
        imem_storage[9]  = enc_reg(`OP_ALU, 5'd13, 5'd12, 5'd11, `FUNC_ADD, 5'd0);
        imem_storage[10] = enc_mov_imm(5'd30, 16'h1000);
        imem_storage[11] = enc_st(5'd13, 5'd30, 11'd0);
        imem_storage[12] = enc_exit();

        run_kernel_profiled(TIMEOUT, cycle_count);
        if (cycle_count >= TIMEOUT) $display("  *** TIMEOUT ***");
        else begin
            print_profile(cycle_count, 13);
            $display("  Result: gmem[0x1000]=%0d (expected 150)", gmem[32'h1000 >> 2]);
        end

        rst_n = 0; #50; rst_n = 1; #50;

        // ==================================================================
        // Bench 5: Mixed ALU+MEM — interleaved LD/compute
        // ==================================================================
        $display("\n[BENCH 5] Mixed ALU+MEM — interleaved LD/compute");
        init_mem;
        gmem[32'h0000 >> 2] = 32'd5;
        gmem[32'h0004 >> 2] = 32'd10;

        imem_storage[0]  = enc_mov_imm(5'd29, 16'h0000);
        imem_storage[1]  = enc_ld(5'd1, 5'd29, 11'd0);
        imem_storage[2]  = enc_mov_imm(5'd10, 16'd100);
        imem_storage[3]  = enc_ld(5'd2, 5'd29, 11'd4);
        imem_storage[4]  = enc_mov_imm(5'd11, 16'd200);
        imem_storage[5]  = enc_reg(`OP_ALU, 5'd20, 5'd1,  5'd10, `FUNC_ADD, 5'd0);
        imem_storage[6]  = enc_reg(`OP_ALU, 5'd21, 5'd2,  5'd11, `FUNC_ADD, 5'd0);
        imem_storage[7]  = enc_reg(`OP_ALU, 5'd22, 5'd20, 5'd21, `FUNC_ADD, 5'd0);
        imem_storage[8]  = enc_mov_imm(5'd30, 16'h1000);
        imem_storage[9]  = enc_st(5'd22, 5'd30, 11'd0);
        imem_storage[10] = enc_exit();

        run_kernel_profiled(TIMEOUT, cycle_count);
        if (cycle_count >= TIMEOUT) $display("  *** TIMEOUT ***");
        else begin
            print_profile(cycle_count, 11);
            $display("  Result: gmem[0x1000]=%0d (expected 315)", gmem[32'h1000 >> 2]);
        end

        $display("\n============================================================");
        $display("  RALPH-10 Benchmark v3 Complete");
        $display("============================================================");
        $finish;
    end

endmodule
