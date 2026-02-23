//============================================================================
// RalphGPU - Softmax Pipeline Bubble Analysis Testbench
// Runs llm_softmax.ptx e2e, captures per-cycle stall/FU breakdown
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"
`include "../rtl/memory_config.vh"

module tb_softmax_pipeline_bubble;

    localparam NUM_WARPS  = `WARPS_PER_SM;
    localparam NUM_LANES  = `THREADS_PER_WARP;
    localparam DATA_WIDTH = `DATA_WIDTH;
    localparam CLK_PERIOD = 10;
    localparam IMEM_WORDS = 1024;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    reg         kernel_start;
    reg  [31:0] kernel_pc;
    reg  [31:0] block_id_x, block_id_y, block_id_z;
    reg  [31:0] block_dim_x, block_dim_y, block_dim_z;
    reg  [31:0] grid_dim_x, grid_dim_y, grid_dim_z;
    wire        kernel_done;

    wire        imem_req;
    wire [31:0] imem_addr;
    wire        imem_ready;
    reg  [63:0] imem_data;
    reg         imem_valid;

    wire        l1d_req_valid;
    wire        l1d_req_write;
    wire [NUM_LANES*32-1:0] l1d_req_addr;
    wire [NUM_LANES*32-1:0] l1d_req_wdata;
    wire [NUM_LANES-1:0] l1d_req_mask;
    reg  [NUM_LANES*32-1:0] l1d_resp_rdata;
    reg         l1d_resp_valid;
    reg         l1d_resp_hit;

    wire [3:0]  m_axi_awid, m_axi_arid;
    wire [31:0] m_axi_awaddr, m_axi_araddr;
    wire [7:0]  m_axi_awlen, m_axi_arlen;
    wire [2:0]  m_axi_awsize, m_axi_arsize;
    wire [1:0]  m_axi_awburst, m_axi_arburst;
    wire        m_axi_awvalid, m_axi_arvalid;
    reg         m_axi_awready, m_axi_arready;
    wire [31:0] m_axi_wdata;
    wire [3:0]  m_axi_wstrb;
    wire        m_axi_wlast, m_axi_wvalid;
    reg         m_axi_wready;
    reg  [3:0]  m_axi_bid, m_axi_rid;
    reg  [1:0]  m_axi_bresp, m_axi_rresp;
    reg         m_axi_bvalid, m_axi_rvalid;
    wire        m_axi_bready, m_axi_rready;
    reg  [31:0] m_axi_rdata;
    reg         m_axi_rlast;

    //------------------------------------------------------------------------
    // Instruction Memory
    //------------------------------------------------------------------------
    reg [31:0] imem_mem [0:IMEM_WORDS-1];
    reg        imem_req_q;
    reg [31:0] imem_addr_q;
    string     imem_file;
    integer    imem_fd;

    integer i;
    initial begin
        for (i = 0; i < IMEM_WORDS; i = i + 1) imem_mem[i] = {`OP_NOP, 26'b0};
        if (!$value$plusargs("imem=%s", imem_file)) imem_file = "llm_softmax.hex";
        imem_fd = $fopen(imem_file, "r");
        if (imem_fd != 0) begin
            $fclose(imem_fd);
            $readmemh(imem_file, imem_mem);
            $display("INFO: loaded %s", imem_file);
        end else begin
            $display("FATAL: cannot load %s", imem_file);
            $finish;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imem_valid <= 0; imem_req_q <= 0; imem_addr_q <= 0; imem_data <= 0;
        end else begin
            imem_valid <= imem_req_q;
            if (imem_req_q) imem_data <= {imem_mem[imem_addr_q[14:2]+1], imem_mem[imem_addr_q[14:2]]};
            imem_req_q <= imem_req;
            if (imem_req) imem_addr_q <= imem_addr;
        end
    end

    //------------------------------------------------------------------------
    // Simple Memory Model (1-cycle L1D hit for stores and loads)
    //------------------------------------------------------------------------
    reg [31:0] gmem [0:16383];
    integer    mem_i;

    initial begin
        for (mem_i = 0; mem_i < 16384; mem_i = mem_i + 1) gmem[mem_i] = 0;
    end

    always @(posedge clk) begin
        l1d_resp_valid <= 0;
        l1d_resp_hit   <= 0;

        if (l1d_req_valid && l1d_req_write) begin
            gmem[l1d_req_addr[15:2]] = l1d_req_wdata[31:0];
            l1d_resp_valid <= 1;
            l1d_resp_hit   <= 1;
        end else if (l1d_req_valid && !l1d_req_write) begin
            for (mem_i = 0; mem_i < NUM_LANES; mem_i = mem_i + 1)
                l1d_resp_rdata[mem_i*32 +: 32] <= gmem[l1d_req_addr[15:2]];
            l1d_resp_valid <= 1;
            l1d_resp_hit   <= 1;
        end
    end

    //------------------------------------------------------------------------
    // DUT
    //------------------------------------------------------------------------
    streaming_multiprocessor_v2 #(
        .SM_ID(0),
        .NUM_WARPS(NUM_WARPS),
        .NUM_LANES(NUM_LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .ICACHE_BYPASS(1), .INIT_WARPS(1)  // Single warp: exposes all pipeline bubbles
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .kernel_start(kernel_start), .kernel_pc(kernel_pc),
        .block_id_x(block_id_x), .block_id_y(block_id_y), .block_id_z(block_id_z),
        .block_dim_x(block_dim_x), .block_dim_y(block_dim_y), .block_dim_z(block_dim_z),
        .grid_dim_x(grid_dim_x), .grid_dim_y(grid_dim_y), .grid_dim_z(grid_dim_z),
        .kernel_done(kernel_done),
        .imem_req(imem_req), .imem_addr(imem_addr), .imem_ready(imem_ready),
        .imem_data(imem_data), .imem_valid(imem_valid),
        .l1d_req_valid(l1d_req_valid), .l1d_req_write(l1d_req_write),
        .l1d_req_addr(l1d_req_addr), .l1d_req_wdata(l1d_req_wdata),
        .l1d_req_mask(l1d_req_mask),
        .l1d_resp_rdata(l1d_resp_rdata), .l1d_resp_valid(l1d_resp_valid),
        .l1d_resp_hit(l1d_resp_hit),
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

    assign imem_ready = 1'b1;

    //------------------------------------------------------------------------
    // AXI Defaults
    //------------------------------------------------------------------------
    initial begin
        m_axi_awready = 1; m_axi_wready = 1;
        m_axi_bvalid = 0; m_axi_bresp = 0; m_axi_bid = 0;
        m_axi_arready = 1;
        m_axi_rvalid = 0; m_axi_rresp = 0; m_axi_rid = 0; m_axi_rdata = 0; m_axi_rlast = 1;
    end

    //------------------------------------------------------------------------
    // Pipeline Bubble Counters
    //------------------------------------------------------------------------
    integer cyc, cnt_issue, cnt_dual;
    integer cnt_stall_scoreboard, cnt_stall_mem, cnt_stall_ifetch, cnt_stall_sync;
    integer cnt_fu_alu, cnt_fu_fpu, cnt_fu_sfu, cnt_fu_ldst, cnt_fu_tensor;
    integer cnt_wb;
    integer cnt_branch_taken, cnt_branch_div;
    integer cnt_active_no_issue;

    wire wb_fire = dut.wb_valid && (dut.wb_rd != 0);
    wire any_stall = dut.perf_stall_scoreboard | dut.perf_stall_mem |
                     dut.perf_stall_ifetch | dut.perf_stall_sync;
    reg kernel_running;
    reg kernel_ever_active;  // True once warp_valid has been non-zero

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kernel_running <= 0;
            kernel_ever_active <= 0;
            cyc <= 0;
            cnt_issue <= 0; cnt_dual <= 0; cnt_wb <= 0;
            cnt_stall_scoreboard <= 0; cnt_stall_mem <= 0;
            cnt_stall_ifetch <= 0; cnt_stall_sync <= 0;
            cnt_fu_alu <= 0; cnt_fu_fpu <= 0; cnt_fu_sfu <= 0;
            cnt_fu_ldst <= 0; cnt_fu_tensor <= 0;
            cnt_branch_taken <= 0; cnt_branch_div <= 0;
            cnt_active_no_issue <= 0;
        end else begin
            if (kernel_start) begin
                kernel_running <= 1;
                kernel_ever_active <= 0;
            end
            if (kernel_running && |dut.warp_valid)
                kernel_ever_active <= 1;
            // Only stop when warps have been active then kernel_done fires
            if (kernel_done && kernel_ever_active) kernel_running <= 0;

            if (kernel_running) begin
                cyc <= cyc + 1;

                if (dut.perf_issue_valid) cnt_issue <= cnt_issue + 1;
                if (dut.perf_dual_issue)  cnt_dual  <= cnt_dual + 1;
                if (wb_fire)              cnt_wb    <= cnt_wb + 1;

                if (dut.perf_stall_scoreboard) cnt_stall_scoreboard <= cnt_stall_scoreboard + 1;
                if (dut.perf_stall_mem)        cnt_stall_mem        <= cnt_stall_mem + 1;
                if (dut.perf_stall_ifetch)     cnt_stall_ifetch     <= cnt_stall_ifetch + 1;
                if (dut.perf_stall_sync)       cnt_stall_sync       <= cnt_stall_sync + 1;

                if (dut.perf_fu_alu_active)    cnt_fu_alu    <= cnt_fu_alu + 1;
                if (dut.perf_fu_fpu_active)    cnt_fu_fpu    <= cnt_fu_fpu + 1;
                if (dut.perf_fu_sfu_active)    cnt_fu_sfu    <= cnt_fu_sfu + 1;
                if (dut.perf_fu_ldst_active)   cnt_fu_ldst   <= cnt_fu_ldst + 1;
                if (dut.perf_fu_tensor_active) cnt_fu_tensor <= cnt_fu_tensor + 1;

                if (dut.perf_branch_taken)     cnt_branch_taken <= cnt_branch_taken + 1;
                if (dut.perf_branch_divergent) cnt_branch_div   <= cnt_branch_div + 1;

                if (!dut.perf_issue_valid && !any_stall)
                    cnt_active_no_issue <= cnt_active_no_issue + 1;

                // Per-cycle trace
                $display("CYC %04d | issue=%b dual=%b | stall: sb=%b mem=%b if=%b sync=%b | fu: alu=%b fpu=%b sfu=%b ldst=%b",
                    cyc,
                    dut.perf_issue_valid, dut.perf_dual_issue,
                    dut.perf_stall_scoreboard, dut.perf_stall_mem,
                    dut.perf_stall_ifetch, dut.perf_stall_sync,
                    dut.perf_fu_alu_active, dut.perf_fu_fpu_active,
                    dut.perf_fu_sfu_active, dut.perf_fu_ldst_active);
                // Debug: show scheduler state for first 30 cycles
                if (cyc < 30)
                    $display("  DBG warp_valid=%b warp_ready=%b inst_buf_v=%b stall_mem=%b stall_fu=%b stall_branch=%b exit_pend=%b warp_active=%b fetch_pc=%h",
                        dut.warp_valid[0], dut.warp_ready[0],
                        dut.warp_inst_buf_valid[0],
                        dut.warp_stalled_mem[0], dut.warp_stalled_fu[0],
                        dut.warp_stalled_branch[0], dut.warp_exit_pending[0],
                        dut.warp_active[0], dut.warp_pc[0]);
                if (cyc < 30)
                    $display("  DEC dec0_v=%b dec_valid=%b dec_stall=%b issue0_fire=%b sched_v=%b lane0_rdy=%b inst=%h",
                        dut.dec0_valid, dut.dec_valid,
                        dut.decode_stalled, dut.issue0_fire,
                        dut.sched_issue_valid_mask[0],
                        dut.lane0_ready,
                        dut.dec0_instruction);
                if (cyc < 30)
                if (cyc < 100)
                    $display("  SCHED hazard=%b ivf=%b wb_v=%b wb_w=%0d wb_r=%0d sb0=%b",
                        dut.u_scheduler.warp_has_hazard[0],
                        dut.warp_inst_valid_fast[0],
                        dut.wb_valid, dut.wb_warp_id, dut.wb_rd,
                        dut.u_scheduler.scoreboard[0]);
                    $display("  PREDEC comp=%b mem=%b branch=%b tensor=%b sfu=%b alu=%b mul=%b fp32=%b fp16=%b eligible=%b wbuf_inst=%h",
                        dut.pd_is_compute[0], dut.pd_is_memory[0],
                        dut.pd_is_branch[0], dut.pd_is_tensor[0],
                        dut.pd_is_sfu[0], dut.pd_is_alu[0],
                        dut.pd_is_mul[0], dut.pd_is_fp32[0], dut.pd_is_fp16[0],
                        dut.u_scheduler.warp_eligible[0],
                        dut.warp_inst_buf[0]);
            end
        end
    end

    //------------------------------------------------------------------------
    // Store Monitor
    //------------------------------------------------------------------------
    reg [31:0] store_log [0:15];
    integer    store_idx;
    initial store_idx = 0;

    always @(posedge clk) begin
        if (l1d_req_valid && l1d_req_write) begin
            store_log[store_idx] = l1d_req_wdata[31:0];
            $display("STORE [0x%08h] = 0x%08h", l1d_req_addr[31:0], l1d_req_wdata[31:0]);
            store_idx = store_idx + 1;
        end
    end

    //------------------------------------------------------------------------
    // Test Sequence
    //------------------------------------------------------------------------
    real ipc;
    real bubble_pct;
    integer total_bubble;

    initial begin
        $display("============================================================");
        $display("RalphGPU Softmax Pipeline Bubble Analysis");
        $display("Kernel: llm_softmax (2-element, single warp)");
        $display("============================================================");

        rst_n = 0; kernel_start = 0; kernel_pc = 0;
        block_id_x = 0; block_id_y = 0; block_id_z = 0;
        block_dim_x = 1; block_dim_y = 1; block_dim_z = 1;
        grid_dim_x = 1; grid_dim_y = 1; grid_dim_z = 1;

        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        @(posedge clk);
        kernel_start = 1;
        kernel_pc = 32'h0;
        @(posedge clk);
        kernel_start = 0;

        // Wait for kernel to actually finish (warps active then done)
        fork
            begin
                wait(kernel_ever_active);
                wait(kernel_done);
            end
            begin
                repeat(2000) @(posedge clk);
                $display("TIMEOUT: kernel did not finish in 2000 cycles");
            end
        join_any
        disable fork;

        // Let pipeline drain
        repeat(10) @(posedge clk);

        // Report
        ipc = (cyc > 0) ? (1.0 * cnt_issue / cyc) : 0.0;
        total_bubble = cnt_stall_scoreboard + cnt_stall_mem + cnt_stall_ifetch
                     + cnt_stall_sync + cnt_active_no_issue;
        bubble_pct = (cyc > 0) ? (100.0 * total_bubble / cyc) : 0.0;

        $display("");
        $display("============================================================");
        $display("  SOFTMAX PIPELINE ANALYSIS REPORT");
        $display("============================================================");
        $display("Total cycles:          %0d", cyc);
        $display("Instructions issued:   %0d", cnt_issue);
        $display("Dual-issued:           %0d", cnt_dual);
        $display("Writebacks:            %0d", cnt_wb);
        $display("IPC:                   %0.3f", ipc);
        $display("");
        $display("--- Stall Breakdown (pipeline bubbles) ---");
        $display("  Scoreboard (RAW/WAW):  %0d cyc", cnt_stall_scoreboard);
        $display("  Memory stall:          %0d cyc", cnt_stall_mem);
        $display("  I-fetch stall:         %0d cyc", cnt_stall_ifetch);
        $display("  Sync/barrier stall:    %0d cyc", cnt_stall_sync);
        $display("  Frontend bubble:       %0d cyc", cnt_active_no_issue);
        $display("  TOTAL BUBBLE:          %0d / %0d cyc (%0.1f%%)", total_bubble, cyc, bubble_pct);
        $display("");
        $display("--- FU Utilization (cycles active) ---");
        $display("  ALU:    %0d", cnt_fu_alu);
        $display("  FPU32:  %0d", cnt_fu_fpu);
        $display("  SFU:    %0d", cnt_fu_sfu);
        $display("  LD/ST:  %0d", cnt_fu_ldst);
        $display("  Tensor: %0d", cnt_fu_tensor);
        $display("");
        $display("--- Branch ---");
        $display("  Taken: %0d  Divergent: %0d", cnt_branch_taken, cnt_branch_div);
        $display("");

        $display("--- Correctness Check ---");
        $display("  Stored values: %0d stores captured", store_idx);
        if (store_idx >= 1)
            $display("  store[0] (softmax sum) = 0x%08h", store_log[0]);
        if (store_idx >= 2)
            $display("  store[1] (softmax x0)  = 0x%08h", store_log[1]);
        if (store_idx >= 3)
            $display("  store[2] (softmax x1)  = 0x%08h", store_log[2]);

        if (kernel_done)
            $display("PASS: softmax kernel completed");
        else
            $display("FAIL: kernel did not complete");

        $display("============================================================");
        $finish;
    end

endmodule
