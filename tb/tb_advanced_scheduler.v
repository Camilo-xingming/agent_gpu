`timescale 1ns / 1ps

module tb_advanced_scheduler();

    parameter NUM_WARPS     = 8;
    parameter INST_WIDTH    = 32;
    parameter NUM_ISSUE     = 2;
    parameter SCOREBOARD_DEPTH = 16;
    parameter WARP_W        = $clog2(NUM_WARPS);

    reg clk;
    reg rst_n;

    // Warp Status
    reg [NUM_WARPS-1:0] warp_valid;
    reg [NUM_WARPS-1:0] warp_ready;
    reg [NUM_WARPS-1:0] warp_diverged;
    reg [NUM_WARPS-1:0] warp_at_barrier;

    // Instruction Buffer
    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0] warp_inst_valid;
    wire [NUM_WARPS-1:0] warp_inst_consume;

    // Decoded Info
    reg [NUM_WARPS*5-1:0] warp_rd;
    reg [NUM_WARPS*5-1:0] warp_rs1;
    reg [NUM_WARPS*5-1:0] warp_rs2;
    reg [NUM_WARPS*5-1:0] warp_rs3;
    reg [NUM_WARPS-1:0] warp_reads_rs3;
    reg [NUM_WARPS-1:0] warp_is_compute;
    reg [NUM_WARPS-1:0] warp_is_tensor;
    reg [NUM_WARPS-1:0] warp_is_memory;
    reg [NUM_WARPS-1:0] warp_is_branch;
    reg [NUM_WARPS-1:0] warp_writes_reg;

    // Pipe Availability
    reg compute_pipe0_ready;
    reg compute_pipe1_ready;
    reg tensor_pipe_ready;
    reg memory_pipe_ready;
    reg branch_unit_ready;

    // Issue Outputs
    wire [NUM_ISSUE-1:0] issue_valid;
    wire [NUM_ISSUE*WARP_W-1:0] issue_warp_id;
    wire [NUM_ISSUE*INST_WIDTH-1:0] issue_inst;
    wire [NUM_ISSUE*3-1:0] issue_pipe;

    // Scoreboard Writeback
    reg wb_valid;
    reg [WARP_W-1:0] wb_warp_id;
    reg [4:0] wb_rd;

    // Stats and Scoreboard
    wire [31:0] stat_cycles;
    wire [31:0] stat_single_issue;
    wire [31:0] stat_dual_issue;
    wire [31:0] stat_stalls;
    wire [NUM_WARPS*4-1:0] issue_seq_out;
    wire [NUM_WARPS*32-1:0] scoreboard_out;

    advanced_warp_scheduler #(
        .NUM_WARPS(NUM_WARPS),
        .INST_WIDTH(INST_WIDTH),
        .NUM_ISSUE(NUM_ISSUE),
        .SCOREBOARD_DEPTH(SCOREBOARD_DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .warp_valid(warp_valid),
        .warp_ready(warp_ready),
        .warp_diverged(warp_diverged),
        .warp_at_barrier(warp_at_barrier),
        .warp_inst(warp_inst),
        .warp_inst_valid(warp_inst_valid),
        .warp_inst_consume(warp_inst_consume),
        .warp_rd(warp_rd),
        .warp_rs1(warp_rs1),
        .warp_rs2(warp_rs2),
        .warp_rs3(warp_rs3),
        .warp_reads_rs3(warp_reads_rs3),
        .warp_is_compute(warp_is_compute),
        .warp_is_tensor(warp_is_tensor),
        .warp_is_memory(warp_is_memory),
        .warp_is_branch(warp_is_branch),
        .warp_writes_reg(warp_writes_reg),
        .compute_pipe0_ready(compute_pipe0_ready),
        .compute_pipe1_ready(compute_pipe1_ready),
        .tensor_pipe_ready(tensor_pipe_ready),
        .memory_pipe_ready(memory_pipe_ready),
        .branch_unit_ready(branch_unit_ready),
        .issue_valid(issue_valid),
        .issue_warp_id(issue_warp_id),
        .issue_inst(issue_inst),
        .issue_pipe(issue_pipe),
        .wb_valid(wb_valid),
        .wb_warp_id(wb_warp_id),
        .wb_rd(wb_rd),
        .stat_cycles(stat_cycles),
        .stat_single_issue(stat_single_issue),
        .stat_dual_issue(stat_dual_issue),
        .stat_stalls(stat_stalls),
        .issue_seq_out(issue_seq_out),
        .scoreboard_out(scoreboard_out)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    integer errors = 0;
    integer i;

    task reset_dut();
        begin
            rst_n = 0;
            warp_valid = 0;
            warp_ready = 0;
            warp_diverged = 0;
            warp_at_barrier = 0;
            warp_inst = 0;
            warp_inst_valid = 0;
            warp_rd = 0; warp_rs1 = 0; warp_rs2 = 0; warp_rs3 = 0;
            warp_reads_rs3 = 0;
            warp_is_compute = 0; warp_is_tensor = 0; warp_is_memory = 0; warp_is_branch = 0;
            warp_writes_reg = 0;
            compute_pipe0_ready = 1; compute_pipe1_ready = 1;
            tensor_pipe_ready = 1; memory_pipe_ready = 1; branch_unit_ready = 1;
            wb_valid = 0; wb_warp_id = 0; wb_rd = 0;
            @(negedge clk);
            rst_n = 1;
        end
    endtask

    task set_warp_inst(input integer w_id, input [31:0] inst, input [4:0] rd, input [4:0] rs1, input [4:0] rs2, input [4:0] rs3, input is_compute, input is_tensor, input is_memory, input is_branch, input writes_reg);
        begin
            warp_valid[w_id] = 1;
            warp_ready[w_id] = 1;
            warp_inst_valid[w_id] = 1;
            warp_inst[w_id*INST_WIDTH +: INST_WIDTH] = (inst & ~(32'h03FFF800)) | (rd << 21) | (rs1 << 16) | (rs2 << 11) | (rs3 << 6);
            warp_rd[w_id*5 +: 5] = rd;
            warp_rs1[w_id*5 +: 5] = rs1;
            warp_rs2[w_id*5 +: 5] = rs2;
            warp_rs3[w_id*5 +: 5] = rs3;
            if (rs3 != 0) warp_reads_rs3[w_id] = 1; else warp_reads_rs3[w_id] = 0;
            warp_is_compute[w_id] = is_compute;
            warp_is_tensor[w_id]  = is_tensor;
            warp_is_memory[w_id]  = is_memory;
            warp_is_branch[w_id]  = is_branch;
            warp_writes_reg[w_id] = writes_reg;
        end
    endtask

    task clear_warp_inst(input integer w_id);
        begin
            warp_valid[w_id] = 0;
            warp_ready[w_id] = 0;
            warp_inst_valid[w_id] = 0;
        end
    endtask

    reg [WARP_W-1:0] saved_w0;
    reg [WARP_W-1:0] saved_w1;

    initial begin
        $dumpfile("tb_advanced_scheduler.vcd");
        $dumpvars(0, tb_advanced_scheduler);

        reset_dut();
        @(negedge clk);

        $display("--- Test 1: Basic Single Issue (Compute) ---");
        set_warp_inst(0, 32'hAAAA_BBBB, 5'd1, 5'd2, 5'd3, 5'd0, 1, 0, 0, 0, 1);
        #4;
        if (issue_valid[0] !== 1'b1 || issue_warp_id[0*WARP_W+:WARP_W] !== 0 || issue_pipe[0*3+:3] !== 3'd0) begin
            $display("FAIL: Test 1 Basic Single Issue");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 1 Basic Single Issue");
        end
        @(negedge clk);
        clear_warp_inst(0);

        @(negedge clk);
        $display("--- Test 2: Dual Issue (Compute + Memory) ---");
        set_warp_inst(1, 32'h1111_1111, 5'd4, 5'd5, 5'd6, 5'd0, 1, 0, 0, 0, 1);
        set_warp_inst(2, 32'h2222_2222, 5'd7, 5'd8, 5'd9, 5'd0, 0, 0, 1, 0, 1);
        #4;
        if (issue_valid !== 2'b11) begin
            $display("FAIL: Test 2 Dual Issue (validity)");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 2 Dual Issue");
        end
        @(negedge clk);
        clear_warp_inst(1);
        clear_warp_inst(2);

        @(negedge clk);
        $display("--- Test 3: Structural Hazard ---");
        set_warp_inst(3, 32'h3333_3333, 5'd10, 5'd11, 5'd12, 5'd0, 1, 0, 0, 0, 1);
        set_warp_inst(4, 32'h4444_4444, 5'd13, 5'd14, 5'd15, 5'd0, 1, 0, 0, 0, 1);
        #4;
        if (issue_valid !== 2'b11) begin
            $display("FAIL: Test 3 Two Compute Instructions");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 3 Two Compute Instructions");
        end
        @(negedge clk);
        clear_warp_inst(3);
        clear_warp_inst(4);

        @(negedge clk);
        $display("--- Test 4: Resource Not Ready Stall ---");
        set_warp_inst(5, 32'h5555_5555, 5'd16, 5'd17, 5'd18, 5'd0, 0, 1, 0, 0, 1);
        tensor_pipe_ready = 0;
        #4;
        if (issue_valid[0] === 1'b1 && issue_pipe[0*3+:3] === 3'd2) begin
            $display("FAIL: Test 4 Resource Not Ready Stall");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 4 Resource Not Ready Stall");
        end
        @(negedge clk);
        tensor_pipe_ready = 1;
        clear_warp_inst(5);

        @(negedge clk);
        $display("--- Test 5: Scoreboard Data Hazard ---");
        set_warp_inst(6, 32'h6666_6666, 5'd20, 5'd21, 5'd22, 5'd0, 1, 0, 0, 0, 1);
        @(negedge clk);
        set_warp_inst(6, 32'h7777_7777, 5'd23, 5'd20, 5'd24, 5'd0, 1, 0, 0, 0, 1);
        #4;
        if (issue_valid[0] === 1'b1 && issue_warp_id[0*WARP_W+:WARP_W] === 6) begin
            $display("FAIL: Test 5 Scoreboard Data Hazard");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 5 Scoreboard Data Hazard");
        end
        @(negedge clk);
        wb_valid = 1; wb_warp_id = 6; wb_rd = 5'd20;
        @(negedge clk); // Scoreboard cleared at posedge during this cycle
        wb_valid = 0;
        #4;
        if (issue_valid[0] === 1'b1 && issue_warp_id[0*WARP_W+:WARP_W] === 6) begin
            $display("PASS: Test 5 Scoreboard Clear");
        end else begin
            $display("FAIL: Test 5 Scoreboard Clear");
            errors = errors + 1;
        end
        @(negedge clk);
        clear_warp_inst(6);

        #20;
        reset_dut();
        @(negedge clk);

        $display("--- Test 6: Priority Arbitration (Branch > Memory > Tensor > Compute) ---");
        set_warp_inst(0, 32'h0000_0000, 5'd1, 5'd2, 5'd3, 5'd0, 1, 0, 0, 0, 1);
        set_warp_inst(1, 32'h1111_1111, 5'd4, 5'd5, 5'd6, 5'd0, 0, 1, 0, 0, 1);
        set_warp_inst(2, 32'h2222_2222, 5'd7, 5'd8, 5'd9, 5'd0, 0, 0, 1, 0, 1);
        set_warp_inst(3, 32'h3333_3333, 5'd0, 5'd0, 5'd0, 5'd0, 0, 0, 0, 1, 0);
        #4;
        
        if (issue_valid !== 2'b11 || issue_pipe[0*3+:3] !== 3'd4 || issue_pipe[1*3+:3] !== 3'd0) begin
            $display("FAIL: Test 6 Priority Arbitration (Cycle 1)");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 6 Priority Arbitration (Cycle 1)");
        end
        @(negedge clk);
        clear_warp_inst(3);
        clear_warp_inst(0);
        #4;
        
        if (issue_valid !== 2'b11 || issue_pipe[0*3+:3] !== 3'd3 || issue_pipe[1*3+:3] !== 3'd2) begin
            $display("FAIL: Test 6 Priority Arbitration (Cycle 2)");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 6 Priority Arbitration (Cycle 2)");
        end
        @(negedge clk);
        clear_warp_inst(2);
        clear_warp_inst(1);

        #20;
        reset_dut();
        @(negedge clk);

        $display("--- Test 7: Multi-Warp Round Robin ---");
        set_warp_inst(0, 32'hAAAA_AAAA, 5'd1, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        set_warp_inst(1, 32'hBBBB_BBBB, 5'd2, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        set_warp_inst(2, 32'hCCCC_CCCC, 5'd3, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        #4;
        if (issue_valid !== 2'b11 || !((issue_warp_id[0*WARP_W+:WARP_W] == 0 && issue_warp_id[1*WARP_W+:WARP_W] == 1) || (issue_warp_id[0*WARP_W+:WARP_W] == 1 && issue_warp_id[1*WARP_W+:WARP_W] == 0))) begin
            $display("FAIL: Test 7 Multi-Warp RR (Cycle 1)");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 7 Multi-Warp RR (Cycle 1)");
        end
        saved_w0 = issue_warp_id[0*WARP_W+:WARP_W];
        saved_w1 = issue_warp_id[1*WARP_W+:WARP_W];
        @(negedge clk);
        clear_warp_inst(saved_w0);
        clear_warp_inst(saved_w1);
        
        #4;
        if (issue_valid[0] !== 1'b1 || issue_warp_id[0*WARP_W+:WARP_W] !== 2) begin
            $display("FAIL: Test 7 Multi-Warp RR (Cycle 2)");
            errors = errors + 1;
        end else begin
            $display("PASS: Test 7 Multi-Warp RR (Cycle 2)");
        end
        @(negedge clk);
        clear_warp_inst(2);

        #20;
        reset_dut();
        @(negedge clk);

        $display("--- Test 8: Edge Case - Full Queues / Back-to-Back Dispatch ---");
        for(i=0; i<8; i=i+1) begin
            set_warp_inst(i, 32'hDDDD_DDDD, i[4:0]+1, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        end
        
        for(i=0; i<4; i=i+1) begin
            #4;
            if (issue_valid !== 2'b11) begin
                $display("FAIL: Test 8 Back-to-Back Dispatch (Cycle %0d)", i+1);
                errors = errors + 1;
            end else begin
                $display("PASS: Test 8 Back-to-Back Dispatch (Cycle %0d) - w%0d and w%0d", i+1, issue_warp_id[0*WARP_W+:WARP_W], issue_warp_id[1*WARP_W+:WARP_W]);
            end
            saved_w0 = issue_warp_id[0*WARP_W+:WARP_W];
            saved_w1 = issue_warp_id[1*WARP_W+:WARP_W];
            @(negedge clk);
            clear_warp_inst(saved_w0);
            clear_warp_inst(saved_w1);
        end

        #50;
        if (errors == 0) begin
            $display("=================================================");
            $display("ALL TESTS PASSED (0 errors)");
            $display("=================================================");
        end else begin
            $display("=================================================");
            $display("TEST FAILED (%0d errors)", errors);
            $display("=================================================");
        end
        $finish;
    end
endmodule
