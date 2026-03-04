`timescale 1ns / 1ps
module tb_advanced_scheduler();
    localparam NUM_WARPS = 8;
    localparam INST_WIDTH = 32;
    localparam NUM_ISSUE = 2;
    localparam WARP_W = $clog2(NUM_WARPS);

    reg clk;
    reg rst_n;
    reg [NUM_WARPS-1:0] warp_valid, warp_ready, warp_diverged, warp_at_barrier;
    reg [NUM_WARPS*INST_WIDTH-1:0] warp_inst;
    reg [NUM_WARPS-1:0] warp_inst_valid;
    wire [NUM_WARPS-1:0] warp_inst_consume;
    reg [NUM_WARPS*5-1:0] warp_rd, warp_rs1, warp_rs2, warp_rs3;
    reg [NUM_WARPS-1:0] warp_reads_rs3, warp_is_compute, warp_is_tensor, warp_is_memory, warp_is_branch, warp_writes_reg;
    reg compute_pipe0_ready, compute_pipe1_ready, tensor_pipe_ready, memory_pipe_ready, branch_unit_ready;
    wire [NUM_ISSUE-1:0] issue_valid;
    wire [NUM_ISSUE*WARP_W-1:0] issue_warp_id;
    wire [NUM_ISSUE*INST_WIDTH-1:0] issue_inst;
    wire [NUM_ISSUE*3-1:0] issue_pipe;
    reg wb_valid;
    reg [WARP_W-1:0] wb_warp_id;
    reg [4:0] wb_rd;
    wire [31:0] stat_cycles, stat_single_issue, stat_dual_issue, stat_stalls;
    wire [NUM_WARPS*4-1:0] issue_seq_out;
    wire [NUM_WARPS*32-1:0] scoreboard_out;

    advanced_warp_scheduler #(.NUM_WARPS(NUM_WARPS), .INST_WIDTH(INST_WIDTH), .NUM_ISSUE(NUM_ISSUE), .SCOREBOARD_DEPTH(16)) dut (
        .clk(clk), .rst_n(rst_n), .warp_valid(warp_valid), .warp_ready(warp_ready), .warp_diverged(warp_diverged), .warp_at_barrier(warp_at_barrier),
        .warp_inst(warp_inst), .warp_inst_valid(warp_inst_valid), .warp_inst_consume(warp_inst_consume),
        .warp_rd(warp_rd), .warp_rs1(warp_rs1), .warp_rs2(warp_rs2), .warp_rs3(warp_rs3), .warp_reads_rs3(warp_reads_rs3),
        .warp_is_compute(warp_is_compute), .warp_is_tensor(warp_is_tensor), .warp_is_memory(warp_is_memory), .warp_is_branch(warp_is_branch), .warp_writes_reg(warp_writes_reg),
        .compute_pipe0_ready(compute_pipe0_ready), .compute_pipe1_ready(compute_pipe1_ready), .tensor_pipe_ready(tensor_pipe_ready), .memory_pipe_ready(memory_pipe_ready), .branch_unit_ready(branch_unit_ready),
        .issue_valid(issue_valid), .issue_warp_id(issue_warp_id), .issue_inst(issue_inst), .issue_pipe(issue_pipe),
        .wb_valid(wb_valid), .wb_warp_id(wb_warp_id), .wb_rd(wb_rd),
        .stat_cycles(stat_cycles), .stat_single_issue(stat_single_issue), .stat_dual_issue(stat_dual_issue), .stat_stalls(stat_stalls),
        .issue_seq_out(issue_seq_out), .scoreboard_out(scoreboard_out)
    );

    always #5 clk = ~clk;

    task set_inst(input [WARP_W-1:0] w_id, input [INST_WIDTH-1:0] inst, input [4:0] rd, input [4:0] rs1, input [4:0] rs2, input [4:0] rs3, input is_compute, input is_tensor, input is_mem, input is_br, input w_reg);
    begin
        warp_valid[w_id] = 1; warp_ready[w_id] = 1; warp_inst_valid[w_id] = 1;
        warp_inst[w_id*INST_WIDTH +: INST_WIDTH] = inst;
        warp_rd[w_id*5 +: 5] = rd; warp_rs1[w_id*5 +: 5] = rs1; warp_rs2[w_id*5 +: 5] = rs2; warp_rs3[w_id*5 +: 5] = rs3;
        warp_reads_rs3[w_id] = (rs3 != 0);
        warp_is_compute[w_id] = is_compute; warp_is_tensor[w_id] = is_tensor; warp_is_memory[w_id] = is_mem; warp_is_branch[w_id] = is_br; warp_writes_reg[w_id] = w_reg;
    end
    endtask

    task clear_inst(input [WARP_W-1:0] w_id);
    begin
        warp_inst_valid[w_id] = 0; warp_is_compute[w_id] = 0; warp_is_tensor[w_id] = 0; warp_is_memory[w_id] = 0; warp_is_branch[w_id] = 0; warp_writes_reg[w_id] = 0;
    end
    endtask

    integer errors;
    initial begin
        clk = 0; rst_n = 0;
        warp_valid = 0; warp_ready = 0; warp_diverged = 0; warp_at_barrier = 0; warp_inst = 0; warp_inst_valid = 0;
        warp_rd = 0; warp_rs1 = 0; warp_rs2 = 0; warp_rs3 = 0; warp_reads_rs3 = 0;
        warp_is_compute = 0; warp_is_tensor = 0; warp_is_memory = 0; warp_is_branch = 0; warp_writes_reg = 0;
        compute_pipe0_ready = 1; compute_pipe1_ready = 1; tensor_pipe_ready = 1; memory_pipe_ready = 1; branch_unit_ready = 1;
        wb_valid = 0; wb_warp_id = 0; wb_rd = 0; errors = 0;

        @(negedge clk); rst_n = 1;
        @(negedge clk);

        // Test 1: Single Compute Issue
        set_inst(0, 32'h00200000, 5'd1, 5'd2, 5'd3, 5'd0, 1, 0, 0, 0, 1);
        #4; // Check just before posedge
        if (warp_inst_consume[0] !== 1'b1) begin $display("ERROR: T1 consume %b", warp_inst_consume[0]); errors++; end
        @(negedge clk); // Clock 1 ends
        clear_inst(0);

        // Test 2: RAW hazard detection
        set_inst(0, 32'h00400000, 5'd2, 5'd1, 5'd3, 5'd0, 1, 0, 0, 0, 1);
        #4; 
        if (issue_valid !== 2'b00) begin $display("ERROR: T2 valid (should be 0) %b", issue_valid); errors++; end
        @(negedge clk); // Clock 2 ends
        
        // Clear dependency
        wb_valid = 1; wb_warp_id = 0; wb_rd = 5'd1;
        @(negedge clk); // Clock 3 ends, scoreboard cleared at posedge during this cycle
        wb_valid = 0;
        #4;
        if (issue_valid !== 2'b01) begin $display("ERROR: T2 resolved valid, was %b", issue_valid); errors++; end
        @(negedge clk);
        clear_inst(0);

        // Test 3: Dual issue across warps
        set_inst(1, 32'h00200000, 5'd1, 5'd2, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        set_inst(2, 32'h00600000, 5'd3, 5'd4, 5'd0, 5'd0, 0, 1, 0, 0, 1);
        #4;
        if (issue_valid !== 2'b11) begin $display("ERROR: T3 dual issue valid %b", issue_valid); errors++; end
        @(negedge clk); 
        clear_inst(1); clear_inst(2);

        // Test 4: Priority Arbitration
        set_inst(3, 32'h00200000, 5'd4, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        set_inst(4, 32'h00200000, 5'd5, 5'd0, 5'd0, 5'd0, 0, 1, 0, 0, 1);
        set_inst(5, 32'h00200000, 5'd6, 5'd0, 5'd0, 5'd0, 0, 0, 1, 0, 1);
        set_inst(6, 32'h00000000, 5'd0, 5'd0, 5'd0, 5'd0, 0, 0, 0, 1, 0);
        #4; 
        if (issue_valid !== 2'b11) begin $display("ERROR: T4 valid"); errors++; end
        if (issue_warp_id[WARP_W-1:0] !== 6) begin $display("ERROR: T4 slot 0 not branch %d", issue_warp_id[WARP_W-1:0]); errors++; end
        if (issue_warp_id[2*WARP_W-1:WARP_W] !== 3) begin $display("ERROR: T4 slot 1 not compute %d", issue_warp_id[2*WARP_W-1:WARP_W]); errors++; end
        @(negedge clk);
        clear_inst(3); clear_inst(4); clear_inst(5); clear_inst(6);

        // Test 5: Multi-Warp Round Robin
        set_inst(0, 32'h00200000, 5'd1, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        set_inst(1, 32'h00400000, 5'd2, 5'd0, 5'd0, 5'd0, 1, 0, 0, 0, 1);
        #4; 
        if (issue_valid !== 2'b11) begin $display("ERROR: T5 dual compute valid: %b", issue_valid); errors++; end
        @(negedge clk);
        clear_inst(0); clear_inst(1);

        if (errors == 0) $display("TB PASS: All tests passed!");
        else $display("TB FAIL: %0d errors found.", errors);
        $finish;
    end
endmodule
