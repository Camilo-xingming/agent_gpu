`timescale 1ns / 1ps

module tb_branch_predictor;

    localparam CLK_PERIOD = 10;

    reg clk;
    reg rst_n;

    reg         pred_req;
    reg  [2:0]  pred_warp_id;
    reg  [31:0] pred_pc;
    reg         pred_is_branch;
    reg         pred_is_call;
    reg         pred_is_return;

    wire        pred_valid;
    wire        pred_taken;
    wire [31:0] pred_target;
    wire [1:0]  pred_confidence;

    reg         update_valid;
    reg  [2:0]  update_warp_id;
    reg  [31:0] update_pc;
    reg         update_taken;
    reg  [31:0] update_target;
    reg         update_is_call;
    reg         update_is_return;
    reg         update_mispredicted;

    wire [31:0] stat_predictions;
    wire [31:0] stat_mispredictions;
    wire [31:0] stat_btb_hits;
    wire [31:0] stat_ras_hits;

    integer pass_count;
    integer fail_count;
    integer test_num;

    branch_predictor #(
        .NUM_WARPS(8),
        .ADDR_WIDTH(32),
        .BTB_ENTRIES(32),
        .BTB_WAYS(4),
        .BHT_ENTRIES(64),
        .TAGE_TABLES(4),
        .TAGE_ENTRIES(32),
        .RAS_DEPTH(4),
        .LOOP_ENTRIES(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pred_req(pred_req),
        .pred_warp_id(pred_warp_id),
        .pred_pc(pred_pc),
        .pred_is_branch(pred_is_branch),
        .pred_is_call(pred_is_call),
        .pred_is_return(pred_is_return),
        .pred_valid(pred_valid),
        .pred_taken(pred_taken),
        .pred_target(pred_target),
        .pred_confidence(pred_confidence),
        .update_valid(update_valid),
        .update_warp_id(update_warp_id),
        .update_pc(update_pc),
        .update_taken(update_taken),
        .update_target(update_target),
        .update_is_call(update_is_call),
        .update_is_return(update_is_return),
        .update_mispredicted(update_mispredicted),
        .stat_predictions(stat_predictions),
        .stat_mispredictions(stat_mispredictions),
        .stat_btb_hits(stat_btb_hits),
        .stat_ras_hits(stat_ras_hits)
    );

    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    task pulse_pred;
        input [31:0] pc;
        input is_branch;
        input is_call;
        input is_return;
        begin
            @(negedge clk);
            pred_req <= 1'b1;
            pred_warp_id <= 3'd0;
            pred_pc <= pc;
            pred_is_branch <= is_branch;
            pred_is_call <= is_call;
            pred_is_return <= is_return;
            @(posedge clk);
            #1;
            @(negedge clk);
            pred_req <= 1'b0;
            pred_is_branch <= 1'b0;
            pred_is_call <= 1'b0;
            pred_is_return <= 1'b0;
        end
    endtask

    task pulse_update;
        input [31:0] pc;
        input taken;
        input [31:0] target;
        input is_call;
        input is_return;
        input mispred;
        begin
            @(negedge clk);
            update_valid <= 1'b1;
            update_warp_id <= 3'd0;
            update_pc <= pc;
            update_taken <= taken;
            update_target <= target;
            update_is_call <= is_call;
            update_is_return <= is_return;
            update_mispredicted <= mispred;
            @(negedge clk);
            update_valid <= 1'b0;
            update_is_call <= 1'b0;
            update_is_return <= 1'b0;
            update_mispredicted <= 1'b0;
        end
    endtask

    task check_result;
        input cond;
        input [255:0] msg;
        begin
            if (cond) begin
                $display("  PASS: %s", msg);
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: %s", msg);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        rst_n = 1'b0;
        pred_req = 1'b0;
        pred_warp_id = 3'd0;
        pred_pc = 32'd0;
        pred_is_branch = 1'b0;
        pred_is_call = 1'b0;
        pred_is_return = 1'b0;

        update_valid = 1'b0;
        update_warp_id = 3'd0;
        update_pc = 32'd0;
        update_taken = 1'b0;
        update_target = 32'd0;
        update_is_call = 1'b0;
        update_is_return = 1'b0;
        update_mispredicted = 1'b0;

        #(CLK_PERIOD * 4);
        rst_n = 1'b1;
        #(CLK_PERIOD * 2);

        $display("====================================================");
        $display("Branch Predictor Testbench");
        $display("====================================================");

        // Test 1: Default conditional branch prediction
        test_num = test_num + 1;
        $display("\n[TEST %0d] Default branch prediction", test_num);
        pulse_pred(32'h0000_0100, 1'b1, 1'b0, 1'b0);
        check_result(pred_valid === 1'b1, "prediction is valid");
        check_result(pred_taken === 1'b0, "default bimodal is not-taken");
        check_result(pred_target === 32'h0000_0104, "no BTB -> fall-through target");
        check_result(pred_confidence == 2'd1, "default confidence is low-medium");

        // Test 2: Call without BTB entry falls through to pc+4
        test_num = test_num + 1;
        $display("\n[TEST %0d] Call prediction without BTB", test_num);
        pulse_pred(32'h0000_0200, 1'b0, 1'b1, 1'b0);
        check_result(pred_valid === 1'b1, "call prediction valid");
        check_result(pred_taken === 1'b1, "call predicted taken");
        check_result(pred_target === 32'h0000_0204, "call target defaults to pc+4 without BTB");
        check_result(pred_confidence == 2'd1, "call without BTB has low confidence");

        // Test 3: Train call target and RAS via update
        test_num = test_num + 1;
        $display("\n[TEST %0d] Update inserts BTB call entry", test_num);
        pulse_update(32'h0000_0200, 1'b1, 32'h0000_0500, 1'b1, 1'b0, 1'b0);

        // Test 4: Call now uses BTB target
        test_num = test_num + 1;
        $display("\n[TEST %0d] Call prediction with BTB hit", test_num);
        pulse_pred(32'h0000_0200, 1'b0, 1'b1, 1'b0);
        check_result(pred_taken === 1'b1, "call still predicted taken");
        check_result(pred_target === 32'h0000_0500, "BTB target returned after update");
        check_result(pred_confidence == 2'd3, "BTB hit raises confidence");

        // Test 5: Return prediction comes from RAS
        test_num = test_num + 1;
        $display("\n[TEST %0d] Return prediction from RAS", test_num);
        pulse_pred(32'h0000_0900, 1'b0, 1'b0, 1'b1);
        check_result(pred_taken === 1'b1, "return predicted taken");
        check_result(pred_target === 32'h0000_0204, "RAS supplies return address from call update");
        check_result(pred_confidence == 2'd3, "return prediction has high confidence");

        // Test 6: Mispredict counter increments on update
        test_num = test_num + 1;
        $display("\n[TEST %0d] Misprediction statistics", test_num);
        pulse_update(32'h0000_0300, 1'b0, 32'h0000_0304, 1'b0, 1'b0, 1'b1);
        #(CLK_PERIOD);
        check_result(stat_predictions == 32'd1, "one branch prediction counted");
        check_result(stat_mispredictions == 32'd1, "one misprediction counted");
        check_result(stat_btb_hits >= 32'd1, "BTB hit counter incremented");
        check_result(stat_ras_hits == 32'd1, "RAS hit counter incremented on return prediction");

        $display("\n====================================================");
        $display("RESULT: PASS=%0d FAIL=%0d", pass_count, fail_count);
        $display("====================================================");

        if (fail_count == 0) begin
            $finish;
        end else begin
            $fatal(1, "tb_branch_predictor failed with %0d checks", fail_count);
        end
    end

endmodule
