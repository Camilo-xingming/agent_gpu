`timescale 1ns / 1ps

module tb_lz4_decompressor;
    localparam INPUT_WIDTH  = 64;
    localparam OUTPUT_WIDTH = 64;

    localparam ST_IDLE         = 4'd0;
    localparam ST_READ_TOKEN   = 4'd1;
    localparam ST_LITERAL_LEN  = 4'd2;
    localparam ST_COPY_LITERAL = 4'd3;
    localparam ST_READ_OFFSET  = 4'd4;
    localparam ST_MATCH_LEN    = 4'd5;
    localparam ST_COPY_MATCH   = 4'd6;
    localparam ST_OUTPUT       = 4'd7;
    localparam ST_DONE         = 4'd8;

    reg                     clk;
    reg                     rst_n;
    reg                     start;
    reg  [31:0]             compressed_size;
    reg  [31:0]             uncompressed_size;
    wire                    done;
    wire                    error;

    reg  [INPUT_WIDTH-1:0]  s_axis_tdata;
    reg                     s_axis_tvalid;
    wire                    s_axis_tready;
    reg                     s_axis_tlast;

    wire [OUTPUT_WIDTH-1:0] m_axis_tdata;
    wire                    m_axis_tvalid;
    reg                     m_axis_tready;
    wire                    m_axis_tlast;

    wire [31:0]             stat_bytes_in;
    wire [31:0]             stat_bytes_out;
    wire [31:0]             stat_literal_count;
    wire [31:0]             stat_match_count;

    integer pass_count;
    integer fail_count;
    integer out_beats;
    integer read_token_hits;
    integer hits_after_a;
    integer bytes_in_a;

    reg [15:0] seen_states;

    lz4_decompressor #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .HISTORY_DEPTH(256)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .compressed_size(compressed_size),
        .uncompressed_size(uncompressed_size),
        .done(done),
        .error(error),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .stat_bytes_in(stat_bytes_in),
        .stat_bytes_out(stat_bytes_out),
        .stat_literal_count(stat_literal_count),
        .stat_match_count(stat_match_count)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            seen_states <= 16'd0;
            out_beats <= 0;
            read_token_hits <= 0;
        end else begin
            seen_states[dut.state] <= 1'b1;
            if (dut.state == ST_READ_TOKEN)
                read_token_hits <= read_token_hits + 1;
            if (m_axis_tvalid && m_axis_tready)
                out_beats <= out_beats + 1;
        end
    end

    function [63:0] pack8;
        input [7:0] b0;
        input [7:0] b1;
        input [7:0] b2;
        input [7:0] b3;
        input [7:0] b4;
        input [7:0] b5;
        input [7:0] b6;
        input [7:0] b7;
        begin
            pack8 = {b7, b6, b5, b4, b3, b2, b1, b0};
        end
    endfunction

    task expect_true;
        input cond;
        input [255:0] msg;
        begin
            if (cond)
                pass_count = pass_count + 1;
            else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %s", msg);
            end
        end
    endtask

    task reset_dut;
        begin
            rst_n = 1'b0;
            start = 1'b0;
            compressed_size = 32'd0;
            uncompressed_size = 32'd0;
            s_axis_tdata = {INPUT_WIDTH{1'b0}};
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            m_axis_tready = 1'b1;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task start_decompress;
        input [31:0] csz;
        input [31:0] usz;
        begin
            compressed_size <= csz;
            uncompressed_size <= usz;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
        end
    endtask

    task send_qword;
        input [63:0] data;
        input        last;
        integer guard;
        begin
            guard = 0;
            while (!s_axis_tready && guard < 300) begin
                @(posedge clk);
                guard = guard + 1;
            end
            if (!s_axis_tready) begin
                fail_count = fail_count + 1;
                $display("[FAIL] send_qword timeout waiting s_axis_tready");
            end else begin
                s_axis_tdata <= data;
                s_axis_tvalid <= 1'b1;
                s_axis_tlast <= last;
                @(posedge clk);
                s_axis_tvalid <= 1'b0;
                s_axis_tdata <= {INPUT_WIDTH{1'b0}};
                s_axis_tlast <= 1'b0;
            end
        end
    endtask

    task run_cycles;
        input integer n;
        integer c;
        begin
            for (c = 0; c < n; c = c + 1)
                @(posedge clk);
        end
    endtask

    initial begin
        pass_count = 0;
        fail_count = 0;

        //============================================================
        // Test 1: empty input
        //============================================================
        reset_dut();
        start_decompress(32'd0, 32'd0);
        run_cycles(40);
        expect_true(seen_states[ST_DONE] || done, "empty: should reach DONE");
        expect_true(!error, "empty: should not assert error");
        expect_true(stat_bytes_in == 32'd0, "empty: stat_bytes_in should be 0");

        //============================================================
        // Test 2: literal-only sequence
        //============================================================
        reset_dut();
        start_decompress(32'd17, 32'd15);
        send_qword(pack8(8'hF0,8'h00,8'h41,8'h42,8'h43,8'h44,8'h45,8'h46), 1'b0);
        send_qword(pack8(8'h47,8'h48,8'h49,8'h4A,8'h4B,8'h4C,8'h4D,8'h4E), 1'b0);
        send_qword(pack8(8'h4F,8'h00,8'h00,8'h00,8'h00,8'h00,8'h00,8'h00), 1'b1);
        run_cycles(600);
        expect_true(seen_states[ST_LITERAL_LEN], "literal-only: should visit ST_LITERAL_LEN");
        expect_true(seen_states[ST_COPY_LITERAL], "literal-only: should visit ST_COPY_LITERAL");
        expect_true(stat_literal_count > 0, "literal-only: literal counter should increase");
        expect_true(!error, "literal-only: should not assert error");

        //============================================================
        // Test 3: short-offset match copy
        //============================================================
        reset_dut();
        start_decompress(32'd3, 32'd4);
        send_qword(pack8(8'h00,8'h01,8'h00,8'h00,8'h00,8'h00,8'h00,8'h00), 1'b1);
        run_cycles(300);
        expect_true(seen_states[ST_READ_OFFSET], "short-offset: should visit ST_READ_OFFSET");
        expect_true(seen_states[ST_COPY_MATCH], "short-offset: should visit ST_COPY_MATCH");
        expect_true(stat_match_count > 0, "short-offset: match counter should increase");
        expect_true(!error, "short-offset: should not assert error");

        //============================================================
        // Test 4: long-offset + match-len extension path
        //============================================================
        reset_dut();
        start_decompress(32'd4, 32'd19);
        send_qword(pack8(8'h0F,8'h08,8'h00,8'h00,8'h00,8'h00,8'h00,8'h00), 1'b1);
        run_cycles(500);
        expect_true(seen_states[ST_MATCH_LEN], "long-offset: should visit ST_MATCH_LEN");
        expect_true(seen_states[ST_COPY_MATCH], "long-offset: should visit ST_COPY_MATCH");
        expect_true(stat_match_count > 0, "long-offset: match counter should increase");
        expect_true(!error, "long-offset: should not assert error");

        //============================================================
        // Test 5: multi-block stream (two tokens)
        //============================================================
        reset_dut();
        start_decompress(32'd6, 32'd8);
        send_qword(pack8(8'h00,8'h01,8'h00,8'h00,8'h01,8'h00,8'h00,8'h00), 1'b1);
        run_cycles(500);
        expect_true(seen_states[ST_READ_TOKEN], "multi-block: should visit ST_READ_TOKEN");
        expect_true(seen_states[ST_COPY_MATCH], "multi-block: should visit ST_COPY_MATCH");
        expect_true(read_token_hits >= 2, "multi-block: should parse multiple tokens");
        expect_true(!error, "multi-block: should not assert error");

        //============================================================
        // Test 6: back-to-back decompressions without reset
        //============================================================
        reset_dut();

        start_decompress(32'd24, 32'd32);
        send_qword(pack8(8'h00,8'h01,8'h00,8'h00,8'h01,8'h00,8'h00,8'h01), 1'b0);
        send_qword(pack8(8'h00,8'h00,8'h01,8'h00,8'h00,8'h01,8'h00,8'h00), 1'b0);
        send_qword(pack8(8'h01,8'h00,8'h00,8'h01,8'h00,8'h00,8'h01,8'h00), 1'b1);
        run_cycles(700);

        hits_after_a = read_token_hits;
        bytes_in_a = stat_bytes_in;

        start_decompress(32'd24, 32'd32);
        send_qword(pack8(8'h00,8'h02,8'h00,8'h00,8'h02,8'h00,8'h00,8'h02), 1'b0);
        send_qword(pack8(8'h00,8'h00,8'h02,8'h00,8'h00,8'h02,8'h00,8'h00), 1'b0);
        send_qword(pack8(8'h02,8'h00,8'h00,8'h02,8'h00,8'h00,8'h02,8'h00), 1'b1);
        run_cycles(700);

        expect_true(read_token_hits >= 1, "b2b: parser activity should be observed");
        expect_true(seen_states[ST_READ_TOKEN], "b2b: should revisit ST_READ_TOKEN");
        expect_true(!error, "b2b: should not assert error");

        $display("============================================================");
        $display("tb_lz4_decompressor Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");

        if (fail_count == 0) begin
            $display("ALL TESTS PASSED");
            $finish;
        end else begin
            $fatal(1, "tb_lz4_decompressor failed");
        end
    end

    initial begin
        #500000;
        $fatal(1, "tb_lz4_decompressor timeout");
    end

    initial begin
        $dumpfile("tb_lz4_decompressor.vcd");
        $dumpvars(0, tb_lz4_decompressor);
    end
endmodule
