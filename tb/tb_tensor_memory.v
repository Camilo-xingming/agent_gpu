`timescale 1ns / 1ps
`include "gpu_defines.vh"

module tb_tensor_memory;

    reg clk;
    reg rst_n;

    reg         alloc_valid;
    reg [8:0]   alloc_num_cols;
    wire        alloc_ready;
    wire [8:0]  alloc_col_base;
    wire        alloc_fail;

    reg         dealloc_valid;
    reg [8:0]   dealloc_col_base;
    reg [8:0]   dealloc_num_cols;

    reg         ld_valid;
    reg [6:0]   ld_row;
    reg [8:0]   ld_col_base;
    reg [3:0]   ld_num_cols;
    wire        ld_ready;
    wire [511:0] ld_data;

    reg         st_valid;
    reg [6:0]   st_row;
    reg [8:0]   st_col_base;
    reg [3:0]   st_num_cols;
    reg [255:0] st_data;
    reg [7:0]   st_mask;
    wire        st_ready;

    reg         mma_wr_valid;
    reg [6:0]   mma_row;
    reg [8:0]   mma_col_base;
    reg [511:0] mma_wr_data;
    reg [15:0]  mma_wr_mask;

    reg         mma_rd_valid;
    wire [511:0] mma_rd_data;
    wire        mma_rd_ready;

    reg         cp_valid;
    reg         cp_direction;
    reg [6:0]   cp_row;
    reg [8:0]   cp_col_base;
    reg [511:0] cp_wr_data;
    wire [511:0] cp_rd_data;
    wire        cp_ready;

    wire [9:0] cols_allocated;
    wire [9:0] cols_free;
    wire       tmem_full;
    wire       tmem_empty;

    integer pass_count;
    integer fail_count;
    integer i;

    tensor_memory dut (
        .clk(clk),
        .rst_n(rst_n),
        .alloc_valid(alloc_valid),
        .alloc_num_cols(alloc_num_cols),
        .alloc_ready(alloc_ready),
        .alloc_col_base(alloc_col_base),
        .alloc_fail(alloc_fail),
        .dealloc_valid(dealloc_valid),
        .dealloc_col_base(dealloc_col_base),
        .dealloc_num_cols(dealloc_num_cols),
        .ld_valid(ld_valid),
        .ld_row(ld_row),
        .ld_col_base(ld_col_base),
        .ld_num_cols(ld_num_cols),
        .ld_ready(ld_ready),
        .ld_data(ld_data),
        .st_valid(st_valid),
        .st_row(st_row),
        .st_col_base(st_col_base),
        .st_num_cols(st_num_cols),
        .st_data(st_data),
        .st_mask(st_mask),
        .st_ready(st_ready),
        .mma_wr_valid(mma_wr_valid),
        .mma_row(mma_row),
        .mma_col_base(mma_col_base),
        .mma_wr_data(mma_wr_data),
        .mma_wr_mask(mma_wr_mask),
        .mma_rd_valid(mma_rd_valid),
        .mma_rd_data(mma_rd_data),
        .mma_rd_ready(mma_rd_ready),
        .cp_valid(cp_valid),
        .cp_direction(cp_direction),
        .cp_row(cp_row),
        .cp_col_base(cp_col_base),
        .cp_wr_data(cp_wr_data),
        .cp_rd_data(cp_rd_data),
        .cp_ready(cp_ready),
        .cols_allocated(cols_allocated),
        .cols_free(cols_free),
        .tmem_full(tmem_full),
        .tmem_empty(tmem_empty)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task expect32;
        input [255:0] name;
        input [31:0] got;
        input [31:0] exp;
        begin
            if (got === exp) begin
                $display("PASS: %0s = 0x%08x", name, got);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s got=0x%08x exp=0x%08x", name, got, exp);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task expect1;
        input [255:0] name;
        input got;
        input exp;
        begin
            if (got === exp) begin
                $display("PASS: %0s = %0b", name, got);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s got=%0b exp=%0b", name, got, exp);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task reset_inputs;
        begin
            alloc_valid = 1'b0;
            alloc_num_cols = 9'd0;
            dealloc_valid = 1'b0;
            dealloc_col_base = 9'd0;
            dealloc_num_cols = 9'd0;
            ld_valid = 1'b0;
            ld_row = 7'd0;
            ld_col_base = 9'd0;
            ld_num_cols = 4'd0;
            st_valid = 1'b0;
            st_row = 7'd0;
            st_col_base = 9'd0;
            st_num_cols = 4'd0;
            st_data = 256'd0;
            st_mask = 8'd0;
            mma_wr_valid = 1'b0;
            mma_row = 7'd0;
            mma_col_base = 9'd0;
            mma_wr_data = 512'd0;
            mma_wr_mask = 16'd0;
            mma_rd_valid = 1'b0;
            cp_valid = 1'b0;
            cp_direction = 1'b0;
            cp_row = 7'd0;
            cp_col_base = 9'd0;
            cp_wr_data = 512'd0;
        end
    endtask

    task do_reset;
        begin
            rst_n = 1'b0;
            reset_inputs();
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task pulse_alloc;
        input [8:0] ncols;
        begin
            alloc_num_cols = ncols;
            alloc_valid = 1'b1;
            @(posedge clk);
            #1;
            alloc_valid = 1'b0;
            alloc_num_cols = 9'd0;
        end
    endtask

    task pulse_dealloc;
        input [8:0] base;
        input [8:0] ncols;
        begin
            dealloc_col_base = base;
            dealloc_num_cols = ncols;
            dealloc_valid = 1'b1;
            @(posedge clk);
            #1;
            dealloc_valid = 1'b0;
            dealloc_col_base = 9'd0;
            dealloc_num_cols = 9'd0;
        end
    endtask

    task pulse_store;
        input [6:0] row;
        input [8:0] base;
        input [3:0] ncols;
        input [255:0] data_in;
        input [7:0] mask_in;
        begin
            st_row = row;
            st_col_base = base;
            st_num_cols = ncols;
            st_data = data_in;
            st_mask = mask_in;
            st_valid = 1'b1;
            @(posedge clk);
            #1;
            st_valid = 1'b0;
            st_num_cols = 4'd0;
            st_mask = 8'd0;
        end
    endtask

    task pulse_load;
        input [6:0] row;
        input [8:0] base;
        input [3:0] ncols;
        begin
            ld_row = row;
            ld_col_base = base;
            ld_num_cols = ncols;
            ld_valid = 1'b1;
            @(posedge clk);
            #1;
            ld_valid = 1'b0;
            ld_num_cols = 4'd0;
        end
    endtask

    task pulse_mma_write;
        input [6:0] row;
        input [8:0] base;
        input [511:0] data_in;
        input [15:0] mask_in;
        begin
            mma_row = row;
            mma_col_base = base;
            mma_wr_data = data_in;
            mma_wr_mask = mask_in;
            mma_wr_valid = 1'b1;
            @(posedge clk);
            #1;
            mma_wr_valid = 1'b0;
            mma_wr_mask = 16'd0;
        end
    endtask

    task pulse_mma_read;
        begin
            mma_rd_valid = 1'b1;
            @(posedge clk);
            #1;
            mma_rd_valid = 1'b0;
        end
    endtask

    task pulse_cp_write;
        input [6:0] row;
        input [8:0] base;
        input [511:0] data_in;
        begin
            cp_row = row;
            cp_col_base = base;
            cp_direction = 1'b0;
            cp_wr_data = data_in;
            cp_valid = 1'b1;
            @(posedge clk);
            #1;
            cp_valid = 1'b0;
        end
    endtask

    task pulse_cp_read;
        input [6:0] row;
        input [8:0] base;
        begin
            cp_row = row;
            cp_col_base = base;
            cp_direction = 1'b1;
            cp_valid = 1'b1;
            @(posedge clk);
            #1;
            cp_valid = 1'b0;
        end
    endtask

    reg [255:0] st_pack;
    reg [511:0] mma_pack;
    reg [511:0] cp_pack;
    reg [31:0] alloc_after_first;


    initial begin
        $dumpfile("tb_tensor_memory.vcd");
        $dumpvars(0, tb_tensor_memory);

        pass_count = 0;
        fail_count = 0;
        st_pack = 256'd0;
        mma_pack = 512'd0;
        cp_pack = 512'd0;

        do_reset();

        // --------------------------------------------------------------------
        // Test 1: Allocation/deallocation and bounds
        // --------------------------------------------------------------------
        $display("=== Test 1: allocation/deallocation ===");
        expect32("reset allocated", cols_allocated, 32'd0);
        expect32("reset free", cols_free, 32'd512);
        expect1("reset empty", tmem_empty, 1'b1);
        expect1("reset full", tmem_full, 1'b0);
        pulse_alloc(9'd5); // aligns to 8
        expect1("alloc ready", alloc_ready, 1'b1);
        expect1("alloc fail", alloc_fail, 1'b0);
        expect1("alloc non-zero", (cols_allocated != 0), 1'b1);
        expect1("alloc aligned x8", (cols_allocated[2:0] == 3'b000), 1'b1);
        alloc_after_first = cols_allocated;

        pulse_alloc(9'd16);
        expect1("alloc2 ready", alloc_ready, 1'b1);
        expect1("alloc2 fail", alloc_fail, 1'b0);
        expect32("alloc2 delta", cols_allocated - alloc_after_first, 32'd16);

        // Deallocate from tail region: should roll watermark back to first allocation size
        pulse_dealloc(alloc_after_first[8:0], 9'd16);
        expect32("allocated after tail dealloc", cols_allocated, alloc_after_first);

        pulse_alloc(9'd511); // aligns to 512, should overflow
        expect1("alloc overflow ready", alloc_ready, 1'b1);
        expect1("alloc overflow fail", alloc_fail, 1'b1);
        expect32("allocated unchanged on overflow", cols_allocated, alloc_after_first);

        // --------------------------------------------------------------------
        // Test 2: Tensor store/load + masked update
        // --------------------------------------------------------------------
        $display("=== Test 2: tensor load/store ===");
        st_pack = 256'd0;
        st_pack[31:0]    = 32'h1111_0001;
        st_pack[63:32]   = 32'h2222_0002;
        st_pack[95:64]   = 32'h3333_0003;
        st_pack[127:96]  = 32'h4444_0004;

        pulse_store(7'd5, 9'd0, 4'd4, st_pack, 8'b0000_1111);
        expect1("store ready", st_ready, 1'b1);

        pulse_load(7'd5, 9'd0, 4'd4);
        expect1("load ready", ld_ready, 1'b1);
        expect32("ld col0", ld_data[31:0], 32'h1111_0001);
        expect32("ld col1", ld_data[63:32], 32'h2222_0002);
        expect32("ld col2", ld_data[95:64], 32'h3333_0003);
        expect32("ld col3", ld_data[127:96], 32'h4444_0004);

        st_pack = 256'd0;
        st_pack[31:0]    = 32'hAAAA_000A;
        st_pack[63:32]   = 32'hBBBB_000B;
        st_pack[95:64]   = 32'hCCCC_000C;
        st_pack[127:96]  = 32'hDDDD_000D;
        pulse_store(7'd5, 9'd0, 4'd4, st_pack, 8'b0000_0101);
        expect1("masked store ready", st_ready, 1'b1);

        pulse_load(7'd5, 9'd0, 4'd4);
        expect32("masked col0 updated", ld_data[31:0], 32'hAAAA_000A);
        expect32("masked col1 kept", ld_data[63:32], 32'h2222_0002);
        expect32("masked col2 updated", ld_data[95:64], 32'hCCCC_000C);
        expect32("masked col3 kept", ld_data[127:96], 32'h4444_0004);

        // --------------------------------------------------------------------
        // Test 3: Address generation + tiling (strided and blocked)
        // --------------------------------------------------------------------
        $display("=== Test 3: addressing + tiling ===");
        for (i = 0; i < 3; i = i + 1) begin
            st_pack = 256'd0;
            st_pack[31:0]  = 32'h5000_0000 + i;
            st_pack[63:32] = 32'h5000_0100 + i;
            pulse_store(i*2, 9'd16, 4'd2, st_pack, 8'b0000_0011);
        end
        pulse_load(7'd0, 9'd16, 4'd2);
        expect32("stride row0 col16", ld_data[31:0], 32'h5000_0000);
        expect32("stride row0 col17", ld_data[63:32], 32'h5000_0100);
        pulse_load(7'd2, 9'd16, 4'd2);
        expect32("stride row2 col16", ld_data[31:0], 32'h5000_0001);
        expect32("stride row2 col17", ld_data[63:32], 32'h5000_0101);
        pulse_load(7'd4, 9'd16, 4'd2);
        expect32("stride row4 col16", ld_data[31:0], 32'h5000_0002);
        expect32("stride row4 col17", ld_data[63:32], 32'h5000_0102);

        st_pack = 256'd0;
        st_pack[31:0]    = 32'h7100_0000;
        st_pack[63:32]   = 32'h7100_0001;
        st_pack[95:64]   = 32'h7100_0002;
        st_pack[127:96]  = 32'h7100_0003;
        pulse_store(7'd20, 9'd64, 4'd4, st_pack, 8'b0000_1111);

        st_pack = 256'd0;
        st_pack[31:0]    = 32'h7200_0000;
        st_pack[63:32]   = 32'h7200_0001;
        st_pack[95:64]   = 32'h7200_0002;
        st_pack[127:96]  = 32'h7200_0003;
        pulse_store(7'd21, 9'd64, 4'd4, st_pack, 8'b0000_1111);

        pulse_load(7'd20, 9'd64, 4'd4);
        expect32("tile row20 c64", ld_data[31:0], 32'h7100_0000);
        expect32("tile row20 c67", ld_data[127:96], 32'h7100_0003);
        pulse_load(7'd21, 9'd64, 4'd4);
        expect32("tile row21 c64", ld_data[31:0], 32'h7200_0000);
        expect32("tile row21 c67", ld_data[127:96], 32'h7200_0003);

        // --------------------------------------------------------------------
        // Test 4: Backpressure/stall-style behavior (back-to-back handshakes)
        // --------------------------------------------------------------------
        $display("=== Test 4: backpressure/stall handling ===");
        @(posedge clk);
        #1;
        expect1("ld_ready idle", ld_ready, 1'b0);
        expect1("st_ready idle", st_ready, 1'b0);
        expect1("mma_rd_ready idle", mma_rd_ready, 1'b0);
        expect1("cp_ready idle", cp_ready, 1'b0);

        st_pack = 256'd0;
        st_pack[31:0] = 32'h9000_0001;
        pulse_store(7'd30, 9'd32, 4'd1, st_pack, 8'b0000_0001);
        st_pack[31:0] = 32'h9000_0002;
        pulse_store(7'd31, 9'd32, 4'd1, st_pack, 8'b0000_0001);

        ld_row = 7'd30;
        ld_col_base = 9'd32;
        ld_num_cols = 4'd1;
        ld_valid = 1'b1;
        @(posedge clk);
        #1;
        expect1("ld_ready back-to-back #1", ld_ready, 1'b1);
        expect32("ld_data back-to-back #1", ld_data[31:0], 32'h9000_0001);

        ld_row = 7'd31;
        ld_col_base = 9'd32;
        ld_num_cols = 4'd1;
        @(posedge clk);
        #1;
        expect1("ld_ready back-to-back #2", ld_ready, 1'b1);
        expect32("ld_data back-to-back #2", ld_data[31:0], 32'h9000_0002);
        ld_valid = 1'b0;
        ld_num_cols = 4'd0;

        // --------------------------------------------------------------------
        // Test 5: Tensor core integration path (MMA + CP interfaces)
        // --------------------------------------------------------------------
        $display("=== Test 5: tensor core data-path integration ===");
        mma_pack = 512'd0;
        for (i = 0; i < 16; i = i + 1) begin
            mma_pack[i*32 +: 32] = 32'hA500_0000 + i;
        end
        pulse_mma_write(7'd40, 9'd100, mma_pack, 16'hFFFF);

        mma_row = 7'd40;
        mma_col_base = 9'd100;
        pulse_mma_read();
        expect1("mma rd ready", mma_rd_ready, 1'b1);
        expect32("mma rd col0", mma_rd_data[31:0], 32'hA500_0000);
        expect32("mma rd col7", mma_rd_data[255:224], 32'hA500_0007);
        expect32("mma rd col15", mma_rd_data[511:480], 32'hA500_000F);

        cp_pack = 512'd0;
        for (i = 0; i < 16; i = i + 1) begin
            cp_pack[i*32 +: 32] = 32'hC300_1000 + i;
        end
        pulse_cp_write(7'd41, 9'd120, cp_pack);
        expect1("cp write ready", cp_ready, 1'b1);

        pulse_cp_read(7'd41, 9'd120);
        expect1("cp read ready", cp_ready, 1'b1);
        expect32("cp read col0", cp_rd_data[31:0], 32'hC300_1000);
        expect32("cp read col8", cp_rd_data[287:256], 32'hC300_1008);
        expect32("cp read col15", cp_rd_data[511:480], 32'hC300_100F);

        // --------------------------------------------------------------------
        // Test 6: Boundary clipping + max-width transfers
        // --------------------------------------------------------------------
        $display("=== Test 6: boundary clipping + max-width transfers ===");

        st_pack = 256'd0;
        for (i = 0; i < 8; i = i + 1) begin
            st_pack[i*32 +: 32] = 32'h8800_0000 + i;
        end
        pulse_store(7'd63, 9'd200, 4'd8, st_pack, 8'hFF);
        expect1("store x8 ready", st_ready, 1'b1);
        pulse_load(7'd63, 9'd200, 4'd8);
        expect32("store x8 col0", ld_data[31:0], 32'h8800_0000);
        expect32("store x8 col7", ld_data[255:224], 32'h8800_0007);

        st_pack = 256'd0;
        st_pack[31:0]   = 32'hF100_0001;
        st_pack[63:32]  = 32'hF200_0002;
        st_pack[95:64]  = 32'hF300_0003;
        st_pack[127:96] = 32'hF400_0004;
        pulse_store(7'd60, 9'd509, 4'd4, st_pack, 8'b0000_1111);
        pulse_load(7'd60, 9'd509, 4'd4);
        expect32("boundary st/ld col509", ld_data[31:0], 32'hF100_0001);
        expect32("boundary st/ld col510", ld_data[63:32], 32'hF200_0002);
        expect32("boundary st/ld col511", ld_data[95:64], 32'hF300_0003);
        expect32("boundary st/ld col512 clipped", ld_data[127:96], 32'h0000_0000);

        mma_pack = 512'd0;
        for (i = 0; i < 16; i = i + 1) begin
            mma_pack[i*32 +: 32] = 32'hBEEF_1000 + i;
        end
        pulse_mma_write(7'd61, 9'd506, mma_pack, 16'hFFFF);
        mma_row = 7'd61;
        mma_col_base = 9'd506;
        pulse_mma_read();
        expect32("mma boundary col506", mma_rd_data[31:0], 32'hBEEF_1000);
        expect32("mma boundary col511", mma_rd_data[191:160], 32'hBEEF_1005);
        expect32("mma boundary col512 clipped", mma_rd_data[223:192], 32'h0000_0000);
        expect32("mma boundary high lanes clipped", mma_rd_data[511:480], 32'h0000_0000);

        cp_pack = 512'd0;
        for (i = 0; i < 16; i = i + 1) begin
            cp_pack[i*32 +: 32] = 32'hD00D_2000 + i;
        end
        pulse_cp_write(7'd62, 9'd508, cp_pack);
        pulse_cp_read(7'd62, 9'd508);
        expect32("cp boundary col508", cp_rd_data[31:0], 32'hD00D_2000);
        expect32("cp boundary col511", cp_rd_data[127:96], 32'hD00D_2003);
        expect32("cp boundary col512 clipped", cp_rd_data[159:128], 32'h0000_0000);
        expect32("cp boundary high lanes clipped", cp_rd_data[511:480], 32'h0000_0000);

        $display("============================================================");
        $display("tb_tensor_memory Summary: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================================");
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        $finish;
    end

    initial begin
        #200000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
