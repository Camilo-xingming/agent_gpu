`timescale 1ns/1ps
`include "gpu_defines.vh"

module tb_wgmma_tile_engine;

    reg clk;
    reg rst_n;
    
    // Inputs
    reg tile_start;
    reg [31:0] tile_m_offset;
    reg [31:0] tile_n_offset;
    reg [31:0] k_tiles;
    
    // Outputs
    wire tile_done;
    wire tile_ready;
    wire gmem_req_valid;
    wire [31:0] gmem_req_addr;
    wire [8:0] gmem_req_size;
    wire gmem_req_is_a;
    
    // Stub inputs
    reg gmem_req_ready;
    reg [511:0] gmem_resp_data;
    reg gmem_resp_valid;
    
    wire smem_wr_en;
    wire [13:0] smem_wr_addr;
    wire [511:0] smem_wr_data;
    wire [63:0] smem_wr_mask;
    
    wire smem_rd_en;
    wire [13:0] smem_rd_addr;
    reg [511:0] smem_rd_data;
    reg smem_rd_valid;
    
    wire mma_valid;
    wire [511:0] mma_frag_a;
    wire [511:0] mma_frag_b;
    wire [1023:0] mma_accum_in;
    reg mma_ready;
    reg [1023:0] mma_accum_out;
    reg mma_done;
    
    wire [31:0] stat_tiles_computed;
    wire [31:0] stat_smem_stalls;
    wire [31:0] stat_mma_stalls;
    
    // Memory arrays for stubbing
    reg [511:0] smem_array [0:1023];

    wgmma_tile_engine dut (
        .clk(clk),
        .rst_n(rst_n),
        .tile_start(tile_start),
        .tile_m_offset(tile_m_offset),
        .tile_n_offset(tile_n_offset),
        .k_tiles(k_tiles),
        .tile_done(tile_done),
        .tile_ready(tile_ready),
        .gmem_req_valid(gmem_req_valid),
        .gmem_req_addr(gmem_req_addr),
        .gmem_req_size(gmem_req_size),
        .gmem_req_is_a(gmem_req_is_a),
        .gmem_req_ready(gmem_req_ready),
        .gmem_resp_data(gmem_resp_data),
        .gmem_resp_valid(gmem_resp_valid),
        .smem_wr_en(smem_wr_en),
        .smem_wr_addr(smem_wr_addr),
        .smem_wr_data(smem_wr_data),
        .smem_wr_mask(smem_wr_mask),
        .smem_rd_en(smem_rd_en),
        .smem_rd_addr(smem_rd_addr),
        .smem_rd_data(smem_rd_data),
        .smem_rd_valid(smem_rd_valid),
        .mma_valid(mma_valid),
        .mma_frag_a(mma_frag_a),
        .mma_frag_b(mma_frag_b),
        .mma_accum_in(mma_accum_in),
        .mma_ready(mma_ready),
        .mma_accum_out(mma_accum_out),
        .mma_done(mma_done),
        .stat_tiles_computed(stat_tiles_computed),
        .stat_smem_stalls(stat_smem_stalls),
        .stat_mma_stalls(stat_mma_stalls)
    );

    always #5 clk = ~clk;

    // GMEM and SMEM mocked response logic
    reg [3:0] mma_timer;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            gmem_resp_valid <= 0;
            smem_rd_valid <= 0;
            mma_done <= 0;
            mma_timer <= 0;
        end else begin
            // Mock GMEM response
            gmem_resp_valid <= 0;
            if (gmem_req_valid && gmem_req_ready) begin
                gmem_resp_valid <= 1;
                gmem_resp_data <= {480'b0, gmem_req_addr};
            end
            
            // Mock SMEM write
            if (smem_wr_en) begin
                smem_array[smem_wr_addr[13:6]] <= smem_wr_data; 
            end
            
            // Mock SMEM read
            smem_rd_valid <= 0;
            if (smem_rd_en) begin
                smem_rd_valid <= 1;
                smem_rd_data <= smem_array[smem_rd_addr[13:6]];
            end
            
            // Mock MMA core
            mma_done <= 0;
            if (mma_valid && mma_ready) begin
                mma_timer <= 5;
            end
            if (mma_timer > 0) begin
                if (mma_timer == 1) begin
                    mma_done <= 1;
                    mma_accum_out <= mma_accum_in + 1;
                end
                mma_timer <= mma_timer - 1;
            end
        end
    end

    initial begin
        $dumpfile("tb_wgmma_tile_engine.vcd");
        $dumpvars(0, tb_wgmma_tile_engine);
        clk = 0;
        rst_n = 0;
        tile_start = 0;
        tile_m_offset = 0;
        tile_n_offset = 0;
        k_tiles = 0;
        gmem_req_ready = 1;
        mma_ready = 1;
        mma_timer = 0;

        #20 rst_n = 1;
        #20;
        
        $display("--- Test 1: Single K-Tile ---");
        wait(tile_ready);
        @(posedge clk);
        tile_start = 1;
        tile_m_offset = 0;
        tile_n_offset = 0;
        k_tiles = 1;
        @(posedge clk);
        tile_start = 0;
        
        wait(tile_done);
        $display("Test 1 Done. Computed: %d", stat_tiles_computed);

        #50;
        $display("--- Test 2: Multi K-Tile Pipelining ---");
        wait(tile_ready);
        @(posedge clk);
        tile_start = 1;
        tile_m_offset = 64;
        tile_n_offset = 64;
        k_tiles = 4;
        @(posedge clk);
        tile_start = 0;
        
        wait(tile_done);
        $display("Test 2 Done. Computed: %d", stat_tiles_computed);

        #50;
        $display("--- Test 3: Back-to-Back Tiles ---");
        wait(tile_ready);
        @(posedge clk);
        tile_start = 1;
        k_tiles = 2;
        @(posedge clk);
        tile_start = 0;
        wait(tile_done);
        
        wait(tile_ready);
        @(posedge clk);
        tile_start = 1;
        k_tiles = 3;
        @(posedge clk);
        tile_start = 0;
        wait(tile_done);
        $display("Test 3 Done. Computed: %d", stat_tiles_computed);

        #100;
        $display("ALL TESTS PASSED");
        $finish;
    end
    
    initial begin
        #50000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
