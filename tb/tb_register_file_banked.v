//============================================================================
// RalphGPU - Banked Register File Testbench
// Verifies banked register file with ECC, conflict detection, and
// operand collector interface
//============================================================================

`timescale 1ns / 1ps

`include "../rtl/gpu_defines.vh"

module tb_register_file_banked;

    //------------------------------------------------------------------------
    // Parameters (small configuration for fast simulation)
    //------------------------------------------------------------------------
    localparam NUM_WARPS       = 2;
    localparam NUM_REGS        = 32;
    localparam NUM_LANES       = 4;
    localparam DATA_WIDTH      = 32;
    localparam NUM_BANKS       = 2;
    localparam NUM_READ_PORTS  = 3;
    localparam NUM_WRITE_PORTS = 1;
    localparam ECC_ENABLE      = 1;
    localparam ECC_BITS        = 7;

    localparam SIMD_WIDTH      = NUM_LANES * DATA_WIDTH;  // 128
    localparam WARP_ID_W       = $clog2(NUM_WARPS);       // 1
    localparam LANE_W          = $clog2(NUM_LANES);       // 2

    localparam CLK_PERIOD      = 10;

    //------------------------------------------------------------------------
    // Clock and Reset
    //------------------------------------------------------------------------
    reg clk;
    reg rst_n;

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //------------------------------------------------------------------------
    // DUT Signals
    //------------------------------------------------------------------------
    // Read ports
    reg  [WARP_ID_W-1:0]        rd_warp_id;
    reg  [WARP_ID_W-1:0]        wr_warp_id;
    reg  [4:0]                   rd_addr_a, rd_addr_b, rd_addr_c;
    wire [SIMD_WIDTH-1:0]        rd_data_a, rd_data_b, rd_data_c;
    wire                         rd_conflict_a, rd_conflict_b, rd_conflict_c;

    // Write port
    reg                          wr_en;
    reg  [4:0]                   wr_addr;
    reg  [SIMD_WIDTH-1:0]        wr_data;
    reg  [NUM_LANES-1:0]         wr_mask;
    wire                         wr_conflict;

    // Operand collector interface
    reg                          oc_valid;
    reg  [WARP_ID_W-1:0]        oc_warp_id;
    reg  [NUM_READ_PORTS*5-1:0]  oc_addr;
    wire [NUM_READ_PORTS*SIMD_WIDTH-1:0] oc_data;
    wire [NUM_READ_PORTS-1:0]    oc_ready;
    wire                         oc_conflict;

    // Statistics
    wire [31:0]                  stat_bank_conflicts;
    wire [31:0]                  stat_total_accesses;

    // ECC
    wire                         ecc_error_corrected;
    wire                         ecc_error_detected;
    wire [31:0]                  stat_ecc_corrections;
    wire [31:0]                  stat_ecc_uncorrectable;
    wire [WARP_ID_W-1:0]        ecc_error_warp;
    wire [4:0]                   ecc_error_reg;
    wire [LANE_W-1:0]           ecc_error_lane;

    //------------------------------------------------------------------------
    // DUT Instantiation
    //------------------------------------------------------------------------
    register_file_banked #(
        .NUM_WARPS       (NUM_WARPS),
        .NUM_REGS        (NUM_REGS),
        .NUM_LANES       (NUM_LANES),
        .DATA_WIDTH      (DATA_WIDTH),
        .NUM_BANKS       (NUM_BANKS),
        .NUM_READ_PORTS  (NUM_READ_PORTS),
        .NUM_WRITE_PORTS (NUM_WRITE_PORTS),
        .ECC_ENABLE      (ECC_ENABLE),
        .ECC_BITS        (ECC_BITS)
    ) dut (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .rd_warp_id            (rd_warp_id),
        .wr_warp_id            (wr_warp_id),
        .rd_addr_a             (rd_addr_a),
        .rd_addr_b             (rd_addr_b),
        .rd_addr_c             (rd_addr_c),
        .rd_data_a             (rd_data_a),
        .rd_data_b             (rd_data_b),
        .rd_data_c             (rd_data_c),
        .rd_conflict_a         (rd_conflict_a),
        .rd_conflict_b         (rd_conflict_b),
        .rd_conflict_c         (rd_conflict_c),
        .wr_en                 (wr_en),
        .wr_addr               (wr_addr),
        .wr_data               (wr_data),
        .wr_mask               (wr_mask),
        .wr_conflict           (wr_conflict),
        .oc_valid              (oc_valid),
        .oc_warp_id            (oc_warp_id),
        .oc_addr               (oc_addr),
        .oc_data               (oc_data),
        .oc_ready              (oc_ready),
        .oc_conflict           (oc_conflict),
        .stat_bank_conflicts   (stat_bank_conflicts),
        .stat_total_accesses   (stat_total_accesses),
        .ecc_error_corrected   (ecc_error_corrected),
        .ecc_error_detected    (ecc_error_detected),
        .stat_ecc_corrections  (stat_ecc_corrections),
        .stat_ecc_uncorrectable(stat_ecc_uncorrectable),
        .ecc_error_warp        (ecc_error_warp),
        .ecc_error_reg         (ecc_error_reg),
        .ecc_error_lane        (ecc_error_lane)
    );

    //------------------------------------------------------------------------
    // Test Counters
    //------------------------------------------------------------------------
    integer pass_count;
    integer fail_count;
    integer test_num;

    //------------------------------------------------------------------------
    // Helper: Build SIMD data vector (base_value + lane_index per lane)
    //------------------------------------------------------------------------
    function [SIMD_WIDTH-1:0] make_simd_data;
        input [DATA_WIDTH-1:0] base_value;
        integer j;
        begin
            make_simd_data = {SIMD_WIDTH{1'b0}};
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                make_simd_data[j*DATA_WIDTH +: DATA_WIDTH] = base_value + j;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Helper: Check one lane of a data bus
    //------------------------------------------------------------------------
    task check_lane;
        input [SIMD_WIDTH-1:0] actual_bus;
        input integer          lane_idx;
        input [DATA_WIDTH-1:0] expected;
        input [255:0]          label;
        reg [DATA_WIDTH-1:0]   actual_lane;
        begin
            actual_lane = actual_bus[lane_idx*DATA_WIDTH +: DATA_WIDTH];
            if (actual_lane === expected) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL [Test %0d] %0s lane %0d: expected %08h, got %08h",
                         test_num, label, lane_idx, expected, actual_lane);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: Check all lanes of a data bus against base+lane pattern
    //------------------------------------------------------------------------
    task check_all_lanes;
        input [SIMD_WIDTH-1:0] actual_bus;
        input [DATA_WIDTH-1:0] base_value;
        input [255:0]          label;
        integer k;
        begin
            for (k = 0; k < NUM_LANES; k = k + 1) begin
                check_lane(actual_bus, k, base_value + k, label);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: Write register
    //------------------------------------------------------------------------
    task write_reg;
        input [WARP_ID_W-1:0]  warp;
        input [4:0]             addr;
        input [SIMD_WIDTH-1:0]  data;
        input [NUM_LANES-1:0]   mask;
        begin
            @(posedge clk);
            wr_en      <= 1'b1;
            wr_warp_id <= warp;
            wr_addr    <= addr;
            wr_data    <= data;
            wr_mask    <= mask;
            @(posedge clk);
            wr_en      <= 1'b0;
            wr_mask    <= {NUM_LANES{1'b0}};
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: Read via direct ports (combinational, settle 1 cycle)
    //------------------------------------------------------------------------
    task read_ports;
        input [WARP_ID_W-1:0] warp;
        input [4:0]            addr_a;
        input [4:0]            addr_b;
        input [4:0]            addr_c;
        begin
            @(posedge clk);
            rd_warp_id <= warp;
            rd_addr_a  <= addr_a;
            rd_addr_b  <= addr_b;
            rd_addr_c  <= addr_c;
            @(posedge clk);  // let combinational logic settle
        end
    endtask

    //------------------------------------------------------------------------
    // Helper: Clear all inputs
    //------------------------------------------------------------------------
    task clear_inputs;
        begin
            wr_en      <= 1'b0;
            wr_warp_id <= 0;
            wr_addr    <= 0;
            wr_data    <= 0;
            wr_mask    <= 0;
            rd_warp_id <= 0;
            rd_addr_a  <= 0;
            rd_addr_b  <= 0;
            rd_addr_c  <= 0;
            oc_valid   <= 1'b0;
            oc_warp_id <= 0;
            oc_addr    <= 0;
        end
    endtask

    //------------------------------------------------------------------------
    // Main Test Sequence
    //------------------------------------------------------------------------
    integer i;

    initial begin
        $dumpfile("tb_register_file_banked.vcd");
        $dumpvars(0, tb_register_file_banked);

        pass_count = 0;
        fail_count = 0;
        test_num   = 0;

        // Initialize all inputs
        rst_n = 1'b0;
        clear_inputs();

        // ================================================================
        // Test 1: Reset - all reads return 0
        // ================================================================
        test_num = 1;
        $display("\n=== Test %0d: Reset - all reads return 0 ===", test_num);

        // Hold reset for several cycles
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // Read register 0, 1, 31 from warp 0 on all three ports
        rd_warp_id <= 0;
        rd_addr_a  <= 5'd0;
        rd_addr_b  <= 5'd1;
        rd_addr_c  <= 5'd31;
        @(posedge clk);
        #1;

        for (i = 0; i < NUM_LANES; i = i + 1) begin
            check_lane(rd_data_a, i, 32'h0, "reset_port_a");
            check_lane(rd_data_b, i, 32'h0, "reset_port_b");
            check_lane(rd_data_c, i, 32'h0, "reset_port_c");
        end
        $display("Test %0d: Reset check done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 2: Write then Read - write to reg 5 warp 0, read back
        // ================================================================
        test_num = 2;
        $display("\n=== Test %0d: Write then Read ===", test_num);

        write_reg(0, 5'd5, make_simd_data(32'hDEAD_0000), {NUM_LANES{1'b1}});

        read_ports(0, 5'd5, 5'd0, 5'd0);
        #1;

        check_all_lanes(rd_data_a, 32'hDEAD_0000, "wr_rd_port_a");
        $display("Test %0d: Write-Read done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 3: Multi-lane write with mask - partial write
        // ================================================================
        test_num = 3;
        $display("\n=== Test %0d: Multi-lane write with mask ===", test_num);

        // Write all lanes of reg 10, warp 0 with known pattern
        write_reg(0, 5'd10, make_simd_data(32'hAAAA_0000), {NUM_LANES{1'b1}});

        // Overwrite only lanes 0 and 2 (mask = 4'b0101)
        write_reg(0, 5'd10, make_simd_data(32'hBBBB_0000), 4'b0101);

        // Read back
        read_ports(0, 5'd10, 5'd0, 5'd0);
        #1;

        // Lane 0: overwritten to 0xBBBB_0000
        check_lane(rd_data_a, 0, 32'hBBBB_0000, "mask_lane0");
        // Lane 1: unchanged at 0xAAAA_0001
        check_lane(rd_data_a, 1, 32'hAAAA_0001, "mask_lane1");
        // Lane 2: overwritten to 0xBBBB_0002
        check_lane(rd_data_a, 2, 32'hBBBB_0002, "mask_lane2");
        // Lane 3: unchanged at 0xAAAA_0003
        check_lane(rd_data_a, 3, 32'hAAAA_0003, "mask_lane3");
        $display("Test %0d: Mask write done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 4: Multi-warp isolation - different values in same reg
        // ================================================================
        test_num = 4;
        $display("\n=== Test %0d: Multi-warp isolation ===", test_num);

        // Write reg 7 in warp 0
        write_reg(0, 5'd7, make_simd_data(32'h1111_0000), {NUM_LANES{1'b1}});
        // Write reg 7 in warp 1 with different data
        write_reg(1, 5'd7, make_simd_data(32'h2222_0000), {NUM_LANES{1'b1}});

        // Read warp 0, reg 7
        read_ports(0, 5'd7, 5'd0, 5'd0);
        #1;
        check_all_lanes(rd_data_a, 32'h1111_0000, "warp0_iso");

        // Read warp 1, reg 7
        read_ports(1, 5'd7, 5'd0, 5'd0);
        #1;
        check_all_lanes(rd_data_a, 32'h2222_0000, "warp1_iso");

        $display("Test %0d: Warp isolation done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 5: Three-port simultaneous read
        // ================================================================
        test_num = 5;
        $display("\n=== Test %0d: Three-port simultaneous read ===", test_num);

        // Write 3 different registers in warp 0
        write_reg(0, 5'd1, make_simd_data(32'hCAFE_0000), {NUM_LANES{1'b1}});
        write_reg(0, 5'd2, make_simd_data(32'hBEEF_0000), {NUM_LANES{1'b1}});
        write_reg(0, 5'd3, make_simd_data(32'hF00D_0000), {NUM_LANES{1'b1}});

        // Read all 3 simultaneously
        read_ports(0, 5'd1, 5'd2, 5'd3);
        #1;

        check_all_lanes(rd_data_a, 32'hCAFE_0000, "3port_a");
        check_all_lanes(rd_data_b, 32'hBEEF_0000, "3port_b");
        check_all_lanes(rd_data_c, 32'hF00D_0000, "3port_c");

        $display("Test %0d: Three-port read done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 6: Direct read-port bank conflict flags
        // ================================================================
        test_num = 6;
        $display("\n=== Test %0d: Direct read-port bank conflict flags ===", test_num);

        // NUM_BANKS=2 => bank = reg_addr[0]
        // A=1(bank1), B=3(bank1), C=2(bank0) => conflicts on A/B only
        read_ports(0, 5'd1, 5'd3, 5'd2);
        #1;
        if (rd_conflict_a && rd_conflict_b && !rd_conflict_c) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL [Test %0d] unexpected rd_conflict flags (case1): a=%b b=%b c=%b",
                     test_num, rd_conflict_a, rd_conflict_b, rd_conflict_c);
        end

        // A=0(bank0), B=1(bank1), C=2(bank0) => conflicts on A/C only
        read_ports(0, 5'd0, 5'd1, 5'd2);
        #1;
        if (rd_conflict_a && !rd_conflict_b && rd_conflict_c) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL [Test %0d] unexpected rd_conflict flags (case2): a=%b b=%b c=%b",
                     test_num, rd_conflict_a, rd_conflict_b, rd_conflict_c);
        end

        $display("Test %0d: Read-port conflict flags done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 7: Operand collector arbitration / partial ready
        // ================================================================
        test_num = 7;
        $display("\n=== Test %0d: Operand collector arbitration ===", test_num);

        // Reuse regs 1,2,3 from test 5.
        @(posedge clk);
        oc_valid   <= 1'b1;
        oc_warp_id <= 0;
        // p0=reg1(bank1), p1=reg2(bank0), p2=reg3(bank1)
        // Expected grant: p0,p1 granted; p2 blocked => oc_ready=3'b011
        oc_addr    <= {5'd3, 5'd2, 5'd1};
        @(posedge clk);
        #1;

        check_all_lanes(oc_data[0*SIMD_WIDTH +: SIMD_WIDTH], 32'hCAFE_0000, "oc_p0_data");
        check_all_lanes(oc_data[1*SIMD_WIDTH +: SIMD_WIDTH], 32'hBEEF_0000, "oc_p1_data");
        if (oc_ready === 3'b011 && oc_conflict) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL [Test %0d] oc_ready/cf case1 mismatch: ready=%b conflict=%b",
                     test_num, oc_ready, oc_conflict);
        end

        // p0=reg2(bank0), p1=reg4(bank0), p2=reg1(bank1)
        // Expected grant: p0,p2 granted; p1 blocked => oc_ready=3'b101
        @(posedge clk);
        oc_addr <= {5'd1, 5'd4, 5'd2};
        @(posedge clk);
        #1;

        if (oc_ready === 3'b101 && oc_conflict) begin
            pass_count = pass_count + 1;
        end else begin
            fail_count = fail_count + 1;
            $display("FAIL [Test %0d] oc_ready/cf case2 mismatch: ready=%b conflict=%b",
                     test_num, oc_ready, oc_conflict);
        end

        oc_valid <= 1'b0;
        $display("Test %0d: Operand collector arbitration done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 8: Write-read forwarding (write then read next cycle)
        // ================================================================
        test_num = 8;
        $display("\n=== Test %0d: Write-read forwarding ===", test_num);

        // Write reg 20 warp 1
        @(posedge clk);
        wr_en      <= 1'b1;
        wr_warp_id <= 1;
        wr_addr    <= 5'd20;
        wr_data    <= make_simd_data(32'hFACE_0000);
        wr_mask    <= {NUM_LANES{1'b1}};
        // Set up read address for the next cycle
        rd_warp_id <= 1;
        rd_addr_a  <= 5'd20;
        rd_addr_b  <= 5'd0;
        rd_addr_c  <= 5'd0;

        @(posedge clk);
        wr_en  <= 1'b0;
        wr_mask <= {NUM_LANES{1'b0}};

        // The write landed on the previous posedge, read should see it now
        #1;
        check_all_lanes(rd_data_a, 32'hFACE_0000, "fwd_rd");
        $display("Test %0d: Write-read forwarding done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Test 9: Statistics counting - stat_total_accesses increments
        // ================================================================
        test_num = 9;
        $display("\n=== Test %0d: Statistics counting ===", test_num);

        begin : stat_block
            reg [31:0] count_before;
            reg [31:0] count_after;

            @(posedge clk);
            #1;
            count_before = stat_total_accesses;

            // Perform an OC access (oc_valid=1 increments stat_total_accesses)
            @(posedge clk);
            oc_valid   <= 1'b1;
            oc_warp_id <= 0;
            oc_addr    <= {5'd1, 5'd2, 5'd3};
            @(posedge clk);
            oc_valid   <= 1'b0;
            @(posedge clk);
            #1;

            count_after = stat_total_accesses;

            if (count_after > count_before) begin
                pass_count = pass_count + 1;
                $display("  stat_total_accesses incremented: %0d -> %0d", count_before, count_after);
            end else begin
                fail_count = fail_count + 1;
                $display("FAIL [Test %0d] stat_total_accesses did not increment: %0d -> %0d",
                         test_num, count_before, count_after);
            end
        end
        $display("Test %0d: Statistics done (pass=%0d, fail=%0d)", test_num, pass_count, fail_count);

        // ================================================================
        // Summary
        // ================================================================
        $display("\n============================================================");
        $display("  REGISTER FILE BANKED TESTBENCH RESULTS");
        $display("============================================================");
        $display("  Tests run : %0d", test_num);
        $display("  Checks    : %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count == 0)
            $display("  STATUS    : ALL PASSED");
        else
            $display("  STATUS    : FAILED");
        $display("============================================================\n");
        $finish;
    end

    //------------------------------------------------------------------------
    // Timeout watchdog
    //------------------------------------------------------------------------
    initial begin
        #100000;
        $display("ERROR: Testbench timed out!");
        $finish;
    end

endmodule
