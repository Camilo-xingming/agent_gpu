//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Test (Using Real RTL)
// Uses actual fp16_mul and fp32_add modules from tensor_core.v
// C = A * B where A, B are 4x4 FP16 matrices, C is FP32
//============================================================================

`timescale 1ns / 1ps

module tb_matrix_4x4_fp16_rtl;

    //------------------------------------------------------------------------
    // Clock
    //------------------------------------------------------------------------
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //------------------------------------------------------------------------
    // Matrix Data - Flat arrays for reliable indexing
    //------------------------------------------------------------------------
    // A[i*4+j], B[i*4+j] for row i, col j
    reg [15:0] A [0:15];
    reg [15:0] B [0:15];
    reg [31:0] C [0:15];

    //------------------------------------------------------------------------
    // RTL Instantiation
    //------------------------------------------------------------------------
    reg  [15:0] mul_a [0:3];
    reg  [15:0] mul_b [0:3];
    wire [31:0] mul_result [0:3];

    fp16_mul u_mul0 (.a(mul_a[0]), .b(mul_b[0]), .result(mul_result[0]));
    fp16_mul u_mul1 (.a(mul_a[1]), .b(mul_b[1]), .result(mul_result[1]));
    fp16_mul u_mul2 (.a(mul_a[2]), .b(mul_b[2]), .result(mul_result[2]));
    fp16_mul u_mul3 (.a(mul_a[3]), .b(mul_b[3]), .result(mul_result[3]));

    wire [31:0] sum_01, sum_23, sum_0123;
    fp32_add u_add01   (.a(mul_result[0]), .b(mul_result[1]), .result(sum_01));
    fp32_add u_add23   (.a(mul_result[2]), .b(mul_result[3]), .result(sum_23));
    fp32_add u_add0123 (.a(sum_01),        .b(sum_23),        .result(sum_0123));

    //------------------------------------------------------------------------
    // FP32 to real conversion
    //------------------------------------------------------------------------
    function real fp32_to_real;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] mant;
        real result;
        integer exp_unbiased;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            mant = fp32[22:0];
            if (exp == 0 && mant == 0) begin
                fp32_to_real = 0.0;
            end else if (exp == 8'hFF) begin
                fp32_to_real = (mant != 0) ? 0.0/0.0 : (sign ? -1.0e38 : 1.0e38);
            end else begin
                exp_unbiased = exp - 127;
                result = 1.0 + (mant * 1.0 / 8388608.0);
                if (exp_unbiased >= 0) begin
                    repeat(exp_unbiased) result = result * 2.0;
                end else begin
                    repeat(-exp_unbiased) result = result / 2.0;
                end
                fp32_to_real = sign ? -result : result;
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // Test
    //------------------------------------------------------------------------
    integer i, j, k;
    real expected;
    real rtl_val;
    real A_real [0:15];
    real B_real [0:15];

    initial begin
        $display("============================================================");
        $display("RalphGPU 4x4 FP16 Matrix Multiplication - RTL Simulation");
        $display("Using actual fp16_mul and fp32_add from tensor_core.v");
        $display("============================================================");
        $display("");

        // Initialize Matrix A (row-major: A[i*4+j] = A[row i][col j])
        // Row 0: [1.0, -1.0, 2.0, 0.5]
        A[0] = 16'h3C00; A_real[0] =  1.0;
        A[1] = 16'hBC00; A_real[1] = -1.0;
        A[2] = 16'h4000; A_real[2] =  2.0;
        A[3] = 16'h3800; A_real[3] =  0.5;
        // Row 1: [-2.0, 3.0, 0.25, -0.5]
        A[4] = 16'hC000; A_real[4] = -2.0;
        A[5] = 16'h4200; A_real[5] =  3.0;
        A[6] = 16'h3400; A_real[6] =  0.25;
        A[7] = 16'hB800; A_real[7] = -0.5;
        // Row 2: [4.0, 0.125, -3.0, 1.5]
        A[8]  = 16'h4400; A_real[8]  =  4.0;
        A[9]  = 16'h3000; A_real[9]  =  0.125;
        A[10] = 16'hC200; A_real[10] = -3.0;
        A[11] = 16'h3E00; A_real[11] =  1.5;
        // Row 3: [-0.25, 2.5, 1.0, -4.0]
        A[12] = 16'hB400; A_real[12] = -0.25;
        A[13] = 16'h4100; A_real[13] =  2.5;
        A[14] = 16'h3C00; A_real[14] =  1.0;
        A[15] = 16'hC400; A_real[15] = -4.0;

        // Initialize Matrix B (row-major)
        // Row 0: [1.5, -1.5, 2.0, 0.125]
        B[0] = 16'h3E00; B_real[0] =  1.5;
        B[1] = 16'hBE00; B_real[1] = -1.5;
        B[2] = 16'h4000; B_real[2] =  2.0;
        B[3] = 16'h3000; B_real[3] =  0.125;
        // Row 1: [3.0, -2.5, 0.5, 4.0]
        B[4] = 16'h4200; B_real[4] =  3.0;
        B[5] = 16'hC100; B_real[5] = -2.5;
        B[6] = 16'h3800; B_real[6] =  0.5;
        B[7] = 16'h4400; B_real[7] =  4.0;
        // Row 2: [-1.0, 1.0, -2.0, 0.25]
        B[8]  = 16'hBC00; B_real[8]  = -1.0;
        B[9]  = 16'h3C00; B_real[9]  =  1.0;
        B[10] = 16'hC000; B_real[10] = -2.0;
        B[11] = 16'h3400; B_real[11] =  0.25;
        // Row 3: [2.0, 1.5, -0.5, -3.0]
        B[12] = 16'h4000; B_real[12] =  2.0;
        B[13] = 16'h3E00; B_real[13] =  1.5;
        B[14] = 16'hB800; B_real[14] = -0.5;
        B[15] = 16'hC200; B_real[15] = -3.0;

        $display("Matrix A (FP16):");
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", A_real[0], A_real[1], A_real[2], A_real[3]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", A_real[4], A_real[5], A_real[6], A_real[7]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", A_real[8], A_real[9], A_real[10], A_real[11]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", A_real[12], A_real[13], A_real[14], A_real[15]);

        $display("");
        $display("Matrix B (FP16):");
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", B_real[0], B_real[1], B_real[2], B_real[3]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", B_real[4], B_real[5], B_real[6], B_real[7]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", B_real[8], B_real[9], B_real[10], B_real[11]);
        $display("  [%7.4f, %7.4f, %7.4f, %7.4f]", B_real[12], B_real[13], B_real[14], B_real[15]);

        $display("");
        $display("------------------------------------------------------------");
        $display("Computing C = A * B using RTL fp16_mul and fp32_add...");
        $display("------------------------------------------------------------");

        // Matrix multiply: C[i][j] = sum_k(A[i][k] * B[k][j])
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                // A[i][k] = A[i*4 + k], B[k][j] = B[k*4 + j]
                mul_a[0] = A[i*4 + 0]; mul_b[0] = B[0*4 + j];
                mul_a[1] = A[i*4 + 1]; mul_b[1] = B[1*4 + j];
                mul_a[2] = A[i*4 + 2]; mul_b[2] = B[2*4 + j];
                mul_a[3] = A[i*4 + 3]; mul_b[3] = B[3*4 + j];

                @(posedge clk);  // Wait one clock cycle
                @(posedge clk);  // Ensure stable
                C[i*4 + j] = sum_0123;
            end
        end

        $display("");
        $display("Result Matrix C = A * B (RTL FP32 output):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%9.5f, %9.5f, %9.5f, %9.5f]",
                     fp32_to_real(C[i*4+0]), fp32_to_real(C[i*4+1]),
                     fp32_to_real(C[i*4+2]), fp32_to_real(C[i*4+3]));
        end

        $display("");
        $display("Expected Result (calculated from FP16 values):");
        for (i = 0; i < 4; i = i + 1) begin
            $write("  [");
            for (j = 0; j < 4; j = j + 1) begin
                expected = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    expected = expected + A_real[i*4+k] * B_real[k*4+j];
                end
                if (j > 0) $write(", ");
                $write("%9.5f", expected);
            end
            $display("]");
        end

        $display("");
        $display("------------------------------------------------------------");
        $display("Element-by-Element Verification:");
        $display("------------------------------------------------------------");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                expected = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    expected = expected + A_real[i*4+k] * B_real[k*4+j];
                end
                rtl_val = fp32_to_real(C[i*4+j]);
                $display("  C[%0d][%0d]: RTL=%9.5f, Expected=%9.5f, Match=%s",
                         i, j, rtl_val, expected,
                         ((rtl_val - expected) < 0.01 && (rtl_val - expected) > -0.01) ? "YES" : "NO");
            end
        end

        $display("");
        $display("============================================================");
        $display("RTL FP16 Matrix Multiplication Test Complete");
        $display("============================================================");

        $display("");
        $display("FP32 Hex Output:");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  Row %0d: [0x%08h, 0x%08h, 0x%08h, 0x%08h]",
                     i, C[i*4], C[i*4+1], C[i*4+2], C[i*4+3]);
        end

        #20;
        $finish;
    end

endmodule
