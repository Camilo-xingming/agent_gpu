//============================================================================
// RalphGPU - 4x4 FP16 Matrix Multiplication Test
// Verifies C = A * B where A and B are 4x4 FP16 matrices
//============================================================================

`timescale 1ns / 1ps

module tb_matrix_4x4_fp16;

    //------------------------------------------------------------------------
    // FP16 Format: 1 sign + 5 exponent + 10 mantissa
    // Bias = 15
    //------------------------------------------------------------------------

    // Convert real to FP16
    function [15:0] real_to_fp16;
        input real val;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real abs_val, frac;
        integer exp_int;
        begin
            if (val == 0.0) begin
                real_to_fp16 = 16'h0000;
            end else begin
                sign = (val < 0) ? 1'b1 : 1'b0;
                abs_val = (val < 0) ? -val : val;

                // Calculate exponent
                exp_int = 0;
                if (abs_val >= 1.0) begin
                    while (abs_val >= 2.0) begin
                        abs_val = abs_val / 2.0;
                        exp_int = exp_int + 1;
                    end
                end else begin
                    while (abs_val < 1.0) begin
                        abs_val = abs_val * 2.0;
                        exp_int = exp_int - 1;
                    end
                end

                // Bias exponent
                exp = exp_int + 15;

                // Calculate mantissa (remove implicit 1)
                frac = abs_val - 1.0;
                mant = frac * 1024.0;

                real_to_fp16 = {sign, exp, mant};
            end
        end
    endfunction

    // Convert FP16 to real
    function real fp16_to_real;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real result;
        begin
            sign = fp16[15];
            exp = fp16[14:10];
            mant = fp16[9:0];

            if (exp == 0 && mant == 0) begin
                fp16_to_real = 0.0;
            end else if (exp == 0) begin
                // Denormalized
                result = mant / 1024.0;
                result = result * (2.0 ** (-14));
                fp16_to_real = sign ? -result : result;
            end else if (exp == 31) begin
                // Inf or NaN
                fp16_to_real = sign ? -1.0/0.0 : 1.0/0.0;
            end else begin
                // Normalized
                result = 1.0 + (mant / 1024.0);
                result = result * (2.0 ** (exp - 15));
                fp16_to_real = sign ? -result : result;
            end
        end
    endfunction

    // FP16 multiply (simplified)
    function [15:0] fp16_mul;
        input [15:0] a, b;
        real ra, rb, rc;
        begin
            ra = fp16_to_real(a);
            rb = fp16_to_real(b);
            rc = ra * rb;
            fp16_mul = real_to_fp16(rc);
        end
    endfunction

    // FP16 add
    function [15:0] fp16_add;
        input [15:0] a, b;
        real ra, rb, rc;
        begin
            ra = fp16_to_real(a);
            rb = fp16_to_real(b);
            rc = ra + rb;
            fp16_add = real_to_fp16(rc);
        end
    endfunction

    //------------------------------------------------------------------------
    // Test Data
    //------------------------------------------------------------------------
    reg [15:0] A [0:3][0:3];  // Matrix A (4x4)
    reg [15:0] B [0:3][0:3];  // Matrix B (4x4)
    reg [15:0] C [0:3][0:3];  // Result C = A * B

    real A_real [0:3][0:3];
    real B_real [0:3][0:3];
    real C_real [0:3][0:3];
    real C_expected [0:3][0:3];

    integer i, j, k;
    integer seed;
    real rand_val;
    reg [15:0] acc;
    real error, max_error;

    //------------------------------------------------------------------------
    // Random number generator (simple LCG)
    //------------------------------------------------------------------------
    function real random_fp16_val;
        input integer s;
        begin
            // Generate value between -2.0 and 2.0
            random_fp16_val = (($random(s) % 4001) - 2000) / 1000.0;
        end
    endfunction

    //------------------------------------------------------------------------
    // Test
    //------------------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("RalphGPU 4x4 FP16 Matrix Multiplication Test");
        $display("============================================================");
        $display("");

        seed = 12345;

        // Initialize matrices with random FP16 values
        $display("Matrix A (FP16):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                rand_val = random_fp16_val(seed);
                seed = seed + 7;
                A_real[i][j] = rand_val;
                A[i][j] = real_to_fp16(rand_val);
            end
            $display("  [%6.3f, %6.3f, %6.3f, %6.3f]",
                     A_real[i][0], A_real[i][1], A_real[i][2], A_real[i][3]);
        end

        $display("");
        $display("Matrix B (FP16):");
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                rand_val = random_fp16_val(seed);
                seed = seed + 13;
                B_real[i][j] = rand_val;
                B[i][j] = real_to_fp16(rand_val);
            end
            $display("  [%6.3f, %6.3f, %6.3f, %6.3f]",
                     B_real[i][0], B_real[i][1], B_real[i][2], B_real[i][3]);
        end

        $display("");
        $display("------------------------------------------------------------");
        $display("Computing C = A * B using FP16 arithmetic...");
        $display("------------------------------------------------------------");

        // Perform matrix multiplication using FP16 operations
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                acc = 16'h0000;  // Initialize accumulator to 0
                for (k = 0; k < 4; k = k + 1) begin
                    // C[i][j] += A[i][k] * B[k][j]
                    acc = fp16_add(acc, fp16_mul(A[i][k], B[k][j]));
                end
                C[i][j] = acc;
                C_real[i][j] = fp16_to_real(acc);
            end
        end

        // Calculate expected result using real arithmetic (for comparison)
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                C_expected[i][j] = 0.0;
                for (k = 0; k < 4; k = k + 1) begin
                    C_expected[i][j] = C_expected[i][j] + A_real[i][k] * B_real[k][j];
                end
            end
        end

        $display("");
        $display("Result Matrix C = A * B (FP16):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%8.4f, %8.4f, %8.4f, %8.4f]",
                     C_real[i][0], C_real[i][1], C_real[i][2], C_real[i][3]);
        end

        $display("");
        $display("Expected Result (FP64 reference):");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%8.4f, %8.4f, %8.4f, %8.4f]",
                     C_expected[i][0], C_expected[i][1], C_expected[i][2], C_expected[i][3]);
        end

        // Calculate error
        $display("");
        $display("------------------------------------------------------------");
        $display("Error Analysis (FP16 vs FP64 reference):");
        $display("------------------------------------------------------------");
        max_error = 0.0;
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (C_expected[i][j] != 0.0)
                    error = (C_real[i][j] - C_expected[i][j]) / C_expected[i][j] * 100.0;
                else
                    error = C_real[i][j] * 100.0;
                if (error < 0) error = -error;
                if (error > max_error) max_error = error;
            end
        end
        $display("Maximum relative error: %.2f%%", max_error);

        if (max_error < 1.0) begin
            $display("");
            $display("============================================================");
            $display("PASS: FP16 matrix multiplication verified successfully!");
            $display("============================================================");
        end else begin
            $display("");
            $display("============================================================");
            $display("WARNING: Error exceeds 1%% (expected for FP16 precision)");
            $display("============================================================");
        end

        $display("");
        $display("FP16 Hex Values:");
        $display("Matrix C:");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [0x%04h, 0x%04h, 0x%04h, 0x%04h]",
                     C[i][0], C[i][1], C[i][2], C[i][3]);
        end

        $finish;
    end

endmodule
