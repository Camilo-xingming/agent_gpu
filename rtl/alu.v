//============================================================================
// RalphGPU - ALU (Arithmetic Logic Unit)
// 32ä½ç®—æœ¯é€»è¾‘å•å…ƒï¼Œæ”¯æŒå®Œæ•´PTXæ•´æ•°è¿®—æŒ‡ä»¤é›†
// æ”¯æŒ: åŸºç¡€ç®—æœ¯ã€ä½æ“ä½œã€ä½åŸŸæ“ä½œã€é€‰æ‹©æ“ä½œ
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module alu (
    input  wire [5:0]  func,        // åŠŸèƒ½ç 
    input  wire [31:0] operand_a,   // æ“ä½œæ•°A
    input  wire [31:0] operand_b,   // æ“ä½œæ•°B
    input  wire [31:0] operand_c,   // æ“ä½œæ•°C (ç”¨äºŽBFI, PRMT, SAD, SELP)
    input  wire        pred_in,     // è°“è¯è¾“å…¥ (ç”¨äºŽSELP)
    input  wire        carry_in,    // è¿›ä½è¾“å…¥ (ç”¨äºŽaddc, subc)
    output reg  [31:0] result,      // ç»“æžœ
    output reg  [31:0] result_hi,   // é«˜32ä½ç»“æžœ (ç”¨äºŽmul.wide)
    output wire        zero,        // é›¶æ ‡å¿—
    output wire        negative,    // è´Ÿæ•°æ ‡å¿—
    output wire        overflow,    // æº¢å‡ºæ ‡å¿—
    output reg         carry_out    // è¿›ä½è¾“å‡º (ç”¨äºŽadd.cc, sub.cc)
);

    //------------------------------------------------------------------------
    // å†…éƒ¨ä¿¡å·
    //------------------------------------------------------------------------
    wire [32:0] add_result;
    wire [32:0] sub_result;
    wire [32:0] addc_result;    // add with carry
    wire [32:0] subc_result;    // sub with borrow
    wire signed [31:0] signed_a;
    wire signed [31:0] signed_b;

    assign signed_a = operand_a;
    assign signed_b = operand_b;
    assign add_result = {1'b0, operand_a} + {1'b0, operand_b};
    assign sub_result = {1'b0, operand_a} - {1'b0, operand_b};
    assign addc_result = {1'b0, operand_a} + {1'b0, operand_b} + {32'b0, carry_in};
    assign subc_result = {1'b0, operand_a} - {1'b0, operand_b} - {32'b0, carry_in};

    //------------------------------------------------------------------------
    // MUL.WIDE (32x32 -> 64-bit result)
    //------------------------------------------------------------------------
    wire [63:0] mul_wide_u = operand_a * operand_b;
    wire signed [63:0] mul_wide_s = signed_a * signed_b;

    //------------------------------------------------------------------------
    // POPC (Population Count) - è®¡ç®—1çš„ä¸ªæ•°
    //------------------------------------------------------------------------
    function [5:0] popc32;
        input [31:0] val;
        integer i;
        begin
            popc32 = 0;
            for (i = 0; i < 32; i = i + 1) begin
                popc32 = popc32 + {5'b0, val[i]};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // CLZ (Count Leading Zeros)
    //------------------------------------------------------------------------
    function [5:0] clz32;
        input [31:0] val;
        integer i;
        reg found;
        begin
            clz32 = 32;
            found = 0;
            for (i = 31; i >= 0; i = i - 1) begin
                if (!found && val[i]) begin
                    clz32 = 6'd31 - i[5:0];
                    found = 1;
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BFIND (Find Most Significant Bit) - è¿”å›žMSBä½ç½®
    //------------------------------------------------------------------------
    function [31:0] bfind32;
        input [31:0] val;
        input        is_signed;
        reg [31:0] search_val;
        integer i;
        reg found;
        begin
            // å¯¹äºŽæœ‰ç¬¦å·æ•°ï¼Œå¦‚æžœæ˜¯è´Ÿæ•°ï¼Œå…ˆå–å
            search_val = (is_signed && val[31]) ? ~val : val;
            bfind32 = 32'hFFFFFFFF;  // -1 è¡¨ç¤ºæœªæ‰¾åˆ°
            found = 0;
            for (i = 31; i >= 0; i = i - 1) begin
                if (!found && search_val[i]) begin
                    bfind32 = i;
                    found = 1;
                end
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BREV (Bit Reverse)
    //------------------------------------------------------------------------
    function [31:0] brev32;
        input [31:0] val;
        integer i;
        begin
            for (i = 0; i < 32; i = i + 1) begin
                brev32[31-i] = val[i];
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // BFE (Bit Field Extract)
    // ä»Žoperand_aä¸­æ–ä»Žä½ç½®poså¼€å§‹çš„lenä½
    // operand_b[7:0] = pos, operand_b[15:8] = len
    //------------------------------------------------------------------------
    wire [4:0] bfe_pos = operand_b[4:0];
    wire [4:0] bfe_len = operand_b[12:8];
    wire [31:0] bfe_mask = (bfe_len == 0) ? 32'b0 : ((32'hFFFFFFFF >> (32 - bfe_len)));
    wire [31:0] bfe_shifted = operand_a >> bfe_pos;
    wire [31:0] bfe_result_u = bfe_shifted & bfe_mask;
    // æœ‰ç¬¦å·æ‰©å±•
    wire bfe_sign_bit = (bfe_len > 0) ? bfe_shifted[bfe_len-1] : 1'b0;
    wire [31:0] bfe_sign_extend = (bfe_sign_bit && bfe_len > 0) ?
                                  (~bfe_mask) : 32'b0;
    wire [31:0] bfe_result_s = bfe_result_u | bfe_sign_extend;

    //------------------------------------------------------------------------
    // BFI (Bit Field Insert)
    // å°†operand_açš„ä½Žlenä½æ’å…¥operand_bçš„posä½ç½®
    // operand_c[7:0] = pos, operand_c[15:8] = len
    //------------------------------------------------------------------------
    wire [4:0] bfi_pos = operand_c[4:0];
    wire [4:0] bfi_len = operand_c[12:8];
    wire [31:0] bfi_mask = (bfi_len == 0) ? 32'b0 :
                           ((32'hFFFFFFFF >> (32 - bfi_len)) << bfi_pos);
    wire [31:0] bfi_insert = (operand_a << bfi_pos) & bfi_mask;
    wire [31:0] bfi_result = (operand_b & ~bfi_mask) | bfi_insert;

    //------------------------------------------------------------------------
    // æ–°å¢žï¼šä½æŽ©ç /æ‰©å±•/æŸ¥æ‰¾/æ¼æ–—ç§»ä½/ä¸‰è¾“å…¥é€»è¾‘
    //------------------------------------------------------------------------
    wire [4:0] bmsk_pos = operand_a[4:0];
    wire [5:0] bmsk_len_ext = {1'b0, operand_b[4:0]};
    wire [31:0] bmsk_base = (bmsk_len_ext == 0) ? 32'b0 :
                            (bmsk_len_ext >= 32) ? 32'hFFFF_FFFF :
                            ((32'h1 << bmsk_len_ext) - 1);
    wire [31:0] bmsk_result = bmsk_base << bmsk_pos;

    wire [5:0] szext_width = {1'b0, operand_b[4:0]};
    wire [31:0] szext_mask = (szext_width == 0) ? 32'b0 :
                             (szext_width >= 32) ? 32'hFFFF_FFFF :
                             ((32'h1 << szext_width) - 1);
    wire szext_sign = (szext_width == 0) ? 1'b0 :
                      (szext_width >= 32) ? operand_a[31] :
                      operand_a[szext_width-1];
    wire [31:0] szext_result = szext_sign ? (operand_a | ~szext_mask) :
                                           (operand_a & szext_mask);

    function [31:0] fns32;
        input [31:0] val;
        integer i;
        begin
            fns32 = 32'hFFFF_FFFF;
            for (i = 0; i < 32; i = i + 1) begin
                if (val[i]) begin
                    fns32 = i;
                    i = 32; // break
                end
            end
        end
    endfunction

    wire [4:0] shf_amt = operand_c[4:0];
    wire [63:0] shf_cat_lr = {operand_a, operand_b};
    wire [63:0] shf_cat_rl = {operand_b, operand_a};
    wire [63:0] shf_l_tmp = shf_cat_lr << shf_amt;
    wire [63:0] shf_r_tmp = shf_cat_rl >> shf_amt;
    wire [31:0] shf_l_res = shf_l_tmp[63:32];
    wire [31:0] shf_r_res = shf_r_tmp[31:0];
    wire [31:0] lop3_res = (operand_a & operand_b) | (~operand_a & operand_c); // LUT 0xCA
    wire [31:0] cnot_res = ~operand_a & operand_b;

    //------------------------------------------------------------------------
    // DP4A / DP2A (int8/int16 dot product with accumulate)
    //------------------------------------------------------------------------
    wire signed [8:0] a_b0_s = {operand_a[7], operand_a[7:0]};
    wire signed [8:0] a_b1_s = {operand_a[15], operand_a[15:8]};
    wire signed [8:0] a_b2_s = {operand_a[23], operand_a[23:16]};
    wire signed [8:0] a_b3_s = {operand_a[31], operand_a[31:24]};

    wire signed [8:0] b_b0_s = {operand_b[7], operand_b[7:0]};
    wire signed [8:0] b_b1_s = {operand_b[15], operand_b[15:8]};
    wire signed [8:0] b_b2_s = {operand_b[23], operand_b[23:16]};
    wire signed [8:0] b_b3_s = {operand_b[31], operand_b[31:24]};

    wire signed [17:0] dp4a_p0 = a_b0_s * b_b0_s;
    wire signed [17:0] dp4a_p1 = a_b1_s * b_b1_s;
    wire signed [17:0] dp4a_p2 = a_b2_s * b_b2_s;
    wire signed [17:0] dp4a_p3 = a_b3_s * b_b3_s;
    wire signed [31:0] dp4a_sum = $signed({{14{dp4a_p0[17]}}, dp4a_p0}) + $signed({{14{dp4a_p1[17]}}, dp4a_p1}) + $signed({{14{dp4a_p2[17]}}, dp4a_p2}) + $signed({{14{dp4a_p3[17]}}, dp4a_p3}) + $signed(operand_c);

    wire signed [16:0] dp2a_p0 = $signed({operand_a[15], operand_a[15:0]}) * $signed({operand_b[15], operand_b[15:0]});
    wire signed [16:0] dp2a_p1 = $signed({operand_a[31], operand_a[31:16]}) * $signed({operand_b[31], operand_b[31:16]});
    wire signed [31:0] dp2a_sum = $signed({{15{dp2a_p0[16]}}, dp2a_p0}) + $signed({{15{dp2a_p1[16]}}, dp2a_p1}) + $signed(operand_c);

    //------------------------------------------------------------------------
    // FP16 <-> FP32 Conversion (for CVT instructions routed through ALU)
    //------------------------------------------------------------------------
    // FP16 format: [15]=sign, [14:10]=exp (bias 15), [9:0]=mantissa
    // FP32 format: [31]=sign, [30:23]=exp (bias 127), [22:0]=mantissa
    function [31:0] fp16_to_fp32;
        input [15:0] fp16;
        reg sign;
        reg [4:0] exp16;
        reg [9:0] man16;
        reg [7:0] exp32;
        reg [22:0] man32;
        begin
            sign = fp16[15];
            exp16 = fp16[14:10];
            man16 = fp16[9:0];

            if (exp16 == 5'h1F) begin
                // Inf or NaN
                exp32 = 8'hFF;
                man32 = {man16, 13'b0};
            end else if (exp16 == 5'h00) begin
                if (man16 == 10'b0) begin
                    // Zero
                    exp32 = 8'h00;
                    man32 = 23'b0;
                end else begin
                    // Denormalized - treat as zero for simplicity
                    exp32 = 8'h00;
                    man32 = 23'b0;
                end
            end else begin
                // Normal number: rebias exponent (15 -> 127)
                exp32 = {3'b0, exp16} + 8'd112;  // 127 - 15 = 112
                man32 = {man16, 13'b0};
            end

            fp16_to_fp32 = {sign, exp32, man32};
        end
    endfunction

    function [15:0] fp32_to_fp16;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp32;
        reg [22:0] man32;
        reg [4:0] exp16;
        reg [9:0] man16;
        begin
            sign = fp32[31];
            exp32 = fp32[30:23];
            man32 = fp32[22:0];

            if (exp32 == 8'hFF) begin
                // Inf or NaN
                exp16 = 5'h1F;
                man16 = man32[22:13];
            end else if (exp32 == 8'h00) begin
                // Zero or denorm
                exp16 = 5'h00;
                man16 = 10'b0;
            end else if (exp32 < 8'd113) begin
                // Underflow to zero
                exp16 = 5'h00;
                man16 = 10'b0;
            end else if (exp32 > 8'd142) begin
                // Overflow to infinity
                exp16 = 5'h1F;
                man16 = 10'b0;
            end else begin
                // Normal number: rebias exponent (127 -> 15)
                begin : fp32_to_fp16_rebias
                    reg [7:0] rebias_tmp;
                    rebias_tmp = exp32 - 8'd112;
                    exp16 = rebias_tmp[4:0];
                end
                man16 = man32[22:13];
            end

            fp32_to_fp16 = {sign, exp16, man16};
        end
    endfunction

    function [31:0] fp32_to_s32;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] man;
        reg [23:0] sig;
        reg signed [8:0] real_exp;
        reg [4:0] shift;
        reg [47:0] shifted_sig;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            man = fp32[22:0];
            sig = {1'b1, man};
            real_exp = {1'b0, exp} - 8'd127;

            if (exp == 8'hFF) begin
                fp32_to_s32 = sign ? 32'h80000000 : 32'h7FFFFFFF;
            end else if (exp == 0 && man == 0) begin
                fp32_to_s32 = 32'b0;
            end else if (real_exp < 0) begin
                fp32_to_s32 = 32'b0;
            end else if (real_exp > 30) begin
                fp32_to_s32 = sign ? 32'h80000000 : 32'h7FFFFFFF;
            end else begin
                if (real_exp <= 23) begin
                    shift = 23 - real_exp[4:0];
                    shifted_sig = {24'b0, sig} >> shift;
                end else begin
                    shift = real_exp[4:0] - 23;
                    shifted_sig = {24'b0, sig} << shift;
                end
                fp32_to_s32 = sign ? -shifted_sig[31:0] : shifted_sig[31:0];
            end
        end
    endfunction

    function [31:0] fp32_to_u32;
        input [31:0] fp32;
        reg sign;
        reg [7:0] exp;
        reg [22:0] man;
        reg [23:0] sig;
        reg signed [8:0] real_exp;
        reg [4:0] shift;
        reg [47:0] shifted_sig;
        begin
            sign = fp32[31];
            exp = fp32[30:23];
            man = fp32[22:0];
            sig = {1'b1, man};
            real_exp = {1'b0, exp} - 8'd127;

            if (exp == 8'hFF) begin
                fp32_to_u32 = sign ? 32'b0 : 32'hFFFFFFFF;
            end else if (sign && exp != 0) begin
                fp32_to_u32 = 32'b0;
            end else if (exp == 0 && man == 0) begin
                fp32_to_u32 = 32'b0;
            end else if (real_exp < 0) begin
                fp32_to_u32 = 32'b0;
            end else if (real_exp > 31) begin
                fp32_to_u32 = 32'hFFFFFFFF;
            end else begin
                if (real_exp <= 23) begin
                    shift = 23 - real_exp[4:0];
                    shifted_sig = {24'b0, sig} >> shift;
                end else begin
                    shift = real_exp[4:0] - 23;
                    shifted_sig = {24'b0, sig} << shift;
                end
                fp32_to_u32 = shifted_sig[31:0];
            end
        end
    endfunction

    function [31:0] s32_to_fp32;
        input [31:0] s32;
        reg sign;
        reg [31:0] abs_val;
        reg [4:0] lzc;
        reg [7:0] exp;
        reg [22:0] man;
        integer i;
        begin
            sign = s32[31];
            abs_val = sign ? -s32 : s32;
            
            if (s32 == 0) begin
                s32_to_fp32 = 32'b0;
            end else begin
                lzc = 0;
                for (i = 31; i >= 0; i = i - 1) begin
                    if (abs_val[i]) begin
                        lzc = 31 - i;
                        i = -1;
                    end
                end
                exp = 8'd127 + (8'd31 - {3'b0, lzc});
                abs_val = abs_val << lzc;
                man = abs_val[30:8];
                s32_to_fp32 = {sign, exp, man};
            end
        end
    endfunction

    function [31:0] u32_to_fp32;
        input [31:0] u32;
        reg [4:0] lzc;
        reg [7:0] exp;
        reg [22:0] man;
        reg [31:0] tmp_u32;
        integer i;
        begin
            tmp_u32 = u32;
            if (u32 == 0) begin
                u32_to_fp32 = 32'b0;
            end else begin
                lzc = 0;
                for (i = 31; i >= 0; i = i - 1) begin
                    if (tmp_u32[i]) begin
                        lzc = 31 - i;
                        i = -1;
                    end
                end
                exp = 8'd127 + (8'd31 - {3'b0, lzc});
                tmp_u32 = tmp_u32 << lzc;
                man = tmp_u32[30:8];
                u32_to_fp32 = {1'b0, exp, man};
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // PRMT (Permute Bytes)
    // æ ¹æ®operand_cé€‰æ‹©operand_aå’Œoperand_bçš„å­—èŠ‚
    //------------------------------------------------------------------------
    wire [63:0] prmt_src = {operand_b, operand_a};  // 8ä¸ªæº­—èŠ‚
    wire [31:0] prmt_result;
    wire [2:0] prmt_sel0 = operand_c[2:0];
    wire [2:0] prmt_sel1 = operand_c[6:4];
    wire [2:0] prmt_sel2 = operand_c[10:8];
    wire [2:0] prmt_sel3 = operand_c[14:12];

    assign prmt_result[7:0]   = prmt_src[prmt_sel0*8 +: 8];
    assign prmt_result[15:8]  = prmt_src[prmt_sel1*8 +: 8];
    assign prmt_result[23:16] = prmt_src[prmt_sel2*8 +: 8];
    assign prmt_result[31:24] = prmt_src[prmt_sel3*8 +: 8];

    //------------------------------------------------------------------------
    // SAD (Sum of Absolute Differences)
    // result = |a - b| + c
    //------------------------------------------------------------------------
    wire signed [31:0] sad_diff = signed_a - signed_b;
    wire [31:0] sad_abs = sad_diff[31] ? (-sad_diff) : sad_diff;
    wire [31:0] sad_result = sad_abs + operand_c;

    //------------------------------------------------------------------------
    // ALU æ“ä½œé€‰æ‹©
    //------------------------------------------------------------------------
    always @(*) begin
        result_hi = 32'b0;
        carry_out = 1'b0;

        /* verilator lint_off CASEOVERLAP */
        case (func)
            // åŸºç¡€è¿®—
            `FUNC_ADD:   begin
                result = add_result[31:0];
            end
            `FUNC_SUB:   begin
                result = sub_result[31:0];
            end
            `FUNC_AND:   result = operand_a & operand_b;
            `FUNC_OR:    result = operand_a | operand_b;
            `FUNC_XOR:   result = operand_a ^ operand_b;
            `FUNC_NOT:   result = ~operand_a;
            `FUNC_SHL:   result = operand_a << operand_b[4:0];
            `FUNC_SHR_U: result = operand_a >> operand_b[4:0];
            `FUNC_SHR_S: result = signed_a >>> operand_b[4:0];

            // PTXæ‰©å±•æ•´æ•°è¿®—
            `FUNC_ABS:   result = signed_a[31] ? (-signed_a) : signed_a;
            `FUNC_NEG:   result = -signed_a;
            `FUNC_MIN_S: result = (signed_a < signed_b) ? operand_a : operand_b;
            `FUNC_MIN_U: result = (operand_a < operand_b) ? operand_a : operand_b;
            `FUNC_MAX_S: result = (signed_a > signed_b) ? operand_a : operand_b;
            `FUNC_MAX_U: result = (operand_a > operand_b) ? operand_a : operand_b;

            // ä½æ“ä½œæŒ‡ä»¤
            `FUNC_POPC:  result = {26'b0, popc32(operand_a)};
            `FUNC_CLZ:   result = {26'b0, clz32(operand_a)};
            `FUNC_BFIND: result = bfind32(operand_a, 1'b1);  // æœ‰ç¬¦å·ç‰ˆæœ¬
            `FUNC_BREV:  result = brev32(operand_a);

            // ä½åŸŸæ“ä½œ
            `FUNC_BFE_S: result = bfe_result_s;
            `FUNC_BFE_U: result = bfe_result_u;
            `FUNC_BFI:   result = bfi_result;
            `FUNC_PRMT:  result = prmt_result;

            // ç‰¹æ®Šè¿®—
            `FUNC_SAD:   result = sad_result;
            `FUNC_CNOT:  result = cnot_res;
            `FUNC_BMSK:  result = bmsk_result;
            `FUNC_SZEXT: result = szext_result;
            `FUNC_FNS:   result = fns32(operand_a);
            `FUNC_SHF_L: result = shf_l_res;
            `FUNC_SHF_R: result = shf_r_res;
            `FUNC_LOP3:  result = lop3_res;
            `VIDEO_DP4A_ALU: result = dp4a_sum;
            `VIDEO_DP2A_ALU: result = dp2a_sum;

            // é€‰æ‹©æ“ä½œ
            `FUNC_SELP:  result = pred_in ? operand_a : operand_b;
            `FUNC_SLCT:  result = signed_b[31] ? operand_a : operand_b;  // æ ¹æ®cçš„ç¬¦å·é€‰æ‹©

            // è¿›ä½è¿®— (add.cc, addc, sub.cc, subc)
            `FUNC_ADD_CC: begin
                result = add_result[31:0];
                carry_out = add_result[32];
            end
            `FUNC_ADDC: begin
                result = addc_result[31:0];
                carry_out = addc_result[32];
            end
            `FUNC_SUB_CC: begin
                result = sub_result[31:0];
                carry_out = sub_result[32];  // borrow
            end
            `FUNC_SUBC: begin
                result = subc_result[31:0];
                carry_out = subc_result[32];  // borrow
            end

            // å®½ä¹˜æ³• (mul.wide: 32x32 -> 64)
            `FUNC_MUL_WIDE: begin
                result = mul_wide_u[31:0];
                result_hi = mul_wide_u[63:32];
            end

            // CVT instructions (FP16 <-> FP32 conversion)
            `CVT_S32_F32: begin
                result = fp32_to_s32(operand_a);
            end
            `CVT_U32_F32: begin
                result = fp32_to_u32(operand_a);
            end
            `CVT_F32_S32: begin
                result = s32_to_fp32(operand_a);
            end
            `CVT_F32_U32: begin
                result = u32_to_fp32(operand_a);
            end
            `CVT_F32_F16: begin
                // Convert FP16 (in low 16 bits of operand_a) to FP32
                result = fp16_to_fp32(operand_a[15:0]);
            end
            `CVT_F16_F32: begin
                // Convert FP32 (in operand_a) to FP16 (result in low 16 bits)
                result = {16'b0, fp32_to_fp16(operand_a)};
            end
            `CVT_PACK: begin
                // Pack low16 of A into lower half, low16 of B into upper half
                result = {operand_b[15:0], operand_a[15:0]};
            end

            default:     result = 32'b0;
                /* verilator lint_on CASEOVERLAP */
        endcase
    end

    //------------------------------------------------------------------------
    // æ ‡å¿—ä½ç”Ÿæˆ
    assign zero     = (result == 32'b0);
    assign negative = result[31];
    assign overflow = 1'b0; // TODO: Implement integer overflow flags

endmodule


//============================================================================
// SIMD ALU - 32ä¸ªå¹¶è¡ŒALUç”¨äºŽWarpæ‰§è¡Œ
//============================================================================
module simd_alu #(
    parameter LANES = 32
)(
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand_a,
    input  wire [LANES*32-1:0]  operand_b,
    input  wire [LANES*32-1:0]  operand_c,
    input  wire [LANES-1:0]     pred_in,
    input  wire [LANES-1:0]     carry_in,
    input  wire [LANES-1:0]     lane_mask,
    output wire [LANES*32-1:0]  result,
    output wire [LANES*32-1:0]  result_hi,
    output wire [LANES-1:0]     zero_flags,
    output wire [LANES-1:0]     neg_flags,
    output wire [LANES-1:0]     ovf_flags,
    output wire [LANES-1:0]     carry_out
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : alu_lane
            wire [31:0] lane_a = operand_a[i*32 +: 32];
            wire [31:0] lane_b = operand_b[i*32 +: 32];
            wire [31:0] lane_c = operand_c[i*32 +: 32];
            wire lane_pred = pred_in[i];
            wire lane_cin = carry_in[i];
            wire [31:0] lane_result;
            wire [31:0] lane_result_hi;
            wire lane_zero, lane_neg, lane_ovf, lane_cout;

            alu u_alu (
                .func      (func),
                .operand_a (lane_a),
                .operand_b (lane_b),
                .operand_c (lane_c),
                .pred_in   (lane_pred),
                .carry_in  (lane_cin),
                .result    (lane_result),
                .result_hi (lane_result_hi),
                .zero      (lane_zero),
                .negative  (lane_neg),
                .overflow  (lane_ovf),
                .carry_out (lane_cout)
            );

            assign result[i*32 +: 32] = lane_mask[i] ? lane_result : 32'b0;
            assign result_hi[i*32 +: 32] = lane_mask[i] ? lane_result_hi : 32'b0;
            assign zero_flags[i] = lane_mask[i] & lane_zero;
            assign neg_flags[i] = lane_mask[i] & lane_neg;
            assign ovf_flags[i] = lane_mask[i] & lane_ovf;
            assign carry_out[i] = lane_mask[i] & lane_cout;
        end
    endgenerate

endmodule
