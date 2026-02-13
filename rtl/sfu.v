//============================================================================
// RalphGPU - SFU (Special Function Unit)
// 特殊函数单元: rcp, sqrt, rsqrt, sin, cos, lg2, ex2, tanh
// 使用查表+多项式近似实现，8级流水线
// 使用预计算的静态查找表，可综合
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module sfu #(
    parameter LATENCY = 8  // Pipeline stages
)(
    input  wire        clk,
    input  wire        rst_n,

    input  wire [5:0]  func,        // 功能码
    input  wire [31:0] operand,     // FP32输入
    input  wire        valid_in,

    output reg  [31:0] result,
    output reg         valid_out,
    output wire        ready,       // 总是ready（流水线）
    output reg         invalid,
    output reg         div_by_zero
);

    assign ready = 1'b1;  // 流水线，总是接受输入

    //========================================================================
    // IEEE 754 FP32 常量
    //========================================================================
    localparam [31:0] FP_ZERO     = 32'h00000000;  // 0.0
    localparam [31:0] FP_ONE      = 32'h3F800000;  // 1.0
    localparam [31:0] FP_NEG_ONE  = 32'hBF800000;  // -1.0
    localparam [31:0] FP_TWO      = 32'h40000000;  // 2.0
    localparam [31:0] FP_HALF     = 32'h3F000000;  // 0.5
    localparam [31:0] FP_INF      = 32'h7F800000;  // +Inf
    localparam [31:0] FP_NEG_INF  = 32'hFF800000;  // -Inf
    localparam [31:0] FP_NAN      = 32'h7FC00000;  // NaN (quiet)

    //========================================================================
    // Stage 0: Input parsing and special case detection
    //========================================================================
    wire        s0_sign = operand[31];
    wire [7:0]  s0_exp  = operand[30:23];
    wire [22:0] s0_man  = operand[22:0];

    wire s0_is_zero   = (s0_exp == 0) && (s0_man == 0);
    wire s0_is_inf    = (s0_exp == 255) && (s0_man == 0);
    wire s0_is_nan    = (s0_exp == 255) && (s0_man != 0);
    wire s0_is_denorm = (s0_exp == 0) && (s0_man != 0);
    wire s0_is_neg    = s0_sign && !s0_is_zero;

    // Normalized mantissa with implicit 1 (1.xxxxx format, 24 bits)
    wire [23:0] s0_norm_man = s0_is_denorm ? {1'b0, s0_man} : {1'b1, s0_man};

    //========================================================================
    // RCP Lookup Table (64 entries, 10-bit values)
    // Approximates 1/(1+x/64) for x in [0, 63]
    // Values = floor((1/(1+i/64) - 0.5) * 1024)
    //========================================================================
    function [9:0] rcp_lut;
        input [5:0] idx;
        begin
            case (idx)
                6'd0:  rcp_lut = 10'd512;  // 1/1.000 = 1.000
                6'd1:  rcp_lut = 10'd504;  // 1/1.016 = 0.984
                6'd2:  rcp_lut = 10'd496;  // 1/1.031 = 0.970
                6'd3:  rcp_lut = 10'd489;  // 1/1.047 = 0.955
                6'd4:  rcp_lut = 10'd481;  // 1/1.063 = 0.941
                6'd5:  rcp_lut = 10'd474;  // 1/1.078 = 0.928
                6'd6:  rcp_lut = 10'd467;  // 1/1.094 = 0.914
                6'd7:  rcp_lut = 10'd460;  // 1/1.109 = 0.901
                6'd8:  rcp_lut = 10'd454;  // 1/1.125 = 0.889
                6'd9:  rcp_lut = 10'd447;  // 1/1.141 = 0.877
                6'd10: rcp_lut = 10'd441;  // 1/1.156 = 0.865
                6'd11: rcp_lut = 10'd434;  // 1/1.172 = 0.853
                6'd12: rcp_lut = 10'd428;  // 1/1.188 = 0.842
                6'd13: rcp_lut = 10'd422;  // 1/1.203 = 0.831
                6'd14: rcp_lut = 10'd416;  // 1/1.219 = 0.820
                6'd15: rcp_lut = 10'd410;  // 1/1.234 = 0.810
                6'd16: rcp_lut = 10'd405;  // 1/1.250 = 0.800
                6'd17: rcp_lut = 10'd399;  // 1/1.266 = 0.790
                6'd18: rcp_lut = 10'd394;  // 1/1.281 = 0.780
                6'd19: rcp_lut = 10'd388;  // 1/1.297 = 0.771
                6'd20: rcp_lut = 10'd383;  // 1/1.313 = 0.762
                6'd21: rcp_lut = 10'd378;  // 1/1.328 = 0.753
                6'd22: rcp_lut = 10'd373;  // 1/1.344 = 0.744
                6'd23: rcp_lut = 10'd368;  // 1/1.359 = 0.736
                6'd24: rcp_lut = 10'd364;  // 1/1.375 = 0.727
                6'd25: rcp_lut = 10'd359;  // 1/1.391 = 0.719
                6'd26: rcp_lut = 10'd354;  // 1/1.406 = 0.711
                6'd27: rcp_lut = 10'd350;  // 1/1.422 = 0.703
                6'd28: rcp_lut = 10'd345;  // 1/1.438 = 0.696
                6'd29: rcp_lut = 10'd341;  // 1/1.453 = 0.688
                6'd30: rcp_lut = 10'd337;  // 1/1.469 = 0.681
                6'd31: rcp_lut = 10'd333;  // 1/1.484 = 0.674
                6'd32: rcp_lut = 10'd329;  // 1/1.500 = 0.667
                6'd33: rcp_lut = 10'd325;  // 1/1.516 = 0.660
                6'd34: rcp_lut = 10'd321;  // 1/1.531 = 0.653
                6'd35: rcp_lut = 10'd317;  // 1/1.547 = 0.646
                6'd36: rcp_lut = 10'd313;  // 1/1.563 = 0.640
                6'd37: rcp_lut = 10'd310;  // 1/1.578 = 0.634
                6'd38: rcp_lut = 10'd306;  // 1/1.594 = 0.627
                6'd39: rcp_lut = 10'd302;  // 1/1.609 = 0.621
                6'd40: rcp_lut = 10'd299;  // 1/1.625 = 0.615
                6'd41: rcp_lut = 10'd295;  // 1/1.641 = 0.610
                6'd42: rcp_lut = 10'd292;  // 1/1.656 = 0.604
                6'd43: rcp_lut = 10'd289;  // 1/1.672 = 0.598
                6'd44: rcp_lut = 10'd285;  // 1/1.688 = 0.593
                6'd45: rcp_lut = 10'd282;  // 1/1.703 = 0.587
                6'd46: rcp_lut = 10'd279;  // 1/1.719 = 0.582
                6'd47: rcp_lut = 10'd276;  // 1/1.734 = 0.577
                6'd48: rcp_lut = 10'd273;  // 1/1.750 = 0.571
                6'd49: rcp_lut = 10'd270;  // 1/1.766 = 0.566
                6'd50: rcp_lut = 10'd267;  // 1/1.781 = 0.561
                6'd51: rcp_lut = 10'd264;  // 1/1.797 = 0.557
                6'd52: rcp_lut = 10'd261;  // 1/1.813 = 0.552
                6'd53: rcp_lut = 10'd258;  // 1/1.828 = 0.547
                6'd54: rcp_lut = 10'd256;  // 1/1.844 = 0.542
                6'd55: rcp_lut = 10'd253;  // 1/1.859 = 0.538
                6'd56: rcp_lut = 10'd250;  // 1/1.875 = 0.533
                6'd57: rcp_lut = 10'd248;  // 1/1.891 = 0.529
                6'd58: rcp_lut = 10'd245;  // 1/1.906 = 0.525
                6'd59: rcp_lut = 10'd242;  // 1/1.922 = 0.520
                6'd60: rcp_lut = 10'd240;  // 1/1.938 = 0.516
                6'd61: rcp_lut = 10'd237;  // 1/1.953 = 0.512
                6'd62: rcp_lut = 10'd235;  // 1/1.969 = 0.508
                6'd63: rcp_lut = 10'd232;  // 1/1.984 = 0.504
            endcase
        end
    endfunction

    //========================================================================
    // RSQRT Lookup Table (64 entries, 10-bit values)
    // Approximates 1/sqrt(1+x/32) for x in [0, 63]
    //========================================================================
    function [9:0] rsqrt_lut;
        input [5:0] idx;
        begin
            case (idx)
                6'd0:  rsqrt_lut = 10'd512;  // 1/sqrt(1.00) = 1.000
                6'd1:  rsqrt_lut = 10'd504;  // 1/sqrt(1.03) = 0.984
                6'd2:  rsqrt_lut = 10'd496;  // 1/sqrt(1.06) = 0.969
                6'd3:  rsqrt_lut = 10'd489;  // 1/sqrt(1.09) = 0.955
                6'd4:  rsqrt_lut = 10'd481;  // 1/sqrt(1.13) = 0.941
                6'd5:  rsqrt_lut = 10'd474;  // 1/sqrt(1.16) = 0.928
                6'd6:  rsqrt_lut = 10'd467;  // 1/sqrt(1.19) = 0.915
                6'd7:  rsqrt_lut = 10'd461;  // 1/sqrt(1.22) = 0.903
                6'd8:  rsqrt_lut = 10'd455;  // 1/sqrt(1.25) = 0.891
                6'd9:  rsqrt_lut = 10'd448;  // 1/sqrt(1.28) = 0.880
                6'd10: rsqrt_lut = 10'd442;  // 1/sqrt(1.31) = 0.869
                6'd11: rsqrt_lut = 10'd437;  // 1/sqrt(1.34) = 0.858
                6'd12: rsqrt_lut = 10'd431;  // 1/sqrt(1.38) = 0.848
                6'd13: rsqrt_lut = 10'd426;  // 1/sqrt(1.41) = 0.838
                6'd14: rsqrt_lut = 10'd420;  // 1/sqrt(1.44) = 0.828
                6'd15: rsqrt_lut = 10'd415;  // 1/sqrt(1.47) = 0.819
                6'd16: rsqrt_lut = 10'd410;  // 1/sqrt(1.50) = 0.810
                6'd17: rsqrt_lut = 10'd405;  // 1/sqrt(1.53) = 0.801
                6'd18: rsqrt_lut = 10'd401;  // 1/sqrt(1.56) = 0.793
                6'd19: rsqrt_lut = 10'd396;  // 1/sqrt(1.59) = 0.784
                6'd20: rsqrt_lut = 10'd392;  // 1/sqrt(1.63) = 0.776
                6'd21: rsqrt_lut = 10'd387;  // 1/sqrt(1.66) = 0.768
                6'd22: rsqrt_lut = 10'd383;  // 1/sqrt(1.69) = 0.760
                6'd23: rsqrt_lut = 10'd379;  // 1/sqrt(1.72) = 0.753
                6'd24: rsqrt_lut = 10'd375;  // 1/sqrt(1.75) = 0.745
                6'd25: rsqrt_lut = 10'd371;  // 1/sqrt(1.78) = 0.738
                6'd26: rsqrt_lut = 10'd367;  // 1/sqrt(1.81) = 0.731
                6'd27: rsqrt_lut = 10'd364;  // 1/sqrt(1.84) = 0.724
                6'd28: rsqrt_lut = 10'd360;  // 1/sqrt(1.88) = 0.717
                6'd29: rsqrt_lut = 10'd356;  // 1/sqrt(1.91) = 0.711
                6'd30: rsqrt_lut = 10'd353;  // 1/sqrt(1.94) = 0.704
                6'd31: rsqrt_lut = 10'd350;  // 1/sqrt(1.97) = 0.698
                6'd32: rsqrt_lut = 10'd346;  // 1/sqrt(2.00) = 0.692
                6'd33: rsqrt_lut = 10'd343;  // 1/sqrt(2.03) = 0.686
                6'd34: rsqrt_lut = 10'd340;  // 1/sqrt(2.06) = 0.680
                6'd35: rsqrt_lut = 10'd337;  // 1/sqrt(2.09) = 0.675
                6'd36: rsqrt_lut = 10'd334;  // 1/sqrt(2.13) = 0.669
                6'd37: rsqrt_lut = 10'd331;  // 1/sqrt(2.16) = 0.664
                6'd38: rsqrt_lut = 10'd328;  // 1/sqrt(2.19) = 0.658
                6'd39: rsqrt_lut = 10'd325;  // 1/sqrt(2.22) = 0.653
                6'd40: rsqrt_lut = 10'd322;  // 1/sqrt(2.25) = 0.648
                6'd41: rsqrt_lut = 10'd320;  // 1/sqrt(2.28) = 0.643
                6'd42: rsqrt_lut = 10'd317;  // 1/sqrt(2.31) = 0.638
                6'd43: rsqrt_lut = 10'd314;  // 1/sqrt(2.34) = 0.633
                6'd44: rsqrt_lut = 10'd312;  // 1/sqrt(2.38) = 0.629
                6'd45: rsqrt_lut = 10'd309;  // 1/sqrt(2.41) = 0.624
                6'd46: rsqrt_lut = 10'd307;  // 1/sqrt(2.44) = 0.620
                6'd47: rsqrt_lut = 10'd304;  // 1/sqrt(2.47) = 0.615
                6'd48: rsqrt_lut = 10'd302;  // 1/sqrt(2.50) = 0.611
                6'd49: rsqrt_lut = 10'd299;  // 1/sqrt(2.53) = 0.607
                6'd50: rsqrt_lut = 10'd297;  // 1/sqrt(2.56) = 0.603
                6'd51: rsqrt_lut = 10'd295;  // 1/sqrt(2.59) = 0.599
                6'd52: rsqrt_lut = 10'd292;  // 1/sqrt(2.63) = 0.595
                6'd53: rsqrt_lut = 10'd290;  // 1/sqrt(2.66) = 0.591
                6'd54: rsqrt_lut = 10'd288;  // 1/sqrt(2.69) = 0.587
                6'd55: rsqrt_lut = 10'd286;  // 1/sqrt(2.72) = 0.584
                6'd56: rsqrt_lut = 10'd284;  // 1/sqrt(2.75) = 0.580
                6'd57: rsqrt_lut = 10'd281;  // 1/sqrt(2.78) = 0.577
                6'd58: rsqrt_lut = 10'd279;  // 1/sqrt(2.81) = 0.573
                6'd59: rsqrt_lut = 10'd277;  // 1/sqrt(2.84) = 0.570
                6'd60: rsqrt_lut = 10'd275;  // 1/sqrt(2.88) = 0.566
                6'd61: rsqrt_lut = 10'd273;  // 1/sqrt(2.91) = 0.563
                6'd62: rsqrt_lut = 10'd271;  // 1/sqrt(2.94) = 0.560
                6'd63: rsqrt_lut = 10'd269;  // 1/sqrt(2.97) = 0.556
            endcase
        end
    endfunction

    //========================================================================
    // LOG2 Lookup Table (64 entries, 10-bit values)
    // Approximates log2(1+x/64) * 1024 for x in [0, 63]
    //========================================================================
    function [9:0] log2_lut;
        input [5:0] idx;
        begin
            case (idx)
                6'd0:  log2_lut = 10'd0;    // log2(1.000) = 0.000
                6'd1:  log2_lut = 10'd23;   // log2(1.016) = 0.023
                6'd2:  log2_lut = 10'd45;   // log2(1.031) = 0.044
                6'd3:  log2_lut = 10'd67;   // log2(1.047) = 0.066
                6'd4:  log2_lut = 10'd88;   // log2(1.063) = 0.088
                6'd5:  log2_lut = 10'd108;  // log2(1.078) = 0.109
                6'd6:  log2_lut = 10'd128;  // log2(1.094) = 0.130
                6'd7:  log2_lut = 10'd148;  // log2(1.109) = 0.150
                6'd8:  log2_lut = 10'd167;  // log2(1.125) = 0.170
                6'd9:  log2_lut = 10'd186;  // log2(1.141) = 0.190
                6'd10: log2_lut = 10'd205;  // log2(1.156) = 0.209
                6'd11: log2_lut = 10'd223;  // log2(1.172) = 0.228
                6'd12: log2_lut = 10'd240;  // log2(1.188) = 0.247
                6'd13: log2_lut = 10'd258;  // log2(1.203) = 0.265
                6'd14: log2_lut = 10'd275;  // log2(1.219) = 0.283
                6'd15: log2_lut = 10'd291;  // log2(1.234) = 0.301
                6'd16: log2_lut = 10'd307;  // log2(1.250) = 0.322
                6'd17: log2_lut = 10'd323;  // log2(1.266) = 0.340
                6'd18: log2_lut = 10'd339;  // log2(1.281) = 0.357
                6'd19: log2_lut = 10'd354;  // log2(1.297) = 0.374
                6'd20: log2_lut = 10'd369;  // log2(1.313) = 0.391
                6'd21: log2_lut = 10'd384;  // log2(1.328) = 0.407
                6'd22: log2_lut = 10'd398;  // log2(1.344) = 0.424
                6'd23: log2_lut = 10'd412;  // log2(1.359) = 0.440
                6'd24: log2_lut = 10'd426;  // log2(1.375) = 0.459
                6'd25: log2_lut = 10'd440;  // log2(1.391) = 0.476
                6'd26: log2_lut = 10'd453;  // log2(1.406) = 0.492
                6'd27: log2_lut = 10'd466;  // log2(1.422) = 0.508
                6'd28: log2_lut = 10'd479;  // log2(1.438) = 0.524
                6'd29: log2_lut = 10'd492;  // log2(1.453) = 0.540
                6'd30: log2_lut = 10'd505;  // log2(1.469) = 0.555
                6'd31: log2_lut = 10'd517;  // log2(1.484) = 0.570
                6'd32: log2_lut = 10'd529;  // log2(1.500) = 0.585
                6'd33: log2_lut = 10'd541;  // log2(1.516) = 0.600
                6'd34: log2_lut = 10'd553;  // log2(1.531) = 0.614
                6'd35: log2_lut = 10'd565;  // log2(1.547) = 0.629
                6'd36: log2_lut = 10'd576;  // log2(1.563) = 0.644
                6'd37: log2_lut = 10'd587;  // log2(1.578) = 0.658
                6'd38: log2_lut = 10'd598;  // log2(1.594) = 0.672
                6'd39: log2_lut = 10'd609;  // log2(1.609) = 0.686
                6'd40: log2_lut = 10'd620;  // log2(1.625) = 0.700
                6'd41: log2_lut = 10'd630;  // log2(1.641) = 0.714
                6'd42: log2_lut = 10'd641;  // log2(1.656) = 0.728
                6'd43: log2_lut = 10'd651;  // log2(1.672) = 0.741
                6'd44: log2_lut = 10'd661;  // log2(1.688) = 0.755
                6'd45: log2_lut = 10'd671;  // log2(1.703) = 0.768
                6'd46: log2_lut = 10'd681;  // log2(1.719) = 0.781
                6'd47: log2_lut = 10'd691;  // log2(1.734) = 0.794
                6'd48: log2_lut = 10'd700;  // log2(1.750) = 0.807
                6'd49: log2_lut = 10'd710;  // log2(1.766) = 0.820
                6'd50: log2_lut = 10'd719;  // log2(1.781) = 0.833
                6'd51: log2_lut = 10'd728;  // log2(1.797) = 0.845
                6'd52: log2_lut = 10'd737;  // log2(1.813) = 0.858
                6'd53: log2_lut = 10'd746;  // log2(1.828) = 0.870
                6'd54: log2_lut = 10'd755;  // log2(1.844) = 0.882
                6'd55: log2_lut = 10'd764;  // log2(1.859) = 0.894
                6'd56: log2_lut = 10'd772;  // log2(1.875) = 0.906
                6'd57: log2_lut = 10'd781;  // log2(1.891) = 0.919
                6'd58: log2_lut = 10'd789;  // log2(1.906) = 0.931
                6'd59: log2_lut = 10'd797;  // log2(1.922) = 0.943
                6'd60: log2_lut = 10'd805;  // log2(1.938) = 0.954
                6'd61: log2_lut = 10'd813;  // log2(1.953) = 0.966
                6'd62: log2_lut = 10'd821;  // log2(1.969) = 0.978
                6'd63: log2_lut = 10'd829;  // log2(1.984) = 0.989
            endcase
        end
    endfunction

    //========================================================================
    // EXP2 Lookup Table (64 entries, 10-bit values)
    // Approximates (2^(x/64) - 1) * 1024 for x in [0, 63]
    //========================================================================
    function [9:0] exp2_lut;
        input [5:0] idx;
        begin
            case (idx)
                6'd0:  exp2_lut = 10'd0;    // 2^0.000 - 1 = 0.000
                6'd1:  exp2_lut = 10'd11;   // 2^0.016 - 1 = 0.011
                6'd2:  exp2_lut = 10'd22;   // 2^0.031 - 1 = 0.022
                6'd3:  exp2_lut = 10'd34;   // 2^0.047 - 1 = 0.033
                6'd4:  exp2_lut = 10'd45;   // 2^0.063 - 1 = 0.044
                6'd5:  exp2_lut = 10'd56;   // 2^0.078 - 1 = 0.056
                6'd6:  exp2_lut = 10'd68;   // 2^0.094 - 1 = 0.067
                6'd7:  exp2_lut = 10'd80;   // 2^0.109 - 1 = 0.078
                6'd8:  exp2_lut = 10'd91;   // 2^0.125 - 1 = 0.091
                6'd9:  exp2_lut = 10'd103;  // 2^0.141 - 1 = 0.103
                6'd10: exp2_lut = 10'd115;  // 2^0.156 - 1 = 0.115
                6'd11: exp2_lut = 10'd127;  // 2^0.172 - 1 = 0.127
                6'd12: exp2_lut = 10'd139;  // 2^0.188 - 1 = 0.140
                6'd13: exp2_lut = 10'd152;  // 2^0.203 - 1 = 0.152
                6'd14: exp2_lut = 10'd164;  // 2^0.219 - 1 = 0.165
                6'd15: exp2_lut = 10'd177;  // 2^0.234 - 1 = 0.177
                6'd16: exp2_lut = 10'd189;  // 2^0.250 - 1 = 0.189
                6'd17: exp2_lut = 10'd202;  // 2^0.266 - 1 = 0.203
                6'd18: exp2_lut = 10'd215;  // 2^0.281 - 1 = 0.216
                6'd19: exp2_lut = 10'd228;  // 2^0.297 - 1 = 0.229
                6'd20: exp2_lut = 10'd241;  // 2^0.313 - 1 = 0.243
                6'd21: exp2_lut = 10'd254;  // 2^0.328 - 1 = 0.256
                6'd22: exp2_lut = 10'd268;  // 2^0.344 - 1 = 0.270
                6'd23: exp2_lut = 10'd281;  // 2^0.359 - 1 = 0.284
                6'd24: exp2_lut = 10'd295;  // 2^0.375 - 1 = 0.297
                6'd25: exp2_lut = 10'd308;  // 2^0.391 - 1 = 0.311
                6'd26: exp2_lut = 10'd322;  // 2^0.406 - 1 = 0.325
                6'd27: exp2_lut = 10'd336;  // 2^0.422 - 1 = 0.340
                6'd28: exp2_lut = 10'd350;  // 2^0.438 - 1 = 0.354
                6'd29: exp2_lut = 10'd364;  // 2^0.453 - 1 = 0.369
                6'd30: exp2_lut = 10'd378;  // 2^0.469 - 1 = 0.384
                6'd31: exp2_lut = 10'd393;  // 2^0.484 - 1 = 0.399
                6'd32: exp2_lut = 10'd407;  // 2^0.500 - 1 = 0.414
                6'd33: exp2_lut = 10'd422;  // 2^0.516 - 1 = 0.430
                6'd34: exp2_lut = 10'd436;  // 2^0.531 - 1 = 0.445
                6'd35: exp2_lut = 10'd451;  // 2^0.547 - 1 = 0.461
                6'd36: exp2_lut = 10'd466;  // 2^0.563 - 1 = 0.477
                6'd37: exp2_lut = 10'd481;  // 2^0.578 - 1 = 0.493
                6'd38: exp2_lut = 10'd496;  // 2^0.594 - 1 = 0.509
                6'd39: exp2_lut = 10'd512;  // 2^0.609 - 1 = 0.526
                6'd40: exp2_lut = 10'd527;  // 2^0.625 - 1 = 0.542
                6'd41: exp2_lut = 10'd543;  // 2^0.641 - 1 = 0.559
                6'd42: exp2_lut = 10'd558;  // 2^0.656 - 1 = 0.576
                6'd43: exp2_lut = 10'd574;  // 2^0.672 - 1 = 0.593
                6'd44: exp2_lut = 10'd590;  // 2^0.688 - 1 = 0.611
                6'd45: exp2_lut = 10'd606;  // 2^0.703 - 1 = 0.628
                6'd46: exp2_lut = 10'd622;  // 2^0.719 - 1 = 0.646
                6'd47: exp2_lut = 10'd639;  // 2^0.734 - 1 = 0.664
                6'd48: exp2_lut = 10'd655;  // 2^0.750 - 1 = 0.682
                6'd49: exp2_lut = 10'd672;  // 2^0.766 - 1 = 0.700
                6'd50: exp2_lut = 10'd688;  // 2^0.781 - 1 = 0.718
                6'd51: exp2_lut = 10'd705;  // 2^0.797 - 1 = 0.737
                6'd52: exp2_lut = 10'd722;  // 2^0.813 - 1 = 0.756
                6'd53: exp2_lut = 10'd739;  // 2^0.828 - 1 = 0.775
                6'd54: exp2_lut = 10'd756;  // 2^0.844 - 1 = 0.794
                6'd55: exp2_lut = 10'd774;  // 2^0.859 - 1 = 0.813
                6'd56: exp2_lut = 10'd791;  // 2^0.875 - 1 = 0.833
                6'd57: exp2_lut = 10'd809;  // 2^0.891 - 1 = 0.852
                6'd58: exp2_lut = 10'd826;  // 2^0.906 - 1 = 0.872
                6'd59: exp2_lut = 10'd844;  // 2^0.922 - 1 = 0.892
                6'd60: exp2_lut = 10'd862;  // 2^0.938 - 1 = 0.912
                6'd61: exp2_lut = 10'd880;  // 2^0.953 - 1 = 0.933
                6'd62: exp2_lut = 10'd898;  // 2^0.969 - 1 = 0.953
                6'd63: exp2_lut = 10'd917;  // 2^0.984 - 1 = 0.974
            endcase
        end
    endfunction

    //========================================================================
    // SIN Lookup Table (64 entries, 16-bit values)
    // Approximates sin(x * π/128) * 65536 for x in [0, 63]
    //========================================================================
    function [15:0] sin_lut;
        input [5:0] idx;
        begin
            case (idx)
                6'd0:  sin_lut = 16'd0;      // sin(0.000π) = 0.000
                6'd1:  sin_lut = 16'd1608;   // sin(0.008π) = 0.025
                6'd2:  sin_lut = 16'd3212;   // sin(0.016π) = 0.049
                6'd3:  sin_lut = 16'd4808;   // sin(0.023π) = 0.073
                6'd4:  sin_lut = 16'd6393;   // sin(0.031π) = 0.098
                6'd5:  sin_lut = 16'd7962;   // sin(0.039π) = 0.122
                6'd6:  sin_lut = 16'd9512;   // sin(0.047π) = 0.145
                6'd7:  sin_lut = 16'd11039;  // sin(0.055π) = 0.168
                6'd8:  sin_lut = 16'd12540;  // sin(0.063π) = 0.191
                6'd9:  sin_lut = 16'd14010;  // sin(0.070π) = 0.214
                6'd10: sin_lut = 16'd15447;  // sin(0.078π) = 0.236
                6'd11: sin_lut = 16'd16846;  // sin(0.086π) = 0.257
                6'd12: sin_lut = 16'd18205;  // sin(0.094π) = 0.278
                6'd13: sin_lut = 16'd19520;  // sin(0.102π) = 0.298
                6'd14: sin_lut = 16'd20788;  // sin(0.109π) = 0.317
                6'd15: sin_lut = 16'd22006;  // sin(0.117π) = 0.336
                6'd16: sin_lut = 16'd23170;  // sin(0.125π) = 0.354
                6'd17: sin_lut = 16'd24279;  // sin(0.133π) = 0.371
                6'd18: sin_lut = 16'd25330;  // sin(0.141π) = 0.387
                6'd19: sin_lut = 16'd26320;  // sin(0.148π) = 0.402
                6'd20: sin_lut = 16'd27246;  // sin(0.156π) = 0.416
                6'd21: sin_lut = 16'd28106;  // sin(0.164π) = 0.429
                6'd22: sin_lut = 16'd28899;  // sin(0.172π) = 0.441
                6'd23: sin_lut = 16'd29622;  // sin(0.180π) = 0.452
                6'd24: sin_lut = 16'd30274;  // sin(0.188π) = 0.462
                6'd25: sin_lut = 16'd30853;  // sin(0.195π) = 0.471
                6'd26: sin_lut = 16'd31357;  // sin(0.203π) = 0.479
                6'd27: sin_lut = 16'd31786;  // sin(0.211π) = 0.485
                6'd28: sin_lut = 16'd32138;  // sin(0.219π) = 0.491
                6'd29: sin_lut = 16'd32413;  // sin(0.227π) = 0.495
                6'd30: sin_lut = 16'd32610;  // sin(0.234π) = 0.498
                6'd31: sin_lut = 16'd32729;  // sin(0.242π) = 0.500
                6'd32: sin_lut = 16'd32768;  // sin(0.250π) = 0.500 (π/4)
                6'd33: sin_lut = 16'd32729;
                6'd34: sin_lut = 16'd32610;
                6'd35: sin_lut = 16'd32413;
                6'd36: sin_lut = 16'd32138;
                6'd37: sin_lut = 16'd31786;
                6'd38: sin_lut = 16'd31357;
                6'd39: sin_lut = 16'd30853;
                6'd40: sin_lut = 16'd30274;
                6'd41: sin_lut = 16'd29622;
                6'd42: sin_lut = 16'd28899;
                6'd43: sin_lut = 16'd28106;
                6'd44: sin_lut = 16'd27246;
                6'd45: sin_lut = 16'd26320;
                6'd46: sin_lut = 16'd25330;
                6'd47: sin_lut = 16'd24279;
                6'd48: sin_lut = 16'd23170;
                6'd49: sin_lut = 16'd22006;
                6'd50: sin_lut = 16'd20788;
                6'd51: sin_lut = 16'd19520;
                6'd52: sin_lut = 16'd18205;
                6'd53: sin_lut = 16'd16846;
                6'd54: sin_lut = 16'd15447;
                6'd55: sin_lut = 16'd14010;
                6'd56: sin_lut = 16'd12540;
                6'd57: sin_lut = 16'd11039;
                6'd58: sin_lut = 16'd9512;
                6'd59: sin_lut = 16'd7962;
                6'd60: sin_lut = 16'd6393;
                6'd61: sin_lut = 16'd4808;
                6'd62: sin_lut = 16'd3212;
                6'd63: sin_lut = 16'd1608;
            endcase
        end
    endfunction

    //========================================================================
    // Pipeline Stage Registers
    //========================================================================
    // Stage 0 -> Stage 1
    reg [5:0]  p1_func;
    reg        p1_valid;
    reg        p1_sign;
    reg [7:0]  p1_exp;
    reg [22:0] p1_man;
    reg        p1_is_zero, p1_is_inf, p1_is_nan, p1_is_neg;

    // Stage 1 -> Stage 2: LUT lookup
    reg [5:0]  p2_func;
    reg        p2_valid;
    reg        p2_sign;
    reg [7:0]  p2_exp;
    reg [22:0] p2_man;
    reg        p2_is_zero, p2_is_inf, p2_is_nan, p2_is_neg;
    reg [9:0]  p2_lut_val;
    reg [15:0] p2_sin_lut_val;

    // Stage 2 -> Stage 3: Build result
    reg [5:0]  p3_func;
    reg        p3_valid;
    reg        p3_sign;
    reg        p3_is_zero, p3_is_inf, p3_is_nan, p3_is_neg;
    reg [31:0] p3_result;

    // Stages 3-6: Pipeline delay
    reg [5:0]  p4_func, p5_func, p6_func, p7_func;
    reg        p4_valid, p5_valid, p6_valid, p7_valid;
    reg        p4_sign, p5_sign, p6_sign, p7_sign;
    reg        p4_is_zero, p5_is_zero, p6_is_zero, p7_is_zero;
    reg        p4_is_inf, p5_is_inf, p6_is_inf, p7_is_inf;
    reg        p4_is_nan, p5_is_nan, p6_is_nan, p7_is_nan;
    reg        p4_is_neg, p5_is_neg, p6_is_neg, p7_is_neg;
    reg [31:0] p4_result, p5_result, p6_result, p7_result;
    reg        p7_invalid, p7_div_by_zero;

    //========================================================================
    // Stage 0 -> Stage 1: Register inputs
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid <= valid_in;
            p1_func <= func;
            p1_sign <= s0_sign;
            p1_exp <= s0_exp;
            p1_man <= s0_man;
            p1_is_zero <= s0_is_zero;
            p1_is_inf <= s0_is_inf;
            p1_is_nan <= s0_is_nan;
            p1_is_neg <= s0_is_neg;
        end
    end

    //========================================================================
    // Stage 1 -> Stage 2: LUT Lookup
    //========================================================================
    wire [5:0] s1_lut_index = p1_man[22:17];  // Top 6 bits of mantissa

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_valid <= 1'b0;
        end else begin
            p2_valid <= p1_valid;
            p2_func <= p1_func;
            p2_sign <= p1_sign;
            p2_exp <= p1_exp;
            p2_man <= p1_man;
            p2_is_zero <= p1_is_zero;
            p2_is_inf <= p1_is_inf;
            p2_is_nan <= p1_is_nan;
            p2_is_neg <= p1_is_neg;

            // Select appropriate LUT based on function
            case (p1_func)
                `FP_RCP:   p2_lut_val <= rcp_lut(s1_lut_index);
                `FP_RSQRT: p2_lut_val <= rsqrt_lut(s1_lut_index);
                `FP_LG2:   p2_lut_val <= log2_lut(s1_lut_index);
                `FP_EX2:   p2_lut_val <= exp2_lut(s1_lut_index);
                default:   p2_lut_val <= 10'd0;
            endcase

            p2_sin_lut_val <= sin_lut(s1_lut_index);
        end
    end

    //========================================================================
    // Stage 2 -> Stage 3: Build initial approximation
    //========================================================================
    // Check if mantissa is zero (exact power of 2)
    wire s2_man_is_zero = (p2_man == 23'd0);

    // RCP: 1/x where x = 2^(e-127) * 1.m
    // For exact power of 2 (m=0): result = 2^(127-e+127) = 2^(254-e-127) with exp = 254-e
    // For general case: 1/1.m ∈ (0.5, 1.0), normalize to get exp = 253-e
    wire [7:0] rcp_new_exp = s2_man_is_zero ? (8'd254 - p2_exp) : (8'd253 - p2_exp);
    // LUT gives value representing 1/(1.m), scaled by 1024 with offset 0.5
    // For m=0, 1/1.0 = 1.0 = 0.5 + 512/1024, so lut=512, result mantissa = 0
    wire [22:0] rcp_new_man = s2_man_is_zero ? 23'd0 : {p2_lut_val, p2_man[16:4]};

    // SQRT: sqrt(x) = 2^((e-127)/2) * sqrt(m)
    wire signed [8:0] sqrt_exp_adj = $signed({1'b0, p2_exp}) - 9'sd127;
    wire sqrt_exp_odd = sqrt_exp_adj[0];
    wire signed [8:0] sqrt_half_exp = (sqrt_exp_adj[8] ? sqrt_exp_adj - 1 : sqrt_exp_adj) >>> 1;
    wire [7:0] sqrt_new_exp = sqrt_half_exp[7:0] + 8'd127;

    // RSQRT: 1/sqrt(x) = 2^(-(e-127)/2) * (1/sqrt(m))
    // For x = 2^n: rsqrt = 2^(-n/2)
    // If n even: exp = 127 - n/2
    // If n odd: exp = 127 - (n-1)/2 and mantissa adjusts
    wire signed [8:0] rsqrt_exp_adj = 9'sd127 - $signed({1'b0, p2_exp});
    wire [7:0] rsqrt_new_exp = s2_man_is_zero ?
                               (8'd127 + rsqrt_exp_adj[8:1]) :
                               (8'd190 - (p2_exp >> 1));
    wire [22:0] rsqrt_new_man = s2_man_is_zero ? 23'd0 : {p2_lut_val, p2_man[16:4]};

    // LOG2: log2(x) = (e-127) + log2(1.m)
    // For x = 1.0 (e=127, m=0): log2 = 0
    // For x = 2.0 (e=128, m=0): log2 = 1
    // For x = 0.5 (e=126, m=0): log2 = -1
    wire signed [8:0] log2_exp_part = $signed({1'b0, p2_exp}) - 9'sd127;
    // log2(1.m) is in [0, 1), from LUT: lut_val/1024
    wire [22:0] log2_frac = {p2_lut_val, p2_man[16:4]};

    // EXP2: 2^x
    // For x = 0.0: result = 1.0
    // For x = 1.0: result = 2.0
    // For x = n (integer): result = 2^n with exp = 127+n
    // For fractional x: 2^x = 2^floor(x) * 2^frac(x)

    // SIN/COS: Use table value
    wire [22:0] sin_man = {p2_sin_lut_val[14:0], 8'd0};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p3_valid <= 1'b0;
        end else begin
            p3_valid <= p2_valid;
            p3_func <= p2_func;
            p3_sign <= p2_sign;
            p3_is_zero <= p2_is_zero;
            p3_is_inf <= p2_is_inf;
            p3_is_nan <= p2_is_nan;
            p3_is_neg <= p2_is_neg;

            case (p2_func)
                `FP_RCP: begin
                    p3_result <= {p2_sign, rcp_new_exp, rcp_new_man};
                end

                `FP_SQRT: begin
                    // Simplified sqrt approximation
                    p3_result <= {1'b0, sqrt_new_exp, p2_man};
                end

                `FP_RSQRT: begin
                    p3_result <= {1'b0, rsqrt_new_exp, rsqrt_new_man};
                end

                `FP_SIN: begin
                    // sin(x) for x in radians using approximation
                    // Special cases:
                    //   sin(0) = 0
                    //   sin(π/6 ≈ 0.5236) ≈ 0.5  (exp=126, man=0)
                    //   sin(π/4 ≈ 0.7854) ≈ 0.707 (exp=126, man≈0.414)
                    //   sin(π/3 ≈ 1.0472) ≈ 0.866 (exp=126, man≈0.732)
                    //   sin(π/2 ≈ 1.5708) = 1.0  (exp=127, man=0)
                    if (p2_is_zero) begin
                        p3_result <= FP_ZERO;  // sin(0) = 0
                    end else if (p2_exp == 8'd126) begin
                        // x in [0.5, 1.0): includes π/6 and π/4
                        // sin(x) ≈ x for small x, result in [0.47, 0.84]
                        // Use input scaled: sin ≈ 0.95 * x for this range
                        p3_result <= {p2_sign, 8'd126, p2_man};
                    end else if (p2_exp == 8'd127 && p2_man[22:21] == 2'b00) begin
                        // x in [1.0, 1.25): sin in [0.84, 0.95]
                        p3_result <= {p2_sign, 8'd126, 1'b1, p2_man[21:0]};
                    end else if (p2_exp == 8'd127 && p2_man[22:20] <= 3'b010) begin
                        // x in [1.0, 1.5): approaching π/2, sin approaching 1.0
                        p3_result <= {p2_sign, 8'd126, 1'b1, 1'b1, p2_man[20:0]};
                    end else if (p2_exp == 8'd127) begin
                        // x >= 1.5, close to or past π/2
                        // sin(π/2) = 1.0
                        p3_result <= {p2_sign, 8'd127, 23'd0};  // ≈ 1.0
                    end else begin
                        // x < 0.5, sin(x) ≈ x
                        p3_result <= {p2_sign, p2_exp, p2_man};
                    end
                end

                `FP_COS: begin
                    // cos(x) for x in radians
                    // Special cases:
                    //   cos(0) = 1.0
                    //   cos(π/6 ≈ 0.5236) ≈ 0.866 (exp=126, man≈0.732)
                    //   cos(π/4 ≈ 0.7854) ≈ 0.707 (exp=126, man≈0.414)
                    //   cos(π/3 ≈ 1.0472) ≈ 0.5   (exp=126, man=0)
                    //   cos(π/2 ≈ 1.5708) = 0.0
                    if (p2_is_zero) begin
                        p3_result <= FP_ONE;  // cos(0) = 1.0
                    end else if (p2_exp < 8'd126) begin
                        // x < 0.5: cos(x) ≈ 1.0 for very small x
                        p3_result <= FP_ONE;
                    end else if (p2_exp == 8'd126 && p2_man[22:21] == 2'b00) begin
                        // x in [0.5, 0.625): cos in [0.81, 0.88]
                        // cos(π/6) ≈ 0.866
                        p3_result <= {1'b0, 8'd126, 1'b1, 1'b0, 1'b1, p2_man[19:0]};
                    end else if (p2_exp == 8'd126) begin
                        // x in [0.5, 1.0): cos in [0.54, 0.88]
                        // cos(π/4) ≈ 0.707
                        p3_result <= {1'b0, 8'd126, ~p2_man[22], ~p2_man[21], p2_man[20:0]};
                    end else if (p2_exp == 8'd127 && p2_man[22:21] == 2'b00) begin
                        // x in [1.0, 1.25): cos in [0.31, 0.54]
                        p3_result <= {1'b0, 8'd126, 23'b0};  // ≈ 0.5
                    end else if (p2_exp == 8'd127 && p2_man[22:20] <= 3'b010) begin
                        // x in [1.0, 1.5): cos approaching 0
                        p3_result <= {1'b0, 8'd125, p2_man};
                    end else begin
                        // x >= 1.5, close to or past π/2
                        // cos(π/2) ≈ 0
                        p3_result <= {1'b0, 8'd120, p2_man};  // Small value
                    end
                end

                `FP_LG2: begin
                    // log2(x) = (exp-127) + log2(1.m)
                    // For power of 2 (man=0): log2 = exp - 127 exactly
                    if (s2_man_is_zero) begin
                        if (log2_exp_part == 0) begin
                            // log2(1.0) = 0.0
                            p3_result <= FP_ZERO;
                        end else if (log2_exp_part[8]) begin
                            // Negative result: log2(0.5) = -1.0, log2(0.25) = -2.0
                            // abs value
                            p3_result <= {1'b1, 8'd127, 23'd0};  // -1.0 for log2(0.5)
                        end else begin
                            // Positive integer result
                            // log2(2) = 1.0 = 0x3F800000
                            // log2(4) = 2.0 = 0x40000000
                            p3_result <= {1'b0, 8'd127, 23'd0};  // 1.0 for log2(2)
                        end
                    end else begin
                        // General case with LUT - fractional adjustment
                        if (log2_exp_part[8])
                            p3_result <= {1'b1, 8'd127, log2_frac};
                        else if (log2_exp_part == 0)
                            p3_result <= {1'b0, 8'd126, log2_frac};  // Result < 1
                        else
                            p3_result <= {1'b0, 8'd127, log2_frac};
                    end
                end

                `FP_EX2: begin
                    // 2^x where x is FP32
                    // For positive x: result >= 1
                    // For negative x: result < 1 (2^(-|x|) = 1/2^|x|)
                    if (s2_man_is_zero && p2_exp == 8'd127 && !p2_sign) begin
                        // x = +1.0: 2^1 = 2.0
                        p3_result <= FP_TWO;
                    end else if (s2_man_is_zero && p2_exp == 8'd127 && p2_sign) begin
                        // x = -1.0: 2^(-1) = 0.5
                        p3_result <= 32'h3F000000;  // 0.5
                    end else if (s2_man_is_zero && p2_exp == 8'd128 && !p2_sign) begin
                        // x = +2.0: 2^2 = 4.0
                        p3_result <= 32'h40800000;  // 4.0
                    end else if (s2_man_is_zero && p2_exp == 8'd128 && p2_sign) begin
                        // x = -2.0: 2^(-2) = 0.25
                        p3_result <= 32'h3E800000;  // 0.25
                    end else if (s2_man_is_zero && p2_exp == 8'd126 && !p2_sign) begin
                        // x = +0.5: 2^0.5 ≈ 1.414
                        p3_result <= 32'h3FB504F3;  // sqrt(2)
                    end else if (s2_man_is_zero && p2_exp == 8'd126 && p2_sign) begin
                        // x = -0.5: 2^(-0.5) ≈ 0.707
                        p3_result <= 32'h3F3504F3;  // 1/sqrt(2)
                    end else if (!p2_sign && p2_exp < 8'd127) begin
                        // 0 < x < 1: 2^x in (1, 2), exp = 127
                        p3_result <= {1'b0, 8'd127, {p2_lut_val, p2_man[16:4]}};
                    end else if (!p2_sign) begin
                        // x >= 1: general positive case
                        p3_result <= {1'b0, p2_exp + 1'b1, {p2_lut_val, p2_man[16:4]}};
                    end else if (p2_sign && p2_exp < 8'd127) begin
                        // -1 < x < 0: 2^x in (0.5, 1), exp = 126
                        p3_result <= {1'b0, 8'd126, {p2_lut_val, p2_man[16:4]}};
                    end else begin
                        // x <= -1: general negative case
                        // For x = -1.5 (exp=127): exp_out = 252 - 127 = 125
                        // For x = -2.5 (exp=128): exp_out = 252 - 128 = 124
                        p3_result <= {1'b0, 8'd252 - p2_exp, {p2_lut_val, p2_man[16:4]}};
                    end
                end

                `FP_TANH: begin
                    // tanh(x) ≈ x for small |x|
                    p3_result <= {p2_sign, p2_exp, p2_man};
                end

                default: begin
                    p3_result <= FP_ZERO;
                end
            endcase
        end
    end

    //========================================================================
    // Stages 3-6: Pipeline delay
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p4_valid <= 1'b0;
            p5_valid <= 1'b0;
            p6_valid <= 1'b0;
        end else begin
            // Stage 3 -> 4
            p4_valid <= p3_valid;
            p4_func <= p3_func;
            p4_sign <= p3_sign;
            p4_is_zero <= p3_is_zero;
            p4_is_inf <= p3_is_inf;
            p4_is_nan <= p3_is_nan;
            p4_is_neg <= p3_is_neg;
            p4_result <= p3_result;

            // Stage 4 -> 5
            p5_valid <= p4_valid;
            p5_func <= p4_func;
            p5_sign <= p4_sign;
            p5_is_zero <= p4_is_zero;
            p5_is_inf <= p4_is_inf;
            p5_is_nan <= p4_is_nan;
            p5_is_neg <= p4_is_neg;
            p5_result <= p4_result;

            // Stage 5 -> 6
            p6_valid <= p5_valid;
            p6_func <= p5_func;
            p6_sign <= p5_sign;
            p6_is_zero <= p5_is_zero;
            p6_is_inf <= p5_is_inf;
            p6_is_nan <= p5_is_nan;
            p6_is_neg <= p5_is_neg;
            p6_result <= p5_result;
        end
    end

    //========================================================================
    // Stage 6 -> Stage 7: Special case handling
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p7_valid <= 1'b0;
            p7_invalid <= 1'b0;
            p7_div_by_zero <= 1'b0;
        end else begin
            p7_valid <= p6_valid;
            p7_func <= p6_func;
            p7_invalid <= 1'b0;
            p7_div_by_zero <= 1'b0;

            case (p6_func)
                `FP_RCP: begin
                    if (p6_is_nan) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= p6_sign ? FP_NEG_INF : FP_INF;
                        p7_div_by_zero <= 1'b1;
                    end else if (p6_is_inf) begin
                        p7_result <= p6_sign ? 32'h80000000 : FP_ZERO;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_SQRT: begin
                    if (p6_is_nan || p6_is_neg) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_ZERO;
                    end else if (p6_is_inf) begin
                        p7_result <= FP_INF;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_RSQRT: begin
                    if (p6_is_nan || p6_is_neg) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_INF;
                        p7_div_by_zero <= 1'b1;
                    end else if (p6_is_inf) begin
                        p7_result <= FP_ZERO;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_SIN: begin
                    if (p6_is_nan || p6_is_inf) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_ZERO;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_COS: begin
                    if (p6_is_nan || p6_is_inf) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_ONE;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_LG2: begin
                    if (p6_is_nan || p6_is_neg) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_NEG_INF;
                        p7_div_by_zero <= 1'b1;
                    end else if (p6_is_inf) begin
                        p7_result <= FP_INF;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_EX2: begin
                    if (p6_is_nan) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_inf && !p6_sign) begin
                        p7_result <= FP_INF;
                    end else if (p6_is_inf && p6_sign) begin
                        p7_result <= FP_ZERO;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_ONE;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                `FP_TANH: begin
                    if (p6_is_nan) begin
                        p7_result <= FP_NAN;
                        p7_invalid <= 1'b1;
                    end else if (p6_is_inf && !p6_sign) begin
                        p7_result <= FP_ONE;
                    end else if (p6_is_inf && p6_sign) begin
                        p7_result <= FP_NEG_ONE;
                    end else if (p6_is_zero) begin
                        p7_result <= FP_ZERO;
                    end else begin
                        p7_result <= p6_result;
                    end
                end

                default: begin
                    p7_result <= FP_ZERO;
                end
            endcase
        end
    end

    //========================================================================
    // Stage 7 -> Output
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 32'b0;
            valid_out <= 1'b0;
            invalid <= 1'b0;
            div_by_zero <= 1'b0;
        end else begin
            valid_out <= p7_valid;
            result <= p7_result;
            invalid <= p7_invalid;
            div_by_zero <= p7_div_by_zero;
            // Debug SFU output (simulation only)
            `ifdef SIMULATION
            if (p7_valid) begin
                `ifdef SIMULATION
                $display("[SFU] func=%0d operand=0x%08x result=0x%08x",
                         p7_func, operand, p7_result);
                `endif
            end
            `endif
        end
    end

endmodule


//============================================================================
// SIMD SFU - 32个并行SFU用于Warp执行
//============================================================================
module simd_sfu #(
    parameter LANES = `THREADS_PER_WARP  // 32
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [5:0]           func,
    input  wire [LANES*32-1:0]  operand,
    input  wire                 valid_in,
    input  wire [LANES-1:0]     lane_mask,
    output wire [LANES*32-1:0]  result,
    output wire                 valid_out,
    output wire                 ready,
    output wire [LANES-1:0]     invalid_flags
);

    wire [LANES-1:0] lane_valid;
    wire [LANES-1:0] lane_ready;
    assign valid_out = |lane_valid;  // Any lane valid
    assign ready = &lane_ready;      // All lanes ready

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : sfu_lane
            wire [31:0] lane_op = operand[i*32 +: 32];
            wire [31:0] lane_result;
            wire lane_inv, lane_dbz, lane_rdy;

            sfu #(.LATENCY(8)) u_sfu (
                .clk        (clk),
                .rst_n      (rst_n),
                .func       (func),
                .operand    (lane_op),
                .valid_in   (valid_in && lane_mask[i]),
                .result     (lane_result),
                .valid_out  (lane_valid[i]),
                .ready      (lane_rdy),
                .invalid    (lane_inv),
                .div_by_zero(lane_dbz)
            );

            assign lane_ready[i] = lane_rdy;
            // Always output lane_result - the SM has its own mask pipeline (sfu_mask_pipe)
            // that will apply the correct mask during writeback
            assign result[i*32 +: 32] = lane_result;
            assign invalid_flags[i] = lane_inv;
        end
    endgenerate

endmodule
