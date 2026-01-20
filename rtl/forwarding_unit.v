//============================================================================
// RalphGPU - Data Forwarding Unit
// 数据转发网络，消除流水线RAW依赖导致的stall
// 支持从EX和MEM阶段转发到ID/EX阶段
//============================================================================

`timescale 1ns / 1ps

module forwarding_unit #(
    parameter THREADS = 32,
    parameter DATA_WIDTH = 32,
    parameter REG_ADDR_WIDTH = 5
)(
    input  wire                         clk,
    input  wire                         rst_n,

    //------------------------------------------------------------------------
    // 当前指令信息 (来自ID阶段)
    //------------------------------------------------------------------------
    input  wire [REG_ADDR_WIDTH-1:0]    id_ra,          // 源操作数A寄存器地址
    input  wire [REG_ADDR_WIDTH-1:0]    id_rb,          // 源操作数B寄存器地址
    input  wire [REG_ADDR_WIDTH-1:0]    id_rc,          // 源操作数C寄存器地址 (FMA)
    input  wire                         id_use_ra,      // 使用ra
    input  wire                         id_use_rb,      // 使用rb
    input  wire                         id_use_rc,      // 使用rc

    //------------------------------------------------------------------------
    // EX阶段信息
    //------------------------------------------------------------------------
    input  wire [REG_ADDR_WIDTH-1:0]    ex_rd,          // EX阶段目标寄存器
    input  wire                         ex_reg_write,   // EX阶段写寄存器
    input  wire [DATA_WIDTH-1:0]        ex_result [0:THREADS-1],  // EX阶段结果

    //------------------------------------------------------------------------
    // MEM阶段信息
    //------------------------------------------------------------------------
    input  wire [REG_ADDR_WIDTH-1:0]    mem_rd,         // MEM阶段目标寄存器
    input  wire                         mem_reg_write,  // MEM阶段写寄存器
    input  wire [DATA_WIDTH-1:0]        mem_result [0:THREADS-1], // MEM阶段结果

    //------------------------------------------------------------------------
    // WB阶段信息 (用于MEM load完成后)
    //------------------------------------------------------------------------
    input  wire [REG_ADDR_WIDTH-1:0]    wb_rd,          // WB阶段目标寄存器
    input  wire                         wb_reg_write,   // WB阶段写寄存器
    input  wire [DATA_WIDTH-1:0]        wb_result [0:THREADS-1],  // WB阶段结果

    //------------------------------------------------------------------------
    // 转发控制输出
    //------------------------------------------------------------------------
    output reg  [1:0]                   forward_a,      // 00=无转发, 01=EX, 10=MEM, 11=WB
    output reg  [1:0]                   forward_b,
    output reg  [1:0]                   forward_c,

    //------------------------------------------------------------------------
    // 转发数据输出
    //------------------------------------------------------------------------
    output reg  [DATA_WIDTH-1:0]        forwarded_a [0:THREADS-1],
    output reg  [DATA_WIDTH-1:0]        forwarded_b [0:THREADS-1],
    output reg  [DATA_WIDTH-1:0]        forwarded_c [0:THREADS-1],

    //------------------------------------------------------------------------
    // Hazard检测
    //------------------------------------------------------------------------
    output reg                          load_use_hazard, // Load-use需要stall

    //------------------------------------------------------------------------
    // Load-Use Hazard检测输入
    //------------------------------------------------------------------------
    input  wire                         ex_is_load       // EX阶段是load指令
);

    //------------------------------------------------------------------------
    // 转发逻辑 - 操作数A
    //------------------------------------------------------------------------
    integer i;

    always @(*) begin
        forward_a = 2'b00;  // 默认无转发

        if (id_use_ra && id_ra != 0) begin  // r0通常是零寄存器，不转发
            // 优先级: EX > MEM > WB (最新的数据优先)
            if (ex_reg_write && ex_rd == id_ra) begin
                forward_a = 2'b01;  // 从EX转发
            end else if (mem_reg_write && mem_rd == id_ra) begin
                forward_a = 2'b10;  // 从MEM转发
            end else if (wb_reg_write && wb_rd == id_ra) begin
                forward_a = 2'b11;  // 从WB转发
            end
        end
    end

    //------------------------------------------------------------------------
    // 转发逻辑 - 操作数B
    //------------------------------------------------------------------------
    always @(*) begin
        forward_b = 2'b00;

        if (id_use_rb && id_rb != 0) begin
            if (ex_reg_write && ex_rd == id_rb) begin
                forward_b = 2'b01;
            end else if (mem_reg_write && mem_rd == id_rb) begin
                forward_b = 2'b10;
            end else if (wb_reg_write && wb_rd == id_rb) begin
                forward_b = 2'b11;
            end
        end
    end

    //------------------------------------------------------------------------
    // 转发逻辑 - 操作数C (用于FMA)
    //------------------------------------------------------------------------
    always @(*) begin
        forward_c = 2'b00;

        if (id_use_rc && id_rc != 0) begin
            if (ex_reg_write && ex_rd == id_rc) begin
                forward_c = 2'b01;
            end else if (mem_reg_write && mem_rd == id_rc) begin
                forward_c = 2'b10;
            end else if (wb_reg_write && wb_rd == id_rc) begin
                forward_c = 2'b11;
            end
        end
    end

    //------------------------------------------------------------------------
    // 转发数据选择
    //------------------------------------------------------------------------
    always @(*) begin
        // 操作数A
        for (i = 0; i < THREADS; i = i + 1) begin
            case (forward_a)
                2'b01:   forwarded_a[i] = ex_result[i];
                2'b10:   forwarded_a[i] = mem_result[i];
                2'b11:   forwarded_a[i] = wb_result[i];
                default: forwarded_a[i] = 0;  // 无转发，使用寄存器文件
            endcase
        end

        // 操作数B
        for (i = 0; i < THREADS; i = i + 1) begin
            case (forward_b)
                2'b01:   forwarded_b[i] = ex_result[i];
                2'b10:   forwarded_b[i] = mem_result[i];
                2'b11:   forwarded_b[i] = wb_result[i];
                default: forwarded_b[i] = 0;
            endcase
        end

        // 操作数C
        for (i = 0; i < THREADS; i = i + 1) begin
            case (forward_c)
                2'b01:   forwarded_c[i] = ex_result[i];
                2'b10:   forwarded_c[i] = mem_result[i];
                2'b11:   forwarded_c[i] = wb_result[i];
                default: forwarded_c[i] = 0;
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Load-Use Hazard检测
    // 当前EX阶段是load，且下一条指令需要使用其结果
    // 这种情况无法转发，必须stall一个周期
    //------------------------------------------------------------------------
    always @(*) begin
        load_use_hazard = 0;

        if (ex_is_load && ex_reg_write && ex_rd != 0) begin
            if ((id_use_ra && id_ra == ex_rd) ||
                (id_use_rb && id_rb == ex_rd) ||
                (id_use_rc && id_rc == ex_rd)) begin
                load_use_hazard = 1;
            end
        end
    end

endmodule


//============================================================================
// 记分板 - 跟踪寄存器依赖
//============================================================================
module scoreboard #(
    parameter NUM_REGS = 32
)(
    input  wire         clk,
    input  wire         rst_n,

    // 指令发射 (标记寄存器为忙)
    input  wire         issue_valid,
    input  wire [4:0]   issue_rd,
    input  wire         issue_reg_write,

    // 指令完成 (标记寄存器为空闲)
    input  wire         complete_valid,
    input  wire [4:0]   complete_rd,

    // 依赖检查
    input  wire [4:0]   check_ra,
    input  wire [4:0]   check_rb,
    input  wire [4:0]   check_rc,
    output wire         ra_busy,
    output wire         rb_busy,
    output wire         rc_busy,
    output wire         any_hazard
);

    // 每个寄存器一个busy位
    reg [NUM_REGS-1:0] busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
        end else begin
            // 发射时设置busy
            if (issue_valid && issue_reg_write && issue_rd != 0) begin
                busy[issue_rd] <= 1;
            end

            // 完成时清除busy
            if (complete_valid && complete_rd != 0) begin
                busy[complete_rd] <= 0;
            end
        end
    end

    // 检查依赖
    assign ra_busy = (check_ra != 0) && busy[check_ra];
    assign rb_busy = (check_rb != 0) && busy[check_rb];
    assign rc_busy = (check_rc != 0) && busy[check_rc];
    assign any_hazard = ra_busy || rb_busy || rc_busy;

endmodule
