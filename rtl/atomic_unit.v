//============================================================================
// RalphGPU - Atomic Unit
// 原子操作单元: add, min, max, inc, dec, and, or, xor, exch, cas
// 支持全局内存和共享内存的原子操作
//============================================================================

`timescale 1ns / 1ps
`include "gpu_defines.vh"

module atomic_unit #(
    parameter NUM_LANES = `THREADS_PER_WARP
) (
    input  wire        clk,
    input  wire        rst_n,

    // 操作请求
    input  wire        req_valid,
    output wire        req_ready,
    input  wire [5:0]  func,           // 原子操作类型
    input  wire [NUM_LANES*32-1:0] addr,       // 内存地址 (每lane)
    input  wire [NUM_LANES*32-1:0] operand_a,  // 操作数A (每lane)
    input  wire [NUM_LANES*32-1:0] operand_b,  // 操作数B (每lane, CAS比较值)
    input  wire [NUM_LANES-1:0]    lane_mask,  // 活跃线程掩码
    input  wire        mem_shared,     // 1=共享内存, 0=全局内存

    // 内存接口 (请求)
    output reg         mem_req,
    output reg         mem_write,
    output reg  [31:0] mem_addr,
    output reg  [31:0] mem_wdata,
    output reg  [5:0]  mem_lane,

    // 内存接口 (响应)
    input  wire        resp_read_valid,
    input  wire        resp_write_valid,
    input  wire [NUM_LANES*32-1:0] mem_rdata,

    // 结果
    output reg  [NUM_LANES*32-1:0] result,  // 原始值(交换前)
    output reg  [NUM_LANES-1:0] result_mask,
    output reg         result_valid,
    output reg         busy
);

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    localparam IDLE      = 3'd0;
    localparam READ_REQ  = 3'd1;
    localparam READ_WAIT = 3'd2;
    localparam COMPUTE   = 3'd3;
    localparam WRITE_REQ = 3'd4;
    localparam WRITE_WAIT= 3'd5;
    localparam DONE      = 3'd6;

    reg [2:0] state;

    // 操作存储
    reg [5:0]  func_reg;
    reg [NUM_LANES*32-1:0] addr_reg;
    reg [NUM_LANES*32-1:0] operand_a_reg;
    reg [NUM_LANES*32-1:0] operand_b_reg;
    reg [NUM_LANES-1:0]    pending_mask;
    reg [5:0]  current_lane;
    reg [31:0] old_value;
    reg [31:0] new_value;
    reg [2:0]  dbg_prev_state;
    reg [15:0] dbg_wait_cycles;

    //------------------------------------------------------------------------
    // 原子操作计算
    //------------------------------------------------------------------------
    // 注意: 该单元只在同一warp内序列化lane访问。
    // 不提供跨warp/跨SM的全局原子性保证，依赖外部内存系统对同地址原子访问进行严格序列化。
    // req_ready仅在IDLE时拉高，防止busy期间丢失请求。
    assign req_ready = (state == IDLE);
    wire signed [31:0] signed_old = old_value;

    function [31:0] lane_slice32;
        input [NUM_LANES*32-1:0] vec;
        input [5:0] lane;
        integer i;
        begin
            lane_slice32 = 32'b0;
            for (i = 0; i < NUM_LANES; i = i + 1) begin
                if (lane == i[5:0]) begin
                    lane_slice32 = vec[i*32 +: 32];
                end
            end
        end
    endfunction

    function [5:0] find_next_lane;
        input [5:0] start;
        input [NUM_LANES-1:0] mask;
        integer j;
        reg found;
        reg [4:0] idx;
        begin
            find_next_lane = start;
            found = 1'b0;
            for (j = 0; j < NUM_LANES; j = j + 1) begin
                idx = start[4:0] + j[4:0];
                if (!found && mask[idx]) begin
                    find_next_lane = {1'b0, idx};
                    found = 1'b1;
                end
            end
        end
    endfunction

    function [NUM_LANES-1:0] lane_onehot;
        input [5:0] lane;
        integer k;
        begin
            lane_onehot = {NUM_LANES{1'b0}};
            for (k = 0; k < NUM_LANES; k = k + 1) begin
                if (lane == k[5:0]) begin
                    lane_onehot[k] = 1'b1;
                end
            end
        end
    endfunction

    wire [31:0] addr_lane = lane_slice32(addr_reg, current_lane);
    wire [31:0] operand_a_lane = lane_slice32(operand_a_reg, current_lane);
    wire [31:0] operand_b_lane = lane_slice32(operand_b_reg, current_lane);
    wire [31:0] mem_rdata_lane = lane_slice32(mem_rdata, current_lane);
    wire signed [31:0] signed_a = operand_a_lane;

    always @(*) begin
        case (func_reg)
            `ATOM_ADD: new_value = old_value + operand_a_lane;

            `ATOM_MIN_S: new_value = (signed_old < signed_a) ? old_value : operand_a_lane;
            `ATOM_MIN_U: new_value = (old_value < operand_a_lane) ? old_value : operand_a_lane;

            `ATOM_MAX_S: new_value = (signed_old > signed_a) ? old_value : operand_a_lane;
            `ATOM_MAX_U: new_value = (old_value > operand_a_lane) ? old_value : operand_a_lane;

            // inc(r, s) = (r >= s) ? 0 : r+1
            `ATOM_INC: new_value = (old_value >= operand_a_lane) ? 32'h0 : (old_value + 1);

            // dec(r, s) = (r == 0 || r > s) ? s : r-1
            `ATOM_DEC: new_value = (old_value == 0 || old_value > operand_a_lane) ?
                                   operand_a_lane : (old_value - 1);

            `ATOM_AND:  new_value = old_value & operand_a_lane;
            `ATOM_OR:   new_value = old_value | operand_a_lane;
            `ATOM_XOR:  new_value = old_value ^ operand_a_lane;

            `ATOM_EXCH: new_value = operand_a_lane;

            // cas(r, s, t) = (r == s) ? t : r
            `ATOM_CAS: new_value = (old_value == operand_a_lane) ? operand_b_lane : old_value;

            default: new_value = old_value;
        endcase
    end

    //------------------------------------------------------------------------
    // 状态机
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= IDLE;
            busy         <= 1'b0;
            result_valid <= 1'b0;
            result       <= {NUM_LANES{32'b0}};
            result_mask  <= {NUM_LANES{1'b0}};
            mem_req      <= 1'b0;
            mem_write    <= 1'b0;
            mem_addr     <= 32'b0;
            mem_wdata    <= 32'b0;
            mem_lane     <= 6'b0;
            pending_mask <= {NUM_LANES{1'b0}};
            current_lane <= 6'b0;
            dbg_prev_state <= IDLE;
            dbg_wait_cycles <= 16'd0;
        end else begin
            // DEBUG TRACE: log transitions + wait handshakes to pinpoint stalls.
            `ifdef SIMULATION
            if (state != dbg_prev_state) begin
                $display("[%0t ATOMIC_FSM] %0d -> %0d lane=%0d pending=0x%08x req_valid=%b busy=%b rvalid=%b wvalid=%b", // keep
                         $time, dbg_prev_state, state, current_lane, pending_mask,
                         req_valid, busy, resp_read_valid, resp_write_valid);
                dbg_prev_state  <= state;
                dbg_wait_cycles <= 16'd0;
            end else if (state == READ_WAIT || state == WRITE_WAIT) begin
                dbg_wait_cycles <= dbg_wait_cycles + 16'd1;

                if (state == READ_WAIT) begin
                    if (resp_read_valid) begin
                        $display("[%0t ATOMIC_READ] lane=%0d addr=0x%08x data=0x%08x wait=%0d", // keep
                                 $time, current_lane, addr_lane, mem_rdata_lane, dbg_wait_cycles);
                    end else if ((dbg_wait_cycles & 16'h003f) == 16'h003f) begin
                        $display("[%0t ATOMIC_READ_WAIT] lane=%0d addr=0x%08x mem_req=%b mem_write=%b wait=%0d rvalid=%b", // keep
                                 $time, current_lane, addr_lane, mem_req, mem_write, dbg_wait_cycles, resp_read_valid);
                    end
                end else begin
                    if (resp_write_valid) begin
                        $display("[%0t ATOMIC_WRITE_ACK] lane=%0d addr=0x%08x old=0x%08x new=0x%08x wait=%0d", // keep
                                 $time, current_lane, addr_lane, old_value, new_value, dbg_wait_cycles);
                    end else if ((dbg_wait_cycles & 16'h003f) == 16'h003f) begin
                        $display("[%0t ATOMIC_WRITE_WAIT] lane=%0d addr=0x%08x mem_req=%b mem_write=%b wait=%0d wvalid=%b", // keep
                                 $time, current_lane, addr_lane, mem_req, mem_write, dbg_wait_cycles, resp_write_valid);
                    end
                end
            end
            `endif

            case (state)
                IDLE: begin
                    result_valid <= 1'b0;
                    if (req_valid) begin
                        `ifdef SIMULATION
                        $display("[%0t ATOMIC_REQ] func=%0d lane_mask=0x%08x mem_shared=%b", $time, func, lane_mask, mem_shared); // keep
                        `endif
                        // 保存操作参数
                        func_reg      <= func;
                        addr_reg      <= addr;
                        operand_a_reg <= operand_a;
                        operand_b_reg <= operand_b;
                        pending_mask  <= lane_mask;
                        result        <= {NUM_LANES{32'b0}};
                        result_mask   <= lane_mask;
                        if (lane_mask != 0) begin
                            current_lane <= find_next_lane(0, lane_mask);
                            busy         <= 1'b1;
                            state        <= READ_REQ;
                        end else begin
                            busy         <= 1'b0;
                            state        <= DONE;
                        end
                    end
                end

                READ_REQ: begin
                    // 发起读请求
                    mem_req   <= 1'b1;
                    mem_write <= 1'b0;
                    mem_addr  <= addr_lane;
                    mem_lane  <= current_lane;
                    state     <= READ_WAIT;
                end

                READ_WAIT: begin
                    if (resp_read_valid) begin
                        old_value <= mem_rdata_lane;
                        mem_req   <= 1'b0;
                        state     <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // new_value已由组合逻辑计算
                    // 检查CAS是否需要写入
                    if (func_reg == `ATOM_CAS && old_value != operand_a_lane) begin
                        // CAS比较失败，不写入
                        begin : cas_fail
                            integer i;
                            for (i = 0; i < NUM_LANES; i = i + 1) begin
                                if (current_lane == i[5:0]) begin
                                    result[i*32 +: 32] <= old_value;
                                end
                            end
                        end
                        pending_mask[current_lane[4:0]] <= 1'b0;
                        if ((pending_mask & ~lane_onehot(current_lane)) != 0) begin
                            current_lane <= find_next_lane(current_lane + 1'b1,
                                                           pending_mask & ~lane_onehot(current_lane));
                            state <= READ_REQ;
                        end else begin
                            state <= DONE;
                        end
                    end else begin
                        state <= WRITE_REQ;
                    end
                end

                WRITE_REQ: begin
                    // 发起写请求
                    mem_req   <= 1'b1;
                    mem_write <= 1'b1;
                    mem_addr  <= addr_lane;
                    mem_wdata <= new_value;
                    mem_lane  <= current_lane;
                    state     <= WRITE_WAIT;
                end

                WRITE_WAIT: begin
                    if (resp_write_valid) begin
                        `ifdef SIMULATION
                        $display("[%0t ATOMIC] Lane %0d WRITE done: addr=0x%08x old=0x%08x new=0x%08x", // keep
                                 $time, current_lane, addr_lane, old_value, new_value);
                        `endif
                        mem_req <= 1'b0;
                        begin : write_done
                            integer i;
                            for (i = 0; i < NUM_LANES; i = i + 1) begin
                                if (current_lane == i[5:0]) begin
                                    result[i*32 +: 32] <= old_value;
                                end
                            end
                        end
                        pending_mask[current_lane[4:0]] <= 1'b0;
                        if ((pending_mask & ~lane_onehot(current_lane)) != 0) begin
                            current_lane <= find_next_lane(current_lane + 1'b1,
                                                           pending_mask & ~lane_onehot(current_lane));
                            state <= READ_REQ;
                        end else begin
                            state <= DONE;
                        end
                    end
                end

                DONE: begin
                    result_valid <= 1'b1;
                    busy         <= 1'b0;
                    state        <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// Reduction Unit
// 归约操作单元 (不返回旧值，只更新内存)
//============================================================================
module reduction_unit (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        req_valid,
    input  wire [5:0]  func,
    input  wire [31:0] addr,
    input  wire [31:0] operand,
    input  wire        mem_shared,

    output reg         mem_req,
    output reg         mem_write,
    output reg  [31:0] mem_addr,
    output reg  [31:0] mem_wdata,

    input  wire        mem_ready,
    input  wire [31:0] mem_rdata,

    output reg         done,
    output reg         busy
);

    // 状态机
    localparam IDLE      = 2'd0;
    localparam READ_REQ  = 2'd1;
    localparam READ_WAIT = 2'd2;
    localparam WRITE     = 2'd3;

    reg [1:0] state;
    reg [5:0]  func_reg;
    reg [31:0] addr_reg;
    reg [31:0] operand_reg;
    reg [31:0] old_value;

    // 归约计算
    wire signed [31:0] signed_old = old_value;
    wire signed [31:0] signed_op  = operand_reg;
    reg [31:0] new_value;

    always @(*) begin
        case (func_reg)
            `ATOM_ADD:   new_value = old_value + operand_reg;
            `ATOM_MIN_S: new_value = (signed_old < signed_op) ? old_value : operand_reg;
            `ATOM_MIN_U: new_value = (old_value < operand_reg) ? old_value : operand_reg;
            `ATOM_MAX_S: new_value = (signed_old > signed_op) ? old_value : operand_reg;
            `ATOM_MAX_U: new_value = (old_value > operand_reg) ? old_value : operand_reg;
            `ATOM_AND:   new_value = old_value & operand_reg;
            `ATOM_OR:    new_value = old_value | operand_reg;
            `ATOM_XOR:   new_value = old_value ^ operand_reg;
            default:     new_value = old_value;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            mem_req   <= 1'b0;
            mem_write <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (req_valid) begin
                        func_reg    <= func;
                        addr_reg    <= addr;
                        operand_reg <= operand;
                        busy        <= 1'b1;
                        mem_req     <= 1'b1;
                        mem_write   <= 1'b0;
                        mem_addr    <= addr;
                        state       <= READ_WAIT;
                    end
                end

                READ_WAIT: begin
                    if (mem_ready) begin
                        old_value <= mem_rdata;
                        mem_req   <= 1'b0;
                        state     <= WRITE;
                    end
                end

                WRITE: begin
                    mem_req   <= 1'b1;
                    mem_write <= 1'b1;
                    mem_addr  <= addr_reg;
                    mem_wdata <= new_value;
                    if (mem_ready) begin
                        mem_req <= 1'b0;
                        done    <= 1'b1;
                        busy    <= 1'b0;
                        state   <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule


//============================================================================
// Memory Barrier Unit
// 内存屏障单元
//============================================================================
module membar_unit (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        req_valid,
    input  wire [1:0]  scope,         // 00=CTA, 01=GL, 10=SYS
    input  wire        all_stores_complete,
    input  wire        all_loads_complete,

    output reg         done,
    output reg         stall_pipeline
);

    localparam IDLE     = 2'd0;
    localparam WAIT_ST  = 2'd1;
    localparam WAIT_LD  = 2'd2;
    localparam COMPLETE = 2'd3;

    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            done           <= 1'b0;
            stall_pipeline <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (req_valid) begin
                        stall_pipeline <= 1'b1;
                        state          <= WAIT_ST;
                    end
                end

                WAIT_ST: begin
                    if (all_stores_complete) begin
                        state <= WAIT_LD;
                    end
                end

                WAIT_LD: begin
                    if (all_loads_complete) begin
                        state <= COMPLETE;
                    end
                end

                COMPLETE: begin
                    stall_pipeline <= 1'b0;
                    done           <= 1'b1;
                    state          <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
