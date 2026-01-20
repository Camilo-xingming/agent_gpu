//============================================================================
// RalphGPU - Instruction Decoder
// 解码PTX风格的32位指令
// 支持完整PTX ISA 8.5+指令集
//============================================================================

`include "gpu_defines.vh"

module decoder (
    input  wire        clk,
    input  wire        rst_n,

    // 指令输入
    input  wire [31:0] instruction,
    input  wire        valid_in,

    // 解码结果
    output reg         valid_out,
    output reg  [5:0]  opcode,
    output reg  [4:0]  rd,          // 目标寄存器
    output reg  [4:0]  ra,          // 源寄存器A
    output reg  [4:0]  rb,          // 源寄存器B
    output reg  [4:0]  rc,          // 源寄存器C / 谓词
    output reg  [5:0]  func,        // 功能码
    output reg  [15:0] imm16,       // 16位立即数
    output reg  [20:0] imm21,       // 21位立即数(分支偏移)

    // 控制信号 - 基础
    output reg         use_imm,     // 使用立即数
    output reg         alu_op,      // ALU操作
    output reg         mul_op,      // 乘法操作
    output reg         div_op,      // 除法操作
    output reg         mem_read,    // 内存读
    output reg         mem_write,   // 内存写
    output reg         mem_shared,  // 共享内存访问
    output reg         branch_op,   // 分支操作
    output reg         sync_op,     // 同步操作
    output reg         special_reg, // 特殊寄存器访问
    output reg         exit_op,     // EXIT/RET操作
    output reg         reg_write,   // 需要写寄存器
    output reg         pred_write,  // 需要写谓词
    output reg  [2:0]  pred_addr,   // 谓词寄存器地址

    // 控制信号 - FP操作
    output reg         fp32_op,     // FP32算术操作
    output reg         fp32_special,// FP32特殊函数
    output reg         fp64_op,     // FP64操作
    output reg         fp16_op,     // FP16/BF16操作
    output reg         cvt_op,      // 类型转换

    // 控制信号 - 扩展内存
    output reg         mem_param,   // 参数内存
    output reg         mem_const,   // 常量内存
    output reg         mem_local,   // 本地内存
    output reg         mem_vector,  // 向量内存访问
    output reg  [1:0]  vec_size,    // 向量大小: 00=1, 01=2, 10=4

    // 控制信号 - 原子操作
    output reg         atomic_op,   // 原子操作
    output reg         reduce_op,   // 归约操作

    // 控制信号 - Warp操作
    output reg         shfl_op,     // Warp shuffle
    output reg         vote_op,     // Warp vote
    output reg         redux_op,    // Warp reduction

    // 控制信号 - Tensor Core
    output reg         wmma_load,   // WMMA加载
    output reg         wmma_store,  // WMMA存储
    output reg         wmma_mma,    // WMMA MMA
    output reg         mma_op,      // MMA指令

    // 控制信号 - 其他
    output reg         call_op,     // 函数调用
    output reg         membar_op,   // 内存屏障

    // 控制信号 - Video
    output reg         video_op,    // Video处理

    // 控制信号 - Texture/Surface
    output reg         tex_op,      // 纹理采样
    output reg         txq_op,      // 纹理查询
    output reg         surf_ld,     // Surface加载
    output reg         surf_st,     // Surface存储
    output reg         surf_red,    // Surface归约

    // 控制信号 - Phase 10: 扩展操作
    output reg         cpasync_op,  // cp.async操作
    output reg         prefetch_op, // prefetch操作
    output reg         wgmma_load,  // WGMMA加载
    output reg         wgmma_store, // WGMMA存储
    output reg         wgmma_mma,   // WGMMA MMA
    output reg  [2:0]  cache_hint,  // 缓存提示

    // 控制信号 - mbarrier (Hopper+)
    output reg         mbarrier_op, // mbarrier操作

    // 控制信号 - Warp-level barrier
    output reg         bar_warp_sync,  // bar.warp.sync (warp-level synchronization)

    // 控制信号 - Cache Policy (Hopper+)
    output reg         cache_policy_op, // Cache policy operations (createpolicy/applypriority/discard)

    // 控制信号 - Stack/Debug/Misc (Phase 6.2)
    output reg         stack_op,        // Stack operations (alloca/stacksave/stackrestore)
    output reg         debug_op,        // Debug operations (brkpt/trap/pmevent)
    output reg         misc_op,         // Misc operations (nanosleep/setmaxnreg)

    // 控制信号 - Async Store/Multimem (Phase 1.2)
    output reg         st_async_op,     // st.async operations
    output reg         multimem_op,     // multimem operations (distributed shared memory)

    // 控制信号 - Barrier Cluster (Phase 3.2)
    output reg         barrier_cluster_op,  // barrier.cluster operations (cross-SM sync)

    // 控制信号 - Warp Collective (Phase 5.1)
    output reg         match_sync_op,       // match.sync operations (warp-level predicate matching)
    output reg         elect_sync_op,       // elect.sync operations (warp-level leader election)
    output reg         red_async_op,        // red.async operations (async reduction to shared memory)

    // 控制信号 - DPX/Sparse (Phase 5.2/5.3)
    output reg         dpx_op,              // DPX operations (dynamic programming)
    output reg         sparse_mma_op,       // Sparse MMA operations (2:4 sparsity)

    // 控制信号 - Address Space/Cache (Phase 4.2)
    output reg         isspacep_op,         // isspacep operations (address space query)
    output reg         mapa_op,             // mapa operations (address mapping)
    output reg         getctarank_op        // getctarank operations (CTA rank in cluster)
);

    //------------------------------------------------------------------------
    // 指令字段提取 (组合逻辑)
    //------------------------------------------------------------------------
    wire [5:0]  inst_opcode = instruction[`INST_OPCODE];
    wire [4:0]  inst_rd     = instruction[`INST_RD];
    wire [4:0]  inst_ra     = instruction[`INST_RA];
    wire [4:0]  inst_rb     = instruction[`INST_RB];
    wire [4:0]  inst_rc     = instruction[`INST_RC];
    wire [5:0]  inst_func   = instruction[`INST_FUNC];
    wire [15:0] inst_imm16  = instruction[`INST_IMM16];
    wire [20:0] inst_imm21  = instruction[`INST_IMM21];

    //------------------------------------------------------------------------
    // 解码逻辑
    //------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            opcode      <= 6'b0;
            rd          <= 5'b0;
            ra          <= 5'b0;
            rb          <= 5'b0;
            rc          <= 5'b0;
            func        <= 6'b0;
            imm16       <= 16'b0;
            imm21       <= 21'b0;
            use_imm     <= 1'b0;
            alu_op      <= 1'b0;
            mul_op      <= 1'b0;
            div_op      <= 1'b0;
            mem_read    <= 1'b0;
            mem_write   <= 1'b0;
            mem_shared  <= 1'b0;
            branch_op   <= 1'b0;
            sync_op     <= 1'b0;
            special_reg <= 1'b0;
            exit_op     <= 1'b0;
            reg_write   <= 1'b0;
            pred_write  <= 1'b0;
            pred_addr   <= 3'b0;
            // 新增控制信号复位
            fp32_op     <= 1'b0;
            fp32_special<= 1'b0;
            fp64_op     <= 1'b0;
            fp16_op     <= 1'b0;
            cvt_op      <= 1'b0;
            mem_param   <= 1'b0;
            mem_const   <= 1'b0;
            mem_local   <= 1'b0;
            mem_vector  <= 1'b0;
            vec_size    <= 2'b0;
            atomic_op   <= 1'b0;
            reduce_op   <= 1'b0;
            shfl_op     <= 1'b0;
            vote_op     <= 1'b0;
            redux_op    <= 1'b0;
            wmma_load   <= 1'b0;
            wmma_store  <= 1'b0;
            wmma_mma    <= 1'b0;
            mma_op      <= 1'b0;
            call_op     <= 1'b0;
            membar_op   <= 1'b0;
            video_op    <= 1'b0;
            tex_op      <= 1'b0;
            txq_op      <= 1'b0;
            surf_ld     <= 1'b0;
            surf_st     <= 1'b0;
            surf_red    <= 1'b0;
            cpasync_op  <= 1'b0;
            prefetch_op <= 1'b0;
            wgmma_load  <= 1'b0;
            wgmma_store <= 1'b0;
            wgmma_mma   <= 1'b0;
            cache_hint  <= 3'b0;
            mbarrier_op <= 1'b0;
            bar_warp_sync <= 1'b0;
            cache_policy_op <= 1'b0;
            stack_op <= 1'b0;
            debug_op <= 1'b0;
            misc_op <= 1'b0;
            st_async_op <= 1'b0;
            multimem_op <= 1'b0;
            barrier_cluster_op <= 1'b0;
            match_sync_op <= 1'b0;
            elect_sync_op <= 1'b0;
            red_async_op <= 1'b0;
            dpx_op <= 1'b0;
            sparse_mma_op <= 1'b0;
            isspacep_op <= 1'b0;
            mapa_op <= 1'b0;
            getctarank_op <= 1'b0;
        end else if (valid_in) begin
            valid_out <= 1'b1;

            // 基本字段
            opcode <= inst_opcode;
            rd     <= inst_rd;
            ra     <= inst_ra;
            rb     <= inst_rb;
            rc     <= inst_rc;
            func   <= inst_func;
            imm16  <= inst_imm16;
            imm21  <= inst_imm21;

            // 默认控制信号
            use_imm     <= 1'b0;
            alu_op      <= 1'b0;
            mul_op      <= 1'b0;
            div_op      <= 1'b0;
            mem_read    <= 1'b0;
            mem_write   <= 1'b0;
            mem_shared  <= 1'b0;
            branch_op   <= 1'b0;
            sync_op     <= 1'b0;
            special_reg <= 1'b0;
            exit_op     <= 1'b0;
            reg_write   <= 1'b0;
            pred_write  <= 1'b0;
            pred_addr   <= inst_rc[2:0];
            // 新增控制信号默认值
            fp32_op     <= 1'b0;
            fp32_special<= 1'b0;
            fp64_op     <= 1'b0;
            fp16_op     <= 1'b0;
            cvt_op      <= 1'b0;
            mem_param   <= 1'b0;
            mem_const   <= 1'b0;
            mem_local   <= 1'b0;
            mem_vector  <= 1'b0;
            vec_size    <= 2'b0;
            atomic_op   <= 1'b0;
            reduce_op   <= 1'b0;
            shfl_op     <= 1'b0;
            vote_op     <= 1'b0;
            redux_op    <= 1'b0;
            wmma_load   <= 1'b0;
            wmma_store  <= 1'b0;
            wmma_mma    <= 1'b0;
            mma_op      <= 1'b0;
            call_op     <= 1'b0;
            membar_op   <= 1'b0;
            video_op    <= 1'b0;
            tex_op      <= 1'b0;
            txq_op      <= 1'b0;
            surf_ld     <= 1'b0;
            surf_st     <= 1'b0;
            surf_red    <= 1'b0;
            cpasync_op  <= 1'b0;
            prefetch_op <= 1'b0;
            wgmma_load  <= 1'b0;
            wgmma_store <= 1'b0;
            wgmma_mma   <= 1'b0;
            cache_hint  <= 3'b0;
            mbarrier_op <= 1'b0;
            bar_warp_sync <= 1'b0;
            cache_policy_op <= 1'b0;
            stack_op <= 1'b0;
            debug_op <= 1'b0;
            misc_op <= 1'b0;
            st_async_op <= 1'b0;
            multimem_op <= 1'b0;
            barrier_cluster_op <= 1'b0;
            match_sync_op <= 1'b0;
            elect_sync_op <= 1'b0;
            red_async_op <= 1'b0;
            dpx_op <= 1'b0;
            sparse_mma_op <= 1'b0;
            isspacep_op <= 1'b0;
            mapa_op <= 1'b0;
            getctarank_op <= 1'b0;

            // 根据OPCODE设置控制信号
            case (inst_opcode)
                `OP_ALU: begin
                    alu_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_ALU_IMM: begin
                    // ALU with 16-bit immediate: rd = ra op imm16
                    // Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:10]=func, [9:0]=imm10
                    alu_op    <= 1'b1;
                    use_imm   <= 1'b1;
                    reg_write <= 1'b1;
                    func      <= inst_imm16[15:10];  // func from upper bits of imm16
                    imm16     <= {6'b0, inst_imm16[9:0]};  // 10-bit immediate
                end

                `OP_MUL: begin
                    mul_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_DIV: begin
                    div_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SETP: begin
                    alu_op     <= 1'b1;  // 比较用ALU
                    pred_write <= 1'b1;
                end

                `OP_BRANCH: begin
                    branch_op <= 1'b1;
                end

                `OP_LD_GLOBAL: begin
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_ST_GLOBAL: begin
                    mem_write <= 1'b1;
                    `ifdef SIMULATION
                    $display("[DECODER] ST_GLOBAL: inst=0x%08h ra=R%0d rb=R%0d", instruction, ra, rb);
                    `endif
                end

                `OP_LD_SHARED: begin
                    mem_read   <= 1'b1;
                    mem_shared <= 1'b1;
                    reg_write  <= 1'b1;
                end

                `OP_ST_SHARED: begin
                    mem_write  <= 1'b1;
                    mem_shared <= 1'b1;
                end

                `OP_MOV_SPECIAL: begin
                    special_reg <= 1'b1;
                    reg_write   <= 1'b1;
                    `ifdef SIMULATION
                    $display("[DECODER] MOV_SPECIAL detected: inst=0x%08h rd=R%0d sreg=%0d", instruction, rd, ra);
                    `endif
                end

                `OP_BAR_SYNC: begin
                    sync_op <= 1'b1;
                end

                `OP_BAR_WARP_SYNC: begin
                    // bar.warp.sync membermask - warp-level synchronization
                    // membermask comes from ra (register A) - 32-bit value specifying participating threads
                    bar_warp_sync <= 1'b1;
                    sync_op <= 1'b1;  // Also set sync_op to trigger stall handling
                end

                `OP_EXIT, `OP_RET: begin
                    exit_op <= 1'b1;
                end

                //============================================================
                // Phase 2: 浮点运算指令
                //============================================================
                `OP_FP32_ARITH: begin
                    fp32_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_FP32_SPECIAL: begin
                    fp32_special <= 1'b1;
                    reg_write    <= 1'b1;
                end

                `OP_FP64_ARITH: begin
                    fp64_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_FP16_ARITH: begin
                    fp16_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_CVT: begin
                    cvt_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 3: 扩展内存操作
                //============================================================
                `OP_LD_PARAM: begin
                    mem_read  <= 1'b1;
                    mem_param <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_LD_CONST: begin
                    mem_read  <= 1'b1;
                    mem_const <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_LD_LOCAL: begin
                    mem_read  <= 1'b1;
                    mem_local <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_ST_LOCAL: begin
                    mem_write <= 1'b1;
                    mem_local <= 1'b1;
                end

                `OP_LD_V2: begin
                    mem_read   <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b01;  // v2
                    reg_write  <= 1'b1;
                end

                `OP_LD_V4: begin
                    mem_read   <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b10;  // v4
                    reg_write  <= 1'b1;
                end

                `OP_ST_V2: begin
                    mem_write  <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b01;
                end

                `OP_ST_V4: begin
                    mem_write  <= 1'b1;
                    mem_vector <= 1'b1;
                    vec_size   <= 2'b10;
                end

                //============================================================
                // Phase 4: 原子操作
                //============================================================
                `OP_ATOM: begin
                    atomic_op <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                    reg_write <= 1'b1;  // 返回旧值
                end

                `OP_RED: begin
                    reduce_op <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                end

                //============================================================
                // Phase 5: Warp级操作
                //============================================================
                `OP_SHFL: begin
                    shfl_op   <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_VOTE: begin
                    vote_op    <= 1'b1;
                    reg_write  <= 1'b1;
                    pred_write <= 1'b1;
                end

                `OP_REDUX: begin
                    redux_op  <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 6: Tensor Core
                //============================================================
                `OP_WMMA_LOAD: begin
                    wmma_load <= 1'b1;
                    mem_read  <= 1'b1;
                end

                `OP_WMMA_STORE: begin
                    wmma_store <= 1'b1;
                    mem_write  <= 1'b1;
                end

                `OP_WMMA_MMA: begin
                    wmma_mma  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_MMA: begin
                    mma_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 7: 控制流扩展
                //============================================================
                `OP_CALL: begin
                    call_op   <= 1'b1;
                    branch_op <= 1'b1;
                end

                `OP_MEMBAR: begin
                    membar_op <= 1'b1;
                    sync_op   <= 1'b1;
                end

                //============================================================
                // Phase 8: Video处理指令
                //============================================================
                `OP_VIDEO: begin
                    video_op  <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 9: 纹理/Surface指令
                //============================================================
                `OP_TEX: begin
                    tex_op    <= 1'b1;
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_TXQ: begin
                    txq_op    <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SULD: begin
                    surf_ld   <= 1'b1;
                    mem_read  <= 1'b1;
                    reg_write <= 1'b1;
                end

                `OP_SUST: begin
                    surf_st   <= 1'b1;
                    mem_write <= 1'b1;
                end

                `OP_SURED: begin
                    surf_red  <= 1'b1;
                    mem_read  <= 1'b1;
                    mem_write <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 10: 扩展操作
                //============================================================
                `OP_CPASYNC: begin
                    // cp.async handled by async engine; do not route through LSU datapath
                    cpasync_op <= 1'b1;
                    cache_hint <= inst_func[2:0];
                end

                `OP_PREFETCH: begin
                    // Prefetch is fire-and-forget hint; no LSU action here
                    prefetch_op <= 1'b1;
                    cache_hint  <= inst_func[2:0];
                end

                `OP_WGMMA_LOAD: begin
                    wgmma_load <= 1'b1;
                    mem_read   <= 1'b1;
                    mem_shared <= 1'b1;
                end

                `OP_WGMMA_STORE: begin
                    wgmma_store <= 1'b1;
                    mem_write   <= 1'b1;
                    mem_shared  <= 1'b1;
                end

                `OP_WGMMA_MMA: begin
                    wgmma_mma <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // mbarrier (Hopper+ Memory Barrier)
                //============================================================
                `OP_MBARRIER: begin
                    mbarrier_op <= 1'b1;
                    // test_wait and try_wait return results to register
                    // func[2] distinguishes wait ops (MBAR_TEST_WAIT=4, MBAR_TRY_WAIT=5)
                    if (inst_func == `MBAR_TEST_WAIT || inst_func == `MBAR_TRY_WAIT) begin
                        reg_write <= 1'b1;
                    end
                end

                //============================================================
                // Cache Policy Instructions (Hopper+)
                //============================================================
                `OP_CACHE_POLICY: begin
                    cache_policy_op <= 1'b1;
                    // createpolicy returns a policy token (writes to register)
                    // applypriority and discard do not return values
                    if (inst_func == `CACHE_CREATEPOLICY) begin
                        reg_write <= 1'b1;
                    end
                end

                //============================================================
                // Stack Operations (Phase 6.2)
                //============================================================
                `OP_STACK: begin
                    stack_op <= 1'b1;
                    case (inst_func)
                        `STACK_ALLOCA: begin
                            // alloca rd, size - allocate stack space, return pointer in rd
                            reg_write <= 1'b1;
                        end
                        `STACK_SAVE: begin
                            // stacksave rd - save current stack pointer to rd
                            reg_write <= 1'b1;
                        end
                        `STACK_RESTORE: begin
                            // stackrestore ra - restore stack pointer from ra
                            // No register write - just updates internal stack pointer
                        end
                        default: begin
                            // Unknown stack operation - treat as NOP
                        end
                    endcase
                end

                //============================================================
                // Debug Operations (Phase 6.2)
                //============================================================
                `OP_DEBUG: begin
                    debug_op <= 1'b1;
                    // Debug operations don't write to registers
                    // brkpt: triggers debugger breakpoint
                    // trap: triggers software trap exception
                    // pmevent: signals performance monitoring event
                end

                //============================================================
                // Misc Operations (Phase 6.2)
                //============================================================
                `OP_MISC: begin
                    misc_op <= 1'b1;
                    case (inst_func)
                        `MISC_NANOSLEEP: begin
                            // nanosleep t - pause execution for t nanoseconds
                            // No register write - just delays warp execution
                        end
                        `MISC_SETMAXNREG: begin
                            // setmaxnreg N - set maximum register count for this thread block
                            // No register write - just configures register limit
                        end
                        default: begin
                            // Unknown misc operation - treat as NOP
                        end
                    endcase
                end

                //============================================================
                // Async Store Operations (Phase 1.2)
                //============================================================
                `OP_ST_ASYNC: begin
                    st_async_op <= 1'b1;
                    case (inst_func)
                        `ST_ASYNC_GLOBAL, `ST_ASYNC_SHARED: begin
                            // st.async.global/shared [addr], data - async store
                            mem_write <= 1'b1;
                            if (inst_func == `ST_ASYNC_SHARED) mem_shared <= 1'b1;
                        end
                        `ST_ASYNC_COMMIT: begin
                            // cp.async.commit_group - commit current async store group
                            // No memory or register operation
                        end
                        `ST_ASYNC_WAIT: begin
                            // cp.async.wait_group N - wait for N groups to complete
                            // Stalls warp until condition met
                        end
                        default: begin
                            // Unknown st.async operation - treat as NOP
                        end
                    endcase
                end

                //============================================================
                // Multimem Operations (Phase 1.2 - Distributed Shared Memory)
                //============================================================
                `OP_MULTIMEM: begin
                    multimem_op <= 1'b1;
                    case (inst_func)
                        `MULTIMEM_LD: begin
                            // multimem.ld rd, [addr] - Load from distributed shared memory
                            mem_read <= 1'b1;
                            mem_shared <= 1'b1;
                            reg_write <= 1'b1;
                        end
                        `MULTIMEM_ST: begin
                            // multimem.st [addr], data, mask - Multicast store to multiple SMs
                            mem_write <= 1'b1;
                            mem_shared <= 1'b1;
                        end
                        `MULTIMEM_RED: begin
                            // multimem.red [addr], data, op - Multicast reduction
                            mem_write <= 1'b1;
                            mem_shared <= 1'b1;
                        end
                        default: begin
                            // Unknown multimem operation - treat as NOP
                        end
                    endcase
                end

                //============================================================
                // Barrier Cluster Operations (Phase 3.2 - Thread Block Cluster Sync)
                //============================================================
                `OP_BARRIER_CLUSTER: begin
                    barrier_cluster_op <= 1'b1;
                    sync_op <= 1'b1;  // All cluster barriers are sync operations
                    case (inst_func)
                        `CLUSTER_BARRIER_ARRIVE: begin
                            // barrier.cluster.arrive - Signal arrival at cluster barrier
                            // Non-blocking, just increments arrive count
                        end
                        `CLUSTER_BARRIER_WAIT: begin
                            // barrier.cluster.wait - Wait for all threads to arrive
                            // Blocks until all threads in cluster have arrived
                        end
                        `CLUSTER_BARRIER_SYNC: begin
                            // barrier.cluster.sync - Arrive and wait (combined)
                            // Equivalent to arrive followed by wait
                        end
                        `CLUSTER_BARRIER_INIT: begin
                            // barrier.cluster.init - Initialize cluster barrier
                            // Sets expected thread count for barrier
                        end
                        default: begin
                            // Unknown barrier.cluster operation - treat as NOP
                        end
                    endcase
                end

                `OP_MOV_IMM: begin
                    // Move immediate value to register
                    // Uses ALU with OR: rd = 0 | imm16
                    alu_op    <= 1'b1;
                    use_imm   <= 1'b1;
                    reg_write <= 1'b1;
                end

                //============================================================
                // Phase 5.1: Warp Collective Operations (Hopper+)
                //============================================================
                `OP_MATCH_SYNC: begin
                    // match.sync membermask, a, b - Warp-level predicate matching
                    // Returns mask of threads where Ra value matches Rb value
                    // membermask from Ra, comparison value from Rb
                    match_sync_op <= 1'b1;
                    sync_op <= 1'b1;  // Requires warp synchronization
                    reg_write <= 1'b1;  // Writes result mask to Rd
                    pred_write <= 1'b1; // Also sets predicate
                end

                `OP_ELECT_SYNC: begin
                    // elect.sync membermask - Warp-level leader election
                    // Elects one thread from participating threads
                    // membermask from Ra, result (lane id) to Rd
                    elect_sync_op <= 1'b1;
                    sync_op <= 1'b1;  // Requires warp synchronization
                    reg_write <= 1'b1;  // Writes elected lane id to Rd
                    pred_write <= 1'b1; // Sets predicate true for elected thread
                end

                `OP_RED_ASYNC: begin
                    // red.async.op dst, src - Async reduction to shared memory
                    // Performs reduction across participating threads asynchronously
                    // Writes result to shared memory address in Ra
                    red_async_op <= 1'b1;
                    // Non-blocking: warp continues execution
                    // Signals mbarrier when complete
                end

                //============================================================
                // Phase 5.2: DPX Instructions (Blackwell Dynamic Programming)
                //============================================================
                `OP_DPX: begin
                    // DPX operations for dynamic programming acceleration
                    // Used in Viterbi, DTW, sequence alignment, Smith-Waterman
                    // func field specifies operation: viaddmin, viaddmax, etc.
                    dpx_op <= 1'b1;
                    reg_write <= 1'b1;  // All DPX ops write result to Rd
                    `ifdef SIMULATION
                    $display("[DECODER] DPX: func=%0d rd=R%0d ra=R%0d rb=R%0d rc=R%0d",
                             inst_func, inst_rd, inst_ra, inst_rb, inst_rc);
                    `endif
                end

                //============================================================
                // Phase 5.3: Sparse MMA Operations (Blackwell 2:4 Sparsity)
                //============================================================
                `OP_SPARSE_MMA: begin
                    // Sparse matrix operations with 2:4 structured sparsity
                    // 50% compression with 2 non-zero values per 4 elements
                    sparse_mma_op <= 1'b1;
                    reg_write <= 1'b1;  // MMA results written to accumulator registers
                    case (inst_func)
                        `SPARSE_COMPRESS: begin
                            // Compress dense matrix to sparse format
                            `ifdef SIMULATION
                            $display("[DECODER] SPARSE_COMPRESS: compressing dense to 2:4 sparse");
                            `endif
                        end
                        `SPARSE_DECOMPRESS: begin
                            // Decompress sparse matrix to dense format
                            `ifdef SIMULATION
                            $display("[DECODER] SPARSE_DECOMPRESS: decompressing 2:4 sparse to dense");
                            `endif
                        end
                        default: begin
                            // Sparse MMA operations
                            `ifdef SIMULATION
                            $display("[DECODER] SPARSE_MMA: func=%0d", inst_func);
                            `endif
                        end
                    endcase
                end

                //============================================================
                // Phase 4.2: Address Space Query Instructions
                //============================================================
                `OP_ISSPACEP: begin
                    // isspacep - Test if address belongs to specific address space
                    // Result is a predicate (1 if in space, 0 otherwise)
                    isspacep_op <= 1'b1;
                    pred_write <= 1'b1;  // Writes predicate result
                    reg_write <= 1'b1;   // Also writes 0/1 to register
                    `ifdef SIMULATION
                    $display("[DECODER] ISSPACEP: func=%0d addr=R%0d", inst_func, inst_ra);
                    `endif
                end

                `OP_MAPA: begin
                    // mapa - Map address between address spaces
                    // Returns mapped address in destination register
                    mapa_op <= 1'b1;
                    reg_write <= 1'b1;
                    `ifdef SIMULATION
                    $display("[DECODER] MAPA: func=%0d src=R%0d dst=R%0d", inst_func, inst_ra, inst_rd);
                    `endif
                end

                `OP_GETCTARANK: begin
                    // getctarank - Get CTA rank within cluster
                    // Returns CTA ID in destination register
                    getctarank_op <= 1'b1;
                    reg_write <= 1'b1;
                    `ifdef SIMULATION
                    $display("[DECODER] GETCTARANK: dst=R%0d", inst_rd);
                    `endif
                end

                `OP_NOP: begin
                    // 空操作，不设置任何控制信号
                end

                default: begin
                    // 未知指令，当作NOP
                end
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
