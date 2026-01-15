# RalphGPU 完善计划

## 目标
实现一个完整功能的 CUDA/PTX 兼容 GPU IP，核数量可配置。

## 当前状态

### 已完成 ✅
- [x] ALU (26测试通过)
- [x] 乘法器 (16测试通过)
- [x] 解码器 (16测试通过)
- [x] 寄存器文件 (11测试通过)
- [x] 共享内存 (8测试通过)
- [x] Warp调度器 (10测试通过)
- [x] 基础架构 (SM, 顶层, AXI接口)
- [x] 核数量可配置 (NUM_SM参数)

### 待修复 🔧
- [ ] 流水线无法正常结束kernel
- [ ] Warp释放机制缺失
- [ ] EXIT指令未实现

### 待实现 📝
- [ ] 分支指令完整实现
- [ ] 比较指令(SETP)测试
- [ ] 除法单元
- [ ] 浮点运算(FP32)

---

## Phase 1: 修复核心流水线 (高优先级)

### 1.1 添加EXIT指令支持
- 在 `gpu_defines.vh` 添加 `OP_EXIT` 操作码
- 在解码器中识别EXIT指令
- 在SM中处理EXIT: 释放当前Warp

### 1.2 实现Warp释放机制
- 当执行EXIT时设置 `dealloc_en = 1`
- 正确更新 `warp_valid` 状态
- 当所有Warp释放后返回IDLE

### 1.3 流水线完成条件
- 检测 `warp_valid == 0` 返回IDLE
- 正确产生 `kernel_done` 信号

---

## Phase 2: 完善PTX指令集

### 2.1 分支指令
- 条件分支 `@p bra target`
- 无条件分支 `bra target`
- 分支目标计算

### 2.2 比较指令
- `setp.eq/ne/lt/le/gt/ge`
- 谓词寄存器管理

### 2.3 除法单元
- `div.s32 rd, ra, rb`
- `rem.s32 rd, ra, rb`

---

## Phase 3: 浮点支持 (FP32)

### 3.1 FPU模块
- `add.f32`, `sub.f32`
- `mul.f32`, `div.f32`
- `fma.rn.f32`

### 3.2 类型转换
- `cvt.f32.s32`
- `cvt.s32.f32`

---

## Phase 4: 高级特性

### 4.1 完整3D索引
- tid.x/y/z
- ctaid.x/y/z
- ntid.x/y/z

### 4.2 向量化
- `ld.v4.f32`
- `st.v4.f32`

---

## 当前进度

**阶段**: Phase 1 - 核心流水线修复 ✅ 完成
**状态**: EXIT指令已实现，基本仿真通过
**下一步**: 调试Multi-SM测试，然后进入Phase 2

---

## 本次迭代完成的工作

### 1. EXIT指令实现
- `gpu_defines.vh`: 添加 `OP_EXIT` (6'b001011) 和 `OP_RET` (6'b001100)
- `decoder.v`: 添加 `exit_op` 输出信号，解码EXIT/RET指令
- `streaming_multiprocessor.v`:
  - 添加 `dec_exit_op` 信号
  - 实现 `warp_exit_en = dec_exit_op && (pipe_state == PIPE_EXEC)`
  - 修改 `dealloc_en` 连接到 `warp_exit_en`
  - EXIT时跳过MEM/WB阶段
  - WB后检查 `warp_valid==0` 返回IDLE

### 2. 测试文件更新
- `tb_ralph_gpu.v`: 添加EXIT指令到kernel末尾
- `tb_vector_add.v`: 添加EXIT指令
- `tb_multi_sm.v`: 添加EXIT指令

### 3. 测试结果
```
单元测试: 全部通过 (87/87)
├─ ALU:          26 PASSED ✅
├─ MUL:          16 PASSED ✅
├─ Decoder:      16 PASSED ✅
├─ Register:     11 PASSED ✅
├─ Shared Mem:    8 PASSED ✅
└─ Warp Sched:   10 PASSED ✅

集成测试:
├─ 基本仿真:     PASSED ✅ (kernel正常完成)
├─ Vector Add:   2/2 PASSED ✅
└─ Multi-SM:     0/6 PASSED ⚠️ (调度时序问题)
```

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 集成测试超时 | 1 | 分析发现dealloc_en=0导致warp不释放 |
| kernel无法完成 | 2 | 添加EXIT指令，修复warp释放逻辑 ✅ |
| Multi-SM测试失败 | 3 | 待调查：调度状态机时序问题 |
