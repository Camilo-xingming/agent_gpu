#============================================================================
# RalphGPU Makefile
# CUDA/PTX兼容GPU IP
#============================================================================

# 工具
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
PYTHON = python3

# 目录
RTL_DIR = rtl
TB_DIR = tb
TOOLS_DIR = tools
EXAMPLES_DIR = examples
BUILD_DIR = build

# Profile selection (LITE/BALANCED/HPC)
GPU_PROFILE ?=

# RTL源文件
RTL_SRCS = \
    $(RTL_DIR)/gpu_defines.vh \
    $(RTL_DIR)/memory_config.vh \
    $(RTL_DIR)/alu.v \
    $(RTL_DIR)/mul_unit.v \
    $(RTL_DIR)/register_file.v \
    $(RTL_DIR)/decoder.v \
    $(RTL_DIR)/warp_scheduler.v \
    $(RTL_DIR)/shared_memory.v \
    $(RTL_DIR)/memory_interface.v \
    $(RTL_DIR)/fpu.v \
    $(RTL_DIR)/sfu.v \
    $(RTL_DIR)/tensor_core.v \
    $(RTL_DIR)/control_flow_unit.v \
    $(RTL_DIR)/warp_shuffle.v \
    $(RTL_DIR)/atomic_unit.v \
    $(RTL_DIR)/streaming_multiprocessor_v2.v \
    $(RTL_DIR)/ralph_gpu_top.v \
    $(RTL_DIR)/memory_controller_hbm.v \
    $(RTL_DIR)/memory_interface_wide.v \
    $(RTL_DIR)/memory_qos.v \
    $(RTL_DIR)/tlb_enhanced.v \
    $(RTL_DIR)/branch_predictor.v \
    $(RTL_DIR)/icache.v \
    $(RTL_DIR)/reconvergence_stack.v \
    $(RTL_DIR)/register_file_banked.v \
    $(RTL_DIR)/advanced_scheduler.v \
    $(RTL_DIR)/wgmma_tile_engine.v \
    $(RTL_DIR)/l2_interconnect.v \
    $(RTL_DIR)/performance_counters.v

# Testbench文件
TB_SRCS = $(TB_DIR)/tb_ralph_gpu.v

# 单元测试文件
TB_ALU = $(TB_DIR)/tb_alu.v
TB_MUL = $(TB_DIR)/tb_mul_unit.v
TB_DEC = $(TB_DIR)/tb_decoder.v
TB_REG = $(TB_DIR)/tb_register_file.v
TB_SMEM = $(TB_DIR)/tb_shared_memory.v
TB_WARP = $(TB_DIR)/tb_warp_scheduler.v
TB_VADD = $(TB_DIR)/tb_vector_add.v
TB_MULTI = $(TB_DIR)/tb_multi_sm.v
TB_MEMSYS = $(TB_DIR)/tb_memory_subsystem.v
TB_TENSOR_FP4 = $(TB_DIR)/tb_tensor_core_fp4.v

# Memory subsystem RTL files (Phase 2)
MEMSYS_SRCS = \
    $(RTL_DIR)/gpu_defines.vh \
    $(RTL_DIR)/memory_config.vh \
    $(RTL_DIR)/l2_cache.v \
    $(RTL_DIR)/tlb.v \
    $(RTL_DIR)/memory_controller.v

# Include路径
INCLUDES = -I$(RTL_DIR)
RTL_DEFINES = -DSM_V2
ifneq ($(GPU_PROFILE),)
RTL_DEFINES += -DGPU_PROFILE_$(GPU_PROFILE)
endif

#============================================================================
# 目标
#============================================================================

.PHONY: all sim wave clean assemble help test test_all
.PHONY: test_alu test_mul test_decoder test_regfile test_smem test_warp
.PHONY: test_sm_v2_perf test_sm_v2_perf_gemm16_ptx test_sm_v2_perf_tensor test_sm_v2_perf_tensor_multiwarp test_tensor_core_fp4
.PHONY: test_vector_add test_multi_sm test_memsys test_phase2

all: $(BUILD_DIR) sim

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#----------------------------------------------------------------------------
# 主仿真
#----------------------------------------------------------------------------
sim: $(BUILD_DIR)/tb_ralph_gpu.vvp
	cd $(BUILD_DIR) && $(VVP) tb_ralph_gpu.vvp

$(BUILD_DIR)/tb_ralph_gpu.vvp: $(RTL_SRCS) $(TB_SRCS) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_SRCS) \
		$(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# 单元测试
#----------------------------------------------------------------------------
test_alu: $(BUILD_DIR)/tb_alu.vvp
	@echo "========================================"
	@echo "Running ALU Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_alu.vvp

$(BUILD_DIR)/tb_alu.vvp: $(RTL_DIR)/alu.v $(RTL_DIR)/gpu_defines.vh $(TB_ALU) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_ALU) $(RTL_DIR)/alu.v

test_mul: $(BUILD_DIR)/tb_mul_unit.vvp
	@echo "========================================"
	@echo "Running Multiply Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_mul_unit.vvp

$(BUILD_DIR)/tb_mul_unit.vvp: $(RTL_DIR)/mul_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_MUL) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_MUL) $(RTL_DIR)/mul_unit.v

test_decoder: $(BUILD_DIR)/tb_decoder.vvp
	@echo "========================================"
	@echo "Running Decoder Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_decoder.vvp

$(BUILD_DIR)/tb_decoder.vvp: $(RTL_DIR)/decoder.v $(RTL_DIR)/gpu_defines.vh $(TB_DEC) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_DEC) $(RTL_DIR)/decoder.v

test_regfile: $(BUILD_DIR)/tb_register_file.vvp
	@echo "========================================"
	@echo "Running Register File Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_register_file.vvp

$(BUILD_DIR)/tb_register_file.vvp: $(RTL_DIR)/register_file.v $(RTL_DIR)/gpu_defines.vh $(TB_REG) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_REG) $(RTL_DIR)/register_file.v

test_smem: $(BUILD_DIR)/tb_shared_memory.vvp
	@echo "========================================"
	@echo "Running Shared Memory Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_shared_memory.vvp

$(BUILD_DIR)/tb_shared_memory.vvp: $(RTL_DIR)/shared_memory.v $(RTL_DIR)/gpu_defines.vh $(TB_SMEM) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_SMEM) $(RTL_DIR)/shared_memory.v

test_warp: $(BUILD_DIR)/tb_warp_scheduler.vvp
	@echo "========================================"
	@echo "Running Warp Scheduler Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_warp_scheduler.vvp

$(BUILD_DIR)/tb_warp_scheduler.vvp: $(RTL_DIR)/warp_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_WARP) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_WARP) $(RTL_DIR)/warp_scheduler.v

#----------------------------------------------------------------------------
# Tensor Core FP4 Sanity Test
#----------------------------------------------------------------------------
test_tensor_core_fp4: $(BUILD_DIR)/tb_tensor_core_fp4.vvp
	@echo "========================================"
	@echo "Running Tensor Core FP4 Sanity Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_tensor_core_fp4.vvp

$(BUILD_DIR)/tb_tensor_core_fp4.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_TENSOR_FP4) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_TENSOR_FP4) $(RTL_DIR)/tensor_core.v

#----------------------------------------------------------------------------
# 集成测试
#----------------------------------------------------------------------------
test_vector_add: $(BUILD_DIR)/tb_vector_add.vvp
	@echo "========================================"
	@echo "Running Vector Addition Integration Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_vector_add.vvp

$(BUILD_DIR)/tb_vector_add.vvp: $(RTL_SRCS) $(TB_VADD) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_VADD) $(filter %.v,$(RTL_SRCS))

test_multi_sm: $(BUILD_DIR)/tb_multi_sm.vvp
	@echo "========================================"
	@echo "Running Multi-SM Parallel Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_multi_sm.vvp

$(BUILD_DIR)/tb_multi_sm.vvp: $(RTL_SRCS) $(TB_MULTI) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_MULTI) $(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# Phase 2 Memory Subsystem Tests
#----------------------------------------------------------------------------
test_memsys: $(BUILD_DIR)/tb_memory_subsystem.vvp
	@echo "========================================"
	@echo "Running Memory Subsystem Test (Phase 2)"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_subsystem.vvp

$(BUILD_DIR)/tb_memory_subsystem.vvp: $(MEMSYS_SRCS) $(TB_MEMSYS) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEMSYS) $(filter %.v,$(MEMSYS_SRCS))

test_phase2: test_memsys
	@echo "========================================"
	@echo "Running Phase 2 Performance Verification"
	@echo "========================================"
	$(PYTHON) tests/phase2_100_percent_verification.py
	@echo "========================================"
	@echo "Phase 2 Verification Complete"
	@echo "========================================"

#----------------------------------------------------------------------------
# SM V2 Integration Tests (Scoreboard, FU Tracking, WB Arbitration)
#----------------------------------------------------------------------------
TB_SM_V2 = $(TB_DIR)/tb_sm_v2_integration.v
TB_SM_V2_PERF = $(TB_DIR)/tb_sm_v2_perf_gemm16.v
TB_SM_V2_PERF_PTX = $(TB_DIR)/tb_sm_v2_perf_gemm16_ptx.v
TB_SM_V2_PERF_TC = $(TB_DIR)/tb_sm_v2_perf_tensor.v
TB_SM_V2_PERF_TC_MW = $(TB_DIR)/tb_sm_v2_perf_tensor_multiwarp.v

# SM V2 RTL sources (define SM_V2 to exclude duplicate wrappers from fpu.v/sfu.v)
SM_V2_SRCS = \
	$(RTL_DIR)/gpu_defines.vh \
	$(RTL_DIR)/memory_config.vh \
	$(RTL_DIR)/streaming_multiprocessor_v2.v \
	$(RTL_DIR)/decoder.v \
	$(RTL_DIR)/register_file_banked.v \
	$(RTL_DIR)/branch_predictor.v \
	$(RTL_DIR)/icache.v \
	$(RTL_DIR)/advanced_scheduler.v \
	$(RTL_DIR)/reconvergence_stack.v \
	$(RTL_DIR)/alu.v \
	$(RTL_DIR)/mul_unit.v \
	$(RTL_DIR)/fpu.v \
	$(RTL_DIR)/sfu.v \
	$(RTL_DIR)/tensor_core.v \
	$(RTL_DIR)/control_flow_unit.v \
	$(RTL_DIR)/shared_memory.v \
	$(RTL_DIR)/memory_interface.v \
	$(RTL_DIR)/warp_shuffle.v \
	$(RTL_DIR)/atomic_unit.v \
	$(RTL_DIR)/wgmma.v \
	$(RTL_DIR)/wgmma_tile_engine.v

SM_V2_DEFINES = -DSM_V2 -DDEBUG_SM_V2

test_sm_v2: test_sm_v2_core
	@echo "SM V2 Architecture Tests Complete"

# Core architecture test (standalone, no external dependencies)
test_sm_v2_core: $(BUILD_DIR)/tb_sm_v2_core.vvp
	@echo "========================================"
	@echo "Running SM V2 Core Architecture Test"
	@echo "  - Scoreboard RAW/WAW hazard detection"
	@echo "  - Multi-cycle FU latency tracking"
	@echo "  - Round-robin writeback arbitration"
	@echo "  - Multi-warp independence"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_core.vvp

$(BUILD_DIR)/tb_sm_v2_core.vvp: $(TB_DIR)/tb_sm_v2_core.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 -o $@ $(TB_DIR)/tb_sm_v2_core.v

# Full integration test (requires all modules)
test_sm_v2_full: $(BUILD_DIR)/tb_sm_v2_integration.vvp
	@echo "========================================"
	@echo "Running SM V2 Full Integration Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_integration.vvp

$(BUILD_DIR)/tb_sm_v2_integration.vvp: $(SM_V2_SRCS) $(TB_SM_V2) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (FP32 FMA stream)
test_sm_v2_perf: $(BUILD_DIR)/tb_sm_v2_perf_gemm16.vvp
	@echo "========================================"
	@echo "Running SM V2 Performance Test (GEMM 16x16x16)"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (PTX-driven FMA stream)
test_sm_v2_perf_gemm16_ptx: $(BUILD_DIR)/tb_sm_v2_perf_gemm16_ptx.vvp
	@echo "========================================"
	@echo "Running SM V2 PTX Performance Test (GEMM 16x16x16)"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16_ptx.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_PTX) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_PTX) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (Tensor Core WMMA stream)
test_sm_v2_perf_tensor: $(BUILD_DIR)/tb_sm_v2_perf_tensor.vvp
	@echo "========================================"
	@echo "Running SM V2 Tensor Core Performance Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_tensor.vvp

$(BUILD_DIR)/tb_sm_v2_perf_tensor.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_TC) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_TC) $(filter %.v,$(SM_V2_SRCS))

# Multi-warp Tensor Core backpressure test
test_sm_v2_perf_tensor_multiwarp: $(BUILD_DIR)/tb_sm_v2_perf_tensor_multiwarp.vvp
	@echo "========================================"
	@echo "Running SM V2 Tensor Core Multi-warp Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_tensor_multiwarp.vvp

$(BUILD_DIR)/tb_sm_v2_perf_tensor_multiwarp.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_TC_MW) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_TC_MW) $(filter %.v,$(SM_V2_SRCS))

#----------------------------------------------------------------------------
# 运行所有测试
#----------------------------------------------------------------------------
test: test_alu test_mul test_decoder test_regfile test_smem test_warp
	@echo "========================================"
	@echo "All Unit Tests Completed"
	@echo "========================================"

test_all: test test_vector_add test_multi_sm sim
	@echo "========================================"
	@echo "All Tests Completed (Unit + Integration)"
	@echo "========================================"

#----------------------------------------------------------------------------
# 波形查看
#----------------------------------------------------------------------------
wave: sim
	$(GTKWAVE) $(BUILD_DIR)/tb_ralph_gpu.vcd &

#----------------------------------------------------------------------------
# PTX汇编
#----------------------------------------------------------------------------
assemble: $(EXAMPLES_DIR)/vector_add.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $(BUILD_DIR)/vector_add.hex
	@echo "Assembled to $(BUILD_DIR)/vector_add.hex"

#----------------------------------------------------------------------------
# 语法检查
#----------------------------------------------------------------------------
lint:
	verilator --lint-only $(INCLUDES) $(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# 清理
#----------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
	rm -f *.vcd

#----------------------------------------------------------------------------
# 帮助
#----------------------------------------------------------------------------
help:
	@echo "RalphGPU - CUDA/PTX Compatible GPU IP"
	@echo ""
	@echo "Main Targets:"
	@echo "  all           - Build and run simulation (default)"
	@echo "  sim           - Run RTL simulation with Icarus Verilog"
	@echo "  wave          - Open waveform viewer (GTKWave)"
	@echo "  assemble      - Assemble PTX example to machine code"
	@echo "  lint          - Run Verilator lint check"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Test Targets:"
	@echo "  test          - Run all unit tests"
	@echo "  test_all      - Run all tests (unit + integration)"
	@echo ""
	@echo "Unit Tests:"
	@echo "  test_alu      - Test ALU operations"
	@echo "  test_mul      - Test multiply unit"
	@echo "  test_decoder  - Test instruction decoder"
	@echo "  test_regfile  - Test register file"
	@echo "  test_smem     - Test shared memory"
	@echo "  test_warp     - Test warp scheduler"
	@echo "  test_tensor_core_fp4 - Tensor Core FP4 sanity test"
	@echo ""
	@echo "Integration Tests:"
	@echo "  test_vector_add - Test vector addition kernel"
	@echo "  test_multi_sm   - Test multi-SM parallel execution"
	@echo "  test_sm_v2_perf - SM V2 FP32 FMA performance microbenchmark"
	@echo "  test_sm_v2_perf_gemm16_ptx - SM V2 PTX-driven GEMM 16x16x16 microbenchmark"
	@echo "  test_sm_v2_perf_tensor - SM V2 Tensor Core WMMA microbenchmark"
	@echo "  test_sm_v2_perf_tensor_multiwarp - SM V2 Tensor Core multi-warp test"
	@echo ""
	@echo "Directory structure:"
	@echo "  rtl/      - RTL source files"
	@echo "  tb/       - Testbenches"
	@echo "  tools/    - Assembler and other tools"
	@echo "  examples/ - PTX example programs"
	@echo "  doc/      - Documentation"
	@echo "  build/    - Build outputs"

#============================================================================
# 配置选项 (可通过make参数覆盖)
#============================================================================
# NUM_SM=4 make sim  - 使用4个SM仿真
# 注意: 需要修改gpu_defines.vh中的参数
