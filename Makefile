SHELL := /bin/bash
export PATH := /opt/homebrew/bin:$(PATH)

#============================================================================
# RalphGPU Makefile
# CUDA/PTX兼容GPU IP
#============================================================================

# 工具
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
PYTHON = python3
VERILATOR ?= /opt/homebrew/bin/verilator
YOSYS ?= /opt/homebrew/bin/yosys

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
    $(RTL_DIR)/fpu64.v \
    $(RTL_DIR)/fp16_unit.v \
    $(RTL_DIR)/sfu.v \
    $(RTL_DIR)/tensor_core.v \
    $(RTL_DIR)/control_flow_unit.v \
    $(RTL_DIR)/warp_shuffle.v \
    $(RTL_DIR)/atomic_unit.v \
    $(RTL_DIR)/async_copy_engine.v \
    $(RTL_DIR)/mbarrier_unit.v \
    $(RTL_DIR)/wgmma.v \
    $(RTL_DIR)/texture_unit.v \
    $(RTL_DIR)/video_unit.v \
    $(RTL_DIR)/streaming_multiprocessor_v2.v \
    $(RTL_DIR)/command_queue.v \
    $(RTL_DIR)/command_processor.v \
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
    $(RTL_DIR)/blackwell_scheduler.v \
    $(RTL_DIR)/wgmma_tile_engine.v \
    $(RTL_DIR)/l2_interconnect.v \
    $(RTL_DIR)/performance_counters.v \
    $(RTL_DIR)/tma_unit.v \
    $(RTL_DIR)/cache_policy_unit.v \
    $(RTL_DIR)/cluster_barrier_unit.v \
    $(RTL_DIR)/cvt_unit.v \
    $(RTL_DIR)/dpx_unit.v \
    $(RTL_DIR)/dual_issue_scheduler.v \
    $(RTL_DIR)/fma_int32.v \
    $(RTL_DIR)/forwarding_unit.v \
    $(RTL_DIR)/griddep_unit.v \
    $(RTL_DIR)/l1_data_cache.v \
    $(RTL_DIR)/l1_data_cache_optimized.v \
    $(RTL_DIR)/l2_cache.v \
    $(RTL_DIR)/memory_coalescing_unit.v \
    $(RTL_DIR)/memory_controller.v \
    $(RTL_DIR)/multimem_unit.v \
    $(RTL_DIR)/st_bulk_unit.v \
    $(RTL_DIR)/stack_debug_unit.v \
    $(RTL_DIR)/tlb.v \
    $(RTL_DIR)/warp_collective_unit.v \
    $(RTL_DIR)/wb_fifo.v \
    $(RTL_DIR)/sm_wbq_bank.v \
    $(RTL_DIR)/sm_special_reg.v \
    $(RTL_DIR)/sm_gmem_arbiter.v \
    $(RTL_DIR)/sm_fetch_pipeline.v \
    $(RTL_DIR)/sm_writeback_arbiter.v \
    $(RTL_DIR)/chi_controller.v \
    $(RTL_DIR)/lz4_decompressor.v \
    $(RTL_DIR)/tensor_memory.v

# Testbench文件
TB_SRCS = $(TB_DIR)/tb_ralph_gpu.v

# 单元测试文件
TB_ALU = $(TB_DIR)/tb_alu.v
TB_MUL = $(TB_DIR)/tb_mul_unit.v
TB_FMA_INT32 = $(TB_DIR)/tb_fma_int32.v
TB_DEC = $(TB_DIR)/tb_decoder.v
TB_REG = $(TB_DIR)/tb_register_file.v
TB_REG_BANKED = $(TB_DIR)/tb_register_file_banked.v
TB_BW_SCHED_SB = $(TB_DIR)/tb_blackwell_scheduler_scoreboard.v
TB_SMEM = $(TB_DIR)/tb_shared_memory.v
TB_MEM_COALESCE = $(TB_DIR)/tb_memory_coalescing_unit.v
TB_MEM_IF = $(TB_DIR)/tb_memory_interface.v
TB_SM_FETCH_PIPE = $(TB_DIR)/tb_sm_fetch_pipeline.v
TB_MEM_QOS = $(TB_DIR)/tb_memory_qos.v
TB_WARP = $(TB_DIR)/tb_warp_scheduler.v
TB_DUAL_ISSUE_SCHED = $(TB_DIR)/tb_dual_issue_scheduler.v
TB_VADD = $(TB_DIR)/tb_vector_add.v
TB_MULTI = $(TB_DIR)/tb_multi_sm.v
TB_MEMSYS = $(TB_DIR)/tb_memory_subsystem.v
TB_MEMCTRL = $(TB_DIR)/tb_memory_controller.v
TB_L1_DATA_CACHE = $(TB_DIR)/tb_l1_data_cache.v
TB_ICACHE_LRU = $(TB_DIR)/tb_icache_lru_conflict.v
TB_TENSOR_FP4 = $(TB_DIR)/tb_tensor_core_fp4.v
TB_TENSOR_FP4_FP8 = $(TB_DIR)/tb_tensor_fp4_fp8.v
TB_TENSOR_MEMORY = $(TB_DIR)/tb_tensor_memory.v
TB_CFU = $(TB_DIR)/tb_control_flow_unit.v
TB_CHI_CTRL = $(TB_DIR)/tb_chi_controller.v
TB_RECONV = $(TB_DIR)/tb_reconvergence_stack.v
TB_LZ4_DECOMP = $(TB_DIR)/tb_lz4_decompressor.v
TB_L2_INTERCONNECT = $(TB_DIR)/tb_l2_interconnect.v

TB_TENSOR_FP4_FP8_FRM_GEN = $(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.v

TENSOR_FRM_SEED ?= 239
TENSOR_FRM_CASES_PER_DTYPE ?= 32

# Memory subsystem RTL files (Phase 2)
MEMSYS_SRCS = \
    $(RTL_DIR)/gpu_defines.vh \
    $(RTL_DIR)/memory_config.vh \
    $(RTL_DIR)/l2_cache.v \
    $(RTL_DIR)/tlb.v \
    $(RTL_DIR)/memory_controller.v

# Include路径
INCLUDES = -I$(RTL_DIR)
RTL_DEFINES = -DSM_V2 -DSIMULATION
ifneq ($(GPU_PROFILE),)
RTL_DEFINES += -DGPU_PROFILE_$(GPU_PROFILE)
endif

#============================================================================
# 目标
#============================================================================

.PHONY: all sim wave clean assemble help test test_all regression
.PHONY: test_alu test_mul test_fma_int32 test_decoder test_regfile test_regfile_banked test_bw_scheduler_scoreboard test_smem test_memory_coalescing_unit test_memory_interface test_sm_fetch_pipeline test_memory_qos test_l2_interconnect test_reconvergence_stack test_lz4_decompressor test_chi_controller test_tensor_memory test_warp test_dual_issue_scheduler test_control_flow_unit test_sfu test_cvt_unit benchmark
.PHONY: test_sm_v2_perf test_sm_v2_perf_gemm16_ptx test_sm_v2_perf_gemm16_wmma_ptx test_sm_v2_perf_gemm64_wgmma_ptx test_sm_v2_perf_tensor test_sm_v2_perf_tensor_multiwarp test_sm_v2_sched_raw_hazard test_raw_hazard test_tensor_core_fp4 test_tensor_fp4_fp8 test_tensor_fp4_fp8_frm test_cron_optimization bench_l1d_ipc_compare bench_mcu_mem_compare dashboard dashboard-sprint21-baseline dashboard-check dashboard-baseline
.PHONY: test_vector_add test_perf_counters test_perf_mem_stream test_perf_mem_gather test_perf_saxpy test_multi_sm test_command_queue test_command_processor test_wb_fifo test_ralph_gpu_top_cp_integration test_memsys test_memory_controller test_l1_data_cache test_icache_lru test_phase2 test_warp_valid_d1

# Audited must-run regression suite (stable gates for local + CI).
# Non-gating/extended tests are tracked separately and can be run via:
# make regression REGRESSION_TARGETS="$(REGRESSION_EXTENDED_TARGETS)"
REGRESSION_MUST_RUN_TARGETS = \
	test \
	test_regfile_banked \
	test_bw_scheduler_scoreboard \
	test_sfu \
	test_cvt_unit \
	test_tensor_fp4_fp8 \
	test_tensor_fp4_fp8_frm \
	test_phase2 \
	test_sm_v2_core \
	test_tensor_core_fp4 \
	test_raw_hazard \
	test_sm_v2_perf_gemm16_ptx \
	test_sm_v2_perf_gemm16_wmma_ptx \
	test_sm_v2_perf_gemm64_wgmma_ptx \
	test_warp_valid_d1 \
	test_command_processor test_wb_fifo \
	test_ralph_gpu_top_cp_integration \
	test_perf_counters \
	test_perf_mem_stream \
	test_perf_mem_gather \
	test_cron_optimization \
	bench_atomic_minimal \
	bench_app_compile

# Known flaky/heavy/non-deterministic tests, kept outside gating set.
REGRESSION_EXTENDED_TARGETS = \
	test_vector_add \
	test_multi_sm \
	test_sm_v2_full \
	test_sm_v2_perf_tensor \
	test_sm_v2_perf_tensor_multiwarp \
	test_l1_data_cache \
	test_icache_lru \
	test_dual_fetch \
	test_ptx \
	bench_atomics \
	bench_divergence

# Allow temporary suite override from CLI:
# make regression REGRESSION_TARGETS="test_alu test_mul test_fma_int32"
REGRESSION_TARGETS ?= $(REGRESSION_MUST_RUN_TARGETS)

# Top-level L1D cache mode for perf counter integration TB:
# 0 = enable L1D cache (default), 1 = bypass cache
TB_L1D_BYPASS ?= 0

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

test_fma_int32: $(BUILD_DIR)/tb_fma_int32.vvp
	@echo "========================================"
	@echo "Running FMA INT32 Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_fma_int32.vvp

$(BUILD_DIR)/tb_fma_int32.vvp: $(RTL_DIR)/fma_int32.v $(RTL_DIR)/gpu_defines.vh $(TB_FMA_INT32) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_FMA_INT32) $(RTL_DIR)/fma_int32.v

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

test_regfile_banked: $(BUILD_DIR)/tb_register_file_banked.vvp
	@echo "========================================"
	@echo "Running Banked Register File Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_register_file_banked.vvp

$(BUILD_DIR)/tb_register_file_banked.vvp: $(RTL_DIR)/register_file_banked.v $(RTL_DIR)/gpu_defines.vh $(TB_REG_BANKED) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_REG_BANKED) $(RTL_DIR)/register_file_banked.v

test_bw_scheduler_scoreboard: $(BUILD_DIR)/tb_blackwell_scheduler_scoreboard.vvp
	@echo "========================================"
	@echo "Running Blackwell Scheduler Scoreboard Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_blackwell_scheduler_scoreboard.vvp

$(BUILD_DIR)/tb_blackwell_scheduler_scoreboard.vvp: $(RTL_DIR)/blackwell_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_BW_SCHED_SB) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_BW_SCHED_SB) $(RTL_DIR)/blackwell_scheduler.v

test_smem: $(BUILD_DIR)/tb_shared_memory.vvp
	@echo "========================================"
	@echo "Running Shared Memory Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_shared_memory.vvp

$(BUILD_DIR)/tb_shared_memory.vvp: $(RTL_DIR)/shared_memory.v $(RTL_DIR)/gpu_defines.vh $(TB_SMEM) | $(BUILD_DIR)
	$(IVERILOG) $(INCLUDES) -o $@ $(TB_SMEM) $(RTL_DIR)/shared_memory.v

test_memory_coalescing_unit: $(BUILD_DIR)/tb_memory_coalescing_unit.vvp
	@echo "========================================"
	@echo "Running Memory Coalescing Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_coalescing_unit.vvp

$(BUILD_DIR)/tb_memory_coalescing_unit.vvp: $(RTL_DIR)/memory_coalescing_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_MEM_COALESCE) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEM_COALESCE) $(RTL_DIR)/memory_coalescing_unit.v

test_memory_interface: $(BUILD_DIR)/tb_memory_interface.vvp
	@echo "========================================"
	@echo "Running Memory Interface Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_interface.vvp

$(BUILD_DIR)/tb_memory_interface.vvp: $(RTL_DIR)/memory_interface.v $(RTL_DIR)/gpu_defines.vh $(TB_MEM_IF) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEM_IF) $(RTL_DIR)/memory_interface.v

test_sm_fetch_pipeline: $(BUILD_DIR)/tb_sm_fetch_pipeline.vvp
	@echo "========================================"
	@echo "Running SM Fetch Pipeline Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_fetch_pipeline.vvp

$(BUILD_DIR)/tb_sm_fetch_pipeline.vvp: $(RTL_SRCS) $(TB_SM_FETCH_PIPE) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_SM_FETCH_PIPE) $(filter %.v,$(RTL_SRCS))

test_memory_qos: $(BUILD_DIR)/tb_memory_qos.vvp
	@echo "========================================"
	@echo "Running Memory QoS Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_qos.vvp

$(BUILD_DIR)/tb_memory_qos.vvp: $(RTL_DIR)/memory_qos.v $(RTL_DIR)/gpu_defines.vh $(TB_MEM_QOS) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEM_QOS) $(RTL_DIR)/memory_qos.v

test_l2_interconnect: $(BUILD_DIR)/tb_l2_interconnect.vvp
	@echo "========================================"
	@echo "Running L2 Interconnect Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_l2_interconnect.vvp

$(BUILD_DIR)/tb_l2_interconnect.vvp: $(RTL_DIR)/l2_interconnect.v $(RTL_DIR)/gpu_defines.vh $(RTL_DIR)/memory_config.vh $(TB_L2_INTERCONNECT) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_L2_INTERCONNECT) $(RTL_DIR)/l2_interconnect.v

test_warp: $(BUILD_DIR)/tb_warp_scheduler.vvp
	@echo "========================================"
	@echo "Running Warp Scheduler Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_warp_scheduler.vvp

$(BUILD_DIR)/tb_warp_scheduler.vvp: $(RTL_DIR)/warp_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_WARP) | $(BUILD_DIR)

	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_WARP) $(RTL_DIR)/warp_scheduler.v
test_dual_issue_scheduler: $(BUILD_DIR)/tb_dual_issue_scheduler.vvp
	@echo "========================================"
	@echo "Running Dual Issue Scheduler Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_dual_issue_scheduler.vvp

$(BUILD_DIR)/tb_dual_issue_scheduler.vvp: $(RTL_DIR)/dual_issue_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_DUAL_ISSUE_SCHED) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DUAL_ISSUE_SCHED) $(RTL_DIR)/dual_issue_scheduler.v

test_control_flow_unit: $(BUILD_DIR)/tb_control_flow_unit.vvp
	@echo "========================================"
	@echo "Running Control Flow Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_control_flow_unit.vvp

$(BUILD_DIR)/tb_control_flow_unit.vvp: $(RTL_DIR)/control_flow_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_CFU) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_CFU) $(RTL_DIR)/control_flow_unit.v

test_chi_controller: $(BUILD_DIR)/tb_chi_controller.vvp
	@echo "========================================"
	@echo "Running CHI Controller Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_chi_controller.vvp

$(BUILD_DIR)/tb_chi_controller.vvp: $(RTL_DIR)/chi_controller.v $(RTL_DIR)/gpu_defines.vh $(TB_CHI_CTRL) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_CHI_CTRL) $(RTL_DIR)/chi_controller.v

test_reconvergence_stack: $(BUILD_DIR)/tb_reconvergence_stack.vvp
	@echo "========================================"
	@echo "Running Reconvergence Stack Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_reconvergence_stack.vvp

$(BUILD_DIR)/tb_reconvergence_stack.vvp: $(RTL_DIR)/reconvergence_stack.v $(RTL_DIR)/gpu_defines.vh $(TB_RECONV) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_RECONV) $(RTL_DIR)/reconvergence_stack.v

test_lz4_decompressor: $(BUILD_DIR)/tb_lz4_decompressor.vvp
	@echo "========================================"
	@echo "Running LZ4 Decompressor Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_lz4_decompressor.vvp

$(BUILD_DIR)/tb_lz4_decompressor.vvp: $(RTL_DIR)/lz4_decompressor.v $(TB_LZ4_DECOMP) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_LZ4_DECOMP) $(RTL_DIR)/lz4_decompressor.v

test_warp_ops: $(BUILD_DIR)/tb_warp_ops.vvp
	@echo "========================================"
	@echo "Running Warp Ops Functional Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_warp_ops.vvp

$(BUILD_DIR)/tb_warp_ops.vvp: $(RTL_DIR)/warp_shuffle.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_warp_ops.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_warp_ops.v $(RTL_DIR)/warp_shuffle.v


test_video_unit: $(BUILD_DIR)/tb_video_unit.vvp
	@echo "========================================"
	@echo "Running Video Unit SIMD Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_video_unit.vvp

$(BUILD_DIR)/tb_video_unit.vvp: $(RTL_DIR)/video_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_video_unit.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_video_unit.v $(RTL_DIR)/video_unit.v

test_tensor_core_e2e: $(BUILD_DIR)/tb_tensor_core_e2e.vvp
	@echo "========================================"
	@echo "Running Tensor Core E2E Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_tensor_core_e2e.vvp

$(BUILD_DIR)/tb_tensor_core_e2e.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_tensor_core_e2e.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_tensor_core_e2e.v $(RTL_DIR)/tensor_core.v
test_tensor_memory: $(BUILD_DIR)/tb_tensor_memory.vvp
	@echo "========================================"
	@echo "Running Tensor Memory Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_tensor_memory.vvp

$(BUILD_DIR)/tb_tensor_memory.vvp: $(RTL_DIR)/tensor_memory.v $(RTL_DIR)/gpu_defines.vh $(TB_TENSOR_MEMORY) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_TENSOR_MEMORY) $(RTL_DIR)/tensor_memory.v

test_texture_unit: $(BUILD_DIR)/tb_texture_unit.vvp
	@echo "========================================"
	@echo "Running Texture Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_texture_unit.vvp

$(BUILD_DIR)/tb_texture_unit.vvp: $(RTL_DIR)/texture_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_texture_unit.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_texture_unit.v $(RTL_DIR)/texture_unit.v
test_sfu: $(BUILD_DIR)/tb_sfu.vvp
	@echo "========================================"
	@echo "Running SFU Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sfu.vvp

$(BUILD_DIR)/tb_sfu.vvp: $(RTL_DIR)/sfu.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_sfu.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_sfu.v $(RTL_DIR)/sfu.v

test_cvt_unit: $(BUILD_DIR)/tb_cvt_unit.vvp
	@echo "========================================"
	@echo "Running CVT Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_cvt_unit.vvp

$(BUILD_DIR)/tb_cvt_unit.vvp: $(TB_DIR)/tb_cvt_unit.v $(RTL_DIR)/cvt_unit.v $(RTL_DIR)/gpu_defines.vh | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_cvt_unit.v $(RTL_DIR)/cvt_unit.v

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

test_tensor_fp4_fp8: $(BUILD_DIR)/tb_tensor_fp4_fp8.vvp
	@echo "========================================"
	@echo "Running Tensor Core FP4/FP8 e2e Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_tensor_fp4_fp8.vvp

$(BUILD_DIR)/tb_tensor_fp4_fp8.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_TENSOR_FP4_FP8) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_TENSOR_FP4_FP8) $(RTL_DIR)/tensor_core.v

test_tensor_fp4_fp8_frm: $(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.vvp
	@echo "========================================"
	@echo "Running Tensor Core FP4/FP8 RTL vs FRM E2E"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_tensor_fp4_fp8_frm_generated.vvp

$(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.v: $(TOOLS_DIR)/tensor_fp4_fp8_frm.py | $(BUILD_DIR)
	$(PYTHON) $(TOOLS_DIR)/tensor_fp4_fp8_frm.py --emit-tb $@ --emit-json $(BUILD_DIR)/tensor_fp4_fp8_frm_vectors.json --seed $(TENSOR_FRM_SEED) --cases-per-dtype $(TENSOR_FRM_CASES_PER_DTYPE)

$(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.vvp: $(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.v $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(BUILD_DIR)/tb_tensor_fp4_fp8_frm_generated.v $(RTL_DIR)/tensor_core.v

#----------------------------------------------------------------------------
# 集成测试
#----------------------------------------------------------------------------
$(BUILD_DIR)/vector_add.hex: $(EXAMPLES_DIR)/vector_add.ptx | $(BUILD_DIR)
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

$(BUILD_DIR)/saxpy.hex: $(EXAMPLES_DIR)/saxpy.ptx | $(BUILD_DIR)
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

test_vector_add: $(BUILD_DIR)/tb_vector_add.vvp $(BUILD_DIR)/vector_add.hex
	@echo "========================================"
	@echo "Running Vector Addition Integration Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_vector_add.vvp

$(BUILD_DIR)/tb_vector_add.vvp: $(RTL_SRCS) $(TB_VADD) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_VADD) $(filter %.v,$(RTL_SRCS))

test_perf_counters: $(BUILD_DIR)/tb_perf_counters_l1d$(TB_L1D_BYPASS).vvp $(BUILD_DIR)/vector_add.hex
	@echo "========================================"
	@echo "Running Performance Counters Integration Test (L1D_BYPASS=$(TB_L1D_BYPASS))"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_perf_counters_l1d$(TB_L1D_BYPASS).vvp | tee perf_counters_l1d$(TB_L1D_BYPASS).log

$(BUILD_DIR)/tb_perf_counters_l1d%.vvp: $(RTL_SRCS) $(TB_DIR)/tb_perf_counters.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -DTB_L1D_BYPASS=$* -o $@ $(TB_DIR)/tb_perf_counters.v $(filter %.v,$(RTL_SRCS))

bench_l1d_ipc_compare: $(BUILD_DIR)/vector_add.hex
	@echo "========================================"
	@echo "Comparing IPC with L1D cache ON/OFF"
	@echo "========================================"
	IVERILOG=$(IVERILOG) VVP=$(VVP) $(PYTHON) scripts/bench-l1d-ipc-compare.py

$(BUILD_DIR)/mem_stream8.hex: $(EXAMPLES_DIR)/mem_stream8.ptx | $(BUILD_DIR)
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

$(BUILD_DIR)/mem_gather.hex: $(EXAMPLES_DIR)/mem_gather.ptx | $(BUILD_DIR)
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

test_perf_mem_stream: $(BUILD_DIR)/tb_perf_mem_stream_l1d$(TB_L1D_BYPASS).vvp $(BUILD_DIR)/mem_stream8.hex
	@echo "========================================"
	@echo "Running Memory Stream Benchmark (L1D_BYPASS=$(TB_L1D_BYPASS))"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_perf_mem_stream_l1d$(TB_L1D_BYPASS).vvp | tee perf_mem_stream_l1d$(TB_L1D_BYPASS).log

test_perf_mem_gather: $(BUILD_DIR)/tb_perf_mem_gather_l1d$(TB_L1D_BYPASS).vvp $(BUILD_DIR)/mem_gather.hex
	@echo "========================================"
	@echo "Running Memory Gather Benchmark (L1D_BYPASS=$(TB_L1D_BYPASS))"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_perf_mem_gather_l1d$(TB_L1D_BYPASS).vvp | tee perf_mem_gather_l1d$(TB_L1D_BYPASS).log

test_perf_saxpy: $(BUILD_DIR)/tb_perf_saxpy_l1d$(TB_L1D_BYPASS).vvp $(BUILD_DIR)/saxpy.hex
	@echo "========================================"
	@echo "Running SAXPY Benchmark (L1D_BYPASS=$(TB_L1D_BYPASS))"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_perf_saxpy_l1d$(TB_L1D_BYPASS).vvp | tee perf_saxpy_l1d$(TB_L1D_BYPASS).log

$(BUILD_DIR)/tb_perf_mem_stream_l1d%.vvp: $(RTL_SRCS) $(TB_DIR)/tb_perf_mem_bench.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -DTB_L1D_BYPASS=$* -o $@ $(TB_DIR)/tb_perf_mem_bench.v $(filter %.v,$(RTL_SRCS))

$(BUILD_DIR)/tb_perf_mem_gather_l1d%.vvp: $(RTL_SRCS) $(TB_DIR)/tb_perf_mem_bench.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -DTB_L1D_BYPASS=$* -DTB_MEM_BENCH_GATHER -o $@ $(TB_DIR)/tb_perf_mem_bench.v $(filter %.v,$(RTL_SRCS))

$(BUILD_DIR)/tb_perf_saxpy_l1d%.vvp: $(RTL_SRCS) $(TB_DIR)/tb_perf_mem_bench.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -DTB_L1D_BYPASS=$* -DTB_MEM_BENCH_SAXPY -o $@ $(TB_DIR)/tb_perf_mem_bench.v $(filter %.v,$(RTL_SRCS))

bench_mcu_mem_compare: $(BUILD_DIR)/mem_stream8.hex $(BUILD_DIR)/mem_gather.hex
	@echo "========================================"
	@echo "Comparing memory benchmarks with L1D+MCU ON/OFF"
	@echo "========================================"
	IVERILOG=$(IVERILOG) VVP=$(VVP) $(PYTHON) scripts/bench-mcu-mem-compare.py

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

test_memory_controller: $(BUILD_DIR)/tb_memory_controller.vvp
	@echo "========================================"
	@echo "Running Memory Controller CDC/Response Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_controller.vvp

$(BUILD_DIR)/tb_memory_controller.vvp: $(RTL_DIR)/memory_controller.v $(RTL_DIR)/gpu_defines.vh $(RTL_DIR)/memory_config.vh $(TB_MEMCTRL) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEMCTRL) $(RTL_DIR)/memory_controller.v

test_l1_data_cache: $(BUILD_DIR)/tb_l1_data_cache.vvp
	@echo "========================================"
	@echo "Running L1 Data Cache Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_l1_data_cache.vvp

$(BUILD_DIR)/tb_l1_data_cache.vvp: $(TB_L1_DATA_CACHE) $(RTL_DIR)/l1_data_cache.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_L1_DATA_CACHE) $(RTL_DIR)/l1_data_cache.v

test_icache_lru: $(BUILD_DIR)/tb_icache_lru_conflict.vvp
	@echo "========================================"
	@echo "Running I-Cache LRU Conflict Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_icache_lru_conflict.vvp

$(BUILD_DIR)/tb_icache_lru_conflict.vvp: $(TB_ICACHE_LRU) $(RTL_DIR)/icache.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_ICACHE_LRU) $(RTL_DIR)/icache.v

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
TB_SM_V2_PERF_PTX = $(TB_DIR)/tb_sm_v2_perf_gemm16_ptx.v
TB_SM_V2_PERF_TC = $(TB_DIR)/tb_sm_v2_perf_tensor.v
TB_SM_V2_PERF_TC_MW = $(TB_DIR)/tb_sm_v2_perf_tensor_multiwarp.v
TB_SM_V2_SCHED_RAW = $(TB_DIR)/tb_sm_v2_sched_raw_hazard.v

# SM V2 RTL sources (define SM_V2 to exclude duplicate wrappers from fpu.v/sfu.v)
SM_V2_SRCS = \
	$(RTL_DIR)/gpu_defines.vh \
	$(RTL_DIR)/memory_config.vh \
	$(RTL_DIR)/streaming_multiprocessor_v2.v \
    $(RTL_DIR)/command_queue.v \
	$(RTL_DIR)/decoder.v \
	$(RTL_DIR)/register_file_banked.v \
	$(RTL_DIR)/branch_predictor.v \
	$(RTL_DIR)/icache.v \
	$(RTL_DIR)/l1_data_cache.v \
	$(RTL_DIR)/advanced_scheduler.v \
	$(RTL_DIR)/blackwell_scheduler.v \
	$(RTL_DIR)/reconvergence_stack.v \
	$(RTL_DIR)/alu.v \
	$(RTL_DIR)/mul_unit.v \
	$(RTL_DIR)/fpu.v \
	$(RTL_DIR)/fpu64.v \
	$(RTL_DIR)/fp16_unit.v \
	$(RTL_DIR)/sfu.v \
	$(RTL_DIR)/tensor_core.v \
	$(RTL_DIR)/control_flow_unit.v \
	$(RTL_DIR)/shared_memory.v \
	$(RTL_DIR)/memory_interface.v \
	$(RTL_DIR)/memory_coalescing_unit.v \
	$(RTL_DIR)/warp_shuffle.v \
	$(RTL_DIR)/atomic_unit.v \
	$(RTL_DIR)/async_copy_engine.v \
	$(RTL_DIR)/tma_unit.v \
	$(RTL_DIR)/mbarrier_unit.v \
	$(RTL_DIR)/wgmma.v \
	$(RTL_DIR)/wgmma_tile_engine.v \
	$(RTL_DIR)/video_unit.v \
	$(RTL_DIR)/texture_unit.v \
	$(RTL_DIR)/sm_fetch_pipeline.v \
	$(RTL_DIR)/sm_writeback_arbiter.v \
	$(RTL_DIR)/wb_fifo.v \
	$(RTL_DIR)/sm_wbq_bank.v \
	$(RTL_DIR)/sm_special_reg.v \
	$(RTL_DIR)/sm_gmem_arbiter.v

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

# Performance microbenchmark (legacy alias kept for compatibility)
test_sm_v2_perf: test_sm_v2_perf_gemm16_ptx
	@echo "Alias target completed: test_sm_v2_perf_gemm16_ptx"

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

# Scheduler RAW hazard test (P2): verify scheduler blocks issue on RAW hazards
test_sm_v2_sched_raw_hazard: $(BUILD_DIR)/tb_sm_v2_sched_raw_hazard.vvp
	@echo "========================================"
	@echo "Running SM V2 Scheduler RAW Hazard Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_sched_raw_hazard.vvp

$(BUILD_DIR)/tb_sm_v2_sched_raw_hazard.vvp: $(SM_V2_SRCS) $(TB_SM_V2_SCHED_RAW) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_SCHED_RAW) $(filter %.v,$(SM_V2_SRCS))

# Alias for regression audit naming consistency.
test_raw_hazard: test_sm_v2_sched_raw_hazard
	@echo "Alias target completed: test_sm_v2_sched_raw_hazard"

#----------------------------------------------------------------------------
# Track 1-2 Performance Benchmarks (Atomics + Divergence)
#----------------------------------------------------------------------------
TB_BENCH_ATOMICS = $(TB_DIR)/tb_bench_atomics.v
TB_BENCH_DIVERGENCE = $(TB_DIR)/tb_bench_divergence.v
TB_ATOMIC_MINIMAL = $(TB_DIR)/tb_atomic_contention_minimal.v
BENCH_DIVERGENCE_MAX_CYCLES ?= 5000

# Atomic operations benchmark
bench_atomics: $(BUILD_DIR)/tb_bench_atomics.vvp tb/bench_atomics.hex
	@echo "========================================"
	@echo "Running Atomic Operations Benchmark"
	@echo "========================================"
	ln -sf ../tb/bench_atomics.hex $(BUILD_DIR)/bench_atomics.hex && cd $(BUILD_DIR) && $(VVP) tb_bench_atomics.vvp | tee bench_atomics.log
	@echo "Output saved to $(BUILD_DIR)/bench_atomics.log"

$(BUILD_DIR)/tb_bench_atomics.vvp: $(RTL_SRCS) $(TB_BENCH_ATOMICS) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_BENCH_ATOMICS) $(filter %.v,$(RTL_SRCS))

tb/bench_atomics.hex: tb/bench_atomics.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Divergence handling benchmark
bench_divergence: $(BUILD_DIR)/tb_bench_divergence.vvp tb/bench_divergence.hex
	@echo "========================================"
	@echo "Running Branch Divergence Benchmark"
	@echo "========================================"
	@echo "Max cycle guard: $(BENCH_DIVERGENCE_MAX_CYCLES)"
	ln -sf ../tb/bench_divergence.hex $(BUILD_DIR)/bench_divergence.hex
	cd $(BUILD_DIR) && $(VVP) tb_bench_divergence.vvp +max_cycles=$(BENCH_DIVERGENCE_MAX_CYCLES) | tee bench_divergence.log
	@echo "Output saved to $(BUILD_DIR)/bench_divergence.log"

$(BUILD_DIR)/tb_bench_divergence.vvp: $(RTL_SRCS) $(TB_BENCH_DIVERGENCE) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_BENCH_DIVERGENCE) $(filter %.v,$(RTL_SRCS))

tb/bench_divergence.hex: tb/bench_divergence.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Minimal atomic contention test (faster)
bench_atomic_minimal: $(BUILD_DIR)/tb_atomic_contention_minimal.vvp asm/bench_atomic_minimal.hex
	@echo "========================================"
	@echo "Running Minimal Atomic Contention Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_atomic_contention_minimal.vvp

$(BUILD_DIR)/tb_atomic_contention_minimal.vvp: $(RTL_SRCS) $(TB_ATOMIC_MINIMAL) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_ATOMIC_MINIMAL) $(filter %.v,$(RTL_SRCS))

asm/bench_atomic_minimal.hex: asm/bench_atomic_minimal.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@


# Application-level PTX compilation
asm/bench_vecadd_atomic.hex: asm/bench_vecadd_atomic.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

asm/bench_matmul_sync.hex: asm/bench_matmul_sync.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

asm/bench_parallel_reduction.hex: asm/bench_parallel_reduction.ptx
	$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Compile all application benchmarks
bench_app_compile: asm/bench_vecadd_atomic.hex asm/bench_matmul_sync.hex asm/bench_parallel_reduction.hex
	@echo "Application benchmarks compiled"

# Run all benchmarks
bench_all: bench_atomics bench_divergence
	@echo "========================================"
	@echo "All Benchmarks Complete"
	@echo "========================================"

# Generate performance report from benchmark logs
# Performance dashboard: run benchmarks, generate IPC/stall/utilization report
benchmark:
	@echo "========================================"
	@echo "Running Standard Benchmark Dashboard"
	@echo "========================================"
	$(PYTHON) tools/perf_dashboard.py --run \
		--repro-cmd "make benchmark" \
		--json $(BUILD_DIR)/benchmark_results.json \
		--csv $(BUILD_DIR)/benchmark_results.csv \
		-o docs/PERF_DASHBOARD.md
	@echo "Benchmark dashboard: docs/PERF_DASHBOARD.md"
	@echo "JSON: $(BUILD_DIR)/benchmark_results.json"
	@echo "CSV:  $(BUILD_DIR)/benchmark_results.csv"

dashboard:
	@echo "========================================"
	@echo "Running Performance Dashboard"
	@echo "========================================"
	$(PYTHON) tools/perf_dashboard.py --run \
		--json $(BUILD_DIR)/perf_results.json \
		--csv $(BUILD_DIR)/perf_results.csv \
		-o docs/PERF_DASHBOARD.md
	@echo "Dashboard: docs/PERF_DASHBOARD.md"
	@echo "JSON:      $(BUILD_DIR)/perf_results.json"
	@echo "CSV:       $(BUILD_DIR)/perf_results.csv"

# Sprint 21 baseline for issue #286 (GEMM/Tensor focused)
dashboard-sprint21-baseline:
	@echo "========================================"
	@echo "Running Sprint 21 PERF baseline (GEMM/Tensor)"
	@echo "========================================"
	$(PYTHON) tools/perf_dashboard.py --run \
		--workload alu_heavy \
		--workload tensor \
		--workload tensor_multiwarp \
		--repro-cmd "make dashboard-sprint21-baseline" \
		--json $(BUILD_DIR)/perf_results_sprint21.json \
		--csv $(BUILD_DIR)/perf_results_sprint21.csv \
		-o docs/PERF_DASHBOARD.md
	@echo "Dashboard: docs/PERF_DASHBOARD.md"
	@echo "JSON:      $(BUILD_DIR)/perf_results_sprint21.json"
	@echo "CSV:       $(BUILD_DIR)/perf_results_sprint21.csv"

# Dashboard with regression check against baseline
dashboard-check:
	@echo "========================================"
	@echo "Performance Dashboard + Regression Check"
	@echo "========================================"
	$(PYTHON) tools/perf_dashboard.py --run \
		--json $(BUILD_DIR)/perf_results.json \
		--csv $(BUILD_DIR)/perf_results.csv \
		--baseline $(BUILD_DIR)/perf_baseline.json \
		-o docs/PERF_DASHBOARD.md

# Save current results as new baseline
dashboard-baseline:
	@echo "Saving current results as baseline..."
	cp $(BUILD_DIR)/perf_results.json $(BUILD_DIR)/perf_baseline.json
	@echo "Baseline saved: $(BUILD_DIR)/perf_baseline.json"

perf_report:
	@echo "========================================"
	@echo "Generating Performance Report"
	@echo "========================================"
	$(PYTHON) scripts/perf_analysis.py \
		--parse $(BUILD_DIR)/bench_atomics.log \
		--parse $(BUILD_DIR)/bench_divergence.log \
		-o docs/PERFORMANCE_REPORT.md
	@echo "Report saved to docs/PERFORMANCE_REPORT.md"

#----------------------------------------------------------------------------
# 运行所有测试
#----------------------------------------------------------------------------
test: test_alu test_mul test_fma_int32 test_decoder test_regfile test_smem test_memory_coalescing_unit test_memory_interface test_sm_fetch_pipeline test_memory_qos test_l2_interconnect test_reconvergence_stack test_lz4_decompressor test_chi_controller test_warp test_dual_issue_scheduler test_warp_ops test_video_unit test_tensor_core_e2e test_tensor_memory test_texture_unit test_command_queue test_command_processor test_wb_fifo test_memory_controller_hbm
	@echo "========================================"
	@echo "All Unit Tests Completed"
	@echo "========================================"

test_ptx: $(BUILD_DIR)
	@echo "========================================"
	@echo "Running PTX Comprehensive Test Suite"
	@echo "========================================"
	$(IVERILOG) -g2012 $(INCLUDES) -o $(BUILD_DIR)/tb_ptx_tests $(TB_DIR)/tb_ptx_tests.v $(filter %.v,$(RTL_SRCS))
	cd $(BUILD_DIR) && $(VVP) tb_ptx_tests

test_all: test test_vector_add test_multi_sm sim
	@echo "========================================"
	@echo "All Tests Completed (Unit + Integration)"
	@echo "========================================"

regression:
	@echo "========================================"
	@echo "Running Audited Regression Suite"
	@echo "========================================"
	@total=0; passed=0; failed=0; failed_targets=""; \
	for target in $(REGRESSION_TARGETS); do \
		total=$$((total + 1)); \
		echo ""; \
		echo "[$$total/$(words $(REGRESSION_TARGETS))] $$target"; \
		if $(MAKE) --no-print-directory $$target; then \
			passed=$$((passed + 1)); \
		else \
			failed=$$((failed + 1)); \
			failed_targets="$$failed_targets $$target"; \
		fi; \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "Regression Summary: $$passed/$$total passed, $$failed failed"; \
	if [ $$failed -ne 0 ]; then \
		echo "Failed targets:$$failed_targets"; \
		echo "========================================"; \
		exit 1; \
	fi; \
	echo "========================================"

test_cron_optimization:
	@echo "========================================"
	@echo "Running Cron Optimization / Heartbeat Tests"
	@echo "========================================"
	bash scripts/test-cron-optimization.sh


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
# Performance Benchmarks (RALPH-7)
#----------------------------------------------------------------------------
PERF_RTL = $(shell find $(RTL_DIR) -name '*.v' | sort)

perf: $(BUILD_DIR)
	@echo "============================================================"
	@echo "RalphGPU Performance Benchmarks"
	@echo "============================================================"
	@echo ""
	@echo "--- Building MatMul 4x4 ---"
	$(IVERILOG) -g2012 $(INCLUDES) -DSM_V2 -DSIMULATION -o $(BUILD_DIR)/tb_matmul_perf.vvp $(PERF_RTL) $(TB_DIR)/tb_matmul_4x4_fp16_gpu_top.v
	@echo "--- Running MatMul 4x4 ---"
	cd $(BUILD_DIR) && $(VVP) tb_matmul_perf.vvp > matmul_perf.log 2>&1
	$(PYTHON) $(TOOLS_DIR)/perf_report.py $(BUILD_DIR)/matmul_perf.log "FP16 MatMul 4x4"
	@echo ""
	@echo "--- Building Tiny MLP ---"
	$(IVERILOG) -g2012 $(INCLUDES) -DSM_V2 -DSIMULATION -o $(BUILD_DIR)/tb_mlp_perf.vvp $(PERF_RTL) $(TB_DIR)/tb_tiny_mlp.v
	@echo "--- Running Tiny MLP ---"
	cd $(BUILD_DIR) && $(VVP) tb_mlp_perf.vvp > mlp_perf.log 2>&1
	$(PYTHON) $(TOOLS_DIR)/perf_report.py $(BUILD_DIR)/mlp_perf.log "Tiny MLP (4->4->1)"
	@echo ""
	@echo "============================================================"
	@echo "Benchmarks Complete"
	@echo "============================================================"

#----------------------------------------------------------------------------
lint:
	$(VERILATOR) --lint-only --top ralph_gpu_top -Wall \
		-Wno-fatal -Wno-BLKLOOPINIT \
		-Wno-DECLFILENAME -Wno-PINCONNECTEMPTY \
		-Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
		-Wno-UNOPTFLAT \
		$(INCLUDES) $(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# 综合检查 (Yosys)
#----------------------------------------------------------------------------
synth: $(BUILD_DIR)
	$(YOSYS) -q -p "read_verilog -sv $(filter %.v,$(RTL_SRCS)); hierarchy -check -top ralph_gpu_top; proc; opt; check -assert" > $(BUILD_DIR)/yosys_synth.log 2>&1

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
	@echo "  synth         - Run Yosys synthesis/check flow"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Test Targets:"
	@echo "  test          - Run all unit tests"
	@echo "  test_l1_data_cache - Test L1 data cache hit/miss/LRU/bank conflict"
	@echo "  test_all      - Run all tests (unit + integration)"
	@echo "  regression    - Run audited must-run test_/bench_ suite with summary"
	@echo "  regression (extended) - Override REGRESSION_TARGETS with REGRESSION_EXTENDED_TARGETS"
	@echo "  test_cron_optimization - Run cron heartbeat/optimization script tests"
	@echo ""
	@echo "Unit Tests:"
	@echo "  test_alu      - Test ALU operations"
	@echo "  test_mul      - Test multiply unit"
	@echo "  test_decoder  - Test instruction decoder"
	@echo "  test_regfile  - Test register file"
	@echo "  test_memory_coalescing_unit - Test memory coalescing logic"
	@echo "  test_smem     - Test shared memory"
	@echo "  test_dual_issue_scheduler - Test dual-issue scheduler"
	@echo  test_dual_issue_scheduler - Test dual-issue scheduler
	@echo "  test_cvt_unit - Test CVT conversion unit"
	@echo "  test_tensor_core_fp4 - Tensor Core FP4 sanity test"
	@echo "  test_tensor_fp4_fp8 - Tensor Core FP4/FP8 handwritten e2e test"
	@echo "  test_tensor_fp4_fp8_frm - Tensor Core FP4/FP8 generated RTL vs FRM e2e"
	@echo ""
	@echo "Integration Tests:"
	@echo "  test_vector_add - Test vector addition kernel"
	@echo "  test_perf_counters - Test top-level perf counters integration (TB_L1D_BYPASS=0/1)"
	@echo "  bench_l1d_ipc_compare - Run perf counters with L1D on/off and print IPC delta"
	@echo "  test_perf_mem_stream - Memory-intensive stream benchmark (TB_L1D_BYPASS=0/1)"
	@echo "  test_perf_mem_gather - Memory-intensive gather benchmark (TB_L1D_BYPASS=0/1)"
	@echo "  test_perf_saxpy - SAXPY benchmark with L1D/L2 profiling"
	@echo "  bench_mcu_mem_compare - Run #306 memory benchmarks with L1D on/off and print table"
	@echo "  test_multi_sm   - Test multi-SM parallel execution"
	@echo "  test_raw_hazard - Alias for SM V2 scheduler RAW hazard gating test"
	@echo "  test_sm_v2_perf - SM V2 FP32 FMA performance microbenchmark"
	@echo "  test_sm_v2_perf_gemm16_ptx - SM V2 PTX-driven GEMM 16x16x16 microbenchmark"
	@echo "  test_sm_v2_perf_gemm16_wmma_ptx - SM V2 PTX WMMA GEMM 16x16x16 path test"
	@echo "  test_sm_v2_perf_gemm64_wgmma_ptx - SM V2 PTX WGMMA GEMM 64x8x16 path test"
	@echo "  test_sm_v2_perf_tensor - SM V2 Tensor Core WMMA microbenchmark"
	@echo "  test_sm_v2_perf_tensor_multiwarp - SM V2 Tensor Core multi-warp test"
	@echo ""
	@echo "Performance Benchmarks (Track 1-2):"
	@echo "  bench_atomics       - Run atomic operations benchmark"
	@echo "  bench_divergence    - Run branch divergence benchmark"
	@echo "  bench_atomic_minimal - Quick atomic contention test"
	@echo "  bench_app_compile   - Compile application-level benchmarks"
	@echo "  bench_all           - Run all benchmarks"
	@echo "  perf_report         - Generate PERFORMANCE_REPORT.md from logs"
	@echo "  dashboard            - Run perf benchmarks + generate dashboard (IPC/stall/util)"
	@echo "  benchmark            - Standardized dashboard report (IPC/occupancy/stalls)"
	@echo "  dashboard-check      - Dashboard + regression check vs baseline"
	@echo "  dashboard-baseline   - Save current results as new baseline"
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

# warp_inst_valid_d1 stall scenario test
test_warp_valid_d1: $(BUILD_DIR)/tb_warp_inst_valid_d1.vvp
	@echo "========================================"
	@echo "Running warp_inst_valid_d1 Stall Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_warp_inst_valid_d1.vvp

$(BUILD_DIR)/tb_warp_inst_valid_d1.vvp: $(TB_DIR)/tb_warp_inst_valid_d1.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_warp_inst_valid_d1.v


# Performance microbenchmark (PTX-driven WMMA GEMM stream)
test_sm_v2_perf_gemm16_wmma_ptx: programs/gemm16_wmma.hex $(BUILD_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.vvp
	@echo "========================================"
	@echo "Running SM V2 PTX WMMA GEMM Path Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16_wmma_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.vvp: $(SM_V2_SRCS) $(TB_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.v $(filter %.v,$(RTL_SRCS))

programs/gemm16_wmma.hex: tests/gemm16_wmma.ptx tools/ptx_assembler.py
	$(PYTHON) tools/ptx_assembler.py tests/gemm16_wmma.ptx -o programs/gemm16_wmma.hex


# Performance microbenchmark (PTX-driven WGMMA GEMM stream)
test_sm_v2_perf_gemm64_wgmma_ptx: programs/gemm64_wgmma.hex $(BUILD_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.vvp
	@echo "========================================"
	@echo "Running SM V2 PTX WGMMA GEMM Path Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm64_wgmma_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.vvp: $(SM_V2_SRCS) $(TB_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.v $(filter %.v,$(RTL_SRCS))

programs/gemm64_wgmma.hex: tests/gemm64_wgmma.ptx tools/ptx_assembler.py
	$(PYTHON) tools/ptx_assembler.py tests/gemm64_wgmma.ptx -o programs/gemm64_wgmma.hex

#----------------------------------------------------------------------------
# Command Processor unit test
#----------------------------------------------------------------------------
test_command_queue: $(BUILD_DIR)/tb_command_queue.vvp
	@echo "========================================"
	@echo "Running Command Queue Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_command_queue.vvp

$(BUILD_DIR)/tb_command_queue.vvp: $(RTL_DIR)/command_queue.v $(RTL_DIR)/gpu_defines.vh tb/tb_command_queue_ring.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ tb/tb_command_queue_ring.v $(RTL_DIR)/command_queue.v

test_command_processor: $(BUILD_DIR)/tb_command_processor.vvp
	@echo "========================================"
	@echo "Running Command Processor Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_command_processor.vvp

$(BUILD_DIR)/tb_command_processor.vvp: $(RTL_DIR)/command_queue.v $(RTL_DIR)/command_processor.v $(RTL_DIR)/gpu_defines.vh tb/tb_command_processor.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ tb/tb_command_processor.v $(RTL_DIR)/command_queue.v $(RTL_DIR)/command_processor.v

test_ralph_gpu_top_cp_integration: $(BUILD_DIR)/tb_ralph_gpu_top_cp_integration.vvp
	@echo "========================================"
	@echo "Running Top-Level Command Processor Integration Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_ralph_gpu_top_cp_integration.vvp

$(BUILD_DIR)/tb_ralph_gpu_top_cp_integration.vvp: $(RTL_SRCS) $(TB_DIR)/tb_ralph_gpu_top_cp_integration.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_DIR)/tb_ralph_gpu_top_cp_integration.v $(filter %.v,$(RTL_SRCS))

# Optional FPGA vendor-flow targets
-include fpga/Makefile.fpga

# Dual-fetch test (#226 True Dual-Issue)
TB_DUAL_FETCH = $(TB_DIR)/tb_dual_fetch.v

test_dual_fetch: $(BUILD_DIR)/tb_dual_fetch.vvp
	@echo "========================================"
	@echo "Running Dual-Fetch Test (#226)"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_dual_fetch.vvp

$(BUILD_DIR)/tb_dual_fetch.vvp: $(SM_V2_SRCS) $(TB_DUAL_FETCH) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_DUAL_FETCH) $(filter %.v,$(SM_V2_SRCS))

# WB FIFO Unit Test
test_wb_fifo: $(BUILD_DIR)/tb_wb_fifo.vvp
	@echo "========================================"
	@echo "Running WB FIFO Unit Test (Overflow Coverage)"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_wb_fifo.vvp

$(BUILD_DIR)/tb_wb_fifo.vvp: $(RTL_DIR)/wb_fifo.v tb/tb_wb_fifo_overflow.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ tb/tb_wb_fifo_overflow.v $(RTL_DIR)/wb_fifo.v

TB_ATOMIC = $(TB_DIR)/tb_atomic_unit.v

test_atomic: $(BUILD_DIR)/tb_atomic_unit.vvp
	@echo "========================================"
	@echo "Running Atomic Unit Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_atomic_unit.vvp

$(BUILD_DIR)/tb_atomic_unit.vvp: $(RTL_DIR)/atomic_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_ATOMIC) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_ATOMIC) $(RTL_DIR)/atomic_unit.v

test_memory_controller_hbm: $(BUILD_DIR)/tb_memory_controller_hbm.vvp
	@echo "========================================"
	@echo "Running HBM Memory Controller Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_memory_controller_hbm.vvp

$(BUILD_DIR)/tb_memory_controller_hbm.vvp: $(RTL_DIR)/memory_controller_hbm.v $(RTL_DIR)/gpu_defines.vh $(RTL_DIR)/memory_config.vh tb/tb_memory_controller_hbm.v | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ tb/tb_memory_controller_hbm.v $(RTL_DIR)/memory_controller_hbm.v
