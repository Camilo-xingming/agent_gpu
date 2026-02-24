SHELL := /bin/bash
export PATH := /opt/homebrew/bin:$(PATH)

#============================================================================
# RalphGPU Makefile
# CUDA/PTXå…¼å®¹GPU IP
#============================================================================

# å·¥å…·
IVERILOG = iverilog
VVP = vvp
GTKWAVE = gtkwave
PYTHON = python3
VERILATOR ?= /opt/homebrew/bin/verilator
YOSYS ?= /opt/homebrew/bin/yosys

# ç›®å½•
RTL_DIR = rtl
TB_DIR = tb
TOOLS_DIR = tools
EXAMPLES_DIR = examples
BUILD_DIR = build

# Profile selection (LITE/BALANCED/HPC)
GPU_PROFILE ?=

# RTLæº–‡ä»¶
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

# Testbenchæ–‡ä»¶
TB_SRCS = $(TB_DIR)/tb_ralph_gpu.v

# å•å…ƒæµ‹è¯•æ–‡ä»¶
TB_ALU = $(TB_DIR)/tb_alu.v
TB_MUL = $(TB_DIR)/tb_mul_unit.v
TB_DEC = $(TB_DIR)/tb_decoder.v
TB_REG = $(TB_DIR)/tb_register_file.v
TB_REG_BANKED = $(TB_DIR)/tb_register_file_banked.v
TB_BW_SCHED_SB = $(TB_DIR)/tb_blackwell_scheduler_scoreboard.v
TB_SMEM = $(TB_DIR)/tb_shared_memory.v
TB_WARP = $(TB_DIR)/tb_warp_scheduler.v
TB_VADD = $(TB_DIR)/tb_vector_add.v
TB_MULTI = $(TB_DIR)/tb_multi_sm.v
TB_MEMSYS = $(TB_DIR)/tb_memory_subsystem.v
TB_L1_DATA_CACHE = $(TB_DIR)/tb_l1_data_cache.v
TB_TENSOR_FP4 = $(TB_DIR)/tb_tensor_core_fp4.v
TB_TENSOR_FP4_FP8 = $(TB_DIR)/tb_tensor_fp4_fp8.v

# Memory subsystem RTL files (Phase 2)
MEMSYS_SRCS = \
    $(RTL_DIR)/gpu_defines.vh \
    $(RTL_DIR)/memory_config.vh \
    $(RTL_DIR)/l2_cache.v \
    $(RTL_DIR)/tlb.v \
    $(RTL_DIR)/memory_controller.v

# Includeè·¯å¾„
INCLUDES = -I$(RTL_DIR)
RTL_DEFINES = -DSM_V2 -DSIMULATION
ifneq ($(GPU_PROFILE),)
RTL_DEFINES += -DGPU_PROFILE_$(GPU_PROFILE)
endif

#============================================================================
# ç›®æ ‡
#============================================================================

.PHONY: all sim wave clean assemble help test test_all
.PHONY: test_alu test_mul test_decoder test_regfile test_regfile_banked test_bw_scheduler_scoreboard test_smem test_warp test_sfu
.PHONY: test_sm_v2_perf test_sm_v2_perf_gemm16_ptx test_sm_v2_perf_gemm16_wmma_ptx test_sm_v2_perf_gemm64_wgmma_ptx test_sm_v2_perf_tensor test_sm_v2_perf_tensor_multiwarp test_sm_v2_sched_raw_hazard test_tensor_core_fp4 dashboard dashboard-check dashboard-baseline
.PHONY: test_vector_add test_multi_sm test_memsys test_l1_data_cache test_phase2

all: $(BUILD_DIR) sim

$(BUILD_DIR):
\tmkdir -p $(BUILD_DIR)

#----------------------------------------------------------------------------
# ä¸»ä»¿çœŸ
#----------------------------------------------------------------------------
sim: $(BUILD_DIR)/tb_ralph_gpu.vvp
\tcd $(BUILD_DIR) && $(VVP) tb_ralph_gpu.vvp

$(BUILD_DIR)/tb_ralph_gpu.vvp: $(RTL_SRCS) $(TB_SRCS) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_SRCS) \
\t\t$(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# å•å…ƒæµ‹è¯•
#----------------------------------------------------------------------------
test_alu: $(BUILD_DIR)/tb_alu.vvp
\t@echo "========================================"
\t@echo "Running ALU Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_alu.vvp

$(BUILD_DIR)/tb_alu.vvp: $(RTL_DIR)/alu.v $(RTL_DIR)/gpu_defines.vh $(TB_ALU) | $(BUILD_DIR)
\t$(IVERILOG) $(INCLUDES) -o $@ $(TB_ALU) $(RTL_DIR)/alu.v

test_mul: $(BUILD_DIR)/tb_mul_unit.vvp
\t@echo "========================================"
\t@echo "Running Multiply Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_mul_unit.vvp

$(BUILD_DIR)/tb_mul_unit.vvp: $(RTL_DIR)/mul_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_MUL) | $(BUILD_DIR)
\t$(IVERILOG) $(INCLUDES) -o $@ $(TB_MUL) $(RTL_DIR)/mul_unit.v

test_decoder: $(BUILD_DIR)/tb_decoder.vvp
\t@echo "========================================"
\t@echo "Running Decoder Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_decoder.vvp

$(BUILD_DIR)/tb_decoder.vvp: $(RTL_DIR)/decoder.v $(RTL_DIR)/gpu_defines.vh $(TB_DEC) | $(BUILD_DIR)
\t$(IVERILOG) $(INCLUDES) -o $@ $(TB_DEC) $(RTL_DIR)/decoder.v

test_regfile: $(BUILD_DIR)/tb_register_file.vvp
\t@echo "========================================"
\t@echo "Running Register File Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_register_file.vvp

$(BUILD_DIR)/tb_register_file.vvp: $(RTL_DIR)/register_file.v $(RTL_DIR)/gpu_defines.vh $(TB_REG) | $(BUILD_DIR)
\t$(IVERILOG) $(INCLUDES) -o $@ $(TB_REG) $(RTL_DIR)/register_file.v

test_regfile_banked: $(BUILD_DIR)/tb_register_file_banked.vvp
\t@echo "========================================"
\t@echo "Running Banked Register File Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_register_file_banked.vvp

$(BUILD_DIR)/tb_register_file_banked.vvp: $(RTL_DIR)/register_file_banked.v $(RTL_DIR)/gpu_defines.vh $(TB_REG_BANKED) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_REG_BANKED) $(RTL_DIR)/register_file_banked.v

test_bw_scheduler_scoreboard: $(BUILD_DIR)/tb_blackwell_scheduler_scoreboard.vvp
\t@echo "========================================"
\t@echo "Running Blackwell Scheduler Scoreboard Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_blackwell_scheduler_scoreboard.vvp

$(BUILD_DIR)/tb_blackwell_scheduler_scoreboard.vvp: $(RTL_DIR)/blackwell_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_BW_SCHED_SB) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_BW_SCHED_SB) $(RTL_DIR)/blackwell_scheduler.v

test_smem: $(BUILD_DIR)/tb_shared_memory.vvp
\t@echo "========================================"
\t@echo "Running Shared Memory Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_shared_memory.vvp

$(BUILD_DIR)/tb_shared_memory.vvp: $(RTL_DIR)/shared_memory.v $(RTL_DIR)/gpu_defines.vh $(TB_SMEM) | $(BUILD_DIR)
\t$(IVERILOG) $(INCLUDES) -o $@ $(TB_SMEM) $(RTL_DIR)/shared_memory.v

test_warp: $(BUILD_DIR)/tb_warp_scheduler.vvp
\t@echo "========================================"
\t@echo "Running Warp Scheduler Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_warp_scheduler.vvp

$(BUILD_DIR)/tb_warp_scheduler.vvp: $(RTL_DIR)/warp_scheduler.v $(RTL_DIR)/gpu_defines.vh $(TB_WARP) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_WARP) $(RTL_DIR)/warp_scheduler.v

test_warp_ops: $(BUILD_DIR)/tb_warp_ops.vvp
\t@echo "========================================"
\t@echo "Running Warp Ops Functional Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_warp_ops.vvp

$(BUILD_DIR)/tb_warp_ops.vvp: $(RTL_DIR)/warp_shuffle.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_warp_ops.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_warp_ops.v $(RTL_DIR)/warp_shuffle.v


test_video_unit: $(BUILD_DIR)/tb_video_unit.vvp
\t@echo "========================================"
\t@echo "Running Video Unit SIMD Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_video_unit.vvp

$(BUILD_DIR)/tb_video_unit.vvp: $(RTL_DIR)/video_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_video_unit.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_video_unit.v $(RTL_DIR)/video_unit.v

test_tensor_core_e2e: $(BUILD_DIR)/tb_tensor_core_e2e.vvp
\t@echo "========================================"
\t@echo "Running Tensor Core E2E Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_tensor_core_e2e.vvp

$(BUILD_DIR)/tb_tensor_core_e2e.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_tensor_core_e2e.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_tensor_core_e2e.v $(RTL_DIR)/tensor_core.v
test_texture_unit: $(BUILD_DIR)/tb_texture_unit.vvp
\t@echo "========================================"
\t@echo "Running Texture Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_texture_unit.vvp

$(BUILD_DIR)/tb_texture_unit.vvp: $(RTL_DIR)/texture_unit.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_texture_unit.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_texture_unit.v $(RTL_DIR)/texture_unit.v
test_sfu: $(BUILD_DIR)/tb_sfu.vvp
\t@echo "========================================"
\t@echo "Running SFU Unit Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sfu.vvp

$(BUILD_DIR)/tb_sfu.vvp: $(RTL_DIR)/sfu.v $(RTL_DIR)/gpu_defines.vh $(TB_DIR)/tb_sfu.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_sfu.v $(RTL_DIR)/sfu.v

#----------------------------------------------------------------------------
# Tensor Core FP4 Sanity Test
#----------------------------------------------------------------------------
test_tensor_core_fp4: $(BUILD_DIR)/tb_tensor_core_fp4.vvp
\t@echo "========================================"
\t@echo "Running Tensor Core FP4 Sanity Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_tensor_core_fp4.vvp

$(BUILD_DIR)/tb_tensor_core_fp4.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_TENSOR_FP4) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_TENSOR_FP4) $(RTL_DIR)/tensor_core.v

test_tensor_fp4_fp8: $(BUILD_DIR)/tb_tensor_fp4_fp8.vvp
\t@echo "========================================"
\t@echo "Running Tensor Core FP4/FP8 e2e Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_tensor_fp4_fp8.vvp

$(BUILD_DIR)/tb_tensor_fp4_fp8.vvp: $(RTL_DIR)/tensor_core.v $(RTL_DIR)/gpu_defines.vh $(TB_TENSOR_FP4_FP8) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_TENSOR_FP4_FP8) $(RTL_DIR)/tensor_core.v

#----------------------------------------------------------------------------
# é›†æˆµ‹è¯•
#----------------------------------------------------------------------------
test_vector_add: $(BUILD_DIR)/tb_vector_add.vvp
\t@echo "========================================"
\t@echo "Running Vector Addition Integration Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_vector_add.vvp

$(BUILD_DIR)/tb_vector_add.vvp: $(RTL_SRCS) $(TB_VADD) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_VADD) $(filter %.v,$(RTL_SRCS))

test_multi_sm: $(BUILD_DIR)/tb_multi_sm.vvp
\t@echo "========================================"
\t@echo "Running Multi-SM Parallel Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_multi_sm.vvp

$(BUILD_DIR)/tb_multi_sm.vvp: $(RTL_SRCS) $(TB_MULTI) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_MULTI) $(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# Phase 2 Memory Subsystem Tests
#----------------------------------------------------------------------------
test_memsys: $(BUILD_DIR)/tb_memory_subsystem.vvp
\t@echo "========================================"
\t@echo "Running Memory Subsystem Test (Phase 2)"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_memory_subsystem.vvp

$(BUILD_DIR)/tb_memory_subsystem.vvp: $(MEMSYS_SRCS) $(TB_MEMSYS) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MEMSYS) $(filter %.v,$(MEMSYS_SRCS))

test_l1_data_cache: $(BUILD_DIR)/tb_l1_data_cache.vvp
\t@echo "========================================"
\t@echo "Running L1 Data Cache Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_l1_data_cache.vvp

$(BUILD_DIR)/tb_l1_data_cache.vvp: $(TB_L1_DATA_CACHE) $(RTL_DIR)/l1_data_cache.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_L1_DATA_CACHE) $(RTL_DIR)/l1_data_cache.v

test_phase2: test_memsys
\t@echo "========================================"
\t@echo "Running Phase 2 Performance Verification"
\t@echo "========================================"
\t$(PYTHON) tests/phase2_100_percent_verification.py
\t@echo "========================================"
\t@echo "Phase 2 Verification Complete"
\t@echo "========================================"

#----------------------------------------------------------------------------
# SM V2 Integration Tests (Scoreboard, FU Tracking, WB Arbitration)
#----------------------------------------------------------------------------
TB_SM_V2 = $(TB_DIR)/tb_sm_v2_integration.v
TB_SM_V2_PERF = $(TB_DIR)/tb_sm_v2_perf_gemm16.v
TB_SM_V2_PERF_PTX = $(TB_DIR)/tb_sm_v2_perf_gemm16_ptx.v
TB_SM_V2_PERF_TC = $(TB_DIR)/tb_sm_v2_perf_tensor.v
TB_SM_V2_PERF_TC_MW = $(TB_DIR)/tb_sm_v2_perf_tensor_multiwarp.v
TB_SM_V2_SCHED_RAW = $(TB_DIR)/tb_sm_v2_sched_raw_hazard.v

# SM V2 RTL sources (define SM_V2 to exclude duplicate wrappers from fpu.v/sfu.v)
SM_V2_SRCS = \
\t$(RTL_DIR)/gpu_defines.vh \
\t$(RTL_DIR)/memory_config.vh \
\t$(RTL_DIR)/streaming_multiprocessor_v2.v \
\t$(RTL_DIR)/decoder.v \
\t$(RTL_DIR)/register_file_banked.v \
\t$(RTL_DIR)/branch_predictor.v \
\t$(RTL_DIR)/icache.v \
\t$(RTL_DIR)/l1_data_cache.v \
\t$(RTL_DIR)/advanced_scheduler.v \
\t$(RTL_DIR)/blackwell_scheduler.v \
\t$(RTL_DIR)/reconvergence_stack.v \
\t$(RTL_DIR)/alu.v \
\t$(RTL_DIR)/mul_unit.v \
\t$(RTL_DIR)/fpu.v \
\t$(RTL_DIR)/fpu64.v \
\t$(RTL_DIR)/fp16_unit.v \
\t$(RTL_DIR)/sfu.v \
\t$(RTL_DIR)/tensor_core.v \
\t$(RTL_DIR)/control_flow_unit.v \
\t$(RTL_DIR)/shared_memory.v \
\t$(RTL_DIR)/memory_interface.v \
\t$(RTL_DIR)/warp_shuffle.v \
\t$(RTL_DIR)/atomic_unit.v \
\t$(RTL_DIR)/async_copy_engine.v \
\t$(RTL_DIR)/tma_unit.v \
\t$(RTL_DIR)/mbarrier_unit.v \
\t$(RTL_DIR)/wgmma.v \
\t$(RTL_DIR)/wgmma_tile_engine.v \
\t$(RTL_DIR)/video_unit.v \
\t$(RTL_DIR)/texture_unit.v \
\t$(RTL_DIR)/sm_fetch_pipeline.v \
\t$(RTL_DIR)/sm_writeback_arbiter.v \
\t$(RTL_DIR)/wb_fifo.v \
\t$(RTL_DIR)/sm_wbq_bank.v \
\t$(RTL_DIR)/sm_special_reg.v \
\t$(RTL_DIR)/sm_gmem_arbiter.v

SM_V2_DEFINES = -DSM_V2 -DDEBUG_SM_V2

test_sm_v2: test_sm_v2_core
\t@echo "SM V2 Architecture Tests Complete"

# Core architecture test (standalone, no external dependencies)
test_sm_v2_core: $(BUILD_DIR)/tb_sm_v2_core.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Core Architecture Test"
\t@echo "  - Scoreboard RAW/WAW hazard detection"
\t@echo "  - Multi-cycle FU latency tracking"
\t@echo "  - Round-robin writeback arbitration"
\t@echo "  - Multi-warp independence"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_core.vvp

$(BUILD_DIR)/tb_sm_v2_core.vvp: $(TB_DIR)/tb_sm_v2_core.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 -o $@ $(TB_DIR)/tb_sm_v2_core.v

# Full integration test (requires all modules)
test_sm_v2_full: $(BUILD_DIR)/tb_sm_v2_integration.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Full Integration Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_integration.vvp

$(BUILD_DIR)/tb_sm_v2_integration.vvp: $(SM_V2_SRCS) $(TB_SM_V2) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (FP32 FMA stream)
test_sm_v2_perf: $(BUILD_DIR)/tb_sm_v2_perf_gemm16.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Performance Test (GEMM 16x16x16)"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (PTX-driven FMA stream)
test_sm_v2_perf_gemm16_ptx: $(BUILD_DIR)/tb_sm_v2_perf_gemm16_ptx.vvp
\t@echo "========================================"
\t@echo "Running SM V2 PTX Performance Test (GEMM 16x16x16)"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16_ptx.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_PTX) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_PTX) $(filter %.v,$(SM_V2_SRCS))

# Performance microbenchmark (Tensor Core WMMA stream)
test_sm_v2_perf_tensor: $(BUILD_DIR)/tb_sm_v2_perf_tensor.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Tensor Core Performance Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_tensor.vvp

$(BUILD_DIR)/tb_sm_v2_perf_tensor.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_TC) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_TC) $(filter %.v,$(SM_V2_SRCS))

# Multi-warp Tensor Core backpressure test
test_sm_v2_perf_tensor_multiwarp: $(BUILD_DIR)/tb_sm_v2_perf_tensor_multiwarp.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Tensor Core Multi-warp Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_tensor_multiwarp.vvp

$(BUILD_DIR)/tb_sm_v2_perf_tensor_multiwarp.vvp: $(SM_V2_SRCS) $(TB_SM_V2_PERF_TC_MW) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_PERF_TC_MW) $(filter %.v,$(SM_V2_SRCS))

# Scheduler RAW hazard test (P2): verify scheduler blocks issue on RAW hazards
test_sm_v2_sched_raw_hazard: $(BUILD_DIR)/tb_sm_v2_sched_raw_hazard.vvp
\t@echo "========================================"
\t@echo "Running SM V2 Scheduler RAW Hazard Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_sched_raw_hazard.vvp

$(BUILD_DIR)/tb_sm_v2_sched_raw_hazard.vvp: $(SM_V2_SRCS) $(TB_SM_V2_SCHED_RAW) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(SM_V2_DEFINES) -o $@ $(TB_SM_V2_SCHED_RAW) $(filter %.v,$(SM_V2_SRCS))

#----------------------------------------------------------------------------
# Track 1-2 Performance Benchmarks (Atomics + Divergence)
#----------------------------------------------------------------------------
TB_BENCH_ATOMICS = $(TB_DIR)/tb_bench_atomics.v
TB_BENCH_DIVERGENCE = $(TB_DIR)/tb_bench_divergence.v
TB_ATOMIC_MINIMAL = $(TB_DIR)/tb_atomic_contention_minimal.v

# Atomic operations benchmark
bench_atomics: $(BUILD_DIR)/tb_bench_atomics.vvp tb/bench_atomics.hex
\t@echo "========================================"
\t@echo "Running Atomic Operations Benchmark"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_bench_atomics.vvp | tee bench_atomics.log
\t@echo "Output saved to $(BUILD_DIR)/bench_atomics.log"

$(BUILD_DIR)/tb_bench_atomics.vvp: $(RTL_SRCS) $(TB_BENCH_ATOMICS) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_BENCH_ATOMICS) $(filter %.v,$(RTL_SRCS))

tb/bench_atomics.hex: tb/bench_atomics.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Divergence handling benchmark
bench_divergence: $(BUILD_DIR)/tb_bench_divergence.vvp tb/bench_divergence.hex
\t@echo "========================================"
\t@echo "Running Branch Divergence Benchmark"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_bench_divergence.vvp | tee bench_divergence.log
\t@echo "Output saved to $(BUILD_DIR)/bench_divergence.log"

$(BUILD_DIR)/tb_bench_divergence.vvp: $(RTL_SRCS) $(TB_BENCH_DIVERGENCE) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_BENCH_DIVERGENCE) $(filter %.v,$(RTL_SRCS))

tb/bench_divergence.hex: tb/bench_divergence.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Minimal atomic contention test (faster)
bench_atomic_minimal: $(BUILD_DIR)/tb_atomic_contention_minimal.vvp asm/bench_atomic_minimal.hex
\t@echo "========================================"
\t@echo "Running Minimal Atomic Contention Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_atomic_contention_minimal.vvp

$(BUILD_DIR)/tb_atomic_contention_minimal.vvp: $(RTL_SRCS) $(TB_ATOMIC_MINIMAL) | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_ATOMIC_MINIMAL) $(filter %.v,$(RTL_SRCS))

asm/bench_atomic_minimal.hex: asm/bench_atomic_minimal.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Application-level PTX compilation
asm/bench_vecadd_atomic.hex: asm/bench_vecadd_atomic.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

asm/bench_matmul_sync.hex: asm/bench_matmul_sync.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

asm/bench_parallel_reduction.hex: asm/bench_parallel_reduction.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $@

# Compile all application benchmarks
bench_app_compile: asm/bench_vecadd_atomic.hex asm/bench_matmul_sync.hex asm/bench_parallel_reduction.hex
\t@echo "Application benchmarks compiled"

# Run all benchmarks
bench_all: bench_atomics bench_divergence
\t@echo "========================================"
\t@echo "All Benchmarks Complete"
\t@echo "========================================"

# Generate performance report from benchmark logs
# Performance dashboard: run benchmarks, generate IPC/stall/utilization report
dashboard:
\t@echo "========================================"
\t@echo "Running Performance Dashboard"
\t@echo "========================================"
\t$(PYTHON) tools/perf_dashboard.py --run \
\t\t\t\t--json $(BUILD_DIR)/perf_results.json \
\t\t\t\t--csv $(BUILD_DIR)/perf_results.csv \
\t\t\t\t-o docs/PERF_DASHBOARD.md
\t@echo "Dashboard: docs/PERF_DASHBOARD.md"
\t@echo "JSON:      $(BUILD_DIR)/perf_results.json"
\t@echo "CSV:       $(BUILD_DIR)/perf_results.csv"

# Dashboard with regression check against baseline
dashboard-check:
\t@echo "========================================"
\t@echo "Performance Dashboard + Regression Check"
\t@echo "========================================"
\t$(PYTHON) tools/perf_dashboard.py --run \
\t\t\t\t--json $(BUILD_DIR)/perf_results.json \
\t\t\t\t--csv $(BUILD_DIR)/perf_results.csv \
\t\t\t\t--baseline $(BUILD_DIR)/perf_baseline.json \
\t\t\t\t-o docs/PERF_DASHBOARD.md

# Save current results as new baseline
dashboard-baseline:
\t@echo "Saving current results as baseline..."
\tcp $(BUILD_DIR)/perf_results.json $(BUILD_DIR)/perf_baseline.json
\t@echo "Baseline saved: $(BUILD_DIR)/perf_baseline.json"

perf_report:
\t@echo "========================================"
\t@echo "Generating Performance Report"
\t@echo "========================================"
\t$(PYTHON) scripts/perf_analysis.py \
\t\t\t\t--parse $(BUILD_DIR)/bench_atomics.log \
\t\t\t\t--parse $(BUILD_DIR)/bench_divergence.log \
\t\t\t\t-o docs/PERFORMANCE_REPORT.md
\t@echo "Report saved to docs/PERFORMANCE_REPORT.md"

#----------------------------------------------------------------------------
# è¿¡Œæ‰€æœ‰æµ‹è¯•
#----------------------------------------------------------------------------
test: test_alu test_mul test_decoder test_regfile test_smem test_warp test_warp_ops test_video_unit test_tensor_core_e2e test_texture_unit
\t@echo "========================================"
\t@echo "All Unit Tests Completed"
\t@echo "========================================"

test_ptx: $(BUILD_DIR)
\t@echo "========================================"
\t@echo "Running PTX Comprehensive Test Suite"
\t@echo "========================================"
\t$(IVERILOG) -g2012 $(INCLUDES) -o $(BUILD_DIR)/tb_ptx_tests $(TB_DIR)/tb_ptx_tests.v $(filter %.v,$(RTL_SRCS))
\tcd $(BUILD_DIR) && $(VVP) tb_ptx_tests

test_all: test test_vector_add test_multi_sm sim
\t@echo "========================================"
\t@echo "All Tests Completed (Unit + Integration)"
\t@echo "========================================"

#----------------------------------------------------------------------------
# æ³¢å½¢æŸ¥çœ‹
#----------------------------------------------------------------------------
wave: sim
\t$(GTKWAVE) $(BUILD_DIR)/tb_ralph_gpu.vcd &

#----------------------------------------------------------------------------
# PTXæ±‡ç¼–
#----------------------------------------------------------------------------
assemble: $(EXAMPLES_DIR)/vector_add.ptx
\t$(PYTHON) $(TOOLS_DIR)/ptx_assembler.py $< -o $(BUILD_DIR)/vector_add.hex
\t@echo "Assembled to $(BUILD_DIR)/vector_add.hex"

#----------------------------------------------------------------------------
# è¯­æ³•æ£€æŸ¥
#----------------------------------------------------------------------------
# Performance Benchmarks (RALPH-7)
#----------------------------------------------------------------------------
PERF_RTL = $(shell find $(RTL_DIR) -name '*.v' | sort)

perf: $(BUILD_DIR)
\t@echo "============================================================"
\t@echo "RalphGPU Performance Benchmarks"
\t@echo "============================================================"
\t@echo ""
\t@echo "--- Building MatMul 4x4 ---"
\t$(IVERILOG) -g2012 $(INCLUDES) -DSM_V2 -DSIMULATION -o $(BUILD_DIR)/tb_matmul_perf.vvp $(PERF_RTL) $(TB_DIR)/tb_matmul_4x4_fp16_gpu_top.v
\t@echo "--- Running MatMul 4x4 ---"
\tcd $(BUILD_DIR) && $(VVP) tb_matmul_perf.vvp > matmul_perf.log 2>&1
\t$(PYTHON) $(TOOLS_DIR)/perf_report.py $(BUILD_DIR)/matmul_perf.log "FP16 MatMul 4x4"
\t@echo ""
\t@echo "--- Building Tiny MLP ---"
\t$(IVERILOG) -g2012 $(INCLUDES) -DSM_V2 -DSIMULATION -o $(BUILD_DIR)/tb_mlp_perf.vvp $(PERF_RTL) $(TB_DIR)/tb_tiny_mlp.v
\t@echo "--- Running Tiny MLP ---"
\tcd $(BUILD_DIR) && $(VVP) tb_mlp_perf.vvp > mlp_perf.log 2>&1
\t$(PYTHON) $(TOOLS_DIR)/perf_report.py $(BUILD_DIR)/mlp_perf.log "Tiny MLP (4->4->1)"
\t@echo ""
\t@echo "============================================================"
\t@echo "Benchmarks Complete"
\t@echo "============================================================"

#----------------------------------------------------------------------------
lint:
\t$(VERILATOR) --lint-only --top ralph_gpu_top -Wall \
\t\t\t\t-Wno-fatal -Wno-BLKLOOPINIT \
\t\t\t\t-Wno-DECLFILENAME -Wno-PINCONNECTEMPTY \
\t\t\t\t-Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
\t\t\t\t-Wno-UNOPTFLAT \
\t\t\t\t$(INCLUDES) $(filter %.v,$(RTL_SRCS))

#----------------------------------------------------------------------------
# ç»¼åæ£€æŸ¥ (Yosys)
#----------------------------------------------------------------------------
synth: $(BUILD_DIR)
\t$(YOSYS) -q -p "read_verilog -sv $(filter %.v,$(RTL_SRCS)); hierarchy -check -top ralph_gpu_top; proc; opt; check -assert" > $(BUILD_DIR)/yosys_synth.log 2>&1

#----------------------------------------------------------------------------
# æ¸…ç
#----------------------------------------------------------------------------
clean:
\trm -rf $(BUILD_DIR)
\trm -f *.vcd

#----------------------------------------------------------------------------
# å¸®åŠ©
#----------------------------------------------------------------------------
help:
\t@echo "RalphGPU - CUDA/PTX Compatible GPU IP"
\t@echo ""
\t@echo "Main Targets:"
\t@echo "  all           - Build and run simulation (default)"
\t@echo "  sim           - Run RTL simulation with Icarus Verilog"
\t@echo "  wave          - Open waveform viewer (GTKWave)"
\t@echo "  assemble      - Assemble PTX example to machine code"
\t@echo "  lint          - Run Verilator lint check"
\t@echo "  synth         - Run Yosys synthesis/check flow"
\t@echo "  clean         - Remove build artifacts"
\t@echo "  help          - Show this help message"
\t@echo ""
\t@echo "Test Targets:"
\t@echo "  test          - Run all unit tests"
\t@echo "  test_l1_data_cache - Test L1 data cache hit/miss/LRU/bank conflict"
\t@echo "  test_all      - Run all tests (unit + integration)"
\t@echo ""
\t@echo "Unit Tests:"
\t@echo "  test_alu      - Test ALU operations"
\t@echo "  test_mul      - Test multiply unit"
\t@echo "  test_decoder  - Test instruction decoder"
\t@echo "  test_regfile  - Test register file"
\t@echo "  test_smem     - Test shared memory"
\t@echo "  test_warp     - Test warp scheduler"
\t@echo "  test_tensor_core_fp4 - Tensor Core FP4 sanity test"
\t@echo ""
\t@echo "Integration Tests:"
\t@echo "  test_vector_add - Test vector addition kernel"
\t@echo "  test_multi_sm   - Test multi-SM parallel execution"
\t@echo "  test_sm_v2_perf - SM V2 FP32 FMA performance microbenchmark"
\t@echo "  test_sm_v2_perf_gemm16_ptx - SM V2 PTX-driven GEMM 16x16x16 microbenchmark"
\t@echo "  test_sm_v2_perf_gemm16_wmma_ptx - SM V2 PTX WMMA GEMM 16x16x16 path test"
\t@echo "  test_sm_v2_perf_gemm64_wgmma_ptx - SM V2 PTX WGMMA GEMM 64x8x16 path test"
\t@echo "  test_sm_v2_perf_tensor - SM V2 Tensor Core WMMA microbenchmark"
\t@echo "  test_sm_v2_perf_tensor_multiwarp - SM V2 Tensor Core multi-warp test"
\t@echo ""
\t@echo "Performance Benchmarks (Track 1-2):"
\t@echo "  bench_atomics       - Run atomic operations benchmark"
\t@echo "  bench_divergence    - Run branch divergence benchmark"
\t@echo "  bench_atomic_minimal - Quick atomic contention test"
\t@echo "  bench_app_compile   - Compile application-level benchmarks"
\t@echo "  bench_all           - Run all benchmarks"
\t@echo "  perf_report         - Generate PERFORMANCE_REPORT.md from logs"
\t@echo "  dashboard            - Run perf benchmarks + generate dashboard (IPC/stall/util)"
\t@echo "  dashboard-check      - Dashboard + regression check vs baseline"
\t@echo "  dashboard-baseline   - Save current results as new baseline"
\t@echo ""
\t@echo "Directory structure:"
\t@echo "  rtl/      - RTL source files"
\t@echo "  tb/       - Testbenches"
\t@echo "  tools/    - Assembler and other tools"
\t@echo "  examples/ - PTX example programs"
\t@echo "  doc/      - Documentation"
\t@echo "  build/    - Build outputs"

#============================================================================
# é…ç½®é€‰é¡¹ (å¯é€šè¿‡makeå‚æ•°è¦†ç›–)
#============================================================================
# NUM_SM=4 make sim  - ä½¿ç”¨4ä¸ªSMä»¿çœŸ
# æ³¨æ„: éœè¦ä¿®æ”¹gpu_defines.vhä¸­çš„å‚æ•°

# warp_inst_valid_d1 stall scenario test
test_warp_valid_d1: $(BUILD_DIR)/tb_warp_inst_valid_d1.vvp
\t@echo "========================================"
\t@echo "Running warp_inst_valid_d1 Stall Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_warp_inst_valid_d1.vvp

$(BUILD_DIR)/tb_warp_inst_valid_d1.vvp: $(TB_DIR)/tb_warp_inst_valid_d1.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_DIR)/tb_warp_inst_valid_d1.v


# Performance microbenchmark (PTX-driven WMMA GEMM stream)
test_sm_v2_perf_gemm16_wmma_ptx: programs/gemm16_wmma.hex $(BUILD_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.vvp
\t@echo "========================================"
\t@echo "Running SM V2 PTX WMMA GEMM Path Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm16_wmma_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.vvp: $(SM_V2_SRCS) $(TB_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_DIR)/tb_sm_v2_perf_gemm16_wmma_ptx.v $(filter %.v,$(RTL_SRCS))

programs/gemm16_wmma.hex: tests/gemm16_wmma.ptx tools/ptx_assembler.py
\t$(PYTHON) tools/ptx_assembler.py tests/gemm16_wmma.ptx -o programs/gemm16_wmma.hex


# Performance microbenchmark (PTX-driven WGMMA GEMM stream)
test_sm_v2_perf_gemm64_wgmma_ptx: programs/gemm64_wgmma.hex $(BUILD_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.vvp
\t@echo "========================================"
\t@echo "Running SM V2 PTX WGMMA GEMM Path Test"
\t@echo "========================================"
\tcd $(BUILD_DIR) && $(VVP) tb_sm_v2_perf_gemm64_wgmma_ptx.vvp

$(BUILD_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.vvp: $(SM_V2_SRCS) $(TB_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.v | $(BUILD_DIR)
\t$(IVERILOG) -g2012 $(INCLUDES) $(RTL_DEFINES) -o $@ $(TB_DIR)/tb_sm_v2_perf_gemm64_wgmma_ptx.v $(filter %.v,$(RTL_SRCS))

programs/gemm64_wgmma.hex: tests/gemm64_wgmma.ptx tools/ptx_assembler.py
\t$(PYTHON) tools/ptx_assembler.py tests/gemm64_wgmma.ptx -o programs/gemm64_wgmma.hex
