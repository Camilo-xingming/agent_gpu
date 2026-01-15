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

# RTL源文件
RTL_SRCS = \
    $(RTL_DIR)/gpu_defines.vh \
    $(RTL_DIR)/alu.v \
    $(RTL_DIR)/mul_unit.v \
    $(RTL_DIR)/register_file.v \
    $(RTL_DIR)/decoder.v \
    $(RTL_DIR)/warp_scheduler.v \
    $(RTL_DIR)/shared_memory.v \
    $(RTL_DIR)/memory_interface.v \
    $(RTL_DIR)/streaming_multiprocessor.v \
    $(RTL_DIR)/ralph_gpu_top.v

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

# Include路径
INCLUDES = -I$(RTL_DIR)

#============================================================================
# 目标
#============================================================================

.PHONY: all sim wave clean assemble help test test_all
.PHONY: test_alu test_mul test_decoder test_regfile test_smem test_warp
.PHONY: test_vector_add test_multi_sm

all: $(BUILD_DIR) sim

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#----------------------------------------------------------------------------
# 主仿真
#----------------------------------------------------------------------------
sim: $(BUILD_DIR)/tb_ralph_gpu.vvp
	cd $(BUILD_DIR) && $(VVP) tb_ralph_gpu.vvp

$(BUILD_DIR)/tb_ralph_gpu.vvp: $(RTL_SRCS) $(TB_SRCS) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_SRCS) \
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
# 集成测试
#----------------------------------------------------------------------------
test_vector_add: $(BUILD_DIR)/tb_vector_add.vvp
	@echo "========================================"
	@echo "Running Vector Addition Integration Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_vector_add.vvp

$(BUILD_DIR)/tb_vector_add.vvp: $(RTL_SRCS) $(TB_VADD) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_VADD) $(filter %.v,$(RTL_SRCS))

test_multi_sm: $(BUILD_DIR)/tb_multi_sm.vvp
	@echo "========================================"
	@echo "Running Multi-SM Parallel Test"
	@echo "========================================"
	cd $(BUILD_DIR) && $(VVP) tb_multi_sm.vvp

$(BUILD_DIR)/tb_multi_sm.vvp: $(RTL_SRCS) $(TB_MULTI) | $(BUILD_DIR)
	$(IVERILOG) -g2012 $(INCLUDES) -o $@ $(TB_MULTI) $(filter %.v,$(RTL_SRCS))

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
	@echo ""
	@echo "Integration Tests:"
	@echo "  test_vector_add - Test vector addition kernel"
	@echo "  test_multi_sm   - Test multi-SM parallel execution"
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
