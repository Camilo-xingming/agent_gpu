# Issue #151: Command Processor Implementation Plan

## 1. Goal
Refactor the kernel dispatch logic from ralph_gpu_top.v into a dedicated command_processor.v module and implement a Command Queue (Ring Buffer) interface for host-to-GPU interaction, matching modern GPU architectures (NVIDIA-style).

## 2. Architecture Changes
- **Current:** Host writes CSRs -> SMs start. Single kernel at a time.
- **Proposed:** 
  - **Command Processor (CP):** A new module that sits between Host CSRs and SMs.
  - **Command Queue:** A ring buffer in Global Memory.
  - **Doorbell:** A CSR where Host writes the 'tail' pointer of the queue.
  - **Dispatcher:** CP fetches commands from Memory, parses them (Kernel Descriptor), and allocates CTAs to SMs.

## 3. Task Breakdown
### Phase 1: Refactoring (Current Session)
- [ ] Create `rtl/command_processor.v`.
- [ ] Move CSR handling and basic SM scheduler logic from `ralph_gpu_top.v` to `command_processor.v`.
- [ ] Re-verify existing kernels still work through the new module (backward compatibility).

### Phase 2: Command Queue Implementation
- [ ] Implement AXI master interface in CP for command fetching.
- [ ] Add Command Queue CSRs (Head, Tail, Base Address).
- [ ] Implement Command Parser for Kernel Descriptors.
- [ ] Support Multi-kernel queuing.

## 4. Proposed Interface for `command_processor.v`
```verilog
module command_processor #(
    parameter NUM_SM = 2
) (
    input  wire        clk,
    input  wire        rst_n,

    // Host CSR Interface
    input  wire        csr_wr_en,
    input  wire [11:0] csr_addr,
    input  wire [31:0] csr_wr_data,
    output wire [31:0] csr_rd_data,

    // SM Control Interface
    output wire [NUM_SM-1:0] sm_kernel_start,
    input  wire [NUM_SM-1:0] sm_done,
    output wire [31:0]       sm_block_id_x [NUM_SM-1:0],
    output wire [31:0]       sm_block_id_y [NUM_SM-1:0],
    output wire [31:0]       sm_block_id_z [NUM_SM-1:0],
    output wire [31:0]       kernel_pc,
    output wire [31:0]       grid_dim_x, grid_dim_y, grid_dim_z,
    output wire [31:0]       block_dim_x, block_dim_y, block_dim_z,

    // Global Memory Interface (for Command Queue)
    output wire        m_axi_req,
    output wire [31:0] m_axi_addr,
    input  wire [63:0] m_axi_data,
    input  wire        m_axi_valid
);
```

## 5. Verification Plan
- Unit test for `command_processor.v` using a mock SM.
- Integration test with `ralph_gpu_top.v`.
- PTX smoke tests (Vector Add).
