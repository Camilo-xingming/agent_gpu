import os
content = """//============================================================================
// RalphGPU - Memory Architecture Configuration
//============================================================================

`ifndef MEMORY_CONFIG_VH
`define MEMORY_CONFIG_VH

`include "gpu_config.vh"

//============================================================================
// Minimal Parameters for Synthesis
//============================================================================
`ifdef SYNTHESIS
    `define L1D_SIZE_KB         1
    `define L1D_WAYS            1
    `define L1D_LINE_SIZE       128
    `define L1I_SIZE_KB         1
    `define L1I_WAYS            1
    `define L1I_LINE_SIZE       64
    `define L2_SIZE_KB          32
    `define L2_NUM_BANKS        2
    `define L2_WAYS             1
    `define L2_LINE_SIZE        128
    `define SMEM_SIZE_KB        1
    `define SMEM_NUM_BANKS      16
    `define RF_SIZE_KB          16
    `define RF_NUM_BANKS        4
    `define TEX_CACHE_SIZE_KB   1
    `define TEX_CACHE_WAYS      1
    `define TEX_LINE_SIZE       128
    `define SYNTH_REDUCED
`endif

//============================================================================
// L1 Data Cache
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_PROFILE_LITE
    `define L1D_SIZE_KB         16
    `define L1D_WAYS            4
    `define L1D_LINE_SIZE       32
`else
    `define L1D_SIZE_KB         64
    `define L1D_WAYS            8
    `define L1D_LINE_SIZE       128
`endif
`endif

`define L1D_SIZE_BYTES      (`L1D_SIZE_KB * 1024)
`define L1D_NUM_SETS        (`L1D_SIZE_BYTES / (`L1D_WAYS * `L1D_LINE_SIZE))
`define L1D_INDEX_BITS      $clog2(`L1D_NUM_SETS)
`define L1D_OFFSET_BITS     $clog2(`L1D_LINE_SIZE)
`define L1D_TAG_BITS        (32 - `L1D_INDEX_BITS - `L1D_OFFSET_BITS)

//============================================================================
// L1 Instruction Cache
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_PROFILE_LITE
    `define L1I_SIZE_KB         8
    `define L1I_WAYS            2
    `define L1I_LINE_SIZE       64
`else
    `define L1I_SIZE_KB         32
    `define L1I_WAYS            4
    `define L1I_LINE_SIZE       64
`endif
`endif

`define L1I_SIZE_BYTES      (`L1I_SIZE_KB * 1024)
`define L1I_NUM_SETS        (`L1I_SIZE_BYTES / (`L1I_WAYS * `L1I_LINE_SIZE))
`define L1I_INDEX_BITS      $clog2(`L1I_NUM_SETS)
`define L1I_OFFSET_BITS     $clog2(`L1I_LINE_SIZE)
`define L1I_TAG_BITS        (32 - `L1I_INDEX_BITS - `L1I_OFFSET_BITS)

//============================================================================
// L2 Cache
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_PROFILE_LITE
    `define L2_SIZE_KB          256
    `define L2_NUM_BANKS        2
    `define L2_WAYS             8
    `define L2_LINE_SIZE        128
`else
    `define L2_SIZE_KB          2048
    `define L2_NUM_BANKS        8
    `define L2_WAYS             16
    `define L2_LINE_SIZE        128
`endif
`endif

`define L2_SIZE_BYTES       (`L2_SIZE_KB * 1024)
`define L2_SIZE_PER_BANK    (`L2_SIZE_BYTES / `L2_NUM_BANKS)
`define L2_SETS_PER_BANK    (`L2_SIZE_PER_BANK / (`L2_WAYS * `L2_LINE_SIZE))
`define L2_INDEX_BITS       $clog2(`L2_SETS_PER_BANK)
`define L2_BANK_BITS        $clog2(`L2_NUM_BANKS)
`define L2_OFFSET_BITS      $clog2(`L2_LINE_SIZE)
`define L2_TAG_BITS         (32 - `L2_INDEX_BITS - `L2_BANK_BITS - `L2_OFFSET_BITS)

//============================================================================
// Shared Memory
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_PROFILE_LITE
    `define SMEM_SIZE_KB        16
    `define SMEM_NUM_BANKS      16
`else
    `define SMEM_SIZE_KB        64
    `define SMEM_NUM_BANKS      32
`endif
`endif

`define SMEM_SIZE_BYTES     (`SMEM_SIZE_KB * 1024)
`define SMEM_ADDR_BITS      $clog2(`SMEM_SIZE_BYTES)
`define SMEM_BANK_BITS      $clog2(`SMEM_NUM_BANKS)
`define SMEM_DEPTH          (`SMEM_SIZE_BYTES / (`SMEM_NUM_BANKS * 4))

//============================================================================
// Register File
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_PROFILE_LITE
    `define RF_SIZE_KB          32
    `define RF_NUM_BANKS        4
`else
    `define RF_SIZE_KB          128
    `define RF_NUM_BANKS        16
`endif
`endif

`define RF_SIZE_BYTES       (`RF_SIZE_KB * 1024)
`define RF_REGS_PER_THREAD  32
`define RF_TOTAL_REGS       (`RF_SIZE_BYTES / 4)

//============================================================================
// Texture Cache
//============================================================================
`ifndef SYNTHESIS
`ifdef GPU_CONFIG_EDGE
    `define TEX_CACHE_SIZE_KB   12
`else
    `define TEX_CACHE_SIZE_KB   48
`endif
`endif

`define TEX_CACHE_SIZE      (`TEX_CACHE_SIZE_KB * 1024)

//============================================================================
// Memory Controller
//============================================================================
`ifdef GPU_PROFILE_LITE
    `define MEM_DATA_WIDTH      128
    `define MEM_NUM_CHANNELS    1
`else
    `define MEM_DATA_WIDTH      256
    `define MEM_NUM_CHANNELS    4
`endif

`define MEM_TOTAL_WIDTH     (`MEM_DATA_WIDTH * `MEM_NUM_CHANNELS)

`endif // MEMORY_CONFIG_VH
"""
with open("rtl/memory_config.vh", "w") as f:
    f.write(content)
