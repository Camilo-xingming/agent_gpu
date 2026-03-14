//============================================================================
// RalphGPU - Memory Architecture Configuration
// Comprehensive memory hierarchy parameters based on GPU best practices
//============================================================================

`ifndef MEMORY_CONFIG_VH
`define MEMORY_CONFIG_VH

`include "gpu_config.vh"

//============================================================================
// GPU Configuration Profiles
// Select one profile or define custom parameters
//============================================================================
// `define GPU_PROFILE_LITE      // Low area / low power
// `define GPU_PROFILE_BALANCED  // Mid-range
// `define GPU_PROFILE_HPC       // High performance
// Legacy (still supported):
// `define GPU_CONFIG_EDGE
// `define GPU_CONFIG_MOBILE
// `define GPU_CONFIG_DESKTOP
// `define GPU_CONFIG_DATACENTER

// If nothing is selected, the default branch below maps to HPC-like values.

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
`else
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
`else
// L1 Data Cache Configuration (Per SM)
//============================================================================
// Best Practice: 32-128KB per SM, 4-8 way set associative
// NVIDIA: 128KB L1 (configurable with shared memory)

`ifdef GPU_PROFILE_LITE
    `define L1D_SIZE_KB         16      // 16KB L1 Data Cache
    `define L1D_WAYS            4       // 4-way set associative
    `define L1D_LINE_SIZE       32      // 32 bytes per cache line
`elsif GPU_CONFIG_EDGE
    `define L1D_SIZE_KB         16      // 16KB L1 Data Cache
    `define L1D_WAYS            4       // 4-way set associative
    `define L1D_LINE_SIZE       32      // 32 bytes per cache line
`elsif GPU_CONFIG_MOBILE
    `define L1D_SIZE_KB         32      // 32KB L1 Data Cache
    `define L1D_WAYS            4       // 4-way set associative
    `define L1D_LINE_SIZE       64      // 64 bytes per cache line
`elsif GPU_PROFILE_BALANCED
    `define L1D_SIZE_KB         64      // 64KB L1 Data Cache
    `define L1D_WAYS            8       // 8-way set associative
    `define L1D_LINE_SIZE       128     // 128 bytes per cache line
`elsif GPU_CONFIG_DESKTOP
    `define L1D_SIZE_KB         64      // 64KB L1 Data Cache
    `define L1D_WAYS            8       // 8-way set associative
    `define L1D_LINE_SIZE       128     // 128 bytes per cache line
`else // HPC / DATACENTER
    `define L1D_SIZE_KB         128     // 128KB L1 Data Cache
    `define L1D_WAYS            8       // 8-way set associative
    `define L1D_LINE_SIZE       128     // 128 bytes per cache line
`endif

// Derived L1D parameters
`define L1D_SIZE_BYTES      (`L1D_SIZE_KB * 1024)
`define L1D_NUM_SETS        (`L1D_SIZE_BYTES / (`L1D_WAYS * `L1D_LINE_SIZE))
`define L1D_INDEX_BITS      $clog2(`L1D_NUM_SETS)
`define L1D_OFFSET_BITS     $clog2(`L1D_LINE_SIZE)
`define L1D_TAG_BITS        (32 - `L1D_INDEX_BITS - `L1D_OFFSET_BITS)

// L1D MSHR (Miss Status Holding Registers)
`define L1D_MSHR_ENTRIES    8       // Non-blocking: 8 outstanding misses
`define L1D_MSHR_SUBENTRIES 4       // Coalescing: 4 requests per MSHR

// L1D Write Buffer
`define L1D_WCB_ENTRIES     8       // Write combining buffer entries
`define L1D_WCB_LINE_SIZE   32      // Bytes per WCB entry

// L1D Prefetcher
`define L1D_PREFETCH_DEPTH  4       // Prefetch queue depth
`define L1D_PREFETCH_DEGREE 2       // Prefetch 2 lines ahead

// L1D Timing (cycles)
`define L1D_HIT_LATENCY     2       // L1 hit latency
`define L1D_MISS_LATENCY    20      // L1 miss to L2 (if L2 hit)

//============================================================================
// L1 Instruction Cache Configuration (Per SM)
//============================================================================
// Best Practice: 32-64KB per SM, high hit rate for loops

`ifdef GPU_PROFILE_LITE
    `define L1I_SIZE_KB         8       // 8KB Instruction Cache
    `define L1I_WAYS            2       // 2-way set associative
    `define L1I_LINE_SIZE       64      // 64 bytes (16 instructions)
`elsif GPU_CONFIG_EDGE
    `define L1I_SIZE_KB         8       // 8KB Instruction Cache
    `define L1I_WAYS            2       // 2-way set associative
    `define L1I_LINE_SIZE       64      // 64 bytes (16 instructions)
`elsif GPU_CONFIG_MOBILE
    `define L1I_SIZE_KB         16      // 16KB Instruction Cache
    `define L1I_WAYS            4       // 4-way set associative
    `define L1I_LINE_SIZE       64      // 64 bytes
`else // BALANCED / HPC / DESKTOP / DATACENTER
    `define L1I_SIZE_KB         32      // 32KB Instruction Cache
    `define L1I_WAYS            4       // 4-way set associative
    `define L1I_LINE_SIZE       64      // 64 bytes
`endif

// Derived L1I parameters
`define L1I_SIZE_BYTES      (`L1I_SIZE_KB * 1024)
`define L1I_NUM_SETS        (`L1I_SIZE_BYTES / (`L1I_WAYS * `L1I_LINE_SIZE))
`define L1I_INDEX_BITS      $clog2(`L1I_NUM_SETS)
`define L1I_OFFSET_BITS     $clog2(`L1I_LINE_SIZE)
`define L1I_TAG_BITS        (32 - `L1I_INDEX_BITS - `L1I_OFFSET_BITS)

//============================================================================
// L2 Cache Configuration (Shared across all SMs)
//============================================================================
// Best Practice: 256KB - 8MB total, 8-16 way, banked
// NVIDIA H100: 50MB L2

`ifdef GPU_PROFILE_LITE
    `define L2_SIZE_KB          256     // 256KB total L2
    `define L2_NUM_BANKS        2       // 2 banks
    `define L2_WAYS             8       // 8-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`elsif GPU_CONFIG_EDGE
    `define L2_SIZE_KB          256     // 256KB total L2
    `define L2_NUM_BANKS        2       // 2 banks
    `define L2_WAYS             8       // 8-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`elsif GPU_CONFIG_MOBILE
    `define L2_SIZE_KB          512     // 512KB total L2
    `define L2_NUM_BANKS        4       // 4 banks
    `define L2_WAYS             8       // 8-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`elsif GPU_PROFILE_BALANCED
    `define L2_SIZE_KB          2048    // 2MB total L2
    `define L2_NUM_BANKS        8       // 8 banks
    `define L2_WAYS             16      // 16-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`elsif GPU_CONFIG_DESKTOP
    `define L2_SIZE_KB          2048    // 2MB total L2
    `define L2_NUM_BANKS        8       // 8 banks
    `define L2_WAYS             16      // 16-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`else // HPC / DATACENTER
    `define L2_SIZE_KB          4096    // 4MB total L2
    `define L2_NUM_BANKS        16      // 16 banks
    `define L2_WAYS             16      // 16-way set associative
    `define L2_LINE_SIZE        128     // 128 bytes per line
`endif

// Derived L2 parameters
`define L2_SIZE_BYTES       (`L2_SIZE_KB * 1024)
`define L2_SIZE_PER_BANK    (`L2_SIZE_BYTES / `L2_NUM_BANKS)
`define L2_SETS_PER_BANK    (`L2_SIZE_PER_BANK / (`L2_WAYS * `L2_LINE_SIZE))
`define L2_INDEX_BITS       $clog2(`L2_SETS_PER_BANK)
`define L2_BANK_BITS        $clog2(`L2_NUM_BANKS)
`define L2_OFFSET_BITS      $clog2(`L2_LINE_SIZE)
`define L2_TAG_BITS         (32 - `L2_INDEX_BITS - `L2_BANK_BITS - `L2_OFFSET_BITS)

// L2 MSHR
`define L2_MSHR_ENTRIES     16      // 16 outstanding misses per bank
`define L2_MSHR_SUBENTRIES  8       // 8 coalesced requests

// L2 Timing
`define L2_HIT_LATENCY      20      // L2 hit latency (cycles)
`define L2_MISS_LATENCY     100     // L2 miss to DRAM

// L2 ECC (optional)
`define L2_ECC_ENABLE       1       // Enable SECDED ECC
`define L2_ECC_BITS         8       // 8 ECC bits per 64-bit word

//============================================================================
// Shared Memory Configuration (Per SM)
//============================================================================
// Best Practice: 16-96KB configurable, 32 banks, conflict detection
// NVIDIA: 0-164KB configurable (combined with L1)

`ifdef GPU_PROFILE_LITE
    `define SMEM_SIZE_KB        16      // 16KB Shared Memory
    `define SMEM_NUM_BANKS      16      // 16 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`elsif GPU_CONFIG_EDGE
    `define SMEM_SIZE_KB        16      // 16KB Shared Memory
    `define SMEM_NUM_BANKS      16      // 16 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`elsif GPU_CONFIG_MOBILE
    `define SMEM_SIZE_KB        32      // 32KB Shared Memory
    `define SMEM_NUM_BANKS      32      // 32 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`elsif GPU_PROFILE_BALANCED
    `define SMEM_SIZE_KB        64      // 64KB Shared Memory
    `define SMEM_NUM_BANKS      32      // 32 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`elsif GPU_CONFIG_DESKTOP
    `define SMEM_SIZE_KB        64      // 64KB Shared Memory
    `define SMEM_NUM_BANKS      32      // 32 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`else // HPC / DATACENTER
    `define SMEM_SIZE_KB        96      // 96KB Shared Memory
    `define SMEM_NUM_BANKS      32      // 32 banks
    `define SMEM_BANK_WIDTH     32      // 32 bits per bank
`endif

// Derived Shared Memory parameters
`define SMEM_SIZE_BYTES     (`SMEM_SIZE_KB * 1024)
`define SMEM_ADDR_BITS      $clog2(`SMEM_SIZE_BYTES)
`define SMEM_BANK_BITS      $clog2(`SMEM_NUM_BANKS)
`define SMEM_DEPTH          (`SMEM_SIZE_BYTES / (`SMEM_NUM_BANKS * 4))

// Shared Memory Timing
`define SMEM_LATENCY        4       // No conflict: 4 cycles
`define SMEM_CONFLICT_PENALTY 1     // Per-way conflict penalty

//============================================================================
// Register File Configuration (Per SM)
//============================================================================
// Best Practice: 64KB-256KB per SM, multi-banked for parallel access
// NVIDIA: 256KB per SM

`ifdef GPU_PROFILE_LITE
    `define RF_SIZE_KB          32      // 32KB Register File per SM
    `define RF_NUM_BANKS        4       // 4 banks
    `define RF_PORTS_READ       4       // 4 read ports (A, B, C, Pred)
    `define RF_PORTS_WRITE      2       // 2 write ports
`elsif GPU_CONFIG_EDGE
    `define RF_SIZE_KB          32      // 32KB Register File per SM
    `define RF_NUM_BANKS        4       // 4 banks
    `define RF_PORTS_READ       4       // 4 read ports (A, B, C, Pred)
    `define RF_PORTS_WRITE      2       // 2 write ports
`elsif GPU_CONFIG_MOBILE
    `define RF_SIZE_KB          64      // 64KB Register File per SM
    `define RF_NUM_BANKS        8       // 8 banks
    `define RF_PORTS_READ       4       // 4 read ports
    `define RF_PORTS_WRITE      2       // 2 write ports
`else // BALANCED / HPC / DESKTOP / DATACENTER
    `define RF_SIZE_KB          128     // 128KB Register File per SM
    `define RF_NUM_BANKS        16      // 16 banks
    `define RF_PORTS_READ       6       // 6 read ports (for dual-issue)
    `define RF_PORTS_WRITE      4       // 4 write ports (for dual-issue)
`endif

// Register File structure
`define RF_REGS_PER_THREAD  32      // 32 architectural registers
`define RF_MAX_REGS         255     // Max allocatable registers
`define RF_REG_WIDTH        32      // 32-bit registers
`define RF_REG_ADDR_BITS    8       // 8-bit register address

// Derived RF parameters
`define RF_SIZE_BYTES       (`RF_SIZE_KB * 1024)
`define RF_TOTAL_REGS       (`RF_SIZE_BYTES / 4)  // 32-bit regs

// RF Timing
`define RF_READ_LATENCY     1       // 1 cycle read
`define RF_WRITE_LATENCY    1       // 1 cycle write

`endif

`endif

//============================================================================
// Constant Memory / Uniform Cache
//============================================================================
// Best Practice: 64KB constant memory, cached

`define CONST_MEM_SIZE_KB   64      // 64KB Constant Memory
`define CONST_CACHE_SIZE_KB 8       // 8KB Constant Cache per SM
`define CONST_CACHE_WAYS    4       // 4-way set associative
`define CONST_LINE_SIZE     32      // 32 bytes per line

`define CONST_MEM_SIZE      (`CONST_MEM_SIZE_KB * 1024)
`define CONST_ADDR_BITS     $clog2(`CONST_MEM_SIZE)

//============================================================================
// Texture Cache Configuration
//============================================================================
// Best Practice: 12-48KB per SM, optimized for 2D locality

`ifdef GPU_CONFIG_EDGE
    `define TEX_CACHE_SIZE_KB   12      // 12KB Texture Cache
    `define TEX_CACHE_WAYS      4       // 4-way
    `define TEX_LINE_SIZE       32      // 32 bytes
`else
    `define TEX_CACHE_SIZE_KB   48      // 48KB Texture Cache
    `define TEX_CACHE_WAYS      8       // 8-way
    `define TEX_LINE_SIZE       64      // 64 bytes
`endif

`define TEX_CACHE_SIZE      (`TEX_CACHE_SIZE_KB * 1024)
`define TEX_LATENCY         20      // Texture fetch latency

//============================================================================
// TLB Configuration (Translation Lookaside Buffer)
//============================================================================
// Best Practice: L1 TLB per SM, L2 TLB shared

// L1 TLB (Per SM)
`define L1_TLB_ENTRIES      32      // 32 entries per SM
`define L1_TLB_WAYS         4       // 4-way set associative
`define L1_TLB_PAGE_SIZE    4096    // 4KB base page
`define L1_TLB_LARGE_PAGE   (2*1024*1024)  // 2MB large page

// L2 TLB (Shared)
`define L2_TLB_ENTRIES      512     // 512 entries shared
`define L2_TLB_WAYS         8       // 8-way set associative

// TLB Timing
`define TLB_HIT_LATENCY     1       // TLB hit adds 1 cycle
`define TLB_MISS_LATENCY    50      // Page table walk

// Virtual Address Space
`define VA_BITS             48      // 48-bit virtual address
`define PA_BITS             40      // 40-bit physical address
`define PAGE_OFFSET_BITS    12      // 4KB page = 12 bits

//============================================================================
// Memory Controller Configuration
//============================================================================
// Best Practice: Wide interface, multiple channels

`ifdef GPU_PROFILE_LITE
    `define MEM_DATA_WIDTH      128     // 128-bit memory interface
    `define MEM_NUM_CHANNELS    1       // 1 memory channel
    `define MEM_BURST_LENGTH    8       // 8-beat burst
`elsif GPU_CONFIG_EDGE
    `define MEM_DATA_WIDTH      128     // 128-bit memory interface
    `define MEM_NUM_CHANNELS    1       // 1 memory channel
    `define MEM_BURST_LENGTH    8       // 8-beat burst
`elsif GPU_CONFIG_MOBILE
    `define MEM_DATA_WIDTH      256     // 256-bit memory interface
    `define MEM_NUM_CHANNELS    2       // 2 memory channels
    `define MEM_BURST_LENGTH    8       // 8-beat burst
`elsif GPU_PROFILE_BALANCED
    `define MEM_DATA_WIDTH      256     // 256-bit per channel
    `define MEM_NUM_CHANNELS    4       // 4 memory channels
    `define MEM_BURST_LENGTH    8       // 8-beat burst
`elsif GPU_CONFIG_DESKTOP
    `define MEM_DATA_WIDTH      256     // 256-bit per channel
    `define MEM_NUM_CHANNELS    4       // 4 memory channels
    `define MEM_BURST_LENGTH    8       // 8-beat burst
`else // HPC / DATACENTER
    `define MEM_DATA_WIDTH      512     // 512-bit per channel
    `define MEM_NUM_CHANNELS    8       // 8 memory channels (HBM)
    `define MEM_BURST_LENGTH    4       // 4-beat burst (HBM style)
`endif

// Memory Interface Type
`define MEM_TYPE_DDR4       0
`define MEM_TYPE_DDR5       1
`define MEM_TYPE_LPDDR5     2
`define MEM_TYPE_HBM2       3
`define MEM_TYPE_HBM3       4

`ifdef GPU_PROFILE_HPC
    `define MEM_TYPE            `MEM_TYPE_HBM3
`elsif GPU_CONFIG_DATACENTER
    `define MEM_TYPE            `MEM_TYPE_HBM3
`elsif GPU_PROFILE_BALANCED
    `define MEM_TYPE            `MEM_TYPE_DDR5
`elsif GPU_CONFIG_DESKTOP
    `define MEM_TYPE            `MEM_TYPE_DDR5
`else
    `define MEM_TYPE            `MEM_TYPE_LPDDR5
`endif

// Memory Timing (in memory clock cycles)
`define MEM_tCL             22      // CAS Latency
`define MEM_tRCD            22      // RAS to CAS delay
`define MEM_tRP             22      // Row Precharge
`define MEM_tRAS            52      // Row Active Time
`define MEM_tRC             74      // Row Cycle Time

// Derived bandwidth
`define MEM_TOTAL_WIDTH     (`MEM_DATA_WIDTH * `MEM_NUM_CHANNELS)
`define MEM_BURST_SIZE      (`MEM_DATA_WIDTH * `MEM_BURST_LENGTH / 8)  // Bytes

// Request Queue
`define MEM_REQ_QUEUE_DEPTH 64      // Outstanding memory requests
`define MEM_REORDER_DEPTH   32      // Reorder buffer entries

//============================================================================
// Memory Coalescing Unit
//============================================================================
`define COAL_WINDOW_SIZE    32      // Coalesce 32 threads
`define COAL_MAX_SEGMENTS   4       // Max memory segments per access
`define COAL_SEGMENT_SIZE   128     // 128-byte segment

//============================================================================
// Atomic Unit Configuration
//============================================================================
`define ATOMIC_QUEUE_DEPTH  16      // Atomic operation queue
`define ATOMIC_LATENCY      50      // Atomic operation latency

//============================================================================
// Summary Table (for documentation)
//============================================================================
// Profile      | L1D  | L1I  | L2    | SMEM | RF   | Channels
// -------------|------|------|-------|------|------|----------
// Lite         | 16KB | 8KB  | 256KB | 16KB | 32KB | 1x128b
// Balanced     | 64KB | 32KB | 2MB   | 64KB | 128KB| 4x256b
// HPC          | 128KB| 32KB | 4MB   | 96KB | 128KB| 8x512b
//
// Legacy Profiles:
// Edge         | 16KB | 8KB  | 256KB | 16KB | 32KB | 1x128b
// Mobile       | 32KB | 16KB | 512KB | 32KB | 64KB | 2x256b
// Desktop      | 64KB | 32KB | 2MB   | 64KB | 128KB| 4x256b
// Datacenter   | 128KB| 32KB | 4MB   | 96KB | 128KB| 8x512b

`endif // MEMORY_CONFIG_VH
