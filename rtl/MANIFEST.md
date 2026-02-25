# RalphGPU RTL Manifest

| File | Description |
|------|-------------|
| advanced_scheduler.v | RalphGPU - Advanced Warp Scheduler |
| alu.v | RalphGPU - ALU (Arithmetic Logic Unit) |
| async_copy_engine.v | Async Copy Engine (cp.async) |
| atomic_unit.v | RalphGPU - Atomic Unit |
| blackwell_scheduler.v | RalphGPU - Blackwell-Style Multi-Scheduler |
| branch_predictor.v | Branch Predictor |
| cache_policy_unit.v | RalphGPU - Cache Policy Unit |
| chi_controller.v | RalphGPU - CHI (Coherent Hub Interface) Controller |
| cluster_barrier_unit.v | RalphGPU - Cluster Barrier Unit |
| command_processor.v | RalphGPU - Command Processor |
| control_flow_unit.v | RalphGPU - Control Flow Unit |
| cvt_unit.v | RalphGPU - CVT Unit (Type Conversion Unit) |
| decoder.v | RalphGPU - Instruction Decoder |
| dpx_unit.v | RalphGPU - DPX Unit (Dynamic Programming Extensions) |
| dual_issue_scheduler.v | RalphGPU - Dual Issue Warp Scheduler |
| fma_int32.v | RalphGPU - Integer FMA Unit (Fused Multiply-Add) |
| forwarding_unit.v | RalphGPU - Data Forwarding Unit |
| fp16_unit.v | FP16/BF16 Unit |
| fpu.v | RalphGPU - FPU (Floating-Point Unit) |
| fpu64.v | RalphGPU - FPU64 (Double-Precision Floating-Point Unit) |
| gpu_defines.vh | Global GPU Definitions and Parameters |
| gpu_config.vh | GPU Configuration Switches |
| griddep_unit.v | RalphGPU - Grid Dependency Control Unit |
| icache.v | RalphGPU - Instruction Cache (I-Cache) |
| l1_data_cache.v | RalphGPU - L1 Data Cache (Replay-Enabled) |
| l1_data_cache_optimized.v | RalphGPU - Optimized L1 Data Cache |
| l2_cache.v | RalphGPU - L2 Cache |
| l2_interconnect.v | RalphGPU - L2 Cache Slice Interconnect |
| lz4_decompressor.v | RalphGPU - LZ4 Hardware Decompressor |
| mbarrier_unit.v | RalphGPU - mbarrier Unit (Hopper+ Memory Barrier) |
| memory_coalescing_unit.v | RalphGPU - Memory Coalescing Unit |
| memory_config.vh | Memory Subsystem Configuration |
| memory_controller.v | RalphGPU - Memory Controller Interface |
| memory_controller_hbm.v | HBM Memory Controller with Real DRAM Latency |
| memory_interface.v | RalphGPU - Global Memory Interface |
| memory_interface_wide.v | RalphGPU - Wide Memory Interface |
| memory_qos.v | RalphGPU - Memory QoS and Bandwidth Management |
| mul_unit.v | RalphGPU - Multiply Unit |
| multimem_unit.v | RalphGPU - Multimem Unit |
| performance_counters.v | RalphGPU - Comprehensive Performance Counters |
| ralph_gpu_top.v | RalphGPU - Top Level Module (NVIDIA Hopper-Class) |
| reconvergence_stack.v | RalphGPU - Enhanced Reconvergence Stack |
| register_file.v | RalphGPU - Register File (Multi-Warp Support) |
| register_file_banked.v | Banked Register File |
| sfu.v | RalphGPU - SFU (Special Function Unit) |
| shared_memory.v | RalphGPU - Shared Memory |
| sm_fetch_pipeline.v | RalphGPU - SM Fetch Pipeline |
| sm_gmem_arbiter.v | Global Memory Request Arbiter |
| sm_special_reg.v | Special Register Execution Unit |
| sm_wbq_bank.v | RalphGPU - Writeback Queue Bank |
| sm_writeback_arbiter.v | Writeback Arbiter for Streaming Multiprocessor |
| st_bulk_unit.v | RalphGPU - Bulk Store Unit |
| stack_debug_unit.v | RalphGPU - Stack and Debug Unit |
| streaming_multiprocessor_v2.v | RalphGPU - Streaming Multiprocessor V2 |
| tensor_core.v | RalphGPU - Tensor Core |
| tensor_memory.v | RalphGPU - Tensor Memory (TMEM) Module |
| texture_unit.v | RalphGPU - Texture/Surface Unit |
| tlb.v | RalphGPU - Translation Lookaside Buffer (TLB) |
| tlb_enhanced.v | Enhanced TLB with Page Walker |
| tma_unit.v | RalphGPU - TMA Unit (Tensor Memory Accelerator) |
| video_unit.v | RalphGPU - Video Processing Unit |
| warp_collective_unit.v | RalphGPU - Warp Collective Unit |
| warp_scheduler.v | RalphGPU - Warp Scheduler |
| warp_shuffle.v | RalphGPU - Warp Shuffle Unit |
| wb_fifo.v | RalphGPU - Write-Back FIFO |
| wgmma.v | RalphGPU - WGMMA (Warpgroup Matrix Multiply-Accumulate) |
| wgmma_tile_engine.v | RalphGPU - WGMMA Tile Engine |
