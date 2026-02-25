# Memory Coalescing Unit (SM Level) Design Document

## 1. Overview
The Memory Coalescing Unit (MCU) is a critical component in RalphGPU\s memory hierarchy. Its primary responsibility is to optimize global and shared memory access by grouping requests from multiple threads within a single Warp into as few memory transactions as possible. This process is essential for achieving high memory bandwidth utilization, as it reduces the number of transactions sent to the L1 Data Cache and beyond.

## 2. Objective
- **Bandwidth Efficiency**: Minimize the number of memory transactions by coalescing requests to the same cache line.
- **Latency Hiding**: By reducing the number of requests, we reduce contention and improve overall throughput.
- **Simplicity**: Provide a unified interface for the Streaming Multiprocessor (SM) to handle Warp-level memory operations without managing individual thread addresses.

## 3. Architecture

### 3.1 Input/Output Ports
- **Warp Side (Input)**:
  - `req_valid`: Indicates a new Warp-level memory request.
  - `req_write`: Write enable for the request.
  - `req_addr[31:0] * 32`: 32 addresses (one for each thread).
  - `req_wdata[31:0] * 32`: 32 write data elements (one for each thread).
  - `req_mask[31:0]`: Active thread mask for the current Warp.
- **L1 Cache Side (Output)**:
  - `l1_req_valid`: Request to the L1 cache.
  - `l1_req_write`: Write enable for the L1 request.
  - `l1_req_addr`: Base address of the cache line being accessed.
  - `l1_req_wdata`: Full cache line write data (128 bytes / 1024 bits).
  - `l1_req_wmask`: Byte-level write mask for the cache line.
- **Response Side (Output)**:
  - `resp_rdata[31:0] * 32`: Collected read data for each thread.
  - `resp_valid`: Indicates that the Warp request is complete.
  - `ready`: Indicates the unit is ready to accept a new Warp request.

### 3.2 Internal Logic
1.  **Address Analysis**: Identify unique cache line addresses from the 32 input thread addresses.
2.  **Request Coalescing**:
    - Iterate through active threads.
    - Match each thread\s address to a set of unique cache lines (typically up to 4 per transaction).
    - Map each thread to its corresponding cache line and byte offset within that line.
3.  **State Machine (FSM)**:
    - **IDLE**: Wait for `req_valid`.
    - **ANALYZE**: Determine unique cache lines and thread mappings.
    - **ISSUE**: Sequentially issue L1 requests for each unique cache line found.
    - **WAIT**: Wait for L1 cache responses.
    - **COLLECT**: Distribute response data from cache lines back to the individual thread\s output registers.
    - **DONE**: Assert `resp_valid` and return to `IDLE`.

### 3.3 Coalescing Rules
- Current implementation supports up to **4 unique cache lines** per Warp-level transaction.
- If more than 4 cache lines are needed, the request may be stalled or handled in multiple passes (depending on complexity).
- Memory addresses must be aligned within the cache line to be coalesced efficiently.

## 4. Implementation Details
The RTL implementation will use a parameterized approach:
- `THREADS`: 32 (standard Warp size)
- `DATA_WIDTH`: 32 bits
- `ADDR_WIDTH`: 32 bits
- `CACHE_LINE_SIZE`: 128 bytes (1024 bits)

## 5. Performance Metrics
- **Coalescing Ratio**: `Total Thread Requests / Total L1 Transactions`. Higher is better.
- **Stall Cycles**: Number of cycles the Warp is stalled waiting for memory transactions to complete.
