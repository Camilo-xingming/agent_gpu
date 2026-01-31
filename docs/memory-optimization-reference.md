# Memory System Optimization Reference

## Source
Article: "Agentic Personal Knowledge Management with OpenClaw, PARA, and QMD"
Author: Nat Eliason (@nateliason)
Date: 2026-01-31

## Key Concepts for RalphGPU Memory Optimization

### Three-Layer Memory Architecture

Inspired by human memory systems:

1. **Knowledge Graph (Long-term Declarative Memory)**
   - Structured facts about entities and relationships
   - Persistent, organized storage
   - PARA framework (Projects/Areas/Resources/Archives)

2. **Daily Notes (Episodic Memory)**
   - Timeline of events
   - Raw logs of what happened when
   - Chronological records

3. **Tacit Knowledge (Procedural Memory)**
   - Patterns and preferences
   - "How things work"
   - Operational rules

### Tiered Retrieval Strategy

**Problem:** Loading everything wastes bandwidth and latency

**Solution:** Two-tier access pattern
- **Tier 1:** Summary/metadata (fast, always loaded)
- **Tier 2:** Full detail (only load when needed)

**Analogy for GPU Memory:**
- L1 cache = Summary/frequently accessed data
- L2/DRAM = Full data store
- Only fetch detailed data on cache miss

### Memory Hierarchy Principles

1. **Structured Organization**
   - Not a flat list — use hierarchical structure
   - Clear categories (Projects/Areas/Resources/Archives)
   - Everything has exactly one place

2. **Atomic Facts with Metadata**
   - Each fact is self-contained
   - Includes: category, timestamp, source, status, relationships
   - Never delete — mark as superseded

3. **Access Patterns Tracking**
   - Track `lastAccessed` and `accessCount`
   - Enable intelligent prefetching
   - Identify "hot" vs "cold" data

## Applying to RalphGPU Memory System

### Current Issues (from Codex analysis)
- Global memory accesses serialized per lane
- No real coalescing
- L1D cache ports unused
- Limited outstanding requests

### Optimization Ideas Inspired by Article

#### 1. Tiered Memory Access
```verilog
// Instead of always fetching full cache line:
// Tier 1: Fetch metadata/summary (tag, validity, access count)
// Tier 2: Fetch full data only if needed

// Example structure:
struct cache_line_metadata {
  valid,
  tag,
  last_access_time,
  access_count,
  dirty
}

// Access pattern:
// 1. Check metadata (fast, small)
// 2. Only load full line on metadata hit
```

#### 2. Access Pattern Tracking
```verilog
// Track per-address access patterns
// Use to predict future accesses

reg [31:0] access_count [0:CACHE_LINES-1];
reg [31:0] last_access_time [0:CACHE_LINES-1];

// Prefetch logic:
// If access_count > threshold && (current_time - last_access) < window:
//   Prefetch related addresses
```

#### 3. Hierarchical Organization
```verilog
// Instead of flat address space:
// Organize memory into zones:
// - "Hot" zone (frequently accessed) → L1
// - "Warm" zone (occasionally accessed) → L2
// - "Cold" zone (rarely accessed) → DRAM

// Automatic migration based on access_count
```

#### 4. Atomic Facts = Memory Transactions
```verilog
// Each memory transaction carries metadata:
struct memory_request {
  addr,
  data,
  timestamp,        // When issued
  source_warp,      // Which warp
  access_count,     // How many times this addr accessed
  related_addrs     // Prefetch hints
}
```

#### 5. Coalescing with Context
```verilog
// Don't just coalesce by address proximity
// Use access patterns to decide what to coalesce:

// Traditional: Coalesce sequential addresses
// Enhanced: Coalesce based on:
//   - Temporal locality (accessed together in time)
//   - Causal relationships (B always follows A)
//   - Warp behavior patterns
```

## Memory Decay & Access Tracking

### Three-Tier System (Hot/Warm/Cold)

**Hot** (accessed in last 7 days)
- Kept in L1 cache or fast memory
- First-class citizen, always checked first
- Highest prefetch priority

**Warm** (accessed 8-30 days ago)
- Kept in L2 cache
- Available but not front-of-mind
- Medium prefetch priority

**Cold** (not accessed in 30+ days)
- Moved to DRAM or evicted
- Still retrievable via search/miss
- Accessing a cold line "reheats" it → back to Hot

### Frequency Resistance

High `accessCount` resists decay:
```
eviction_score = recency_weight / (1 + log(accessCount))
```

Lines accessed frequently stay warm even if not accessed recently.

### Metadata Tracking (per cache line)

```verilog
struct cache_line_metadata {
  tag,
  valid,
  dirty,
  last_access_cycle,   // When last accessed
  access_count,        // How many times accessed
  tier,                // Hot (0) / Warm (1) / Cold (2)
  superseded_by        // Pointer to newer version (if any)
}
```

### The No-Deletion Rule

**Never delete cache lines — mark as superseded instead**

When cache line is updated:
1. Old line: `valid = 0`, `superseded_by = <new_line_index>`
2. New line: created with fresh data
3. Keep chain: old → new → newer

Benefits:
- Full history preserved
- Can trace data evolution
- Debugging: see what changed when
- Rollback possible

## Practical Takeaways for Current Optimization

### 1. Implement metadata-first access
```verilog
// Stage 1: Load metadata (tag, valid, tier, access_count)
// Stage 2: Only load data if metadata hit && tier == Hot/Warm
```
Saves bandwidth when data isn't needed.

### 2. Track access patterns
```verilog
always @(posedge clk) begin
  if (cache_hit) begin
    access_count[line_idx] <= access_count[line_idx] + 1;
    last_access_cycle[line_idx] <= current_cycle;
  end
end
```

### 3. Hierarchical memory zones with automatic migration
```verilog
// Weekly/periodic rebalance:
for (each cache line) {
  age = current_cycle - last_access_cycle;
  
  if (age < 7_cycles) 
    tier = HOT;    // Promote to L1
  else if (age < 30_cycles) 
    tier = WARM;   // Keep in L2
  else
    tier = COLD;   // Demote to DRAM
    
  // Frequency resistance:
  if (access_count > HIGH_THRESHOLD)
    tier = min(tier, WARM);  // Don't let high-freq go cold
}
```

### 4. Rich transaction metadata
```verilog
struct memory_request {
  addr,
  data,
  timestamp,           // When issued
  source_warp_id,      // Which warp
  access_count_hint,   // Predicted access frequency
  related_addrs[3:0],  // Prefetch hints
  tier_hint            // Suggested placement tier
}
```

### 5. Never delete, mark superseded
```verilog
// On cache line update:
cache[old_idx].valid = 0;
cache[old_idx].superseded_by = new_idx;
cache[new_idx] = new_data;
cache[new_idx].valid = 1;

// Debugging: trace history
function trace_history(addr) {
  idx = find_latest(addr);
  while (cache[idx].superseded_by != NULL) {
    print($time, ":", cache[idx].data);
    idx = cache[idx].superseded_by;
  }
endfunction
```

### 6. QMD-style Search Layer for Memory

**Problem:** GPU has thousands of addresses, scanning all is slow

**Solution:** Indexed retrieval
- **BM25-style keyword search:** Fast address lookup by pattern
- **Vector similarity:** Find "similar" memory access patterns
- **Combined query:** Best for prefetch prediction

Example:
```verilog
// Instead of scanning all cache lines:
// 1. Index: Build hash table of frequently accessed addresses
// 2. Query: "Find addresses accessed together with addr X"
// 3. Prefetch: Load top-K results into cache
```

## References
- PARA Method: Tiago Forte
- Original article: X/Twitter @nateliason
- Applied to GPU memory hierarchy

---

*Created: 2026-01-31*
*For: RalphGPU memory system optimization (Track 3)*
