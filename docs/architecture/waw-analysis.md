# WAW (Write-After-Write) Hazard Analysis

## Scoreboard Architecture

### SET Path (issue time)
- Per-scheduler loop: `scoreboard[warp][rd] <= 1'b1` on issue
- Both scheduler slots can SET in same cycle (different warps)
- Tensor ops: deferred SET via `tensor_sb_set_valid` (when push succeeds)

### CLEAR Path (writeback time)  
- **Single writeback port**: `wb_valid` → `scoreboard[wb_warp_id][wb_rd] <= 1'b0`
- Round-robin arbiter across 17 FU queues (ALU, MUL, FPU32, ..., tensor, mem, etc.)
- Only ONE register cleared per cycle

## WAW Scenarios

### Scenario 1: Same warp, same dest reg, different FUs
```
IADD R1, R2, R3    ; ALU writes R1, scoreboard[w][R1] = 1
IMUL R1, R4, R5    ; MUL writes R1, scoreboard[w][R1] = 1 (already set)
```
**Analysis:** 
- Both instructions SET scoreboard[w][R1] = 1 (idempotent, no conflict)
- MUL result may complete before ALU (different pipeline latency)
- First WB clears scoreboard[w][R1] = 0 → **PREMATURE CLEAR**
- Second WB clears again (no-op, already 0)
- **BUG: Instruction after second write could issue before second WB completes**

**Severity:** HIGH for correctness. The scoreboard uses a 1-bit busy flag, not a counter.
If two instructions write the same register, the first WB clears the busy bit even though
the second write is still in-flight.

### Scenario 2: Different warps, same reg
```
Warp 0: IADD R1, R2, R3
Warp 1: IADD R1, R4, R5
```
**Analysis:** Safe — scoreboard is per-warp (`scoreboard[0][R1]` vs `scoreboard[1][R1]`).

### Scenario 3: Same-cycle SET and CLEAR (NBA race)
```
Cycle N: issue IADD R1 → SET scoreboard[w][R1] = 1
Cycle N: wb prev IADD R1 → CLEAR scoreboard[w][R1] = 0
```
**Analysis:** Both are NBA in the same always block:
```verilog
// SET path (line 403)
scoreboard[issue_warp_r[sb_s]][warp_rd[issue_warp_r[sb_s]]] <= 1'b1;
// CLEAR path (line 445)  
scoreboard[wb_warp_id][wb_rd] <= 1'b0;
```
**Last NBA wins in Verilog.** Since CLEAR is AFTER SET in the code, CLEAR wins → **BUG**.
New instruction's scoreboard bit gets cleared by the old instruction's writeback.

### Scenario 4: Slot 0 and Slot 1 SET same warp/reg
Not possible in current design — each scheduler handles different warps
(warp_id % NUM_SCHEDULERS). Two schedulers cannot issue for the same warp.

## Risk Assessment

| Scenario | Risk | Current Impact |
|---|---|---|
| 1: Same warp, same reg, different FU | HIGH | Premature clear, data corruption |
| 2: Different warps, same reg | NONE | Per-warp isolation |
| 3: Same-cycle SET/CLEAR NBA race | HIGH | New instruction's bit cleared by old WB |
| 4: Cross-slot same warp | NONE | Impossible by design |

## Current Mitigation
- With `lane0_stall_raw = 1'b0`, WAW is irrelevant (no stalls at all)
- After P2 enables RAW stall, WAW becomes exploitable
- GPU workloads rarely write same reg from different FUs in same warp
- Warp interleaving reduces probability but doesn't eliminate

## Recommended Fixes (Post-P2)

### Option A: Reference counter (2-bit scoreboard)
Replace 1-bit busy with 2-bit counter:
```verilog
reg [1:0] scoreboard [0:NUM_WARPS-1][0:31];  // 0-3 outstanding writes
```
SET increments, CLEAR decrements. Stall when > 0.
**Cost:** 2× scoreboard storage (256 bits → 512 bits for 4 warps × 32 regs)

### Option B: SET-wins priority
Add explicit check: if SET and CLEAR target same warp/reg in same cycle, SET wins:
```verilog
if (wb_valid && wb_warp_id == issue_warp && wb_rd == issue_rd)
    scoreboard[wb_warp_id][wb_rd] <= 1'b1;  // SET wins over CLEAR
else if (wb_valid)
    scoreboard[wb_warp_id][wb_rd] <= 1'b0;
```
**Cost:** Combinational check, no extra storage. Fixes Scenario 3 only.

### Option C: Stall on WAW
If `scoreboard[warp][rd]` is already set when issuing a new write to same reg, stall:
```verilog
wire lane0_stall_waw = dec0_valid && warp_writes_reg && scoreboard[warp][rd];
```
**Cost:** 1 extra stall condition. Simplest, most conservative.

## Recommendation
Start with **Option C** (stall on WAW) — simplest, prevents all WAW hazards.
Add **Option B** (SET-wins) for the NBA race. Both are 1-2 line changes.
