# Atomic Contention Analysis

## Scope
This note summarizes the atomic unit review and the attempt to create a minimal atomic contention testbench. The goal was to validate correctness under multi-warp contention (all threads incrementing the same address). The testbench did not complete, so the findings below focus on code review and the partial test attempt.

## 1) atomic_unit.v review findings

### High-severity issues
1) **No backpressure/queueing on `req_valid`**
   - `atomic_unit` only samples `req_valid` in `IDLE`. Any request asserted while `busy` is ignored.
   - Under contention (multiple warps issuing atomics close together), upstream must perfectly stall on `busy` or requests can be dropped.
   - Risk: lost atomic updates, non-deterministic results under contention.

2) **Cross-warp atomicity depends entirely on external arbitration**
   - The unit serializes lanes **within a warp**, but provides no global locking across warps/SMs.
   - Correctness under contention is only guaranteed if the memory system serializes all atomic accesses to the same address.
   - Risk: inter-warp races and lost updates if the external memory path does not strictly serialize atomics.

### Additional concerns (Medium/Low)
- **Handshake assumptions**: `mem_req` is pulsed in `READ_REQ` and `WRITE_REQ`, while `mem_ready` is only sampled later in `*_WAIT`. If the memory model uses a pulse-style ready, it can be missed and stall indefinitely.
- **`mem_rdata` shape mismatch**: the unit uses lane-indexed slices of `mem_rdata` even though a single `mem_addr` is issued. This is only correct if the memory interface returns a full lane vector per single-lane access (unlikely for global memory).
- **`mem_shared` unused**: shared vs global atomic behavior is not differentiated inside the unit.

## 2) Minimal atomic contention testbench attempt

### Testbench created
- **File:** `tb/tb_atomic_contention_minimal.v`
- **Goal:** Launch 4 warps (128 threads), run the existing PTX test `test_23_mem_consistency_atomicity` (atomic increment + barrier), and check result at `0x2000`.
- **Instruction source:** `sim/test_23_mem_consistency_atomicity.hex` (link to `asm/ptx_comprehensive_tests/test_23_mem_consistency_atomicity.ptx`).

### Issues encountered
- The simulation **hangs with no output** (even the initial `$display`), and `/tmp/atomic_contention_minimal.log` is empty.
- Early logs (from a prior run before modifications) showed `warp_valid=0000` and no kernel launch activity.
- Launch sequence in the testbench mirrors `tb_ptx_minimal.v` (CSR writes for grid/block dims + `GPU_CONTROL=1`), but the kernel still fails to start.
- `vvp` appears to run but produces no stdout; likely buffering or a stalled sim at time 0.

## Recommendations / Next Steps

1) **Add explicit backpressure/queueing to `atomic_unit`**
   - Introduce `req_ready` or a small FIFO to avoid dropping atomics when `busy`.

2) **Clarify/align memory interface for atomics**
   - Make `mem_rdata` a single 32-bit value for the addressed lane, or issue per-lane addresses with matching per-lane responses.
   - Ensure `mem_ready`/`mem_req` handshake is level-based and cannot be missed.

3) **Fix/verify kernel launch in the contention testbench**
   - Compare with `tb_ptx_minimal.v` and instrument kernel start signals (`kernel_start`, `warp_valid`, `sm_kernel_start`) to confirm CSR writes are applied.
   - If needed, reuse the proven `tb_ptx_minimal.v` scaffolding and swap the program memory to `test_23_mem_consistency_atomicity.hex`.

4) **Run contention test after launch fix**
   - Expected result: PASS marker `0xCAFE` at `0x2000` and counter at `0x1000` equals `ntid.x` (128 threads).
   - If it fails, likely due to contention serialization in `atomic_unit` or memory pipeline.

---

This captures the current understanding: atomic correctness under contention is not guaranteed by the unit itself, and the minimal contention testbench needs a launch/visibility fix before it can validate behavior.
