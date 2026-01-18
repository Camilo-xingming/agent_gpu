# PTX ISA 9.1 Gap & Implementation Plan

Scope: Map PTX ISA 9.1 against current RalphGPU RTL/assembler support, identify missing/partial areas, and plan implementation.

## Execution Order & Gates
- Execute Phase A immediately, then run its directed/unit and SM benches before enabling any Phase B work.
- Start Phase B only after Phase A passes the gate; Phase C starts after Phase B’s gate.
- Verification plan below is sequenced to match these gates.

## Current Coverage Snapshot (2026-01-18)
- Implemented & wired: ALU (+setp), mul.lo/hi/mad.lo, mul.wide, mul24/mad24, mad.cc/madc (carry-in), bmsk/szext/fns/lop3/shf/cnot, integer div/rem (s/u 32b), basic branch/call/ret/exit, mov special regs (incl. laneid/warpid/smid/activemask), bar.sync, ld/st global/shared/param/const/local (+v2/v4), FP32/FP16/BF16/FP64 arithmetic, FP32 special funcs (rcp/sqrt/rsqrt/sin/cos/lg2/ex2/tanh), copysign/testp, CVT (incl. cvt.pack), ATOM/RED, SHFL/VOTE/REDUX, WMMA/MMA, MEMBAR.
- Partially integrated: cp.async/prefetch decoded; SM tracks cp.async with per-warp pending counters and fixed-latency drain (wait_group/all stalls warp) but no real LSU traffic yet; bar.sync clears once all valid warps reach barrier (single-SM scope).
- Decoded but not integrated: WGMMA load/store/mma_async, texture/surface ops, video ops, async/tensormap extras.
- Missing/partial hardware blocks: dp4a/dp2a still not wired to video unit; FP compare variants (half/mixed) still missing; sync primitives (bar.warp.sync, barrier.cluster, red.async, match.sync, griddepcontrol, elect.sync, mbarrier, tensormap.*); st.async/st.bulk/multimem.*, prefetch/applypriority/discard/createpolicy/isspacep/mapa/getctarank; stack ops; misc debug (brkpt/trap/nanosleep/pmevent/setmaxnreg); texture/surface/video not wired; WGMMA/5th-gen tensor path absent.

## Gaps by Category (vs PTX 9.1)
- Integer: dp4a/dp2a DONE (wired via ALU); bfind signed/unsigned parity unknown.
- FP: half/mixed FP compare variants missing; special math beyond current set still uncovered.
- Compare/select: integer/half `set` forms not distinct (setp/selp/slct present).
- Data move/convert: st.async/st.bulk/multimem.*, prefetch/applypriority/discard/createpolicy/isspacep/mapa/getctarank missing; cache-policy state machine absent.
- Sync/comm: missing bar.warp.sync, barrier.cluster, red.async, match.sync, griddepcontrol, elect.sync, mbarrier, tensormap.*; activemask opcode is present via special reg.
- Tensor/async: WGMMA/5th-gen tensor memory/ops not integrated; cp.async currently stubbed (fixed-latency counter, no LSU); prefetch is hint-only.
- Texture/surface/video: modules exist but SM wiring absent; tex/txq/suld/sust/sured and SIMD video ops unsupported.
- Stack/misc: alloca/stacksave/stackrestore, brkpt/trap/nanosleep/pmevent/setmaxnreg not handled.

## Implementation Plan
### Phase A – Correctness Fill-Ins (status: mostly complete)
1) Integer div/rem: DONE (s/u 32b, integrated into mul_unit + SM issue/WB).  
2) Mul/carry & 24-bit: DONE (mul24/mad24, mad.cc/madc, fns/szext/bmsk, dp4a/dp2a wired via ALU path).  
3) Logic/shift: DONE (lop3/shf/cnot).  
4) FP side: copysign/testp DONE; FP compare variants (half/mixed) still open; rcp.approx.ftz.f64 mapped to RCP.  
5) Special regs: activemask/laneid/warpid/smid DONE; integer `set` non-predicated forms TBD if decode requires.

**Phase A Verification & Gate**
- Assembler: encodings for new integer/logic/FP ops DONE; add FP compare (half/mixed) tests when implemented.  
- Directed: div/rem latency, mul24/mad24/mad.cc/madc, lop3/shf/cnot, copysign/testp, dp4a/dp2a all covered; add FP compare (half/mixed) benches when implemented.  
- SM integration: smoke passes with new ops; proceed to Phase B with noted residuals.

### Phase B – Memory & Sync Features
1) Async memory (owner: LSU): replace cp.async stub with real path. Options: (a) instantiate `async_copy_engine` (global→shared) and drive it from `OP_CPASYNC` with cache hints; (b) route through LSU queues with a cp.async tag. Implement commit/wait ordering (group counters), wire `cp_async_pending` to actual completion, and add st.async/st.bulk/multimem datapaths with write-completion tracking.  
2) Sync primitives (owner: control/scoreboard): add bar.warp.sync, barrier.cluster (multi-SM token), red.async, match.sync, griddepcontrol, elect.sync, mbarrier, tensormap.*. Extend scoreboard with async-group tokens and barrier IDs; ensure interaction with membar/atomics and cp.async wait_all.  
3) Decode & helpers (owner: decode/ALU/LSU): add mapa/getctarank/isspacep/createpolicy/applypriority/discard. Map policy ops to a small cache-policy state machine; map issapacep/mapa/getctarank to ALU/special-reg paths. (cvt.pack already done.)  
4) Prefetch: plumb cache hints to L1/L2 prefetch path; ensure no architectural side effects.

**Phase B Verification & Gate**
- Assembler: cp.async/st.async/prefetch/createpolicy/discard/isspacep/mapa/getctarank/bar.warp/barrier.cluster/red.async/match.sync/mbarrier/tensormap coverage tests.  
- Directed LSU/sync: cp.async group commit/wait (incl. wait_all), st.async ordering & fences, barrier.cluster topology, red.async accumulation, match.sync corner cases, mbarrier token reuse, tensormap control words.  
- SM/cluster integration: kernels mixing cp.async with compute; warp-level barrier microbench; cluster barrier microbench; async reduction under contention.  
- Gate: rerun Phase A suite + new LSU/sync/SM/cluster cases before Phase C.

### Phase C – Tensor/Graphics Extras
1) Tensor: wire WGMMA load/store/mma_async with fence/commit/wait; decide reuse vs dedicated tensor path; update scheduler/scoreboard for long tensor ops.  
2) Texture/surface/video: connect texture_unit + caches to SM issue/WB; enable tex/txq/suld/sust/sured; wire SIMD video ops (dp2a/dp4a variants, vadd/vmad) with saturation rules.  
3) Stack/misc: implement alloca/stacksave/stackrestore in LSU/control; add brkpt/trap/nanosleep/pmevent/setmaxnreg handling in control block with safe no-ops where hardware not modeled.

**Phase C Verification & Gate**
- Assembler: WGMMA/texture/surface/video/stack op encodings.  
- Directed tensor/texture: WGMMA throughput/latency, fence/commit/wait ordering; texture/surface correctness on cacheable/non-cacheable paths; video SIMD saturation cases.  
- SM integration: kernels mixing WGMMA with cp.async/st.async; texture + compute; stack save/restore under deep call.  
- Gate: full regression (Phase A+B suites) + new tensor/texture/video/stack benches before marking PTX 9.1 coverage complete.

## Verification Plan (rollup)
- Stage-gate per phase as above; no promotion without prior gate passing.  
- Extend `tools/ptx_assembler.py` for all new opcodes/operands; keep json specs in sync.  
- Maintain directed FU benches for each new datapath plus SM integration benches per feature area.  
- Keep nightly regression running 87+ legacy tests plus accumulating Phase A→C suites; track coverage deltas after each phase.
