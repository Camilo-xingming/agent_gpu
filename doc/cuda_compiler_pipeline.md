# CUDA-like C -> PTX -> Binary Pipeline (Issue #157)

## Scope
This is a framework and PoC pipeline for compiling a narrow CUDA-like C subset into RalphGPU PTX, then assembling PTX into machine-code hex.

Current tool:
- `tools/cuda_kernel_compiler.py`

## Pipeline Stages
1. Parse CUDA-like kernel source (`__global__ void ...`) into a lightweight IR.
2. Lower built-in index expressions:
   - `blockIdx.x`, `blockDim.x`, `threadIdx.x`
3. Pattern-match supported kernel bodies.
4. Emit RalphGPU-friendly PTX.
5. Optional PTX -> hex assembly via existing `tools/ptx_assembler.py`.

## Supported Source Subset
- Kernel signature: `__global__ void name(params...)`
- Global index form:
  - `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
- Guard form:
  - `if (idx < n) { ... }`
- Pattern A (vector add):
  - `out[idx] = a[idx] + b[idx];`
- Pattern B (shared staging copy):
  - `__shared__ int tile[N];`
  - `tile[threadIdx.x] = in[idx];`
  - `__syncthreads();`
  - `out[idx] = tile[threadIdx.x];`

## Address and Scalar Binding
Since this PoC does not yet model full CUDA parameter space lowering, pointer/scalar kernel params are bound by CLI flags:
- `--pointer-base NAME=VALUE`
- `--scalar-value NAME=VALUE`

Example:
```bash
python3 tools/cuda_kernel_compiler.py examples/vector_add.cu \
  -o build/vector_add.ptx \
  --hex-output build/vector_add.hex \
  --pointer-base a=0x1000 --pointer-base b=0x2000 --pointer-base c=0x3000 \
  --scalar-value n=256
```

## Tests
- `tests/test_cuda_kernel_compiler.py`
  - Vector-add PTX emission checks
  - Shared-memory PTX emission checks
  - End-to-end PTX->hex generation check

Run:
```bash
python3 tests/test_cuda_kernel_compiler.py
```

## Next Expansion Candidates
- General expression lowering (not only pattern match)
- PTX parameter-space lowering (`ld.param`) and launch metadata
- More built-ins (`blockIdx.y/z`, `gridDim`, `warpSize`)
- Additional control flow (`for` loops, nested conditionals)
