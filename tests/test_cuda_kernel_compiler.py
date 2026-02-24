#!/usr/bin/env python3
"""Unit tests for tools/cuda_kernel_compiler.py."""

import os
import re
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from cuda_kernel_compiler import CompileOptions, KernelCompiler, compile_file


VECTOR_ADD_SOURCE = """
__global__ void vector_add(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""


SHARED_COPY_SOURCE = """
__global__ void shared_copy(const int* src, int* dst, int n) {
    __shared__ int tile[64];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tile[threadIdx.x] = src[idx];
        __syncthreads();
        dst[idx] = tile[threadIdx.x];
    }
}
"""


class CudaKernelCompilerTest(unittest.TestCase):
    def test_vector_add_compiles_expected_instructions(self) -> None:
        options = CompileOptions(
            pointer_bases={"a": 0x1000, "b": 0x2000, "c": 0x3000},
            scalar_values={"n": 64},
        )
        result = KernelCompiler(options).compile_source(VECTOR_ADD_SOURCE)
        ptx = result.ptx

        self.assertIn(".entry vector_add:", ptx)
        self.assertIn("mov.u32 r0, %ctaid.x", ptx)
        self.assertIn("mov.u32 r1, %ntid.x", ptx)
        self.assertIn("mov.u32 r2, %tid.x", ptx)
        self.assertIn("setp.lt.s32 p0, r4, r5", ptx)
        self.assertIn("bra.z p0, DONE", ptx)
        self.assertIn("ld.global.s32 r12, [r11]", ptx)
        self.assertIn("ld.global.s32 r15, [r14]", ptx)
        self.assertIn("st.global.s32 [r18], r16", ptx)

    def test_shared_copy_emits_shared_memory_sequence(self) -> None:
        options = CompileOptions(
            pointer_bases={"src": 0x4000, "dst": 0x5000},
            scalar_values={"n": 32},
        )
        result = KernelCompiler(options).compile_source(SHARED_COPY_SOURCE)
        ptx = result.ptx

        self.assertIn(".entry shared_copy:", ptx)
        self.assertIn("st.shared.s32 [r22], r12", ptx)
        self.assertIn("bar.sync 0", ptx)
        self.assertIn("ld.shared.s32 r23, [r22]", ptx)
        self.assertIn("st.global.s32 [r25], r23", ptx)

    def test_end_to_end_generates_ptx_and_hex(self) -> None:
        options = CompileOptions(
            pointer_bases={"a": 0x1000, "b": 0x2000, "c": 0x3000},
            scalar_values={"n": 16},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            src_path = temp / "vector_add.cu"
            ptx_path = temp / "vector_add.ptx"
            hex_path = temp / "vector_add.hex"

            src_path.write_text(VECTOR_ADD_SOURCE, encoding="utf-8")
            compile_file(src_path, ptx_path, hex_path, options)

            self.assertTrue(ptx_path.exists())
            self.assertTrue(hex_path.exists())

            ptx_text = ptx_path.read_text(encoding="utf-8")
            self.assertIn(".entry vector_add:", ptx_text)

            hex_lines = [line.strip() for line in hex_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(len(hex_lines), 0)
            for line in hex_lines:
                self.assertRegex(line, r"^[0-9a-fA-F]{8}$")


if __name__ == "__main__":
    unittest.main()
