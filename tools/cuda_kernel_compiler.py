#!/usr/bin/env python3
"""RalphGPU CUDA-like kernel compiler pipeline (C subset -> PTX -> HEX)."""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ptx_assembler import assemble_file


@dataclass
class KernelParam:
    ctype: str
    name: str
    is_pointer: bool


@dataclass
class SharedBuffer:
    ctype: str
    name: str
    length: int


@dataclass
class KernelIR:
    name: str
    params: List[KernelParam]
    idx_var: str
    bound_expr: str
    shared_buffers: List[SharedBuffer]
    operation: str
    op_args: Dict[str, str]
    uses_syncthreads: bool


@dataclass
class CompileOptions:
    pointer_bases: Dict[str, int] = field(default_factory=dict)
    scalar_values: Dict[str, int] = field(default_factory=dict)
    default_pointer_base: int = 0x1000
    pointer_stride: int = 0x1000


@dataclass
class CompilationResult:
    ir: KernelIR
    ptx: str
    pointer_bases: Dict[str, int]
    scalar_values: Dict[str, int]


class CSubsetParser:
    """Parser for a narrow CUDA-like C kernel subset."""

    _KERNEL_RE = re.compile(
        r"__global__\s+void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{",
        flags=re.S,
    )

    _SHARED_RE = re.compile(
        r"__shared__\s+([A-Za-z_]\w*(?:\s*\*)?)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*;",
        flags=re.S,
    )

    def parse(self, source: str) -> KernelIR:
        clean = self._strip_comments(source)
        kernel_name, params_raw, body = self._extract_kernel(clean)

        params = self._parse_params(params_raw)
        shared_buffers = self._parse_shared_buffers(body)
        idx_var = self._parse_global_idx_var(body)
        bound_expr, operation_scope = self._extract_guard_scope(body, idx_var, params)
        operation, op_args = self._parse_operation(operation_scope, idx_var)

        return KernelIR(
            name=kernel_name,
            params=params,
            idx_var=idx_var,
            bound_expr=bound_expr,
            shared_buffers=shared_buffers,
            operation=operation,
            op_args=op_args,
            uses_syncthreads="__syncthreads" in body,
        )

    @staticmethod
    def _strip_comments(source: str) -> str:
        source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
        lines = []
        for line in source.splitlines():
            lines.append(line.split("//", 1)[0])
        return "\n".join(lines)

    def _extract_kernel(self, source: str) -> Tuple[str, str, str]:
        match = self._KERNEL_RE.search(source)
        if not match:
            raise ValueError("Expected one '__global__ void <kernel>(...) { ... }' definition")

        kernel_name = match.group(1)
        params_raw = match.group(2)
        brace_start = match.end() - 1
        body, _ = self._extract_brace_block(source, brace_start)
        return kernel_name, params_raw, body

    @staticmethod
    def _extract_brace_block(text: str, open_brace_idx: int) -> Tuple[str, int]:
        if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
            raise ValueError("Invalid brace block start")

        depth = 0
        start = open_brace_idx + 1
        for idx in range(open_brace_idx, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx], idx
        raise ValueError("Unbalanced braces in kernel body")

    @staticmethod
    def _parse_params(params_raw: str) -> List[KernelParam]:
        params: List[KernelParam] = []
        if not params_raw.strip():
            return params

        for raw in params_raw.split(","):
            token = " ".join(raw.strip().split())
            if not token:
                continue
            match = re.match(r"(.+?)\s*([A-Za-z_]\w*)$", token)
            if not match:
                raise ValueError(f"Unable to parse kernel parameter: '{raw.strip()}'")

            ctype = match.group(1).strip()
            name = match.group(2).strip()
            is_pointer = "*" in ctype
            params.append(KernelParam(ctype=ctype, name=name, is_pointer=is_pointer))

        return params

    def _parse_shared_buffers(self, body: str) -> List[SharedBuffer]:
        buffers: List[SharedBuffer] = []
        for match in self._SHARED_RE.finditer(body):
            ctype = " ".join(match.group(1).split())
            name = match.group(2)
            length = int(match.group(3))
            buffers.append(SharedBuffer(ctype=ctype, name=name, length=length))
        return buffers

    @staticmethod
    def _parse_global_idx_var(body: str) -> str:
        pattern = re.compile(
            r"(?:int|unsigned|size_t)\s+([A-Za-z_]\w*)\s*=\s*"
            r"(?:blockIdx\.x\s*\*\s*blockDim\.x|blockDim\.x\s*\*\s*blockIdx\.x)\s*"
            r"\+\s*threadIdx\.x\s*;",
            flags=re.S,
        )
        match = pattern.search(body)
        if not match:
            raise ValueError(
                "Expected global index pattern like 'int idx = blockIdx.x * blockDim.x + threadIdx.x;'"
            )
        return match.group(1)

    def _extract_guard_scope(
        self,
        body: str,
        idx_var: str,
        params: List[KernelParam],
    ) -> Tuple[str, str]:
        guard_re = re.compile(
            rf"if\s*\(\s*{re.escape(idx_var)}\s*<\s*([A-Za-z_]\w*|\d+)\s*\)\s*\{{",
            flags=re.S,
        )
        match = guard_re.search(body)
        if match:
            bound_expr = match.group(1)
            block, _ = self._extract_brace_block(body, match.end() - 1)
            return bound_expr, block

        scalar_params = [p.name for p in params if not p.is_pointer]
        if scalar_params:
            return scalar_params[0], body
        return "32", body

    @staticmethod
    def _parse_operation(scope: str, idx_var: str) -> Tuple[str, Dict[str, str]]:
        idx = re.escape(idx_var)

        shared_stage = re.search(
            rf"([A-Za-z_]\w*)\s*\[\s*threadIdx\.x\s*\]\s*=\s*([A-Za-z_]\w*)\s*\[\s*{idx}\s*\]\s*;",
            scope,
            flags=re.S,
        )
        shared_writeback = re.search(
            rf"([A-Za-z_]\w*)\s*\[\s*{idx}\s*\]\s*=\s*([A-Za-z_]\w*)\s*\[\s*threadIdx\.x\s*\]\s*;",
            scope,
            flags=re.S,
        )
        if shared_stage and shared_writeback and shared_stage.group(1) == shared_writeback.group(2):
            return "shared_copy", {
                "shared": shared_stage.group(1),
                "src": shared_stage.group(2),
                "dst": shared_writeback.group(1),
            }

        vector_add = re.search(
            rf"([A-Za-z_]\w*)\s*\[\s*{idx}\s*\]\s*=\s*"
            rf"([A-Za-z_]\w*)\s*\[\s*{idx}\s*\]\s*\+\s*"
            rf"([A-Za-z_]\w*)\s*\[\s*{idx}\s*\]\s*;",
            scope,
            flags=re.S,
        )
        if vector_add:
            return "vector_add", {
                "dst": vector_add.group(1),
                "src_a": vector_add.group(2),
                "src_b": vector_add.group(3),
            }

        raise ValueError(
            "Unsupported kernel body. Supported patterns: vector add or shared-memory stage+copy."
        )


class KernelCompiler:
    """Compiler pipeline orchestrator for the supported C subset."""

    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options or CompileOptions()
        self.parser = CSubsetParser()

    def compile_source(self, source: str) -> CompilationResult:
        ir = self.parser.parse(source)
        pointer_bases = self._resolve_pointer_bases(ir)
        scalar_values = self._resolve_scalar_values(ir)
        ptx = self._emit_ptx(ir, pointer_bases, scalar_values)
        return CompilationResult(
            ir=ir,
            ptx=ptx,
            pointer_bases=pointer_bases,
            scalar_values=scalar_values,
        )

    def _resolve_pointer_bases(self, ir: KernelIR) -> Dict[str, int]:
        result: Dict[str, int] = {}
        used = set()

        for name, addr in self.options.pointer_bases.items():
            value = int(addr)
            result[name] = value
            used.add(value)

        next_base = self.options.default_pointer_base
        for param in ir.params:
            if not param.is_pointer:
                continue
            if param.name in result:
                continue

            while next_base in used:
                next_base += self.options.pointer_stride
            result[param.name] = next_base
            used.add(next_base)
            next_base += self.options.pointer_stride

        return result

    def _resolve_scalar_values(self, ir: KernelIR) -> Dict[str, int]:
        result = {name: int(value) for name, value in self.options.scalar_values.items()}

        for param in ir.params:
            if param.is_pointer or param.name in result:
                continue
            if param.name.lower() in {"n", "len", "length", "size"}:
                result[param.name] = 16
            else:
                result[param.name] = 1

        return result

    def _resolve_bound(self, ir: KernelIR, scalar_values: Dict[str, int]) -> int:
        token = ir.bound_expr
        if re.fullmatch(r"\d+", token):
            return int(token)
        if token in scalar_values:
            return int(scalar_values[token])
        raise ValueError(f"Unable to resolve bound expression '{token}'. Provide --scalar-value {token}=...'")

    @staticmethod
    def _required_pointer(pointer_bases: Dict[str, int], name: str) -> int:
        if name not in pointer_bases:
            raise ValueError(f"Missing pointer base for '{name}'. Provide --pointer-base {name}=0x....")
        return pointer_bases[name]

    def _emit_ptx(
        self,
        ir: KernelIR,
        pointer_bases: Dict[str, int],
        scalar_values: Dict[str, int],
    ) -> str:
        bound = self._resolve_bound(ir, scalar_values)

        lines: List[str] = [
            "// Auto-generated by tools/cuda_kernel_compiler.py",
            f"// kernel={ir.name}",
            "// pointer-bases=" + ", ".join(f"{k}=0x{v:08x}" for k, v in sorted(pointer_bases.items())),
            "// scalar-values=" + ", ".join(f"{k}={v}" for k, v in sorted(scalar_values.items())),
            "",
            f".entry {ir.name}:",
            "    mov.u32 r0, %ctaid.x",
            "    mov.u32 r1, %ntid.x",
            "    mov.u32 r2, %tid.x",
            "    mul.lo.s32 r3, r0, r1",
            "    add.s32 r4, r3, r2",
            f"    mov.u32 r5, {bound}",
            "    setp.lt.s32 p0, r4, r5",
            "    bra.z p0, DONE",
            "    shl.b32 r6, r4, 2",
            "",
        ]

        if ir.operation == "vector_add":
            src_a = self._required_pointer(pointer_bases, ir.op_args["src_a"])
            src_b = self._required_pointer(pointer_bases, ir.op_args["src_b"])
            dst = self._required_pointer(pointer_bases, ir.op_args["dst"])

            lines.extend(
                [
                    f"    mov.u32 r10, 0x{src_a:08x}",
                    "    add.s32 r11, r10, r6",
                    "    ld.global.s32 r12, [r11]",
                    f"    mov.u32 r13, 0x{src_b:08x}",
                    "    add.s32 r14, r13, r6",
                    "    ld.global.s32 r15, [r14]",
                    "    add.s32 r16, r12, r15",
                    f"    mov.u32 r17, 0x{dst:08x}",
                    "    add.s32 r18, r17, r6",
                    "    st.global.s32 [r18], r16",
                ]
            )

        elif ir.operation == "shared_copy":
            if not ir.uses_syncthreads:
                raise ValueError("shared_copy pattern requires __syncthreads() in source")

            src = self._required_pointer(pointer_bases, ir.op_args["src"])
            dst = self._required_pointer(pointer_bases, ir.op_args["dst"])

            lines.extend(
                [
                    f"    mov.u32 r10, 0x{src:08x}",
                    "    add.s32 r11, r10, r6",
                    "    ld.global.s32 r12, [r11]",
                    "    shl.b32 r20, r2, 2",
                    "    mov.u32 r21, 0",
                    "    add.s32 r22, r21, r20",
                    "    st.shared.s32 [r22], r12",
                    "    bar.sync 0",
                    "    ld.shared.s32 r23, [r22]",
                    f"    mov.u32 r24, 0x{dst:08x}",
                    "    add.s32 r25, r24, r6",
                    "    st.global.s32 [r25], r23",
                ]
            )
        else:
            raise ValueError(f"Unsupported operation kind: {ir.operation}")

        lines.extend(
            [
                "",
                "DONE:",
                "    exit",
                ".end",
            ]
        )

        return "\n".join(lines)


def _parse_mapping(values: List[str], flag: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected {flag} in NAME=VALUE format: '{item}'")
        name, value_text = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Expected non-empty name in {flag}: '{item}'")
        try:
            value = int(value_text.strip(), 0)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for {flag}: '{item}'") from exc
        mapping[name] = value
    return mapping


def format_ir(ir: KernelIR) -> str:
    params = ", ".join(f"{p.ctype} {p.name}" for p in ir.params)
    shared = ", ".join(f"{b.ctype} {b.name}[{b.length}]" for b in ir.shared_buffers) or "<none>"
    return (
        f"kernel={ir.name}\n"
        f"params={params}\n"
        f"idx_var={ir.idx_var}\n"
        f"bound={ir.bound_expr}\n"
        f"operation={ir.operation} {ir.op_args}\n"
        f"shared={shared}\n"
        f"uses_syncthreads={ir.uses_syncthreads}"
    )


def compile_file(
    input_path: Path,
    ptx_output: Path,
    hex_output: Optional[Path],
    options: CompileOptions,
    dump_ir: bool = False,
) -> CompilationResult:
    source = input_path.read_text(encoding="utf-8")
    compiler = KernelCompiler(options)
    result = compiler.compile_source(source)

    ptx_output.parent.mkdir(parents=True, exist_ok=True)
    ptx_output.write_text(result.ptx + "\n", encoding="utf-8")

    if hex_output is not None:
        hex_output.parent.mkdir(parents=True, exist_ok=True)
        assemble_file(str(ptx_output), str(hex_output))

    if dump_ir:
        print(format_ir(result.ir))

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a minimal CUDA-like C kernel into RalphGPU PTX and optional HEX"
    )
    parser.add_argument("input", type=Path, help="Input .cu/.c kernel source file")
    parser.add_argument(
        "-o",
        "--ptx-output",
        type=Path,
        default=None,
        help="Output PTX path (default: <input>.ptx)",
    )
    parser.add_argument(
        "--hex-output",
        type=Path,
        default=None,
        help="Optional output HEX path (assembles PTX with tools/ptx_assembler.py)",
    )
    parser.add_argument(
        "--pointer-base",
        action="append",
        default=[],
        help="Pointer base mapping NAME=VALUE (e.g., a=0x1000). Repeat as needed.",
    )
    parser.add_argument(
        "--scalar-value",
        action="append",
        default=[],
        help="Scalar value mapping NAME=VALUE (e.g., n=256). Repeat as needed.",
    )
    parser.add_argument(
        "--dump-ir",
        action="store_true",
        help="Print parsed kernel IR summary",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    ptx_output = args.ptx_output or args.input.with_suffix(".ptx")

    options = CompileOptions(
        pointer_bases=_parse_mapping(args.pointer_base, "--pointer-base"),
        scalar_values=_parse_mapping(args.scalar_value, "--scalar-value"),
    )

    compile_file(
        input_path=args.input,
        ptx_output=ptx_output,
        hex_output=args.hex_output,
        options=options,
        dump_ir=args.dump_ir,
    )

    print(f"Wrote PTX: {ptx_output}")
    if args.hex_output is not None:
        print(f"Wrote HEX: {args.hex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
