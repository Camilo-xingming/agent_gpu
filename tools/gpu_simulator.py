#!/usr/bin/env python3
"""
RalphGPU Functional Simulator (FRM)
用Python模拟GPU执行，验证设计逻辑
Expanded to cover FP32, Warp, and extended operations
"""

import struct
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import IntEnum

# 操作码 - matches gpu_defines.vh
class Opcode(IntEnum):
    ALU = 0b000000
    MUL = 0b000001
    DIV = 0b000010
    SETP = 0b000011
    BRANCH = 0b000100
    LD_GLOBAL = 0b000101
    ST_GLOBAL = 0b000110
    LD_SHARED = 0b000111
    ST_SHARED = 0b001000
    MOV_SPECIAL = 0b001001
    BAR_SYNC = 0b001010
    EXIT = 0b001011
    RET = 0b001100
    # FP operations
    FP32_ARITH = 0b001101
    FP32_SPECIAL = 0b001110
    FP64_ARITH = 0b001111
    FP16_ARITH = 0b010000
    CVT = 0b010001
    # Extended memory
    LD_PARAM = 0b010010
    LD_CONST = 0b010011
    ATOM = 0b011010
    # Warp operations
    SHFL = 0b011100
    VOTE = 0b011101
    REDUX = 0b011110
    # Memory barrier
    MEMBAR = 0b100100
    # Video/DP
    VIDEO = 0b100101
    # Move immediate
    MOV_IMM = 0b110000
    ALU_IMM = 0b110001
    NOP = 0b111111

# ALU功能码
class AluFunc(IntEnum):
    ADD = 0b000000
    SUB = 0b000001
    AND = 0b000010
    OR = 0b000011
    XOR = 0b000100
    NOT = 0b000101
    SHL = 0b000110
    SHR_U = 0b000111
    SHR_S = 0b001000
    ABS = 0b001001
    NEG = 0b001010
    MIN_S = 0b001011
    MIN_U = 0b001100
    MAX_S = 0b001101
    MAX_U = 0b001110
    POPC = 0b001111
    CLZ = 0b010000

# FP32 功能码
class Fp32Func(IntEnum):
    ADD = 0b000000
    SUB = 0b000001
    MUL = 0b000010
    DIV = 0b000011
    FMA = 0b000100
    NEG = 0b000101
    ABS = 0b000110
    MIN = 0b000111
    MAX = 0b001000

# FP32 Special 功能码
class Fp32SpecialFunc(IntEnum):
    RCP = 0b000000
    SQRT = 0b000001
    RSQRT = 0b000010
    SIN = 0b000011
    COS = 0b000100
    LG2 = 0b000101
    EX2 = 0b000110
    TANH = 0b000111

# FP16 功能码
class Fp16Func(IntEnum):
    ADD = 0b000000
    SUB = 0b000001
    MUL = 0b000010
    FMA = 0b000011
    NEG = 0b000100
    ABS = 0b000101
    MIN = 0b000110
    MAX = 0b000111

# Video功能码 (DP4A/DP2A)
class VideoFunc(IntEnum):
    DP4A_S32_S32 = 0b010000
    DP4A_S32_U32 = 0b010001
    DP4A_U32_S32 = 0b010010
    DP4A_U32_U32 = 0b010011
    DP2A_S32_S32 = 0b010100
    DP2A_S32_U32 = 0b010101

# Atomic功能码
class AtomicFunc(IntEnum):
    ADD = 0
    EXCH = 1
    CAS = 2
    AND = 3
    OR = 4
    XOR = 5
    MIN = 6
    MAX = 7

# MEMBAR功能码
class MembarFunc(IntEnum):
    CTA = 0b00  # membar.cta - CTA level
    GL = 0b01   # membar.gl - Global level
    SYS = 0b10  # membar.sys - System level

# 特殊寄存器
class SpecialReg(IntEnum):
    TID_X = 0
    TID_Y = 1
    TID_Z = 2
    CTAID_X = 3
    CTAID_Y = 4
    CTAID_Z = 5
    NTID_X = 6
    NTID_Y = 7
    NTID_Z = 8
    LANEID = 9
    WARPID = 10
    SMID = 11


@dataclass
class ThreadState:
    """单个线程的状态"""
    tid: int
    registers: List[int] = field(default_factory=lambda: [0] * 32)
    predicates: List[bool] = field(default_factory=lambda: [False] * 8)
    active: bool = True


@dataclass
class WarpState:
    """Warp状态 (32个线程)"""
    warp_id: int
    threads: List[ThreadState] = field(default_factory=list)
    pc: int = 0
    at_barrier: bool = False  # Warp is waiting at a barrier

    def __post_init__(self):
        if not self.threads:
            self.threads = [ThreadState(tid=i) for i in range(32)]


@dataclass
class CTAState:
    """CTA/Block状态 - tracks block-level synchronization"""
    block_id: tuple = (0, 0, 0)
    warps: List[WarpState] = field(default_factory=list)
    # Barrier tracking: barrier_id -> (arrived_count, target_count)
    barriers: Dict[int, tuple] = field(default_factory=dict)
    # Memory visibility tracking (for membar)
    pending_writes: List[tuple] = field(default_factory=list)

    def barrier_arrive(self, barrier_id: int, thread_count: int, total_threads: int) -> bool:
        """Thread arrives at barrier. Returns True if barrier should release."""
        if barrier_id not in self.barriers:
            self.barriers[barrier_id] = (0, total_threads)
        arrived, target = self.barriers[barrier_id]
        arrived += thread_count
        if arrived >= target:
            # Barrier releases - reset for next use
            self.barriers[barrier_id] = (0, target)
            return True
        else:
            self.barriers[barrier_id] = (arrived, target)
            return False

    def flush_writes(self, scope: int):
        """Flush pending writes for membar (no-op in instant FRM)"""
        # In FRM, memory operations are instantaneous
        # This is a placeholder for future cache/memory modeling
        self.pending_writes.clear()


class RalphGPUSimulator:
    """GPU功能仿真器"""

    def __init__(self, num_sm: int = 2, warps_per_sm: int = 4):
        self.num_sm = num_sm
        self.warps_per_sm = warps_per_sm
        self.threads_per_warp = 32

        # 内存
        self.global_memory: Dict[int, int] = {}
        self.shared_memory: List[Dict[int, int]] = [{} for _ in range(num_sm)]
        self.param_memory: Dict[int, int] = {}
        self.const_memory: Dict[int, int] = {}

        # 指令内存
        self.instruction_memory: List[int] = []

        # 执行上下文
        self.block_dim = (32, 1, 1)
        self.grid_dim = (1, 1, 1)
        self.current_block_id = (0, 0, 0)
        self.current_cta: Optional[CTAState] = None

        # 统计
        self.cycle_count = 0
        self.instruction_count = 0

    def load_program(self, hex_file: str):
        """从hex文件加载程序"""
        self.instruction_memory = []
        with open(hex_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instruction_memory.append(int(line, 16))
        print(f"Loaded {len(self.instruction_memory)} instructions")

    def init_memory(self, data: Dict[int, int]):
        """初始化全局内存"""
        self.global_memory.update(data)

    def decode_instruction(self, inst: int) -> dict:
        """解码指令"""
        return {
            'opcode': (inst >> 26) & 0x3F,
            'rd': (inst >> 21) & 0x1F,
            'ra': (inst >> 16) & 0x1F,
            'rb': (inst >> 11) & 0x1F,
            'rc': (inst >> 6) & 0x1F,
            'func': inst & 0x3F,
        }

    def execute_alu(self, func: int, a: int, b: int) -> int:
        """执行ALU操作"""
        # 转换为32位有符号数处理
        a = a & 0xFFFFFFFF
        b = b & 0xFFFFFFFF

        if func == AluFunc.ADD:
            result = (a + b) & 0xFFFFFFFF
        elif func == AluFunc.SUB:
            result = (a - b) & 0xFFFFFFFF
        elif func == AluFunc.AND:
            result = a & b
        elif func == AluFunc.OR:
            result = a | b
        elif func == AluFunc.XOR:
            result = a ^ b
        elif func == AluFunc.NOT:
            result = (~a) & 0xFFFFFFFF
        elif func == AluFunc.SHL:
            result = (a << (b & 0x1F)) & 0xFFFFFFFF
        elif func == AluFunc.SHR_U:
            result = a >> (b & 0x1F)
        elif func == AluFunc.SHR_S:
            # 算术右移
            if a & 0x80000000:
                result = ((a >> (b & 0x1F)) | (0xFFFFFFFF << (32 - (b & 0x1F)))) & 0xFFFFFFFF
            else:
                result = a >> (b & 0x1F)
        else:
            result = 0

        return result

    def uint_to_float(self, v: int) -> float:
        """Convert uint32 bit pattern to float32"""
        return struct.unpack('f', struct.pack('I', v & 0xFFFFFFFF))[0]

    def float_to_uint(self, f: float) -> int:
        """Convert float32 to uint32 bit pattern"""
        return struct.unpack('I', struct.pack('f', f))[0]

    def uint16_to_fp16(self, v: int) -> float:
        """Convert uint16 bit pattern to float16 (stored in lower 16 bits)"""
        return struct.unpack('e', struct.pack('H', v & 0xFFFF))[0]

    def fp16_to_uint16(self, f: float) -> int:
        """Convert float to FP16 bit pattern (uint16)"""
        return struct.unpack('H', struct.pack('e', f))[0]

    def execute_fp16_arith(self, func: int, a: int, b: int, c: int = 0) -> int:
        """Execute FP16 arithmetic operations

        FP16 values are stored in the lower 16 bits of a 32-bit register.
        Result is returned in lower 16 bits, upper bits zeroed.
        """
        fa = self.uint16_to_fp16(a)
        fb = self.uint16_to_fp16(b)
        fc = self.uint16_to_fp16(c)

        try:
            if func == Fp16Func.ADD:
                result = fa + fb
            elif func == Fp16Func.SUB:
                result = fa - fb
            elif func == Fp16Func.MUL:
                result = fa * fb
            elif func == Fp16Func.FMA:
                result = fa * fb + fc
            elif func == Fp16Func.NEG:
                result = -fa
            elif func == Fp16Func.ABS:
                result = abs(fa)
            elif func == Fp16Func.MIN:
                result = min(fa, fb)
            elif func == Fp16Func.MAX:
                result = max(fa, fb)
            else:
                result = 0.0
        except:
            result = float('nan')

        return self.fp16_to_uint16(result)

    def execute_fp32_arith(self, func: int, a: int, b: int, c: int = 0) -> int:
        """Execute FP32 arithmetic operations"""
        fa = self.uint_to_float(a)
        fb = self.uint_to_float(b)
        fc = self.uint_to_float(c)

        try:
            if func == Fp32Func.ADD:
                result = fa + fb
            elif func == Fp32Func.SUB:
                result = fa - fb
            elif func == Fp32Func.MUL:
                result = fa * fb
            elif func == Fp32Func.DIV:
                result = fa / fb if fb != 0 else float('inf') if fa >= 0 else float('-inf')
            elif func == Fp32Func.FMA:
                result = fa * fb + fc
            elif func == Fp32Func.NEG:
                result = -fa
            elif func == Fp32Func.ABS:
                result = abs(fa)
            elif func == Fp32Func.MIN:
                result = min(fa, fb)
            elif func == Fp32Func.MAX:
                result = max(fa, fb)
            else:
                result = 0.0
        except:
            result = float('nan')

        return self.float_to_uint(result)

    def execute_fp32_special(self, func: int, a: int) -> int:
        """Execute FP32 special functions (SFU)"""
        fa = self.uint_to_float(a)

        try:
            if func == Fp32SpecialFunc.RCP:
                result = 1.0 / fa if fa != 0 else float('inf')
            elif func == Fp32SpecialFunc.SQRT:
                result = math.sqrt(fa) if fa >= 0 else float('nan')
            elif func == Fp32SpecialFunc.RSQRT:
                result = 1.0 / math.sqrt(fa) if fa > 0 else float('inf') if fa == 0 else float('nan')
            elif func == Fp32SpecialFunc.SIN:
                result = math.sin(fa)
            elif func == Fp32SpecialFunc.COS:
                result = math.cos(fa)
            elif func == Fp32SpecialFunc.LG2:
                result = math.log2(fa) if fa > 0 else float('-inf') if fa == 0 else float('nan')
            elif func == Fp32SpecialFunc.EX2:
                result = 2.0 ** fa if fa < 128 else float('inf')
            elif func == Fp32SpecialFunc.TANH:
                result = math.tanh(fa)
            else:
                result = 0.0
        except:
            result = float('nan')

        return self.float_to_uint(result)

    def execute_dp4a(self, a: int, b: int, c: int, signed_a: bool = True, signed_b: bool = True) -> int:
        """Execute DP4A dot product of 4 bytes"""
        result = c
        for i in range(4):
            a_byte = (a >> (i * 8)) & 0xFF
            b_byte = (b >> (i * 8)) & 0xFF
            if signed_a and a_byte > 127:
                a_byte -= 256
            if signed_b and b_byte > 127:
                b_byte -= 256
            result += a_byte * b_byte
        return result & 0xFFFFFFFF

    def get_special_reg(self, thread: ThreadState, warp: 'WarpState', sm_id: int, reg_id: int) -> int:
        """获取特殊寄存器值"""
        if reg_id == SpecialReg.TID_X:
            return thread.tid
        elif reg_id == SpecialReg.TID_Y:
            return 0
        elif reg_id == SpecialReg.TID_Z:
            return 0
        elif reg_id == SpecialReg.CTAID_X:
            return self.current_block_id[0]
        elif reg_id == SpecialReg.CTAID_Y:
            return self.current_block_id[1]
        elif reg_id == SpecialReg.CTAID_Z:
            return self.current_block_id[2]
        elif reg_id == SpecialReg.NTID_X:
            return self.block_dim[0]
        elif reg_id == SpecialReg.NTID_Y:
            return self.block_dim[1]
        elif reg_id == SpecialReg.NTID_Z:
            return self.block_dim[2]
        elif reg_id == SpecialReg.LANEID:
            return thread.tid % 32
        elif reg_id == SpecialReg.WARPID:
            return warp.warp_id if warp else 0
        elif reg_id == SpecialReg.SMID:
            return sm_id
        return 0

    def execute_warp(self, warp: WarpState, sm_id: int) -> bool:
        """执行一个Warp的一条指令，返回是否继续"""
        if warp.pc >= len(self.instruction_memory):
            return False

        inst = self.instruction_memory[warp.pc]
        decoded = self.decode_instruction(inst)
        opcode = decoded['opcode']

        self.instruction_count += 1

        # 对每个活跃线程执行
        for thread in warp.threads:
            if not thread.active:
                continue

            rd = decoded['rd']
            ra = decoded['ra']
            rb = decoded['rb']
            rc = decoded['rc']
            func = decoded['func']

            if opcode == Opcode.NOP:
                pass

            elif opcode == Opcode.ALU:
                a = thread.registers[ra]
                b = thread.registers[rb]
                thread.registers[rd] = self.execute_alu(func, a, b)

            elif opcode == Opcode.MUL:
                a = thread.registers[ra]
                b = thread.registers[rb]
                result = a * b
                if func == 0:  # mul.lo
                    thread.registers[rd] = result & 0xFFFFFFFF
                elif func == 1:  # mul.hi
                    thread.registers[rd] = (result >> 32) & 0xFFFFFFFF
                elif func == 2:  # mad.lo
                    c = thread.registers[rc]
                    thread.registers[rd] = ((a * b) + c) & 0xFFFFFFFF

            elif opcode == Opcode.DIV:
                a = thread.registers[ra] & 0xFFFFFFFF
                b = thread.registers[rb] & 0xFFFFFFFF
                if func == 0:  # div.s32 (signed division)
                    # Convert to signed
                    a_s = a if a < 0x80000000 else a - 0x100000000
                    b_s = b if b < 0x80000000 else b - 0x100000000
                    if b_s != 0:
                        # Python integer division rounds toward negative infinity
                        # PTX/C integer division truncates toward zero
                        result = int(a_s / b_s)  # Use true division then truncate
                    else:
                        result = 0xFFFFFFFF if a_s >= 0 else 1  # Undefined, use INT_MAX/-1
                    thread.registers[rd] = result & 0xFFFFFFFF
                elif func == 1:  # div.u32 (unsigned division)
                    if b != 0:
                        result = a // b
                    else:
                        result = 0xFFFFFFFF  # Undefined, use UINT_MAX
                    thread.registers[rd] = result & 0xFFFFFFFF
                elif func == 2:  # rem.s32 (signed remainder)
                    a_s = a if a < 0x80000000 else a - 0x100000000
                    b_s = b if b < 0x80000000 else b - 0x100000000
                    if b_s != 0:
                        # Remainder has same sign as dividend (truncation toward zero)
                        result = a_s - int(a_s / b_s) * b_s
                    else:
                        result = a_s  # Undefined, return dividend
                    thread.registers[rd] = result & 0xFFFFFFFF
                elif func == 3:  # rem.u32 (unsigned remainder)
                    if b != 0:
                        result = a % b
                    else:
                        result = a  # Undefined, return dividend
                    thread.registers[rd] = result & 0xFFFFFFFF

            elif opcode == Opcode.MOV_SPECIAL:
                thread.registers[rd] = self.get_special_reg(thread, warp, sm_id, ra)

            elif opcode == Opcode.MOV_IMM:
                # MOV_IMM: rd = imm16 (lower 16 bits of instruction)
                imm16 = inst & 0xFFFF
                thread.registers[rd] = imm16

            elif opcode == Opcode.ALU_IMM:
                # ALU_IMM: rd = ra op imm10, func in bits [15:10], imm10 in bits [9:0]
                alu_func = (inst >> 10) & 0x3F
                imm10 = inst & 0x3FF
                a = thread.registers[ra]
                thread.registers[rd] = self.execute_alu(alu_func, a, imm10)

            elif opcode == Opcode.FP32_ARITH:
                a = thread.registers[ra]
                b = thread.registers[rb]
                c = thread.registers[rc]
                thread.registers[rd] = self.execute_fp32_arith(func, a, b, c)

            elif opcode == Opcode.FP16_ARITH:
                # FP16 values stored in lower 16 bits of registers
                a = thread.registers[ra] & 0xFFFF
                b = thread.registers[rb] & 0xFFFF
                c = thread.registers[rc] & 0xFFFF
                thread.registers[rd] = self.execute_fp16_arith(func, a, b, c)

            elif opcode == Opcode.FP32_SPECIAL:
                a = thread.registers[ra]
                thread.registers[rd] = self.execute_fp32_special(func, a)

            elif opcode == Opcode.VIDEO:
                # DP4A and DP2A operations
                a = thread.registers[ra]
                b = thread.registers[rb]
                c = thread.registers[rc]
                if func == VideoFunc.DP4A_S32_S32:
                    thread.registers[rd] = self.execute_dp4a(a, b, c, True, True)
                elif func == VideoFunc.DP4A_U32_U32:
                    thread.registers[rd] = self.execute_dp4a(a, b, c, False, False)
                elif func == VideoFunc.DP4A_S32_U32:
                    thread.registers[rd] = self.execute_dp4a(a, b, c, True, False)
                elif func == VideoFunc.DP4A_U32_S32:
                    thread.registers[rd] = self.execute_dp4a(a, b, c, False, True)

            elif opcode == Opcode.EXIT:
                thread.active = False

            elif opcode == Opcode.LD_GLOBAL:
                addr = thread.registers[ra] + ((rb << 6) | func)
                thread.registers[rd] = self.global_memory.get(addr, 0)

            elif opcode == Opcode.ST_GLOBAL:
                addr = thread.registers[ra] + ((rc << 6) | func)
                self.global_memory[addr] = thread.registers[rb]

            elif opcode == Opcode.LD_SHARED:
                addr = thread.registers[ra] + ((rb << 6) | func)
                thread.registers[rd] = self.shared_memory[sm_id].get(addr, 0)

            elif opcode == Opcode.ST_SHARED:
                addr = thread.registers[ra] + ((rc << 6) | func)
                self.shared_memory[sm_id][addr] = thread.registers[rb]

            elif opcode == Opcode.LD_PARAM:
                addr = thread.registers[ra] + ((rb << 6) | func)
                thread.registers[rd] = self.param_memory.get(addr, 0)

            elif opcode == Opcode.LD_CONST:
                addr = thread.registers[ra] + ((rb << 6) | func)
                thread.registers[rd] = self.const_memory.get(addr, 0)

            elif opcode == Opcode.ATOM:
                addr = thread.registers[ra]
                old_val = self.global_memory.get(addr, 0) & 0xFFFFFFFF
                thread.registers[rd] = old_val

                op_val = thread.registers[rb] & 0xFFFFFFFF
                try:
                    atomic_func = AtomicFunc(func)
                except ValueError:
                    atomic_func = None

                new_val = old_val
                if atomic_func == AtomicFunc.ADD:
                    new_val = (old_val + op_val) & 0xFFFFFFFF
                elif atomic_func == AtomicFunc.EXCH:
                    new_val = op_val
                elif atomic_func == AtomicFunc.CAS:
                    # CAS: compare with rc, swap with rb (op_val)
                    compare_val = thread.registers[rc] & 0xFFFFFFFF
                    new_val = op_val if old_val == compare_val else old_val
                elif atomic_func == AtomicFunc.AND:
                    new_val = old_val & op_val
                elif atomic_func == AtomicFunc.OR:
                    new_val = old_val | op_val
                elif atomic_func == AtomicFunc.XOR:
                    new_val = old_val ^ op_val
                elif atomic_func == AtomicFunc.MIN:
                    old_signed = old_val if old_val < 0x80000000 else old_val - 0x100000000
                    op_signed = op_val if op_val < 0x80000000 else op_val - 0x100000000
                    new_val = old_val if old_signed <= op_signed else op_val
                elif atomic_func == AtomicFunc.MAX:
                    old_signed = old_val if old_val < 0x80000000 else old_val - 0x100000000
                    op_signed = op_val if op_val < 0x80000000 else op_val - 0x100000000
                    new_val = old_val if old_signed >= op_signed else op_val

                self.global_memory[addr] = new_val & 0xFFFFFFFF

            elif opcode == Opcode.SETP:
                a = thread.registers[ra]
                b = thread.registers[rb]
                if func == 0:  # eq
                    thread.predicates[rd & 0x7] = (a == b)
                elif func == 1:  # ne
                    thread.predicates[rd & 0x7] = (a != b)
                elif func == 2:  # lt
                    thread.predicates[rd & 0x7] = (a < b)
                elif func == 3:  # le
                    thread.predicates[rd & 0x7] = (a <= b)
                elif func == 4:  # gt
                    thread.predicates[rd & 0x7] = (a > b)
                elif func == 5:  # ge
                    thread.predicates[rd & 0x7] = (a >= b)

            elif opcode == Opcode.BRANCH:
                if thread.tid != 0:
                    continue
                imm16 = inst & 0xFFFF
                if imm16 & 0x8000:
                    imm16 -= 0x10000
                target_pc = warp.pc + imm16

                if rd & 0x10:
                    predicate_idx = rc & 0x7
                    take_branch = thread.predicates[predicate_idx]
                else:
                    branch_type = (rd >> 3) & 0x3
                    take_branch = branch_type in (0b00, 0b11)
                    if branch_type == 0b01:
                        take_branch = (thread.registers[ra] == 0)
                    elif branch_type == 0b10:
                        take_branch = (thread.registers[ra] != 0)
                if take_branch:
                    warp.pc = target_pc - 1  # -1 because warp.pc is incremented after the loop

            elif opcode == Opcode.BAR_SYNC:
                # Block-level barrier synchronization
                # bar.sync barrier_id, thread_count
                # barrier_id = ra, thread_count = rb (0 = all threads in CTA)
                if thread.tid == 0:
                    barrier_id = ra
                    thread_count_reg = thread.registers[rb] if rb else 0
                    total_threads = self.block_dim[0] * self.block_dim[1] * self.block_dim[2]

                    # Count active threads in this warp arriving at barrier
                    active_count = sum(1 for t in warp.threads if t.active)

                    # Use CTA state for proper tracking if available
                    if self.current_cta:
                        target = thread_count_reg if thread_count_reg > 0 else total_threads
                        released = self.current_cta.barrier_arrive(barrier_id, active_count, target)
                        if not released:
                            # In multi-warp mode, warp would stall here
                            warp.at_barrier = True
                    # Single-warp mode: barrier always releases immediately
                    # (all 32 threads arrive together)

            elif opcode == Opcode.MEMBAR:
                # Memory barrier - ensures memory ordering
                # func encodes scope: CTA=0, GL=1, SYS=2
                if thread.tid == 0:
                    scope = func & 0x3
                    # Flush any pending writes to ensure visibility
                    if self.current_cta:
                        self.current_cta.flush_writes(scope)
                    # In FRM, memory operations are instantaneous
                    # This enforces a sequencing point for correctness

        warp.pc += 1
        self.cycle_count += 1

        # Check if all threads exited
        all_exited = all(not t.active for t in warp.threads)

        # 检查是否到达程序末尾或NOP或所有线程已退出
        return (warp.pc < len(self.instruction_memory) and
                opcode != Opcode.NOP and
                opcode != Opcode.EXIT and
                not all_exited)

    def run_kernel(self, entry_pc: int = 0, max_cycles: int = 10000):
        """运行Kernel"""
        print(f"\n{'='*60}")
        print("RalphGPU Simulation Start")
        print(f"Grid: {self.grid_dim}, Block: {self.block_dim}")
        print(f"{'='*60}\n")

        total_blocks = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

        for block_idx in range(total_blocks):
            self.current_block_id = (
                block_idx % self.grid_dim[0],
                (block_idx // self.grid_dim[0]) % self.grid_dim[1],
                block_idx // (self.grid_dim[0] * self.grid_dim[1])
            )

            # Create CTA state for this block
            self.current_cta = CTAState(block_id=self.current_block_id)

            # 创建Warp
            warp = WarpState(warp_id=0)
            warp.pc = entry_pc
            self.current_cta.warps.append(warp)

            # 执行直到完成
            while self.execute_warp(warp, sm_id=0):
                if self.cycle_count > max_cycles:
                    print("ERROR: Max cycles exceeded!")
                    return

        print(f"\n{'='*60}")
        print("Simulation Complete")
        print(f"Cycles: {self.cycle_count}")
        print(f"Instructions: {self.instruction_count}")
        print(f"{'='*60}\n")

    def dump_registers(self, warp: WarpState, num_threads: int = 4):
        """打印寄存器状态"""
        print("\nRegister State:")
        print("-" * 60)
        for t in range(min(num_threads, 32)):
            thread = warp.threads[t]
            regs = [f"r{i}={thread.registers[i]}" for i in range(8) if thread.registers[i] != 0]
            print(f"Thread {t}: {', '.join(regs) if regs else '(all zero)'}")

    def dump_memory(self, start: int, count: int):
        """打印内存内容"""
        print(f"\nGlobal Memory [{start}:{start+count}]:")
        print("-" * 60)
        for i in range(count):
            addr = start + i * 4
            val = self.global_memory.get(addr, 0)
            if val != 0:
                print(f"  [{addr:08x}] = {val}")


def test_vector_add():
    """测试向量加法"""
    print("\n" + "=" * 60)
    print("Test: Vector Addition")
    print("C[i] = A[i] + B[i]")
    print("=" * 60)

    sim = RalphGPUSimulator(num_sm=1)

    # 初始化数据
    # A[0-31] = 0, 1, 2, ..., 31  @ address 0x0000
    # B[0-31] = 100, 101, ..., 131  @ address 0x1000
    for i in range(32):
        sim.global_memory[i * 4] = i              # A[i]
        sim.global_memory[0x1000 + i * 4] = 100 + i  # B[i]

    # 简化的向量加法程序 (使用立即数偏移)
    # 由于我们需要在指令中编码偏移，我们使用一个更简单的方法
    instructions = [
        # mov.u32 r0, %tid.x  -- r0 = thread_id
        (Opcode.MOV_SPECIAL << 26) | (0 << 21) | (SpecialReg.TID_X << 16),

        # add r15, r0, r0  -- r15 = tid * 2 (临时)
        (Opcode.ALU << 26) | (15 << 21) | (0 << 16) | (0 << 11) | AluFunc.ADD,

        # add r1, r15, r15  -- r1 = tid * 4 (字节偏移)
        (Opcode.ALU << 26) | (1 << 21) | (15 << 16) | (15 << 11) | AluFunc.ADD,

        # ld.global r2, [r1+0]  -- r2 = A[tid]
        (Opcode.LD_GLOBAL << 26) | (2 << 21) | (1 << 16) | (0 << 11) | 0,

        # 计算B偏移: 需要r1 + 0x1000
        # 先构造0x1000: 使用多次左移
        # r10 = 1, r10 = r10 << 12 = 0x1000
        # 简化：使用小偏移测试，把B放在偏移64处 (16个元素后)

        # 重新设计：把B放在地址64 (0x40)，C放在地址128 (0x80)
        # 这样偏移在6位以内可以编码

        # ld.global r3, [r1+64]  -- r3 = B[tid]
        # offset 64 = 1 << 6, 编码在 rb(5bit)<<6 | func(6bit) = (1<<6)|0 = 64
        (Opcode.LD_GLOBAL << 26) | (3 << 21) | (1 << 16) | (1 << 11) | 0,

        # add.s32 r4, r2, r3  -- r4 = A[tid] + B[tid]
        (Opcode.ALU << 26) | (4 << 21) | (2 << 16) | (3 << 11) | AluFunc.ADD,

        # st.global [r1+128], r4  -- C[tid] = r4
        # offset 128 = 2 << 6, 编码在 rc(5bit)<<6 | func(6bit) = (2<<6)|0 = 128
        (Opcode.ST_GLOBAL << 26) | (4 << 11) | (1 << 16) | (2 << 6) | 0,

        # nop (结束)
        (Opcode.NOP << 26),
    ]

    # 重新初始化内存使用更小的偏移
    sim.global_memory.clear()
    for i in range(32):
        sim.global_memory[i * 4] = i              # A[i] at 0
        sim.global_memory[64 + i * 4] = 100 + i   # B[i] at 64

    sim.instruction_memory = instructions

    # 创建warp
    warp = WarpState(warp_id=0)

    # 运行
    sim.block_dim = (32, 1, 1)
    sim.grid_dim = (1, 1, 1)

    # 执行
    print("\nExecuting kernel...")
    cycle = 0
    while sim.execute_warp(warp, sm_id=0) and cycle < 100:
        cycle += 1

    # 打印一些调试信息
    print(f"\nThread 0 registers after execution:")
    for i in range(5):
        print(f"  r{i} = {warp.threads[0].registers[i]}")

    # 验证结果
    print("\nResults (first 8 elements):")
    print("-" * 50)
    errors = 0
    for i in range(8):
        a_val = sim.global_memory.get(i * 4, 0)
        b_val = sim.global_memory.get(64 + i * 4, 0)
        c_val = sim.global_memory.get(128 + i * 4, 0)
        expected = a_val + b_val
        status = "OK" if c_val == expected else "FAIL"
        if c_val != expected:
            errors += 1
        print(f"C[{i}] = {c_val:4d} (A={a_val}, B={b_val}, expected={expected}) [{status}]")

    print("-" * 50)
    print(f"Cycles: {sim.cycle_count}")
    if errors == 0:
        print("TEST PASSED!")
    else:
        print(f"TEST FAILED: {errors} errors")

    return errors == 0


def test_simple_alu():
    """测试简单ALU操作"""
    print("\n" + "=" * 60)
    print("Test: Simple ALU Operations")
    print("=" * 60)

    sim = RalphGPUSimulator(num_sm=1)

    # 测试程序:
    # mov r0, %tid.x    // r0 = 0,1,2,...,31
    # add r1, r0, r0    // r1 = r0 + r0 = 0,2,4,...,62
    # mul.lo r2, r0, r1 // r2 = r0 * r1
    # nop

    instructions = [
        (Opcode.MOV_SPECIAL << 26) | (0 << 21) | (SpecialReg.TID_X << 16),
        (Opcode.ALU << 26) | (1 << 21) | (0 << 16) | (0 << 11) | AluFunc.ADD,
        (Opcode.MUL << 26) | (2 << 21) | (0 << 16) | (1 << 11) | 0,  # mul.lo
        (Opcode.NOP << 26),
    ]

    sim.instruction_memory = instructions

    warp = WarpState(warp_id=0)
    sim.block_dim = (32, 1, 1)

    # 执行
    while sim.execute_warp(warp, sm_id=0):
        pass

    # 验证
    print("\nResults (first 8 threads):")
    print("-" * 50)
    print(f"{'Thread':>6} {'r0':>6} {'r1':>6} {'r2':>8} {'Expected':>10}")
    print("-" * 50)

    errors = 0
    for i in range(8):
        thread = warp.threads[i]
        r0 = thread.registers[0]
        r1 = thread.registers[1]
        r2 = thread.registers[2]
        expected_r1 = i * 2
        expected_r2 = i * (i * 2)

        status = ""
        if r1 != expected_r1 or r2 != expected_r2:
            status = " FAIL"
            errors += 1

        print(f"{i:>6} {r0:>6} {r1:>6} {r2:>8} {expected_r2:>10}{status}")

    print("-" * 50)
    print(f"Cycles: {sim.cycle_count}")
    if errors == 0:
        print("TEST PASSED!")
    else:
        print(f"TEST FAILED: {errors} errors")

    return errors == 0


def test_fp32_arith():
    """测试FP32算术操作"""
    print("\n" + "=" * 60)
    print("Test: FP32 Arithmetic Operations")
    print("=" * 60)

    sim = RalphGPUSimulator(num_sm=1)

    # Helper to encode float as int
    def f2i(f):
        return struct.unpack('I', struct.pack('f', f))[0]

    # Helper to decode int as float
    def i2f(i):
        return struct.unpack('f', struct.pack('I', i))[0]

    # Test program:
    # r1 = 3.0f, r2 = 2.0f
    # r3 = r1 + r2 = 5.0f
    # r4 = r1 * r2 = 6.0f
    # r5 = sin(r1) ≈ 0.1411

    three_f = f2i(3.0)
    two_f = f2i(2.0)

    # Build instruction sequence using MOV_IMM for 32-bit values
    # We need to load 32-bit floats, so use the multi-instruction approach
    hi_3 = (three_f >> 16) & 0xFFFF
    lo_3 = three_f & 0xFFFF
    hi_2 = (two_f >> 16) & 0xFFFF
    lo_2 = two_f & 0xFFFF

    instructions = [
        # Load 3.0 into r1 (hi16 + shl + or lo16)
        (Opcode.MOV_IMM << 26) | (1 << 21) | hi_3,  # r1 = hi_3
        (Opcode.ALU_IMM << 26) | (1 << 21) | (1 << 16) | (AluFunc.SHL << 10) | 16,  # r1 = r1 << 16
        (Opcode.MOV_IMM << 26) | (31 << 21) | lo_3,  # r31 = lo_3
        (Opcode.ALU << 26) | (1 << 21) | (1 << 16) | (31 << 11) | AluFunc.OR,  # r1 = r1 | r31

        # Load 2.0 into r2
        (Opcode.MOV_IMM << 26) | (2 << 21) | hi_2,
        (Opcode.ALU_IMM << 26) | (2 << 21) | (2 << 16) | (AluFunc.SHL << 10) | 16,
        (Opcode.MOV_IMM << 26) | (31 << 21) | lo_2,
        (Opcode.ALU << 26) | (2 << 21) | (2 << 16) | (31 << 11) | AluFunc.OR,

        # FP32 add: r3 = r1 + r2
        (Opcode.FP32_ARITH << 26) | (3 << 21) | (1 << 16) | (2 << 11) | Fp32Func.ADD,

        # FP32 mul: r4 = r1 * r2
        (Opcode.FP32_ARITH << 26) | (4 << 21) | (1 << 16) | (2 << 11) | Fp32Func.MUL,

        # FP32 sin: r5 = sin(r1)
        (Opcode.FP32_SPECIAL << 26) | (5 << 21) | (1 << 16) | Fp32SpecialFunc.SIN,

        # Exit
        (Opcode.EXIT << 26),
    ]

    sim.instruction_memory = instructions
    warp = WarpState(warp_id=0)
    sim.block_dim = (32, 1, 1)

    # Execute
    while sim.execute_warp(warp, sm_id=0):
        pass

    # Verify thread 0 results
    thread = warp.threads[0]
    r1 = i2f(thread.registers[1])
    r2 = i2f(thread.registers[2])
    r3 = i2f(thread.registers[3])
    r4 = i2f(thread.registers[4])
    r5 = i2f(thread.registers[5])

    print(f"\nResults (thread 0):")
    print("-" * 50)
    print(f"r1 = {r1} (expected 3.0)")
    print(f"r2 = {r2} (expected 2.0)")
    print(f"r3 = r1 + r2 = {r3} (expected 5.0)")
    print(f"r4 = r1 * r2 = {r4} (expected 6.0)")
    print(f"r5 = sin(r1) = {r5} (expected ~0.1411)")
    print("-" * 50)

    errors = 0
    if abs(r3 - 5.0) > 0.001:
        print(f"ERROR: r3 = {r3}, expected 5.0")
        errors += 1
    if abs(r4 - 6.0) > 0.001:
        print(f"ERROR: r4 = {r4}, expected 6.0")
        errors += 1
    if abs(r5 - math.sin(3.0)) > 0.01:
        print(f"ERROR: r5 = {r5}, expected {math.sin(3.0)}")
        errors += 1

    if errors == 0:
        print("TEST PASSED!")
    else:
        print(f"TEST FAILED: {errors} errors")

    return errors == 0


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("RalphGPU Functional Simulator")
    print("=" * 60)

    all_passed = True

    # 运行测试
    all_passed &= test_simple_alu()
    all_passed &= test_vector_add()
    all_passed &= test_fp32_arith()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
