#!/usr/bin/env python3
"""
RalphGPU Functional Simulator
用Python模拟GPU执行，验证设计逻辑
"""

import struct
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import IntEnum

# 操作码
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

    def __post_init__(self):
        if not self.threads:
            self.threads = [ThreadState(tid=i) for i in range(32)]


class RalphGPUSimulator:
    """GPU功能仿真器"""

    def __init__(self, num_sm: int = 2, warps_per_sm: int = 4):
        self.num_sm = num_sm
        self.warps_per_sm = warps_per_sm
        self.threads_per_warp = 32

        # 内存
        self.global_memory: Dict[int, int] = {}
        self.shared_memory: List[Dict[int, int]] = [{} for _ in range(num_sm)]

        # 指令内存
        self.instruction_memory: List[int] = []

        # 执行上下文
        self.block_dim = (32, 1, 1)
        self.grid_dim = (1, 1, 1)
        self.current_block_id = (0, 0, 0)

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

    def get_special_reg(self, thread: ThreadState, reg_id: int) -> int:
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

            elif opcode == Opcode.MOV_SPECIAL:
                thread.registers[rd] = self.get_special_reg(thread, ra)

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

        warp.pc += 1
        self.cycle_count += 1

        # 检查是否到达程序末尾或NOP
        return warp.pc < len(self.instruction_memory) and opcode != Opcode.NOP

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

            # 创建Warp
            warp = WarpState(warp_id=0)
            warp.pc = entry_pc

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


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("RalphGPU Functional Simulator")
    print("=" * 60)

    all_passed = True

    # 运行测试
    all_passed &= test_simple_alu()
    all_passed &= test_vector_add()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
