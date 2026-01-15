#!/usr/bin/env python3
"""
RalphGPU PTX Assembler
将PTX风格的汇编转换为机器码

用法:
    python ptx_assembler.py input.ptx -o output.hex
"""

import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

# 操作码定义
OPCODES = {
    'alu':        0b000000,
    'mul':        0b000001,
    'div':        0b000010,
    'setp':       0b000011,
    'bra':        0b000100,
    'ld.global':  0b000101,
    'st.global':  0b000110,
    'ld.shared':  0b000111,
    'st.shared':  0b001000,
    'mov':        0b001001,
    'bar.sync':   0b001010,
    'nop':        0b111111,
}

# ALU功能码
ALU_FUNCS = {
    'add':   0b000000,
    'sub':   0b000001,
    'and':   0b000010,
    'or':    0b000011,
    'xor':   0b000100,
    'not':   0b000101,
    'shl':   0b000110,
    'shr.u': 0b000111,
    'shr.s': 0b001000,
}

# MUL功能码
MUL_FUNCS = {
    'mul.lo': 0b000000,
    'mul.hi': 0b000001,
    'mad.lo': 0b000010,
}

# 比较功能码
CMP_FUNCS = {
    'eq': 0b000000,
    'ne': 0b000001,
    'lt': 0b000010,
    'le': 0b000011,
    'gt': 0b000100,
    'ge': 0b000101,
}

# 特殊寄存器
SPECIAL_REGS = {
    '%tid.x':    0,
    '%tid.y':    1,
    '%tid.z':    2,
    '%ctaid.x':  3,
    '%ctaid.y':  4,
    '%ctaid.z':  5,
    '%ntid.x':   6,
    '%ntid.y':   7,
    '%ntid.z':   8,
    '%nctaid.x': 9,
    '%nctaid.y': 10,
    '%nctaid.z': 11,
}


@dataclass
class Instruction:
    """表示一条指令"""
    opcode: int = 0
    rd: int = 0
    ra: int = 0
    rb: int = 0
    rc: int = 0
    func: int = 0

    def encode(self) -> int:
        """编码为32位机器码"""
        return ((self.opcode & 0x3F) << 26 |
                (self.rd & 0x1F) << 21 |
                (self.ra & 0x1F) << 16 |
                (self.rb & 0x1F) << 11 |
                (self.rc & 0x1F) << 6 |
                (self.func & 0x3F))


def parse_register(reg_str: str) -> int:
    """解析寄存器名 (r0-r31)"""
    reg_str = reg_str.strip().lower()
    if reg_str.startswith('r'):
        num = int(reg_str[1:])
        if 0 <= num <= 31:
            return num
    raise ValueError(f"Invalid register: {reg_str}")


def parse_special_reg(reg_str: str) -> int:
    """解析特殊寄存器"""
    reg_str = reg_str.strip().lower()
    if reg_str in SPECIAL_REGS:
        return SPECIAL_REGS[reg_str]
    raise ValueError(f"Invalid special register: {reg_str}")


def parse_immediate(imm_str: str) -> int:
    """解析立即数"""
    imm_str = imm_str.strip()
    if imm_str.startswith('0x'):
        return int(imm_str, 16)
    elif imm_str.startswith('0b'):
        return int(imm_str, 2)
    else:
        return int(imm_str)


class PTXAssembler:
    """PTX汇编器"""

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[Instruction] = []
        self.current_addr = 0

    def first_pass(self, lines: List[str]):
        """第一遍：收集标签"""
        addr = 0
        for line in lines:
            line = line.split('//')[0].strip()  # 移除注释
            if not line:
                continue

            # 检查标签
            if line.endswith(':'):
                label = line[:-1].strip()
                self.labels[label] = addr
            else:
                addr += 4  # 每条指令4字节

    def second_pass(self, lines: List[str]) -> List[int]:
        """第二遍：生成机器码"""
        machine_code = []

        for line_num, line in enumerate(lines, 1):
            line = line.split('//')[0].strip()
            if not line or line.endswith(':'):
                continue

            try:
                inst = self.parse_instruction(line)
                machine_code.append(inst.encode())
                self.current_addr += 4
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                print(f"  {line}")
                raise

        return machine_code

    def parse_instruction(self, line: str) -> Instruction:
        """解析单条指令"""
        # 分割指令和操作数
        parts = line.replace(',', ' ').split()
        if not parts:
            return Instruction(opcode=OPCODES['nop'])

        mnemonic = parts[0].lower()
        operands = parts[1:] if len(parts) > 1 else []

        # NOP
        if mnemonic == 'nop':
            return Instruction(opcode=OPCODES['nop'])

        # ALU指令: add.s32 rd, ra, rb
        if mnemonic.startswith(('add', 'sub', 'and', 'or', 'xor', 'not', 'shl', 'shr')):
            return self.parse_alu(mnemonic, operands)

        # 乘法指令: mul.lo.s32 rd, ra, rb
        if mnemonic.startswith(('mul', 'mad')):
            return self.parse_mul(mnemonic, operands)

        # 比较指令: setp.eq.s32 p, ra, rb
        if mnemonic.startswith('setp'):
            return self.parse_setp(mnemonic, operands)

        # 分支: bra target / @p bra target
        if mnemonic == 'bra' or mnemonic.startswith('@'):
            return self.parse_branch(mnemonic, operands)

        # 内存加载: ld.global.s32 rd, [addr]
        if mnemonic.startswith('ld'):
            return self.parse_load(mnemonic, operands)

        # 内存存储: st.global.s32 [addr], rs
        if mnemonic.startswith('st'):
            return self.parse_store(mnemonic, operands)

        # 特殊寄存器: mov.u32 rd, %tid.x
        if mnemonic.startswith('mov'):
            return self.parse_mov(mnemonic, operands)

        # 同步: bar.sync 0
        if mnemonic.startswith('bar'):
            return Instruction(opcode=OPCODES['bar.sync'])

        raise ValueError(f"Unknown instruction: {mnemonic}")

    def parse_alu(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析ALU指令"""
        # 提取操作类型 (add.s32 -> add)
        op = mnemonic.split('.')[0]
        if op not in ALU_FUNCS:
            # 处理 shr.u/shr.s
            op = '.'.join(mnemonic.split('.')[:2])

        if op not in ALU_FUNCS:
            raise ValueError(f"Unknown ALU operation: {op}")

        inst = Instruction(opcode=OPCODES['alu'], func=ALU_FUNCS[op])

        if op == 'not':  # 一元操作
            inst.rd = parse_register(operands[0])
            inst.ra = parse_register(operands[1])
        else:  # 二元操作
            inst.rd = parse_register(operands[0])
            inst.ra = parse_register(operands[1])
            # 检查是否是立即数
            try:
                inst.rb = parse_register(operands[2])
            except ValueError:
                # 立即数 - 需要特殊处理，这里简化为固定值
                inst.rb = parse_immediate(operands[2]) & 0x1F

        return inst

    def parse_mul(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析乘法指令"""
        op = '.'.join(mnemonic.split('.')[:2])  # mul.lo
        if op not in MUL_FUNCS:
            raise ValueError(f"Unknown MUL operation: {op}")

        inst = Instruction(opcode=OPCODES['mul'], func=MUL_FUNCS[op])
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])

        if op == 'mad.lo' and len(operands) > 3:
            inst.rc = parse_register(operands[3])

        return inst

    def parse_setp(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析比较指令"""
        # setp.eq.s32 -> eq
        parts = mnemonic.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid setp format: {mnemonic}")
        cmp_op = parts[1]

        if cmp_op not in CMP_FUNCS:
            raise ValueError(f"Unknown comparison: {cmp_op}")

        inst = Instruction(opcode=OPCODES['setp'], func=CMP_FUNCS[cmp_op])
        # 谓词寄存器 (p0-p7)
        pred = operands[0].lower()
        if pred.startswith('p'):
            inst.rd = int(pred[1:]) & 0x7
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])

        return inst

    def parse_branch(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析分支指令"""
        inst = Instruction(opcode=OPCODES['bra'])

        # 条件分支 @p bra target
        if mnemonic.startswith('@'):
            pred = mnemonic[1:].lower()
            if pred.startswith('p'):
                inst.rc = int(pred[1:]) & 0x7
            target = operands[1] if len(operands) > 1 else operands[0]
        else:
            target = operands[0]

        # 计算分支偏移
        if target in self.labels:
            offset = (self.labels[target] - self.current_addr - 4) // 4
            inst.ra = (offset >> 16) & 0x1F
            inst.rb = (offset >> 11) & 0x1F
            inst.func = offset & 0x3F
        else:
            # 尝试解析为数字
            offset = parse_immediate(target)
            inst.ra = (offset >> 16) & 0x1F
            inst.rb = (offset >> 11) & 0x1F
            inst.func = offset & 0x3F

        return inst

    def parse_load(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析加载指令"""
        if 'shared' in mnemonic:
            opcode = OPCODES['ld.shared']
        else:
            opcode = OPCODES['ld.global']

        inst = Instruction(opcode=opcode)
        inst.rd = parse_register(operands[0])

        # 解析地址 [ra] 或 [ra+offset]
        addr_str = ' '.join(operands[1:]).strip('[]')
        if '+' in addr_str:
            parts = addr_str.split('+')
            inst.ra = parse_register(parts[0])
            # offset存储在rb和func中
            offset = parse_immediate(parts[1])
            inst.rb = (offset >> 6) & 0x1F
            inst.func = offset & 0x3F
        else:
            inst.ra = parse_register(addr_str)

        return inst

    def parse_store(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析存储指令"""
        if 'shared' in mnemonic:
            opcode = OPCODES['st.shared']
        else:
            opcode = OPCODES['st.global']

        inst = Instruction(opcode=opcode)

        # 解析地址
        addr_str = operands[0].strip('[]')
        if '+' in addr_str:
            parts = addr_str.split('+')
            inst.ra = parse_register(parts[0])
            offset = parse_immediate(parts[1])
            inst.rc = (offset >> 6) & 0x1F
            inst.func = offset & 0x3F
        else:
            inst.ra = parse_register(addr_str)

        inst.rb = parse_register(operands[1])

        return inst

    def parse_mov(self, mnemonic: str, operands: List[str]) -> Instruction:
        """解析MOV指令"""
        inst = Instruction(opcode=OPCODES['mov'])
        inst.rd = parse_register(operands[0])

        # 检查是否是特殊寄存器
        src = operands[1].lower()
        if src.startswith('%'):
            inst.ra = parse_special_reg(src)
        else:
            # 立即数加载
            imm = parse_immediate(src)
            inst.ra = (imm >> 16) & 0x1F
            inst.rb = (imm >> 11) & 0x1F
            inst.func = imm & 0x3F

        return inst


def assemble_file(input_file: str, output_file: str):
    """汇编单个文件"""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    assembler = PTXAssembler()

    # 两遍扫描
    assembler.first_pass(lines)
    machine_code = assembler.second_pass(lines)

    # 输出
    with open(output_file, 'w') as f:
        for i, code in enumerate(machine_code):
            f.write(f"{code:08x}\n")

    print(f"Assembled {len(machine_code)} instructions to {output_file}")
    return machine_code


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: ptx_assembler.py <input.ptx> [-o output.hex]")
        print("\nExample PTX code:")
        print("  mov.u32 r0, %tid.x     // r0 = thread_id")
        print("  add.s32 r1, r0, r0     // r1 = r0 + r0")
        print("  mul.lo.s32 r2, r0, r1  // r2 = r0 * r1")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '-o' else 'output.hex'

    assemble_file(input_file, output_file)


if __name__ == '__main__':
    main()
