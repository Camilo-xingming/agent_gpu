#!/usr/bin/env python3
"""
RalphGPU PTX Assembler - Complete PTX ISA 9.1 Support
Assembles PTX-style instructions to RalphGPU machine code

Usage:
    python ptx_assembler.py input.ptx -o output.hex
    python ptx_assembler.py --test  # Run instruction coverage test
"""

import sys
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum

#============================================================================
# Opcode Definitions (6-bit) - matches gpu_defines.vh
#============================================================================
class Opcode(IntEnum):
    # Basic operations
    ALU         = 0b000000
    MUL         = 0b000001
    DIV         = 0b000010
    SETP        = 0b000011
    BRANCH      = 0b000100
    LD_GLOBAL   = 0b000101
    ST_GLOBAL   = 0b000110
    LD_SHARED   = 0b000111
    ST_SHARED   = 0b001000
    MOV_SPECIAL = 0b001001
    BAR_SYNC    = 0b001010
    EXIT        = 0b001011
    RET         = 0b001100

    # FP operations
    FP32_ARITH   = 0b001101
    FP32_SPECIAL = 0b001110
    FP64_ARITH   = 0b001111
    FP16_ARITH   = 0b010000
    CVT          = 0b010001

    # Extended memory
    LD_PARAM    = 0b010010
    LD_CONST    = 0b010011
    LD_LOCAL    = 0b010100
    ST_LOCAL    = 0b010101
    LD_V2       = 0b010110
    LD_V4       = 0b010111
    ST_V2       = 0b011000
    ST_V4       = 0b011001

    # Atomic operations
    ATOM        = 0b011010
    RED         = 0b011011

    # Warp operations
    SHFL        = 0b011100
    VOTE        = 0b011101
    REDUX       = 0b011110

    # Tensor core
    WMMA_LOAD   = 0b011111
    WMMA_STORE  = 0b100000
    WMMA_MMA    = 0b100001
    MMA         = 0b100010

    # Control flow extension
    CALL        = 0b100011
    MEMBAR      = 0b100100

    # Video
    VIDEO       = 0b100101

    # Texture/Surface
    TEX         = 0b100110
    TXQ         = 0b100111
    SULD        = 0b101000
    SUST        = 0b101001
    SURED       = 0b101010

    # Extended operations (Phase 10)
    CPASYNC     = 0b101011
    PREFETCH    = 0b101100
    WGMMA_LOAD  = 0b101101
    WGMMA_STORE = 0b101110
    WGMMA_MMA   = 0b101111

    MOV_IMM     = 0b110000  # Move immediate to register
    ALU_IMM     = 0b110001  # ALU with 16-bit immediate

    NOP         = 0b111111

#============================================================================
# ALU Function Codes (6-bit)
#============================================================================
class AluFunc(IntEnum):
    ADD     = 0b000000
    SUB     = 0b000001
    AND     = 0b000010
    OR      = 0b000011
    XOR     = 0b000100
    NOT     = 0b000101
    SHL     = 0b000110
    SHR_U   = 0b000111
    SHR_S   = 0b001000
    ABS     = 0b001001
    NEG     = 0b001010
    MIN_S   = 0b001011
    MIN_U   = 0b001100
    MAX_S   = 0b001101
    MAX_U   = 0b001110
    POPC    = 0b001111
    CLZ     = 0b010000
    BFIND   = 0b010001
    BREV    = 0b010010
    BFE_S   = 0b010011
    BFE_U   = 0b010100
    BFI     = 0b010101
    PRMT    = 0b010110
    SAD     = 0b010111
    CNOT    = 0b011010
    BMSK    = 0b011011
    SZEXT   = 0b011100
    FNS     = 0b011101
    SHF_L   = 0b011110
    SHF_R   = 0b011111
    LOP3    = 0b100101
    SELP    = 0b011000
    SLCT    = 0b011001
    # Carry operations
    ADD_CC  = 0b100000
    ADDC    = 0b100001
    SUB_CC  = 0b100010
    SUBC    = 0b100011
    MUL_WIDE = 0b100100

#============================================================================
# MUL Function Codes
#============================================================================
class MulFunc(IntEnum):
    MUL_LO  = 0b000000
    MUL_HI  = 0b000001
    MAD_LO  = 0b000010
    MAD_HI  = 0b000011
    MUL24   = 0b000100
    MAD24   = 0b000101
    MAD_LO_CC = 0b100101
    MADC_LO   = 0b100110

# Division/Modulo Function Codes (OP_DIV)
class DivFunc(IntEnum):
    DIV_S = 0b000000
    DIV_U = 0b000001
    REM_S = 0b000010
    REM_U = 0b000011

#============================================================================
# Compare Function Codes
#============================================================================
class CmpFunc(IntEnum):
    EQ = 0b000000
    NE = 0b000001
    LT = 0b000010
    LE = 0b000011
    GT = 0b000100
    GE = 0b000101

#============================================================================
# FP32 Function Codes
#============================================================================
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

#============================================================================
# FP32 Special Function Codes
#============================================================================
class Fp32SpecialFunc(IntEnum):
    RCP   = 0b000000
    SQRT  = 0b000001
    RSQRT = 0b000010
    SIN   = 0b000011
    COS   = 0b000100
    LG2   = 0b000101
    EX2   = 0b000110
    TANH  = 0b000111
    TESTP = 0b001000
    COPYSIGN = 0b001001

#============================================================================
# FP64 Function Codes
#============================================================================
class Fp64Func(IntEnum):
    ADD   = 0b000000
    SUB   = 0b000001
    MUL   = 0b000010
    DIV   = 0b000011
    FMA   = 0b000100
    NEG   = 0b000101
    ABS   = 0b000110
    MIN   = 0b000111
    MAX   = 0b001000
    SQRT  = 0b001001
    RSQRT = 0b001010
    RCP   = 0b001011
    COPYSIGN = 0b001100
    TESTP = 0b001101

#============================================================================
# FP16/BF16 Function Codes
#============================================================================
class Fp16Func(IntEnum):
    ADD     = 0b000000
    SUB     = 0b000001
    MUL     = 0b000010
    FMA     = 0b000011
    NEG     = 0b000100
    ABS     = 0b000101
    MIN     = 0b000110
    MAX     = 0b000111
    TANH    = 0b001000
    EX2     = 0b001001
    # BF16
    BF16_ADD = 0b010000
    BF16_SUB = 0b010001
    BF16_MUL = 0b010010
    BF16_FMA = 0b010011
    # Packed FP16x2
    F16X2_ADD = 0b100000
    F16X2_SUB = 0b100001
    F16X2_MUL = 0b100010
    F16X2_FMA = 0b100011

#============================================================================
# CVT Function Codes
#============================================================================
class CvtFunc(IntEnum):
    S32_F32 = 0b000000
    U32_F32 = 0b000001
    F32_S32 = 0b000010
    F32_U32 = 0b000011
    F32_F64 = 0b000100
    F64_F32 = 0b000101
    F32_F16 = 0b101000  # unique code 40, not conflicting with ALU FUNC_SHL
    F16_F32 = 0b101001  # unique code 41, not conflicting with ALU FUNC_SHR_U
    S64_F64 = 0b001000
    U64_F64 = 0b001001
    F64_S64 = 0b001010
    F64_U64 = 0b001011
    PACK    = 0b101100

#============================================================================
# Atomic Function Codes
#============================================================================
class AtomFunc(IntEnum):
    ADD   = 0b000000
    MIN_S = 0b000001
    MIN_U = 0b000010
    MAX_S = 0b000011
    MAX_U = 0b000100
    INC   = 0b000101
    DEC   = 0b000110
    AND   = 0b000111
    OR    = 0b001000
    XOR   = 0b001001
    EXCH  = 0b001010
    CAS   = 0b001011

#============================================================================
# Shuffle Function Codes
#============================================================================
class ShflFunc(IntEnum):
    IDX  = 0b000000
    UP   = 0b000001
    DOWN = 0b000010
    BFLY = 0b000011

#============================================================================
# Vote Function Codes
#============================================================================
class VoteFunc(IntEnum):
    ALL    = 0b000000
    ANY    = 0b000001
    UNI    = 0b000010
    BALLOT = 0b000011

#============================================================================
# Redux Function Codes
#============================================================================
class ReduxFunc(IntEnum):
    ADD = 0b000000
    MIN = 0b000001
    MAX = 0b000010
    AND = 0b000011
    OR  = 0b000100

#============================================================================
# Texture Function Codes
#============================================================================
class TexFunc(IntEnum):
    TEX_1D    = 0b000000
    TEX_2D    = 0b000001
    TEX_3D    = 0b000010
    TEX_CUBE  = 0b000011
    TEX_A1D   = 0b000100
    TEX_A2D   = 0b000101
    TEX_LEVEL = 0b001000
    TEX_GRAD  = 0b001001
    TEX_GATHER = 0b001010
    # Query
    TXQ_WIDTH  = 0b010000
    TXQ_HEIGHT = 0b010001
    TXQ_DEPTH  = 0b010010
    TXQ_LEVELS = 0b010011

#============================================================================
# Surface Function Codes
#============================================================================
class SurfFunc(IntEnum):
    SURF_1D  = 0b000000
    SURF_2D  = 0b000001
    SURF_3D  = 0b000010
    SURF_A1D = 0b000100
    SURF_A2D = 0b000101

#============================================================================
# Video Function Codes
#============================================================================
class VideoFunc(IntEnum):
    VADD      = 0b000000
    VSUB      = 0b000001
    VABSDIFF  = 0b000010
    VMIN      = 0b000011
    VMAX      = 0b000100
    VSHL      = 0b000101
    VSHR      = 0b000110
    VMAD      = 0b000111
    VSET      = 0b001000
    VADD4     = 0b010000
    VSUB4     = 0b010001
    VABSDIFF4 = 0b010010
    VADD2     = 0b010100
    VSUB2     = 0b010101
    VMUL2     = 0b010110
    DP4A      = 0b100010  # VIDEO_DP4A_ALU - routed through ALU path
    DP2A      = 0b100011  # VIDEO_DP2A_ALU - routed through ALU path

#============================================================================
# WMMA Function Codes
#============================================================================
class WmmaFunc(IntEnum):
    M16N16K16 = 0b000000
    M8N8K4    = 0b000001
    M32N8K16  = 0b000010

#============================================================================
# Tensor Core Data Types (matches gpu_defines.vh)
#============================================================================
class TensorDataType(IntEnum):
    FP16     = 0b000
    BF16     = 0b001
    INT8     = 0b010
    INT4     = 0b011
    FP8_E4M3 = 0b100
    FP8_E5M2 = 0b101
    FP4_E2M1 = 0b110
    FP4_E3M0 = 0b111

#============================================================================
# WGMMA Function Codes
#============================================================================
class WgmmaFunc(IntEnum):
    M64N8K16   = 0b000000
    M64N16K16  = 0b000001
    M64N32K16  = 0b000010
    M64N64K16  = 0b000011
    M64N128K16 = 0b000100
    M64N256K16 = 0b000101
    FENCE      = 0b010000
    COMMIT     = 0b010001
    WAIT       = 0b010010

#============================================================================
# Async Copy Function Codes
#============================================================================
class CpAsyncFunc(IntEnum):
    CA       = 0b000000
    CG       = 0b000001
    COMMIT   = 0b000010
    WAIT     = 0b000011
    WAIT_ALL = 0b000100
    BULK     = 0b001000

#============================================================================
# Prefetch Function Codes
#============================================================================
class PrefetchFunc(IntEnum):
    L1  = 0b000000
    L2  = 0b000001
    L1U = 0b000010

#============================================================================
# Membar Function Codes
#============================================================================
class MembarFunc(IntEnum):
    CTA = 0b000000
    GL  = 0b000001
    SYS = 0b000010

#============================================================================
# Special Registers
#============================================================================
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
    '%laneid':   12,
    '%warpid':   13,
    '%smid':     14,
    '%activemask': 15,
}

#============================================================================
# Instruction Encoding
#============================================================================
@dataclass
class Instruction:
    """Represents a single 32-bit instruction"""
    opcode: int = 0
    rd: int = 0
    ra: int = 0
    rb: int = 0
    rc: int = 0
    func: int = 0
    imm16: int = 0  # For immediate-mode instructions

    def encode(self) -> int:
        """Encode to 32-bit machine code"""
        if self.opcode == Opcode.MOV_IMM:
            # MOV_IMM format: [31:26]=opcode, [25:21]=rd, [15:0]=imm16
            return ((self.opcode & 0x3F) << 26 |
                    (self.rd & 0x1F) << 21 |
                    (self.imm16 & 0xFFFF))
        elif self.opcode == Opcode.ALU_IMM:
            # ALU_IMM format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:0]=imm16
            # (imm16 contains func in [15:10] and imm10 in [9:0])
            return ((self.opcode & 0x3F) << 26 |
                    (self.rd & 0x1F) << 21 |
                    (self.ra & 0x1F) << 16 |
                    (self.imm16 & 0xFFFF))
        elif self.opcode == Opcode.BRANCH:
            # BRANCH format: {opcode[31:26], type[25:24], unused[23:21], ra[20:16], offset[15:0]}
            # rd contains branch_type in bits [4:3]
            # ra contains condition register
            # rb, rc, func combined form offset[15:0]
            offset = ((self.rb & 0x1F) << 11 |
                      (self.rc & 0x1F) << 6 |
                      (self.func & 0x3F))
            return ((self.opcode & 0x3F) << 26 |
                    (self.rd & 0x1F) << 21 |
                    (self.ra & 0x1F) << 16 |
                    (offset & 0xFFFF))
        else:
            return ((self.opcode & 0x3F) << 26 |
                    (self.rd & 0x1F) << 21 |
                    (self.ra & 0x1F) << 16 |
                    (self.rb & 0x1F) << 11 |
                    (self.rc & 0x1F) << 6 |
                    (self.func & 0x3F))

#============================================================================
# Parser Utilities
#============================================================================
def parse_register(reg_str: str) -> int:
    """Parse register name (r0-r31 or rd0-rd15 for 64-bit)"""
    reg_str = reg_str.strip().lower()
    if reg_str.startswith('rd'):  # 64-bit register pair
        num = int(reg_str[2:])
        return num * 2  # Map to even register
    elif reg_str.startswith('r'):
        num = int(reg_str[1:])
        if 0 <= num <= 31:
            return num
    elif reg_str.startswith('p'):  # Predicate register
        num = int(reg_str[1:])
        return num & 0x7
    raise ValueError(f"Invalid register: {reg_str}")

def parse_special_reg(reg_str: str) -> int:
    """Parse special register"""
    reg_str = reg_str.strip().lower()
    if reg_str in SPECIAL_REGS:
        return SPECIAL_REGS[reg_str]
    raise ValueError(f"Invalid special register: {reg_str}")

def parse_immediate(imm_str: str) -> int:
    """Parse immediate value"""
    imm_str = imm_str.strip()
    if imm_str.startswith('0x'):
        return int(imm_str, 16)
    elif imm_str.startswith('0b'):
        return int(imm_str, 2)
    elif imm_str.startswith('-'):
        return int(imm_str)
    else:
        return int(imm_str)

def is_signed_type(type_str: str) -> bool:
    """Check if type is signed"""
    return '.s' in type_str or type_str.endswith('s32') or type_str.endswith('s64')

#============================================================================
# PTX Assembler
#============================================================================
class PTXAssembler:
    """Complete PTX ISA 9.1 Assembler"""

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.current_addr = 0
        self.instructions_assembled: List[str] = []

    def first_pass(self, lines: List[str]):
        """First pass: collect labels"""
        addr = 0
        for line in lines:
            line = line.split('//')[0].strip()
            if not line or line.startswith('.'):
                continue
            if line.endswith(':'):
                label = line[:-1].strip()
                self.labels[label] = addr
            else:
                addr += 4

    def second_pass(self, lines: List[str]) -> List[int]:
        """Second pass: generate machine code"""
        machine_code = []
        self.current_addr = 0

        for line_num, line in enumerate(lines, 1):
            line = line.split('//')[0].strip()
            if not line or line.endswith(':') or line.startswith('.'):
                continue

            try:
                inst = self.parse_instruction(line)
                machine_code.append(inst.encode())
                self.instructions_assembled.append(line.split()[0].lower())
                self.current_addr += 4
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                print(f"  {line}")
                raise

        return machine_code

    def parse_instruction(self, line: str) -> Instruction:
        """Parse a single instruction"""
        # Handle predicated instructions: @p bra target
        predicate = None
        if line.startswith('@'):
            parts = line.split(None, 1)
            pred_str = parts[0][1:]  # Remove @
            if pred_str.startswith('!'):
                predicate = (parse_register(pred_str[1:]), True)  # Negated
            else:
                predicate = (parse_register(pred_str), False)
            line = parts[1] if len(parts) > 1 else ''

        # Split instruction and operands
        parts = line.replace(',', ' ').split()
        if not parts:
            return Instruction(opcode=Opcode.NOP)

        mnemonic = parts[0].lower()
        operands = parts[1:] if len(parts) > 1 else []

        # Dispatch to appropriate handler
        inst = self._dispatch_instruction(mnemonic, operands)

        # Add predicate if present
        if predicate is not None:
            inst.rc = predicate[0] | (0x10 if predicate[1] else 0)

        return inst

    def _dispatch_instruction(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Dispatch to appropriate instruction parser"""

        # NOP
        if mnemonic == 'nop':
            return Instruction(opcode=Opcode.NOP)

        # EXIT/RET
        if mnemonic == 'exit':
            return Instruction(opcode=Opcode.EXIT)
        if mnemonic == 'ret':
            return Instruction(opcode=Opcode.RET)

        #================================================================
        # Integer Arithmetic (ALU)
        #================================================================
        if mnemonic.startswith(('add.s', 'add.u', 'add.b')):
            return self._parse_alu_binary(AluFunc.ADD, operands)
        if mnemonic.startswith('add.cc'):
            return self._parse_alu_binary(AluFunc.ADD_CC, operands)
        if mnemonic.startswith('addc'):
            return self._parse_alu_binary(AluFunc.ADDC, operands)
        if mnemonic.startswith(('sub.s', 'sub.u', 'sub.b')):
            return self._parse_alu_binary(AluFunc.SUB, operands)
        if mnemonic.startswith('sub.cc'):
            return self._parse_alu_binary(AluFunc.SUB_CC, operands)
        if mnemonic.startswith('subc'):
            return self._parse_alu_binary(AluFunc.SUBC, operands)
        if mnemonic.startswith(('and.b', 'and.pred')):
            return self._parse_alu_binary(AluFunc.AND, operands)
        if mnemonic.startswith(('or.b', 'or.pred')):
            return self._parse_alu_binary(AluFunc.OR, operands)
        if mnemonic.startswith(('xor.b', 'xor.pred')):
            return self._parse_alu_binary(AluFunc.XOR, operands)
        if mnemonic.startswith('not.b'):
            return self._parse_alu_unary(AluFunc.NOT, operands)
        if mnemonic.startswith('shl.b'):
            return self._parse_alu_binary(AluFunc.SHL, operands)
        if mnemonic.startswith('shr.u'):
            return self._parse_alu_binary(AluFunc.SHR_U, operands)
        if mnemonic.startswith('shr.s'):
            return self._parse_alu_binary(AluFunc.SHR_S, operands)
        if mnemonic.startswith('abs.s'):
            return self._parse_alu_unary(AluFunc.ABS, operands)
        if mnemonic.startswith('neg.s'):
            return self._parse_alu_unary(AluFunc.NEG, operands)
        if mnemonic.startswith('min.s'):
            return self._parse_alu_binary(AluFunc.MIN_S, operands)
        if mnemonic.startswith('min.u'):
            return self._parse_alu_binary(AluFunc.MIN_U, operands)
        if mnemonic.startswith('max.s'):
            return self._parse_alu_binary(AluFunc.MAX_S, operands)
        if mnemonic.startswith('max.u'):
            return self._parse_alu_binary(AluFunc.MAX_U, operands)
        if mnemonic.startswith('popc.b'):
            return self._parse_alu_unary(AluFunc.POPC, operands)
        if mnemonic.startswith('clz.b'):
            return self._parse_alu_unary(AluFunc.CLZ, operands)
        if mnemonic.startswith('bfind'):
            return self._parse_alu_unary(AluFunc.BFIND, operands)
        if mnemonic.startswith('brev.b'):
            return self._parse_alu_unary(AluFunc.BREV, operands)
        if mnemonic.startswith('bfe.s'):
            return self._parse_alu_ternary(AluFunc.BFE_S, operands)
        if mnemonic.startswith('bfe.u'):
            return self._parse_alu_ternary(AluFunc.BFE_U, operands)
        if mnemonic.startswith('bfi.b'):
            return self._parse_alu_ternary(AluFunc.BFI, operands)
        if mnemonic.startswith('prmt.b'):
            return self._parse_alu_ternary(AluFunc.PRMT, operands)
        if mnemonic.startswith('sad'):
            return self._parse_alu_ternary(AluFunc.SAD, operands)
        if mnemonic.startswith('cnot'):
            return self._parse_alu_binary(AluFunc.CNOT, operands)
        if mnemonic.startswith('bmsk'):
            return self._parse_alu_binary(AluFunc.BMSK, operands)
        if mnemonic.startswith('szext'):
            return self._parse_alu_binary(AluFunc.SZEXT, operands)
        if mnemonic.startswith('fns'):
            return self._parse_alu_unary(AluFunc.FNS, operands)
        if mnemonic.startswith('shf.l'):
            return self._parse_alu_ternary(AluFunc.SHF_L, operands)
        if mnemonic.startswith('shf.r'):
            return self._parse_alu_ternary(AluFunc.SHF_R, operands)
        if mnemonic.startswith('lop3'):
            return self._parse_alu_ternary(AluFunc.LOP3, operands)
        if mnemonic.startswith('selp'):
            return self._parse_alu_ternary(AluFunc.SELP, operands)
        if mnemonic.startswith('slct'):
            return self._parse_alu_ternary(AluFunc.SLCT, operands)
        if mnemonic.startswith('mul.wide'):
            return self._parse_mul_wide(operands)

        #================================================================
        # Multiplication
        #================================================================
        if mnemonic.startswith('mul.lo'):
            return self._parse_mul(MulFunc.MUL_LO, operands)
        if mnemonic.startswith('mul.hi'):
            return self._parse_mul(MulFunc.MUL_HI, operands)
        if mnemonic.startswith('mad.lo'):
            return self._parse_mad(MulFunc.MAD_LO, operands)
        if mnemonic.startswith('mad.hi'):
            return self._parse_mad(MulFunc.MAD_HI, operands)
        if mnemonic.startswith('mul24'):
            return self._parse_mul(MulFunc.MUL24, operands)
        if mnemonic.startswith('mad24'):
            return self._parse_mad(MulFunc.MAD24, operands)
        if mnemonic.startswith('mad.cc'):
            return self._parse_mad(MulFunc.MAD_LO_CC, operands)
        if mnemonic.startswith('madc'):
            return self._parse_mad(MulFunc.MADC_LO, operands)

        #================================================================
        # Division
        #================================================================
        if mnemonic.startswith(('div.s', 'div.u')):
            return self._parse_div(mnemonic, operands, is_rem=False)
        if mnemonic.startswith(('rem.s', 'rem.u')):
            return self._parse_div(mnemonic, operands, is_rem=True)

        #================================================================
        # Comparison (setp)
        #================================================================
        if mnemonic.startswith('setp'):
            return self._parse_setp(mnemonic, operands)

        #================================================================
        # Branch/Call
        #================================================================
        if mnemonic == 'bra' or mnemonic.startswith('bra.'):
            return self._parse_branch(mnemonic, operands)
        if mnemonic == 'call':
            return self._parse_call(operands)

        #================================================================
        # Memory Operations
        #================================================================
        if mnemonic.startswith('ld.global'):
            return self._parse_load(Opcode.LD_GLOBAL, operands)
        if mnemonic.startswith('st.global'):
            return self._parse_store(Opcode.ST_GLOBAL, operands)
        if mnemonic.startswith('ld.shared'):
            return self._parse_load(Opcode.LD_SHARED, operands)
        if mnemonic.startswith('st.shared'):
            return self._parse_store(Opcode.ST_SHARED, operands)
        if mnemonic.startswith('ld.param'):
            return self._parse_load(Opcode.LD_PARAM, operands)
        if mnemonic.startswith('ld.const'):
            return self._parse_load(Opcode.LD_CONST, operands)
        if mnemonic.startswith('ld.local'):
            return self._parse_load(Opcode.LD_LOCAL, operands)
        if mnemonic.startswith('st.local'):
            return self._parse_store(Opcode.ST_LOCAL, operands)

        # Vector loads/stores
        if mnemonic.startswith('ld.v2'):
            return self._parse_load(Opcode.LD_V2, operands)
        if mnemonic.startswith('ld.v4'):
            return self._parse_load(Opcode.LD_V4, operands)
        if mnemonic.startswith('st.v2'):
            return self._parse_store(Opcode.ST_V2, operands)
        if mnemonic.startswith('st.v4'):
            return self._parse_store(Opcode.ST_V4, operands)

        # Cache hints
        if mnemonic.startswith('ld.ca'):
            return self._parse_load_cached(Opcode.LD_GLOBAL, operands, 0b001)
        if mnemonic.startswith('ld.cg'):
            return self._parse_load_cached(Opcode.LD_GLOBAL, operands, 0b010)
        if mnemonic.startswith('ld.cs'):
            return self._parse_load_cached(Opcode.LD_GLOBAL, operands, 0b011)
        if mnemonic.startswith('ld.lu'):
            return self._parse_load_cached(Opcode.LD_GLOBAL, operands, 0b100)
        if mnemonic.startswith('ld.cv'):
            return self._parse_load_cached(Opcode.LD_GLOBAL, operands, 0b101)
        if mnemonic.startswith('st.wb'):
            return self._parse_store_cached(Opcode.ST_GLOBAL, operands, 0b110)
        if mnemonic.startswith('st.wt'):
            return self._parse_store_cached(Opcode.ST_GLOBAL, operands, 0b111)

        #================================================================
        # MOV / Special Registers
        #================================================================
        if mnemonic.startswith('mov'):
            return self._parse_mov(operands)

        #================================================================
        # Synchronization
        #================================================================
        if mnemonic.startswith('bar.sync'):
            return self._parse_bar_sync(operands)
        if mnemonic.startswith('membar'):
            return self._parse_membar(mnemonic, operands)

        #================================================================
        # FP32 Arithmetic
        #================================================================
        if mnemonic.startswith('add.f32') or mnemonic.startswith('add.rn.f32'):
            return self._parse_fp32(Fp32Func.ADD, operands)
        if mnemonic.startswith('sub.f32') or mnemonic.startswith('sub.rn.f32'):
            return self._parse_fp32(Fp32Func.SUB, operands)
        if mnemonic.startswith('mul.f32') or mnemonic.startswith('mul.rn.f32'):
            return self._parse_fp32(Fp32Func.MUL, operands)
        if mnemonic.startswith('div.f32') or mnemonic.startswith('div.rn.f32'):
            return self._parse_fp32(Fp32Func.DIV, operands)
        if mnemonic.startswith('fma.rn.f32') or mnemonic.startswith('fma.f32'):
            return self._parse_fp32_fma(operands)
        if mnemonic.startswith('neg.f32'):
            return self._parse_fp32_unary(Fp32Func.NEG, operands)
        if mnemonic.startswith('abs.f32'):
            return self._parse_fp32_unary(Fp32Func.ABS, operands)
        if mnemonic.startswith('min.f32'):
            return self._parse_fp32(Fp32Func.MIN, operands)
        if mnemonic.startswith('max.f32'):
            return self._parse_fp32(Fp32Func.MAX, operands)

        #================================================================
        # FP32 Special Functions
        #================================================================
        if mnemonic.startswith('rcp.f32') or mnemonic.startswith('rcp.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.RCP, operands)
        if mnemonic.startswith('sqrt.f32') or mnemonic.startswith('sqrt.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.SQRT, operands)
        if mnemonic.startswith('rsqrt.f32') or mnemonic.startswith('rsqrt.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.RSQRT, operands)
        if mnemonic.startswith('sin.f32') or mnemonic.startswith('sin.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.SIN, operands)
        if mnemonic.startswith('cos.f32') or mnemonic.startswith('cos.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.COS, operands)
        if mnemonic.startswith('lg2.f32') or mnemonic.startswith('lg2.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.LG2, operands)
        if mnemonic.startswith('ex2.f32') or mnemonic.startswith('ex2.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.EX2, operands)
        if mnemonic.startswith('tanh.f32') or mnemonic.startswith('tanh.approx.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.TANH, operands)
        if mnemonic.startswith('testp.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.TESTP, operands)
        if mnemonic.startswith('copysign.f32'):
            return self._parse_fp32_special(Fp32SpecialFunc.COPYSIGN, operands)

        #================================================================
        # FP64 Arithmetic
        #================================================================
        if mnemonic.startswith('add.f64') or mnemonic.startswith('add.rn.f64'):
            return self._parse_fp64(Fp64Func.ADD, operands)
        if mnemonic.startswith('sub.f64') or mnemonic.startswith('sub.rn.f64'):
            return self._parse_fp64(Fp64Func.SUB, operands)
        if mnemonic.startswith('mul.f64') or mnemonic.startswith('mul.rn.f64'):
            return self._parse_fp64(Fp64Func.MUL, operands)
        if mnemonic.startswith('div.f64') or mnemonic.startswith('div.rn.f64'):
            return self._parse_fp64(Fp64Func.DIV, operands)
        if mnemonic.startswith('fma.rn.f64') or mnemonic.startswith('fma.f64'):
            return self._parse_fp64_fma(operands)
        if mnemonic.startswith('neg.f64'):
            return self._parse_fp64_unary(Fp64Func.NEG, operands)
        if mnemonic.startswith('abs.f64'):
            return self._parse_fp64_unary(Fp64Func.ABS, operands)
        if mnemonic.startswith('min.f64'):
            return self._parse_fp64(Fp64Func.MIN, operands)
        if mnemonic.startswith('max.f64'):
            return self._parse_fp64(Fp64Func.MAX, operands)
        if mnemonic.startswith('sqrt.f64'):
            return self._parse_fp64_unary(Fp64Func.SQRT, operands)
        if mnemonic.startswith('rsqrt.f64'):
            return self._parse_fp64_unary(Fp64Func.RSQRT, operands)
        if mnemonic.startswith('rcp.f64') or mnemonic.startswith('rcp.approx.f64') or mnemonic.startswith('rcp.approx.ftz.f64'):
            return self._parse_fp64_unary(Fp64Func.RCP, operands)
        if mnemonic.startswith('copysign.f64'):
            return self._parse_fp64_unary(Fp64Func.COPYSIGN, operands)
        if mnemonic.startswith('testp.f64'):
            return self._parse_fp64_unary(Fp64Func.TESTP, operands)

        #================================================================
        # FP16/BF16
        #================================================================
        if mnemonic.startswith('add.f16x2'):
            return self._parse_fp16(Fp16Func.F16X2_ADD, operands)
        if mnemonic.startswith('sub.f16x2'):
            return self._parse_fp16(Fp16Func.F16X2_SUB, operands)
        if mnemonic.startswith('mul.f16x2'):
            return self._parse_fp16(Fp16Func.F16X2_MUL, operands)
        if mnemonic.startswith('fma.f16x2') or mnemonic.startswith('fma.rn.f16x2'):
            return self._parse_fp16_fma(Fp16Func.F16X2_FMA, operands)
        if mnemonic.startswith('add.f16'):
            return self._parse_fp16(Fp16Func.ADD, operands)
        if mnemonic.startswith('sub.f16'):
            return self._parse_fp16(Fp16Func.SUB, operands)
        if mnemonic.startswith('mul.f16'):
            return self._parse_fp16(Fp16Func.MUL, operands)
        if mnemonic.startswith('fma.f16') or mnemonic.startswith('fma.rn.f16'):
            return self._parse_fp16_fma(Fp16Func.FMA, operands)
        if mnemonic.startswith('neg.f16'):
            return self._parse_fp16_unary(Fp16Func.NEG, operands)
        if mnemonic.startswith('abs.f16'):
            return self._parse_fp16_unary(Fp16Func.ABS, operands)
        if mnemonic.startswith('min.f16'):
            return self._parse_fp16(Fp16Func.MIN, operands)
        if mnemonic.startswith('max.f16'):
            return self._parse_fp16(Fp16Func.MAX, operands)
        if mnemonic.startswith('tanh.f16'):
            return self._parse_fp16_unary(Fp16Func.TANH, operands)
        if mnemonic.startswith('ex2.f16'):
            return self._parse_fp16_unary(Fp16Func.EX2, operands)
        # BF16
        if mnemonic.startswith('add.bf16'):
            return self._parse_fp16(Fp16Func.BF16_ADD, operands)
        if mnemonic.startswith('sub.bf16'):
            return self._parse_fp16(Fp16Func.BF16_SUB, operands)
        if mnemonic.startswith('mul.bf16'):
            return self._parse_fp16(Fp16Func.BF16_MUL, operands)
        if mnemonic.startswith('fma.bf16'):
            return self._parse_fp16_fma(Fp16Func.BF16_FMA, operands)

        #================================================================
        # Type Conversion
        #================================================================
        if mnemonic.startswith('cvt'):
            return self._parse_cvt(mnemonic, operands)

        #================================================================
        # Atomic Operations
        #================================================================
        if mnemonic.startswith('atom.add'):
            return self._parse_atomic(AtomFunc.ADD, operands)
        if mnemonic.startswith('atom.min.s'):
            return self._parse_atomic(AtomFunc.MIN_S, operands)
        if mnemonic.startswith('atom.min.u') or mnemonic.startswith('atom.min'):
            return self._parse_atomic(AtomFunc.MIN_U, operands)
        if mnemonic.startswith('atom.max.s'):
            return self._parse_atomic(AtomFunc.MAX_S, operands)
        if mnemonic.startswith('atom.max.u') or mnemonic.startswith('atom.max'):
            return self._parse_atomic(AtomFunc.MAX_U, operands)
        if mnemonic.startswith('atom.inc'):
            return self._parse_atomic(AtomFunc.INC, operands)
        if mnemonic.startswith('atom.dec'):
            return self._parse_atomic(AtomFunc.DEC, operands)
        if mnemonic.startswith('atom.and'):
            return self._parse_atomic(AtomFunc.AND, operands)
        if mnemonic.startswith('atom.or'):
            return self._parse_atomic(AtomFunc.OR, operands)
        if mnemonic.startswith('atom.xor'):
            return self._parse_atomic(AtomFunc.XOR, operands)
        if mnemonic.startswith('atom.exch'):
            return self._parse_atomic(AtomFunc.EXCH, operands)
        if mnemonic.startswith('atom.cas'):
            return self._parse_atomic_cas(operands)

        #================================================================
        # Reduction Operations
        #================================================================
        if mnemonic.startswith('red.add'):
            return self._parse_reduction(AtomFunc.ADD, operands)
        if mnemonic.startswith('red.min'):
            return self._parse_reduction(AtomFunc.MIN_U, operands)
        if mnemonic.startswith('red.max'):
            return self._parse_reduction(AtomFunc.MAX_U, operands)
        if mnemonic.startswith('red.and'):
            return self._parse_reduction(AtomFunc.AND, operands)
        if mnemonic.startswith('red.or'):
            return self._parse_reduction(AtomFunc.OR, operands)

        #================================================================
        # Warp Shuffle
        #================================================================
        if mnemonic.startswith('shfl.sync.idx'):
            return self._parse_shfl(ShflFunc.IDX, operands)
        if mnemonic.startswith('shfl.sync.up'):
            return self._parse_shfl(ShflFunc.UP, operands)
        if mnemonic.startswith('shfl.sync.down'):
            return self._parse_shfl(ShflFunc.DOWN, operands)
        if mnemonic.startswith('shfl.sync.bfly'):
            return self._parse_shfl(ShflFunc.BFLY, operands)

        #================================================================
        # Warp Vote
        #================================================================
        if mnemonic.startswith('vote.sync.all'):
            return self._parse_vote(VoteFunc.ALL, operands)
        if mnemonic.startswith('vote.sync.any'):
            return self._parse_vote(VoteFunc.ANY, operands)
        if mnemonic.startswith('vote.sync.uni'):
            return self._parse_vote(VoteFunc.UNI, operands)
        if mnemonic.startswith('vote.sync.ballot'):
            return self._parse_vote(VoteFunc.BALLOT, operands)

        #================================================================
        # Warp Reduction
        #================================================================
        if mnemonic.startswith('redux.sync.add'):
            return self._parse_redux(ReduxFunc.ADD, operands)
        if mnemonic.startswith('redux.sync.min'):
            return self._parse_redux(ReduxFunc.MIN, operands)
        if mnemonic.startswith('redux.sync.max'):
            return self._parse_redux(ReduxFunc.MAX, operands)
        if mnemonic.startswith('redux.sync.and'):
            return self._parse_redux(ReduxFunc.AND, operands)
        if mnemonic.startswith('redux.sync.or'):
            return self._parse_redux(ReduxFunc.OR, operands)

        #================================================================
        # WMMA (Tensor Core)
        #================================================================
        if mnemonic.startswith('wmma.load.a'):
            return self._parse_wmma_load(mnemonic, operands, 0)
        if mnemonic.startswith('wmma.load.b'):
            return self._parse_wmma_load(mnemonic, operands, 1)
        if mnemonic.startswith('wmma.load.c'):
            return self._parse_wmma_load(mnemonic, operands, 2)
        if mnemonic.startswith('wmma.store.d'):
            return self._parse_wmma_store(mnemonic, operands)
        if mnemonic.startswith('wmma.mma'):
            return self._parse_wmma_mma(mnemonic, operands)
        if mnemonic.startswith('mma.sync'):
            return self._parse_mma_sync(mnemonic, operands)

        #================================================================
        # WGMMA (Hopper Tensor Core)
        #================================================================
        if mnemonic.startswith('wgmma.mma_async'):
            return self._parse_wgmma_mma(mnemonic, operands)
        if mnemonic.startswith('wgmma.fence'):
            return self._parse_wgmma_control(WgmmaFunc.FENCE)
        if mnemonic.startswith('wgmma.commit_group'):
            return self._parse_wgmma_control(WgmmaFunc.COMMIT)
        if mnemonic.startswith('wgmma.wait_group'):
            return self._parse_wgmma_control(WgmmaFunc.WAIT)

        #================================================================
        # Texture Operations
        #================================================================
        if mnemonic.startswith('tex.1d'):
            return self._parse_tex(TexFunc.TEX_1D, operands)
        if mnemonic.startswith('tex.2d'):
            return self._parse_tex(TexFunc.TEX_2D, operands)
        if mnemonic.startswith('tex.3d'):
            return self._parse_tex(TexFunc.TEX_3D, operands)
        if mnemonic.startswith('tex.cube'):
            return self._parse_tex(TexFunc.TEX_CUBE, operands)
        if mnemonic.startswith('tex.level'):
            return self._parse_tex(TexFunc.TEX_LEVEL, operands)
        if mnemonic.startswith('txq.width'):
            return self._parse_txq(TexFunc.TXQ_WIDTH, operands)
        if mnemonic.startswith('txq.height'):
            return self._parse_txq(TexFunc.TXQ_HEIGHT, operands)
        if mnemonic.startswith('txq.depth'):
            return self._parse_txq(TexFunc.TXQ_DEPTH, operands)
        if mnemonic.startswith('txq.num_mipmap_levels'):
            return self._parse_txq(TexFunc.TXQ_LEVELS, operands)

        #================================================================
        # Surface Operations
        #================================================================
        if mnemonic.startswith('suld.b.1d'):
            return self._parse_suld(SurfFunc.SURF_1D, operands)
        if mnemonic.startswith('suld.b.2d'):
            return self._parse_suld(SurfFunc.SURF_2D, operands)
        if mnemonic.startswith('suld.b.3d'):
            return self._parse_suld(SurfFunc.SURF_3D, operands)
        if mnemonic.startswith('sust.b.1d'):
            return self._parse_sust(SurfFunc.SURF_1D, operands)
        if mnemonic.startswith('sust.b.2d'):
            return self._parse_sust(SurfFunc.SURF_2D, operands)
        if mnemonic.startswith('sust.b.3d'):
            return self._parse_sust(SurfFunc.SURF_3D, operands)
        if mnemonic.startswith('sured'):
            return self._parse_sured(operands)

        #================================================================
        # Video/SIMD Operations
        #================================================================
        if mnemonic.startswith('vadd'):
            return self._parse_video(VideoFunc.VADD, operands)
        if mnemonic.startswith('vsub'):
            return self._parse_video(VideoFunc.VSUB, operands)
        if mnemonic.startswith('vabsdiff'):
            return self._parse_video(VideoFunc.VABSDIFF, operands)
        if mnemonic.startswith('vmin'):
            return self._parse_video(VideoFunc.VMIN, operands)
        if mnemonic.startswith('vmax'):
            return self._parse_video(VideoFunc.VMAX, operands)
        if mnemonic.startswith('vshl'):
            return self._parse_video(VideoFunc.VSHL, operands)
        if mnemonic.startswith('vshr'):
            return self._parse_video(VideoFunc.VSHR, operands)
        if mnemonic.startswith('vmad'):
            return self._parse_video_mad(operands)
        if mnemonic.startswith('vadd4'):
            return self._parse_video(VideoFunc.VADD4, operands)
        if mnemonic.startswith('vsub4'):
            return self._parse_video(VideoFunc.VSUB4, operands)
        if mnemonic.startswith('vabsdiff4'):
            return self._parse_video(VideoFunc.VABSDIFF4, operands)
        if mnemonic.startswith('vadd2'):
            return self._parse_video(VideoFunc.VADD2, operands)
        if mnemonic.startswith('vsub2'):
            return self._parse_video(VideoFunc.VSUB2, operands)
        if mnemonic.startswith('vmul2'):
            return self._parse_video(VideoFunc.VMUL2, operands)
        if mnemonic.startswith('dp4a'):
            return self._parse_dp4a(operands)
        if mnemonic.startswith('dp2a'):
            return self._parse_dp2a(operands)

        #================================================================
        # Async Copy Operations
        #================================================================
        if mnemonic.startswith('cp.async.ca'):
            return self._parse_cpasync(CpAsyncFunc.CA, operands)
        if mnemonic.startswith('cp.async.cg'):
            return self._parse_cpasync(CpAsyncFunc.CG, operands)
        if mnemonic.startswith('cp.async.commit_group'):
            return self._parse_cpasync_control(CpAsyncFunc.COMMIT)
        if mnemonic.startswith('cp.async.wait_group'):
            return self._parse_cpasync_control(CpAsyncFunc.WAIT)
        if mnemonic.startswith('cp.async.wait_all'):
            return self._parse_cpasync_control(CpAsyncFunc.WAIT_ALL)
        if mnemonic.startswith('cp.async.bulk'):
            return self._parse_cpasync(CpAsyncFunc.BULK, operands)

        #================================================================
        # Prefetch Operations
        #================================================================
        if mnemonic.startswith('prefetch.l1') or mnemonic.startswith('prefetch.L1'):
            return self._parse_prefetch(PrefetchFunc.L1, operands)
        if mnemonic.startswith('prefetch.l2') or mnemonic.startswith('prefetch.L2'):
            return self._parse_prefetch(PrefetchFunc.L2, operands)
        if mnemonic.startswith('prefetchu.l1') or mnemonic.startswith('prefetchu.L1'):
            return self._parse_prefetch(PrefetchFunc.L1U, operands)

        raise ValueError(f"Unknown instruction: {mnemonic}")

    #========================================================================
    # Instruction Parsers
    #========================================================================

    def _parse_alu_binary(self, func: int, operands: List[str]) -> Instruction:
        """Parse binary ALU instruction: op rd, ra, rb"""
        inst = Instruction(opcode=Opcode.ALU, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        try:
            inst.rb = parse_register(operands[2])
        except ValueError:
            # Immediate value - use ALU_IMM opcode
            # Format: [31:26]=opcode, [25:21]=rd, [20:16]=ra, [15:10]=func, [9:0]=imm10
            imm = parse_immediate(operands[2])
            if imm < 0:
                # Handle negative immediates with sign extension
                imm = imm & 0x3FF  # 10-bit wrap
            inst.opcode = Opcode.ALU_IMM
            inst.imm16 = ((func & 0x3F) << 10) | (imm & 0x3FF)
        return inst

    def _parse_alu_unary(self, func: int, operands: List[str]) -> Instruction:
        """Parse unary ALU instruction: op rd, ra"""
        inst = Instruction(opcode=Opcode.ALU, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_alu_ternary(self, func: int, operands: List[str]) -> Instruction:
        """Parse ternary ALU instruction: op rd, ra, rb, rc"""
        inst = Instruction(opcode=Opcode.ALU, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        if len(operands) > 3:
            try:
                inst.rc = parse_register(operands[3])
            except ValueError:
                inst.rc = parse_immediate(operands[3]) & 0x1F
        return inst

    def _parse_mul(self, func: int, operands: List[str]) -> Instruction:
        """Parse multiplication: mul.lo rd, ra, rb"""
        inst = Instruction(opcode=Opcode.MUL, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_mad(self, func: int, operands: List[str]) -> Instruction:
        """Parse multiply-add: mad.lo rd, ra, rb, rc"""
        inst = Instruction(opcode=Opcode.MUL, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        if len(operands) > 3:
            inst.rc = parse_register(operands[3])
        return inst

    def _parse_mul_wide(self, operands: List[str]) -> Instruction:
        """Parse wide multiplication: mul.wide rd, ra, rb (32x32->64)"""
        inst = Instruction(opcode=Opcode.ALU, func=AluFunc.MUL_WIDE)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_div(self, mnemonic: str, operands: List[str], is_rem: bool) -> Instruction:
        """Parse division/remainder: div rd, ra, rb"""
        signed = '.s' in mnemonic
        if is_rem:
            func = DivFunc.REM_S if signed else DivFunc.REM_U
        else:
            func = DivFunc.DIV_S if signed else DivFunc.DIV_U

        inst = Instruction(opcode=Opcode.DIV, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_setp(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse comparison: setp.eq.s32 p, ra, rb"""
        parts = mnemonic.split('.')
        cmp_op = parts[1] if len(parts) > 1 else 'eq'

        cmp_funcs = {'eq': CmpFunc.EQ, 'ne': CmpFunc.NE, 'lt': CmpFunc.LT,
                     'le': CmpFunc.LE, 'gt': CmpFunc.GT, 'ge': CmpFunc.GE}

        inst = Instruction(opcode=Opcode.SETP, func=cmp_funcs.get(cmp_op, CmpFunc.EQ))
        pred = operands[0].lower()
        if pred.startswith('p'):
            inst.rd = int(pred[1:]) & 0x7
        inst.ra = parse_register(operands[1])
        try:
            inst.rb = parse_register(operands[2])
        except ValueError:
            inst.rb = parse_immediate(operands[2]) & 0x1F
        return inst

    def _parse_branch(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse branch: bra target, bra.z ra target, bra.nz ra target, bra.uni target

        Encoding: {opcode[31:26], type[25:24], unused[23:21], ra[20:16], offset[15:0]}
        Types: 00=unconditional, 01=if_zero, 10=if_not_zero, 11=uniform
        """
        inst = Instruction(opcode=Opcode.BRANCH)

        # Determine branch type from mnemonic
        branch_type = 0b00  # Default: unconditional
        if '.z' in mnemonic or '.eq' in mnemonic:
            branch_type = 0b01  # Branch if zero
        elif '.nz' in mnemonic or '.ne' in mnemonic:
            branch_type = 0b10  # Branch if not zero
        elif '.uni' in mnemonic:
            branch_type = 0b11  # Uniform branch

        # Parse operands based on branch type
        if branch_type in [0b01, 0b10]:  # Conditional branch
            # Format: bra.z ra, target OR bra.nz ra, target
            if len(operands) >= 2:
                cond_reg = parse_register(operands[0])
                target = operands[1]
            else:
                raise ValueError(f"Conditional branch requires register and target: {operands}")
        else:
            # Unconditional or uniform: bra target
            cond_reg = 0
            target = operands[0]

        # Calculate offset
        if target in self.labels:
            # Offset is in bytes from current instruction
            offset = self.labels[target] - self.current_addr
        else:
            offset = parse_immediate(target)

        # Encode: rd[4:3]=branch_type, ra=condition register, imm16=offset
        # In hardware: issue_rd[4:3] is branch_type, issue_ra is condition reg
        inst.rd = (branch_type << 3)  # Put branch_type in bits [4:3] of rd field
        inst.ra = cond_reg
        # Offset goes in imm16 (bits [15:0])
        inst.rb = (offset >> 11) & 0x1F
        inst.rc = (offset >> 6) & 0x1F
        inst.func = offset & 0x3F
        return inst

    def _parse_call(self, operands: List[str]) -> Instruction:
        """Parse function call: call target"""
        inst = Instruction(opcode=Opcode.CALL)
        target = operands[0]

        if target in self.labels:
            offset = (self.labels[target] - self.current_addr - 4) // 4
        else:
            offset = parse_immediate(target)

        inst.ra = (offset >> 16) & 0x1F
        inst.rb = (offset >> 11) & 0x1F
        inst.rc = (offset >> 6) & 0x1F
        inst.func = offset & 0x3F
        return inst

    def _parse_load(self, opcode: int, operands: List[str]) -> Instruction:
        """Parse load: ld.global rd, [ra+offset]"""
        inst = Instruction(opcode=opcode)
        inst.rd = parse_register(operands[0])

        addr_str = ' '.join(operands[1:]).strip('[]')
        if '+' in addr_str:
            parts = addr_str.split('+')
            inst.ra = parse_register(parts[0])
            offset = parse_immediate(parts[1])
            inst.rb = (offset >> 6) & 0x1F
            inst.func = offset & 0x3F
        else:
            inst.ra = parse_register(addr_str)
        return inst

    def _parse_store(self, opcode: int, operands: List[str]) -> Instruction:
        """Parse store: st.global [ra+offset], rs"""
        inst = Instruction(opcode=opcode)

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

    def _parse_load_cached(self, opcode: int, operands: List[str], hint: int) -> Instruction:
        """Parse cached load with hint"""
        inst = self._parse_load(opcode, operands)
        inst.rc = hint
        return inst

    def _parse_store_cached(self, opcode: int, operands: List[str], hint: int) -> Instruction:
        """Parse cached store with hint"""
        inst = self._parse_store(opcode, operands)
        inst.rc = (inst.rc & 0x18) | hint
        return inst

    def _parse_mov(self, operands: List[str]) -> Instruction:
        """Parse MOV instruction"""
        inst = Instruction(opcode=Opcode.MOV_SPECIAL)
        inst.rd = parse_register(operands[0])

        src = operands[1].lower()
        if src.startswith('%'):
            # Special register (tid.x, ctaid.x, etc.)
            inst.ra = parse_special_reg(src)
        else:
            # Immediate value - use MOV_IMM opcode
            # Format: [31:26]=opcode, [25:21]=rd, [15:0]=imm16
            imm = parse_immediate(src)
            if imm < 0 or imm > 0xFFFF:
                raise ValueError(f"MOV immediate {imm} out of 16-bit range")
            inst.opcode = Opcode.MOV_IMM
            inst.imm16 = imm
        return inst

    def _parse_bar_sync(self, operands: List[str]) -> Instruction:
        """Parse barrier: bar.sync n"""
        inst = Instruction(opcode=Opcode.BAR_SYNC)
        if operands:
            inst.ra = parse_immediate(operands[0]) & 0x1F
        return inst

    def _parse_membar(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse memory barrier: membar.cta/gl/sys"""
        inst = Instruction(opcode=Opcode.MEMBAR)
        if '.cta' in mnemonic:
            inst.func = MembarFunc.CTA
        elif '.gl' in mnemonic:
            inst.func = MembarFunc.GL
        elif '.sys' in mnemonic:
            inst.func = MembarFunc.SYS
        return inst

    def _parse_fp32(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP32 binary: add.f32 rd, ra, rb"""
        inst = Instruction(opcode=Opcode.FP32_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_fp32_unary(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP32 unary: neg.f32 rd, ra"""
        inst = Instruction(opcode=Opcode.FP32_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_fp32_fma(self, operands: List[str]) -> Instruction:
        """Parse FP32 FMA: fma.rn.f32 rd, ra, rb, rc"""
        inst = Instruction(opcode=Opcode.FP32_ARITH, func=Fp32Func.FMA)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_fp32_special(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP32 special: sin.f32 rd, ra"""
        inst = Instruction(opcode=Opcode.FP32_SPECIAL, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_fp64(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP64 binary"""
        inst = Instruction(opcode=Opcode.FP64_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_fp64_unary(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP64 unary"""
        inst = Instruction(opcode=Opcode.FP64_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_fp64_fma(self, operands: List[str]) -> Instruction:
        """Parse FP64 FMA"""
        inst = Instruction(opcode=Opcode.FP64_ARITH, func=Fp64Func.FMA)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_fp16(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP16/BF16 binary"""
        inst = Instruction(opcode=Opcode.FP16_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_fp16_unary(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP16/BF16 unary"""
        inst = Instruction(opcode=Opcode.FP16_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_fp16_fma(self, func: int, operands: List[str]) -> Instruction:
        """Parse FP16/BF16 FMA"""
        inst = Instruction(opcode=Opcode.FP16_ARITH, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_cvt(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse type conversion: cvt.s32.f32 rd, ra"""
        inst = Instruction(opcode=Opcode.CVT)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])

        # Determine conversion type from mnemonic
        cvt_map = {
            'cvt.s32.f32': CvtFunc.S32_F32,
            'cvt.u32.f32': CvtFunc.U32_F32,
            'cvt.f32.s32': CvtFunc.F32_S32,
            'cvt.f32.u32': CvtFunc.F32_U32,
            'cvt.f32.f64': CvtFunc.F32_F64,
            'cvt.f64.f32': CvtFunc.F64_F32,
            'cvt.f32.f16': CvtFunc.F32_F16,
            'cvt.f16.f32': CvtFunc.F16_F32,
            'cvt.s64.f64': CvtFunc.S64_F64,
            'cvt.u64.f64': CvtFunc.U64_F64,
            'cvt.f64.s64': CvtFunc.F64_S64,
            'cvt.f64.u64': CvtFunc.F64_U64,
            'cvt.pack':    CvtFunc.PACK,
        }

        # Find matching conversion
        for pattern, func in cvt_map.items():
            if mnemonic.startswith(pattern):
                inst.func = func
                break
        return inst

    def _parse_atomic(self, func: int, operands: List[str]) -> Instruction:
        """Parse atomic operation: atom.add rd, [ra], rb"""
        inst = Instruction(opcode=Opcode.ATOM, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1].strip('[]'))
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_atomic_cas(self, operands: List[str]) -> Instruction:
        """Parse atomic CAS: atom.cas rd, [ra], rb, rc"""
        inst = Instruction(opcode=Opcode.ATOM, func=AtomFunc.CAS)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1].strip('[]'))
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_reduction(self, func: int, operands: List[str]) -> Instruction:
        """Parse reduction: red.add [ra], rb"""
        inst = Instruction(opcode=Opcode.RED, func=func)
        inst.ra = parse_register(operands[0].strip('[]'))
        inst.rb = parse_register(operands[1])
        return inst

    def _parse_shfl(self, func: int, operands: List[str]) -> Instruction:
        """Parse shuffle: shfl.sync.idx rd, ra, rb/imm, rc/imm"""
        inst = Instruction(opcode=Opcode.SHFL, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        # Third operand can be register or immediate (lane index)
        try:
            inst.rb = parse_register(operands[2])
        except ValueError:
            inst.rb = parse_immediate(operands[2]) & 0x1F
        # Fourth operand can be register or immediate (width/mask)
        if len(operands) > 3:
            try:
                inst.rc = parse_register(operands[3])
            except ValueError:
                inst.rc = parse_immediate(operands[3]) & 0x1F
        return inst

    def _parse_vote(self, func: int, operands: List[str]) -> Instruction:
        """Parse vote: vote.sync.all p, pred, mask"""
        inst = Instruction(opcode=Opcode.VOTE, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        if len(operands) > 2:
            inst.rb = parse_immediate(operands[2]) & 0x1F
        return inst

    def _parse_redux(self, func: int, operands: List[str]) -> Instruction:
        """Parse warp reduction: redux.sync.add rd, ra, mask"""
        inst = Instruction(opcode=Opcode.REDUX, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        if len(operands) > 2:
            inst.rb = parse_immediate(operands[2]) & 0x1F
        return inst

    def _parse_tensor_dtype(self, mnemonic: str) -> int:
        """Parse WMMA/MMA data type from mnemonic"""
        tokens = mnemonic.lower().split('.')
        has_fp4 = any(tok in ('f4',) for tok in tokens)
        has_fp8 = any(tok in ('f8',) for tok in tokens)
        has_fp4_e2m1 = 'e2m1' in tokens
        has_fp4_e3m0 = 'e3m0' in tokens
        has_fp8_e4m3 = 'e4m3' in tokens
        has_fp8_e5m2 = 'e5m2' in tokens
        has_int4 = any(tok in ('s4', 'u4', 'int4', 's4x8', 'u4x8') for tok in tokens)
        has_int8 = any(tok in ('s8', 'u8', 'int8', 's8x4', 'u8x4') for tok in tokens)
        has_bf16 = any(tok in ('bf16',) for tok in tokens)
        has_fp16 = any(tok in ('f16',) for tok in tokens)

        if has_fp4_e3m0:
            return TensorDataType.FP4_E3M0
        if has_fp4_e2m1 or has_fp4:
            return TensorDataType.FP4_E2M1
        if has_fp8_e5m2:
            return TensorDataType.FP8_E5M2
        if has_fp8_e4m3 or has_fp8:
            return TensorDataType.FP8_E4M3
        if has_int4:
            return TensorDataType.INT4
        if has_int8:
            return TensorDataType.INT8
        if has_bf16:
            return TensorDataType.BF16
        if has_fp16:
            return TensorDataType.FP16
        return TensorDataType.FP16

    def _parse_wmma_shape(self, mnemonic: str) -> int:
        """Parse WMMA/MMA shape from mnemonic"""
        shape_map = {
            'm16n16k16': WmmaFunc.M16N16K16,
            'm8n8k4': WmmaFunc.M8N8K4,
            'm32n8k16': WmmaFunc.M32N8K16,
            'm16n8k8': WmmaFunc.M16N16K16,
        }
        for shape, func in shape_map.items():
            if shape in mnemonic:
                return func
        return WmmaFunc.M16N16K16

    def _encode_wmma_func(self, shape: int, dtype: int) -> int:
        """Encode WMMA/MMA func field: shape[5:3] + dtype[2:0]"""
        return ((shape & 0x7) << 3) | (dtype & 0x7)

    def _parse_wmma_load(self, mnemonic: str, operands: List[str], matrix: int) -> Instruction:
        """Parse WMMA load"""
        inst = Instruction(opcode=Opcode.WMMA_LOAD)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1].strip('[]'))
        dtype = self._parse_tensor_dtype(mnemonic)
        inst.func = self._encode_wmma_func(matrix, dtype)  # matrix in [5:3], dtype in [2:0]
        return inst

    def _parse_wmma_store(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse WMMA store"""
        inst = Instruction(opcode=Opcode.WMMA_STORE)
        inst.ra = parse_register(operands[0].strip('[]'))
        inst.rb = parse_register(operands[1])
        dtype = self._parse_tensor_dtype(mnemonic)
        inst.func = self._encode_wmma_func(3, dtype)  # D matrix id = 3
        return inst

    def _parse_wmma_mma(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse WMMA MMA"""
        inst = Instruction(opcode=Opcode.WMMA_MMA)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        shape = self._parse_wmma_shape(mnemonic)
        dtype = self._parse_tensor_dtype(mnemonic)
        inst.func = self._encode_wmma_func(shape, dtype)
        return inst

    def _parse_mma_sync(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse MMA sync instruction"""
        inst = Instruction(opcode=Opcode.MMA)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])

        shape = self._parse_wmma_shape(mnemonic)
        dtype = self._parse_tensor_dtype(mnemonic)
        inst.func = self._encode_wmma_func(shape, dtype)
        return inst

    def _parse_wgmma_mma(self, mnemonic: str, operands: List[str]) -> Instruction:
        """Parse WGMMA async MMA"""
        inst = Instruction(opcode=Opcode.WGMMA_MMA)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])

        # Determine shape
        shape_map = {
            'm64n8k16': WgmmaFunc.M64N8K16,
            'm64n16k16': WgmmaFunc.M64N16K16,
            'm64n32k16': WgmmaFunc.M64N32K16,
            'm64n64k16': WgmmaFunc.M64N64K16,
            'm64n128k16': WgmmaFunc.M64N128K16,
            'm64n256k16': WgmmaFunc.M64N256K16,
        }
        for shape, func in shape_map.items():
            if shape in mnemonic:
                inst.func = func
                break
        return inst

    def _parse_wgmma_control(self, func: int) -> Instruction:
        """Parse WGMMA control instruction"""
        return Instruction(opcode=Opcode.WGMMA_MMA, func=func)

    def _parse_tex(self, func: int, operands: List[str]) -> Instruction:
        """Parse texture sample"""
        inst = Instruction(opcode=Opcode.TEX, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])  # Texture handle
        inst.rb = parse_register(operands[2])  # Coordinates
        return inst

    def _parse_txq(self, func: int, operands: List[str]) -> Instruction:
        """Parse texture query"""
        inst = Instruction(opcode=Opcode.TXQ, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        return inst

    def _parse_suld(self, func: int, operands: List[str]) -> Instruction:
        """Parse surface load"""
        inst = Instruction(opcode=Opcode.SULD, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_sust(self, func: int, operands: List[str]) -> Instruction:
        """Parse surface store"""
        inst = Instruction(opcode=Opcode.SUST, func=func)
        inst.ra = parse_register(operands[0])
        inst.rb = parse_register(operands[1])
        inst.rc = parse_register(operands[2])
        return inst

    def _parse_sured(self, operands: List[str]) -> Instruction:
        """Parse surface reduction"""
        inst = Instruction(opcode=Opcode.SURED)
        inst.ra = parse_register(operands[0])
        inst.rb = parse_register(operands[1])
        inst.rc = parse_register(operands[2])
        return inst

    def _parse_video(self, func: int, operands: List[str]) -> Instruction:
        """Parse video instruction"""
        inst = Instruction(opcode=Opcode.VIDEO, func=func)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        return inst

    def _parse_video_mad(self, operands: List[str]) -> Instruction:
        """Parse video MAD"""
        inst = Instruction(opcode=Opcode.VIDEO, func=VideoFunc.VMAD)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_dp4a(self, operands: List[str]) -> Instruction:
        """Parse DP4A instruction"""
        inst = Instruction(opcode=Opcode.VIDEO, func=VideoFunc.DP4A)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_dp2a(self, operands: List[str]) -> Instruction:
        """Parse DP2A instruction"""
        inst = Instruction(opcode=Opcode.VIDEO, func=VideoFunc.DP2A)
        inst.rd = parse_register(operands[0])
        inst.ra = parse_register(operands[1])
        inst.rb = parse_register(operands[2])
        inst.rc = parse_register(operands[3])
        return inst

    def _parse_cpasync(self, func: int, operands: List[str]) -> Instruction:
        """Parse async copy"""
        inst = Instruction(opcode=Opcode.CPASYNC, func=func)
        inst.ra = parse_register(operands[0].strip('[]'))  # Shared dest
        inst.rb = parse_register(operands[1].strip('[]'))  # Global src
        if len(operands) > 2:
            inst.rc = parse_immediate(operands[2]) & 0x1F  # Size
        return inst

    def _parse_cpasync_control(self, func: int) -> Instruction:
        """Parse async copy control"""
        return Instruction(opcode=Opcode.CPASYNC, func=func)

    def _parse_prefetch(self, func: int, operands: List[str]) -> Instruction:
        """Parse prefetch"""
        inst = Instruction(opcode=Opcode.PREFETCH, func=func)
        inst.ra = parse_register(operands[0].strip('[]'))
        return inst

#============================================================================
# Assembly Function
#============================================================================
def assemble_file(input_file: str, output_file: str) -> List[int]:
    """Assemble a PTX file to machine code"""
    with open(input_file, 'r') as f:
        lines = f.readlines()

    assembler = PTXAssembler()
    assembler.first_pass(lines)
    machine_code = assembler.second_pass(lines)

    with open(output_file, 'w') as f:
        for code in machine_code:
            f.write(f"{code:08x}\n")

    print(f"Assembled {len(machine_code)} instructions to {output_file}")
    return machine_code

#============================================================================
# Instruction Coverage Test
#============================================================================
def run_coverage_test() -> Tuple[int, int, float]:
    """Run comprehensive instruction coverage test"""

    # All PTX instructions to test (~285 unique instructions)
    test_instructions = [
        # Integer Arithmetic (~35)
        "add.s32 r0, r1, r2",
        "add.u32 r0, r1, r2",
        "sub.s32 r0, r1, r2",
        "sub.u32 r0, r1, r2",
        "mul.lo.s32 r0, r1, r2",
        "mul.lo.u32 r0, r1, r2",
        "mul.hi.s32 r0, r1, r2",
        "mul.hi.u32 r0, r1, r2",
        "mad.lo.s32 r0, r1, r2, r3",
        "mad.lo.u32 r0, r1, r2, r3",
        "div.s32 r0, r1, r2",
        "div.u32 r0, r1, r2",
        "rem.s32 r0, r1, r2",
        "rem.u32 r0, r1, r2",
        "abs.s32 r0, r1",
        "neg.s32 r0, r1",
        "min.s32 r0, r1, r2",
        "min.u32 r0, r1, r2",
        "max.s32 r0, r1, r2",
        "max.u32 r0, r1, r2",
        "popc.b32 r0, r1",
        "clz.b32 r0, r1",
        "bfind.s32 r0, r1",
        "brev.b32 r0, r1",
        "bfe.s32 r0, r1, r2, r3",
        "bfe.u32 r0, r1, r2, r3",
        "bfi.b32 r0, r1, r2, r3",
        "prmt.b32 r0, r1, r2, r3",
        "sad.s32 r0, r1, r2, r3",
        "add.cc.s32 r0, r1, r2",
        "addc.s32 r0, r1, r2",
        "sub.cc.s32 r0, r1, r2",
        "subc.s32 r0, r1, r2",
        "mul.wide.s32 r0, r1, r2",

        # Logic (~10)
        "and.b32 r0, r1, r2",
        "or.b32 r0, r1, r2",
        "xor.b32 r0, r1, r2",
        "not.b32 r0, r1",
        "shl.b32 r0, r1, r2",
        "shr.u32 r0, r1, r2",
        "shr.s32 r0, r1, r2",
        "selp.b32 r0, r1, r2, r3",
        "slct.f32.s32 r0, r1, r2, r3",

        # Comparison (~6)
        "setp.eq.s32 p0, r1, r2",
        "setp.ne.s32 p0, r1, r2",
        "setp.lt.s32 p0, r1, r2",
        "setp.le.s32 p0, r1, r2",
        "setp.gt.s32 p0, r1, r2",
        "setp.ge.s32 p0, r1, r2",

        # FP32 Arithmetic (~9)
        "add.f32 r0, r1, r2",
        "sub.f32 r0, r1, r2",
        "mul.f32 r0, r1, r2",
        "div.f32 r0, r1, r2",
        "fma.rn.f32 r0, r1, r2, r3",
        "neg.f32 r0, r1",
        "abs.f32 r0, r1",
        "min.f32 r0, r1, r2",
        "max.f32 r0, r1, r2",

        # FP32 Special (~8)
        "rcp.f32 r0, r1",
        "sqrt.f32 r0, r1",
        "rsqrt.f32 r0, r1",
        "sin.f32 r0, r1",
        "cos.f32 r0, r1",
        "lg2.f32 r0, r1",
        "ex2.f32 r0, r1",
        "tanh.f32 r0, r1",

        # FP64 (~12)
        "add.f64 r0, r2, r4",
        "sub.f64 r0, r2, r4",
        "mul.f64 r0, r2, r4",
        "div.f64 r0, r2, r4",
        "fma.rn.f64 r0, r2, r4, r6",
        "neg.f64 r0, r2",
        "abs.f64 r0, r2",
        "min.f64 r0, r2, r4",
        "max.f64 r0, r2, r4",
        "sqrt.f64 r0, r2",
        "rsqrt.f64 r0, r2",
        "rcp.f64 r0, r2",

        # FP16 (~18)
        "add.f16 r0, r1, r2",
        "sub.f16 r0, r1, r2",
        "mul.f16 r0, r1, r2",
        "fma.f16 r0, r1, r2, r3",
        "neg.f16 r0, r1",
        "abs.f16 r0, r1",
        "min.f16 r0, r1, r2",
        "max.f16 r0, r1, r2",
        "tanh.f16 r0, r1",
        "ex2.f16 r0, r1",
        "add.bf16 r0, r1, r2",
        "sub.bf16 r0, r1, r2",
        "mul.bf16 r0, r1, r2",
        "fma.bf16 r0, r1, r2, r3",
        "add.f16x2 r0, r1, r2",
        "sub.f16x2 r0, r1, r2",
        "mul.f16x2 r0, r1, r2",
        "fma.f16x2 r0, r1, r2, r3",

        # Type Conversion (~12)
        "cvt.s32.f32 r0, r1",
        "cvt.u32.f32 r0, r1",
        "cvt.f32.s32 r0, r1",
        "cvt.f32.u32 r0, r1",
        "cvt.f32.f64 r0, r2",
        "cvt.f64.f32 r0, r1",
        "cvt.f32.f16 r0, r1",
        "cvt.f16.f32 r0, r1",
        "cvt.s64.f64 r0, r2",
        "cvt.u64.f64 r0, r2",
        "cvt.f64.s64 r0, r2",
        "cvt.f64.u64 r0, r2",

        # Data Movement (~40)
        "ld.global.s32 r0, [r1]",
        "st.global.s32 [r0], r1",
        "ld.shared.s32 r0, [r1]",
        "st.shared.s32 [r0], r1",
        "ld.param.s32 r0, [r1]",
        "ld.const.s32 r0, [r1]",
        "ld.local.s32 r0, [r1]",
        "st.local.s32 [r0], r1",
        "ld.v2.s32 r0, [r1]",
        "ld.v4.s32 r0, [r1]",
        "st.v2.s32 [r0], r1",
        "st.v4.s32 [r0], r1",
        "mov.u32 r0, %tid.x",
        "mov.u32 r0, %tid.y",
        "mov.u32 r0, %tid.z",
        "mov.u32 r0, %ctaid.x",
        "mov.u32 r0, %ntid.x",
        "mov.u32 r0, %laneid",
        "mov.u32 r0, %warpid",
        "mov.u32 r0, %smid",
        "ld.ca.s32 r0, [r1]",
        "ld.cg.s32 r0, [r1]",
        "ld.cs.s32 r0, [r1]",
        "ld.lu.s32 r0, [r1]",
        "ld.cv.s32 r0, [r1]",
        "st.wb.s32 [r0], r1",
        "st.wt.s32 [r0], r1",
        "prefetch.L1 [r0]",
        "prefetch.L2 [r0]",
        "prefetchu.L1 [r0]",

        # Control Flow (~10)
        "bra 0",
        "bra.uni 0",
        "@p0 bra 0",
        "call 0",
        "ret",
        "exit",

        # Synchronization (~8)
        "bar.sync 0",
        "membar.cta",
        "membar.gl",
        "membar.sys",

        # Atomic (~12)
        "atom.add.s32 r0, [r1], r2",
        "atom.min.s32 r0, [r1], r2",
        "atom.min.u32 r0, [r1], r2",
        "atom.max.s32 r0, [r1], r2",
        "atom.max.u32 r0, [r1], r2",
        "atom.inc.u32 r0, [r1], r2",
        "atom.dec.u32 r0, [r1], r2",
        "atom.and.b32 r0, [r1], r2",
        "atom.or.b32 r0, [r1], r2",
        "atom.xor.b32 r0, [r1], r2",
        "atom.exch.b32 r0, [r1], r2",
        "atom.cas.b32 r0, [r1], r2, r3",

        # Reduction (~5)
        "red.add.s32 [r0], r1",
        "red.min.s32 [r0], r1",
        "red.max.s32 [r0], r1",
        "red.and.b32 [r0], r1",
        "red.or.b32 [r0], r1",

        # Warp Shuffle (~4)
        "shfl.sync.idx.b32 r0, r1, r2, r3",
        "shfl.sync.up.b32 r0, r1, r2, r3",
        "shfl.sync.down.b32 r0, r1, r2, r3",
        "shfl.sync.bfly.b32 r0, r1, r2, r3",

        # Warp Vote (~4)
        "vote.sync.all.pred p0, p1, 0xFFFFFFFF",
        "vote.sync.any.pred p0, p1, 0xFFFFFFFF",
        "vote.sync.uni.pred p0, p1, 0xFFFFFFFF",
        "vote.sync.ballot.b32 r0, p1, 0xFFFFFFFF",

        # Warp Redux (~5)
        "redux.sync.add.s32 r0, r1, 0xFFFFFFFF",
        "redux.sync.min.s32 r0, r1, 0xFFFFFFFF",
        "redux.sync.max.s32 r0, r1, 0xFFFFFFFF",
        "redux.sync.and.b32 r0, r1, 0xFFFFFFFF",
        "redux.sync.or.b32 r0, r1, 0xFFFFFFFF",

        # WMMA (~7)
        "wmma.load.a.sync.m16n16k16.f16 r0, [r1]",
        "wmma.load.b.sync.m16n16k16.f16 r0, [r1]",
        "wmma.load.c.sync.m16n16k16.f32 r0, [r1]",
        "wmma.store.d.sync.m16n16k16.f32 [r0], r1",
        "wmma.mma.sync.m16n16k16.f32.f16 r0, r1, r2, r3",
        "wmma.mma.sync.m16n16k16.f32.bf16.bf16.f32 r0, r1, r2, r3",
        "wmma.mma.sync.m16n16k16.f32.s8.s8.s32 r0, r1, r2, r3",
        "wmma.mma.sync.m16n16k16.f32.f4.f4.f32 r0, r1, r2, r3",
        "wmma.mma.sync.m16n16k16.f32.e5m2.e5m2.f32 r0, r1, r2, r3",
        "wmma.mma.sync.m16n16k16.f32.e3m0.e3m0.f32 r0, r1, r2, r3",
        "mma.sync.m8n8k4.f32.f16 r0, r1, r2, r3",
        "mma.sync.m16n8k8.f32.f16 r0, r1, r2, r3",

        # WGMMA (~9)
        "wgmma.mma_async.m64n8k16.f32 r0, r1, r2",
        "wgmma.mma_async.m64n16k16.f32 r0, r1, r2",
        "wgmma.mma_async.m64n32k16.f32 r0, r1, r2",
        "wgmma.mma_async.m64n64k16.f32 r0, r1, r2",
        "wgmma.mma_async.m64n128k16.f32 r0, r1, r2",
        "wgmma.mma_async.m64n256k16.f32 r0, r1, r2",
        "wgmma.fence",
        "wgmma.commit_group",
        "wgmma.wait_group",

        # Texture (~13)
        "tex.1d.v4.f32 r0, r1, r2",
        "tex.2d.v4.f32 r0, r1, r2",
        "tex.3d.v4.f32 r0, r1, r2",
        "tex.cube.v4.f32 r0, r1, r2",
        "tex.level.2d.v4.f32 r0, r1, r2",
        "txq.width.b32 r0, r1",
        "txq.height.b32 r0, r1",
        "txq.depth.b32 r0, r1",
        "txq.num_mipmap_levels.b32 r0, r1",
        "suld.b.1d.b32 r0, r1, r2",
        "suld.b.2d.b32 r0, r1, r2",
        "sust.b.1d.b32 r0, r1, r2",
        "sust.b.2d.b32 r0, r1, r2",
        "sured.b32 r0, r1, r2",

        # Video (~12)
        "vadd.s32.s32.s32 r0, r1, r2",
        "vsub.s32.s32.s32 r0, r1, r2",
        "vabsdiff.s32.s32.s32 r0, r1, r2",
        "vmin.s32.s32.s32 r0, r1, r2",
        "vmax.s32.s32.s32 r0, r1, r2",
        "vshl.u32 r0, r1, r2",
        "vshr.u32 r0, r1, r2",
        "vmad.s32.s32.s32 r0, r1, r2, r3",
        "vadd4.s8 r0, r1, r2",
        "vadd2.s16 r0, r1, r2",
        "dp4a.s32.s32 r0, r1, r2, r3",
        "dp2a.s32.s32 r0, r1, r2, r3",

        # Async Copy (~6)
        "cp.async.ca.shared.global [r0], [r1], 16",
        "cp.async.cg.shared.global [r0], [r1], 16",
        "cp.async.commit_group",
        "cp.async.wait_group 0",
        "cp.async.wait_all",
        "cp.async.bulk.shared.global [r0], [r1], 128",

        # NOP
        "nop",
    ]

    assembler = PTXAssembler()
    passed = 0
    failed = 0
    failed_instructions = []

    for inst_str in test_instructions:
        try:
            inst = assembler.parse_instruction(inst_str)
            code = inst.encode()
            passed += 1
        except Exception as e:
            failed += 1
            failed_instructions.append((inst_str, str(e)))

    total = passed + failed
    coverage = (passed / total * 100) if total > 0 else 0

    print("=" * 60)
    print("PTX Instruction Coverage Test Results")
    print("=" * 60)
    print(f"Total instructions tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Coverage: {coverage:.1f}%")
    print("=" * 60)

    if failed_instructions:
        print("\nFailed instructions:")
        for inst, err in failed_instructions[:10]:
            print(f"  {inst}")
            print(f"    Error: {err}")
        if len(failed_instructions) > 10:
            print(f"  ... and {len(failed_instructions) - 10} more")

    return passed, total, coverage

#============================================================================
# Main
#============================================================================
def main():
    if len(sys.argv) < 2:
        print("RalphGPU PTX Assembler - Complete PTX ISA 9.1 Support")
        print("")
        print("Usage:")
        print("  python ptx_assembler.py <input.ptx> [-o output.hex]")
        print("  python ptx_assembler.py --test  # Run coverage test")
        print("")
        print("Examples:")
        print("  python ptx_assembler.py vector_add.ptx -o vector_add.hex")
        print("  python ptx_assembler.py --test")
        return

    if sys.argv[1] == '--test':
        run_coverage_test()
        return

    input_file = sys.argv[1]
    output_file = 'output.hex'

    if len(sys.argv) > 3 and sys.argv[2] == '-o':
        output_file = sys.argv[3]

    assemble_file(input_file, output_file)

if __name__ == '__main__':
    main()
