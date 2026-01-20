# RalphGPU Toolchain

## PTX Assembly Flow

```
PTX Source (.ptx)
       │
       ▼
┌─────────────────────┐
│ tools/ptx_assembler.py │  (ptxas-stub)
│ - Parses PTX syntax    │
│ - Encodes instructions │
│ - Outputs hex binary   │
└─────────────────────┘
       │
       ▼
HEX Binary (.hex)
       │
       ▼
┌─────────────────────┐
│ Testbench ($readmemh)  │
│ - Loads into imem      │
│ - Drives DUT           │
└─────────────────────┘
       │
       ▼
RTL Simulation (iverilog/vvp)
```

## PTX Assembler (ptxas-stub)

**Location**: `tools/ptx_assembler.py`

### Usage
```bash
# Assemble PTX to HEX
python tools/ptx_assembler.py input.ptx -o output.hex

# Run instruction coverage test
python tools/ptx_assembler.py --test
```

### Instruction Format (32-bit)
```
[31:26] opcode (6 bits)
[25:21] rd (5 bits)
[20:16] ra (5 bits)
[15:11] rb (5 bits)
[10:6]  rc (5 bits)
[5:0]   func (6 bits)
```

### Extended Format (for immediates)
```
[31:26] opcode (6 bits)
[25:21] rd (5 bits)
[20:16] ra (5 bits)
[15:0]  immediate (16 bits)
```

## Verification Framework

**Location**: `tools/verification_framework.py`

### Components
- Test case generation
- RTL simulation driver
- Output comparison

## GPU Simulator

**Location**: `tools/gpu_simulator.py`

### Purpose
- Software reference model for functional verification
- Executes PTX semantically

## HEX Format

- ASCII hex dump, one 32-bit word per line
- Little-endian word order
- Example:
```
00000000
12345678
DEADBEEF
```

## Compilation Command

```bash
# Compile RTL for simulation
iverilog -g2012 -Irtl -s <testbench> tb/<testbench>.v rtl/*.v

# Run simulation
vvp a.out
```

## Test Organization

```
asm/           # PTX source files
*.hex          # Assembled binaries (in project root)
tb/tb_*.v      # Testbenches
```
