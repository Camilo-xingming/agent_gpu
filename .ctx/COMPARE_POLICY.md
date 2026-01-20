# RalphGPU Comparison Policy

## Integer Operations
- **Policy**: Bit-exact comparison
- **Rationale**: Integer arithmetic must produce identical results across implementations

## Floating Point (FP32/FP64)
- **Policy**: Bit-exact by default
- **NaN Handling**: Compare via `isnan()` only; payload bits are ignored
- **Special Cases**:
  - +0.0 and -0.0 are considered equal for comparison purposes
  - Infinity values must match sign exactly

## Floating Point (FP16)
- **Policy**: Bit-exact
- **Rationale**: Lower precision still requires exact reproduction of IEEE 754 behavior

## Tensor Core Operations
- **Policy**: Tolerance-based comparison
- **Thresholds**:
  - FP16 accumulator: relative tolerance 1e-3
  - FP32 accumulator: relative tolerance 1e-5
  - INT8 accumulator: bit-exact
- **Rationale**: Tensor operations may have implementation-defined rounding

## Approximation Functions (SFU)
- **Policy**: Tolerance-based
- **Thresholds**:
  - sin/cos: absolute tolerance 1e-4 (within valid input range)
  - ex2/lg2: relative tolerance 1e-3
  - rcp/rsqrt: relative tolerance 1e-4
- **Rationale**: PTX ISA defines these as "approximations"

## Memory Operations
- **Policy**: Bit-exact for data
- **Address Alignment**: Must match PTX specification
- **Timing**: Not compared (functional correctness only)

## Predicate Results
- **Policy**: Bit-exact (0 or 1)
- **Rationale**: Predicates control execution flow

## Comparison Function (Python)

```python
def compare_results(expected, actual, op_type):
    if op_type == 'int':
        return expected == actual
    elif op_type == 'fp32':
        if is_nan(expected) and is_nan(actual):
            return True  # NaN == NaN for test purposes
        return expected == actual
    elif op_type == 'tensor_fp16':
        return abs(expected - actual) / max(abs(expected), 1e-10) < 1e-3
    elif op_type == 'sfu':
        return abs(expected - actual) < 1e-4
```

## Failure Reporting
When comparison fails:
1. Report expected vs actual values
2. Report bit patterns in hex
3. Report operation type and operands
4. Log to test output for debugging
