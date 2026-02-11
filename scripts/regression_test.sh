#!/bin/bash
# RalphGPU Regression Test Suite
# Usage: ./scripts/regression_test.sh [--quick|--full|--perf]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine test mode
MODE="${1:---quick}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="test_results/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}=== RalphGPU Regression Test Suite ===${NC}"
echo "Mode: ${MODE}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Core unit tests (always run)
CORE_TESTS=(
    "test_alu"
    "test_mul"
    "test_decoder"
    "test_regfile"
    "test_smem"
    "test_warp"
)

# Integration tests
INTEGRATION_TESTS=(
    "test_tensor_core_fp4"
    "test_vector_add"
    "test_multi_sm"
)

# Performance tests
PERF_TESTS=(
    "test_sm_v2_perf"
    "test_sm_v2_perf_gemm16_ptx"
    "test_sm_v2_perf_tensor"
    "test_sm_v2_perf_tensor_multiwarp"
)

# Run a single test
run_test() {
    local test_name=$1
    local log_file="${RESULTS_DIR}/${test_name}.log"
    
    echo -ne "Running ${test_name}... "
    
    if make ${test_name} > "${log_file}" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        echo "${test_name}: PASS" >> "${RESULTS_DIR}/summary.txt"
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        echo "${test_name}: FAIL" >> "${RESULTS_DIR}/summary.txt"
        echo "  Log: ${log_file}"
        return 1
    fi
}

# Extract metrics from performance test log
extract_metrics() {
    local test_name=$1
    local log_file="${RESULTS_DIR}/${test_name}.log"
    
    if [ -f "${log_file}" ]; then
        grep -E "Cycles:|Writebacks:|Issues:|IPC:|Stalls:|PASS|FAIL" "${log_file}" > "${RESULTS_DIR}/${test_name}_metrics.txt" 2>/dev/null || true
    fi
}

# Main test execution
FAILED_TESTS=()

echo -e "${YELLOW}=== Core Unit Tests ===${NC}"
for test in "${CORE_TESTS[@]}"; do
    if ! run_test "${test}"; then
        FAILED_TESTS+=("${test}")
    fi
done

if [ "${MODE}" == "--full" ] || [ "${MODE}" == "--integration" ]; then
    echo ""
    echo -e "${YELLOW}=== Integration Tests ===${NC}"
    for test in "${INTEGRATION_TESTS[@]}"; do
        if ! run_test "${test}"; then
            FAILED_TESTS+=("${test}")
        fi
    done
fi

if [ "${MODE}" == "--full" ] || [ "${MODE}" == "--perf" ]; then
    echo ""
    echo -e "${YELLOW}=== Performance Tests ===${NC}"
    for test in "${PERF_TESTS[@]}"; do
        if ! run_test "${test}"; then
            FAILED_TESTS+=("${test}")
        fi
        extract_metrics "${test}"
    done
fi

# Summary
echo ""
echo -e "${BLUE}=== Test Summary ===${NC}"
cat "${RESULTS_DIR}/summary.txt"

echo ""
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ ${#FAILED_TESTS[@]} test(s) failed:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - ${test}"
    done
    exit 1
fi
