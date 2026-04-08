#!/bin/bash
################################################################################
# Run All Configurations
# 
# This script runs all current provider/mode combinations:
# - AIBOM configs: RAG+Ollama, RAG+OpenRouter, Direct+Ollama, Direct+OpenRouter
# - DataBOM configs: RAG+Ollama, RAG+OpenRouter, Direct+Ollama, Direct+OpenRouter
#
# Usage:
#   ./run_all_8_configs.sh [--limit N] [--force-restart]
#                           [--embedding-provider local]
#                           [--embedding-model <name-or-path>]
#                           [--openrouter-model <model-name>]
################################################################################

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}         ${BLUE}Running All Provider Configurations${NC}                    ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Wrapper-level defaults
EMBEDDING_PROVIDER="local"
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
OPENROUTER_MODEL="qwen/qwen3-coder:free"  # Using paid model (you have credits!)

# Parse wrapper-specific args, pass everything else through to test_tool.py
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding-provider)
            EMBEDDING_PROVIDER="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --openrouter-model)
            OPENROUTER_MODEL="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Track statistics
TOTAL=8
SUCCESS=0
FAILED=0
FAILED_CONFIGS=()

# Function to run a config
run_config() {
    local name=$1
    local mode=$2
    local provider=$3
    local model=$4
    local csv=$5
    shift 5  # Remove first 5 arguments, leaving only the extra flags
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}▶ Config $((SUCCESS + FAILED + 1))/$TOTAL: ${BLUE}$name${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    
    if python tests/test_tool.py \
        --mode "$mode" \
        --provider "$provider" \
        --model "$model" \
        --input-csv "$csv" \
        --embedding-provider "$EMBEDDING_PROVIDER" \
        --embedding-model "$EMBEDDING_MODEL" \
        "$@"; then
        SUCCESS=$((SUCCESS + 1))
        echo -e "${GREEN}✓ SUCCESS: $name${NC}"
    else
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$name")
        echo -e "${RED}✗ FAILED: $name${NC}"
    fi
}

echo -e "${YELLOW}Starting processing...${NC}"
echo ""



# Run all current 8 configs
run_config "AIBOM - RAG + Ollama" "rag" "ollama" "qwen2.5:14b-instruct" "Golden_Set/AIBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "AIBOM - RAG + OpenRouter" "rag" "openrouter" "$OPENROUTER_MODEL" "Golden_Set/AIBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "AIBOM - Direct + Ollama" "direct" "ollama" "qwen2.5:14b-instruct" "Golden_Set/AIBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "AIBOM - Direct + OpenRouter" "direct" "openrouter" "$OPENROUTER_MODEL" "Golden_Set/AIBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"

run_config "DataBOM - RAG + Ollama" "rag" "ollama" "qwen2.5:14b-instruct" "Golden_Set/DataBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "DataBOM - RAG + OpenRouter" "rag" "openrouter" "$OPENROUTER_MODEL" "Golden_Set/DataBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "DataBOM - Direct + Ollama" "direct" "ollama" "qwen2.5:14b-instruct" "Golden_Set/DataBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"
run_config "DataBOM - Direct + OpenRouter" "direct" "openrouter" "$OPENROUTER_MODEL" "Golden_Set/DataBOM_Golden-Set_main-version.csv" "${PASSTHROUGH_ARGS[@]}"



# Print summary
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                        ${BLUE}FINAL SUMMARY${NC}                             ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Total Configs:${NC}    $TOTAL"
echo -e "${GREEN}Successful:${NC}       $SUCCESS"
echo -e "${RED}Failed:${NC}           $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed Configurations:${NC}"
    for config in "${FAILED_CONFIGS[@]}"; do
        echo -e "  ${RED}✗${NC} $config"
    done
    echo ""
fi

echo -e "${CYAN}Results saved in:${NC} results-golden-set/"
echo -e "  ${YELLOW}JSON:${NC}  results-golden-set/json/"
echo -e "  ${YELLOW}CSV:${NC}   results-golden-set/csv/"
echo ""

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All 8 configurations completed successfully.${NC}"
    exit 0
fi
