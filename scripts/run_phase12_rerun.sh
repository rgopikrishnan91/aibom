#!/usr/bin/env bash
# Run the 10-BOM golden-set rerun with the Phase 12 pipeline.
# Mirrors the baseline provider/model so the diff is apples-to-apples.

set -u
cd "$(dirname "$0")/.."

OUT=test_outputs/phase12_rerun
mkdir -p "$OUT/ai" "$OUT/data"

# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env; set +a

PROVIDER=openrouter
MODEL=openai/gpt-4o-mini
COMMON_ARGS=(--provider "$PROVIDER" --model "$MODEL" --mode rag --use-case complete --no-validate-spdx -y)

run_ai() {
  local short="$1" repo="$2" arxiv="$3" github="$4"
  local out_json="$OUT/ai/${short}.json"
  local out_log="$OUT/ai/${short}.log"
  if [[ -s "$out_json" ]]; then
    echo "[skip] $short — output already exists at $out_json"
    return 0
  fi
  echo "[run]  AI $short ($repo)"
  python -m aikaboom.cli generate --type ai \
    --repo "$repo" --arxiv "$arxiv" --github "$github" \
    "${COMMON_ARGS[@]}" -o "$out_json" > "$out_log" 2>&1
  echo "[done] AI $short → $(stat -c %s "$out_json" 2>/dev/null || echo 0) bytes"
}

run_data() {
  local short="$1" repo="$2" arxiv="$3" github="$4"
  local out_json="$OUT/data/${short}.json"
  local out_log="$OUT/data/${short}.log"
  if [[ -s "$out_json" ]]; then
    echo "[skip] $short — output already exists at $out_json"
    return 0
  fi
  echo "[run]  DATA $short ($repo)"
  local hf_url="https://huggingface.co/datasets/$repo"
  python -m aikaboom.cli generate --type data \
    --hf-url "$hf_url" --arxiv "$arxiv" --github "$github" \
    "${COMMON_ARGS[@]}" -o "$out_json" > "$out_log" 2>&1
  echo "[done] DATA $short → $(stat -c %s "$out_json" 2>/dev/null || echo 0) bytes"
}

# ---------------------------------------------------------------------------
# 5 AI BOMs
# ---------------------------------------------------------------------------
run_ai bert       "google-bert/bert-base-uncased"     "https://arxiv.org/abs/1810.04805" "https://github.com/google-research/bert/blob/master/README.md"
run_ai distilbert "distilbert/distilbert-base-uncased" "https://arxiv.org/abs/1910.01108" "https://github.com/huggingface/transformers-research-projects/tree/main/distillation"
run_ai qwen       "Qwen/Qwen2.5-7B-Instruct"          "https://arxiv.org/abs/2407.10671" "https://github.com/QwenLM/Qwen3"
run_ai clip       "openai/clip-vit-base-patch32"      "https://arxiv.org/abs/2103.00020" "https://github.com/openai/CLIP"
run_ai roberta    "FacebookAI/roberta-large"          "https://arxiv.org/pdf/1907.11692" "https://github.com/facebookresearch/fairseq/tree/main/examples/roberta"

# ---------------------------------------------------------------------------
# 5 Dataset BOMs
# ---------------------------------------------------------------------------
run_data mbpp           "google-research-datasets/mbpp" "https://arxiv.org/pdf/2108.07732" "https://github.com/google-research/google-research/tree/master/mbpp"
run_data code_contests  "deepmind/code_contests"        "https://arxiv.org/pdf/2203.07814" "https://github.com/google-deepmind/code_contests"
run_data c4             "allenai/c4"                    "https://arxiv.org/pdf/1910.10683" "https://github.com/google-research/text-to-text-transfer-transformer#datasets"
run_data gsm8k          "openai/gsm8k"                  "https://arxiv.org/pdf/2110.14168" "https://github.com/openai/grade-school-math"
run_data mmlu           "cais/mmlu"                     "https://arxiv.org/pdf/2009.03300" "https://github.com/hendrycks/test"

echo "[all done] Phase 12 rerun complete; outputs in $OUT/"
