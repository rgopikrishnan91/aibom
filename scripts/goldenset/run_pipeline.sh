#!/usr/bin/env bash
# Goldenset AIBOM pipeline: runs OWASP aibom-generator and ALOHA on all 25
# models in Golden_Set/AIBOM_Golden-Set_main-version.csv, then maps each
# CycloneDX 1.6 output to SPDX 3.0.1 (AI Profile) using cdx_to_spdx.py.
#
# Usage:
#   bash scripts/goldenset/run_pipeline.sh
#
# Outputs land in goldenset_results/.
# Requires: python3 >= 3.10, pip, git, network access to huggingface.co.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${WORK_DIR:-/tmp/goldenset-tools}"
OUT_DIR="${REPO_ROOT}/goldenset_results"
CSV="${REPO_ROOT}/Golden_Set/AIBOM_Golden-Set_main-version.csv"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "${WORK_DIR}" \
         "${OUT_DIR}/aibom-generator/cyclonedx-1.6" \
         "${OUT_DIR}/aibom-generator/spdx" \
         "${OUT_DIR}/aloha/cyclonedx-1.6"
: > "${OUT_DIR}/aibom-generator/errors.log"
: > "${OUT_DIR}/aloha/errors.log"

# 1. Preflight: HF reachability
log "Preflight: checking huggingface.co reachability..."
hf_status=$(curl -s -o /dev/null -w '%{http_code}' \
  https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 || echo "000")
if [[ "${hf_status}" != "200" ]]; then
  echo "ERROR: huggingface.co returned HTTP ${hf_status}. Add huggingface.co + *.huggingface.co to your egress allowlist before running this script." >&2
  exit 1
fi
log "  HF reachable (HTTP 200)."

# 2. Clone tools
if [[ ! -d "${WORK_DIR}/aibom-generator" ]]; then
  log "Cloning aibom-generator..."
  git clone --depth 1 https://github.com/GenAI-Security-Project/aibom-generator.git \
    "${WORK_DIR}/aibom-generator"
fi
if [[ ! -d "${WORK_DIR}/ALOHA" ]]; then
  log "Cloning ALOHA..."
  git clone --depth 1 https://github.com/MSR4SBOM/ALOHA.git "${WORK_DIR}/ALOHA"
fi

# 3. Create venv and install deps
VENV_DIR="${WORK_DIR}/venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating Python virtual environment..."
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Installing Python dependencies..."
pip install --quiet \
  blinker beautifulsoup4 huggingface_hub jsonschema license-expression \
  packageurl-python pydantic 'PyYAML>=6' requests httpx jinja2 datasets flask \
  fastapi uvicorn starlette gunicorn python-multipart 2>&1 | tail -5
pip install --quiet -r "${WORK_DIR}/ALOHA/requirements.txt" 2>&1 | tail -3 || true

# 4. Read model IDs from CSV
mapfile -t MODELS < <(python3 -c '
import csv, sys
with open(sys.argv[1]) as f:
    r = csv.reader(f); next(r)
    for row in r:
        if row and row[0].strip(): print(row[0].strip())
' "${CSV}")
log "Found ${#MODELS[@]} models in goldenset."

# 5. Run both tools per model
ok_aibg=0; fail_aibg=0; ok_aloha=0; fail_aloha=0
for model in "${MODELS[@]}"; do
  safe="${model//\//_}"
  log "→ ${model}"

  # 5a. aibom-generator (writes _1_6.json + _1_7.json; we keep 1.6)
  out_base="${OUT_DIR}/aibom-generator/cyclonedx-1.6/${safe}_ai_sbom_1_6.json"
  ( cd "${WORK_DIR}/aibom-generator" && \
    timeout "${TIMEOUT_SEC}" python3 -m src.cli "${model}" --output "${out_base}" \
      >>"${OUT_DIR}/aibom-generator/errors.log" 2>&1 )
  if [[ -s "${out_base}" ]]; then
    ok_aibg=$((ok_aibg+1))
    # discard the 1.7 sibling the CLI also writes
    rm -f "${out_base%_1_6.json}_1_7.json" "${out_base%.json}.html" 2>/dev/null
  else
    fail_aibg=$((fail_aibg+1))
    echo "FAIL aibom-generator: ${model}" >> "${OUT_DIR}/aibom-generator/errors.log"
  fi

  # 5b. ALOHA (writes <safe>.json into output dir)
  ( cd "${WORK_DIR}/ALOHA" && \
    timeout "${TIMEOUT_SEC}" python3 ALOHA.py "${model}" -o "${OUT_DIR}/aloha/cyclonedx-1.6/" \
      >>"${OUT_DIR}/aloha/errors.log" 2>&1 )
  # ALOHA names output after the model — find latest json that mentions the model
  if compgen -G "${OUT_DIR}/aloha/cyclonedx-1.6/*${safe##*_}*.json" >/dev/null 2>&1 \
     || compgen -G "${OUT_DIR}/aloha/cyclonedx-1.6/${safe}*.json" >/dev/null 2>&1; then
    ok_aloha=$((ok_aloha+1))
  else
    fail_aloha=$((fail_aloha+1))
    echo "FAIL ALOHA: ${model}" >> "${OUT_DIR}/aloha/errors.log"
  fi
done

# 6. Map aibom-generator CycloneDX → SPDX
log "Mapping CycloneDX → SPDX..."
python3 "${SCRIPT_DIR}/cdx_to_spdx.py" \
  --in-dir "${OUT_DIR}/aibom-generator/cyclonedx-1.6" \
  --out-dir "${OUT_DIR}/aibom-generator/spdx"

# 7. Summary
log "Done."
echo
echo "Results in: ${OUT_DIR}"
echo "  aibom-generator: ${ok_aibg} ok, ${fail_aibg} failed"
echo "  ALOHA:           ${ok_aloha} ok, ${fail_aloha} failed"
echo "  SPDX-mapped:     $(ls "${OUT_DIR}/aibom-generator/spdx" 2>/dev/null | wc -l) files"
