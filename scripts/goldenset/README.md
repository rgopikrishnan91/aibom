# Goldenset AIBOM pipeline

Self-contained scripts to run two AIBOM tools against the 25 Hugging Face
models in `Golden_Set/AIBOM_Golden-Set_main-version.csv` and produce both
CycloneDX 1.6 and SPDX 3.0.1 (AI Profile) outputs.

Tools:
- [OWASP aibom-generator](https://github.com/GenAI-Security-Project/aibom-generator) — emits CycloneDX 1.6 + 1.7
- [ALOHA](https://github.com/MSR4SBOM/ALOHA) — emits CycloneDX 1.6

The CycloneDX → SPDX mapping follows
[`docs/aibom-field-mapping/README.md`](https://github.com/GenAI-Security-Project/aibom-generator/blob/main/docs/aibom-field-mapping/README.md)
in the aibom-generator repo. Datasets in `DataBOM_Golden-Set_main-version.csv`
are intentionally skipped — both tools are model-centric.

## Requirements
- Python ≥ 3.10
- `pip`, `git`, `curl`
- Outbound HTTPS to `huggingface.co` and `*.huggingface.co`

## Run
```bash
bash scripts/goldenset/run_pipeline.sh
```

Outputs land in `goldenset_results/`:
```
goldenset_results/
  aibom-generator/
    cyclonedx-1.6/   # one JSON per model
    spdx/            # SPDX 3.0.1 JSON-LD, mapped from the CDX above
    errors.log
  aloha/
    cyclonedx-1.6/   # one JSON per model
    errors.log
```

The script clones both tools into `/tmp/goldenset-tools/` (override with
`WORK_DIR=...`), installs the minimal Python deps (no torch — `--summarize`
is intentionally not used), and runs each tool with a 180 s timeout per
model (override with `TIMEOUT_SEC=...`).

## SPDX mapping notes

`cdx_to_spdx.py` only emits fields covered by the field-mapping table.
Missing source data is omitted, not synthesized. The output is one
JSON-LD document per model with:
- An `SpdxDocument` element
- A `software_Package` (license + identity)
- An `ai_AIPackage` (AI Profile fields: `primaryPurpose`, `domain`,
  `typeOfModel`, `hyperparameter`, `metric`, `energyConsumption`,
  `informationAboutTraining`, `modelDataPreprocessing`, `limitation`,
  `safetyRiskAssessment`, `useSensitivePersonalInformation`,
  `modelExplainability`)
- `Relationship`s for `hasDeclaredLicense`, `ancestorOf`, `trainedOn`,
  `testedOn`, `dependsOn`

You can also run the mapper standalone against any directory of
CycloneDX-1.6 JSON files:
```bash
python3 scripts/goldenset/cdx_to_spdx.py \
  --in-dir path/to/cdx-files \
  --out-dir path/to/spdx-out
```
