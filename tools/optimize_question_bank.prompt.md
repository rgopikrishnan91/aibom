# Optimize one question-bank field

You are optimising one entry in an AIBOM (AI software bill of materials)
question bank. The bank drives retrieval-augmented extraction over three
source types: HuggingFace model cards, GitHub READMEs, and arXiv papers.

Your output will be embedded by a sentence-transformer (`BAAI/bge-small-en-v1.5`,
512-token window) to retrieve relevant chunks (HyDE / Hypothetical Document
Embeddings, Gao et al. ACL 2023), used as a BM25 sparse query for exact-match
retrieval, and fed to an LLM to extract the field value from the retrieved
chunks.

---

## Source registers — write in this voice

### HuggingFace model cards
Markdown sections, concrete phrasing. Examples:
- "This model is a fine-tuned version of `bert-base-uncased` on the SQuAD v2 dataset."
- "Trained for 3 epochs with AdamW, learning_rate=5e-5, batch_size=32 on 8 V100 GPUs."
- "Released under the **Apache-2.0** license."
- "**Limitations:** Performance degrades on out-of-domain text. May reflect biases in the training corpus."
- "Energy use: ~120 kWh of training compute on AWS us-east-1."

### GitHub READMEs
Project-flavoured prose, install instructions, badges. Examples:
- "## Installation\n\n```pip install transformers```"
- "MIT licensed. See LICENSE file for details."
- "Built with PyTorch 2.0 and supports CUDA 11.8+."
- "**Caveats:** This is research code; not production-ready."

### arXiv papers
Academic register, passive voice, citations. Examples:
- "We pre-train the model on the C4 corpus following the recipe of Raffel et al. (2020)."
- "Evaluation is performed on the GLUE benchmark using the standard metrics."
- "Carbon emissions during training are estimated at 552 kg CO₂eq following Strubell et al. (2019)."
- "Our approach is limited by the size of the available training data and the assumption of i.i.d. examples."

---

## Output format

Return a single JSON object — no envelope, no markdown fences, no
commentary:

```json
{
  "retrieval": {
    "hypothetical_passage": "<one paragraph, ≤100 tokens, in the practitioner register above>",
    "bm25_terms": ["<term>", "<term>", "..."]
  },
  "extraction": {
    "instruction": "<direct imperative; what the LLM should extract>",
    "field_spec": "<the SPDX/internal contract: field name, legal values, format>",
    "output_guidance": "<edge-case decision rules: missing info, noAssertion threshold, multi-value handling>"
  }
}
```

### Field-by-field guardrails

- `retrieval.hypothetical_passage`: blends vocabulary from all three
  registers so the FAISS embedding lands close to chunks from any
  source. Reads like a paragraph from a real model card or paper, NOT
  like a spec definition. No phrases like "this field captures…",
  "the SPDX property…", "the value indicates…". Aim for 50–100 tokens.
- `retrieval.bm25_terms`: 10–20 exact strings you'd literally Ctrl-F in
  a model card / README / paper. Include synonyms, abbreviations,
  unit symbols, license SPDX IDs, framework names. Skip generic words
  like "information", "value", "describes". Each term is one token
  string (a single word OR a multi-word phrase as one string element).
- `extraction.instruction`: one imperative sentence. "Extract the X
  reported in the source. If multiple Xs, list them comma-separated."
- `extraction.field_spec`: the SPDX property name (or "AIkaBoOM-internal"
  for fields without one), the legal value space (enum members /
  free-form text / non-negative integer / etc.), and any format
  constraints (units, ISO-8601, comma-separated, etc.).
- `extraction.output_guidance`: rules for edge cases. When to answer
  `noAssertion`. How to handle conflicting source info. Whether to
  include reasoning vs just the bare value. What "Not found." means.

---

## Worked example — `ai/energyConsumption`

Input field metadata:
- `field`: `energyConsumption`
- `bom_type`: `ai`
- `aikaboom_internal`: `false`
- `spdx_property`: `ai_energyConsumption`
- `spec_url`: `https://spdx.github.io/spdx-spec/v3.0.1/model/AI/Properties/energyConsumption/`
- `summary`: "Indicates the amount of energy consumed when training the AIPackage."
- `description`: "Specifies the amount of energy consumed by an AI model. The energy consumed is divided into 3 stages: training, fine-tuning, and inference."

Expected output:

```json
{
  "retrieval": {
    "hypothetical_passage": "Training this model required approximately 120 kWh of energy on 8 NVIDIA A100 GPUs over 72 hours, equivalent to roughly 50 kg CO₂eq following Strubell et al. (2019). Fine-tuning on the downstream task adds another 5 kWh. Inference cost is around 0.001 kWh per 1k tokens. Carbon emissions were estimated using the ML CO2 Impact calculator and the AWS us-east-1 grid mix.",
    "bm25_terms": ["kWh", "energy consumption", "training compute", "GPU hours", "FLOPs", "carbon emissions", "CO2eq", "kg CO2", "Strubell", "ML CO2 Impact", "watt-hours", "joules", "training cost", "inference cost", "fine-tuning energy"]
  },
  "extraction": {
    "instruction": "Extract the energy consumed by the AI model across training, fine-tuning, and inference stages. Report each stage with its quantity and unit; if only total energy is reported, state that.",
    "field_spec": "SPDX property `ai_energyConsumption`. Composite of up to three EnergyConsumptionDescription records (training, fine-tuning, inference), each with `energyQuantity` (xsd:decimal) and `energyUnit` (kilowatt-hour | megajoule | other). Format: 'training: 120 kWh; fine-tuning: 5 kWh; inference: 0.001 kWh/1k tokens'.",
    "output_guidance": "If the source reports CO₂eq instead of energy, capture both. If only one stage is reported, return that stage and mark the others as not reported. If no energy figure appears, return 'Not found.' rather than inventing a value. Carbon-equivalent estimates require a method citation (e.g., Strubell et al., ML CO2 Impact); without one, note the unattributed estimate verbatim."
  }
}
```

---

## Field to optimise

You are now optimising this field. Read the metadata below, then produce
the JSON output above. Do not output anything else.

```
field:               {field}
bom_type:            {bom_type}
aikaboom_internal:   {aikaboom_internal}
spdx_property:       {spdx_property}
spec_url:            {spec_url}
existing_question:   {question}
existing_keywords:   {keywords}
summary:             {summary}
description:         {description}
```

Important:
- For AIkaBoOM-internal fields (`aikaboom_internal: true`), there is no
  SPDX page. Use the existing `description` as the source of truth and
  set `field_spec` to "AIkaBoOM-internal" with the relevant value space.
- The `existing_*` fields are reference material to inform your output;
  do not copy them verbatim. Improve.
- BM25 terms list MUST contain only strings (not objects). Each string
  may be a single word or a multi-word phrase.
- Output exactly one JSON object as specified above.
