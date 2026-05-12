# Citations

AIkaBoOM reuses parsing logic and methodological insight from the works below.
Each citation lists the upstream URL and the file(s) in this repository that
adapt the work. The adapted modules carry the same citation in their docstring
header.

## SAIL Research

Two SAIL Research scripts contribute parsing logic to AIkaBoOM's Phase 12
conflict-detection rebuild. Reuse confirmed permissive by the upstream
maintainers (2026-05-08); the upstream repos do not yet ship a `LICENSE`
file, so this note stands in until they do.

### Synchronization patterns between HuggingFace and GitHub (2025)

Ajibode, A., et al. (2025). *Synchronization Patterns between HuggingFace
and GitHub for Pre-Trained Language Models.*
<https://github.com/SAILResearch/replication-25-synchronization-Patterns>

Adapts: `Codes/GH_link_extraction_from_model_card.py` — GitHub-URL
filtering by model-name segment overlap.

Used in: `src/aikaboom/utils/sail_link_extractor.py`. The standalone
harvest used to populate the alias index now lives in
`scripts/harvest_supplier_roots.py` (simple HF-org listing + clustering;
no model-card fetching).

### Naming and versioning practices of pre-trained language models (2024)

Ajibode, A., et al. (2024). *On the Naming and Versioning Practices of
Pre-Trained Language Models on HuggingFace.*
<https://github.com/SAILResearch/wip-24-adekunle-lm-release>

Adapts: `codes/extracting_sizes_from_names.py` (size token detection) +
`codes/variants_collection.py` (variant keyword classification).

Used in: `src/aikaboom/utils/sail_version_extractor.py`, which feeds the
`packageVersion` cascade in `src/aikaboom/core/processors.py`.

**Security note.** The upstream repository contains a hardcoded
HuggingFace API token (`codes/version_manual.py:14` and
`codes/variants_collection.py:16`). It has been reported to the
maintainers for rotation. AIkaBoOM never copies those lines — only the
parsing logic — and pulls its own token from the user's `.env`.
