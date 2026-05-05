# AIkaBoOM Documentation

## Contents

- [Field Resolution Strategies](./FIELD_STRATEGIES.md) — per-field reference for source priority, normalisation, and SPDX/CycloneDX export shape.
- [SPDX 3.0.1 Field Reference](./SPDX_3.0.1_FIELD_REFERENCE.md) — verbatim Summary + Description blocks harvested from the [`spdx/spdx-3-model`](https://github.com/spdx/spdx-3-model) repo. Canonical text for the question-bank `description` slots.
- [HuggingFace Spaces Deployment](./HF_SPACES.md)
- [Local Embeddings Guide](./LOCAL_EMBEDDINGS.md)
- [Migration Guide](./migration.md)

## Quick Links

- [Installation Instructions](../README.md#installation)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/rgopikrishnan91/aikaboom)

## Overview

AIkaBoOM is a comprehensive solution for generating Bills of Materials for AI models and datasets. It supports:

- **Dual Processing Modes**: RAG (Retrieval-Augmented Generation) and Direct LLM
- **Multiple Sources**: GitHub, HuggingFace, arXiv
- **SPDX 3.0.1**: Validated JSON-LD export (lightweight JSON Schema by default; optional SHACL strict pass)
- **CycloneDX 1.7 (beta)**: ML-BOM export with the `modelCard` extension
- **Recursive BOMs (beta)**: Walks the dependency tree of an AI BOM and emits per-child BOMs plus a single linked SPDX bundle
- **Conflict Resolution**: Inter-source and intra-source conflicts are surfaced as triplets (`{value, source, conflict}`) on every field

## Architecture Overview

```
┌─────────────────┐
│   Web UI        │
│  (Flask App)    │
└────────┬────────┘
         │
    ┌────▼─────┐
    │Processors│
    └────┬─────┘
         │
    ┌────▼────────┬──────────┐
    │             │          │
┌───▼───┐   ┌────▼────┐ ┌──▼──────┐
│  RAG  │   │ Direct  │ │Metadata │
│Engine │   │   LLM   │ │Fetcher  │
└───┬───┘   └────┬────┘ └──┬──────┘
    │            │         │
    └────────────┴─────────┘
              │
         ┌────▼─────┐
         │SPDX Gen  │
         └──────────┘
```

For detailed documentation, see individual files in this directory.
