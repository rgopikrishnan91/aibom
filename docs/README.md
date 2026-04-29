# AIkaBoOM Documentation

## Contents

- [HuggingFace Spaces Deployment](./HF_SPACES.md)
- [Local Embeddings Guide](./LOCAL_EMBEDDINGS.md)
- [Migration Guide](./migration.md)

## Quick Links

- [Installation Instructions](../README.md#installation)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/rgopikrishnan91/aibom)

## Overview

AIkaBoOM is a comprehensive solution for generating Bills of Materials for AI models and datasets. It supports:

- **Dual Processing Modes**: RAG (Retrieval-Augmented Generation) and Direct LLM
- **Multiple Sources**: GitHub, HuggingFace, ArXiv
- **SPDX 3.0.1 Compliance**: Automatic conversion to SPDX format
- **Conflict Resolution**: Intelligent handling of conflicting metadata

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
