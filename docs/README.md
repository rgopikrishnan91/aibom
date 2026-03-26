# BOM Tools Documentation

## Table of Contents

1. [Getting Started](./getting_started.md)
2. [Architecture](./architecture.md)
3. [API Reference](./api.md)
4. [User Guide](./user_guide.md)
5. [Configuration](./configuration.md)
6. [Development](./development.md)

## Quick Links

- [Installation Instructions](../README.md#installation)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/yourusername/BOM_Tools)

## Overview

BOM Tools is a comprehensive solution for generating Bill of Materials (BOM) for AI models and datasets. It supports:

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
