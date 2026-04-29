# SPDX 3.0.1 Schema Files

This directory should contain bundled copies of the official SPDX 3.0.1 validation schemas.

## Required files

Download these manually if they are not present (the build environment may not
have access to spdx.org):

```bash
curl -o spdx-model.ttl https://spdx.org/rdf/3.0.1/spdx-model.ttl
curl -o spdx-json-schema.json https://spdx.org/schema/3.0.1/spdx-json-schema.json
```

## What they do

- **spdx-model.ttl** - SHACL shapes for semantic validation (correct object/property usage, relationships, cardinality)
- **spdx-json-schema.json** - JSON Schema for structural validation (field names, types, required properties)

Both are published by the SPDX project and are the canonical validation artifacts
referenced in the [SPDX 3 validation guide](https://github.com/spdx/spdx-3-model/blob/main/serialization/jsonld/validation.md).

If these files are missing at runtime, the validator falls back to AIkaBoOM's
built-in structural checks.
