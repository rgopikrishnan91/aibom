"""Version + size + variant extraction from a HuggingFace model name.

Combines parsing logic from two SAIL Research scripts:

  - ``codes/extracting_sizes_from_names.py`` — split-on-``-``/``_`` size
    token detection (``7B``, ``350M``, ``0.5B``) — the size component.
  - ``codes/variants_collection.py`` — keyword classification of variant
    tag (``quantized``, ``compressed``, ``deduplicated``,
    ``instruction-tuned``) — the variant component.

Adds an explicit ``vX.Y[.Z]`` regex (the version-marker component)
which the upstream scripts don't isolate.

Citation:
    Ajibode, A. et al. (2024). "On the Naming and Versioning Practices
    of Pre-Trained Language Models on HuggingFace."
    https://github.com/SAILResearch/wip-24-adekunle-lm-release

Security note (2026-05-08):
    The upstream repo has a hardcoded HuggingFace token in
    ``codes/version_manual.py:14`` and ``codes/variants_collection.py:16``.
    Reported to maintainers; never copy those lines. AIkaBoOM uses the
    parsing logic only, never the leaked token.

License:
    SAIL upstream has no LICENSE file at time of lift.
    Reuse confirmed verbally by the maintainer (per AIkaBoOM Phase 12
    project notes); when SAIL adds a LICENSE, mirror it here.
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Variant keyword sets (from variants_collection.py)
# ---------------------------------------------------------------------------
_QUANTIZED_TOKENS = {
    "4bits", "4bit", "8bits", "8bit", "awq", "gptq", "gguf",
    "float32", "float16", "int8", "int4", "ptq", "q8", "q4",
    "quantized", "qat",
}
_COMPRESSED_TOKENS = {"distilled", "distil"}
_DEDUPED_TOKENS = {"deduped"}
_INSTRUCT_TOKENS = {"instruct", "chat", "sft", "dpo", "rlhf", "it"}

# Explicit version markers like vX, vX.Y, vX.Y.Z (AIkaBoOM addition)
_EXPLICIT_VERSION_RE = re.compile(
    r'(?<![A-Za-z0-9])v(\d+(?:\.\d+){0,3})(?![A-Za-z0-9])',
    re.IGNORECASE,
)


def _last_segment(repo_id: str) -> str:
    """Return the model-name segment (after the org slash)."""
    if not repo_id:
        return ""
    return repo_id.split('/')[-1]


def extract_size_component(repo_id: str) -> Optional[str]:
    """Extract a size token (``7B``, ``350M``, ``0.5B``) from the model name.

    Adapted from SAIL ``extracting_sizes_from_names.py``: split on
    ``-``/``_``, find tokens whose suffix is ``b``/``B``/``m``/``M`` and
    whose prefix is a number. Returns canonicalised form (uppercase
    unit). Returns ``None`` if no size token found.

    Examples
    --------
    >>> extract_size_component("mistralai/Mistral-7B-v0.1")
    '7B'
    >>> extract_size_component("Qwen/Qwen2.5-7B-Instruct")
    '7B'
    >>> extract_size_component("google-bert/bert-base-uncased") is None
    True
    """
    name = _last_segment(repo_id)
    if not name:
        return None
    parts = re.split(r'[-_]', name)
    for part in parts:
        m = re.fullmatch(r'(\d+(?:\.\d+)?)([bBmM])', part)
        if m:
            return f"{m.group(1)}{m.group(2).upper()}"
    return None


def extract_variant_tag(repo_id: str) -> Optional[str]:
    """Classify the variant by keyword overlap with the model name.

    Priority order (first match wins): quantized > compressed >
    deduplicated > instruction-tuned. Returns ``None`` if no signal.

    Examples
    --------
    >>> extract_variant_tag("TheBloke/Llama-2-7B-Chat-GPTQ")
    'quantized'
    >>> extract_variant_tag("Qwen/Qwen2.5-7B-Instruct")
    'instruction-tuned'
    >>> extract_variant_tag("distilbert/distilbert-base-uncased")
    'compressed'
    >>> extract_variant_tag("EleutherAI/pythia-1B-deduped")
    'deduplicated'
    >>> extract_variant_tag("google-bert/bert-base-uncased") is None
    True
    """
    name = _last_segment(repo_id).lower()
    if not name:
        return None
    parts = set(re.split(r'[-_]', name))
    if parts & _QUANTIZED_TOKENS:
        return "quantized"
    # Check for distil substring (distilbert, distilgpt2 etc. don't split cleanly)
    if any(t in name for t in _COMPRESSED_TOKENS):
        return "compressed"
    if parts & _DEDUPED_TOKENS:
        return "deduplicated"
    if parts & _INSTRUCT_TOKENS:
        return "instruction-tuned"
    return None


def extract_explicit_version(repo_id: str) -> Optional[str]:
    """Find an explicit ``vX``/``vX.Y``/``vX.Y.Z`` marker.

    AIkaBoOM addition (not present in the SAIL scripts).

    Examples
    --------
    >>> extract_explicit_version("mistralai/Mistral-7B-v0.1")
    'v0.1'
    >>> extract_explicit_version("openai/clip-vit-base-patch32") is None
    True
    """
    name = _last_segment(repo_id)
    if not name:
        return None
    m = _EXPLICIT_VERSION_RE.search(name)
    if m:
        return f"v{m.group(1)}"
    return None


def extract_version_from_repo_id(repo_id: str) -> Optional[str]:
    """Combine size + version + variant components into one string.

    Designed for the BOM ``packageVersion`` field. Produces
    human-readable identifiers like ``7B-v0.1`` from
    ``mistralai/Mistral-7B-v0.1`` — far more useful than a git SHA.

    Returns ``None`` if no component could be extracted (e.g.
    ``google-bert/bert-base-uncased``); caller should fall back to
    HF git SHA / GitHub tag in that case.

    Examples
    --------
    >>> extract_version_from_repo_id("mistralai/Mistral-7B-v0.1")
    '7B-v0.1'
    >>> extract_version_from_repo_id("Qwen/Qwen2.5-7B-Instruct")
    '7B-instruction-tuned'
    >>> extract_version_from_repo_id("TheBloke/Llama-2-7B-Chat-GPTQ")
    '7B-quantized'
    >>> extract_version_from_repo_id("google-bert/bert-base-uncased") is None
    True
    """
    parts = []
    sz = extract_size_component(repo_id)
    if sz:
        parts.append(sz)
    ver = extract_explicit_version(repo_id)
    if ver:
        parts.append(ver)
    var = extract_variant_tag(repo_id)
    if var:
        parts.append(var)
    return "-".join(parts) if parts else None


__all__ = [
    "extract_size_component",
    "extract_variant_tag",
    "extract_explicit_version",
    "extract_version_from_repo_id",
]
