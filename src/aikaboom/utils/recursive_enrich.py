"""Recursive-walk enricher: invokes the BOM-generation tool per child.

`recursive_bom.generate_recursive_boms` accepts an ``enrich_fn`` callback
that, given a discovered target dict, returns the full metadata for that
target. Without one, it falls back to seed-only metadata. This module
provides the production callback that delegates to
:class:`aikaboom.core.processors.AIBOMProcessor` and
:class:`aikaboom.core.processors.DATABOMProcessor`.

The closure is stateless: each child gets a fresh processor instance
with its own retrievers and FAISS index. Slow but correct — there is no
shared state across siblings, and a transient failure in one child does
not corrupt another.

Lives in a separate module from ``recursive_bom`` so that the recursive
walker stays free of ``processors`` imports (avoids circulars).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


EnrichFn = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


def build_enrich_fn(
    use_case: str = "complete",
    mode: str = "rag",
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
) -> EnrichFn:
    """Return a recursive-walk enricher closure.

    Args:
        use_case: The processor use-case (``"complete"`` by default).
        mode: Processing mode (``"rag"`` by default).
        llm_provider: Optional LLM provider override (e.g. ``"openai"``,
            ``"anthropic"``); defaults to the processor's own default.
        model: Optional LLM model identifier override.

    Returns:
        A callable ``(target_dict) -> metadata_dict | None``. Returns
        ``None`` when the target cannot be resolved to a HuggingFace
        identifier, or when the inner processor raises (network failure,
        missing repo, etc.) — in those cases the recursive walker falls
        back to seed-only metadata for that child.
    """

    def enrich(target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (target.get("target") or "").strip()
        bom_type = target.get("bom_type")
        if not name or bom_type not in ("ai", "data"):
            return None

        identifier = _resolve_identifier(
            name, bom_type, target.get("resolvable_hint", False)
        )
        if identifier is None:
            log.info(
                "recursive enrich: cannot resolve %r as %s", name, bom_type
            )
            return None

        try:
            if bom_type == "ai":
                from aikaboom.core.processors import AIBOMProcessor

                proc = _build_processor(
                    AIBOMProcessor, use_case, mode, llm_provider, model
                )
                return proc.process_ai_model(
                    repo_id=identifier, arxiv_url="", github_url=""
                )

            from aikaboom.core.processors import DATABOMProcessor

            proc = _build_processor(
                DATABOMProcessor, use_case, mode, llm_provider, model
            )
            hf_url = f"https://huggingface.co/datasets/{identifier}"
            return proc.process_dataset(
                arxiv_url="", github_url="", hf_url=hf_url
            )

        except Exception as exc:  # noqa: BLE001 - we deliberately trap all
            log.warning(
                "recursive enrich failed for %s/%s: %s",
                bom_type,
                identifier,
                exc,
            )
            return None

    return enrich


def _build_processor(cls, use_case, mode, llm_provider, model):
    """Construct an AIBOMProcessor / DATABOMProcessor.

    The processor constructors take ``model``, ``mode``, ``llm_provider``,
    ``use_case``. We forward only the kwargs that were supplied so the
    processor's own defaults handle anything left as ``None`` — keeps us
    forward-compatible with constructor changes.
    """
    kwargs: Dict[str, Any] = {"use_case": use_case, "mode": mode}
    if llm_provider is not None:
        kwargs["llm_provider"] = llm_provider
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)


def _resolve_identifier(
    name: str, bom_type: str, resolvable_hint: bool
) -> Optional[str]:
    """Map a free-text target to a HF identifier, or ``None`` if unresolvable.

    The resolver is conservative — better to skip an unresolvable target
    (the walker records the skip in its audit) than to invent identifiers
    that point at the wrong artifact.
    """
    name = name.strip()
    if not name:
        return None
    if resolvable_hint:
        # Already in 'org/name' form (a slash, no spaces) — use directly.
        return name

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.info(
            "recursive enrich: huggingface_hub not installed; cannot search "
            "for %r as %s",
            name,
            bom_type,
        )
        return None

    api = HfApi()
    try:
        if bom_type == "data":
            results = api.list_datasets(search=name, limit=1)
        else:
            results = api.list_models(search=name, limit=1)
        first = next(iter(results), None)
        return first.id if first is not None else None
    except Exception as exc:  # noqa: BLE001
        log.info(
            "recursive enrich: HF search failed for %r as %s: %s",
            name,
            bom_type,
            exc,
        )
        return None
