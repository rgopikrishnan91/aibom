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
    find_links: bool = True,
) -> EnrichFn:
    """Return a recursive-walk enricher closure.

    Args:
        use_case: The processor use-case (``"complete"`` by default).
        mode: Processing mode (``"rag"`` by default).
        llm_provider: Optional LLM provider override (e.g. ``"openai"``,
            ``"anthropic"``); defaults to the processor's own default.
        model: Optional LLM model identifier override.
        find_links: When True (default), each child is run through the
            link-fallback finder to discover its GitHub repo and arXiv
            paper before the BOM is built — the same source discovery a
            top-level run performs. Without it a child is processed from
            its HuggingFace card alone, which under-extracts
            ``modelLineage`` / ``trainedOnDatasets`` / ``testedOnDatasets``
            (those fields are mined mostly from the README and the
            paper, not the card metadata). Degrades gracefully to
            HuggingFace-only when no ``GEMINI_API_KEY`` is configured.

    Returns:
        A callable ``(target_dict) -> metadata_dict | None``. Returns
        ``None`` when the target cannot be resolved to a HuggingFace
        identifier, or when the inner processor raises (network failure,
        missing repo, etc.) — in those cases the recursive walker greys
        the child as identified-but-not-generated.
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

        # Source discovery: give the child the same GitHub repo + arXiv
        # paper a top-level run would gather, instead of HuggingFace alone.
        arxiv_url, github_url = "", ""
        if find_links:
            arxiv_url, github_url = _discover_links(identifier, bom_type)

        try:
            if bom_type == "ai":
                from aikaboom.core.processors import AIBOMProcessor

                proc = _build_processor(
                    AIBOMProcessor, use_case, mode, llm_provider, model
                )
                return proc.process_ai_model(
                    repo_id=identifier,
                    arxiv_url=arxiv_url,
                    github_url=github_url,
                )

            from aikaboom.core.processors import DATABOMProcessor

            proc = _build_processor(
                DATABOMProcessor, use_case, mode, llm_provider, model
            )
            hf_url = f"https://huggingface.co/datasets/{identifier}"
            return proc.process_dataset(
                arxiv_url=arxiv_url, github_url=github_url, hf_url=hf_url
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


def _discover_links(identifier: str, bom_type: str) -> tuple:
    """Discover a child's GitHub repo + arXiv paper via the link-fallback
    finder, mirroring the source discovery a top-level run performs.

    A recursive child enriched from its HuggingFace card alone loses the
    GitHub README and the arXiv paper that the RAG pipeline mines for
    ``modelLineage`` / ``trainedOnDatasets`` / ``testedOnDatasets``, so a
    child's relationships come out under-extracted. Running the finder
    here puts a child on equal footing with a top-level run.

    Returns ``(arxiv_url, github_url)``; ``("", "")`` when discovery is
    unavailable (no ``GEMINI_API_KEY``, missing dependency, or any
    failure) so the walk degrades to HuggingFace-only rather than break.
    """
    # Discovery is silent by default, which makes it impossible to tell
    # whether a child was sourced fully or fell back to HuggingFace-only.
    # Every branch below prints one ``[recursive] link discovery …`` line
    # — visible on the CLI and mirrored into the web log/SSE stream — so
    # the outcome for each child is always observable.
    try:
        from aikaboom.utils.link_fallback import LinkFallbackFinder

        finder = LinkFallbackFinder()
        # No key / missing deps → the finder disables itself (client None).
        if getattr(finder, "client", None) is None:
            print(
                f"[recursive] link discovery DISABLED for {identifier} — "
                f"LinkFallbackFinder unavailable (missing GEMINI_API_KEY or "
                f"google-genai). Child built from the HuggingFace card only."
            )
            return "", ""
        if bom_type == "data":
            _hf, arxiv_url, github_url, _status = finder.find_missing_links(
                hf_repo_id=identifier, arxiv_url="", github_url="",
            )
        else:
            _hf, arxiv_url, github_url, _status = finder.find_missing_links(
                repo_id=identifier, hf_repo_id=identifier,
                arxiv_url="", github_url="",
            )
        arxiv_url, github_url = (arxiv_url or ""), (github_url or "")
        print(
            f"[recursive] link discovery for {identifier}: "
            f"github={'✓ ' + github_url if github_url else '✗ none'} | "
            f"arxiv={'✓ ' + arxiv_url if arxiv_url else '✗ none'}"
        )
        log.info(
            "recursive enrich: link discovery for %s → github=%s arxiv=%s",
            identifier, bool(github_url), bool(arxiv_url),
        )
        return arxiv_url, github_url
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        print(
            f"[recursive] link discovery FAILED for {identifier}: {exc} — "
            f"child built from the HuggingFace card only."
        )
        log.info(
            "recursive enrich: link discovery failed for %s: %s",
            identifier, exc,
        )
        return "", ""


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
