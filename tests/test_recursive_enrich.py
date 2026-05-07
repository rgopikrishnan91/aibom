"""Tests for the recursive-walk enricher.

Covers:
  - Identifier resolution: slash form passes through; bare names go
    through HF search; failures return None.
  - Processor selection by ``bom_type``.
  - Exception safety: inner processor raise → enrich returns None.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from aikaboom.utils.recursive_enrich import (
    build_enrich_fn,
    _resolve_identifier,
)


# ---------------------------------------------------------------------------
# Identifier resolution
# ---------------------------------------------------------------------------


def test_slash_form_passes_through_without_search():
    """``org/name`` is already a HF identifier — never call HF search."""
    with patch("huggingface_hub.HfApi") as mock_api:
        ident = _resolve_identifier("microsoft/DialoGPT", "ai", resolvable_hint=True)
    assert ident == "microsoft/DialoGPT"
    mock_api.assert_not_called()


def test_bare_name_searches_hf_and_returns_id():
    fake_hit = SimpleNamespace(id="imagenet-1k/imagenet-1k")
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_datasets.return_value = iter([fake_hit])
        ident = _resolve_identifier("ImageNet", "data", resolvable_hint=False)
    assert ident == "imagenet-1k/imagenet-1k"


def test_unresolvable_bare_name_returns_none():
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_datasets.return_value = iter([])
        ident = _resolve_identifier("ZzzNotARealDataset", "data", resolvable_hint=False)
    assert ident is None


def test_hf_search_failure_returns_none():
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_models.side_effect = RuntimeError("network down")
        ident = _resolve_identifier("LLaMA", "ai", resolvable_hint=False)
    assert ident is None


def test_empty_name_returns_none():
    assert _resolve_identifier("", "ai", resolvable_hint=False) is None
    assert _resolve_identifier("   ", "ai", resolvable_hint=False) is None


# ---------------------------------------------------------------------------
# build_enrich_fn closure: dispatches to the right processor and is safe
# under failure
# ---------------------------------------------------------------------------


def test_enrich_invokes_ai_processor_for_ai_target():
    enriched = {"model_id": "test", "rag_fields": {"model_name": "test"}}

    class FakeAI:
        def __init__(self, **_kw):
            pass

        def process_ai_model(self, repo_id, arxiv_url, github_url):
            return enriched

    with patch("aikaboom.core.processors.AIBOMProcessor", FakeAI):
        enrich = build_enrich_fn(use_case="complete")
        out = enrich({
            "target": "meta-llama/Llama-2-7b",
            "bom_type": "ai",
            "resolvable_hint": True,
        })
    assert out == enriched


def test_enrich_invokes_data_processor_for_data_target():
    enriched = {"dataset_id": "imagenet"}

    class FakeData:
        def __init__(self, **_kw):
            pass

        def process_dataset(self, arxiv_url, github_url, hf_url):
            assert hf_url == "https://huggingface.co/datasets/imagenet-1k/imagenet-1k"
            return enriched

    with patch("aikaboom.core.processors.DATABOMProcessor", FakeData):
        enrich = build_enrich_fn(use_case="complete")
        out = enrich({
            "target": "imagenet-1k/imagenet-1k",
            "bom_type": "data",
            "resolvable_hint": True,
        })
    assert out == enriched


def test_enrich_returns_none_when_processor_raises():
    class ExplodingAI:
        def __init__(self, **_kw):
            pass

        def process_ai_model(self, **_kw):
            raise RuntimeError("HF API blew up")

    with patch("aikaboom.core.processors.AIBOMProcessor", ExplodingAI):
        enrich = build_enrich_fn(use_case="complete")
        out = enrich({
            "target": "meta-llama/Llama-2-7b",
            "bom_type": "ai",
            "resolvable_hint": True,
        })
    assert out is None


def test_enrich_returns_none_for_unknown_bom_type():
    enrich = build_enrich_fn(use_case="complete")
    out = enrich({"target": "whatever", "bom_type": "package", "resolvable_hint": True})
    assert out is None


def test_enrich_returns_none_when_target_unresolvable():
    """If the bare name can't be searched (HF unreachable), the closure
    must not call any processor — resolution failure short-circuits."""
    with patch("huggingface_hub.HfApi") as MockApi:
        MockApi.return_value.list_models.return_value = iter([])
        enrich = build_enrich_fn(use_case="complete")
        out = enrich({
            "target": "NonexistentModelBlahBlah",
            "bom_type": "ai",
            "resolvable_hint": False,
        })
    assert out is None
