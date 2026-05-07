"""Phase 9 — fixes for the 10 medium/low findings remaining after Phase 8.

  #8  Dataset SPDX strict-mode 4 errors  → discovery-driven; covered by #16/#17.
  #9  sbom-utility doesn't validate CycloneDX 1.7 → emit 1.6.
  #11 Web /config defaults to OpenAI even when only OpenRouter set.
  #12 pick_free_openrouter_model picks audio/music models.
  #13 LangChainPendingDeprecationWarning on every CLI invocation.
  #16 SPDX emits stub dataset_DatasetPackage named "noAssertion".
  #17 modelLineage arrow strings emitted as a single fake node, wrong package class.
  #18 LLM hallucinates benchmark numbers (documented; not patched).
  #19 Self-referential modelLineage for base models.
  #20 Conflict shape differs between direct/RAG fields (documented).
"""
from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# #9 — CycloneDX 1.6
# ---------------------------------------------------------------------------


def test_cyclonedx_exporter_emits_1_6():
    """The exporter emits ``specVersion: "1.6"`` so sbom-utility can
    validate it; previously it emitted 1.7 which sbom-utility v0.18.x
    can't read."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter, CDX_SPEC_VERSION

    bom = {
        "repo_id": "test/model",
        "model_id": "test_model",
        "direct_fields": {},
        "rag_fields": {},
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    assert CDX_SPEC_VERSION == "1.6"
    assert cdx["specVersion"] == "1.6"
    assert cdx["bomFormat"] == "CycloneDX"


def test_cyclonedx_internal_validator_expects_1_6():
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    exporter = CycloneDXExporter(bom_type="ai")
    bad_bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",  # wrong; we expect 1.6 now
        "serialNumber": "urn:uuid:fake",
        "components": [{"type": "machine-learning-model", "name": "x"}],
    }
    is_valid, errors = exporter.validate_cyclonedx(bad_bom)
    assert is_valid is False
    assert any("1.6" in e for e in errors)


# ---------------------------------------------------------------------------
# #11 — provider auto-detect
# ---------------------------------------------------------------------------


def _clear_provider_env(monkeypatch):
    for v in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(v, raising=False)


def test_detect_default_provider_picks_openrouter_when_set(monkeypatch):
    from aikaboom.utils.provider_resolver import detect_default_provider

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider, model = detect_default_provider()
    assert provider == "openrouter"
    assert model and ":free" in model


def test_detect_default_provider_returns_none_when_nothing_set(monkeypatch):
    from aikaboom.utils.provider_resolver import detect_default_provider

    _clear_provider_env(monkeypatch)
    provider, model = detect_default_provider()
    assert provider is None
    assert model is None


def test_detect_default_provider_prefers_openrouter_over_openai(monkeypatch):
    """OpenAI alone → openai. Both set → openrouter wins (free tier
    available; matches CLI's auto-detection priority)."""
    from aikaboom.utils.provider_resolver import detect_default_provider

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    provider, _ = detect_default_provider()
    assert provider == "openrouter"


# ---------------------------------------------------------------------------
# #12 — free-model picker filters non-chat
# ---------------------------------------------------------------------------


def test_is_text_chat_model_uses_modality_when_available():
    from aikaboom.utils.openrouter_models import _is_text_chat_model

    assert _is_text_chat_model({
        "id": "meta-llama/llama-3.3-70b",
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        },
    })
    assert not _is_text_chat_model({
        "id": "google/lyria-3-pro-preview",
        "architecture": {
            "input_modalities": ["text"],
            "output_modalities": ["audio"],
        },
    })


def test_is_text_chat_model_falls_back_to_name_blocklist():
    """Curated fallback entries don't carry architecture data; the picker
    must still filter obvious non-chat names."""
    from aikaboom.utils.openrouter_models import _is_text_chat_model

    assert _is_text_chat_model({
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "architecture": {},
    })
    assert not _is_text_chat_model({
        "id": "google/lyria-3-pro-preview",
        "architecture": {},
    })
    assert not _is_text_chat_model({
        "id": "openrouter/owl-alpha:free",
        "architecture": {},
    })


def test_list_free_openrouter_models_drops_non_chat():
    """End-to-end: with the API mocked to return a music model alongside
    a text model, the free list contains only the text one."""
    from aikaboom.utils import openrouter_models

    fake_models = [
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Llama 3.3 free",
            "context_length": 128_000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"input_modalities": ["text"],
                             "output_modalities": ["text"]},
        },
        {
            "id": "google/lyria-3-pro-preview:free",
            "name": "Lyria",
            "context_length": 1_000_000,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"input_modalities": ["text"],
                             "output_modalities": ["audio"]},
        },
    ]
    with patch.object(openrouter_models, "list_openrouter_models",
                      return_value=fake_models):
        free = openrouter_models.list_free_openrouter_models()
    assert [m["id"] for m in free] == ["meta-llama/llama-3.3-70b-instruct:free"]


# ---------------------------------------------------------------------------
# #13 — deprecation warning suppressed at CLI module top
# ---------------------------------------------------------------------------


def test_cli_module_installs_warning_filter():
    """The CLI module must install a ``DeprecationWarning`` filter scoped
    to ``langgraph.cache.base`` BEFORE any aikaboom imports so the noisy
    warning doesn't print on every invocation.

    Static-source check (rather than inspecting ``warnings.filters`` at
    runtime) because pytest resets the filter list between tests, so a
    runtime check is order-dependent. Verifies the filter call is wired
    in cli.py at module top.
    """
    import inspect
    import aikaboom.cli

    src = inspect.getsource(aikaboom.cli)
    assert "filterwarnings" in src
    assert "langgraph" in src
    # The filter must run BEFORE the dotenv/argparse imports so it
    # catches the langgraph import chain.
    filter_idx = src.find("filterwarnings")
    argparse_idx = src.find("import argparse")
    assert 0 < filter_idx < argparse_idx, (
        "warnings.filterwarnings must run before argparse / aikaboom imports"
    )


# ---------------------------------------------------------------------------
# #16 — noAssertion sentinel skipped
# ---------------------------------------------------------------------------


def test_is_nil_value_recognises_common_sentinels():
    from aikaboom.utils.spdx_validator import _is_nil_value

    assert _is_nil_value("noAssertion")
    assert _is_nil_value("noassertion")
    assert _is_nil_value("Not found.")
    assert _is_nil_value("")
    assert _is_nil_value(None)
    assert _is_nil_value("N/A")
    assert not _is_nil_value("Apache-2.0")
    assert not _is_nil_value("ImageNet")


def test_build_dataset_relationships_skips_nil_value():
    """A field whose value is ``noAssertion`` must not produce a stub
    package or relationship — earlier behaviour fabricated a
    ``dataset_DatasetPackage`` with name ``"noAssertion"`` (Finding #16)."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    stubs, rels = v._build_dataset_relationships(
        value="noAssertion",
        rel_type="trainedOn",
        from_id="urn:spdx:AIPackage-fake",
        creation_uuid="ci-fake",
    )
    assert stubs == []
    assert rels == []


# ---------------------------------------------------------------------------
# #17 — arrow split + per-rel_type package class
# ---------------------------------------------------------------------------


def test_arrow_separated_lineage_is_split_into_targets():
    """``A -> B -> C`` becomes three ``ai_AIPackage`` stubs, not one fake
    node named with the literal arrow string."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    stubs, rels = v._build_dataset_relationships(
        value="LLaMA -> LLaMA-2 -> LLaMA-3",
        rel_type="dependsOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
    )
    names = sorted(s["name"] for s in stubs)
    assert names == ["LLaMA", "LLaMA-2", "LLaMA-3"]
    # Every stub must be ai_AIPackage (modelLineage points at AI ancestors,
    # not datasets) — this was wrong before Phase 9 (Finding #17b).
    assert all(s["type"] == "ai_AIPackage" for s in stubs)
    assert len(rels) == 3
    assert all(r["relationshipType"] == "dependsOn" for r in rels)


def test_dataset_relationship_keeps_dataset_package_type():
    """trainedOn / testedOn must still emit dataset_DatasetPackage stubs."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    stubs, _ = v._build_dataset_relationships(
        value="ImageNet, COCO",
        rel_type="trainedOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
    )
    assert sorted(s["name"] for s in stubs) == ["COCO", "ImageNet"]
    assert all(s["type"] == "dataset_DatasetPackage" for s in stubs)


# ---------------------------------------------------------------------------
# #19 — self-loop reject
# ---------------------------------------------------------------------------


def test_self_referential_modellineage_dropped():
    """A base model whose modelLineage points at itself produces no
    stubs/edges. Earlier behaviour emitted a ``dependsOn`` from the
    parent to a fake child with the parent's own name (Finding #19)."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    stubs, rels = v._build_dataset_relationships(
        value="mistralai/Mistral-7B-v0.1 -> mistralai/Mistral-7B-v0.1",
        rel_type="dependsOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
        parent_identifier="mistralai/Mistral-7B-v0.1",
    )
    assert stubs == []
    assert rels == []


def test_self_loop_dropped_among_genuine_targets():
    """When the lineage list mixes the parent's own name with real
    ancestors, only the self-loop is dropped."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    stubs, rels = v._build_dataset_relationships(
        value="parent/model -> ancestor/A -> ancestor/B",
        rel_type="dependsOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
        parent_identifier="parent/model",
    )
    assert sorted(s["name"] for s in stubs) == ["ancestor/A", "ancestor/B"]
    assert len(rels) == 2


# ---------------------------------------------------------------------------
# #18, #20 — documented (no code path); spot-check README mentions both
# ---------------------------------------------------------------------------


def test_readme_documents_both_conflict_shapes_and_known_limits():
    """Spot-check that the README has the new sections so the
    documentation is part of the verifiable surface (Findings #18, #20)."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "Known limitations" in readme
    assert "may be hallucinated" in readme
    assert "two shapes" in readme
    assert "internal_conflicts" in readme
