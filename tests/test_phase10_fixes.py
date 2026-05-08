"""Phase 10 — fixes from the Mistral-7B-v0.1 re-test report.

  #13 langgraph PendingDeprecationWarning suppression actually works.
  #17 (follow-through) recursive walker arrow-splits modelLineage.
  #22 SPDX relationship cap is user-configurable; default unbounded.
  #23 CycloneDX 1.6 dataset entries don't carry the 1.7-only type field.
  #24 CycloneDX nil sentinels don't fabricate stub child packages.
  #12+#6 free-model surface deleted; importable / flag presence regressions.
"""
from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# #22 — configurable SPDX relationship cap
# ---------------------------------------------------------------------------


def test_spdx_relationship_cap_unbounded_by_default():
    """Default (no cap) emits every target — closes the silent ``[:10]``
    truncation that swallowed children 11+ in the user's re-test."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")  # default relationship_cap=None
    names = ", ".join(f"ds-{i}" for i in range(20))
    stubs, rels = v._build_dataset_relationships(
        value=names,
        rel_type="testedOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
    )
    assert len(stubs) == 20, f"expected 20, got {len(stubs)}"
    assert len(rels) == 20


def test_spdx_relationship_cap_applies_when_set():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai", relationship_cap=5)
    names = ", ".join(f"ds-{i}" for i in range(20))
    stubs, _ = v._build_dataset_relationships(
        value=names,
        rel_type="testedOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
    )
    assert len(stubs) == 5


def test_spdx_relationship_cap_negative_treated_as_unbounded():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai", relationship_cap=-1)
    names = ", ".join(f"ds-{i}" for i in range(15))
    stubs, _ = v._build_dataset_relationships(
        value=names,
        rel_type="testedOn",
        from_id="urn:spdx:AIPackage-parent",
        creation_uuid="ci-fake",
    )
    assert len(stubs) == 15


def test_validate_bom_to_spdx_passes_cap_through():
    """The convenience wrapper threads ``relationship_cap`` into the
    constructor."""
    from aikaboom.utils.spdx_validator import validate_bom_to_spdx

    bom = {
        "repo_id": "test/m",
        "model_id": "test_m",
        "direct_fields": {},
        "rag_fields": {
            "testedOnDatasets": {
                "value": ", ".join(f"d-{i}" for i in range(20)),
                "source": "huggingface",
                "conflict": None,
            },
        },
    }
    spdx = validate_bom_to_spdx(bom, bom_type="ai", validate=False, relationship_cap=3)
    dataset_pkgs = [n for n in spdx["@graph"] if n.get("type") == "dataset_DatasetPackage"]
    assert len(dataset_pkgs) == 3


def test_cli_exposes_spdx_relationship_cap_flag():
    """CLI argparse must accept ``--spdx-relationship-cap N`` (Phase 10
    user requirement: parameter exposed in CLI / web / HF Space)."""
    import inspect
    import aikaboom.cli

    src = inspect.getsource(aikaboom.cli)
    assert "--spdx-relationship-cap" in src
    assert "spdx_relationship_cap" in src  # both flag form and arg attr form


def test_web_route_reads_spdx_relationship_cap_from_request_body():
    """The /process route pulls ``spdx_relationship_cap`` off the JSON
    body so the web UI + HF Space form fields work."""
    import inspect
    import aikaboom.web.app as app_module

    src = inspect.getsource(app_module)
    assert "spdx_relationship_cap" in src
    assert "data.get('spdx_relationship_cap')" in src or "data.get(\"spdx_relationship_cap\")" in src


# ---------------------------------------------------------------------------
# #23 — CycloneDX 1.6 conformance: no 1.7-only dataset type field
# ---------------------------------------------------------------------------


def test_cyclonedx_dataset_entries_have_no_type_field():
    """The 1.7-only ``type: "training"`` / ``type: "evaluation"`` keys are
    schema errors against the 1.6 spec (sbom-utility v0.18.x reported
    41-85 errors per BOM in the re-test). The exporter now omits the
    type field; train/test split lives in custom properties instead."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "test/m",
        "model_id": "test_m",
        "direct_fields": {},
        "rag_fields": {
            "trainedOnDatasets": {"value": "ImageNet, COCO", "source": "huggingface", "conflict": None},
            "testedOnDatasets": {"value": "MMLU, HellaSwag", "source": "huggingface", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    comp = cdx["components"][0]
    datasets = comp.get("modelCard", {}).get("modelParameters", {}).get("datasets", [])
    assert datasets, "datasets list should be populated"
    for d in datasets:
        assert "type" not in d, f"dataset entry has 1.7-only type field: {d}"
        assert "name" in d


def test_cyclonedx_train_test_split_surfaces_via_properties():
    """The train/test distinction (lost when ``type`` was dropped) is
    surfaced via ``aikaboom:training-set:names`` /
    ``aikaboom:evaluation-set:names`` properties so auditors can still
    tell them apart."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "test/m",
        "model_id": "test_m",
        "direct_fields": {},
        "rag_fields": {
            "trainedOnDatasets": {"value": "ImageNet, COCO", "source": "huggingface", "conflict": None},
            "testedOnDatasets": {"value": "MMLU", "source": "huggingface", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    comp = cdx["components"][0]
    props = {p["name"]: p["value"] for p in comp.get("properties", [])}
    assert "aikaboom:training-set:names" in props
    assert "ImageNet" in props["aikaboom:training-set:names"]
    assert "aikaboom:evaluation-set:names" in props
    assert "MMLU" in props["aikaboom:evaluation-set:names"]


# ---------------------------------------------------------------------------
# #24 — CycloneDX nil sentinels don't fabricate child entries
# ---------------------------------------------------------------------------


def test_cyclonedx_noassertion_dataset_filtered():
    """``trainedOnDatasets.value == "noAssertion"`` must NOT produce a
    fake child dataset entry named ``"noAssertion"`` (Finding #24)."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "test/m",
        "model_id": "test_m",
        "direct_fields": {},
        "rag_fields": {
            "trainedOnDatasets": {"value": "noAssertion", "source": "huggingface", "conflict": None},
            "testedOnDatasets": {"value": "Not found.", "source": "huggingface", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    comp = cdx["components"][0]
    datasets = comp.get("modelCard", {}).get("modelParameters", {}).get("datasets", [])
    assert datasets == [], f"silent fields should not fabricate datasets: {datasets}"
    # And no aikaboom:training-set:names property either.
    props = {p["name"]: p["value"] for p in comp.get("properties", [])}
    assert "aikaboom:training-set:names" not in props


def test_cyclonedx_noassertion_safety_filtered():
    """``safetyRiskAssessment == "noAssertion"`` must not produce an
    ``ethicalConsiderations[0] = {name: "safetyRiskAssessment",
    description: "noAssertion"}`` stub."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "test/m",
        "model_id": "test_m",
        "direct_fields": {},
        "rag_fields": {
            "safetyRiskAssessment": {"value": "noAssertion", "source": "huggingface", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    considerations = cdx["components"][0].get("modelCard", {}).get("considerations", {})
    assert "ethicalConsiderations" not in considerations, considerations


# ---------------------------------------------------------------------------
# #17 follow-through — recursive walker splits arrow lineage
# ---------------------------------------------------------------------------


def test_recursive_walker_arrow_split_via_shared_helper():
    """Arrow-typed modelLineage strings (``"A -> B"``) must produce
    distinct child targets in the recursive walker, not a single fake
    child named with the literal arrow string. SPDX builder already did
    this; Phase 10 ports the same shared helper to the walker."""
    from aikaboom.utils.recursive_bom import _split_targets

    triplet = {
        "value": "ancestor/A -> ancestor/B",
        "source": "huggingface",
        "conflict": None,
    }
    targets = _split_targets(triplet, parent_identifier="parent/model")
    assert sorted(targets) == ["ancestor/A", "ancestor/B"]


def test_recursive_walker_self_loop_drop():
    """Same helper drops the parent's own id when it appears as a
    lineage segment (a base model whose modelLineage points at itself
    — Finding #19)."""
    from aikaboom.utils.recursive_bom import _split_targets

    triplet = {
        "value": "parent/model -> ancestor/A",
        "source": "huggingface",
        "conflict": None,
    }
    targets = _split_targets(triplet, parent_identifier="parent/model")
    assert targets == ["ancestor/A"]


# ---------------------------------------------------------------------------
# #13 — langgraph warning suppression actually fires
# ---------------------------------------------------------------------------


def test_langgraph_warning_suppressed_at_package_init():
    """The Phase 9 filter in cli.py was installed too late (the package
    __init__ runs first when CLI is invoked, and pulls langgraph
    transitively). Phase 10 moves the suppression to
    ``aikaboom/__init__.py`` and monkey-patches
    ``langchain_core._api.deprecation.surface_langchain_deprecation_warnings``
    to a no-op so it can't re-enable the filter."""
    import inspect
    import aikaboom

    src = inspect.getsource(aikaboom)
    assert "surface_langchain_deprecation_warnings" in src
    assert "lambda: None" in src or "lambda :None" in src.replace(" ", "")
    assert "simplefilter" in src and "PendingDeprecationWarning" in src
    # The filter has to run BEFORE any aikaboom.utils.* import so it
    # catches the langgraph chain triggered by spdx_validator.
    init_path = inspect.getfile(aikaboom)
    text = open(init_path, encoding="utf-8").read()
    filter_idx = text.find("simplefilter")
    spdx_idx = text.find("from aikaboom.utils.spdx_validator")
    assert 0 < filter_idx < spdx_idx, (
        "warnings filter must precede the aikaboom.utils.spdx_validator import"
    )


# ---------------------------------------------------------------------------
# #12 + #6 — free-model surface removal regression tests
# ---------------------------------------------------------------------------


def test_free_model_helpers_no_longer_importable():
    """Phase 10 retired the free-tier path; the picker / filter / curated
    fallback / modality helper are gone. If a future change re-introduces
    any of them this test will fail and force a conscious decision."""
    from aikaboom.utils import openrouter_models

    for name in (
        "list_free_openrouter_models",
        "pick_free_openrouter_model",
        "_is_free",
        "_is_text_chat_model",
        "_NON_CHAT_NAME_PATTERNS",
        "CURATED_FREE_FALLBACK",
        "_curated_fallback_models",
    ):
        assert not hasattr(openrouter_models, name), (
            f"{name} should have been deleted in Phase 10"
        )


def test_free_flag_no_longer_in_cli_argparse():
    """``--free`` / ``--pick-free-model`` flags are gone from the CLI."""
    import inspect
    import aikaboom.cli

    src = inspect.getsource(aikaboom.cli)
    assert "--free" not in src
    assert "--pick-free-model" not in src
    assert "pick_free_openrouter_model" not in src
    assert "list_free_openrouter_models" not in src


def test_provider_resolver_default_is_paid():
    """OpenRouter default model in ``provider_resolver.PROVIDER_PRIORITY``
    must NOT carry the ``:free`` suffix (Phase 10)."""
    from aikaboom.utils.provider_resolver import PROVIDER_PRIORITY

    or_entry = next((e for e in PROVIDER_PRIORITY if e[0] == "openrouter"), None)
    assert or_entry is not None
    default_model = or_entry[2]
    assert ":free" not in default_model, (
        f"OpenRouter default still references a :free model: {default_model}"
    )


def test_package_init_no_longer_exports_free_helpers():
    import aikaboom

    assert "list_free_openrouter_models" not in aikaboom.__all__
    assert "pick_free_openrouter_model" not in aikaboom.__all__
    assert not hasattr(aikaboom, "list_free_openrouter_models")
    assert not hasattr(aikaboom, "pick_free_openrouter_model")
