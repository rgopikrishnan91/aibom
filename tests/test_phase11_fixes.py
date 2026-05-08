"""Phase 11 — re-test follow-ups on Mistral-7B-v0.1.

  #27 license now resolves via the *direct* path; both HF cardData and
      GitHub repo licence APIs already expose it. CDX `licenses` and SPDX
      `simplelicensing_LicenseExpression` populate from `direct.license`.
  #23 (final) every CDX modelCard dataset entry carries `type: "dataset"`.
  #26 recursive walker prefers `repo_id` over `model_id` so self-loops in
      modelLineage match the canonical form and get skipped.
  #28 SPDX `_as_list` / `_dictionary_entries` filter nil sentinels at every
      level so `["noAssertion"]` and the `dataset_sensor` stub disappear.
  #29 CDX dataset description reads from `rag.description`, falls back to
      `rag.intendedUse`; `"noAssertion"` doesn't end up as the literal text.
  URI consistency: dataset SPDX uses `urn:spdx:` everywhere, matching AI.
"""
from __future__ import annotations

import json
import re

import pytest


# ---------------------------------------------------------------------------
# #27 — license resolves on the direct path
# ---------------------------------------------------------------------------


def test_resolve_direct_fields_ai_includes_license():
    """HF and GitHub inspectors emit `license`; resolver must thread it
    into `direct_fields` so exporters don't need a RAG fallback."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {"license": "Apache-2.0"}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta)

    assert "license" in direct
    assert direct["license"] == "Apache-2.0"
    assert direct["license_source"] == "huggingface"
    assert direct["license_conflicts"] is None


def test_resolve_direct_fields_ai_license_conflict():
    """Cross-source disagreement surfaces in `license_conflicts`; HF wins
    because direct_fields default priority is huggingface > github."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {"license": "MIT"}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta)

    assert direct["license"] == "Apache-2.0"
    assert direct["license_source"] == "huggingface"
    assert direct["license_conflicts"]
    assert "github" in direct["license_conflicts"]
    assert "MIT" in direct["license_conflicts"]


def test_resolve_direct_fields_data_includes_license():
    """Dataset path mirror: HF dataset cards carry `cardData.license`."""
    from aikaboom.core.processors import DATABOMProcessor

    proc = DATABOMProcessor.__new__(DATABOMProcessor)
    hf_meta = {"license": "cc-by-sa-4.0"}
    gh_meta = {}

    direct = proc._resolve_direct_fields_data(hf_meta, gh_meta)

    assert "license" in direct
    # LicenseNormalizer should map cc-by-sa-4.0 → CC-BY-SA-4.0
    assert direct["license"].lower().startswith("cc-by-sa")


# ---------------------------------------------------------------------------
# #23 (final) — every CDX dataset entry carries type: "dataset"
# ---------------------------------------------------------------------------


def test_cdx_dataset_entries_carry_type_dataset():
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "x/y",
        "model_id": "x_y",
        "direct_fields": {
            "license": {"value": "MIT", "source": "hf", "conflict": None},
        },
        "rag_fields": {
            "trainedOnDatasets": {"value": "ds-a, ds-b", "source": "hf", "conflict": None},
            "testedOnDatasets": {"value": "ds-c", "source": "hf", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    datasets = cdx["components"][0]["modelCard"]["modelParameters"]["datasets"]

    assert len(datasets) == 3
    assert all(d["type"] == "dataset" for d in datasets), datasets
    assert {d["name"] for d in datasets} == {"ds-a", "ds-b", "ds-c"}


# ---------------------------------------------------------------------------
# #27 — exporter pass-through (direct.license is the canonical source)
# ---------------------------------------------------------------------------


def test_cdx_license_reads_from_direct_license_only():
    """Once `direct.license` is populated, no RAG fallback is needed and
    the exporter should not consult `rag.license`."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "repo_id": "x/y",
        "model_id": "x_y",
        "direct_fields": {
            "license": {"value": "Apache-2.0", "source": "hf", "conflict": None},
        },
        "rag_fields": {
            "license": {"value": "GPL-3.0", "source": "arxiv", "conflict": None},
        },
    }
    cdx = CycloneDXExporter(bom_type="ai").validate_and_convert(bom)
    licenses = cdx["components"][0].get("licenses", [])
    ids = [l["license"]["id"] for l in licenses]
    assert "Apache-2.0" in ids
    # RAG license must NOT win — direct is the canonical source post-Phase 11.
    assert "GPL-3.0" not in ids


def test_spdx_license_reads_from_direct_license():
    from aikaboom.utils.spdx_validator import SPDXValidator

    bom = {
        "repo_id": "x/y",
        "model_id": "x_y",
        "direct_fields": {
            "license": "Apache-2.0",
            "suppliedBy": "x",
            "downloadLocation": "https://hf.co/x/y",
        },
        "rag_fields": {},
    }
    spdx = SPDXValidator(bom_type="ai").validate_and_convert(bom)
    licenses = [
        e for e in spdx["@graph"]
        if e.get("type") == "simplelicensing_LicenseExpression"
    ]
    assert licenses
    assert licenses[0]["simplelicensing_licenseExpression"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# #26 — recursive walker prefers repo_id (canonical) over model_id (slug)
# ---------------------------------------------------------------------------


def test_walker_uses_repo_id_for_self_loop_visit_key():
    """If `model_id` is the slug `mistralai_Mistral-7B-v0.1` but `repo_id`
    is `mistralai/Mistral-7B-v0.1`, the visit-set must record the canonical
    form so a modelLineage triplet that points back at the parent is
    detected as a duplicate."""
    from aikaboom.utils.recursive_bom import discover_recursive_targets

    metadata = {
        "model_id": "mistralai_Mistral-7B-v0.1",
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "rag_fields": {
            "modelLineage": {
                "value": "fine-tuned from mistralai/Mistral-7B-v0.1",
                "source": "hf",
                "conflict": None,
            },
        },
    }
    targets, _ = discover_recursive_targets(metadata, bom_type="ai")
    parents = {t["parent"] for t in targets}
    assert parents == {"mistralai/Mistral-7B-v0.1"}, (
        f"expected canonical repo_id form, got {parents}"
    )


# ---------------------------------------------------------------------------
# #28 — SPDX list helpers filter nil sentinels at every level
# ---------------------------------------------------------------------------


def test_spdx_as_list_drops_nil_scalar():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    assert v._as_list("noAssertion") == []
    assert v._as_list("Not found.") == []
    assert v._as_list("") == []


def test_spdx_as_list_drops_nil_inside_list():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    assert v._as_list(["good", "noAssertion", "also-good"]) == ["good", "also-good"]


def test_spdx_as_list_drops_nil_segments_in_string():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    assert v._as_list("a, noAssertion, b") == ["a", "b"]


def test_spdx_dictionary_entries_drop_nil_sensor():
    """The `dataset_sensor` field used to fabricate
    `{"key": "value1", "value": "noAssertion"}` when the source said
    nothing. With the nil filter the helper returns an empty list."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    assert v._dictionary_entries("noAssertion") == []
    assert v._dictionary_entries({"sensor1": "noAssertion"}) == []
    # Real values still pass through.
    assert v._dictionary_entries({"sensor1": "GPS"}) == [
        {"type": "DictionaryEntry", "key": "sensor1", "value": "GPS"}
    ]


# ---------------------------------------------------------------------------
# #29 — CDX dataset description prefers rag.description over intendedUse
# ---------------------------------------------------------------------------


def test_cdx_dataset_description_prefers_description_field():
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "dataset_id": "squad",
        "direct_metadata": {"name": "SQuAD", "license": "CC-BY-SA-4.0"},
        "rag_metadata": {
            "description": "Reading comprehension dataset",
            "intendedUse": "QA research",
        },
    }
    cdx = CycloneDXExporter(bom_type="data").validate_and_convert(bom)
    assert cdx["components"][0]["description"] == "Reading comprehension dataset"


def test_cdx_dataset_description_falls_back_to_intendedUse():
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "dataset_id": "squad",
        "direct_metadata": {"name": "SQuAD", "license": "CC-BY-SA-4.0"},
        "rag_metadata": {"intendedUse": "QA research"},
    }
    cdx = CycloneDXExporter(bom_type="data").validate_and_convert(bom)
    assert cdx["components"][0]["description"] == "QA research"


def test_cdx_dataset_description_filters_nil_sentinel():
    """`"noAssertion"` must never end up as the literal description."""
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter

    bom = {
        "dataset_id": "squad",
        "direct_metadata": {"name": "SQuAD", "license": "CC-BY-SA-4.0"},
        "rag_metadata": {"description": "noAssertion", "intendedUse": "noAssertion"},
    }
    cdx = CycloneDXExporter(bom_type="data").validate_and_convert(bom)
    assert cdx["components"][0]["description"] == ""


# ---------------------------------------------------------------------------
# URI consistency — dataset SPDX uses urn:spdx: everywhere
# ---------------------------------------------------------------------------


def test_dataset_spdx_uses_urn_spdx_uri_form():
    from aikaboom.utils.spdx_validator import SPDXValidator

    spdx = SPDXValidator(bom_type="data").validate_and_convert(
        {
            "dataset_id": "squad",
            "direct_metadata": {
                "name": "SQuAD",
                "license": "CC-BY-SA-4.0",
                "originatedBy": "Stanford NLP",
                "downloadLocation": "https://hf.co/datasets/squad",
            },
            "rag_metadata": {},
            "urls": {"huggingface": "https://hf.co/datasets/squad"},
        }
    )
    serialised = json.dumps(spdx)
    assert "https://spdx.org/spdxdocs/" not in serialised, (
        "dataset path must not emit legacy https://spdx.org/spdxdocs/ URIs"
    )
    # Every spdxId on a non-blank node should now use urn:spdx:.
    for elem in spdx["@graph"]:
        sid = elem.get("spdxId")
        if sid:
            assert sid.startswith("urn:spdx:"), elem


# ---------------------------------------------------------------------------
# Phase 11.5 — license intra-source conflict (license-only special case)
# ---------------------------------------------------------------------------


def test_intra_source_license_conflict_hf():
    """HF cardData says Apache-2.0 but the same source's README says MIT
    → intra-source conflict on huggingface."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {"license": "Apache-2.0"}
    readmes = {
        "huggingface": "This software is licensed under the MIT License.",
        "github": "Released under the Apache License 2.0.",
    }

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta, named_readmes=readmes)

    intra = direct["license_intra_conflicts"]
    assert "huggingface" in intra
    assert "Apache-2.0" in intra["huggingface"]
    assert "MIT" in intra["huggingface"]
    # github agrees with itself, no conflict there.
    assert "github" not in intra


def test_intra_source_no_conflict_when_aliases_match():
    """cardData=Apache-2.0 vs README \"Apache License 2.0\" — same license
    after SPDX normalisation, no intra conflict."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {}
    readmes = {"huggingface": "Released under the Apache License 2.0."}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta, named_readmes=readmes)
    assert direct["license_intra_conflicts"] == {}


def test_intra_source_no_conflict_when_readme_silent():
    """README has no license mention → nothing to compare, no conflict."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {}
    readmes = {"huggingface": "This is a great model. No license info here."}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta, named_readmes=readmes)
    assert direct["license_intra_conflicts"] == {}


def test_intra_source_no_readmes_at_all():
    """Back-compat: resolver called without READMEs (the
    fetch_direct_metadata path) yields an empty intra dict, no crash."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {"license": "Apache-2.0"}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta)
    assert direct["license_intra_conflicts"] == {}


def test_intra_source_inter_check_still_fires():
    """Regression: adding the intra check must not break the existing
    cross-source license conflict detection."""
    from aikaboom.core.processors import AIBOMProcessor

    proc = AIBOMProcessor.__new__(AIBOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {"license": "MIT"}
    readmes = {"huggingface": "", "github": ""}

    direct = proc._resolve_direct_fields_ai(hf_meta, gh_meta, named_readmes=readmes)
    # Inter-source still flagged.
    assert direct["license_conflicts"]
    assert "github" in direct["license_conflicts"]
    # Intra empty (no README disagreements with cardData).
    assert direct["license_intra_conflicts"] == {}


def test_intra_source_data_path_works_too():
    """Dataset BOM path also runs the intra check — datasets often have
    license info in both cardData and README."""
    from aikaboom.core.processors import DATABOMProcessor

    proc = DATABOMProcessor.__new__(DATABOMProcessor)
    hf_meta = {"license": "Apache-2.0"}
    gh_meta = {}
    readmes = {"huggingface": "This dataset is released under the GPL-3.0."}

    direct = proc._resolve_direct_fields_data(hf_meta, gh_meta, named_readmes=readmes)
    intra = direct["license_intra_conflicts"]
    assert "huggingface" in intra
    assert "Apache-2.0" in intra["huggingface"]
    assert "GPL-3.0" in intra["huggingface"]
