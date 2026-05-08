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


# ---------------------------------------------------------------------------
# Phase 11.6 — license canonicalisation regressions (#30, #31, #32)
#              + #28 residual (dataPreprocessing / knownBias)
# ---------------------------------------------------------------------------


def test_cdx_license_id_for_canonical_spdx():
    """Canonical SPDX id from the alias table → license.id block."""
    from aikaboom.utils.normalise import cdx_license_block

    assert cdx_license_block("apache-2.0") == {"license": {"id": "Apache-2.0"}}
    assert cdx_license_block("Apache-2.0") == {"license": {"id": "Apache-2.0"}}
    assert cdx_license_block("MIT License") == {"license": {"id": "MIT"}}
    assert cdx_license_block("cc-by-sa-4.0") == {"license": {"id": "CC-BY-SA-4.0"}}


def test_cdx_license_name_for_unknown():
    """Non-SPDX-list strings (custom ML licences, OR-expressions) → license.name."""
    from aikaboom.utils.normalise import cdx_license_block

    assert cdx_license_block("OpenRAIL") == {"license": {"name": "OpenRAIL"}}
    assert cdx_license_block("Llama-2") == {"license": {"name": "Llama-2"}}
    # OR-expressions aren't SPDX ids on their own; emit as a name,
    # preserving the caller's casing.
    assert cdx_license_block("MIT OR Apache-2.0") == {
        "license": {"name": "MIT OR Apache-2.0"}
    }


def test_cdx_license_empty_omits_block():
    """None / empty / nil sentinels → no licenses[] entry."""
    from aikaboom.utils.normalise import cdx_license_block

    assert cdx_license_block(None) is None
    assert cdx_license_block("") is None
    assert cdx_license_block("NOASSERTION") is None
    assert cdx_license_block("noAssertion") is None
    assert cdx_license_block("Not found.") is None


def test_hf_inspector_unwraps_single_element_license_list():
    """HF cardData.license = ['mit'] → 'MIT' (string, canonical SPDX id)."""
    from aikaboom.utils.metadata_fetcher import _clean_hf_license

    assert _clean_hf_license(["mit"]) == "MIT"
    assert _clean_hf_license(["apache-2.0"]) == "Apache-2.0"
    assert _clean_hf_license(["cc-by-sa-3.0"]) == "CC-BY-SA-3.0"


def test_hf_inspector_joins_multi_element_license_list():
    """HF cardData.license = ['mit', 'apache-2.0'] → SPDX OR expression."""
    from aikaboom.utils.metadata_fetcher import _clean_hf_license

    assert _clean_hf_license(["mit", "apache-2.0"]) == "MIT OR Apache-2.0"


def test_hf_inspector_canonicalises_lowercase():
    """Bare HF lowercase string canonicalises through the alias table."""
    from aikaboom.utils.metadata_fetcher import _clean_hf_license

    assert _clean_hf_license("apache-2.0") == "Apache-2.0"
    assert _clean_hf_license("cc-by-sa-4.0") == "CC-BY-SA-4.0"
    # Empty / None pass through as None.
    assert _clean_hf_license(None) is None
    assert _clean_hf_license("") is None
    assert _clean_hf_license([]) is None
    assert _clean_hf_license([None, ""]) is None


def test_spdx_dataset_license_none_becomes_noassertion():
    """Linked-bundle children with no cardData.license → license_expr = NOASSERTION,
    not null. Reproduces #32."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {
            "license": {"value": None, "source": None, "conflict": None},
        },
        "rag_fields": {},
    }
    spdx = v.validate_and_convert(bom)
    license_elements = [
        e for e in spdx["@graph"]
        if e.get("type") == "simplelicensing_LicenseExpression"
    ]
    assert license_elements, "expected a LicenseExpression element"
    for e in license_elements:
        assert e["simplelicensing_licenseExpression"] == "NOASSERTION"


def test_spdx_dataset_license_list_handled_via_inspector():
    """If ['mit'] reaches the SPDX builder, it's a regression somewhere
    upstream — but defensive `_extract_value` plus the cleaning in
    `_clean_hf_license` should make sure it never does. This guards
    the boundary."""
    from aikaboom.utils.metadata_fetcher import _clean_hf_license
    from aikaboom.utils.spdx_validator import SPDXValidator

    # The inspector must clean lists before they reach the resolver.
    assert _clean_hf_license(["unknown"]) == "unknown"
    # Even the multi-element case becomes a single string.
    assert " OR " in _clean_hf_license(["mit", "apache-2.0"])

    # And if a list still slipped past for any reason, the SPDX builder's
    # belt-and-suspenders join must collapse it to a string.
    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {
            "license": {"value": ["mit"], "source": "huggingface", "conflict": None},
        },
        "rag_fields": {},
    }
    spdx = v.validate_and_convert(bom)
    for e in spdx["@graph"]:
        if e.get("type") == "simplelicensing_LicenseExpression":
            expr = e["simplelicensing_licenseExpression"]
            assert isinstance(expr, str), "license expression must be a string"
            assert expr == "mit"


def test_spdx_ai_license_none_becomes_noassertion():
    """AI BOM path mirror: None license → NOASSERTION, not null."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    bom = {
        "model_id": "x_y",
        "repo_id": "x/y",
        "direct_fields": {
            "license": {"value": None, "source": None, "conflict": None},
        },
        "rag_fields": {},
    }
    spdx = v.validate_and_convert(bom)
    license_elements = [
        e for e in spdx["@graph"]
        if e.get("type") == "simplelicensing_LicenseExpression"
    ]
    assert license_elements
    for e in license_elements:
        assert e["simplelicensing_licenseExpression"] == "NOASSERTION"


def test_spdx_dataset_data_preprocessing_drops_nil():
    """`dataset_dataPreprocessing: ['noAssertion']` regression — the inline
    `[x] if x else []` block was bypassing the Phase 11D `_is_nil_value`
    filter in `_as_list`."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {},
        "rag_fields": {
            "dataPreprocessing": {"value": "noAssertion", "source": None, "conflict": None},
        },
    }
    spdx = v.validate_and_convert(bom)
    dataset_packages = [
        e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage"
    ]
    assert dataset_packages
    for ds in dataset_packages:
        assert ds.get("dataset_dataPreprocessing", []) == [], ds


def test_spdx_dataset_known_bias_drops_nil():
    """Same as above for `knownBias`."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {},
        "rag_fields": {
            "knownBias": {"value": "noAssertion", "source": None, "conflict": None},
        },
    }
    spdx = v.validate_and_convert(bom)
    dataset_packages = [
        e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage"
    ]
    assert dataset_packages
    for ds in dataset_packages:
        assert ds.get("dataset_knownBias", []) == [], ds


# ---------------------------------------------------------------------------
# Phase 11.7 — truthy-triplet or-chain bug (#33) at the dataset SPDX builder
#              and the AI suppliedBy site
# ---------------------------------------------------------------------------


def test_first_non_nil_unwraps_triplet_with_none_value():
    """`_first_non_nil` falls past a triplet dict whose value is None,
    where the bare `or` chain would short-circuit on the truthy triplet."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    triplet_none = {"value": None, "source": None, "conflict": None}
    triplet_real = {"value": "realvalue", "source": "huggingface", "conflict": None}

    # Dead-branch: triplet with None value should be skipped, default used.
    assert v._first_non_nil(triplet_none, default="X") == "X"
    # Real value: returned as-is.
    assert v._first_non_nil(triplet_real, default="X") == "realvalue"
    # Cascade: first non-nil wins.
    assert v._first_non_nil(triplet_none, triplet_real, default="X") == "realvalue"
    # Nil sentinels skipped.
    assert v._first_non_nil({"value": "noAssertion"}, default="X") == "X"
    assert v._first_non_nil("", default="X") == "X"


def test_spdx_dataset_download_location_falls_back_when_direct_none():
    """#33: when an unresolved child has direct.downloadLocation = None,
    the SPDX bundle must fall back to urls.github / urls.huggingface /
    NOASSERTION instead of emitting `software_downloadLocation: null`."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "allenai/quac",
        "name": "allenai/quac",
        "direct_fields": {
            "downloadLocation": {"value": None, "source": None, "conflict": None},
            "originatedBy": {"value": None, "source": None, "conflict": None},
            "builtTime": {"value": None, "source": None, "conflict": None},
            "releaseTime": {"value": None, "source": None, "conflict": None},
            "license": {"value": None, "source": None, "conflict": None},
        },
        "rag_fields": {},
        "urls": {"huggingface": "https://huggingface.co/datasets/allenai/quac"},
    }
    spdx = v.validate_and_convert(bom)
    dataset_packages = [
        e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage"
    ]
    assert dataset_packages
    for ds in dataset_packages:
        dl = ds.get("software_downloadLocation")
        assert dl is not None, "software_downloadLocation must never be null"
        assert dl != "", "software_downloadLocation must never be empty"
        # Either the HF url fallback or NOASSERTION — both valid.
        assert dl in (
            "https://huggingface.co/datasets/allenai/quac",
            "NOASSERTION",
        ), f"unexpected downloadLocation: {dl!r}"


def test_spdx_dataset_download_location_uses_direct_when_present():
    """Regression: when direct.downloadLocation IS populated, that's
    what we use — the new fallback logic must not override real values."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {
            "downloadLocation": {
                "value": "https://huggingface.co/datasets/x/y",
                "source": "huggingface",
                "conflict": None,
            },
        },
        "rag_fields": {},
        "urls": {"github": "https://github.com/should-not-win"},
    }
    spdx = v.validate_and_convert(bom)
    dataset_packages = [
        e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage"
    ]
    for ds in dataset_packages:
        assert ds["software_downloadLocation"] == "https://huggingface.co/datasets/x/y"


def test_spdx_dataset_originated_by_falls_back_to_unknown():
    """#33 sibling: originatedBy with None triplet must fall back to "Unknown",
    not propagate None into Person/Organization elements."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="data")
    bom = {
        "dataset_id": "x/y",
        "name": "x/y",
        "direct_fields": {
            "originatedBy": {"value": None, "source": None, "conflict": None},
        },
        "rag_fields": {},
    }
    spdx = v.validate_and_convert(bom)
    # Walk the @graph; nothing should carry a literal None as name.
    for e in spdx["@graph"]:
        if e.get("type") in {"Person", "Organization"}:
            assert e.get("name") is not None, e


def test_spdx_ai_supplied_by_falls_back_to_unknown():
    """AI builder sibling at line 447: triplet with None value previously
    produced supplied_by = None; now falls back to "Unknown"."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    bom = {
        "model_id": "x_y",
        "repo_id": "x/y",
        "direct_fields": {
            "suppliedBy": {"value": None, "source": None, "conflict": None},
        },
        "rag_fields": {},
    }
    spdx = v.validate_and_convert(bom)
    for e in spdx["@graph"]:
        if e.get("type") in {"Person", "Organization"}:
            assert e.get("name") is not None, e
