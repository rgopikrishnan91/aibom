"""Conflict → SPDX Annotation emission.

The SPDX writer must produce SPDX 3.0.1 ``Annotation`` Elements (one per
detected conflict) attached via ``subject`` to the root AIPackage /
DatasetPackage. The ``statement`` carries the structured conflict
provenance as JSON so downstream consumers can deep-link from a UI row
back to the exact graph node without re-running the auditor.

Recursive coverage: when ``build_linked_spdx_bundle`` merges per-child
SPDX docs, every child's annotations come along and resolve to elements
that exist in the merged graph (i.e. no dangling ``subject`` refs).
"""
from __future__ import annotations

import json
import pytest

from aikaboom.utils.spdx_validator import SPDXValidator, validate_spdx_export
from aikaboom.utils.recursive_bom import build_linked_spdx_bundle, generate_recursive_boms


def _ai_bom_with_conflicts():
    return {
        "repo_id": "acme/test-model",
        "direct_fields": {
            "license": {
                "value": "MIT",
                "source": "github",
                "conflict": {"value": "Apache-2.0", "source": "arxiv", "type": "inter"},
            },
            "suppliedBy": {"value": "ACME", "source": "github", "conflict": None},
        },
        "rag_fields": {
            "model_name": {
                "value": "Test",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "No"},
            },
            "intended_use": {
                "value": "Chatbot",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "Yes"},
                "trace": {
                    "claims": {
                        "huggingface": "Chatbot for customer support",
                        "arxiv": "General language model evaluation",
                    },
                    "selected_sources": ["huggingface"],
                    "internal_conflicts": {},
                    "external_conflicts": [{
                        "sources": ["huggingface", "arxiv"],
                        "description": 'A says "X" vs B says "Y"',
                        "statements": ["X", "Y"],
                        "grounding": 0.85,
                    }],
                },
            },
            "training_information": {
                "value": "Trained on web-crawled text",
                "source": "huggingface",
                "conflict": {"internal": "Yes", "external": "No"},
                "trace": {
                    "claims": {"huggingface": "Trained on web-crawled text"},
                    "selected_sources": ["huggingface"],
                    "internal_conflicts": {
                        "huggingface": {
                            "narrative": 'web vs curated',
                            "statements": ["web", "curated"],
                            "grounding": 0.55,
                        },
                    },
                    "external_conflicts": [],
                },
            },
        },
    }


def test_ai_bom_emits_one_annotation_per_conflict():
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    annotations = [e for e in doc["@graph"] if e.get("type") == "Annotation"]
    # license (direct-inter) + intended_use (rag-external) + training_information (rag-internal)
    assert len(annotations) == 3

    kinds = sorted(json.loads(a["statement"])["kind"] for a in annotations)
    assert kinds == ["inter", "rag-external", "rag-internal"]


def test_phantom_rag_envelope_does_not_emit_annotation():
    """``{internal: No, external: No}`` is a no-op envelope — must not annotate."""
    bom = {
        "repo_id": "acme/quiet",
        "direct_fields": {
            "license": {"value": "MIT", "source": "github", "conflict": None},
        },
        "rag_fields": {
            "model_name": {
                "value": "Quiet",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "No"},
            },
        },
    }
    doc = SPDXValidator(bom_type="ai").validate_and_convert(bom)
    annotations = [e for e in doc["@graph"] if e.get("type") == "Annotation"]
    assert annotations == []


def test_annotation_subject_resolves_to_ai_package():
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    types_by_id = {
        (e.get("spdxId") or e.get("@id")): e.get("type")
        for e in doc["@graph"]
    }
    for a in [e for e in doc["@graph"] if e.get("type") == "Annotation"]:
        assert a["subject"] in types_by_id, "subject must reference an existing Element"
        assert types_by_id[a["subject"]] == "ai_AIPackage"


def test_annotation_required_properties():
    """``annotationType`` and ``subject`` are SHACL-required."""
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    for a in [e for e in doc["@graph"] if e.get("type") == "Annotation"]:
        assert a["annotationType"] in {"other", "review"}
        assert a["subject"].startswith("urn:spdx:")
        assert a["spdxId"].startswith("urn:spdx:Annotation-")
        assert a["creationInfo"]  # must reference CreationInfo
        # contentType pattern: ``type/subtype``
        assert "/" in a.get("contentType", "")


def test_annotation_statement_is_valid_json():
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    for a in [e for e in doc["@graph"] if e.get("type") == "Annotation"]:
        payload = json.loads(a["statement"])
        assert "field" in payload
        assert "kind" in payload
        assert "section" in payload
        assert "spdx_property" in payload


def test_annotation_statement_carries_spdx_property_name():
    """``statement.spdx_property`` is the SPDX 3.0.1 property IRI suffix,
    not the AIkaBoOM-internal field key.

    Lets a downstream SPDX viewer say *"this is a conflict on
    ai_informationAboutTraining"* without having to know AIkaBoOM's
    snake_case → camelCase → SPDX-prefixed translation pipeline.
    """
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    by_field = {}
    for a in [e for e in doc["@graph"] if e.get("type") == "Annotation"]:
        payload = json.loads(a["statement"])
        by_field[payload["field"]] = payload["spdx_property"]
    assert by_field["license"] == "simplelicensing_licenseExpression"
    assert by_field["intended_use"] == "ai_informationAboutApplication"
    assert by_field["training_information"] == "ai_informationAboutTraining"


def test_annotation_name_uses_spdx_property():
    """The annotation ``name`` is keyed on the SPDX property, not the
    BOM-internal field key — same value an SPDX consumer would see in
    ``statement.spdx_property``.
    """
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    names = sorted(
        a["name"] for a in doc["@graph"] if a.get("type") == "Annotation"
    )
    assert names == sorted([
        "aikaboom:conflict:ai_informationAboutApplication",
        "aikaboom:conflict:ai_informationAboutTraining",
        "aikaboom:conflict:simplelicensing_licenseExpression",
    ])


def test_two_distinct_property_conflicts_emit_two_annotations():
    """Concrete worked example from the design discussion: conflicts on
    both ``training_information`` and ``safety_risk_assessment`` produce
    two annotations whose ``name`` / ``statement.spdx_property`` carry
    the two SPDX property names, both with subject = the AIPackage.
    """
    bom = {
        "repo_id": "acme/two-conflict",
        "direct_fields": {},
        "rag_fields": {
            "training_information": {
                "value": "Trained on web text",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "Yes"},
                "trace": {
                    "claims": {"huggingface": "web text", "arxiv": "curated"},
                    "selected_sources": ["huggingface"],
                    "internal_conflicts": {},
                    "external_conflicts": [{
                        "sources": ["huggingface", "arxiv"],
                        "description": "web vs curated",
                        "grounding": 0.8,
                    }],
                },
            },
            "safety_risk_assessment": {
                "value": "high",
                "source": "huggingface",
                "conflict": {"internal": "Yes", "external": "No"},
                "trace": {
                    "claims": {"huggingface": "high"},
                    "selected_sources": ["huggingface"],
                    "internal_conflicts": {
                        "huggingface": {
                            "narrative": '"high" vs "low"',
                            "grounding": 0.65,
                        },
                    },
                    "external_conflicts": [],
                },
            },
        },
    }
    doc = SPDXValidator(bom_type="ai").validate_and_convert(bom)
    annotations = [e for e in doc["@graph"] if e.get("type") == "Annotation"]
    assert len(annotations) == 2

    by_property = {
        json.loads(a["statement"])["spdx_property"]: a for a in annotations
    }
    assert set(by_property) == {
        "ai_informationAboutTraining",
        "ai_safetyRiskAssessment",
    }
    # Both annotations share the same subject — the AIPackage.
    ai_pkg = next(
        e for e in doc["@graph"] if e.get("type") == "ai_AIPackage"
    )
    for a in annotations:
        assert a["subject"] == ai_pkg["spdxId"]
        assert a["name"].startswith("aikaboom:conflict:ai_")


def test_annotated_ai_bom_passes_json_schema():
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    result = validate_spdx_export(doc, strict=False, bom_type="ai")
    assert result["valid"], result["errors"]


def test_annotated_ai_bom_passes_strict_shacl():
    pytest.importorskip("pyshacl")
    doc = SPDXValidator(bom_type="ai").validate_and_convert(_ai_bom_with_conflicts())
    result = validate_spdx_export(doc, strict=True, bom_type="ai")
    assert result["valid"], result["errors"]


def test_dataset_bom_emits_annotations():
    """Use the section names the real DATABOMProcessor pipeline emits
    (``direct_fields`` / ``rag_fields`` — see processors.py:894–895),
    not the legacy ``direct_metadata`` / ``rag_metadata`` shape.
    """
    bom = {
        "dataset_id": "acme/qa",
        "direct_fields": {
            "name": {"value": "qa", "source": "huggingface", "conflict": None},
            "license": {
                "value": "MIT",
                "source": "github",
                "conflict": {"value": "Apache-2.0", "source": "arxiv", "type": "inter"},
            },
        },
        "rag_fields": {
            "intendedUse": {"value": "QA", "source": "huggingface", "conflict": None},
            "datasetType": {
                "value": "text",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "Yes"},
                "trace": {
                    "claims": {"huggingface": "text", "arxiv": "categorical"},
                    "selected_sources": ["huggingface"],
                    "internal_conflicts": {},
                    "external_conflicts": [{
                        "sources": ["huggingface", "arxiv"],
                        "description": "text vs categorical",
                        "grounding": 0.75,
                    }],
                },
            },
        },
        "urls": {},
    }
    doc = SPDXValidator(bom_type="data").validate_and_convert(bom)
    annotations = [e for e in doc["@graph"] if e.get("type") == "Annotation"]
    # license (direct-inter) + datasetType (rag-external)
    assert len(annotations) == 2
    for a in annotations:
        assert a["subject"].startswith("urn:spdx:DatasetPackage-")

    by_property = {
        json.loads(a["statement"])["spdx_property"]: a for a in annotations
    }
    # Verify the SPDX property mapping kicks in for both the license
    # special case and a regular dataset_ property.
    assert "simplelicensing_licenseExpression" in by_property
    assert "dataset_datasetType" in by_property

    result = validate_spdx_export(doc, strict=False, bom_type="data")
    assert result["valid"], result["errors"]


def test_dataset_bom_legacy_metadata_shape_still_works():
    """Hand-authored / pre-pipeline BOMs may still use
    ``direct_metadata`` / ``rag_metadata`` keys. The writer accepts both
    shapes (spdx_validator.py:1149 fallback), and annotations should
    still be emitted so legacy BOMs don't silently lose conflict
    provenance after upgrading.
    """
    bom = {
        "dataset_id": "acme/qa-legacy",
        "direct_metadata": {
            "name": {"value": "qa-legacy", "source": "huggingface", "conflict": None},
            "license": {
                "value": "MIT",
                "source": "github",
                "conflict": {"value": "Apache-2.0", "source": "arxiv", "type": "inter"},
            },
        },
        "rag_metadata": {
            "intendedUse": {"value": "QA", "source": "huggingface", "conflict": None},
        },
        "urls": {},
    }
    doc = SPDXValidator(bom_type="data").validate_and_convert(bom)
    annotations = [e for e in doc["@graph"] if e.get("type") == "Annotation"]
    assert len(annotations) == 1
    payload = json.loads(annotations[0]["statement"])
    assert payload["spdx_property"] == "simplelicensing_licenseExpression"


def _enrich_with_conflict(target):
    if target["bom_type"] != "data":
        return None
    name = target["target"]
    return {
        "dataset_id": name,
        "direct_metadata": {
            "name": {"value": name, "source": "huggingface", "conflict": None},
            "license": {
                "value": "CC-BY",
                "source": "github",
                "conflict": {"value": "CC-BY-NC", "source": "huggingface", "type": "inter"},
            },
        },
        "rag_metadata": {
            "intendedUse": {"value": "Eval", "source": "huggingface", "conflict": None},
        },
        "urls": {},
    }


def test_linked_recursive_bundle_propagates_annotations():
    parent = _ai_bom_with_conflicts()
    parent["rag_fields"]["trainedOnDatasets"] = {
        "value": "squad, glue",
        "source": "github",
        "conflict": {"internal": "No", "external": "No"},
    }
    recursive = generate_recursive_boms(
        parent, bom_type="ai", max_depth=2, enrich_fn=_enrich_with_conflict,
    )
    assert recursive["generated_count"] == 2

    bundle = build_linked_spdx_bundle(parent, recursive, bom_type="ai")
    graph = bundle["@graph"]
    annotations = [e for e in graph if e.get("type") == "Annotation"]
    # 3 parent annotations + 1 per child dataset = 5
    assert len(annotations) == 5

    id_to_type = {(e.get("spdxId") or e.get("@id")): e.get("type") for e in graph}
    for a in annotations:
        assert a["subject"] in id_to_type, f"dangling subject: {a['subject']}"

    # Annotations must point at one of: parent AIPackage, child DatasetPackages.
    target_types = {id_to_type[a["subject"]] for a in annotations}
    assert target_types <= {"ai_AIPackage", "dataset_DatasetPackage"}


def test_linked_recursive_bundle_passes_strict_shacl():
    pytest.importorskip("pyshacl")
    parent = _ai_bom_with_conflicts()
    recursive = generate_recursive_boms(
        parent, bom_type="ai", max_depth=2, enrich_fn=_enrich_with_conflict,
    )
    bundle = build_linked_spdx_bundle(parent, recursive, bom_type="ai")
    result = validate_spdx_export(bundle, strict=True, bom_type="ai")
    assert result["valid"], result["errors"]


def test_web_annotate_conflicts_with_spdx_ids_matches_by_field_and_kind():
    from aikaboom.web.app import _annotate_conflicts_with_spdx_ids, _extract_conflicts

    bom = _ai_bom_with_conflicts()
    doc = SPDXValidator(bom_type="ai").validate_and_convert(bom)
    conflicts = _extract_conflicts(bom)
    annotated = _annotate_conflicts_with_spdx_ids(conflicts, doc)

    # Every conflict row should be cross-referenced to an Annotation
    assert all("spdx_annotation_id" in c for c in annotated), (
        "Every conflict row should be cross-referenced to an annotation."
    )
    ids = [c["spdx_annotation_id"] for c in annotated]
    assert len(set(ids)) == len(ids), "Each conflict gets a unique annotation."
