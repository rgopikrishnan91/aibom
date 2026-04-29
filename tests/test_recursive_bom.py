from aikaboom.utils.recursive_bom import (
    build_linked_spdx_bundle,
    discover_recursive_targets,
    generate_recursive_boms,
    linked_bundle_summary,
)
from aikaboom.utils.spdx_validator import validate_spdx_export


def _clean_triplet(value):
    return {"value": value, "source": "huggingface", "conflict": None}


def _conflict_triplet(value, kind="internal"):
    other = "external" if kind == "internal" else "internal"
    return {
        "value": value,
        "source": "huggingface",
        "conflict": {kind: f"Yes: {kind} contradiction", other: "No"},
    }


def test_discover_recursive_targets_from_clean_relationships():
    metadata = {
        "model_id": "parent-model",
        "rag_fields": {
            "trainedOnDatasets": _clean_triplet("SQuAD, Common Crawl"),
            "testedOnDatasets": _clean_triplet("MMLU"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert [t["relationship_type"] for t in targets] == [
        "trainedOn", "trainedOn", "testedOn", "dependsOn",
    ]
    assert audit["skipped_due_to_conflict"] == []
    assert targets[-1]["bom_type"] == "ai"
    assert targets[-1]["resolvable_hint"] is True


def test_internal_conflict_blocks_recursion():
    metadata = {"rag_fields": {"trainedOnDatasets": _conflict_triplet("squad", "internal")}}
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert targets == []
    assert audit["skipped_due_to_conflict"][0]["field"] == "trainedOnDatasets"
    assert audit["skipped_due_to_conflict"][0]["reason"] == "conflict-detected"


def test_external_conflict_blocks_recursion():
    metadata = {"rag_fields": {"modelLineage": _conflict_triplet("meta-llama/Llama-2", "external")}}
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert targets == []
    assert audit["skipped_due_to_conflict"][0]["field"] == "modelLineage"


def test_depth_zero_returns_no_targets():
    out = generate_recursive_boms(
        {"rag_fields": {"trainedOnDatasets": _clean_triplet("squad")}},
        bom_type="ai",
        max_depth=0,
    )
    assert out["enabled"] is True
    assert out["generated_count"] == 0
    assert out["deepest_level_reached"] == 0


def test_non_ai_bom_returns_no_targets():
    targets, audit = discover_recursive_targets({"rag_fields": {}}, bom_type="data")
    assert targets == []
    assert audit["reason"].startswith("recursion only")


def test_default_depth_is_one_level():
    metadata = {
        "model_id": "parent",
        "rag_fields": {
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    out = generate_recursive_boms(metadata, bom_type="ai")  # default max_depth=1
    assert out["max_depth"] == 1
    assert out["generated_count"] == 2
    assert {n["bom_type"] for n in out["generated"]} == {"data", "ai"}
    assert all(n["depth"] == 1 for n in out["generated"])


def test_generates_both_data_and_ai_children_when_present():
    metadata = {
        "model_id": "parent-model",
        "rag_fields": {
            "trainedOnDatasets": _clean_triplet("squad"),
            "testedOnDatasets": _clean_triplet("mmlu"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    out = generate_recursive_boms(metadata, bom_type="ai", max_depth=1)
    by_rel = {n["relationship_type"]: n for n in out["generated"]}
    assert by_rel["trainedOn"]["bom_type"] == "data"
    assert by_rel["trainedOn"]["spdx_validation"]["valid"] is True
    assert by_rel["testedOn"]["bom_type"] == "data"
    assert by_rel["dependsOn"]["bom_type"] == "ai"
    assert by_rel["dependsOn"]["cyclonedx_data"]["bomFormat"] == "CycloneDX"


def test_true_recursion_with_enrich_callback_walks_tree():
    """Provide an enrich callback that adds relationship fields to a child;
    the walker should descend deeper until max_depth or exhaustion."""
    grandchildren_per_model = {
        "meta-llama/Llama-3": {
            "model_id": "Llama-3",
            "repo_id": "meta-llama/Llama-3",
            "rag_fields": {
                "trainedOnDatasets": _clean_triplet("the-pile"),
                "modelLineage": _clean_triplet("meta-llama/Llama-2"),
            },
        },
        "meta-llama/Llama-2": {
            "model_id": "Llama-2",
            "repo_id": "meta-llama/Llama-2",
            "rag_fields": {
                "trainedOnDatasets": _clean_triplet("c4"),
            },
        },
    }

    def enrich(target):
        if target["bom_type"] == "ai":
            return grandchildren_per_model.get(target["target"])
        return None  # data leaves stay as seeds

    parent = {
        "model_id": "parent",
        "rag_fields": {"modelLineage": _clean_triplet("meta-llama/Llama-3")},
    }

    out = generate_recursive_boms(parent, bom_type="ai", max_depth=3, enrich_fn=enrich)

    rels = [(n["relationship_type"], n["target"], n["depth"]) for n in out["generated"]]
    assert ("dependsOn", "meta-llama/Llama-3", 1) in rels
    assert ("trainedOn", "the-pile", 2) in rels
    assert ("dependsOn", "meta-llama/Llama-2", 2) in rels
    assert ("trainedOn", "c4", 3) in rels
    assert out["deepest_level_reached"] == 3
    assert out["tree_exhausted"] is True


def test_max_depth_truncates_walk():
    grandchildren = {
        "meta-llama/Llama-3": {
            "rag_fields": {"modelLineage": _clean_triplet("meta-llama/Llama-2")},
        },
        "meta-llama/Llama-2": {
            "rag_fields": {"modelLineage": _clean_triplet("meta-llama/Llama-1")},
        },
    }

    def enrich(target):
        return grandchildren.get(target["target"])

    parent = {
        "model_id": "parent",
        "rag_fields": {"modelLineage": _clean_triplet("meta-llama/Llama-3")},
    }

    out = generate_recursive_boms(parent, bom_type="ai", max_depth=1, enrich_fn=enrich)
    assert out["deepest_level_reached"] == 1
    assert out["tree_exhausted"] is False  # we stopped early
    assert {n["target"] for n in out["generated"]} == {"meta-llama/Llama-3"}


def test_visited_set_prevents_cycle():
    enriched = {
        "model-a": {"rag_fields": {"modelLineage": _clean_triplet("model-b")}},
        "model-b": {"rag_fields": {"modelLineage": _clean_triplet("model-a")}},  # cycle
    }

    def enrich(target):
        return enriched.get(target["target"])

    parent = {
        "model_id": "parent",
        "rag_fields": {"modelLineage": _clean_triplet("model-a")},
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=10, enrich_fn=enrich)
    targets = [n["target"] for n in out["generated"]]
    assert targets == ["model-a", "model-b"]
    assert any(d["target"] == "model-a" for d in out["duplicates"])
    assert out["tree_exhausted"] is True


def test_duplicate_dataset_referenced_twice_is_not_duplicated():
    grandchildren = {
        "meta-llama/Llama-3": {
            "rag_fields": {"trainedOnDatasets": _clean_triplet("squad")},
        },
    }

    def enrich(target):
        return grandchildren.get(target["target"])

    parent = {
        "model_id": "parent",
        "rag_fields": {
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=3, enrich_fn=enrich)
    squad_nodes = [n for n in out["generated"] if n["target"].lower() == "squad"]
    assert len(squad_nodes) == 1
    assert any(d["target"].lower() == "squad" for d in out["duplicates"])


def test_warns_about_resource_cost_in_payload():
    out = generate_recursive_boms(
        {"rag_fields": {"trainedOnDatasets": _clean_triplet("squad")}},
        bom_type="ai",
        max_depth=1,
    )
    joined = " ".join(out["warnings"]).lower()
    assert "beta" in joined
    assert "tree" in joined or "dependency" in joined


def test_linked_spdx_bundle_links_parent_to_children_with_relationships():
    parent = {
        "model_id": "parent-model",
        "repo_id": "org/parent",
        "direct_fields": {"license": "MIT"},
        "rag_fields": {
            "model_name": "Parent",
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    rec = generate_recursive_boms(parent, bom_type="ai", max_depth=1)

    bundle = build_linked_spdx_bundle(parent, rec, bom_type="ai")

    assert bundle["@context"]
    rel_types = {
        e["relationshipType"] for e in bundle["@graph"] if e.get("type") == "Relationship"
    }
    assert {"trainedOn", "dependsOn"} <= rel_types

    # Every relationship resolves: 'from' and the first 'to' must be in the graph.
    ids = {e.get("spdxId") or e.get("@id") for e in bundle["@graph"]}
    for r in (e for e in bundle["@graph"] if e.get("type") == "Relationship"):
        assert r["from"] in ids
        assert r["to"][0] in ids

    pkg_types = [e.get("type") for e in bundle["@graph"]]
    assert pkg_types.count("ai_AIPackage") >= 2  # parent + lineage child
    assert "dataset_DatasetPackage" in pkg_types

    # The bundle is spec-clean (no AIkaBoOM-private keys at root).
    assert set(bundle.keys()) == {"@context", "@graph"}

    summary = linked_bundle_summary(bundle, rec)
    assert summary["beta"] is True
    assert summary["recursive_edge_count"] >= 2
    assert summary["node_count"] == len(bundle["@graph"])


def test_linked_spdx_bundle_passes_lightweight_validation():
    parent = {
        "model_id": "parent-model",
        "repo_id": "org/parent",
        "direct_fields": {"license": "MIT"},
        "rag_fields": {
            "model_name": "Parent",
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    rec = generate_recursive_boms(parent, bom_type="ai", max_depth=1)
    bundle = build_linked_spdx_bundle(parent, rec, bom_type="ai")

    out = validate_spdx_export(bundle, strict=False, bom_type="ai")
    assert out["valid"], f"Linked bundle failed lightweight validation: {out['errors']}"
    assert out["validator"] == "jsonschema"
    assert out["errors"] == []


def test_linked_spdx_bundle_passes_strict_validation():
    parent = {
        "model_id": "parent-model",
        "repo_id": "org/parent",
        "direct_fields": {"license": "MIT"},
        "rag_fields": {
            "model_name": "Parent",
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    rec = generate_recursive_boms(parent, bom_type="ai", max_depth=1)
    bundle = build_linked_spdx_bundle(parent, rec, bom_type="ai")

    out = validate_spdx_export(bundle, strict=True, bom_type="ai")
    assert out["valid"], f"Linked bundle failed strict validation: {out['errors']}"
    assert out["validator"] == "jsonschema+shacl"


def test_linked_spdx_bundle_validates_after_multi_level_walk():
    """A multi-level enriched walk must still produce a spec-conformant bundle."""
    grand = {
        "meta-llama/Llama-3": {
            "model_id": "L3", "repo_id": "meta-llama/Llama-3",
            "rag_fields": {
                "trainedOnDatasets": _clean_triplet("the-pile"),
                "modelLineage": _clean_triplet("meta-llama/Llama-2"),
            },
        },
        "meta-llama/Llama-2": {
            "model_id": "L2", "repo_id": "meta-llama/Llama-2",
            "rag_fields": {"trainedOnDatasets": _clean_triplet("c4")},
        },
    }
    parent = {
        "model_id": "parent", "repo_id": "org/parent",
        "direct_fields": {"license": "MIT"},
        "rag_fields": {
            "model_name": "Parent",
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    rec = generate_recursive_boms(
        parent, bom_type="ai", max_depth=4,
        enrich_fn=lambda t: grand.get(t["target"]),
    )
    bundle = build_linked_spdx_bundle(parent, rec, bom_type="ai")

    light = validate_spdx_export(bundle, strict=False, bom_type="ai")
    assert light["valid"], f"3-deep bundle lightweight failed: {light['errors']}"
    strict = validate_spdx_export(bundle, strict=True, bom_type="ai")
    assert strict["valid"], f"3-deep bundle strict failed: {strict['errors']}"

    summary = linked_bundle_summary(bundle, rec)
    assert summary["deepest_level_reached"] == 3
    assert summary["recursive_edge_count"] >= 5
