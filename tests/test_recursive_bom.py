from aikaboom.utils.recursive_bom import (
    EXHAUST_DEPTH,
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


def _enriched_leaf(target):
    """Minimal 'successfully enriched' metadata for a target.

    The walker greys a child only when enrich_fn returns None, so a test
    enricher must return a dict for every target that should generate —
    the production enricher does the same (it fetches a full BOM for
    datasets too, not just models)."""
    if target.get("bom_type") == "data":
        return {
            "dataset_id": target["target"],
            "direct_metadata": {"name": target["target"]},
            "rag_metadata": {},
        }
    return {
        "model_id": target["target"],
        "repo_id": target["target"],
        "rag_fields": {"model_name": target["target"]},
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
    # modelLineage is discovered first so its dependsOn edge is never
    # starved by a large trainedOn/testedOn dataset fan-out under the cap.
    assert [t["relationship_type"] for t in targets] == [
        "dependsOn", "trainedOn", "trainedOn", "testedOn",
    ]
    assert audit["skipped_due_to_conflict"] == []
    assert targets[0]["relationship_type"] == "dependsOn"
    assert targets[0]["bom_type"] == "ai"
    assert targets[0]["resolvable_hint"] is True


def test_internal_conflict_flags_edge_but_walks_target():
    """A field with an internal conflict no longer blocks recursion — the
    target is still walked, just tagged with has_conflict so the UI/CLI
    can mark the edge. (Changed from skip-on-conflict so depth 3+ can
    still be reached when a depth-2 model's lineage is auditor-flagged.)"""
    metadata = {"rag_fields": {"trainedOnDatasets": _conflict_triplet("squad", "internal")}}
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert len(targets) == 1
    assert targets[0]["target"] == "squad"
    assert targets[0]["has_conflict"] is True
    assert audit["conflict_flagged"][0]["field"] == "trainedOnDatasets"
    assert audit["conflict_flagged"][0]["reason"] == "conflict-detected"
    # Back-compat alias still points at the same data
    assert audit["skipped_due_to_conflict"] == audit["conflict_flagged"]


def test_external_conflict_flags_edge_but_walks_target():
    """modelLineage with an external conflict — same: walked with ⚠."""
    metadata = {"rag_fields": {"modelLineage": _conflict_triplet("meta-llama/Llama-2", "external")}}
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert len(targets) == 1
    assert targets[0]["has_conflict"] is True
    assert targets[0]["relationship_type"] == "dependsOn"
    assert audit["conflict_flagged"][0]["field"] == "modelLineage"


def test_depth_zero_returns_no_targets():
    out = generate_recursive_boms(
        {"rag_fields": {"trainedOnDatasets": _clean_triplet("squad")}},
        bom_type="ai",
        max_depth=0,
    )
    assert out["enabled"] is True
    assert out["generated_count"] == 0
    assert out["deepest_level_reached"] == 0


def test_unsupported_bom_type_returns_no_targets():
    targets, audit = discover_recursive_targets({"rag_fields": {}}, bom_type="other")
    assert targets == []
    assert "not supported" in audit["reason"]


def test_data_bom_walks_sourceinfo_into_dependson_children():
    """Dataset BOMs derive child seeds from ``sourceInfo`` (upstream
    datasets), filtering out arXiv/paper-style references which are not
    walkable as BOMs."""
    metadata = {
        "dataset_id": "child_dataset",
        "rag_metadata": {
            "sourceInfo": _clean_triplet([
                "Common Crawl",
                "WebText",
                "arXiv:1910.10683",   # paper ref — must be filtered
                "2110.14168",          # bare arxiv ID — must be filtered
                "https://example.com/foo",  # URL — must be filtered
            ]),
        },
    }
    targets, audit = discover_recursive_targets(metadata, bom_type="data")
    target_names = sorted(t["target"] for t in targets)
    assert target_names == ["Common Crawl", "WebText"]
    assert all(t["relationship_type"] == "dependsOn" for t in targets)
    assert all(t["bom_type"] == "data" for t in targets)
    assert audit["skipped_due_to_conflict"] == []


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
        return _enriched_leaf(target)  # data leaves enrich to leaf nodes

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
    out = generate_recursive_boms(
        parent, bom_type="ai", max_depth=3,
        enrich_fn=lambda t: enrich(t) or _enriched_leaf(t),
    )
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
    # All four bullets we promise users must be present.
    assert "beta" in joined
    assert "unique-target set" in joined
    # New warning text covers conflict-walked edges (walked, not skipped).
    assert "conflict" in joined and "walked" in joined
    assert "enrich" in joined  # documents the enrich_fn escape hatch


def test_seed_only_walk_terminates_after_one_level():
    """No enrich_fn → seed children have no relationship fields, so the walk
    naturally exhausts even when max_depth is large."""
    parent = {
        "model_id": "parent",
        "rag_fields": {
            "trainedOnDatasets": _clean_triplet("squad"),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=5)
    assert out["tree_exhausted"] is True
    assert out["deepest_level_reached"] == 1
    assert all(n["depth"] == 1 for n in out["generated"])
    # Seed children are not enriched.
    assert all(n["enriched"] is False for n in out["generated"])


def test_max_depth_zero_never_invokes_enrich_callback():
    calls = []

    def enrich(target):
        calls.append(target["target"])
        return None

    parent = {
        "model_id": "parent",
        "rag_fields": {"trainedOnDatasets": _clean_triplet("squad")},
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=0, enrich_fn=enrich)
    assert calls == []
    assert out["generated_count"] == 0


def test_three_node_cycle_does_not_loop():
    enriched = {
        "model-a": {"rag_fields": {"modelLineage": _clean_triplet("model-b")}},
        "model-b": {"rag_fields": {"modelLineage": _clean_triplet("model-c")}},
        "model-c": {"rag_fields": {"modelLineage": _clean_triplet("model-a")}},
    }

    def enrich(target):
        return enriched.get(target["target"])

    parent = {
        "model_id": "parent",
        "rag_fields": {"modelLineage": _clean_triplet("model-a")},
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=10, enrich_fn=enrich)
    assert [n["target"] for n in out["generated"]] == ["model-a", "model-b", "model-c"]
    assert any(d["target"] == "model-a" for d in out["duplicates"])
    assert out["tree_exhausted"] is True


def test_diamond_dependency_visits_target_once():
    enriched = {
        "model-a": {
            "rag_fields": {
                "modelLineage": _clean_triplet("model-b"),
                "trainedOnDatasets": _clean_triplet("shared"),
            },
        },
        "model-b": {"rag_fields": {"trainedOnDatasets": _clean_triplet("shared")}},
    }

    def enrich(target):
        return enriched.get(target["target"]) or _enriched_leaf(target)

    parent = {
        "model_id": "parent",
        "rag_fields": {"modelLineage": _clean_triplet("model-a")},
    }
    out = generate_recursive_boms(parent, bom_type="ai", max_depth=4, enrich_fn=enrich)
    shared = [n for n in out["generated"] if n["target"] == "shared"]
    assert len(shared) == 1, "diamond dependency must produce exactly one node"
    assert any(d["target"] == "shared" for d in out["duplicates"])


def test_string_shaped_conflict_flags_edge():
    """The validator still recognises string-shaped conflict values (e.g.
    "github: squad-v2" from SourceHandler) as real conflicts — the edge
    is now walked but tagged ``has_conflict=True``."""
    metadata = {
        "rag_fields": {
            "trainedOnDatasets": {
                "value": "squad",
                "source": "huggingface",
                "conflict": "github: squad-v2",
            },
        },
    }
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert len(targets) == 1
    assert targets[0]["has_conflict"] is True
    assert audit["conflict_flagged"][0]["conflict"]["type"] == "inter"


def test_no_conflict_string_does_not_block_recursion():
    metadata = {
        "rag_fields": {
            "trainedOnDatasets": {"value": "squad", "conflict": "No conflict detected"},
        },
    }
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert [t["target"] for t in targets] == ["squad"]
    assert audit["skipped_due_to_conflict"] == []


def test_default_depth_is_one_asserts_tree_exhausted():
    """Regression: default-depth tests previously didn't assert the natural
    termination signal."""
    parent = {
        "model_id": "p",
        "rag_fields": {"trainedOnDatasets": _clean_triplet("squad")},
    }
    out = generate_recursive_boms(parent, bom_type="ai")
    assert out["tree_exhausted"] is True
    assert out["max_depth"] == 1


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
        enrich_fn=lambda t: grand.get(t["target"]) or _enriched_leaf(t),
    )
    bundle = build_linked_spdx_bundle(parent, rec, bom_type="ai")

    light = validate_spdx_export(bundle, strict=False, bom_type="ai")
    assert light["valid"], f"3-deep bundle lightweight failed: {light['errors']}"
    strict = validate_spdx_export(bundle, strict=True, bom_type="ai")
    assert strict["valid"], f"3-deep bundle strict failed: {strict['errors']}"

    summary = linked_bundle_summary(bundle, rec)
    assert summary["deepest_level_reached"] == 3
    assert summary["recursive_edge_count"] >= 5


def test_exhaust_mode_hits_safety_cap():
    """Under EXHAUST_DEPTH, an ever-fanning enrich callback must terminate at
    ``safety_cap`` rather than running forever. The walker records the
    cut-off in ``safety_capped`` (separated from conflict-walked edges in
    the conflict-tagged refactor) and flips ``tree_exhausted`` to False so
    the auditor can see the walk was bounded, not natural."""
    counter = {"n": 0}

    def enrich(target):
        # Each enriched node names a fresh child via modelLineage so the
        # frontier never empties on its own.
        counter["n"] += 1
        return {
            "model_id": target["target"],
            "rag_fields": {
                "modelLineage": _clean_triplet(f"org/child-{counter['n']:04d}"),
            },
        }

    parent = {
        "model_id": "root",
        "rag_fields": {"modelLineage": _clean_triplet("org/seed-1")},
    }

    out = generate_recursive_boms(
        parent,
        bom_type="ai",
        max_depth=EXHAUST_DEPTH,
        safety_cap=5,
        enrich_fn=enrich,
        validate_spdx=False,
    )
    assert out["generated_count"] == 5, (
        f"expected exactly safety_cap nodes, got {out['generated_count']}"
    )
    assert out["tree_exhausted"] is False
    capped = [s for s in out.get("safety_capped", [])
              if s.get("reason") == "safety-cap-reached"]
    assert capped, "safety-cap-reached entries must be recorded in safety_capped"


def test_conflict_flagging_under_phase4_structured_shape():
    """Phase 4's conflict trace can land in the triplet under a richer
    ``{type: "intra"|"inter", ...}`` shape (no legacy ``internal`` /
    ``external`` strings). ``_conflict_of`` must still recognise it so the
    walker tags the edge with has_conflict=True (instead of the old
    skip-and-continue behaviour)."""
    metadata = {
        "rag_fields": {
            "trainedOnDatasets": {
                "value": "squad",
                "source": "huggingface",
                "conflict": {
                    "type": "intra",
                    "value": "squad vs SQuAD-1.1",
                    "source": "huggingface",
                },
            },
        },
    }
    targets, audit = discover_recursive_targets(metadata, bom_type="ai")
    assert len(targets) == 1
    assert targets[0]["has_conflict"] is True
    assert audit["conflict_flagged"][0]["field"] == "trainedOnDatasets"


def test_modellineage_not_starved_by_dataset_fanout_under_safety_cap():
    """A model with a large testedOn fan-out plus one modelLineage edge.

    Regression: the safety cap is global and targets were discovered in
    trainedOn/testedOn/modelLineage order, so a big dataset fan-out
    consumed every slot and the modelLineage dependsOn child — the only
    recursable edge — was dropped. modelLineage must be discovered first
    so it is never crowded out by dataset leaves.
    """
    datasets = ", ".join(f"DS{i}" for i in range(20))
    metadata = {
        "repo_id": "microsoft/Magma-8B",
        "rag_fields": {
            "testedOnDatasets": _clean_triplet(datasets),
            "modelLineage": _clean_triplet("meta-llama/Llama-3-8B-Instruct"),
        },
    }
    out = generate_recursive_boms(metadata, bom_type="ai", max_depth=1, safety_cap=5)
    rels = {(n["relationship_type"], n["target"]) for n in out["generated"]}
    assert ("dependsOn", "meta-llama/Llama-3-8B-Instruct") in rels, (
        "modelLineage dependsOn child was starved by the dataset fan-out; "
        f"generated: {sorted(rels)}"
    )


def test_linked_bundle_has_no_duplicate_ai_package_stub():
    """modelLineage produces a recursive AI child; the parent SPDX's
    auto-generated dependsOn stub for the same model must be suppressed
    so the linked bundle carries exactly one ai_AIPackage per model.
    """
    metadata = {
        "repo_id": "microsoft/Magma-8B",
        "rag_fields": {"modelLineage": _clean_triplet("meta-llama/Llama-3-8B-Instruct")},
    }
    rec = generate_recursive_boms(metadata, bom_type="ai", max_depth=1)
    bundle = build_linked_spdx_bundle(metadata, rec, bom_type="ai")
    names = [
        e.get("name") for e in bundle["@graph"]
        if e.get("type") == "ai_AIPackage"
    ]
    assert names.count("meta-llama/Llama-3-8B-Instruct") == 1, (
        f"duplicate ai_AIPackage stub in linked bundle: {names}"
    )
    result = validate_spdx_export(bundle, bom_type="ai")
    assert result["valid"], (
        f"linked bundle with deduped stub must still validate: {result['errors']}"
    )


def test_breadth_caps_per_node_fanout():
    """breadth limits how many children a single node expands. Lineage is
    discovered first so the dependsOn edge always survives the budget."""
    datasets = ", ".join(f"DS{i}" for i in range(20))
    metadata = {
        "repo_id": "p/model",
        "rag_fields": {
            "testedOnDatasets": _clean_triplet(datasets),
            "modelLineage": _clean_triplet("meta-llama/Llama-3"),
        },
    }
    out = generate_recursive_boms(
        metadata, bom_type="ai", max_depth=1, breadth=4, safety_cap=200
    )
    assert out["generated_count"] == 4
    rels = {(n["relationship_type"], n["target"]) for n in out["generated"]}
    assert ("dependsOn", "meta-llama/Llama-3") in rels
    # 21 targets discovered, 4 expanded → 17 recorded as breadth-capped.
    assert len(out["breadth_capped"]) == 17
    assert all(e["reason"] == "breadth-cap-reached" for e in out["breadth_capped"])


def test_breadth_and_safety_cap_are_independent():
    """breadth is per-node; safety_cap is the absolute total. The smaller
    effective limit wins, and each capped target is labelled by its cause."""
    datasets = ", ".join(f"DS{i}" for i in range(20))
    metadata = {
        "repo_id": "p/model",
        "rag_fields": {"testedOnDatasets": _clean_triplet(datasets)},
    }
    out = generate_recursive_boms(
        metadata, bom_type="ai", max_depth=1, breadth=15, safety_cap=6
    )
    # safety_cap (6) is tighter than breadth (15) here.
    assert out["generated_count"] == 6
    assert len(out["safety_capped"]) == 14


def test_enrichment_failure_greys_node_instead_of_generating():
    """A child whose enrich_fn returns None is recorded as identified-
    but-not-generated, kept out of `generated`, and not walked deeper."""
    def failing_enrich(target):
        return None

    metadata = {
        "repo_id": "p/model",
        "rag_fields": {"modelLineage": _clean_triplet("some-unresolvable-base")},
    }
    out = generate_recursive_boms(
        metadata, bom_type="ai", max_depth=3, enrich_fn=failing_enrich
    )
    assert out["generated_count"] == 0
    assert len(out["enrichment_failed"]) == 1
    entry = out["enrichment_failed"][0]
    assert entry["target"] == "some-unresolvable-base"
    assert entry["reason"] == "unresolved"


def test_enrichment_exception_greys_node_with_error():
    """An enrich_fn that raises greys the node and records the error text."""
    def raising_enrich(target):
        raise RuntimeError("network down")

    metadata = {
        "repo_id": "p/m",
        "rag_fields": {"modelLineage": _clean_triplet("base-x")},
    }
    out = generate_recursive_boms(
        metadata, bom_type="ai", max_depth=2, enrich_fn=raising_enrich
    )
    assert out["generated_count"] == 0
    assert len(out["enrichment_failed"]) == 1
    entry = out["enrichment_failed"][0]
    assert entry["reason"] == "enrichment-failed"
    assert "network down" in (entry.get("error") or "")


def test_generate_single_node_on_demand():
    """generate_single_node enriches and builds one node for a greyed target."""
    from aikaboom.utils.recursive_bom import generate_single_node

    def enrich(target):
        return {
            "model_id": target["target"],
            "repo_id": target["target"],
            "rag_fields": {"model_name": target["target"]},
        }

    result = generate_single_node(
        {
            "target": "meta-llama/Llama-3", "bom_type": "ai",
            "relationship_type": "dependsOn", "parent": "p/model", "depth": 2,
        },
        enrich_fn=enrich,
    )
    assert result["ok"] is True
    node = result["node"]
    assert node["target"] == "meta-llama/Llama-3"
    assert node["depth"] == 2
    assert node["enriched"] is True
    assert "spdx_data" in node


def test_generate_single_node_unresolved_target():
    """A target the enricher cannot resolve returns ok=False, not a node."""
    from aikaboom.utils.recursive_bom import generate_single_node

    result = generate_single_node(
        {"target": "nonsense", "bom_type": "ai"},
        enrich_fn=lambda t: None,
    )
    assert result["ok"] is False
    assert result["error"] == "unresolved"
