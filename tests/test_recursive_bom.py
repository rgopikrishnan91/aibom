from aikaboom.utils.recursive_bom import (
    discover_recursive_targets,
    generate_recursive_boms,
)


def test_discover_recursive_targets_from_clean_relationships():
    metadata = {
        "model_id": "parent-model",
        "rag_fields": {
            "trainedOnDatasets": {
                "value": "SQuAD, Common Crawl",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "No"},
            },
            "testedOnDatasets": {
                "value": "MMLU",
                "source": "arxiv",
                "conflict": None,
            },
            "modelLineage": {
                "value": "meta-llama/Llama-3",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "No"},
            },
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
    metadata = {
        "rag_fields": {
            "trainedOnDatasets": {
                "value": "squad",
                "source": "huggingface",
                "conflict": {"internal": "Yes: README mentions both squad and squad_v2", "external": "No"},
            },
        },
    }

    targets, audit = discover_recursive_targets(metadata, bom_type="ai")

    assert targets == []
    assert len(audit["skipped_due_to_conflict"]) == 1
    skip = audit["skipped_due_to_conflict"][0]
    assert skip["field"] == "trainedOnDatasets"
    assert skip["reason"] == "conflict-detected"
    assert skip["conflict"]["internal"].startswith("Yes")


def test_external_conflict_blocks_recursion():
    metadata = {
        "rag_fields": {
            "modelLineage": {
                "value": "meta-llama/Llama-2",
                "source": "huggingface",
                "conflict": {"internal": "No", "external": "Yes: arxiv lists Llama-3"},
            },
        },
    }

    targets, audit = discover_recursive_targets(metadata, bom_type="ai")

    assert targets == []
    assert audit["skipped_due_to_conflict"][0]["field"] == "modelLineage"


def test_depth_zero_returns_no_targets():
    metadata = {"rag_fields": {"trainedOnDatasets": {"value": "squad", "conflict": None}}}
    out = generate_recursive_boms(metadata, bom_type="ai", max_depth=0)
    assert out["enabled"] is True
    assert out["generated_count"] == 0
    assert out["generated"] == []
    assert out["skipped_due_to_conflict"] == []


def test_non_ai_bom_returns_no_targets():
    targets, audit = discover_recursive_targets({"rag_fields": {}}, bom_type="data")
    assert targets == []
    assert audit["reason"].startswith("recursion only")


def test_generate_recursive_boms_emits_child_exports():
    metadata = {
        "model_id": "parent-model",
        "rag_fields": {
            "trainedOnDatasets": {"value": "squad", "source": "huggingface", "conflict": None},
        },
    }

    result = generate_recursive_boms(metadata, bom_type="ai", max_depth=1)

    assert result["beta"] is True
    assert result["strategy"] == "conflict-gated relationship recursion"
    assert result["generated_count"] == 1
    child = result["generated"][0]
    assert child["bom_type"] == "data"
    assert child["relationship_type"] == "trainedOn"
    assert child["spdx_validation"]["valid"] is True
    assert child["cyclonedx_data"]["bomFormat"] == "CycloneDX"


def test_generate_recursive_boms_reports_skips_in_payload():
    metadata = {
        "rag_fields": {
            "trainedOnDatasets": {
                "value": "squad",
                "source": "huggingface",
                "conflict": {"internal": "Yes: contradiction", "external": "No"},
            },
            "modelLineage": {"value": "meta-llama/Llama-3", "conflict": None},
        },
    }

    result = generate_recursive_boms(metadata, bom_type="ai", max_depth=1)

    assert result["generated_count"] == 1
    assert result["generated"][0]["relationship_type"] == "dependsOn"
    assert len(result["skipped_due_to_conflict"]) == 1
    assert result["skipped_due_to_conflict"][0]["field"] == "trainedOnDatasets"
