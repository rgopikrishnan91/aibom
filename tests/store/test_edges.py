"""Edge extraction + persistence: artifact-to-artifact relationships."""

from aikaboom.store.edges import extract_relationship_targets


def _bom_with(field, value):
    return {"direct_fields": {field: {"value": value, "source": "huggingface",
                                      "conflict": None}}, "rag_fields": {}}


def test_extracts_trainedon_target():
    bom = _bom_with("trainedOnDatasets", "squad")
    assert ("trainedOn", "squad") in extract_relationship_targets(bom)


def test_extracts_testedon_and_dependson():
    assert ("testedOn", "glue") in extract_relationship_targets(
        _bom_with("testedOnDatasets", "glue"))
    assert ("dependsOn", "bert-base") in extract_relationship_targets(
        _bom_with("modelLineage", "bert-base"))


def test_splits_multi_value_strings():
    targets = extract_relationship_targets(_bom_with("trainedOnDatasets", "squad, glue; mnli"))
    names = {t for _, t in targets}
    assert names == {"squad", "glue", "mnli"}


def test_drops_non_walkable_targets():
    # arXiv refs are filtered by _is_walkable_target
    targets = extract_relationship_targets(_bom_with("sourceInfo", "arXiv:2108.07732"))
    assert targets == []


def test_ignores_unknown_and_empty_fields():
    assert extract_relationship_targets({"direct_fields": {}, "rag_fields": {}}) == []
    assert extract_relationship_targets(_bom_with("trainedOnDatasets", None)) == []
