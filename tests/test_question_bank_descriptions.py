"""Regression test: every question-bank `description` matches the SPDX 3.0.1
spec text (verbatim) or is explicitly marked as AIkaBoOM-internal.

Re-run ``python tools/harvest_spdx_3_0_1.py`` and
``python tools/sync_question_bank_descriptions.py --apply`` to refresh
when SPDX revs."""
import json
import os

import pytest


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
_QB_ROOT = os.path.join(_REPO_ROOT, "src", "aikaboom", "question_bank")
_INDEX_PATH = os.path.join(_REPO_ROOT, "docs", "SPDX_3.0.1_FIELD_REFERENCE.json")


def _load_index():
    with open(_INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def _expected_description(prop: dict) -> str:
    """Mirror tools/sync_question_bank_descriptions.py::_spdx_description."""
    summary = (prop.get("summary") or "").strip()
    description = (prop.get("description") or "").strip()
    if summary and description:
        return f"{summary}\n\n{description}"
    return summary or description


def _entries():
    """Yield (bom_type, field, entry_dict) for every question-bank JSON."""
    for bom_type in ("ai", "data"):
        folder = os.path.join(_QB_ROOT, bom_type)
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as f:
                yield bom_type, name[:-5], json.load(f)


@pytest.fixture(scope="module")
def index():
    return _load_index()


class TestReferenceIndexExists:
    """The reference doc must be present and well-formed; sync depends on it."""

    def test_index_loads(self, index):
        assert index["spdx_version"] == "3.0.1"
        assert "properties" in index
        assert "aikaboom_field_to_spdx" in index

    def test_mapping_covers_every_question_bank_field(self, index):
        """Every JSON entry in question_bank/ has a mapping in the
        reference index — either to an SPDX property or to ``None``
        (AIkaBoOM-internal)."""
        for bom_type, field, _ in _entries():
            assert field in index["aikaboom_field_to_spdx"][bom_type], (
                f"{bom_type}/{field}: question_bank JSON exists but no entry "
                f"in tools/harvest_spdx_3_0_1.py:{bom_type.upper()}_FIELD_TO_SPDX"
            )


class TestDescriptionsMatchSpec:
    """Every entry's description is either verbatim SPDX text (for fields
    mapped to a published SPDX property) or carries the
    ``aikaboom_internal: true`` flag (for AIkaBoOM-only relationship
    targets and aggregates)."""

    def test_every_entry_matches_or_is_internal(self, index):
        properties = index["properties"]
        mapping = index["aikaboom_field_to_spdx"]
        failures = []
        for bom_type, field, entry in _entries():
            spdx_name = mapping[bom_type].get(field)
            if spdx_name is None:
                if entry.get("aikaboom_internal") is not True:
                    failures.append(
                        f"{bom_type}/{field}.json: AIkaBoOM-internal field must "
                        f"declare `\"aikaboom_internal\": true`"
                    )
                continue
            if entry.get("aikaboom_internal") is True:
                failures.append(
                    f"{bom_type}/{field}.json: maps to SPDX `{spdx_name}` but "
                    f"is marked aikaboom_internal — drop the flag and re-sync"
                )
                continue
            prop = properties.get(spdx_name)
            if prop is None:
                failures.append(
                    f"{bom_type}/{field}.json -> SPDX `{spdx_name}` not found "
                    f"in reference index; re-run harvest"
                )
                continue
            expected = _expected_description(prop)
            if entry["description"] != expected:
                failures.append(
                    f"{bom_type}/{field}.json: description drifted from SPDX "
                    f"3.0.1 `{spdx_name}` — re-run "
                    f"`python tools/sync_question_bank_descriptions.py --apply`"
                )
        assert not failures, "\n".join(failures)


class TestInternalFieldsAreOnlyTheKnownOnes:
    """Lock the small set of AIkaBoOM-internal fields so adding a new one
    is an explicit decision the test forces a reviewer to make."""

    EXPECTED_INTERNAL = {
        ("ai", "trainedOnDatasets"),
        ("ai", "testedOnDatasets"),
        ("ai", "modelLineage"),
        ("data", "sourceInfo"),
    }

    def test_only_known_fields_are_internal(self, index):
        actual = {
            (bom_type, field)
            for bom_type, field, entry in _entries()
            if entry.get("aikaboom_internal") is True
        }
        assert actual == self.EXPECTED_INTERNAL, (
            f"unexpected internal: {actual - self.EXPECTED_INTERNAL}; "
            f"missing: {self.EXPECTED_INTERNAL - actual}"
        )
