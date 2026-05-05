"""Tests for the per-field JSON question bank under
``src/aikaboom/question_bank/``."""
import json
import os

import pytest

from aikaboom.utils import question_bank as qb


_QB_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "src", "aikaboom", "question_bank",
)


def _list_field_files(bom_type):
    folder = os.path.join(_QB_ROOT, bom_type)
    return sorted(
        f[:-5] for f in os.listdir(folder)
        if f.endswith(".json")
    )


def _read_raw(bom_type, field):
    path = os.path.join(_QB_ROOT, bom_type, f"{field}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestPerFieldFiles:
    """Every expected field has a JSON file with the required keys."""

    EXPECTED_AI = {
        "autonomyType", "domain", "energyConsumption", "hyperparameter",
        "informationAboutApplication", "informationAboutTraining",
        "limitation", "metric", "metricDecisionThreshold",
        "modelDataPreprocessing", "modelExplainability",
        "safetyRiskAssessment", "standardCompliance", "typeOfModel",
        "useSensitivePersonalInformation", "trainedOnDatasets",
        "testedOnDatasets", "modelLineage", "license", "primaryPurpose",
    }
    EXPECTED_DATA = {
        "anonymizationMethodUsed", "confidentialityLevel",
        "dataCollectionProcess", "dataPreprocessing",
        "datasetAvailability", "datasetNoise", "datasetSize",
        "datasetType", "datasetUpdateMechanism",
        "hasSensitivePersonalInformation", "intendedUse", "knownBias",
        "sensorUsed", "license", "primaryPurpose", "description",
        "sourceInfo",
    }

    def test_ai_files_present(self):
        actual = set(_list_field_files("ai"))
        assert actual == self.EXPECTED_AI, (
            f"missing: {self.EXPECTED_AI - actual}; "
            f"unexpected: {actual - self.EXPECTED_AI}"
        )

    def test_data_files_present(self):
        actual = set(_list_field_files("data"))
        assert actual == self.EXPECTED_DATA, (
            f"missing: {self.EXPECTED_DATA - actual}; "
            f"unexpected: {actual - self.EXPECTED_DATA}"
        )

    @pytest.mark.parametrize("bom_type,field", [
        (b, f) for b, fields in [("ai", EXPECTED_AI), ("data", EXPECTED_DATA)]
        for f in fields
    ])
    def test_each_entry_has_required_keys(self, bom_type, field):
        raw = _read_raw(bom_type, field)
        # Schema: every file declares its own field, bom_type, question,
        # keywords, description (post_process is optional).
        assert raw["field"] == field
        assert raw["bom_type"] == bom_type
        for key in ("question", "keywords", "description"):
            assert key in raw and isinstance(raw[key], str), (
                f"{bom_type}/{field}.json: missing/non-str '{key}'"
            )
        # Priority must NOT be in the JSON — it lives in source_priority.json.
        assert "priority" not in raw, (
            f"{bom_type}/{field}.json: priority must live in source_priority.json"
        )


class TestLoader:
    """`load_question_bank` and `load_with_priorities` produce dicts in
    the legacy `FIXED_QUESTIONS_*` shape so consumers don't notice the
    move."""

    def test_load_question_bank_ai(self):
        bank = qb.load_question_bank("ai")
        assert "license" in bank
        entry = bank["license"]
        assert entry["question"].startswith("Under what license")
        assert entry["post_process"] == "normalize_license"
        # Priority is empty until overlay runs.
        assert entry["priority"] == []

    def test_load_with_priorities_overlays(self):
        bank = qb.load_with_priorities("ai")
        assert bank["license"]["priority"] == [
            "huggingface", "github", "arxiv",
        ]
        assert bank["limitation"]["priority"][0] == "arxiv"

    def test_data_bank_has_expected_post_processors(self):
        bank = qb.load_with_priorities("data")
        # Per the design: only license / description / sourceInfo carry a
        # post_process; primaryPurpose / datasetAvailability are
        # human-readable in the Provenance BOM.
        assert bank["license"]["post_process"] == "normalize_license"
        assert bank["description"]["post_process"] == "collapse_whitespace"
        assert bank["sourceInfo"]["post_process"] == "dedupe_named_entities"
        assert bank["primaryPurpose"]["post_process"] is None
        assert bank["datasetAvailability"]["post_process"] is None

    def test_unknown_bom_type_returns_empty(self):
        assert qb.load_question_bank("not_a_real_bom_type") == {}


class TestMalformedEntry:
    """Loader skips a malformed file with a warning rather than
    crashing the whole bank."""

    def test_missing_required_key_skipped(self, tmp_path, monkeypatch, capsys):
        # Build a minimal fake bank under tmp_path mirroring the package
        # layout so we can substitute it via importlib.resources.
        bom_folder = tmp_path / "ai"
        bom_folder.mkdir()
        (bom_folder / "good.json").write_text(json.dumps({
            "field": "good", "bom_type": "ai",
            "question": "?", "keywords": "k", "description": "d",
        }))
        (bom_folder / "bad.json").write_text(json.dumps({
            "field": "bad",  # missing question/keywords/description
        }))
        # Stub _entry_files to return our temp files.
        from pathlib import Path as P
        class _Ref:
            def __init__(self, p): self._p = p; self.name = p.name
            def open(self, encoding=None): return open(self._p, encoding=encoding)
        monkeypatch.setattr(
            qb, "_entry_files",
            lambda bt: [_Ref(p) for p in sorted((tmp_path / bt).glob("*.json"))]
            if bt == "ai" else [],
        )
        bank = qb.load_question_bank("ai")
        assert "good" in bank
        assert "bad" not in bank
        warning = capsys.readouterr().err
        assert "bad.json" in warning


class TestQuestionBankShipsInWheel:
    """Bundled JSON files must be reachable via importlib.resources so
    `pip install` of the package picks them up. Locks
    pyproject.toml/setup.py package_data."""

    def test_files_resolve_via_importlib_resources(self):
        from importlib import resources
        ai_lic = resources.files("aikaboom.question_bank").joinpath(
            "ai", "license.json",
        )
        with resources.as_file(ai_lic) as p:
            data = json.loads(open(p, encoding="utf-8").read())
            assert data["field"] == "license"
            assert data["post_process"] == "normalize_license"
