"""Tests for the community-editable source-priority config."""
import json

import pytest

from aikaboom.core.source_handler import SourceHandler
from aikaboom.utils import source_priority as sp


@pytest.fixture(autouse=True)
def _flush_cache():
    """Each test sees a clean loader state regardless of prior overrides."""
    sp.set_source_priority_path(None)
    yield
    sp.set_source_priority_path(None)


class TestBundledConfig:
    def test_config_parses_and_has_required_sections(self):
        cfg = sp.load_source_priority()
        assert "direct_fields" in cfg
        assert "rag_fields_ai" in cfg
        assert "rag_fields_data" in cfg
        assert isinstance(cfg["direct_fields"]["default"], list)
        assert isinstance(cfg["rag_fields_ai"]["default"], list)

    def test_config_covers_every_question_bank_entry(self):
        """The shipped source-priority config must declare a priority
        for every field that has a question-bank JSON file. After moving
        the question bank to JSON, the source-priority config remains
        the single place priorities are declared — every JSON-declared
        field needs an entry here."""
        from aikaboom.utils.question_bank import load_question_bank

        cfg = sp.load_source_priority()
        for field in load_question_bank("ai"):
            assert field in cfg["rag_fields_ai"], (
                f"rag_fields_ai.{field}: question-bank JSON exists but no "
                f"config priority is declared"
            )
        for field in load_question_bank("data"):
            assert field in cfg["rag_fields_data"], (
                f"rag_fields_data.{field}: question-bank JSON exists but no "
                f"config priority is declared"
            )

    def test_get_direct_priority_returns_default_for_unknown_field(self):
        result = sp.get_direct_priority("non_existent_field")
        assert result == sp.load_source_priority()["direct_fields"]["default"]

    def test_get_rag_priority_per_bom_type_default(self):
        ai_default = sp.load_source_priority()["rag_fields_ai"]["default"]
        data_default = sp.load_source_priority()["rag_fields_data"]["default"]
        assert sp.get_rag_priority("nope", bom_type="ai") == ai_default
        assert sp.get_rag_priority("nope", bom_type="data") == data_default
        # 'dataset' is an alias for 'data'
        assert sp.get_rag_priority("nope", bom_type="dataset") == data_default

    def test_get_rag_priority_returns_field_specific_when_set(self):
        # 'limitation' prefers arxiv per the shipped config
        assert sp.get_rag_priority("limitation", bom_type="ai")[0] == "arxiv"

    def test_config_is_canonical_design_choice(self):
        """Lock the runtime behaviour to the bundled config exactly. Every
        explicit entry in `source_priority.json` must be returned verbatim
        by the loader; no implicit override / fallback path may quietly
        return something else.

        Priorities are a working-group design choice that the config
        exposes for community editing — this test catches anyone who
        accidentally introduces a hard-coded override."""
        import os
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aikaboom", "config", "source_priority.json",
        )
        with open(cfg_path, encoding="utf-8") as f:
            shipped = json.load(f)

        for field, expected in shipped.get("direct_fields", {}).items():
            if field == "default":
                continue
            assert sp.get_direct_priority(field) == expected, (
                f"direct_fields.{field}: code returned {sp.get_direct_priority(field)} "
                f"but config says {expected}"
            )
        for field, expected in shipped.get("rag_fields_ai", {}).items():
            if field == "default":
                continue
            assert sp.get_rag_priority(field, bom_type="ai") == expected, (
                f"rag_fields_ai.{field}: code returned "
                f"{sp.get_rag_priority(field, bom_type='ai')} but config says {expected}"
            )
        for field, expected in shipped.get("rag_fields_data", {}).items():
            if field == "default":
                continue
            assert sp.get_rag_priority(field, bom_type="data") == expected, (
                f"rag_fields_data.{field}: code returned "
                f"{sp.get_rag_priority(field, bom_type='data')} but config says {expected}"
            )


class TestUserOverride:
    def test_env_var_override_changes_priority(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.json"
        custom.write_text(
            json.dumps({"direct_fields": {"license": ["github", "huggingface"]}})
        )
        monkeypatch.setenv("AIKABOOM_SOURCE_PRIORITY", str(custom))
        sp.set_source_priority_path(None)  # flush cache so env var is re-read

        assert sp.get_direct_priority("license") == ["github", "huggingface"]
        # Other fields fall back to the bundled defaults.
        assert sp.get_direct_priority("suppliedBy") == ["huggingface", "github"]

    def test_explicit_path_pin_takes_precedence(self, tmp_path):
        custom = tmp_path / "custom.json"
        custom.write_text(
            json.dumps({"rag_fields_ai": {"trainedOnDatasets": ["github", "huggingface", "arxiv"]}})
        )
        sp.set_source_priority_path(str(custom))
        assert sp.get_rag_priority("trainedOnDatasets") == ["github", "huggingface", "arxiv"]

    def test_user_config_merges_per_field_over_defaults(self, tmp_path):
        custom = tmp_path / "tiny.json"
        custom.write_text(json.dumps({"rag_fields_ai": {"limitation": ["github"]}}))
        sp.set_source_priority_path(str(custom))
        # Overridden field reflects the user value
        assert sp.get_rag_priority("limitation") == ["github"]
        # Untouched fields keep the bundled value
        assert sp.get_rag_priority("trainedOnDatasets") == [
            "huggingface", "arxiv", "github",
        ]

    def test_invalid_json_falls_back_to_bundled(self, tmp_path, capsys):
        bad = tmp_path / "broken.json"
        bad.write_text("{ this is not json")
        sp.set_source_priority_path(str(bad))
        # Loader logs a warning and uses the bundled defaults.
        assert sp.get_direct_priority("license") == ["huggingface", "github"]
        warning = capsys.readouterr().err
        assert "not valid JSON" in warning

    def test_missing_path_falls_back_to_bundled(self, tmp_path, capsys):
        sp.set_source_priority_path(str(tmp_path / "does-not-exist.json"))
        assert sp.get_rag_priority("trainedOnDatasets") == [
            "huggingface", "arxiv", "github",
        ]
        warning = capsys.readouterr().err
        assert "not found" in warning


class TestSourceHandlerWrapper:
    def test_priority_flips_chosen_value(self):
        hf = {"license": "MIT"}
        gh = {"license": "Apache-2.0"}

        v, src, conflict = SourceHandler.get_field_conflict_with_priority(
            "license", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"],
        )
        assert v == "MIT" and src == "huggingface"
        assert conflict is not None

        v, src, conflict = SourceHandler.get_field_conflict_with_priority(
            "license", {"huggingface": hf, "github": gh},
            priority=["github", "huggingface"],
        )
        assert v == "Apache-2.0" and src == "github"
        assert conflict is not None

    def test_majority_still_wins_over_priority(self):
        """If 2-of-3 sources agree, priority is ignored regardless of order."""
        hf = {"license": "MIT"}
        gh = {"license": "MIT"}
        arxiv = {"license": "Apache-2.0"}
        v, src, _ = SourceHandler.get_field_conflict_with_priority(
            "license",
            {"arxiv": arxiv, "huggingface": hf, "github": gh},
            priority=["arxiv", "huggingface", "github"],
        )
        assert v == "MIT"
        assert src in {"huggingface", "github"}

    def test_none_sources_are_skipped(self):
        v, src, conflict = SourceHandler.get_field_conflict_with_priority(
            "license", {"huggingface": None, "github": {"license": "MIT"}},
            priority=["huggingface", "github"],
        )
        assert v == "MIT" and src == "github" and conflict is None

    def test_unknown_priority_names_are_ignored(self):
        v, src, _ = SourceHandler.get_field_conflict_with_priority(
            "license", {"huggingface": {"license": "MIT"}},
            priority=["arxiv", "huggingface"],
        )
        assert v == "MIT" and src == "huggingface"

    def test_extras_appended_when_priority_partial(self):
        """Sources not listed in priority are appended in dict-iteration order."""
        v, src, _ = SourceHandler.get_field_conflict_with_priority(
            "license",
            {"github": {"license": "Apache-2.0"}, "huggingface": {"license": "MIT"}},
            priority=["arxiv"],  # both real sources are extras
        )
        # github is iterated first in this dict, so it wins the priority tie.
        assert v == "Apache-2.0" and src == "github"
