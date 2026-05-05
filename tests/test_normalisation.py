"""Unit tests for the field-comparison normalisers in
:mod:`aikaboom.utils.normalise`.

Each test pins one normaliser's contract so future edits can't regress
the comparison semantics that the direct/RAG pipelines rely on.
"""
from datetime import datetime

import pytest

from aikaboom.utils import normalise as N


class TestNormalizeUrl:
    def test_lowercases_host_and_scheme(self):
        assert N.normalize_url("HTTPS://Github.com/Org/Repo") == "https://github.com/Org/Repo"

    def test_strips_trailing_slash(self):
        assert N.normalize_url("https://huggingface.co/m/x/") == "https://huggingface.co/m/x"

    def test_drops_fragment(self):
        assert N.normalize_url("https://huggingface.co/m/x#install") == "https://huggingface.co/m/x"

    def test_strips_www(self):
        assert N.normalize_url("https://www.huggingface.co/m/x") == "https://huggingface.co/m/x"

    def test_preserves_query_string(self):
        assert N.normalize_url("https://github.com/o/r?branch=main") == "https://github.com/o/r?branch=main"

    def test_empty_input(self):
        assert N.normalize_url("") == ""
        assert N.normalize_url(None) == ""
        assert N.normalize_url(123) == ""


class TestNormalizeVersion:
    def test_strips_leading_v(self):
        assert N.normalize_version("v1.2.3") == "1.2.3"
        assert N.normalize_version("V2.0") == "2.0"

    def test_drops_build_metadata(self):
        assert N.normalize_version("1.2.3+sha.abc") == "1.2.3"
        assert N.normalize_version("v1.0+build5") == "1.0"

    def test_preserves_prerelease(self):
        assert N.normalize_version("1.2.3-rc1") == "1.2.3-rc1"
        assert N.normalize_version("v0.1-alpha") == "0.1-alpha"

    def test_lowercases(self):
        assert N.normalize_version("RELEASE-1.0") == "release-1.0"

    def test_empty_input(self):
        assert N.normalize_version("") == ""
        assert N.normalize_version(None) == ""


class TestNormalizeOrg:
    def test_lowercases_and_strips(self):
        assert N.normalize_org(" Meta ") == "meta"
        assert N.normalize_org("HuggingFace") == "huggingface"

    def test_empty_alias_map_is_no_op(self):
        # Default ships empty.
        assert N.normalize_org("Meta") == "meta"

    def test_explicit_aliases_apply(self):
        aliases = {"meta": "facebookresearch", "google": "googleresearch"}
        assert N.normalize_org("Meta", aliases=aliases) == "facebookresearch"
        assert N.normalize_org("openai", aliases=aliases) == "openai"  # not in map

    def test_empty_input(self):
        assert N.normalize_org(None) == ""
        assert N.normalize_org("") == ""
        assert N.normalize_org("   ") == ""


class TestDateWindowConflict:
    def test_within_window_no_conflict(self):
        c = N.date_window_conflict(
            "2024-08-15", "huggingface",
            [("github", "2024-08-13")], window_days=7,
        )
        assert c is None

    def test_outside_window_flags(self):
        c = N.date_window_conflict(
            "2024-08-15", "huggingface",
            [("github", "2024-01-01")], window_days=7,
        )
        assert c is not None
        assert c["delta_days"] > 200
        assert c["source"] == "github"
        assert c["type"] == "inter"

    def test_picks_largest_delta_when_multiple(self):
        c = N.date_window_conflict(
            "2024-08-15", "huggingface",
            [("github", "2024-08-01"), ("arxiv", "2023-01-01")],
            window_days=7,
        )
        assert c["source"] == "arxiv"

    def test_unparseable_dates_ignored(self):
        c = N.date_window_conflict(
            "2024-08-15", "huggingface",
            [("github", "not-a-date"), ("arxiv", None)],
            window_days=7,
        )
        assert c is None

    def test_chosen_source_excluded_from_runners_up(self):
        c = N.date_window_conflict(
            "2024-08-15", "huggingface",
            [("huggingface", "2020-01-01")],  # same source — ignored
            window_days=7,
        )
        assert c is None

    def test_unparseable_chosen(self):
        c = N.date_window_conflict("not-a-date", "huggingface", [], window_days=7)
        assert c is None


class TestDedupeNamedEntities:
    def test_splits_on_commas_and_semicolons(self):
        assert N.dedupe_named_entities("squad, MMLU; squad") == ["squad", "MMLU"]

    def test_strips_bullets(self):
        assert N.dedupe_named_entities("- squad\n- mmlu") == ["squad", "mmlu"]

    def test_dedupes_case_insensitive_preserves_first_case(self):
        assert N.dedupe_named_entities(["SQuAD", "squad", "Squad"]) == ["SQuAD"]

    def test_drops_placeholders(self):
        assert N.dedupe_named_entities("noAssertion, unknown, n/a, none, squad") == ["squad"]

    def test_empty_inputs(self):
        assert N.dedupe_named_entities(None) == []
        assert N.dedupe_named_entities("") == []
        assert N.dedupe_named_entities([]) == []

    def test_dict_input_uses_values(self):
        assert N.dedupe_named_entities({"a": "x", "b": "y"}) == ["x", "y"]


class TestCollapseWhitespace:
    def test_collapses_runs(self):
        assert N.collapse_whitespace("  a   \n  b ") == "a b"

    def test_none_returns_empty(self):
        assert N.collapse_whitespace(None) == ""


class TestNormalizeLicense:
    def test_alias_canonicalisation(self):
        assert N.normalize_license("mit license") == "MIT"
        assert N.normalize_license("Apache 2.0") == "Apache-2.0"
        assert N.normalize_license("GPLv3") == "GPL-3.0"

    def test_unknown_passes_through_lowercased(self):
        assert N.normalize_license("Custom Whacky") in {"custom whacky"}

    def test_none_returns_empty(self):
        assert N.normalize_license(None) == ""


class TestEnumPostProcessors:
    def test_purpose_normalises_to_spdx_enum(self):
        assert N.normalize_purpose_enum("model") == "model"
        assert N.normalize_purpose_enum("MODEL") == "model"
        assert N.normalize_purpose_enum("not-a-known-purpose") == "other"

    def test_availability_normalises_to_spdx_enum(self):
        assert N.normalize_availability_enum("clickthrough") == "clickthrough"
        assert N.normalize_availability_enum("REGISTRATION") == "registration"
        assert N.normalize_availability_enum("free download") == "directDownload"


class TestCoerceDatasetSizeBytes:
    """Tests for the SPDX emitter's unit-aware byte-count parser."""

    def test_plain_int(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes(12345) == 12345
        assert _coerce_dataset_size_bytes("12345") == 12345

    def test_thousands_separator(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("1,234,567") == 1_234_567
        assert _coerce_dataset_size_bytes("1_234_567") == 1_234_567

    def test_decimal_si_suffixes(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("1 KB") == 1_000
        assert _coerce_dataset_size_bytes("1 MB") == 1_000_000
        assert _coerce_dataset_size_bytes("1.5 GB") == 1_500_000_000
        assert _coerce_dataset_size_bytes("5 TB") == 5_000_000_000_000

    def test_iec_binary_suffixes(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("1 KiB") == 1024
        assert _coerce_dataset_size_bytes("1 MiB") == 1024 ** 2
        assert _coerce_dataset_size_bytes("2 GiB") == 2 * 1024 ** 3

    def test_bare_kmgt_treated_as_decimal(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("5T") == 5_000_000_000_000
        assert _coerce_dataset_size_bytes("100M") == 100_000_000

    def test_non_byte_units_return_none(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        # "examples" / "tokens" / "rows" aren't byte units → noAssertion.
        assert _coerce_dataset_size_bytes("10000 examples") is None
        assert _coerce_dataset_size_bytes("1.5 million tokens") is None

    def test_empty_or_invalid(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("") is None
        assert _coerce_dataset_size_bytes(None) is None
        assert _coerce_dataset_size_bytes("nonsense") is None
        assert _coerce_dataset_size_bytes(True) is None  # bool rejected

    def test_negative_treated_as_invalid(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes("-500 MB") is None

    def test_float_input(self):
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes(1024.5) == 1024  # truncated

    def test_zero_int_is_no_assertion(self):
        """Regression: a literal ``0`` byte count is treated as no-assertion
        (caller omits the property) rather than emitted as ``"0 bytes"``."""
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes(0) is None
        assert _coerce_dataset_size_bytes("0") is None
        assert _coerce_dataset_size_bytes("0 KB") is None

    def test_sub_byte_float_is_no_assertion(self):
        """Regression: a fractional value that rounds to 0 bytes (e.g.
        ``0.5``) is no-assertion, not ``"0 bytes"``."""
        from aikaboom.utils.spdx_validator import _coerce_dataset_size_bytes
        assert _coerce_dataset_size_bytes(0.5) is None
        assert _coerce_dataset_size_bytes(0.999) is None


class TestSPDXCoercesFromHumanReadable:
    """The Provenance BOM keeps the raw human-readable LLM answer; the SPDX
    emitter's _normalize_enum / _normalize_enum_list helpers must turn it
    into the right SPDX enum at export time."""

    def test_purpose_enum_alias(self):
        from aikaboom.utils.spdx_validator import SPDXValidator, _SOFTWARE_PURPOSES
        v = SPDXValidator(bom_type="ai")
        assert v._normalize_enum("model", _SOFTWARE_PURPOSES, "other") == "model"
        assert v._normalize_enum("MODEL", _SOFTWARE_PURPOSES, "other") == "model"
        assert v._normalize_enum("text-generation system", _SOFTWARE_PURPOSES, "other") == "other"

    def test_availability_enum_alias(self):
        from aikaboom.utils.spdx_validator import SPDXValidator, _DATASET_AVAILABILITY_VALUES
        v = SPDXValidator(bom_type="data")
        assert v._normalize_enum(
            "clickthrough", _DATASET_AVAILABILITY_VALUES, "directDownload",
        ) == "clickthrough"
        assert v._normalize_enum(
            "REGISTRATION", _DATASET_AVAILABILITY_VALUES, "directDownload",
        ) == "registration"
        # Unknown phrase falls back to default.
        assert v._normalize_enum(
            "publicly downloadable", _DATASET_AVAILABILITY_VALUES, "directDownload",
        ) == "directDownload"

    def test_dataset_emitter_omits_size_on_no_assertion(self):
        from aikaboom.utils.spdx_validator import SPDXValidator
        v = SPDXValidator(bom_type="data")
        # Free-text answer that doesn't yield bytes → property omitted.
        spdx = v.validate_and_convert({
            "dataset_id": "test/ds",
            "direct_metadata": {"name": "Test"},
            "rag_metadata": {"datasetSize": "10000 examples"},
            "urls": {},
        })
        ds = next(e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage")
        assert "dataset_datasetSize" not in ds

    def test_dataset_emitter_keeps_size_when_parseable(self):
        from aikaboom.utils.spdx_validator import SPDXValidator
        v = SPDXValidator(bom_type="data")
        spdx = v.validate_and_convert({
            "dataset_id": "test/ds",
            "direct_metadata": {"name": "Test"},
            "rag_metadata": {"datasetSize": "1.2 GB"},
            "urls": {},
        })
        ds = next(e for e in spdx["@graph"] if e.get("type") == "dataset_DatasetPackage")
        assert ds["dataset_datasetSize"] == 1_200_000_000


class TestPostProcessorDispatch:
    def test_known_names_resolve(self):
        assert N.get_post_processor("normalize_license") is N.normalize_license
        assert N.get_post_processor("dedupe_named_entities") is N.dedupe_named_entities
        assert N.get_post_processor("normalize_purpose_enum") is N.normalize_purpose_enum
        assert N.get_post_processor("normalize_availability_enum") is N.normalize_availability_enum
        assert N.get_post_processor("collapse_whitespace") is N.collapse_whitespace

    def test_none_or_missing_returns_none(self):
        assert N.get_post_processor(None) is None
        assert N.get_post_processor("") is None

    def test_unknown_name_warns_and_returns_none(self, capsys):
        out = N.get_post_processor("definitely-not-a-real-postprocessor")
        assert out is None
        captured = capsys.readouterr()
        assert "definitely-not-a-real-postprocessor" in (captured.out + captured.err)

    def test_unknown_name_warning_is_deduplicated(self, capsys):
        """Regression: unknown post_process names warn at most once per
        name so a typo doesn't flood stderr when the dispatch is hit
        per-question on every run."""
        # Reset the warned-set so this test is order-independent.
        N._WARNED_UNKNOWN_POST_PROCESSORS.discard("typo-name-once")
        N.get_post_processor("typo-name-once")
        first = capsys.readouterr()
        N.get_post_processor("typo-name-once")
        N.get_post_processor("typo-name-once")
        N.get_post_processor("typo-name-once")
        rest = capsys.readouterr()
        assert "typo-name-once" in (first.out + first.err)
        assert "typo-name-once" not in (rest.out + rest.err)
