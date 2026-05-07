"""Phase 8 — fixes from real-user testing on Mistral-7B-v0.1.

Each test corresponds to a finding in the user's testing report:

  #1  ``--mode direct`` crashed because ``DirectLLM.process_ai_model``
      didn't accept the ``structured_chunks`` kwarg the orchestrator
      passes unconditionally.
  #2  HF/GH metadata was only fetched when a token env var was set;
      anonymous access works for public repos and the constructor was
      gating on a token unnecessarily.
  #4  ``extract_repo_id_from_hf_url`` mishandled the short-form
      ``/datasets/<name>`` URL and returned ``"datasets/<name>"``.
  #5  ``--use-case license`` was silently ignored from the CLI.
  #6  429 / rate-limit errors weren't classified as transient.
  #7  ``_normalize_timestamp`` rejected ``+00:00`` offsets and silently
      replaced real timestamps with the wall-clock now.
  #10 End-of-run summary helper bins fields as populated vs missing.
  #15 Per-BOM-type ``QUESTION_TO_KEY`` lookup so data-BOM RAG fields
      no longer all collapse to ``"unknown"``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# #1 — DirectLLM signature parity
# ---------------------------------------------------------------------------


def test_directllm_process_ai_model_accepts_structured_chunks():
    """``AIBOMProcessor.process_ai_model`` always passes ``structured_chunks``
    through to the inner processor. The Direct-LLM backend must accept and
    ignore it; previously it raised ``TypeError`` so ``--mode direct`` was
    completely broken."""
    from aikaboom.core.agentic_rag import DirectLLM
    import inspect
    sig = inspect.signature(DirectLLM.process_ai_model)
    # Either an explicit kwarg or **kwargs is fine.
    has_chunks = "structured_chunks" in sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_chunks or has_var_kw, sig

    sig_d = inspect.signature(DirectLLM.process_dataset)
    has_chunks_d = "structured_chunks" in sig_d.parameters
    has_var_kw_d = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_d.parameters.values()
    )
    assert has_chunks_d or has_var_kw_d, sig_d


# ---------------------------------------------------------------------------
# #4 — short-form HF dataset URL
# ---------------------------------------------------------------------------


def test_short_form_dataset_url_returns_bare_name():
    from aikaboom.utils.metadata_fetcher import MetadataFetcher

    out = MetadataFetcher.extract_repo_id_from_hf_url(
        "https://huggingface.co/datasets/squad"
    )
    assert out == "squad"


def test_namespaced_dataset_url_keeps_slash():
    from aikaboom.utils.metadata_fetcher import MetadataFetcher

    out = MetadataFetcher.extract_repo_id_from_hf_url(
        "https://huggingface.co/datasets/rajpurkar/squad"
    )
    assert out == "rajpurkar/squad"


def test_model_url_returns_namespace_slash_repo():
    from aikaboom.utils.metadata_fetcher import MetadataFetcher

    out = MetadataFetcher.extract_repo_id_from_hf_url(
        "https://huggingface.co/mistralai/Mistral-7B-v0.1"
    )
    assert out == "mistralai/Mistral-7B-v0.1"


def test_invalid_url_returns_none():
    from aikaboom.utils.metadata_fetcher import MetadataFetcher

    assert MetadataFetcher.extract_repo_id_from_hf_url("") is None
    assert MetadataFetcher.extract_repo_id_from_hf_url(None) is None


# ---------------------------------------------------------------------------
# #5 — shared use-case filter
# ---------------------------------------------------------------------------


def test_use_case_filter_license_keeps_only_license_fields():
    from aikaboom.utils.use_case import filter_questions_by_use_case

    full_ai = {
        "license": {"q": "..."},
        "standardCompliance": {"q": "..."},
        "domain": {"q": "..."},
        "limitation": {"q": "..."},
    }
    filtered = filter_questions_by_use_case("license", "ai", full_ai)
    assert set(filtered.keys()) == {"license", "standardCompliance"}


def test_use_case_filter_complete_returns_full():
    from aikaboom.utils.use_case import filter_questions_by_use_case

    full = {"a": 1, "b": 2}
    assert filter_questions_by_use_case("complete", "ai", full) == full


def test_use_case_filter_unknown_falls_back_to_complete():
    from aikaboom.utils.use_case import filter_questions_by_use_case

    full = {"a": 1, "b": 2}
    assert filter_questions_by_use_case("nonsense", "ai", full) == full


def test_use_case_filter_data_bom_has_distinct_presets():
    """The data presets reference data fields, not AI fields. A 'license'
    preset for a data BOM should keep just the data 'license' question."""
    from aikaboom.utils.use_case import filter_questions_by_use_case

    full_data = {
        "license": {"q": "data license question"},
        "intendedUse": {"q": "..."},
        "knownBias": {"q": "..."},
    }
    filtered = filter_questions_by_use_case("license", "data", full_data)
    assert set(filtered.keys()) == {"license"}


# ---------------------------------------------------------------------------
# #6 — 429 / rate-limit retry
# ---------------------------------------------------------------------------


def _make_429(message="HTTP 429: Too Many Requests"):
    """Build an exception whose str() matches the rate-limit detector."""
    return RuntimeError(message)


def test_invoke_with_retry_treats_429_as_transient():
    from aikaboom.core.agentic_rag import _invoke_with_retry

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_429("Error code: 429 - Too Many Requests")
        return "ok"

    with patch("aikaboom.core.agentic_rag.time.sleep") as mock_sleep:
        out = _invoke_with_retry(fn, max_retries=4, initial_delay=0.5)
    assert out == "ok"
    assert calls["n"] == 3
    assert mock_sleep.call_count >= 2


def test_invoke_with_retry_honours_x_ratelimit_reset_seconds():
    """If the caught exception has ``response.headers['X-RateLimit-Reset']``
    set to a small seconds-from-now value, the retry sleep should match it
    instead of using the exponential default."""
    import time as time_mod
    from aikaboom.core.agentic_rag import _retry_sleep_seconds

    class FakeResp:
        headers = {"X-RateLimit-Reset": str(int(time_mod.time()) + 5)}

    err = RuntimeError("rate limit")
    err.response = FakeResp()
    delay = _retry_sleep_seconds(err, attempt=0, initial_delay=2.0)
    # Within ±2s of the requested 5s offset
    assert 3.0 <= delay <= 7.0, delay


def test_invoke_with_retry_caps_excessive_reset_header():
    """A misformatted reset header that would imply an hour-long sleep is
    capped to 60s so a worker can't hang."""
    from aikaboom.core.agentic_rag import _retry_sleep_seconds

    class FakeResp:
        headers = {"X-RateLimit-Reset": "9999999999"}  # year 2286

    err = RuntimeError("rate limit")
    err.response = FakeResp()
    delay = _retry_sleep_seconds(err, attempt=0, initial_delay=2.0)
    assert delay <= 60.0


# ---------------------------------------------------------------------------
# #7 — fromisoformat timestamp
# ---------------------------------------------------------------------------


def test_normalize_timestamp_accepts_offset_iso():
    """``2025-07-24T16:44:02+00:00`` (HF's ``last_modified`` shape) must
    round-trip to the equivalent ``Z``-suffixed instant, not get silently
    replaced by ``_get_current_timestamp``."""
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    out = v._normalize_timestamp("2025-07-24T16:44:02+00:00")
    assert out == "2025-07-24T16:44:02Z"


def test_normalize_timestamp_accepts_z_suffix():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    assert v._normalize_timestamp("2025-07-24T16:44:02Z") == "2025-07-24T16:44:02Z"


def test_normalize_timestamp_accepts_bare_date():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    assert v._normalize_timestamp("2025-07-24") == "2025-07-24T00:00:00Z"


def test_normalize_timestamp_offset_other_than_zero_converts_to_z():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    # 2025-07-24T18:44:02+02:00 is the same instant as 16:44:02Z
    assert v._normalize_timestamp("2025-07-24T18:44:02+02:00") == "2025-07-24T16:44:02Z"


def test_normalize_timestamp_garbage_falls_back():
    from aikaboom.utils.spdx_validator import SPDXValidator

    v = SPDXValidator(bom_type="ai")
    out = v._normalize_timestamp("not a timestamp", default="2020-01-01T00:00:00Z")
    assert out == "2020-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# #10 — end-of-run summary
# ---------------------------------------------------------------------------


def test_summarise_rag_fields_classifies_populated_vs_missing():
    from aikaboom.cli import _summarise_rag_fields

    bom = {
        "rag_fields": {
            "license": {"value": "Apache-2.0"},
            "intendedUse": {"value": "Open-source LLM."},
            "knownBias": {"value": "noAssertion"},
            "datasetNoise": {"value": "Not found."},
            "primaryPurpose": {"value": ""},
            "datasetType": {"value": None},
        },
    }
    populated, missing = _summarise_rag_fields(bom)
    assert set(populated) == {"license", "intendedUse"}
    assert set(missing) == {"knownBias", "datasetNoise", "primaryPurpose", "datasetType"}


def test_summarise_handles_missing_rag_section():
    from aikaboom.cli import _summarise_rag_fields

    populated, missing = _summarise_rag_fields({})
    assert populated == []
    assert missing == []


# ---------------------------------------------------------------------------
# #15 — per-bom-type QUESTION_TO_KEY
# ---------------------------------------------------------------------------


def test_question_key_map_distinguishes_ai_and_data():
    """A data-BOM run must resolve data question texts back to their data
    field keys; previously the AI-only map silently routed every data
    question to ``'unknown'`` so every dataset BOM had ``rag_fields ==
    {'unknown': ...}`` (Finding #15)."""
    from aikaboom.core.agentic_rag import (
        FIXED_QUESTIONS_AI,
        FIXED_QUESTIONS_DATA,
        _question_key_map,
    )

    ai_map = _question_key_map("ai")
    data_map = _question_key_map("data")

    # Both maps are non-empty and have the right number of entries.
    assert len(ai_map) == len(FIXED_QUESTIONS_AI)
    assert len(data_map) == len(FIXED_QUESTIONS_DATA)

    # Pick one AI-only key and one data-only key (both bom types have a
    # 'license' field, but with different question texts).
    sample_ai_key = next(iter(FIXED_QUESTIONS_AI.keys()))
    sample_ai_q = FIXED_QUESTIONS_AI[sample_ai_key]["question"]
    assert ai_map[sample_ai_q] == sample_ai_key

    sample_data_key = next(iter(FIXED_QUESTIONS_DATA.keys()))
    sample_data_q = FIXED_QUESTIONS_DATA[sample_data_key]["question"]
    assert data_map[sample_data_q] == sample_data_key
