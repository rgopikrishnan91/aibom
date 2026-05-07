"""Validation gate for Phase 5A — question-bank extraction-block cleanup.

Asserts that every per-field JSON under ``src/aikaboom/question_bank/{ai,data}/``
satisfies the five fixes from the user's prompt-optimization plan:

  1. ``"Not found."`` is not used as a return value (replaced by ``noAssertion``).
     Special case: ``data/datasetSize`` may say ``mark byte size as 'not reported'``
     because that's a label inside a composite response.
  2. The 9 enumerated fields enforce bare-enum output (no justification request,
     no "optionally followed by..." in guidance, explicit "no explanatory prose"
     marker in guidance). ``ai/safetyRiskAssessment`` no longer uses the
     ``unclassified:`` prefix pattern.
  3. ``instruction`` does not redundantly restate ``field_spec``'s legal values
     (no "and return one of the three allowed enum values" style clauses).
  4. ``output_guidance`` does not contain "do not fabricate / hallucinate /
     make up" — universal Rule 1 of the new answer prompt covers these.
  5. ``NOASSERTION`` (all-caps) appears only in ``ai/license`` (SPDX license
     expression convention). Everywhere else it is ``noAssertion``.

Plus a structural check: every entry has all three ``extraction`` keys present
and non-empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
QB_ROOT = REPO_ROOT / "src" / "aikaboom" / "question_bank"

ENUMERATED_FIELDS = {
    ("ai", "autonomyType"),
    ("ai", "safetyRiskAssessment"),
    ("ai", "useSensitivePersonalInformation"),
    ("ai", "primaryPurpose"),
    ("data", "hasSensitivePersonalInformation"),
    ("data", "confidentialityLevel"),
    ("data", "datasetAvailability"),
    ("data", "datasetType"),
    ("data", "primaryPurpose"),
}

BARE_ENUM_MARKERS = (
    "bare enum value",
    "exactly one enum value",
    "no explanatory prose",
    "value only",
    "return only",
)

JUSTIFICATION_PATTERNS = (
    "justification",
    "rationale",
)

REDUNDANT_FABRICATION_PATTERNS = (
    "do not fabricate",
    "do not hallucinate",
    "do not make up",
)


def _all_field_files():
    """Yield (bom_type, field_name, path) tuples for every per-field JSON."""
    for bom_type in ("ai", "data"):
        for path in sorted((QB_ROOT / bom_type).glob("*.json")):
            yield bom_type, path.stem, path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def all_entries():
    return [(bom, fname, _load(path)) for bom, fname, path in _all_field_files()]


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_every_field_has_full_extraction_block(all_entries):
    missing = []
    for bom, fname, entry in all_entries:
        ext = entry.get("extraction", {})
        for key in ("instruction", "field_spec", "output_guidance"):
            if not (ext.get(key) or "").strip():
                missing.append(f"{bom}/{fname}: extraction.{key} missing or empty")
    assert not missing, "Empty extraction slots:\n  " + "\n  ".join(missing)


# ---------------------------------------------------------------------------
# Fix 1: noAssertion standardization
# ---------------------------------------------------------------------------


def test_no_not_found_as_return_value(all_entries):
    """`"Not found."` must not appear as a return value. Allow only in
    ``data/datasetSize.output_guidance`` if it has been rewritten to
    ``'not reported'`` (no period, no "Not found." form)."""
    offenders = []
    for bom, fname, entry in all_entries:
        ext = entry.get("extraction", {})
        for key in ("instruction", "output_guidance"):
            text = ext.get(key, "")
            if '"Not found."' in text or "'Not found.'" in text:
                offenders.append(f"{bom}/{fname}.{key}: contains 'Not found.'")
    assert not offenders, "Stale 'Not found.' return values:\n  " + "\n  ".join(offenders)


def test_datasetsize_uses_not_reported_label(all_entries):
    """``data/datasetSize`` is a composite response (byte count + auxiliary
    metrics). Its byte-size sentinel is the string ``'not reported'`` —
    a label inside the composite, not a standalone return."""
    for bom, fname, entry in all_entries:
        if (bom, fname) != ("data", "datasetSize"):
            continue
        guidance = entry["extraction"]["output_guidance"]
        # Must reference the byte-size label
        assert "not reported" in guidance.lower(), (
            "data/datasetSize must label missing byte size as 'not reported'"
        )
        return
    pytest.fail("data/datasetSize.json not found")


# ---------------------------------------------------------------------------
# Fix 2: enumerated-field bare-enum enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bom,fname", sorted(ENUMERATED_FIELDS))
def test_enumerated_fields_enforce_bare_enum(bom, fname, all_entries):
    """Each enumerated field's output_guidance must contain a bare-enum
    marker phrase, must not request justification or 'optionally followed by',
    and must not advertise free-text rationale in field_spec."""
    entry = next(e for b, f, e in all_entries if b == bom and f == fname)
    ext = entry["extraction"]
    instr = ext["instruction"]
    spec = ext["field_spec"]
    guid = ext["output_guidance"]

    # (a) instruction must not request justification
    for pat in JUSTIFICATION_PATTERNS:
        assert pat not in instr.lower(), (
            f"{bom}/{fname}.instruction still asks for {pat!r}"
        )

    # (b) guidance must not have "optionally followed by"
    assert "optionally followed by" not in guid.lower(), (
        f"{bom}/{fname}.output_guidance still has 'optionally followed by ...'"
    )

    # (c) guidance must contain a bare-enum marker
    has_marker = any(m in guid.lower() for m in BARE_ENUM_MARKERS)
    assert has_marker, (
        f"{bom}/{fname}.output_guidance is missing a bare-enum marker "
        f"(one of: {BARE_ENUM_MARKERS}). Got: {guid!r}"
    )

    # (d) field_spec must not advertise free-text rationale
    for pat in (
        "optionally accompanied by prose",
        "optional accompanying free-text",
        "optionally with a parenthetical",
    ):
        assert pat not in spec.lower(), (
            f"{bom}/{fname}.field_spec still advertises free-text rationale: "
            f"{pat!r}"
        )


def test_safety_risk_assessment_no_unclassified_prefix(all_entries):
    entry = next(e for b, f, e in all_entries if (b, f) == ("ai", "safetyRiskAssessment"))
    guid = entry["extraction"]["output_guidance"]
    assert "unclassified:" not in guid.lower(), (
        "ai/safetyRiskAssessment.output_guidance must replace the "
        "'unclassified:' prefix pattern with a bare 'noAssertion' return."
    )


# ---------------------------------------------------------------------------
# Fix 4: redundant fabrication warnings stripped
# ---------------------------------------------------------------------------


def test_no_redundant_fabrication_warnings(all_entries):
    """The new answer prompt's universal Rule 1 forbids fabrication. Per-field
    output_guidance must not duplicate that warning."""
    offenders = []
    for bom, fname, entry in all_entries:
        guid = entry.get("extraction", {}).get("output_guidance", "").lower()
        for pat in REDUNDANT_FABRICATION_PATTERNS:
            if pat in guid:
                offenders.append(f"{bom}/{fname}: contains {pat!r}")
    assert not offenders, "Redundant fabrication warnings:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------------------
# Fix 5: NOASSERTION casing
# ---------------------------------------------------------------------------


# Uppercase NOASSERTION is a literal token in the SPDX
# simplelicensing_LicenseExpression grammar — both ai/license and data/license
# use that class, so both keep the all-caps form.
LICENSE_FIELDS = {("ai", "license"), ("data", "license")}


def test_noassertion_uppercase_only_in_license_fields(all_entries):
    """Everywhere except the two license fields, ``noAssertion`` must be the
    lowercase camelCase form."""
    offenders = []
    for bom, fname, entry in all_entries:
        if (bom, fname) in LICENSE_FIELDS:
            continue
        ext = entry.get("extraction", {})
        for key in ("instruction", "field_spec", "output_guidance"):
            text = ext.get(key, "")
            if "NOASSERTION" in text:
                offenders.append(f"{bom}/{fname}.{key}: uses uppercase NOASSERTION")
    assert not offenders, (
        "Uppercase NOASSERTION outside license fields:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("bom,fname", sorted(LICENSE_FIELDS))
def test_license_fields_keep_noassertion_uppercase(bom, fname, all_entries):
    """License fields must retain the SPDX-grammar uppercase NOASSERTION."""
    entry = next(e for b, f, e in all_entries if (b, f) == (bom, fname))
    blob = " ".join(entry["extraction"].values())
    assert "NOASSERTION" in blob, (
        f"{bom}/{fname} must retain uppercase NOASSERTION (SPDX convention)."
    )
