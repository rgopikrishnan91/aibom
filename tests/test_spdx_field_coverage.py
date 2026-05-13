"""Field-coverage regression: every populated provenance field reaches SPDX.

Before this PR the SPDX builder looked up RAG fields by snake_case keys
(``intended_use``, ``training_information``, …) while the actual BOM
emitted camelCase IDs (``informationAboutApplication``,
``informationAboutTraining``, …). Every multi-word field silently
dropped out of the SPDX export, leaving only ``ai_domain`` in the
AI Package.

These tests pin the camelCase mapping and the snake_case fallback so
the export can't regress in either direction.
"""
from __future__ import annotations

import pytest

from aikaboom.utils.spdx_validator import SPDXValidator


def _ai_pkg(spdx: dict) -> dict:
    """Pull the ai_AIPackage element out of an SPDX 3.0.1 graph."""
    graph = spdx.get("@graph") or spdx.get("elements") or []
    return next(
        e for e in graph if isinstance(e, dict) and e.get("type") == "ai_AIPackage"
    )


def _full_camelcase_ai_bom() -> dict:
    """BOM in the format the live pipeline emits — camelCase RAG keys."""
    def t(value):
        return {"value": value, "source": "huggingface", "conflict": None}

    return {
        "repo_id": "google-bert/bert-base-uncased",
        "model_id": "bert-base-uncased",
        "direct_fields": {
            "releaseTime":      t("2024-01-01T00:00:00Z"),
            "downloadLocation": t("https://huggingface.co/google-bert/bert-base-uncased"),
            "packageVersion":   t("1.0"),
            "license":          t("Apache-2.0"),
        },
        "rag_fields": {
            "autonomyType":                     t("noAssertion"),
            "domain":                           t("natural language processing"),
            "energyConsumption":                t("100 kWh"),
            "hyperparameter":                   t("learning_rate=3e-4; batch_size=64"),
            "informationAboutApplication":      t("Used for fine-tuning on NLP tasks."),
            "informationAboutTraining":         t("Trained on BookCorpus and Wikipedia."),
            "limitation":                       t("English-only; not for production safety-critical use."),
            "metric":                           t("accuracy=0.85; f1=0.82"),
            "metricDecisionThreshold":          t("0.5"),
            "modelDataPreprocessing":           t("lowercase; wordpiece tokenization"),
            "modelExplainability":              t("attention visualization"),
            "safetyRiskAssessment":             t("low"),
            "standardCompliance":               t("ISO/IEC 23053:2022"),
            "typeOfModel":                      t("transformer"),
            "useSensitivePersonalInformation":  t("no"),
        },
    }


def test_camelcase_rag_fields_all_reach_spdx_ai_package():
    """The live pipeline emits camelCase RAG keys. Every populated field
    must surface as the matching ai_* property in the AIPackage."""
    spdx = SPDXValidator(bom_type="ai").validate_and_convert(_full_camelcase_ai_bom())
    ai = _ai_pkg(spdx)

    expected_ai_props = {
        "ai_autonomyType",
        "ai_domain",
        "ai_energyConsumption",
        "ai_hyperparameter",
        "ai_informationAboutApplication",
        "ai_informationAboutTraining",
        "ai_limitation",
        "ai_metric",
        "ai_metricDecisionThreshold",
        "ai_modelDataPreprocessing",
        "ai_modelExplainability",
        "ai_safetyRiskAssessment",
        "ai_standardCompliance",
        "ai_typeOfModel",
        "ai_useSensitivePersonalInformation",
    }
    missing = expected_ai_props - set(ai.keys())
    assert not missing, (
        f"AIPackage is missing {len(missing)} ai_* properties: {sorted(missing)}\n"
        f"present ai_* properties: {sorted(k for k in ai if k.startswith('ai_'))}"
    )


def test_camelcase_scalar_values_round_trip():
    """Scalar text fields should arrive verbatim, not normalized away."""
    spdx = SPDXValidator(bom_type="ai").validate_and_convert(_full_camelcase_ai_bom())
    ai = _ai_pkg(spdx)
    assert ai["ai_informationAboutApplication"].startswith("Used for fine-tuning")
    assert "BookCorpus" in ai["ai_informationAboutTraining"]
    assert "English-only" in ai["ai_limitation"]
    assert ai["ai_energyConsumption"] == "100 kWh"


def test_camelcase_enum_values_normalize():
    """Enum-shaped fields (autonomy, sensitive PII, safety) go through
    _normalize_enum and end up as one of the allowed literals."""
    spdx = SPDXValidator(bom_type="ai").validate_and_convert(_full_camelcase_ai_bom())
    ai = _ai_pkg(spdx)
    assert ai["ai_autonomyType"] == "noAssertion"
    assert ai["ai_useSensitivePersonalInformation"] == "no"
    assert ai["ai_safetyRiskAssessment"] == "low"


def test_noassertion_fields_emit_canonical_sentinel():
    """A BOM where every optional field is ``noAssertion`` should still
    produce a complete AIPackage — losing the field hides the audit
    signal that "we asked, the source had nothing to say"."""
    def t():
        return {"value": "noAssertion", "source": "huggingface", "conflict": None}

    bom = {
        "repo_id": "x/y",
        "model_id": "y",
        "direct_fields": {"license": t()},
        "rag_fields": {
            k: t() for k in (
                "autonomyType domain energyConsumption hyperparameter "
                "informationAboutApplication informationAboutTraining limitation "
                "metric metricDecisionThreshold modelDataPreprocessing "
                "modelExplainability safetyRiskAssessment standardCompliance "
                "typeOfModel useSensitivePersonalInformation"
            ).split()
        },
    }
    ai = _ai_pkg(SPDXValidator(bom_type="ai").validate_and_convert(bom))

    # All 15 ai_* properties should be present.
    ai_props = sorted(k for k in ai if k.startswith("ai_"))
    assert len(ai_props) == 15, ai_props
    # Each should carry a noAssertion sentinel of some kind.
    for prop in ai_props:
        v = ai[prop]
        flat = str(v).lower()
        assert "noassertion" in flat, f"{prop} did not carry a noAssertion sentinel: {v!r}"


def test_snake_case_aliases_still_work():
    """Legacy callers that hand-rolled BOMs in snake_case keep working.

    The in-tree pipeline has used camelCase for a long time but external
    scripts and notebooks shipped before the rename may still build BOMs
    with snake_case keys. Both should map cleanly.
    """
    def t(v):
        return {"value": v, "source": "hf", "conflict": None}

    bom = {
        "repo_id": "x/y",
        "rag_fields": {
            "intended_use":                      t("Application X"),
            "training_information":              t("Trained on Y"),
            "limitations":                       t("Z"),
            "hyperparameters":                   t("lr=1e-4"),
            "performance_metrics":               t("acc=0.9"),
            "decision_threshold":                t("0.5"),
            "data_preprocessing":                t("tokenize"),
            "model_explainability":              t("attention"),
            "standard_compliance":               t("ISO X"),
            "model_type":                        t("transformer"),
            "autonomy_type":                     t("noAssertion"),
            "sensitive_personal_information":    t("no"),
            "safety_risk_assessment":            t("low"),
            "energy_consumption":                t("50 kWh"),
        },
    }
    ai = _ai_pkg(SPDXValidator(bom_type="ai").validate_and_convert(bom))

    assert ai["ai_informationAboutApplication"] == "Application X"
    assert ai["ai_informationAboutTraining"]    == "Trained on Y"
    assert ai["ai_limitation"]                  == "Z"
    assert "lr=1e-4" in str(ai["ai_hyperparameter"])
    assert "acc=0.9" in str(ai["ai_metric"])
    assert "0.5" in str(ai["ai_metricDecisionThreshold"])
    assert ai["ai_modelDataPreprocessing"]      == ["tokenize"]
    assert ai["ai_modelExplainability"]         == ["attention"]
    assert ai["ai_standardCompliance"]          == ["ISO X"]
    assert ai["ai_typeOfModel"]                 == ["transformer"]
    assert ai["ai_autonomyType"]                == "noAssertion"
    assert ai["ai_useSensitivePersonalInformation"] == "no"
    assert ai["ai_safetyRiskAssessment"]        == "low"
    assert ai["ai_energyConsumption"]           == "50 kWh"


def test_partial_rag_fields_only_emit_what_is_present():
    """A BOM with only a subset of optional fields must emit exactly
    those fields (no phantom defaults, no missing fields)."""
    def t(v):
        return {"value": v, "source": "hf", "conflict": None}

    bom = {
        "repo_id": "x/y",
        "rag_fields": {
            "domain":                t("computer vision"),
            "limitation":            t("Bench-tested only."),
            "useSensitivePersonalInformation": t("no"),
            # The other 12 fields are deliberately absent.
        },
    }
    ai = _ai_pkg(SPDXValidator(bom_type="ai").validate_and_convert(bom))
    ai_props = {k for k in ai if k.startswith("ai_")}
    # We expect exactly the three fields we set.
    assert ai_props == {"ai_domain", "ai_limitation", "ai_useSensitivePersonalInformation"}, (
        f"unexpected ai_* properties: {sorted(ai_props)}"
    )
