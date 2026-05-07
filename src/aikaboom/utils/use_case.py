"""Use-case presets and question filtering shared by CLI + web.

A "use case" maps a short tag like ``"safety"`` or ``"license"`` to a
subset of the field-extraction questions. The web UI has had this for
a while; the CLI was missing it (Finding #5 in real-user testing —
``--use-case license`` ran all 20 questions). This module is the single
source of truth so both entry points filter the same way.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional


# ---------------------------------------------------------------------------
# Presets
#
# ``fields=None`` means "all fields" (the ``complete`` use case). Any other
# value is the explicit subset of question keys to keep. Lifted from the
# original definitions in ``aikaboom.web.app`` so behaviour is identical.
# ---------------------------------------------------------------------------

USE_CASE_PRESETS_AI: Dict[str, Dict] = {
    "complete": {
        "label": "Complete AI BOM",
        "description": "Process every inference-backed field plus direct metadata.",
        "fields": None,
    },
    "safety": {
        "label": "Safety & Compliance",
        "description": "Focus on safety risks, standards, limitations, and lifecycle transparency.",
        "fields": [
            "standardCompliance", "limitation", "modelExplainability",
            "informationAboutApplication", "informationAboutTraining",
            "modelDataPreprocessing", "domain",
        ],
    },
    "security": {
        "label": "Security & Risk",
        "description": "Highlight security posture, sensitive data usage, and operational context.",
        "fields": [
            "safetyRiskAssessment", "informationAboutApplication", "domain",
            "useSensitivePersonalInformation", "autonomyType",
        ],
    },
    "lineage": {
        "label": "Model Lineage",
        "description": "Capture training provenance, preprocessing, and supporting context.",
        "fields": [
            "informationAboutTraining", "modelDataPreprocessing",
            "hyperparameter", "metric", "metricDecisionThreshold",
        ],
    },
    "license": {
        "label": "License Compliance",
        "description": "Surface licensing and compliance-adjacent information.",
        "fields": ["standardCompliance", "license"],
    },
}


USE_CASE_PRESETS_DATA: Dict[str, Dict] = {
    "complete": {
        "label": "Complete Data BOM",
        "description": "Process every inference-backed field plus direct metadata.",
        "fields": None,
    },
    "safety": {
        "label": "Safety & Bias",
        "description": "Focus on bias, noise, and data quality aspects.",
        "fields": [
            "knownBias", "datasetNoise", "datasetUpdateMechanism",
            "dataPreprocessing", "dataCollectionProcess", "intendedUse",
        ],
    },
    "security": {
        "label": "Security & Privacy",
        "description": "Focus on intended use, purpose, and anonymization.",
        "fields": ["intendedUse", "primaryPurpose", "anonymizationMethodUsed"],
    },
    "lineage": {
        "label": "Data Lineage",
        "description": "Capture data origin, collection, and preprocessing information.",
        "fields": [
            "datasetAvailability", "dataPreprocessing", "dataCollectionProcess",
            "releaseTime", "originatedBy",
        ],
    },
    "license": {
        "label": "License & Rights",
        "description": "Focus on licensing and usage rights information.",
        "fields": ["license"],
    },
}


def _presets_for(bom_type: str) -> Dict[str, Dict]:
    return USE_CASE_PRESETS_AI if bom_type == "ai" else USE_CASE_PRESETS_DATA


def normalize_use_case(use_case: Optional[str], bom_type: str = "ai") -> str:
    """Map a user-supplied use-case tag to a known preset key, falling back
    to ``"complete"`` when unrecognised."""
    key = (use_case or "complete").strip().lower()
    presets = _presets_for(bom_type)
    return key if key in presets else "complete"


def get_use_case_label(use_case: str, bom_type: str = "ai") -> str:
    """Display label for a use-case key."""
    presets = _presets_for(bom_type)
    return presets.get(use_case, presets["complete"])["label"]


def filter_questions_by_use_case(
    use_case: Optional[str],
    bom_type: str,
    full_config: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Return the subset of ``full_config`` covered by the given use case.

    ``full_config`` is the per-field question configuration (e.g. the output
    of :func:`aikaboom.core.agentic_rag.get_fixed_questions`). When the
    use case is ``"complete"`` (or unrecognised) the full config is
    returned unchanged.
    """
    use_case = normalize_use_case(use_case, bom_type)
    presets = _presets_for(bom_type)
    fields: Optional[Iterable[str]] = presets[use_case]["fields"]
    if fields is None:
        return dict(full_config)
    keep = set(fields)
    return {k: v for k, v in full_config.items() if k in keep}


__all__ = [
    "USE_CASE_PRESETS_AI",
    "USE_CASE_PRESETS_DATA",
    "filter_questions_by_use_case",
    "get_use_case_label",
    "normalize_use_case",
]
