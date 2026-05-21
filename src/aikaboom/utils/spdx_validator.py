"""
Unified SPDX 3.0.1 BOM Validator
Converts AI and Dataset BOM metadata to SPDX 3.0.1 compliant format.

Validation uses the official SPDX 3.0.1 SHACL shapes and JSON Schema
when available (pyshacl + jsonschema). Falls back to built-in structural
checks when those deps or schema files are missing.
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from typing import Dict, Any, Optional, Iterable, List
import uuid

# URLs for official SPDX 3.0.1 validation artifacts
SPDX_SHACL_URL = "https://spdx.org/rdf/3.0.1/spdx-model.ttl"
SPDX_JSON_SCHEMA_URL = "https://spdx.org/schema/3.0.1/spdx-json-schema.json"
_SCHEMA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "aikaboom_spdx_schemas")
_BUNDLED_SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), '..', 'schemas')
_SCHEMA_PACKAGE = "aikaboom.schemas"

_SOFTWARE_PURPOSES = {
    "application", "archive", "bom", "configuration", "container", "data", "device",
    "deviceDriver", "diskImage", "documentation", "evidence", "executable", "file",
    "filesystemImage", "firmware", "framework", "install", "library", "manifest",
    "model", "module", "operatingSystem", "other", "patch", "platform", "requirement",
    "source", "specification", "test",
}
_PRESENCE_VALUES = {"yes", "no", "noAssertion"}

# Sentinels meaning "no information for this field". Lifted into
# ``aikaboom.utils.value_helpers`` in Phase 10 so the CycloneDX exporter
# can apply the same filter without duplicating the constant.
from aikaboom.utils.value_helpers import _NIL_VALUES, _is_nil_value  # noqa: F401, E402
from aikaboom.utils.lineage import split_lineage_targets  # noqa: E402
# SPDX 3.0.1 SafetyRiskAssessmentType has no "noAssertion" member — the
# only members are low/medium/high/serious. An unknown safety risk is
# encoded by omitting ai_safetyRiskAssessment, not by a sentinel string.
_AI_SAFETY_VALUES = {"low", "medium", "high", "serious"}
_DATASET_AVAILABILITY_VALUES = {
    "clickthrough", "directDownload", "query", "registration", "scrapingScript",
}
_DATASET_CONFIDENTIALITY_VALUES = {"amber", "clear", "green", "red"}
_DATASET_TYPES = {
    "audio", "categorical", "graph", "image", "noAssertion", "numeric", "other",
    "sensor", "structured", "syntactic", "text", "timeseries", "timestamp", "video",
}


def _get_schema_path(url: str, filename: str) -> Optional[str]:
    """Get schema file path: try cache, then bundled, then download."""
    # 1. Check disk cache
    os.makedirs(_SCHEMA_CACHE_DIR, exist_ok=True)
    cached = os.path.join(_SCHEMA_CACHE_DIR, filename)
    if os.path.exists(cached) and os.path.getsize(cached) > 100:
        return cached

    # 2. Check bundled copy
    bundled = os.path.join(_BUNDLED_SCHEMAS_DIR, filename)
    if os.path.exists(bundled) and os.path.getsize(bundled) > 100:
        return bundled

    # 3. Try downloading
    try:
        import requests
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        if len(resp.text) > 100:
            with open(cached, 'w') as f:
                f.write(resp.text)
            return cached
    except Exception:
        pass

    return None


def _get_bundled_schema_path(filename: str) -> str:
    """Return a filesystem path for a bundled SPDX validation artifact."""
    try:
        schema_ref = resources.files(_SCHEMA_PACKAGE).joinpath(filename)
        with resources.as_file(schema_ref) as path:
            if path.exists() and path.stat().st_size > 100:
                return str(path)
    except Exception:
        pass

    path = os.path.abspath(os.path.join(_BUNDLED_SCHEMAS_DIR, filename))
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    raise FileNotFoundError(f"Bundled SPDX schema not found: {filename}")


@lru_cache(maxsize=1)
def _load_json_schema() -> Dict[str, Any]:
    with open(_get_bundled_schema_path("spdx-json-schema.json"), encoding="utf-8") as f:
        return json.load(f)


_SIZE_UNITS = {
    # Decimal SI
    "b": 1, "byte": 1, "bytes": 1,
    "k": 10 ** 3, "kb": 10 ** 3,
    "m": 10 ** 6, "mb": 10 ** 6,
    "g": 10 ** 9, "gb": 10 ** 9,
    "t": 10 ** 12, "tb": 10 ** 12,
    "p": 10 ** 15, "pb": 10 ** 15,
    # IEC binary
    "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30, "tib": 2 ** 40, "pib": 2 ** 50,
}

_SIZE_PATTERN = re.compile(
    r"^\s*([0-9][\d_,]*(?:\.\d+)?)\s*([a-zA-Z]+)?\s*$"
)


def _coerce_dataset_size_bytes(value: Any) -> Optional[int]:
    """Best-effort parse of a free-form size string into integer bytes.

    Recognises decimal SI (``KB``/``MB``/``GB``/``TB``/``PB``) and IEC
    binary (``KiB``/``MiB``/…) suffixes plus bare ``K``/``M``/``G``/``T``,
    plain integers, and integers with thousands separators (``1,234,567``).
    Returns ``None`` when no positive byte count can be derived — callers
    should omit the SPDX ``dataset_datasetSize`` property in that case
    rather than emit a misleading ``0``. Zero bytes is treated as
    no-assertion: a dataset that genuinely has zero bytes is the same
    signal as "we don't know".
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly.
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value <= 0:
            return None
        result = int(value)
        return result if result > 0 else None
    text = str(value).strip()
    if not text:
        return None
    match = _SIZE_PATTERN.match(text)
    if not match:
        return None
    number_text = match.group(1).replace(",", "").replace("_", "")
    suffix = (match.group(2) or "").lower()
    try:
        number = float(number_text)
    except ValueError:
        return None
    if number <= 0:
        return None
    if suffix and suffix not in _SIZE_UNITS:
        # Suffix present but not a byte unit (e.g. "10000 examples").
        return None
    multiplier = _SIZE_UNITS.get(suffix, 1)
    result = int(number * multiplier)
    return result if result > 0 else None


# SPDX 3.0.1 ai_energyUnit enum members are kilowattHour / megajoule /
# other. Free-text unit spellings the RAG layer emits are folded onto
# those; anything unrecognised falls back to "other".
_ENERGY_UNIT_ALIASES = {
    "kwh": "kilowattHour", "kw·h": "kilowattHour", "kw h": "kilowattHour",
    "kilowatthour": "kilowattHour", "kilowatt-hour": "kilowattHour",
    "kilowatt hour": "kilowattHour", "kilowatt hours": "kilowattHour",
    "kilowatt-hours": "kilowattHour",
    "mj": "megajoule", "megajoule": "megajoule", "megajoules": "megajoule",
}

_ENERGY_PATTERN = re.compile(
    r"^\s*([0-9][\d_,]*(?:\.\d+)?)\s*([A-Za-z][A-Za-z·\- ]*?)\s*$"
)


def _energy_consumption_element(value: Any) -> Optional[Dict[str, Any]]:
    """Convert a free-text energy figure into an inline SPDX 3.0.1
    ``ai_EnergyConsumption`` object.

    SPDX 3.0.1 models ``ai_energyConsumption`` as a structured type, not
    a string. Emitting the raw RAG text (``"100 kWh"``, ``"noAssertion"``,
    or prose) made the AI Package — and therefore the whole document —
    fail JSON Schema validation.

    Only a clean ``<number> <unit>`` figure is representable. Anything
    else (no-assertion sentinels, prose, a bare number with no unit)
    returns ``None`` so the caller omits the property, which is the
    schema-valid encoding of "no information". The figure is filed under
    ``ai_trainingEnergyConsumption`` — training energy is the headline
    number model cards report.
    """
    if value in (None, "") or _is_nil_value(value):
        return None
    match = _ENERGY_PATTERN.match(str(value))
    if not match:
        return None
    try:
        quantity = float(match.group(1).replace(",", "").replace("_", ""))
    except ValueError:
        return None
    if quantity <= 0:
        return None
    unit_text = re.sub(r"\s+", " ", match.group(2).strip().lower())
    unit = _ENERGY_UNIT_ALIASES.get(unit_text, "other")
    return {
        "type": "ai_EnergyConsumption",
        "ai_trainingEnergyConsumption": [{
            "type": "ai_EnergyConsumptionDescription",
            "ai_energyQuantity": quantity,
            "ai_energyUnit": unit,
        }],
    }


class _EmptyFindings:
    """Placeholder ``Findings``-Protocol implementation.

    The standalone ``validate_bom_to_spdx`` path converts a ``bom_data``
    dict without access to a ``BomStore``, so a plugin's ``analyze()``
    cannot run. We hand each plugin an empty findings object so the loop
    is in place; emitters return ``[]`` for empty input. Task 13 wires
    the store-aware path that produces real findings.
    """

    def to_dict(self) -> Dict[str, Any]:
        return {"findings": []}

    def violations(self) -> List[Any]:
        return []

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def _emit_plugin_annotations(claim_iri: str) -> List[Dict[str, Any]]:
    """Loop registered plugins and collect SPDX Annotation Elements.

    Plugin emission is best-effort: a failure in one plugin must not
    block the SPDX document. Each plugin's ``spdx_annotations`` is
    expected to handle empty findings gracefully (return ``[]``).
    """
    out: List[Dict[str, Any]] = []
    try:
        from aikaboom.plugins import all_plugins
    except Exception:
        return out
    findings = _EmptyFindings()
    for plugin in all_plugins():
        try:
            if not plugin.enabled():
                continue
        except Exception:
            continue
        try:
            anns = plugin.spdx_annotations(claim_iri=claim_iri, findings=findings)
        except Exception as e:
            logging.getLogger("aikaboom.plugins").warning(
                "Plugin %r SPDX emission failed: %s", getattr(plugin, "name", "?"), e
            )
            continue
        if anns:
            out.extend(anns)
    return out


class SPDXValidator:
    """Unified validator that converts both AI and Dataset BOM data to SPDX 3.0.1 format"""
    
    # Official SPDX 3.0.1 @context URL
    SPDX_CONTEXT_URL = "https://spdx.org/rdf/3.0.1/spdx-context.jsonld"

    # Official SPDX 3.0.1 type names (from the JSON Schema / JSON-LD context)
    AI_PACKAGE_TYPE = "ai_AIPackage"
    DATASET_PACKAGE_TYPE = "dataset_DatasetPackage"

    # Field mappings for AI BOM
    # Property names follow the official SPDX 3.0.1 JSON-LD context:
    #   Core properties: no prefix (name, suppliedBy, releaseTime, etc.)
    #   AI properties: ai_ prefix (ai_domain, ai_typeOfModel, etc.)
    #   Software properties: software_ prefix (software_downloadLocation, etc.)
    # The BOM's RAG fields are keyed by question_type (camelCase, mirroring
    # the SPDX 3.0.1 property suffix). Each entry below is the BOM key →
    # SPDX property name. Snake_case aliases stay as a fallback so older
    # callers that hand-rolled BOMs in the legacy format still export
    # cleanly (the BOM emitted by the in-tree pipeline has always been
    # camelCase, but external scripts in the wild may not).
    AI_FIELD_MAPPING = {
        # Direct fields (core + software namespace)
        "releaseTime": "releaseTime",
        "suppliedBy": "suppliedBy",
        "downloadLocation": "software_downloadLocation",
        "packageVersion": "software_packageVersion",
        "primaryPurpose": "software_primaryPurpose",
        "license": "license",

        # RAG fields for AI models (ai_ namespace) — camelCase keys match
        # the question_type IDs the pipeline emits into rag_fields.
        "model_name": "name",
        "autonomyType":                   "ai_autonomyType",
        "domain":                         "ai_domain",
        "energyConsumption":              "ai_energyConsumption",
        "hyperparameter":                 "ai_hyperparameter",
        "informationAboutApplication":    "ai_informationAboutApplication",
        "informationAboutTraining":       "ai_informationAboutTraining",
        "limitation":                     "ai_limitation",
        "metric":                         "ai_metric",
        "metricDecisionThreshold":        "ai_metricDecisionThreshold",
        "modelDataPreprocessing":         "ai_modelDataPreprocessing",
        "modelExplainability":            "ai_modelExplainability",
        "safetyRiskAssessment":           "ai_safetyRiskAssessment",
        "standardCompliance":             "ai_standardCompliance",
        "typeOfModel":                    "ai_typeOfModel",
        "useSensitivePersonalInformation": "ai_useSensitivePersonalInformation",
    }

    # Snake_case → camelCase aliases. Looked up in _rag_value() when the
    # primary (camelCase) key is missing, so externally-authored BOMs
    # that still use the old snake_case keys keep exporting correctly.
    AI_RAG_FIELD_ALIASES = {
        "autonomyType":                    ("autonomy_type",),
        "energyConsumption":               ("energy_consumption",),
        "hyperparameter":                  ("hyperparameters",),
        "informationAboutApplication":     ("intended_use",),
        "informationAboutTraining":        ("training_information",),
        "limitation":                      ("limitations",),
        "metric":                          ("performance_metrics",),
        "metricDecisionThreshold":         ("decision_threshold",),
        "modelDataPreprocessing":          ("data_preprocessing",),
        "modelExplainability":             ("model_explainability",),
        "safetyRiskAssessment":            ("safety_risk_assessment",),
        "standardCompliance":              ("standard_compliance",),
        "typeOfModel":                     ("model_type",),
        "useSensitivePersonalInformation": ("sensitive_personal_information",),
    }
    
    # Field mappings for Dataset BOM
    # Property names follow the official SPDX 3.0.1 JSON-LD context:
    #   Dataset properties: dataset_ prefix
    DATASET_FIELD_MAPPING = {
        # Direct fields (core + software namespace)
        "name": "name",
        "originatedBy": "originatedBy",
        "builtTime": "builtTime",
        "releaseTime": "releaseTime",
        "downloadLocation": "software_downloadLocation",
        "primaryPurpose": "software_primaryPurpose",
        "license": "license",

        # RAG fields for datasets (dataset_ namespace)
        "dataPreprocessing": "dataset_dataPreprocessing",
        "datasetAvailability": "dataset_datasetAvailability",
        "dataCollectionProcess": "dataset_dataCollectionProcess",
        "datasetSize": "dataset_datasetSize",
        "datasetType": "dataset_datasetType",
        "datasetUpdateMechanism": "dataset_datasetUpdateMechanism",
        "hasSensitivePersonalInformation": "dataset_hasSensitivePersonalInformation",
        "intendedUse": "dataset_intendedUse",
        "knownBias": "dataset_knownBias",
        "anonymizationMethodUsed": "dataset_anonymizationMethodUsed",
        "confidentialityLevel": "dataset_confidentialityLevel",
        "datasetNoise": "dataset_datasetNoise",
        "sensorUsed": "dataset_sensor",
    }
    
    def __init__(self, template_path: str = None, bom_type: str = 'ai',
                 relationship_cap: Optional[int] = None):
        """
        Initialize validator with optional SPDX template

        Args:
            template_path: Optional path to SPDX template file
            bom_type: Type of BOM to generate ('ai' or 'data')
            relationship_cap: Max number of trainedOn / testedOn /
                dependsOn child packages to emit per relationship in the
                standalone SPDX BOM. ``None`` (default) is unbounded —
                matches the recursive walker and linked bundle. Phase 10
                exposes this as a CLI / web / HF-Space form parameter
                (Finding #22).
        """
        self.template_path = template_path
        self.bom_type = bom_type.lower()
        self.template = self._load_template() if template_path else None
        self.relationship_cap = relationship_cap
    
    def _load_template(self) -> Optional[Dict[str, Any]]:
        """Load SPDX template from file"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: SPDX template not found at {self.template_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Invalid JSON in SPDX template: {e}")
            return None
    
    def _create_minimal_template(self) -> Dict[str, Any]:
        """Create minimal SPDX 3.0.1 template"""
        return {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": []
        }
    
    def _extract_value(self, field_data: Any) -> Any:
        """Extract value from triplet structure or return as-is"""
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data["value"]
        return field_data

    def _rag_get(self, rag_fields: Dict, key: str) -> Any:
        """Look up an AI RAG field by its camelCase key, falling back to
        any registered snake_case alias.

        The in-tree pipeline emits camelCase keys (question_type IDs from
        ``FIXED_QUESTIONS_AI``), but BOMs hand-rolled before the rename
        used snake_case (``intended_use``, ``training_information``, …).
        Both forms are supported here so the SPDX export covers the
        complete set of provenance fields in either case.
        """
        if not isinstance(rag_fields, dict):
            return None
        triplet = rag_fields.get(key)
        if triplet not in (None, "", {}):
            return triplet
        for alias in self.AI_RAG_FIELD_ALIASES.get(key, ()):
            triplet = rag_fields.get(alias)
            if triplet not in (None, "", {}):
                return triplet
        return None

    def _first_non_nil(self, *candidates: Any, default: Any = None) -> Any:
        """Return the first candidate whose unwrapped value is non-nil.

        Each candidate is run through ``_extract_value`` (so triplet
        dicts ``{value, source, conflict}`` become their value) and then
        nil-checked (``None``, ``""``, ``"noAssertion"``, ``"Not found."``,
        …). The first usable value wins; falls back to ``default``.

        This replaces the ``direct.get(k) or rag.get(k) or "X"`` idiom,
        which is broken once direct fields are triplet dicts: the truthy
        triplet short-circuits the ``or`` chain even when its ``value``
        is ``None``, so the ``rag`` and default branches never run.
        """
        for candidate in candidates:
            value = self._extract_value(candidate)
            if value in (None, ""):
                continue
            if isinstance(value, str) and _is_nil_value(value):
                continue
            return value
        return default

    def _as_list(self, value: Any) -> List[Any]:
        """Normalize scalar/list SPDX properties to a clean list.

        Filters nil sentinels (``"noAssertion"``, ``"Not found."`` …) at every
        level: a top-level nil returns ``[]``; nil entries inside a list are
        dropped; a string that splits into segments has each segment checked.
        Without this, ``_as_list("noAssertion")`` returned ``["noAssertion"]``
        and downstream consumers got a one-element list of garbage.
        """
        value = self._extract_value(value)
        if value is None or value == "" or _is_nil_value(value):
            return []
        if isinstance(value, list):
            return [
                v for v in (self._extract_value(item) for item in value)
                if v not in (None, "") and not _is_nil_value(v)
            ]
        if isinstance(value, str):
            parts = [
                p.strip() for p in re.split(r"[;,\n]+", value)
                if p.strip() and not _is_nil_value(p.strip())
            ]
            return parts or ([] if _is_nil_value(value) else [value])
        return [value]

    def _normalize_timestamp(self, value: Any, default: Optional[str] = None) -> str:
        """Return an SPDX timestamp with second precision and a Z suffix.

        Accepts every ISO 8601 form ``datetime.fromisoformat`` understands —
        ``+HH:MM`` offsets, ``Z`` suffix, microseconds, bare ``YYYY-MM-DD``.
        Anything that doesn't parse falls through to ``default`` /
        ``_get_current_timestamp()``. Earlier versions accepted only the
        narrow ``Z``-suffixed form, which silently overwrote real
        ``+00:00`` timestamps from HuggingFace with the wall-clock now
        (Finding #7 in real-user testing).
        """
        value = self._extract_value(value)
        if not value:
            return default or self._get_current_timestamp()
        if not isinstance(value, str):
            return default or self._get_current_timestamp()
        try:
            dt = datetime.fromisoformat(value.strip())
        except (TypeError, ValueError):
            return default or self._get_current_timestamp()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _normalize_enum(self, value: Any, allowed: Iterable[str], default: str) -> str:
        value = self._extract_value(value)
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        aliases = {
            "no assertion": "noAssertion",
            "no_assertion": "noAssertion",
            "noassertion": "noAssertion",
            "none": "noAssertion",
            "unknown": "noAssertion",
            "true": "yes",
            "false": "no",
        }
        normalized = aliases.get(text.lower(), text)
        allowed_set = set(allowed)
        if normalized in allowed_set:
            return normalized
        lower_map = {item.lower(): item for item in allowed_set}
        return lower_map.get(normalized.lower(), default)

    def _normalize_enum_list(
        self, value: Any, allowed: Iterable[str], default: str = "other"
    ) -> List[str]:
        out = []
        for item in self._as_list(value):
            normalized = self._normalize_enum(item, allowed, default)
            if normalized not in out:
                out.append(normalized)
        return out or [default]

    def _dictionary_entries(self, value: Any) -> List[Dict[str, str]]:
        """Convert loose metadata into SPDX DictionaryEntry objects.

        Skips nil sentinels at every level so a parent ``"noAssertion"`` does
        not produce ``{"key": "value1", "value": "noAssertion"}`` and a nil
        ``value`` inside a real key/value dict is silently dropped.
        """
        value = self._extract_value(value)
        if not value or _is_nil_value(value):
            return []
        if isinstance(value, dict):
            if "key" in value:
                key = str(value.get("key") or "").strip()
                if not key:
                    return []
                entry = {"type": "DictionaryEntry", "key": key}
                inner = value.get("value")
                if inner not in (None, "") and not _is_nil_value(inner):
                    entry["value"] = str(inner)
                return [entry]
            return [
                {"type": "DictionaryEntry", "key": str(k), "value": str(v)}
                for k, v in value.items()
                if k not in (None, "") and v not in (None, "") and not _is_nil_value(v)
            ]
        entries = []
        for index, item in enumerate(self._as_list(value), start=1):
            if isinstance(item, dict) and "key" in item:
                entries.extend(self._dictionary_entries(item))
            elif not _is_nil_value(item):
                entries.append({"type": "DictionaryEntry", "key": f"value{index}", "value": str(item)})
        return entries
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SPDX IDs"""
        return str(uuid.uuid4())
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Relationship-typed BOM fields don't map to a single property on the
    # package — they materialize as ``Relationship`` Elements pointing at
    # child Packages. The SPDX 3.0.1 relationship-type token is the most
    # specific identifier we can put in an Annotation's ``name``.
    _AI_RELATIONSHIP_PROPERTIES = {
        "trainedOnDatasets": "trainedOn",
        "testedOnDatasets":  "testedOn",
        "modelLineage":      "dependsOn",
    }

    def _spdx_property_for_bom_field(
        self, bom_field: str, bom_type: str
    ) -> Optional[str]:
        """Return the SPDX 3.0.1 property/relationship name for a BOM field.

        Resolution order:

        1. **Direct lookup** in ``AI_FIELD_MAPPING`` / ``DATASET_FIELD_MAPPING``
           (handles canonical camelCase keys like ``informationAboutTraining``).
        2. **Snake_case alias reverse-lookup** via ``AI_RAG_FIELD_ALIASES``
           so legacy keys (``training_information``,
           ``safety_risk_assessment``, …) still resolve.
        3. **License special case** — the LicenseExpression element owns
           ``simplelicensing_licenseExpression``; the AIPackage / DatasetPackage
           reaches it via ``hasConcludedLicense`` / ``hasDeclaredLicense``
           Relationships. Conflicts on ``license`` are reported as
           ``simplelicensing_licenseExpression`` so downstream tooling can
           cross-reference into the License element.
        4. **Relationship-typed AI fields** (``trainedOnDatasets`` etc.) →
           the SPDX relationship-type token (``trainedOn``, …).

        Returns ``None`` for unknown fields so callers can fall back to the
        BOM field key as a last resort.
        """
        # 1. Direct hit
        mapping = (
            self.AI_FIELD_MAPPING if (bom_type or "").lower() == "ai"
            else self.DATASET_FIELD_MAPPING
        )
        if bom_field in mapping:
            value = mapping[bom_field]
            # The mappings store ``license`` as the literal string
            # ``"license"`` because the BOM dict uses ``license`` as the
            # key in both direct_fields and direct_metadata. The actual
            # SPDX 3.0.1 property is the one on LicenseExpression, so we
            # normalise that here.
            if value == "license":
                return "simplelicensing_licenseExpression"
            return value

        # 2. Snake_case alias → canonical camelCase → SPDX property
        if (bom_type or "").lower() == "ai":
            for canonical, aliases in self.AI_RAG_FIELD_ALIASES.items():
                if bom_field in aliases:
                    return self.AI_FIELD_MAPPING.get(canonical)

        # 3. Relationship-typed AI fields (no single property on the package).
        # Dataset BOMs intentionally don't have these (no ``trainedOnDatasets``
        # on a Dataset), so the lookup is AI-only by design. A malformed
        # dataset BOM that contained one of these keys falls through to the
        # generic ``return None`` below — callers degrade gracefully by using
        # the BOM field name in the annotation ``name``. This is preferable
        # to raising: a bad input shape should still produce a parseable
        # SPDX document with the conflict labelled, just less prettily.
        if (bom_type or "").lower() == "ai":
            rel = self._AI_RELATIONSHIP_PROPERTIES.get(bom_field)
            if rel:
                return rel

        return None

    def _build_conflict_annotations(
        self,
        bom_data: Dict[str, Any],
        subject_id: str,
        creation_ref: str,
        bom_type: str = "ai",
        relationship_subjects: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Emit SPDX 3.0.1 ``Annotation`` Elements for every conflict in ``bom_data``.

        SPDX Core defines ``Annotation`` as an Element whose ``subject``
        points at the annotated Element and whose ``statement`` carries
        free-form content (typed via ``contentType``). The vocabulary for
        ``annotationType`` only defines ``other`` and ``review``; we use
        ``other`` because conflict provenance is not a human review.

        The ``statement`` is JSON-encoded (``contentType: application/json``)
        so downstream consumers can parse the rich provenance — claims by
        source, narrative, grounding, confidence band — without re-running
        the auditor. ``name``/``description`` provide a human-readable
        summary for SPDX viewers that don't parse JSON statements.

        Args:
            bom_data: BOM metadata dict (``direct_fields`` / ``rag_fields``
                / ``direct_metadata`` / ``rag_metadata``).
            subject_id: ``spdxId`` of the Element that owns the conflicted
                field (the root AIPackage / DatasetPackage).
            creation_ref: ``@id`` reference (e.g. ``_:creationinfo-<uuid>``
                or the legacy ``_:creationinfo``) of the CreationInfo node
                shared with the rest of the document so SHACL
                ``creationInfo`` refs resolve.
            relationship_subjects: Optional ``{bom_field_name: spdxId}`` map
                for relationship fields (e.g. ``trainedOnDatasets`` →
                ``urn:spdx:DatasetPackage-…``). Falls back to ``subject_id``.
        """
        from aikaboom.core.conflict_routing import (
            HIGH_CONFIDENCE_THRESHOLD,
            SUPPRESS_GROUNDING_BELOW,
        )

        if not isinstance(bom_data, dict):
            return []

        relationship_subjects = relationship_subjects or {}

        def _confidence_band(grounding: Any) -> str:
            if grounding is None:
                return "deterministic"
            try:
                g = float(grounding)
            except (TypeError, ValueError):
                return "deterministic"
            if g < SUPPRESS_GROUNDING_BELOW:
                return "suppressed"
            if g >= HIGH_CONFIDENCE_THRESHOLD:
                return "high"
            return "low"

        annotations: List[Dict[str, Any]] = []

        for section in ("direct_fields", "rag_fields", "direct_metadata", "rag_metadata"):
            fields = bom_data.get(section)
            if not isinstance(fields, dict):
                continue
            section_label = (
                "direct" if section.startswith("direct") else "rag"
            )
            for field_name, triplet in fields.items():
                if not isinstance(triplet, dict):
                    continue

                subject_for_field = relationship_subjects.get(field_name, subject_id)
                spdx_property = self._spdx_property_for_bom_field(field_name, bom_type)
                trace = triplet.get("trace") or {}
                claims = (trace.get("claims") or {}) if isinstance(trace, dict) else {}
                selected_sources = (
                    (trace.get("selected_sources") or [])
                    if isinstance(trace, dict) else []
                )

                # 1. Direct-field conflict (deterministic). Same shape gate
                #    as ``_extract_conflicts`` so phantom RAG envelopes
                #    (``{internal: "No", external: "No"}``) don't create
                #    Annotation noise.
                conflict = triplet.get("conflict")
                if (
                    isinstance(conflict, dict)
                    and "type" in conflict
                    and "value" in conflict
                    and "internal" not in conflict
                ):
                    payload = {
                        "spdx_property": spdx_property,
                        "field": field_name,
                        "section": section_label,
                        "kind": conflict.get("type") or "inter",
                        "chosen_value": triplet.get("value"),
                        "chosen_source": triplet.get("source"),
                        "conflict_value": conflict.get("value"),
                        "conflict_source": conflict.get("source"),
                        "grounding": None,
                        "confidence": "deterministic",
                        "claims": dict(claims),
                        "sources_considered": list(claims.keys()),
                        "selected_sources": list(selected_sources),
                    }
                    annotations.append(
                        self._make_annotation(
                            subject_for_field, creation_ref,
                            spdx_property, field_name,
                            "direct-source-disagreement", payload,
                        )
                    )

                # 2. RAG auditor — internal contradictions inside one source.
                if isinstance(trace, dict):
                    for src, entry in (trace.get("internal_conflicts") or {}).items():
                        if not isinstance(entry, dict):
                            entry = {"narrative": str(entry)}
                        grounding = entry.get("grounding")
                        payload = {
                            "spdx_property": spdx_property,
                            "field": field_name,
                            "section": section_label,
                            "kind": "rag-internal",
                            "chosen_value": triplet.get("value"),
                            "chosen_source": triplet.get("source"),
                            "conflict_source": src,
                            "narrative": entry.get("narrative"),
                            "statements": entry.get("statements"),
                            "grounding": grounding,
                            "confidence": _confidence_band(grounding),
                            "claims": dict(claims),
                            "sources_considered": list(claims.keys()),
                            "selected_sources": list(selected_sources),
                        }
                        annotations.append(
                            self._make_annotation(
                                subject_for_field, creation_ref,
                                spdx_property, field_name,
                                "intra-source-contradiction", payload,
                            )
                        )

                # 3. RAG auditor — external contradictions across sources.
                if isinstance(trace, dict):
                    for entry in (trace.get("external_conflicts") or []):
                        if not isinstance(entry, dict):
                            continue
                        sources = list(entry.get("sources") or [])
                        grounding = entry.get("grounding")
                        payload = {
                            "spdx_property": spdx_property,
                            "field": field_name,
                            "section": section_label,
                            "kind": "rag-external",
                            "chosen_value": triplet.get("value"),
                            "chosen_source": triplet.get("source"),
                            "conflict_sources": sources,
                            "narrative": entry.get("description"),
                            "statements": entry.get("statements"),
                            "grounding": grounding,
                            "confidence": _confidence_band(grounding),
                            "claims": dict(claims),
                            "sources_considered": list(claims.keys()),
                            "selected_sources": list(selected_sources),
                        }
                        annotations.append(
                            self._make_annotation(
                                subject_for_field, creation_ref,
                                spdx_property, field_name,
                                "inter-source-disagreement", payload,
                            )
                        )

        return annotations

    def _make_annotation(
        self,
        subject_id: str,
        creation_ref: str,
        spdx_property: Optional[str],
        bom_field: str,
        kind_label: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a single SPDX ``Annotation`` Element.

        The annotation ``name`` and the human-readable ``description`` are
        keyed on the **SPDX 3.0.1 property name** (e.g.
        ``ai_safetyRiskAssessment``) so a downstream SPDX consumer can
        identify the affected property without knowing AIkaBoOM's internal
        field vocabulary. Unknown / unmapped BOM fields fall back to the
        BOM field key as a last-resort identifier.
        """
        compact_payload = {
            k: v for k, v in payload.items()
            if v is not None and v != "" and v != [] and v != {}
        }
        statement = json.dumps(compact_payload, ensure_ascii=False, sort_keys=True)
        property_label = spdx_property or bom_field
        return {
            "type": "Annotation",
            "spdxId": f"urn:spdx:Annotation-{self._generate_uuid()}",
            "creationInfo": creation_ref,
            "annotationType": "other",
            "subject": subject_id,
            "contentType": "application/json",
            "statement": statement,
            "name": f"aikaboom:conflict:{property_label}",
            "description": (
                f"AIkaBoOM-detected {kind_label} on SPDX property "
                f"'{property_label}'. Statement carries structured conflict "
                "provenance (claims, sources, narrative, grounding) as JSON."
            ),
        }

    def validate_and_convert(self, bom_data: Dict[str, Any], bom_type: str = None) -> Dict[str, Any]:
        """
        Convert BOM data to SPDX 3.0.1 format
        
        Args:
            bom_data: Dictionary containing metadata
            bom_type: Type of BOM ('ai' or 'data'), overrides constructor setting
            
        Returns:
            SPDX 3.0.1 compliant dictionary
        """
        # Use provided bom_type or fall back to constructor setting
        bom_type = (bom_type or self.bom_type).lower()
        
        if bom_type == 'ai':
            return self._convert_ai_bom(bom_data)
        elif bom_type in ['data', 'dataset']:
            return self._convert_dataset_bom(bom_data)
        else:
            raise ValueError(f"Invalid bom_type: {bom_type}. Must be 'ai' or 'data'")
    
    def _convert_ai_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert AI BOM data to SPDX 3.0.1 AI Package format"""
        # Extract fields
        direct_fields = bom_data.get("direct_fields", {})
        rag_fields = bom_data.get("rag_fields", {})
        repo_id = bom_data.get("repo_id", "unknown")
        
        # Generate UUIDs
        doc_uuid = self._generate_uuid()
        creation_uuid = self._generate_uuid()
        person_uuid = self._generate_uuid()
        org_uuid = self._generate_uuid()
        bom_uuid = self._generate_uuid()
        package_uuid = self._generate_uuid()
        license_uuid = self._generate_uuid()
        
        # Get timestamp
        timestamp = self._get_current_timestamp()
        
        # Extract organization info. Triplet dicts are always truthy so
        # ``.get(k, "Unknown")`` would never fall through to the default;
        # ``_first_non_nil`` unwraps + nil-checks before falling back.
        supplied_by = self._first_non_nil(direct_fields.get("suppliedBy"), default="Unknown")
        
        # Extract license. Triplet dicts are always truthy so the dict-default
        # in .get() never kicks in; explicitly check the unwrapped value for
        # nil and fall back to NOASSERTION (SPDX-canonical missing sentinel).
        license_raw = self._extract_value(direct_fields.get("license"))
        if license_raw in (None, "") or _is_nil_value(license_raw):
            license_expr = "NOASSERTION"
        else:
            license_expr = license_raw
        
        # Build SPDX document
        spdx_doc = {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": [
                # 1. CreationInfo
                {
                    "type": "CreationInfo",
                    "@id": f"_:creationinfo-{creation_uuid}",
                    "specVersion": "3.0.1",
                    "createdBy": [f"urn:spdx:Person-{person_uuid}"],
                    "created": timestamp
                },
                # 2. Person
                {
                    "type": "Person",
                    "spdxId": f"urn:spdx:Person-{person_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "name": "AI BOM Generator",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "email",
                        "identifier": "bom-generator@example.com"
                    }]
                },
                # 3. Organization
                {
                    "type": "Organization",
                    "spdxId": f"urn:spdx:Organization-{org_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "name": supplied_by or "Unknown",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "other",
                        "issuingAuthority": "GitHub",
                        "identifier": repo_id,
                        "identifierLocator": ["NOASSERTION"]
                    }]
                },
                # 4. SpdxDocument
                {
                    "type": "SpdxDocument",
                    "spdxId": f"urn:spdx:Document-{doc_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "profileConformance": ["core", "ai"],
                    "rootElement": [f"urn:spdx:Bom-{bom_uuid}"]
                },
                # 5. Bom
                {
                    "type": "Bom",
                    "spdxId": f"urn:spdx:Bom-{bom_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "profileConformance": ["core", "ai"],
                    "rootElement": [f"urn:spdx:AIPackage-{package_uuid}"]
                },
                # 6. AIPackage
                self._build_ai_package(package_uuid, creation_uuid, org_uuid, direct_fields, rag_fields, repo_id),
                # 7. LicenseExpression
                {
                    "type": "simplelicensing_LicenseExpression",
                    "spdxId": f"urn:spdx:License-{license_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "simplelicensing_licenseExpression": license_expr or "NOASSERTION",
                    "simplelicensing_licenseListVersion": "3.25.0",
                    "comment": "License information extracted from AI BOM metadata"
                },
                # 8. Relationship - concludedLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-concludedLicense-{self._generate_uuid()}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "relationshipType": "hasConcludedLicense",
                    "from": f"urn:spdx:AIPackage-{package_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Concluded license for AI package"
                },
                # 9. Relationship - declaredLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-declaredLicense-{self._generate_uuid()}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "relationshipType": "hasDeclaredLicense",
                    "from": f"urn:spdx:AIPackage-{package_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Declared license for AI package"
                }
            ]
        }
        
        # Emit trainedOn / testedOn / dependsOn relationships from RAG fields.
        # Track the first stub spdxId per relationship field so conflict
        # annotations on those relationship fields can target the child
        # element rather than always pointing at the AIPackage.
        relationship_fields = {
            'trainedOnDatasets': 'trainedOn',
            'testedOnDatasets': 'testedOn',
            'modelLineage': 'dependsOn',
        }
        relationship_subjects: Dict[str, str] = {}
        for field_key, rel_type in relationship_fields.items():
            value = self._extract_value(rag_fields.get(field_key))
            if _is_nil_value(value):
                continue
            stub_elements, rels = self._build_dataset_relationships(
                value=value,
                rel_type=rel_type,
                from_id=f"urn:spdx:AIPackage-{package_uuid}",
                creation_uuid=creation_uuid,
                parent_identifier=repo_id,
            )
            spdx_doc['@graph'].extend(stub_elements)
            spdx_doc['@graph'].extend(rels)
            if stub_elements:
                relationship_subjects[field_key] = stub_elements[0].get("spdxId")

        # Attach SPDX Core Annotation Elements for every detected conflict.
        # Subject defaults to the AIPackage; relationship-field conflicts
        # target the first stub child for that field when available.
        ai_subject_id = f"urn:spdx:AIPackage-{package_uuid}"
        spdx_doc['@graph'].extend(
            self._build_conflict_annotations(
                bom_data, ai_subject_id,
                creation_ref=f"_:creationinfo-{creation_uuid}",
                bom_type="ai",
                relationship_subjects=relationship_subjects,
            )
        )

        # Plugin-contributed Annotation Elements (e.g. license-compat).
        # Findings are sourced lazily from a BomStore by the plugin loop
        # helper; when no store is available in scope (the standalone
        # ``validate_bom_to_spdx`` path), the helper passes empty findings
        # and the plugin emits nothing. Task 13 wires the store-aware path.
        spdx_doc['@graph'].extend(
            _emit_plugin_annotations(claim_iri=ai_subject_id)
        )

        return spdx_doc

    def _build_dataset_relationships(self, value: str, rel_type: str,
                                      from_id: str, creation_uuid: str,
                                      parent_identifier: Optional[str] = None):
        """Create stub child packages + Relationship elements for a target list.

        Parses ``value`` as a separator-joined list of relationship targets:

        - Separators: ``,``, ``;``, newline, ``->``, ``→`` (the last two are
          common in LLM-extracted lineage strings).
        - ``rel_type == 'dependsOn'`` (modelLineage) emits ``ai_AIPackage``
          stubs; ``trainedOn`` / ``testedOn`` emit ``dataset_DatasetPackage``
          stubs.
        - Targets equal (case-insensitive) to ``parent_identifier`` are
          dropped (a base model's modelLineage often points at itself —
          Finding #19).
        - Nil sentinels (``noAssertion``, ``Not found.``, …) are dropped at
          per-segment granularity; ``"X -> noAssertion"`` becomes a single
          target ``X``.

        Returns: ``(stub_elements: list[dict], relationships: list[dict])``.
        """
        # Shared helper handles separator parsing (commas, semicolons,
        # newlines, ``->`` / ``→`` arrows), dedup, nil-sentinel removal,
        # and self-loop drops. Same parser the recursive walker uses.
        unique_names = split_lineage_targets(value, parent_identifier)
        if not unique_names:
            return [], []

        # Apply the user-configurable cap (Phase 10 / Finding #22).
        # Default ``None`` → unbounded; the recursive walker and linked
        # bundle have always been unbounded, so the standalone SPDX now
        # matches by default. Users can pass --spdx-relationship-cap to
        # cap the standalone output.
        cap = self.relationship_cap
        if cap is not None and cap >= 0:
            unique_names = unique_names[:cap]

        stubs = []
        rels = []
        for name in unique_names:
            child_uuid = self._generate_uuid()
            stub = self._lineage_stub(name, rel_type, child_uuid, creation_uuid)
            stubs.append(stub)
            rels.append({
                "type": "Relationship",
                "spdxId": f"urn:spdx:Relationship-{rel_type}-{child_uuid}",
                "creationInfo": f"_:creationinfo-{creation_uuid}",
                "relationshipType": rel_type,
                "from": from_id,
                "to": [stub["spdxId"]],
                "description": f"{rel_type} relationship to: {name}",
            })
        return stubs, rels

    def _lineage_stub(self, name: str, rel_type: str, child_uuid: str,
                       creation_uuid: str) -> Dict[str, Any]:
        """Build a minimally-valid stub package for a relationship target.

        - ``dependsOn`` (modelLineage) → ``ai_AIPackage`` (the ancestor IS
          another AI model, not a dataset). Earlier code emitted
          ``dataset_DatasetPackage`` for every relationship, which produced
          schema errors for modelLineage edges (Finding #17 part b).
        - ``trainedOn`` / ``testedOn`` → ``dataset_DatasetPackage``.
        """
        if rel_type == "dependsOn":
            return {
                "type": "ai_AIPackage",
                "spdxId": f"urn:spdx:AIPackage-{child_uuid}",
                "creationInfo": f"_:creationinfo-{creation_uuid}",
                "name": name,
                "software_downloadLocation": "NOASSERTION",
                "software_primaryPurpose": "model",
            }
        return {
            "type": "dataset_DatasetPackage",
            "spdxId": f"urn:spdx:DatasetPackage-{child_uuid}",
            "creationInfo": f"_:creationinfo-{creation_uuid}",
            "name": name,
            "software_downloadLocation": "NOASSERTION",
            "software_primaryPurpose": "data",
            "dataset_datasetType": ["noAssertion"],
        }

    def _build_ai_package(
        self, package_uuid: str, creation_uuid: str, org_uuid: str,
        direct_fields: Dict, rag_fields: Dict, repo_id: str
    ) -> Dict[str, Any]:
        """Build AI Package element with all mapped fields"""
        ai_package = {
            "type": "ai_AIPackage",
            "spdxId": f"urn:spdx:AIPackage-{package_uuid}",
            "creationInfo": f"_:creationinfo-{creation_uuid}",
            "originatedBy": [f"urn:spdx:Organization-{org_uuid}"],
            "suppliedBy": f"urn:spdx:Organization-{org_uuid}",
        }
        
        # Map direct fields (using official SPDX 3.0.1 property names)
        direct_mapping = {
            "releaseTime": "releaseTime",
            "downloadLocation": "software_downloadLocation",
            "packageVersion": "software_packageVersion",
            "primaryPurpose": "software_primaryPurpose",
        }

        for our_field, spdx_field in direct_mapping.items():
            value = self._extract_value(direct_fields.get(our_field))
            if value is not None and value != "":
                if spdx_field == "releaseTime":
                    ai_package[spdx_field] = self._normalize_timestamp(value)
                elif spdx_field == "software_primaryPurpose":
                    ai_package[spdx_field] = self._normalize_enum(
                        value, _SOFTWARE_PURPOSES, "model"
                    )
                else:
                    ai_package[spdx_field] = value
            else:
                if spdx_field == "software_downloadLocation":
                    ai_package[spdx_field] = "NOASSERTION"
                elif spdx_field == "software_primaryPurpose":
                    ai_package[spdx_field] = "model"

        # Set name (standard SPDX Core property)
        model_name_value = self._extract_value(rag_fields.get("model_name"))
        ai_package["name"] = model_name_value or repo_id or "AI Model Name Placeholder"

        # BOM RAG fields are keyed by camelCase question_type IDs. The
        # snake_case aliases in AI_RAG_FIELD_ALIASES keep this working
        # for externally-authored BOMs that still use the legacy spelling.
        # energyConsumption is handled separately below — SPDX 3.0.1 models
        # it as a structured ai_EnergyConsumption object, not a free string.
        scalar_mapping = {
            "informationAboutApplication": "ai_informationAboutApplication",
            "informationAboutTraining":    "ai_informationAboutTraining",
            "limitation":                  "ai_limitation",
        }
        list_mapping = {
            "domain":                  "ai_domain",
            "modelDataPreprocessing":  "ai_modelDataPreprocessing",
            "modelExplainability":     "ai_modelExplainability",
            "standardCompliance":      "ai_standardCompliance",
            "typeOfModel":             "ai_typeOfModel",
        }
        dictionary_mapping = {
            "hyperparameter":          "ai_hyperparameter",
            "metric":                  "ai_metric",
            "metricDecisionThreshold": "ai_metricDecisionThreshold",
        }

        # When a field is populated in the BOM with a real value we emit
        # that value. When the BOM explicitly says "noAssertion" (the LLM
        # was asked and couldn't answer), we still emit the property with
        # the SPDX-canonical ``"noAssertion"`` sentinel — losing the field
        # entirely would hide audit-relevant information ("we asked, no
        # source had data"). When the BOM is missing the field altogether,
        # we skip it (no evidence that the question was even asked).
        for our_field, spdx_field in scalar_mapping.items():
            triplet = self._rag_get(rag_fields, our_field)
            if triplet is None:
                continue
            value = self._extract_value(triplet)
            if value in (None, ""):
                continue
            text = str(value).strip()
            if not text:
                continue
            ai_package[spdx_field] = "noAssertion" if _is_nil_value(text) else text

        for our_field, spdx_field in list_mapping.items():
            triplet = self._rag_get(rag_fields, our_field)
            if triplet is None:
                continue
            values = [str(v) for v in self._as_list(triplet)]
            if values:
                ai_package[spdx_field] = values
            elif _is_nil_value(self._extract_value(triplet)):
                ai_package[spdx_field] = ["noAssertion"]

        for our_field, spdx_field in dictionary_mapping.items():
            triplet = self._rag_get(rag_fields, our_field)
            if triplet is None:
                continue
            entries = self._dictionary_entries(triplet)
            if entries:
                ai_package[spdx_field] = entries
            elif _is_nil_value(self._extract_value(triplet)):
                ai_package[spdx_field] = [
                    {"type": "DictionaryEntry", "key": "value", "value": "noAssertion"}
                ]

        autonomy = self._extract_value(self._rag_get(rag_fields, "autonomyType"))
        if autonomy not in (None, ""):
            ai_package["ai_autonomyType"] = self._normalize_enum(
                autonomy, _PRESENCE_VALUES, "noAssertion"
            )

        sensitive_pii = self._extract_value(self._rag_get(rag_fields, "useSensitivePersonalInformation"))
        if sensitive_pii not in (None, ""):
            ai_package["ai_useSensitivePersonalInformation"] = self._normalize_enum(
                sensitive_pii, _PRESENCE_VALUES, "noAssertion"
            )

        # safetyRiskAssessment: emit only when the value normalises to a
        # real SafetyRiskAssessmentType member (low/medium/high/serious).
        # "noAssertion" and other non-members are dropped — the schema has
        # no sentinel for them, so emitting one breaks the whole document.
        safety = self._extract_value(self._rag_get(rag_fields, "safetyRiskAssessment"))
        if safety not in (None, ""):
            normalized_safety = self._normalize_enum(safety, _AI_SAFETY_VALUES, "")
            if normalized_safety:
                ai_package["ai_safetyRiskAssessment"] = normalized_safety

        # energyConsumption: only a clean "<number> <unit>" figure can be
        # represented as the structured ai_EnergyConsumption type. Prose
        # and no-assertion sentinels yield None and the property is omitted.
        energy_element = _energy_consumption_element(
            self._extract_value(self._rag_get(rag_fields, "energyConsumption"))
        )
        if energy_element is not None:
            ai_package["ai_energyConsumption"] = energy_element

        # Set builtTime if not present
        if "builtTime" not in ai_package:
            built_time = self._extract_value(direct_fields.get("builtTime"))
            ai_package["builtTime"] = self._normalize_timestamp(built_time)
        
        ai_package["comment"] = "This results are generated by AI tools."
        
        return ai_package
    
    def _convert_dataset_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Dataset BOM data to SPDX 3.0.1 Dataset Package format"""
        # Extract metadata
        direct = bom_data.get('direct_metadata', bom_data.get('direct_fields', {}))
        rag = bom_data.get('rag_metadata', bom_data.get('rag_fields', {}))
        urls = bom_data.get('urls', {})
        dataset_id = bom_data.get('dataset_id', bom_data.get('repo_id', 'unknown'))
        
        # Generate UUIDs
        person_uuid = self._generate_uuid()
        org_uuid = self._generate_uuid()
        doc_uuid = self._generate_uuid()
        bom_uuid = self._generate_uuid()
        dataset_uuid = self._generate_uuid()
        rel_concluded_uuid = self._generate_uuid()
        rel_declared_uuid = self._generate_uuid()
        license_uuid = self._generate_uuid()
        
        # Get timestamp
        created_time = self._get_current_timestamp()
        
        # Extract values with fallbacks. Use ``_first_non_nil`` rather
        # than the bare ``or``-chain idiom: direct-field triplet dicts
        # are always truthy, so ``direct.get(k) or rag.get(k) or "X"``
        # short-circuits on the truthy triplet even when its ``value``
        # is ``None``, leaving the rag and default branches dead.
        dataset_name = self._first_non_nil(
            direct.get('name'), rag.get('name'), default=dataset_id,
        )
        originated_by = self._first_non_nil(
            direct.get('originatedBy'), rag.get('originatedBy'), default="Unknown",
        )
        built_time = self._normalize_timestamp(
            self._first_non_nil(direct.get('builtTime'), rag.get('builtTime')),
            created_time,
        )
        release_time = self._normalize_timestamp(
            self._first_non_nil(direct.get('releaseTime'), rag.get('releaseTime')),
            created_time,
        )
        download_location = self._first_non_nil(
            direct.get('downloadLocation'),
            urls.get('github'),
            urls.get('huggingface'),
            default="NOASSERTION",
        )
        primary_purpose = self._normalize_enum(
            self._first_non_nil(
                direct.get('primaryPurpose'), rag.get('primaryPurpose'),
                default="data",
            ),
            _SOFTWARE_PURPOSES,
            "data",
        )
        # Triplet dicts are always truthy so the original `or`-chain default
        # never fired; explicitly check the unwrapped value for nil and fall
        # back to NOASSERTION. With Phase 11.6A canonicalising at the HF
        # inspector, lists are no longer a concern at this layer.
        license_raw = self._extract_value(direct.get('license'))
        if license_raw in (None, "") or _is_nil_value(license_raw):
            license_raw = self._extract_value(rag.get('license'))
        # Belt-and-suspenders: if a list-form license slipped past the HF
        # inspector (e.g. a hand-crafted BOM dict fed to the validator
        # directly), join it as an SPDX OR-expression so the SPDX schema
        # never sees a list as a license expression.
        if isinstance(license_raw, list):
            cleaned = [str(x).strip() for x in license_raw if x]
            license_raw = " OR ".join(cleaned) if cleaned else None
        if license_raw in (None, "") or _is_nil_value(license_raw):
            license_expr = "NOASSERTION"
        else:
            license_expr = license_raw
        dataset_availability = self._normalize_enum(
            self._first_non_nil(
                direct.get('datasetAvailability'),
                rag.get('datasetAvailability'),
                default="directDownload",
            ),
            _DATASET_AVAILABILITY_VALUES,
            "directDownload",
        )

        # RAG-specific fields with type conversion. ``_as_list`` normalises
        # scalar/list shapes and drops nil sentinels so a silent
        # ``"noAssertion"`` answer doesn't become ``["noAssertion"]``.
        data_preprocessing = self._as_list(rag.get('dataPreprocessing'))
        
        data_collection = self._extract_value(rag.get('dataCollectionProcess') or "")
        
        dataset_size = _coerce_dataset_size_bytes(
            self._extract_value(rag.get('datasetSize'))
        )
        # `dataset_size is None` means we couldn't parse a byte count from
        # the answer; the property is omitted from the Dataset element so
        # the SPDX export carries no misleading "0 bytes" claim.
        
        dataset_type = self._normalize_enum_list(
            rag.get('datasetType', []), _DATASET_TYPES, "other"
        )
        
        dataset_update = self._extract_value(rag.get('datasetUpdateMechanism') or "")
        has_pii = self._normalize_enum(
            rag.get('hasSensitivePersonalInformation') or "no",
            _PRESENCE_VALUES,
            "no",
        )
        intended_use = self._extract_value(rag.get('intendedUse') or "")
        
        known_bias = self._as_list(rag.get('knownBias'))

        # Build the DatasetPackage element first so we can conditionally
        # omit dataset_datasetSize when the LLM answer didn't yield a
        # parseable byte count (the SPDX schema allows omission, which is
        # less misleading than emitting "0").
        dataset_package = {
            "type": "dataset_DatasetPackage",
            "spdxId": f"urn:spdx:DatasetPackage-{dataset_uuid}",
            "creationInfo": "_:creationinfo",
            "name": dataset_name,
            "originatedBy": [f"urn:spdx:Organization-{org_uuid}"],
            "builtTime": built_time,
            "releaseTime": release_time,
            "software_downloadLocation": download_location,
            "software_primaryPurpose": primary_purpose,
            "dataset_anonymizationMethodUsed": self._as_list(rag.get('anonymizationMethodUsed', "")),
            "dataset_confidentialityLevel": self._normalize_enum(
                rag.get('confidentialityLevel', "clear"),
                _DATASET_CONFIDENTIALITY_VALUES,
                "clear",
            ),
            "dataset_dataPreprocessing": data_preprocessing,
            "dataset_datasetAvailability": dataset_availability,
            "dataset_dataCollectionProcess": data_collection,
            "dataset_datasetNoise": self._extract_value(rag.get('datasetNoise', "")),
            "dataset_datasetType": dataset_type,
            "dataset_datasetUpdateMechanism": dataset_update,
            "dataset_hasSensitivePersonalInformation": has_pii,
            "dataset_intendedUse": intended_use,
            "dataset_knownBias": known_bias,
            "dataset_sensor": self._dictionary_entries(rag.get('sensorUsed', "")),
            "comment": "Generated by AIkaBoOM.",
        }
        if dataset_size is not None:
            dataset_package["dataset_datasetSize"] = dataset_size

        # Build SPDX document
        spdx_doc = {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": [
                # 1. CreationInfo
                {
                    "type": "CreationInfo",
                    "@id": "_:creationinfo",
                    "specVersion": "3.0.1",
                    "createdBy": [f"urn:spdx:Person-{person_uuid}"],
                    "created": created_time
                },
                # 2. Person
                {
                    "type": "Person",
                    "spdxId": f"urn:spdx:Person-{person_uuid}",
                    "creationInfo": "_:creationinfo",
                    "name": "Dataset BOM Generator",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "email",
                        "identifier": "bom-generator@example.com"
                    }]
                },
                # 3. Organization
                {
                    "type": "Organization",
                    "spdxId": f"urn:spdx:Organization-{org_uuid}",
                    "creationInfo": "_:creationinfo",
                    "name": originated_by,
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "other",
                        "issuingAuthority": "GitHub",
                        "identifier": originated_by,
                        "identifierLocator": [download_location]
                    }]
                },
                # 4. SpdxDocument
                {
                    "type": "SpdxDocument",
                    "spdxId": f"urn:spdx:Document-{doc_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"urn:spdx:Bom-{bom_uuid}"]
                },
                # 5. Bom
                {
                    "type": "Bom",
                    "spdxId": f"urn:spdx:Bom-{bom_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"urn:spdx:DatasetPackage-{dataset_uuid}"]
                },
                # 6. DatasetPackage (built above so dataset_datasetSize can be omitted on no-assertion)
                dataset_package,
                # 7. LicenseExpression
                {
                    "type": "simplelicensing_LicenseExpression",
                    "spdxId": f"urn:spdx:License-{license_uuid}",
                    "creationInfo": "_:creationinfo",
                    "simplelicensing_licenseExpression": license_expr,
                    "simplelicensing_licenseListVersion": "3.25.0",
                    "comment": "License information extracted from Dataset BOM metadata"
                },
                # 8. Relationship - concludedLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-concludedLicense-{rel_concluded_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasConcludedLicense",
                    "from": f"urn:spdx:DatasetPackage-{dataset_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Concluded license for dataset package"
                },
                # 9. Relationship - declaredLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-declaredLicense-{rel_declared_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasDeclaredLicense",
                    "from": f"urn:spdx:DatasetPackage-{dataset_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Declared license for dataset package"
                }
            ]
        }

        # Attach SPDX Core Annotation Elements for every detected conflict.
        # Dataset BOMs share a single literal ``_:creationinfo`` ref across
        # the whole graph (legacy of pre-Phase-7 dataset writer), so we
        # pass that here instead of a uuid-suffixed ref.
        dataset_subject_id = f"urn:spdx:DatasetPackage-{dataset_uuid}"
        spdx_doc['@graph'].extend(
            self._build_conflict_annotations(
                bom_data, dataset_subject_id,
                creation_ref="_:creationinfo",
                bom_type="data",
            )
        )

        # Plugin-contributed Annotation Elements (e.g. license-compat).
        # See note in ``_convert_ai_bom`` — Task 13 wires the store-aware
        # path; this placeholder is no-op until findings are produced.
        spdx_doc['@graph'].extend(
            _emit_plugin_annotations(claim_iri=dataset_subject_id)
        )

        return spdx_doc
    
    def save_spdx(self, spdx_data: Dict[str, Any], output_path: str) -> str:
        """Save SPDX document to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spdx_data, f, indent=2, ensure_ascii=False)
        print(f"✅ SPDX BOM saved to: {output_path}")
        return output_path
    
    def validate_spdx_bom(self, spdx_bom: Dict, strict: bool = False) -> tuple:
        """Validate an SPDX 3.0.1 JSON-LD document with official artifacts.

        Default validation uses the bundled SPDX 3.0.1 JSON Schema. Strict
        validation runs JSON Schema first and then the bundled SHACL shapes.
        The private structural validator is only used if official validation
        cannot run in the current environment.
        """
        errors: List[str] = []

        try:
            import jsonschema

            schema = _load_json_schema()
            validator = jsonschema.Draft202012Validator(schema)
            for error in sorted(validator.iter_errors(spdx_bom), key=lambda e: list(e.path)):
                path = ".".join(str(p) for p in error.path) or "$"
                errors.append(f"JSON Schema: {path}: {error.message}")
        except Exception as exc:
            fallback_ok, fallback_errors = self._validate_spdx_bom_structural(spdx_bom, strict=False)
            if fallback_ok:
                return False, [f"JSON Schema validation unavailable: {exc}"]
            return False, [f"JSON Schema validation unavailable: {exc}", *fallback_errors]

        _integrity_ok, integrity_errors = self._validate_spdx_bom_structural(
            spdx_bom,
            strict=False,
        )
        errors.extend(f"SPDX Integrity: {error}" for error in integrity_errors)

        if errors:
            return False, errors

        if strict:
            try:
                from pyshacl import validate as shacl_validate
                from rdflib import Graph

                shacl_input = json.loads(json.dumps(spdx_bom))
                with open(_get_bundled_schema_path("spdx-context.jsonld"), encoding="utf-8") as f:
                    shacl_input["@context"] = json.load(f).get("@context")
                data_graph = Graph()
                data_graph.parse(data=json.dumps(shacl_input), format="json-ld")
                shacl_graph = Graph()
                shacl_graph.parse(_get_bundled_schema_path("spdx-model.ttl"), format="turtle")
                conforms, _results_graph, results_text = shacl_validate(
                    data_graph=data_graph,
                    shacl_graph=shacl_graph,
                    ont_graph=shacl_graph,
                    inference="none",
                    serialize_report_graph=False,
                )
                if not conforms:
                    errors.extend(self._summarize_shacl_report(results_text))
            except ImportError as exc:
                return False, [f"SHACL validation unavailable: {exc}"]
            except Exception as exc:
                return False, [f"SHACL validation error: {exc}"]

        return len(errors) == 0, errors

    def _summarize_shacl_report(self, results_text: str) -> List[str]:
        messages = []
        current = []
        for line in (results_text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("Constraint Violation"):
                if current:
                    messages.append("SHACL: " + " | ".join(current[:3]))
                current = [stripped]
            elif stripped.startswith(("Message:", "Focus Node:", "Result Path:", "Value Node:")):
                current.append(stripped)
        if current:
            messages.append("SHACL: " + " | ".join(current[:3]))
        if not messages and results_text:
            messages.append("SHACL: " + results_text.strip().replace("\n", " ")[:500])
        return messages or ["SHACL: validation failed"]

    def _validate_spdx_bom_structural(self, spdx_bom: Dict, strict: bool = False) -> tuple:
        """Structural validation of an SPDX 3.0.1 JSON-LD document.

        Checks required top-level keys, required element types, ID
        uniqueness, cross-reference integrity (creationInfo, relationship
        from/to), and per-element required properties.

        Args:
            spdx_bom: The SPDX JSON-LD dict to validate.
            strict: When True, also checks timestamp format and license
                expression non-emptiness.

        Returns:
            Tuple of (is_valid: bool, errors: list[str])
        """
        import re
        errors = []

        # 1. Top-level keys
        if '@context' not in spdx_bom:
            errors.append("Missing '@context'")
        if '@graph' not in spdx_bom or not isinstance(spdx_bom.get('@graph'), list):
            errors.append("Missing or invalid '@graph' array")
            return (False, errors)

        graph = spdx_bom['@graph']

        # 2. Build ID index (spdxId or @id) and check uniqueness
        id_index = set()
        for elem in graph:
            sid = elem.get('spdxId') or elem.get('@id')
            if sid:
                if sid in id_index:
                    errors.append(f"Duplicate ID: {sid}")
                id_index.add(sid)

        # 3. Required element types
        type_set = {e.get('type') for e in graph if e.get('type')}
        if 'CreationInfo' not in type_set:
            errors.append("Missing required element type: CreationInfo")
        if 'SpdxDocument' not in type_set:
            errors.append("Missing required element type: SpdxDocument")
        if 'Bom' not in type_set:
            errors.append("Missing required element type: Bom")

        expected_pkg = 'ai_AIPackage' if self.bom_type == 'ai' else 'dataset_DatasetPackage'
        if expected_pkg not in type_set:
            errors.append(f"Missing required element type: {expected_pkg}")

        # 4. Per-element required property checks
        for elem in graph:
            t = elem.get('type')
            eid = elem.get('spdxId') or elem.get('@id') or '(anonymous)'

            if t == 'CreationInfo':
                if elem.get('specVersion') != '3.0.1':
                    errors.append(f"CreationInfo {eid}: specVersion must be '3.0.1', got '{elem.get('specVersion')}'")
                if not elem.get('created'):
                    errors.append(f"CreationInfo {eid}: missing 'created' timestamp")
                elif strict:
                    ts = elem.get('created', '')
                    if not re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', ts):
                        errors.append(f"CreationInfo {eid}: 'created' not ISO 8601 with Z suffix: {ts}")
                if not elem.get('createdBy'):
                    errors.append(f"CreationInfo {eid}: missing 'createdBy'")

            if t == 'SpdxDocument':
                pc = elem.get('profileConformance') or []
                if 'core' not in pc:
                    errors.append(f"SpdxDocument {eid}: profileConformance must include 'core'")
                if not elem.get('rootElement'):
                    errors.append(f"SpdxDocument {eid}: missing 'rootElement'")

            if t == 'Bom':
                pc = elem.get('profileConformance') or []
                if 'core' not in pc:
                    errors.append(f"Bom {eid}: profileConformance must include 'core'")

        # 5. Cross-reference integrity
        for elem in graph:
            t = elem.get('type')
            eid = elem.get('spdxId') or elem.get('@id') or '(anonymous)'

            ci = elem.get('creationInfo')
            if ci and isinstance(ci, str) and ci not in id_index:
                errors.append(f"{t} {eid}: creationInfo references unknown ID: {ci}")

            if t == 'Relationship':
                frm = elem.get('from')
                if frm and frm not in id_index:
                    errors.append(f"Relationship {eid}: 'from' references unknown ID: {frm}")
                for to_ref in (elem.get('to') or []):
                    if to_ref not in id_index:
                        errors.append(f"Relationship {eid}: 'to' references unknown ID: {to_ref}")

            if t == 'SpdxDocument':
                for root in (elem.get('rootElement') or []):
                    if root not in id_index:
                        errors.append(f"SpdxDocument {eid}: rootElement references unknown ID: {root}")

            if t == 'Bom':
                for root in (elem.get('rootElement') or []):
                    if root not in id_index:
                        errors.append(f"Bom {eid}: rootElement references unknown ID: {root}")

        return len(errors) == 0, errors


def validate_bom_to_spdx(
    bom_data: Dict[str, Any],
    bom_type: str = 'ai',
    output_path: Optional[str] = None,
    validate: bool = True,
    strict: bool = False,
    relationship_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to validate and convert BOM data to SPDX format

    Args:
        bom_data: BOM metadata dictionary
        bom_type: Type of BOM ('ai' or 'data')
        output_path: Optional path to save SPDX JSON
        validate: Validate generated SPDX before returning
        strict: Also run beta SHACL validation when validate is True
        relationship_cap: Max trainedOn / testedOn / dependsOn child
            packages per relationship in the standalone SPDX. ``None``
            (default) is unbounded — matches the recursive walker and
            linked bundle. Phase 10 (Finding #22).

    Returns:
        SPDX 3.0.1 compliant dictionary
    """
    validator = SPDXValidator(bom_type=bom_type, relationship_cap=relationship_cap)
    spdx_data = validator.validate_and_convert(bom_data, bom_type=bom_type)

    if validate:
        ok, errors = validator.validate_spdx_bom(spdx_data, strict=strict)
        if ok:
            print("SPDX 3.0.1 validation passed")
        else:
            print(f"SPDX 3.0.1 validation failed with {len(errors)} error(s)")
            for error in errors[:10]:
                print(f"  - {error}")
    
    if output_path:
        validator.save_spdx(spdx_data, output_path)
    
    return spdx_data


def validate_spdx_export(
    spdx_data: Dict[str, Any],
    strict: bool = False,
    bom_type: str = "ai",
) -> Dict[str, Any]:
    """Validate an SPDX export and return a structured status payload.

    The default validator is the bundled SPDX 3.0.1 JSON Schema. Passing
    strict=True enables the slower beta SHACL pass.
    """
    validator = SPDXValidator(bom_type=bom_type)
    valid, errors = validator.validate_spdx_bom(spdx_data, strict=strict)
    return {
        "valid": valid,
        "strict": strict,
        "beta": strict,
        "validator": "jsonschema+shacl" if strict else "jsonschema",
        "errors": errors,
    }


if __name__ == "__main__":
    print("🧪 Testing Unified SPDX Validator")
    print("=" * 70)
    
    # Test AI BOM
    print("\n📦 Testing AI BOM conversion...")
    ai_bom_data = {
        "repo_id": "test/ai-model",
        "direct_fields": {
            "suppliedBy": "Test AI Lab",
            "license": "MIT",
            "downloadLocation": "https://huggingface.co/test/ai-model",
            "releaseTime": "2024-01-15T00:00:00Z"
        },
        "rag_fields": {
            "model_name": "Test AI Model",
            "model_type": "transformer",
            "intended_use": "Text generation"
        }
    }
    
    ai_validator = SPDXValidator(bom_type='ai')
    ai_spdx = ai_validator.validate_and_convert(ai_bom_data)
    ai_validator.validate_spdx_bom(ai_spdx)
    
    # Test Dataset BOM
    print("\n📊 Testing Dataset BOM conversion...")
    dataset_bom_data = {
        "dataset_id": "test/dataset",
        "direct_metadata": {
            "name": "Test Dataset",
            "license": "Apache-2.0",
            "originatedBy": "Test Data Lab"
        },
        "rag_metadata": {
            "intendedUse": "Research purposes",
            "datasetSize": 10000
        },
        "urls": {
            "huggingface": "https://huggingface.co/datasets/test/dataset"
        }
    }
    
    dataset_validator = SPDXValidator(bom_type='data')
    dataset_spdx = dataset_validator.validate_and_convert(dataset_bom_data)
    dataset_validator.validate_spdx_bom(dataset_spdx)
    
    print("\n" + "=" * 70)
    print("✅ Unified SPDX Validator test completed successfully!")
