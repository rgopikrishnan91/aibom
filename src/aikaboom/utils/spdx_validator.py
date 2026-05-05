"""
Unified SPDX 3.0.1 BOM Validator
Converts AI and Dataset BOM metadata to SPDX 3.0.1 compliant format.

Validation uses the official SPDX 3.0.1 SHACL shapes and JSON Schema
when available (pyshacl + jsonschema). Falls back to built-in structural
checks when those deps or schema files are missing.
"""

import json
import os
import re
import tempfile
from datetime import datetime
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
    AI_FIELD_MAPPING = {
        # Direct fields (core + software namespace)
        "releaseTime": "releaseTime",
        "suppliedBy": "suppliedBy",
        "downloadLocation": "software_downloadLocation",
        "packageVersion": "software_packageVersion",
        "primaryPurpose": "software_primaryPurpose",
        "license": "license",

        # RAG fields for AI models (ai_ namespace)
        "model_name": "name",
        "autonomy_type": "ai_autonomyType",
        "domain": "ai_domain",
        "energy_consumption": "ai_energyConsumption",
        "hyperparameters": "ai_hyperparameter",
        "intended_use": "ai_informationAboutApplication",
        "training_information": "ai_informationAboutTraining",
        "limitations": "ai_limitation",
        "performance_metrics": "ai_metric",
        "decision_threshold": "ai_metricDecisionThreshold",
        "data_preprocessing": "ai_modelDataPreprocessing",
        "model_explainability": "ai_modelExplainability",
        "safety_risk_assessment": "ai_safetyRiskAssessment",
        "standard_compliance": "ai_standardCompliance",
        "model_type": "ai_typeOfModel",
        "sensitive_personal_information": "ai_useSensitivePersonalInformation",
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
    
    def __init__(self, template_path: str = None, bom_type: str = 'ai'):
        """
        Initialize validator with optional SPDX template
        
        Args:
            template_path: Optional path to SPDX template file
            bom_type: Type of BOM to generate ('ai' or 'data')
        """
        self.template_path = template_path
        self.bom_type = bom_type.lower()
        self.template = self._load_template() if template_path else None
    
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

    def _as_list(self, value: Any) -> List[Any]:
        """Normalize scalar/list SPDX properties to a clean list."""
        value = self._extract_value(value)
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [v for v in (self._extract_value(item) for item in value) if v not in (None, "")]
        if isinstance(value, str):
            parts = [p.strip() for p in re.split(r"[;,\n]+", value) if p.strip()]
            return parts or [value]
        return [value]

    def _normalize_timestamp(self, value: Any, default: Optional[str] = None) -> str:
        """Return an SPDX timestamp with second precision and a Z suffix."""
        value = self._extract_value(value)
        if not value:
            return default or self._get_current_timestamp()
        if isinstance(value, str):
            if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", value):
                return value
            if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                return f"{value}T00:00:00Z"
        return default or self._get_current_timestamp()

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
        """Convert loose metadata into SPDX DictionaryEntry objects."""
        value = self._extract_value(value)
        if not value:
            return []
        if isinstance(value, dict):
            if "key" in value:
                key = str(value.get("key") or "").strip()
                if not key:
                    return []
                entry = {"type": "DictionaryEntry", "key": key}
                if value.get("value") not in (None, ""):
                    entry["value"] = str(value["value"])
                return [entry]
            return [
                {"type": "DictionaryEntry", "key": str(k), "value": str(v)}
                for k, v in value.items()
                if k not in (None, "") and v not in (None, "")
            ]
        entries = []
        for index, item in enumerate(self._as_list(value), start=1):
            if isinstance(item, dict) and "key" in item:
                entries.extend(self._dictionary_entries(item))
            else:
                entries.append({"type": "DictionaryEntry", "key": f"value{index}", "value": str(item)})
        return entries
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SPDX IDs"""
        return str(uuid.uuid4())
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
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
        
        # Extract organization info
        supplied_by = self._extract_value(direct_fields.get("suppliedBy", "Unknown"))
        
        # Extract license
        license_expr = self._extract_value(direct_fields.get("license", "NOASSERTION"))
        
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
        
        # Emit trainedOn / testedOn / dependsOn relationships from RAG fields
        relationship_fields = {
            'trainedOnDatasets': 'trainedOn',
            'testedOnDatasets': 'testedOn',
            'modelLineage': 'dependsOn',
        }
        for field_key, rel_type in relationship_fields.items():
            value = self._extract_value(rag_fields.get(field_key))
            if value and isinstance(value, str) and value.lower() not in ('not found', 'not found.', ''):
                stub_elements, rels = self._build_dataset_relationships(
                    value=value,
                    rel_type=rel_type,
                    from_id=f"urn:spdx:AIPackage-{package_uuid}",
                    creation_uuid=creation_uuid,
                )
                spdx_doc['@graph'].extend(stub_elements)
                spdx_doc['@graph'].extend(rels)

        return spdx_doc

    def _build_dataset_relationships(self, value: str, rel_type: str,
                                      from_id: str, creation_uuid: str):
        """Create stub DatasetPackage elements and Relationship elements.

        Parses a comma/semicolon separated string of dataset names, creates
        a minimal DatasetPackage stub for each, and emits a Relationship
        linking the AIPackage to each dataset.

        Returns:
            (stub_elements: list[dict], relationships: list[dict])
        """
        import re
        names = [n.strip() for n in re.split(r'[;,\n]+', value) if n.strip()]
        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for n in names:
            low = n.lower()
            if low not in seen:
                seen.add(low)
                unique_names.append(n)

        stubs = []
        rels = []
        for name in unique_names[:10]:
            ds_uuid = self._generate_uuid()
            ds_id = f"urn:spdx:DatasetPackage-{ds_uuid}"
            stubs.append({
                "type": "dataset_DatasetPackage",
                "spdxId": ds_id,
                "creationInfo": f"_:creationinfo-{creation_uuid}",
                "name": name,
                "software_downloadLocation": "NOASSERTION",
                "software_primaryPurpose": "data",
                "dataset_datasetType": ["noAssertion"],
            })
            rels.append({
                "type": "Relationship",
                "spdxId": f"urn:spdx:Relationship-{rel_type}-{ds_uuid}",
                "creationInfo": f"_:creationinfo-{creation_uuid}",
                "relationshipType": rel_type,
                "from": from_id,
                "to": [ds_id],
                "description": f"{rel_type} relationship to dataset: {name}",
            })
        return stubs, rels

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

        scalar_mapping = {
            "intended_use": "ai_informationAboutApplication",
            "training_information": "ai_informationAboutTraining",
            "limitations": "ai_limitation",
        }
        list_mapping = {
            "domain": "ai_domain",
            "data_preprocessing": "ai_modelDataPreprocessing",
            "model_explainability": "ai_modelExplainability",
            "standard_compliance": "ai_standardCompliance",
            "model_type": "ai_typeOfModel",
        }
        dictionary_mapping = {
            "hyperparameters": "ai_hyperparameter",
            "performance_metrics": "ai_metric",
            "decision_threshold": "ai_metricDecisionThreshold",
        }

        for our_field, spdx_field in scalar_mapping.items():
            value = self._extract_value(rag_fields.get(our_field))
            if value not in (None, ""):
                ai_package[spdx_field] = str(value)

        for our_field, spdx_field in list_mapping.items():
            values = [str(v) for v in self._as_list(rag_fields.get(our_field))]
            if values:
                ai_package[spdx_field] = values

        for our_field, spdx_field in dictionary_mapping.items():
            entries = self._dictionary_entries(rag_fields.get(our_field))
            if entries:
                ai_package[spdx_field] = entries

        autonomy = self._extract_value(rag_fields.get("autonomy_type"))
        if autonomy not in (None, ""):
            ai_package["ai_autonomyType"] = self._normalize_enum(
                autonomy, _PRESENCE_VALUES, "noAssertion"
            )

        sensitive_pii = self._extract_value(rag_fields.get("sensitive_personal_information"))
        if sensitive_pii not in (None, ""):
            ai_package["ai_useSensitivePersonalInformation"] = self._normalize_enum(
                sensitive_pii, _PRESENCE_VALUES, "noAssertion"
            )

        safety = self._extract_value(rag_fields.get("safety_risk_assessment"))
        if safety not in (None, ""):
            normalized_safety = self._normalize_enum(safety, _AI_SAFETY_VALUES, "")
            if normalized_safety:
                ai_package["ai_safetyRiskAssessment"] = normalized_safety
        
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
        
        # Extract values with fallbacks
        dataset_name = self._extract_value(direct.get('name') or rag.get('name') or dataset_id)
        originated_by = self._extract_value(direct.get('originatedBy') or rag.get('originatedBy') or "Unknown")
        built_time = self._normalize_timestamp(direct.get('builtTime') or rag.get('builtTime'), created_time)
        release_time = self._normalize_timestamp(direct.get('releaseTime') or rag.get('releaseTime'), created_time)
        download_location = self._extract_value(direct.get('downloadLocation') or urls.get('github') or urls.get('huggingface') or "NOASSERTION")
        primary_purpose = self._normalize_enum(
            direct.get('primaryPurpose') or rag.get('primaryPurpose') or "data",
            _SOFTWARE_PURPOSES,
            "data",
        )
        license_expr = self._extract_value(direct.get('license') or rag.get('license') or "NOASSERTION")
        dataset_availability = self._normalize_enum(
            direct.get('datasetAvailability') or rag.get('datasetAvailability') or "directDownload",
            _DATASET_AVAILABILITY_VALUES,
            "directDownload",
        )
        
        # RAG-specific fields with type conversion
        data_preprocessing = self._extract_value(rag.get('dataPreprocessing', []))
        if isinstance(data_preprocessing, str):
            data_preprocessing = [data_preprocessing] if data_preprocessing else []
        
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
        
        known_bias = self._extract_value(rag.get('knownBias', []))
        if isinstance(known_bias, str):
            known_bias = [known_bias] if known_bias else []

        # Build the DatasetPackage element first so we can conditionally
        # omit dataset_datasetSize when the LLM answer didn't yield a
        # parseable byte count (the SPDX schema allows omission, which is
        # less misleading than emitting "0").
        dataset_package = {
            "type": "dataset_DatasetPackage",
            "spdxId": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
            "creationInfo": "_:creationinfo",
            "name": dataset_name,
            "originatedBy": [f"https://spdx.org/spdxdocs/Organization1-{org_uuid}"],
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
                    "createdBy": [f"https://spdx.org/spdxdocs/Person1-{person_uuid}"],
                    "created": created_time
                },
                # 2. Person
                {
                    "type": "Person",
                    "spdxId": f"https://spdx.org/spdxdocs/Person1-{person_uuid}",
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
                    "spdxId": f"https://spdx.org/spdxdocs/Organization1-{org_uuid}",
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
                    "spdxId": f"https://spdx.org/spdxdocs/Document1-{doc_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"https://spdx.org/spdxdocs/BOM1-{bom_uuid}"]
                },
                # 5. Bom
                {
                    "type": "Bom",
                    "spdxId": f"https://spdx.org/spdxdocs/BOM1-{bom_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}"]
                },
                # 6. DatasetPackage (built above so dataset_datasetSize can be omitted on no-assertion)
                dataset_package,
                # 7. LicenseExpression
                {
                    "type": "simplelicensing_LicenseExpression",
                    "spdxId": f"https://spdx.org/licenses/{license_uuid}",
                    "creationInfo": "_:creationinfo",
                    "simplelicensing_licenseExpression": license_expr,
                    "simplelicensing_licenseListVersion": "3.25.0",
                    "comment": "License information extracted from Dataset BOM metadata"
                },
                # 8. Relationship - concludedLicense
                {
                    "type": "Relationship",
                    "spdxId": f"https://spdx.org/spdxdocs/Relationship/concludedLicense-{rel_concluded_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasConcludedLicense",
                    "from": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
                    "to": [f"https://spdx.org/licenses/{license_uuid}"],
                    "description": "Concluded license for dataset package"
                },
                # 9. Relationship - declaredLicense
                {
                    "type": "Relationship",
                    "spdxId": f"https://spdx.org/spdxdocs/Relationship/declaredLicense-{rel_declared_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasDeclaredLicense",
                    "from": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
                    "to": [f"https://spdx.org/licenses/{license_uuid}"],
                    "description": "Declared license for dataset package"
                }
            ]
        }
        
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
) -> Dict[str, Any]:
    """
    Convenience function to validate and convert BOM data to SPDX format
    
    Args:
        bom_data: BOM metadata dictionary
        bom_type: Type of BOM ('ai' or 'data')
        output_path: Optional path to save SPDX JSON
        validate: Validate generated SPDX before returning
        strict: Also run beta SHACL validation when validate is True
        
    Returns:
        SPDX 3.0.1 compliant dictionary
    """
    validator = SPDXValidator(bom_type=bom_type)
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
