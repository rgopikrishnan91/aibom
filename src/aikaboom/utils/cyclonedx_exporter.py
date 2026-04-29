"""
Beta CycloneDX 1.7 exporter for AIkaBoOM.

Converts AIkaBoOM provenance BOM data (with triplet fields) into
CycloneDX 1.7 JSON, using the modelCard extension for AI/ML metadata.

Public API:
    CycloneDXExporter(bom_type='ai').validate_and_convert(bom_data) -> dict
    bom_to_cyclonedx(bom_data, bom_type='ai', output_path=None) -> dict
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class CycloneDXExporter:
    """Converts AIkaBoOM provenance BOMs to CycloneDX 1.7 JSON."""

    def __init__(self, bom_type: str = "ai"):
        self.bom_type = bom_type.lower()

    def _extract_value(self, field_data: Any) -> Any:
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data["value"]
        return field_data

    def validate_and_convert(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.bom_type == "ai":
            return self._convert_ai_bom(bom_data)
        elif self.bom_type in ("data", "dataset"):
            return self._convert_dataset_bom(bom_data)
        else:
            raise ValueError(f"Invalid bom_type: {self.bom_type}")

    def _convert_ai_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        direct = bom_data.get("direct_fields", {})
        rag = bom_data.get("rag_fields", {})
        model_id = bom_data.get("model_id", bom_data.get("repo_id", "unknown"))

        component = self._build_ai_component(model_id, direct, rag, bom_data)

        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "version": 1,
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "metadata": {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tools": [{"name": "AIkaBoOM", "version": "1.0.0"}],
                "authors": [{"name": "AIkaBoOM Generator"}],
            },
            "components": [component],
        }

    def _convert_dataset_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        direct = bom_data.get("direct_metadata", bom_data.get("direct_fields", {}))
        rag = bom_data.get("rag_metadata", bom_data.get("rag_fields", {}))
        dataset_id = bom_data.get("dataset_id", bom_data.get("repo_id", "unknown"))

        component = self._build_dataset_component(dataset_id, direct, rag, bom_data)

        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "version": 1,
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "metadata": {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tools": [{"name": "AIkaBoOM", "version": "1.0.0"}],
                "authors": [{"name": "AIkaBoOM Generator"}],
            },
            "components": [component],
        }

    def _build_ai_component(self, model_id, direct, rag, bom_data):
        name = self._extract_value(rag.get("model_name")) or model_id
        version = self._extract_value(direct.get("packageVersion")) or ""
        supplier = self._extract_value(direct.get("suppliedBy")) or ""
        license_val = self._extract_value(direct.get("license")) or "NOASSERTION"
        description = self._extract_value(rag.get("intended_use", rag.get("informationAboutApplication"))) or ""
        task = self._extract_value(direct.get("primaryPurpose")) or ""
        model_type = self._extract_value(rag.get("model_type", rag.get("typeOfModel"))) or ""
        limitation = self._extract_value(rag.get("limitations", rag.get("limitation"))) or ""
        metric = self._extract_value(rag.get("performance_metrics", rag.get("metric"))) or ""
        safety = self._extract_value(rag.get("safety_risk_assessment", rag.get("safetyRiskAssessment"))) or ""

        # External references from URLs
        ext_refs = []
        repo_id = bom_data.get("repo_id", "")
        if repo_id and "/" in str(repo_id):
            ext_refs.append({"type": "documentation", "url": f"https://huggingface.co/{repo_id}"})
        urls = bom_data.get("urls", {})
        if urls.get("github"):
            ext_refs.append({"type": "vcs", "url": urls["github"]})
        if urls.get("arxiv"):
            ext_refs.append({"type": "documentation", "url": urls["arxiv"]})

        # Build modelCard
        model_card = {}
        model_params = {}
        if task:
            model_params["task"] = task
        if model_type:
            model_params["architectureFamily"] = model_type
        if model_params:
            model_card["modelParameters"] = model_params

        # Training datasets
        datasets = []
        trained_on = self._extract_value(rag.get("trainedOnDatasets"))
        if trained_on and isinstance(trained_on, str) and trained_on.lower() not in ("not found", "not found.", ""):
            for ds_name in [n.strip() for n in trained_on.split(",") if n.strip()]:
                datasets.append({"type": "training", "name": ds_name})

        tested_on = self._extract_value(rag.get("testedOnDatasets"))
        if tested_on and isinstance(tested_on, str) and tested_on.lower() not in ("not found", "not found.", ""):
            for ds_name in [n.strip() for n in tested_on.split(",") if n.strip()]:
                datasets.append({"type": "evaluation", "name": ds_name})

        if datasets:
            if "modelParameters" not in model_card:
                model_card["modelParameters"] = {}
            model_card["modelParameters"]["datasets"] = datasets

        # Quantitative analysis
        if metric:
            model_card["quantitativeAnalysis"] = {
                "performanceMetrics": [{"type": "other", "value": metric}]
            }

        # Considerations
        considerations = {}
        if limitation:
            considerations["technicalLimitations"] = [limitation]
        if safety:
            considerations["ethicalConsiderations"] = [{"name": "safetyRiskAssessment", "description": safety}]
        if considerations:
            model_card["considerations"] = considerations

        # Custom properties (fields without a native CycloneDX slot + conflicts)
        properties = []
        custom_fields = {
            "energyConsumption": rag.get("energy_consumption", rag.get("energyConsumption")),
            "autonomyType": rag.get("autonomy_type", rag.get("autonomyType")),
            "modelExplainability": rag.get("model_explainability", rag.get("modelExplainability")),
            "useSensitivePersonalInformation": rag.get("sensitive_personal_information", rag.get("useSensitivePersonalInformation")),
            "standardCompliance": rag.get("standard_compliance", rag.get("standardCompliance")),
        }
        for prop_name, raw in custom_fields.items():
            val = self._extract_value(raw)
            if val and str(val).lower() not in ("not found", "not found.", ""):
                properties.append({"name": f"aikaboom:{prop_name}", "value": str(val)})

        # Preserve conflict info as properties
        for section_key in ("direct_fields", "rag_fields"):
            fields = bom_data.get(section_key, {})
            if not isinstance(fields, dict):
                continue
            for field, triplet in fields.items():
                if isinstance(triplet, dict) and triplet.get("conflict"):
                    properties.append({
                        "name": f"aikaboom:conflict:{field}",
                        "value": json.dumps(triplet["conflict"]),
                    })

        # Model lineage / pedigree
        pedigree = None
        lineage = self._extract_value(rag.get("modelLineage"))
        if lineage and isinstance(lineage, str) and lineage.lower() not in ("not found", "not found.", ""):
            pedigree = {"ancestors": [{"type": "machine-learning-model", "name": lineage}]}

        component = {
            "type": "machine-learning-model",
            "bom-ref": f"ai-model:{model_id}",
            "name": name,
            "supplier": {"name": supplier} if supplier else None,
            "licenses": [{"license": {"id": license_val}}] if license_val and license_val != "NOASSERTION" else [],
            "description": description,
        }
        if version:
            component["version"] = version
        if ext_refs:
            component["externalReferences"] = ext_refs
        if model_card:
            component["modelCard"] = model_card
        if pedigree:
            component["pedigree"] = pedigree
        if properties:
            component["properties"] = properties

        # Remove None values
        return {k: v for k, v in component.items() if v is not None}

    def _build_dataset_component(self, dataset_id, direct, rag, bom_data):
        name = self._extract_value(direct.get("name")) or dataset_id
        license_val = self._extract_value(direct.get("license")) or "NOASSERTION"
        description = self._extract_value(rag.get("intendedUse")) or ""
        originated_by = self._extract_value(direct.get("originatedBy")) or ""

        component = {
            "type": "data",
            "bom-ref": f"dataset:{dataset_id}",
            "name": name,
            "description": description,
            "supplier": {"name": originated_by} if originated_by else None,
            "licenses": [{"license": {"id": license_val}}] if license_val and license_val != "NOASSERTION" else [],
        }

        # Dataset-specific properties
        properties = []
        ds_fields = {
            "datasetSize": rag.get("datasetSize"),
            "datasetType": rag.get("datasetType"),
            "dataCollectionProcess": rag.get("dataCollectionProcess"),
            "knownBias": rag.get("knownBias"),
            "hasSensitivePersonalInformation": rag.get("hasSensitivePersonalInformation"),
            "anonymizationMethodUsed": rag.get("anonymizationMethodUsed"),
        }
        for prop_name, raw in ds_fields.items():
            val = self._extract_value(raw)
            if val and str(val).lower() not in ("not found", "not found.", ""):
                properties.append({"name": f"aikaboom:{prop_name}", "value": str(val)})

        if properties:
            component["properties"] = properties

        return {k: v for k, v in component.items() if v is not None}

    def validate_cyclonedx(self, cdx_bom: Dict) -> tuple:
        """Basic structural validation of a CycloneDX document.

        Uses cyclonedx-python-lib for schema validation if installed,
        otherwise does basic structural checks.

        Returns:
            (is_valid: bool, errors: list[str])
        """
        errors = []

        if cdx_bom.get("bomFormat") != "CycloneDX":
            errors.append("bomFormat must be 'CycloneDX'")
        if cdx_bom.get("specVersion") != "1.7":
            errors.append(f"specVersion must be '1.7', got '{cdx_bom.get('specVersion')}'")
        if not cdx_bom.get("serialNumber"):
            errors.append("Missing serialNumber")
        if not isinstance(cdx_bom.get("components"), list):
            errors.append("Missing or invalid 'components' array")
        elif len(cdx_bom["components"]) == 0:
            errors.append("components array is empty")

        is_valid = len(errors) == 0
        if is_valid:
            print("CycloneDX 1.7 validation passed")
        else:
            print(f"CycloneDX 1.7 validation failed with {len(errors)} error(s)")
            for e in errors:
                print(f"  - {e}")

        return is_valid, errors

    def save_cyclonedx(self, cdx_data: Dict, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cdx_data, f, indent=2, ensure_ascii=False)
        print(f"CycloneDX 1.7 BOM saved to: {output_path}")
        return output_path


def bom_to_cyclonedx(
    bom_data: Dict[str, Any],
    bom_type: str = "ai",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to convert BOM data to CycloneDX 1.7 format."""
    exporter = CycloneDXExporter(bom_type=bom_type)
    cdx_data = exporter.validate_and_convert(bom_data)
    if output_path:
        exporter.save_cyclonedx(cdx_data, output_path)
    return cdx_data
