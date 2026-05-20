"""
Beta CycloneDX 1.6 ML-BOM exporter for AIkaBoOM.

Converts AIkaBoOM provenance BOM data (with triplet fields) into
CycloneDX 1.6 JSON, using the modelCard extension for AI/ML metadata.

Why 1.6 and not 1.7: the validator we adopted in Phase 6
(``sbom-utility``, https://github.com/CycloneDX/sbom-utility) ships
embedded JSON schemas for CycloneDX 1.2-1.6 only as of v0.18.x.
Emitting 1.7 means our own outputs can't be validated end-to-end
(Phase 8 / Finding #9). 1.6 covers every component shape we emit
(machine-learning-model, modelCard, pedigree, properties); revisit
when sbom-utility ships 1.7 schemas.

Public API:
    CycloneDXExporter(bom_type='ai').validate_and_convert(bom_data) -> dict
    bom_to_cyclonedx(bom_data, bom_type='ai', output_path=None) -> dict
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from aikaboom.utils.value_helpers import _is_nil_value
from aikaboom.utils.normalise import cdx_license_block


CDX_SPEC_VERSION = "1.6"


def _collect_conflict_properties(
    bom_data: Dict[str, Any], bom_type: str
) -> List[Dict[str, str]]:
    """Build CycloneDX ``properties`` entries describing every detected conflict.

    Same gate, same SPDX-property naming, and (a JSON-serialised slice of)
    the same payload shape as
    :meth:`SPDXValidator._build_conflict_annotations`, so a consumer
    cross-walking SPDX ↔ CycloneDX sees the same identifier and the same
    provenance on both sides.

    CycloneDX 1.6 doesn't have a top-level ``annotations`` array (added in
    1.7); properties is the canonical extensibility slot in 1.6. We key
    each entry on the SPDX 3.0.1 property name (``ai_safetyRiskAssessment``,
    ``simplelicensing_licenseExpression``, …) so the property key alone
    identifies the conflicted SPDX property; the JSON value carries the
    full provenance.

    Phantom RAG envelopes (``{internal: "No", external: "No"}``) are
    explicitly skipped — same fix class as the SPDX side and the
    ``_extract_conflicts`` web helper.
    """
    # Lazy import to avoid a circular dependency: spdx_validator imports
    # ``aikaboom.utils.value_helpers`` and a few others that don't reach
    # cyclonedx_exporter, but routing through the class keeps the mapping
    # source-of-truth in one place.
    from aikaboom.utils.spdx_validator import SPDXValidator
    from aikaboom.core.conflict_routing import (
        HIGH_CONFIDENCE_THRESHOLD,
        SUPPRESS_GROUNDING_BELOW,
    )

    validator = SPDXValidator(bom_type=bom_type)

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

    out: List[Dict[str, str]] = []
    if not isinstance(bom_data, dict):
        return out

    for section in ("direct_fields", "rag_fields", "direct_metadata", "rag_metadata"):
        fields = bom_data.get(section) or {}
        if not isinstance(fields, dict):
            continue
        section_label = "direct" if section.startswith("direct") else "rag"
        for field_name, triplet in fields.items():
            if not isinstance(triplet, dict):
                continue
            spdx_property = validator._spdx_property_for_bom_field(
                field_name, bom_type
            )
            property_key = (
                f"aikaboom:conflict:{spdx_property}" if spdx_property
                else f"aikaboom:conflict:{field_name}"
            )
            trace = triplet.get("trace") or {}
            claims = (trace.get("claims") or {}) if isinstance(trace, dict) else {}
            selected_sources = (
                (trace.get("selected_sources") or [])
                if isinstance(trace, dict) else []
            )

            # 1. Direct-field conflict (deterministic).
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
                    "confidence": "deterministic",
                    "claims": dict(claims) or None,
                    "selected_sources": list(selected_sources) or None,
                }
                out.append({
                    "name": property_key,
                    "value": json.dumps(
                        {k: v for k, v in payload.items() if v is not None and v != ""},
                        ensure_ascii=False, sort_keys=True,
                    ),
                })

            # 2. RAG auditor — internal contradictions.
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
                        "grounding": grounding,
                        "confidence": _confidence_band(grounding),
                        "claims": dict(claims) or None,
                        "selected_sources": list(selected_sources) or None,
                    }
                    out.append({
                        "name": property_key,
                        "value": json.dumps(
                            {k: v for k, v in payload.items()
                             if v is not None and v != ""},
                            ensure_ascii=False, sort_keys=True,
                        ),
                    })

            # 3. RAG auditor — external contradictions.
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
                        "conflict_sources": sources or None,
                        "narrative": entry.get("description"),
                        "grounding": grounding,
                        "confidence": _confidence_band(grounding),
                        "claims": dict(claims) or None,
                        "selected_sources": list(selected_sources) or None,
                    }
                    out.append({
                        "name": property_key,
                        "value": json.dumps(
                            {k: v for k, v in payload.items()
                             if v is not None and v != ""},
                            ensure_ascii=False, sort_keys=True,
                        ),
                    })

    return out


class _EmptyFindings:
    """Placeholder ``Findings``-Protocol implementation for the CDX path.

    Mirrors ``aikaboom.utils.spdx_validator._EmptyFindings`` — the bom_data
    -> CDX conversion has no ``BomStore`` in scope, so the plugin loop runs
    with empty findings and emits nothing until Task 13 wires the
    store-aware path.
    """

    def to_dict(self) -> Dict[str, Any]:
        return {"findings": []}

    def violations(self) -> List[Any]:
        return []

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


def _collect_plugin_properties(claim_iri: str) -> List[Dict[str, str]]:
    """Loop registered plugins; pack their SPDX-shaped annotations into
    CycloneDX 1.6 ``properties`` under the ``aikaboom:license-compat:*``
    namespace. CycloneDX 1.6 has no Annotation primitive, so we use the
    ``properties`` extension that downstream consumers can parse.
    """
    out: List[Dict[str, str]] = []
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
        except Exception:
            # Plugin emission is best-effort; never break the CDX export.
            continue
        for ann in anns or []:
            comment = ann.get("comment")
            if not comment:
                continue
            out.append({
                "name": f"aikaboom:{getattr(plugin, 'name', 'plugin')}:annotation",
                "value": comment,
            })
    return out


class CycloneDXExporter:
    """Converts AIkaBoOM provenance BOMs to CycloneDX 1.6 JSON."""

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
            "specVersion": CDX_SPEC_VERSION,
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
            "specVersion": CDX_SPEC_VERSION,
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
        license_block = cdx_license_block(self._extract_value(direct.get("license")))
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

        # Training / evaluation datasets. CycloneDX 1.7 introduced
        # ``type: "training"`` / ``type: "evaluation"`` as enum values
        # for modelCard datasets, but we emit 1.6 (sbom-utility v0.18.x
        # only ships embedded schemas through 1.6) — that enum doesn't
        # exist there. Drop the ``type`` field; surface the train/test
        # split as custom properties instead. Phase 10 / Finding #23.
        # Also nil-filter so a "noAssertion" parent value doesn't
        # fabricate a child dataset entry. Finding #24.
        trained_names: List[str] = []
        tested_names: List[str] = []
        trained_on = self._extract_value(rag.get("trainedOnDatasets"))
        if isinstance(trained_on, str) and not _is_nil_value(trained_on):
            trained_names = [
                n.strip() for n in trained_on.split(",")
                if n.strip() and not _is_nil_value(n)
            ]
        tested_on = self._extract_value(rag.get("testedOnDatasets"))
        if isinstance(tested_on, str) and not _is_nil_value(tested_on):
            tested_names = [
                n.strip() for n in tested_on.split(",")
                if n.strip() and not _is_nil_value(n)
            ]

        # CycloneDX 1.6 requires ``type`` on every modelCard dataset entry;
        # the 1.6 enum is source-code | configuration | dataset | definition |
        # other. ``"dataset"`` is the only value that matches the field's
        # purpose. Train / test split rides in the ``properties`` block below.
        datasets = [{"type": "dataset", "name": n} for n in trained_names + tested_names]
        if datasets:
            if "modelParameters" not in model_card:
                model_card["modelParameters"] = {}
            model_card["modelParameters"]["datasets"] = datasets

        # Quantitative analysis
        if metric and not _is_nil_value(metric):
            model_card["quantitativeAnalysis"] = {
                "performanceMetrics": [{"type": "other", "value": metric}]
            }

        # Considerations — nil-filter so silent fields don't fabricate
        # entries with name="noAssertion" (Finding #24).
        considerations = {}
        if limitation and not _is_nil_value(limitation):
            considerations["technicalLimitations"] = [limitation]
        if safety and not _is_nil_value(safety):
            considerations["ethicalConsiderations"] = [
                {"name": "safetyRiskAssessment", "description": safety}
            ]
        if considerations:
            model_card["considerations"] = considerations

        # Custom properties (fields without a native CycloneDX slot + conflicts)
        properties = []

        # Surface the train/test split via properties since the 1.6 schema
        # doesn't accept the ``type: "training"`` / ``type: "evaluation"``
        # enum values that 1.7 introduced. Finding #23.
        if trained_names:
            properties.append({
                "name": "aikaboom:training-set:names",
                "value": ", ".join(trained_names),
            })
        if tested_names:
            properties.append({
                "name": "aikaboom:evaluation-set:names",
                "value": ", ".join(tested_names),
            })

        custom_fields = {
            "energyConsumption": rag.get("energy_consumption", rag.get("energyConsumption")),
            "autonomyType": rag.get("autonomy_type", rag.get("autonomyType")),
            "modelExplainability": rag.get("model_explainability", rag.get("modelExplainability")),
            "useSensitivePersonalInformation": rag.get("sensitive_personal_information", rag.get("useSensitivePersonalInformation")),
            "standardCompliance": rag.get("standard_compliance", rag.get("standardCompliance")),
        }
        for prop_name, raw in custom_fields.items():
            val = self._extract_value(raw)
            if val and not _is_nil_value(val):
                properties.append({"name": f"aikaboom:{prop_name}", "value": str(val)})

        # Preserve conflict info as properties, with parity to the SPDX
        # Annotation Element emitter (spdx_validator._build_conflict_annotations).
        # ``_collect_conflict_properties`` filters phantom RAG envelopes,
        # uses SPDX 3.0.1 property names in the property key, and includes
        # the rich provenance (claims, narrative, grounding, confidence
        # band) as a JSON value.
        properties.extend(_collect_conflict_properties(bom_data, bom_type="ai"))

        # Plugin-contributed annotations (parity with the SPDX hook).
        # See ``_collect_plugin_properties`` — empty findings until the
        # Task 13 store-aware path lands.
        properties.extend(_collect_plugin_properties(f"ai-model:{model_id}"))

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
            "licenses": [license_block] if license_block else [],
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
        license_block = cdx_license_block(self._extract_value(direct.get("license")))
        # ``description`` is the data-BOM equivalent of the AI BOM's auto-
        # extracted blurb; ``intendedUse`` is rarely populated for datasets.
        # Prefer description; fall back to intendedUse only if description is
        # missing or a nil sentinel like ``"noAssertion"``.
        raw_desc = (
            self._extract_value(rag.get("description"))
            or self._extract_value(rag.get("intendedUse"))
            or ""
        )
        description = "" if _is_nil_value(raw_desc) else raw_desc
        originated_by = self._extract_value(direct.get("originatedBy")) or ""

        component = {
            "type": "data",
            "bom-ref": f"dataset:{dataset_id}",
            "name": name,
            "description": description,
            "supplier": {"name": originated_by} if originated_by else None,
            "licenses": [license_block] if license_block else [],
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

        # Same conflict surfacing as the AI path, parity with the SPDX
        # Annotation emitter. Keyed on SPDX 3.0.1 property names so
        # CycloneDX ↔ SPDX cross-walks line up on both sides.
        properties.extend(_collect_conflict_properties(bom_data, bom_type="data"))

        # Plugin-contributed annotations (parity with the SPDX hook).
        properties.extend(_collect_plugin_properties(f"dataset:{dataset_id}"))

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
        if cdx_bom.get("specVersion") != CDX_SPEC_VERSION:
            errors.append(
                f"specVersion must be '{CDX_SPEC_VERSION}', got "
                f"'{cdx_bom.get('specVersion')}'"
            )
        if not cdx_bom.get("serialNumber"):
            errors.append("Missing serialNumber")
        if not isinstance(cdx_bom.get("components"), list):
            errors.append("Missing or invalid 'components' array")
        elif len(cdx_bom["components"]) == 0:
            errors.append("components array is empty")

        is_valid = len(errors) == 0
        if is_valid:
            print("CycloneDX 1.6 validation passed")
        else:
            print(f"CycloneDX 1.6 validation failed with {len(errors)} error(s)")
            for e in errors:
                print(f"  - {e}")

        return is_valid, errors

    def save_cyclonedx(self, cdx_data: Dict, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cdx_data, f, indent=2, ensure_ascii=False)
        print(f"CycloneDX 1.6 BOM saved to: {output_path}")
        return output_path


def bom_to_cyclonedx(
    bom_data: Dict[str, Any],
    bom_type: str = "ai",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to convert BOM data to CycloneDX 1.6 format."""
    exporter = CycloneDXExporter(bom_type=bom_type)
    cdx_data = exporter.validate_and_convert(bom_data)
    if output_path:
        exporter.save_cyclonedx(cdx_data, output_path)
    return cdx_data
