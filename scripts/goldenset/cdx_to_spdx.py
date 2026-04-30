#!/usr/bin/env python3
"""Map a CycloneDX 1.6 AI BOM (as emitted by OWASP aibom-generator) to an
SPDX 3.0.1 JSON-LD document with the AI Profile.

Mapping follows the field-mapping doc:
  https://github.com/GenAI-Security-Project/aibom-generator/blob/main/docs/aibom-field-mapping/README.md

Only fields covered by that doc are emitted. Missing source data is skipped
rather than synthesized.
"""

import argparse
import json
import pathlib
import sys
import uuid
from typing import Any, Dict, List, Optional


def _get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
            cur = cur[p]
        else:
            return default
    return cur


def _component(cdx: dict) -> dict:
    """Return the ML model component from the CycloneDX BOM.

    ALOHA puts the model directly in metadata.component (with modelCard).
    aibom-generator puts a job wrapper in metadata.component and the actual
    model in the components[] array (type=machine-learning-model, has modelCard).
    Prefer the component that carries a modelCard; fall back to metadata.component.
    """
    meta_comp = _get(cdx, "metadata", "component", default={}) or {}
    if meta_comp.get("modelCard"):
        return meta_comp
    # Search components[] for an ML model with a modelCard
    for comp in cdx.get("components", []) or []:
        if comp.get("modelCard"):
            return comp
        if comp.get("type") == "machine-learning-model":
            return comp
    return meta_comp


def _props(node: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in node.get("properties", []) or []:
        n = p.get("name")
        if n:
            out.setdefault(n, p.get("value"))
    return out


def _licenses(comp: dict) -> List[str]:
    ids: List[str] = []
    for lic in comp.get("licenses", []) or []:
        node = lic.get("license") or {}
        if node.get("id"):
            ids.append(node["id"])
        elif node.get("name"):
            ids.append(node["name"])
        elif lic.get("expression"):
            ids.append(lic["expression"])
    return ids


def _ext_ref(comp: dict, ref_type: str) -> Optional[str]:
    for r in comp.get("externalReferences", []) or []:
        if r.get("type") == ref_type:
            return r.get("url")
    return None


def cdx_to_spdx(cdx: dict) -> dict:
    """Convert a CycloneDX 1.6 AI BOM dict to an SPDX 3.0.1 JSON-LD dict."""
    comp = _component(cdx)
    mc = comp.get("modelCard") or {}
    mparams = mc.get("modelParameters") or {}
    cprops = _props(comp)
    mprops = _props(mc)

    base = (comp.get("name") or "model").replace("/", "_")
    pkg_id = f"urn:spdx:{base}-{uuid.uuid4()}"
    ai_id = f"urn:spdx:ai-{base}-{uuid.uuid4()}"

    # AI package: fields from the AI Profile
    ai_pkg: Dict[str, Any] = {
        "type": "ai_AIPackage",
        "spdxId": ai_id,
        "name": comp.get("name"),
    }

    # Direct mappings per field-mapping table
    if comp.get("description"):
        ai_pkg["summary"] = comp["description"]
    if _ext_ref(comp, "distribution"):
        ai_pkg["downloadLocation"] = _ext_ref(comp, "distribution")
    if _ext_ref(comp, "vcs"):
        ai_pkg["packageUrl"] = _ext_ref(comp, "vcs")
    elif _ext_ref(comp, "website"):
        ai_pkg["packageUrl"] = _ext_ref(comp, "website")
    if comp.get("supplier"):
        sup = comp["supplier"]
        ai_pkg["suppliedBy"] = sup.get("name") if isinstance(sup, dict) else sup
    if comp.get("version"):
        ai_pkg["packageVersion"] = comp["version"]

    # AI-Profile-specific fields
    if mparams.get("task"):
        ai_pkg["primaryPurpose"] = [mparams["task"]]
    tags = comp.get("tags") or []
    if tags:
        ai_pkg["domain"] = tags
    if mparams.get("approach") or mparams.get("architectureFamily"):
        ai_pkg["typeOfModel"] = [mparams.get("approach") or mparams.get("architectureFamily")]
    # Hyperparameters: collect anything in modelParameters that isn't a structural field
    structural = {"task", "approach", "architectureFamily", "modelArchitecture", "datasets", "inputs", "outputs"}
    hp = {k: v for k, v in mparams.items() if k not in structural and isinstance(v, (str, int, float, bool))}
    if hp:
        ai_pkg["hyperparameter"] = [{"key": k, "value": str(v)} for k, v in hp.items()]
    qa = mc.get("quantitativeAnalysis") or {}
    metrics = qa.get("performanceMetrics") or []
    if metrics:
        ai_pkg["metric"] = [
            {"key": m.get("type") or m.get("name") or "metric",
             "value": str(m.get("value", ""))} for m in metrics
        ]
    if mprops.get("metricDecisionThreshold"):
        ai_pkg["metricDecisionThreshold"] = [
            {"key": "threshold", "value": str(mprops["metricDecisionThreshold"])}
        ]
    if mprops.get("energyConsumption") or comp.get("environmentalConsiderations"):
        ai_pkg["energyConsumption"] = (
            mprops.get("energyConsumption")
            or json.dumps(comp.get("environmentalConsiderations"))
        )
    formulation = cdx.get("formulation") or comp.get("formulation")
    if formulation:
        ai_pkg["informationAboutTraining"] = (
            json.dumps(formulation) if not isinstance(formulation, str) else formulation
        )
        ai_pkg["modelDataPreprocessing"] = ai_pkg["informationAboutTraining"]
    if mprops.get("useSensitivePersonalInformation"):
        ai_pkg["useSensitivePersonalInformation"] = mprops["useSensitivePersonalInformation"]
    if mprops.get("modelExplainability"):
        ai_pkg["modelExplainability"] = mprops["modelExplainability"]
    # Limitations / safety
    cons = mc.get("considerations") or {}
    lim = cons.get("technicalLimitations") or mprops.get("limitation")
    if lim:
        ai_pkg["limitation"] = lim if isinstance(lim, str) else json.dumps(lim)
    risk = cons.get("ethicalConsiderations") or mprops.get("safetyRiskAssessment")
    if risk:
        ai_pkg["safetyRiskAssessment"] = risk if isinstance(risk, str) else json.dumps(risk)

    # Standard SPDX Package wrapper (license + identity)
    pkg: Dict[str, Any] = {
        "type": "software_Package",
        "spdxId": pkg_id,
        "name": comp.get("name"),
    }
    if comp.get("version"):
        pkg["packageVersion"] = comp["version"]
    if _ext_ref(comp, "distribution"):
        pkg["downloadLocation"] = _ext_ref(comp, "distribution")
    licenses = _licenses(comp)
    if licenses:
        pkg["licenseDeclared"] = " AND ".join(licenses) if len(licenses) > 1 else licenses[0]

    relationships: List[dict] = []
    # License is an explicit relationship in SPDX 3
    for lic_id in licenses:
        relationships.append({
            "type": "Relationship",
            "spdxId": f"urn:spdx:rel-{uuid.uuid4()}",
            "relationshipType": "hasDeclaredLicense",
            "from": pkg_id,
            "to": [lic_id],
        })
    # Lineage: pedigree.ancestors → ancestorOf
    for anc in _get(comp, "pedigree", "ancestors", default=[]) or []:
        a_id = anc.get("bom-ref") or anc.get("name")
        if not a_id:
            continue
        relationships.append({
            "type": "Relationship",
            "spdxId": f"urn:spdx:rel-{uuid.uuid4()}",
            "relationshipType": "ancestorOf",
            "from": a_id,
            "to": [ai_id],
        })
    # Datasets: modelCard.datasets → trainedOn (default; tested-on if marked)
    for ds in mparams.get("datasets", []) or mc.get("datasets", []) or []:
        ref = ds.get("ref") or ds.get("name")
        if not ref:
            continue
        rel_type = "testedOn" if (ds.get("type") or "").lower() == "test" else "trainedOn"
        relationships.append({
            "type": "Relationship",
            "spdxId": f"urn:spdx:rel-{uuid.uuid4()}",
            "relationshipType": rel_type,
            "from": ai_id,
            "to": [ref],
        })
    # Runtime dependencies → dependsOn
    for dep in cdx.get("dependencies", []) or []:
        src = dep.get("ref")
        for tgt in dep.get("dependsOn", []) or []:
            relationships.append({
                "type": "Relationship",
                "spdxId": f"urn:spdx:rel-{uuid.uuid4()}",
                "relationshipType": "dependsOn",
                "from": src,
                "to": [tgt],
            })

    doc_id = f"urn:spdx:doc-{base}-{uuid.uuid4()}"
    doc: Dict[str, Any] = {
        "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
        "@graph": [
            {
                "type": "SpdxDocument",
                "spdxId": doc_id,
                "specVersion": "SPDX-3.0.1",
                "profileConformance": ["core", "software", "ai"],
                "name": f"AIBOM for {comp.get('name', 'unknown')}",
                "creationInfo": {
                    "type": "CreationInfo",
                    "created": cdx.get("metadata", {}).get("timestamp"),
                    "createdBy": ["Tool-cdx_to_spdx.py"],
                },
                "rootElement": [ai_id],
                "element": [pkg_id, ai_id] + [r["spdxId"] for r in relationships],
            },
            pkg,
            ai_pkg,
            *relationships,
        ],
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory of CycloneDX 1.6 JSON files")
    ap.add_argument("--out-dir", required=True, help="Where to write SPDX JSON-LD files")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    for src in sorted(in_dir.glob("*.json")):
        try:
            cdx = json.loads(src.read_text())
            spdx = cdx_to_spdx(cdx)
            dst = out_dir / src.name.replace("_ai_sbom_1_6", "_spdx_3_0_1")
            dst.write_text(json.dumps(spdx, indent=2))
            n_ok += 1
        except Exception as e:
            print(f"FAIL {src.name}: {e}", file=sys.stderr)
            n_fail += 1
    print(f"SPDX mapping: {n_ok} ok, {n_fail} failed → {out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
