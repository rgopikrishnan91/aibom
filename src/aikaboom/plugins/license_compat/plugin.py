"""LicenseCompatPlugin — wires the engine + walker + emitters into the plugin contract."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from aikaboom.plugins import ConflictRecord, GraphOverlay, Scope, TabSpec
from aikaboom.plugins.license_compat.engine import (
    Finding,
    Findings,
    Recommendation,
    check_compat,
    find_breaking_nodes,
    find_compatible_subchains,
    recommend,
)
from aikaboom.plugins.license_compat.matrix import LicenseMatrix, load_matrix


class LicenseCompatPlugin:
    name = "license-compat"

    def __init__(self, matrix: Optional[LicenseMatrix] = None):
        self._matrix_override: Optional[Path] = None
        self._matrix_cache: Optional[LicenseMatrix] = matrix

    def enabled(self) -> bool:
        return os.environ.get("AIKABOOM_LICENSE_COMPAT_DISABLED", "").lower() not in ("1", "true", "yes")

    def _matrix(self) -> LicenseMatrix:
        if self._matrix_cache is None:
            override = os.environ.get("AIKABOOM_LICENSE_MATRIX")
            self._matrix_cache = load_matrix(matrix_path=Path(override) if override else None)
        return self._matrix_cache

    def analyze(self, store, scope: Scope) -> Findings:
        from aikaboom.plugins.license_compat.walker import (
            compute_license_frequencies,
            enumerate_edges,
            resolve_artifact_license,
        )

        matrix = self._matrix()
        freqs = compute_license_frequencies(store, matrix)
        findings: list[Finding] = []
        for edge in enumerate_edges(store, scope):
            d = resolve_artifact_license(store, edge.downstream_iri, matrix)
            u = resolve_artifact_license(store, edge.upstream_iri, matrix)
            d_licenses = d.licenses or frozenset({None})
            for d_lic in d_licenses:
                verdict = check_compat(d_lic, u.licenses, matrix)
                rec = (
                    recommend(u.licenses, matrix, freqs)
                    if verdict.status == "violation"
                    else None
                )
                findings.append(
                    Finding(
                        downstream_iri=edge.downstream_iri,
                        downstream_label=edge.downstream_label,
                        upstream_iri=edge.upstream_iri,
                        upstream_label=edge.upstream_label,
                        predicate=edge.predicate,
                        downstream_license=d_lic,
                        upstream_licenses=u.licenses,
                        verdict=verdict,
                        recommendation=rec,
                    )
                )
        return Findings(findings)

    def register_cli(self, parent_subparsers) -> None:
        from aikaboom.plugins.license_compat.cli import register_cli as _register
        _register(parent_subparsers, self)

    def web_blueprint(self):
        from aikaboom.plugins.license_compat.web import build_blueprint
        return build_blueprint(self)

    def bom_viewer_tab(self) -> Optional[TabSpec]:
        return TabSpec(
            label="License compatibility",
            url_template="/license-compat/{artifact_id}",
            sort_order=50,
        )

    def spdx_annotations(self, claim_iri: str, findings: Findings) -> list[dict]:
        from aikaboom.plugins.license_compat.spdx import emit_annotations
        return emit_annotations(claim_iri, findings, matrix=self._matrix())

    def spdx_elements(self, claim_iri: str, findings: Findings) -> list[dict]:
        # license_compat emits only Annotations; no arbitrary SPDX elements.
        return []

    def graph_overlay(self, findings: Findings) -> GraphOverlay:
        from aikaboom.plugins.license_compat.overlay import build_overlay
        return build_overlay(findings, plugin_name=self.name)

    def conflict_findings(self, findings: Findings) -> list[ConflictRecord]:
        records: list[ConflictRecord] = []
        for f in findings.violations():
            records.append(ConflictRecord(
                category="license-compat",
                severity="high",
                subject_iri=f.downstream_iri,
                title=f"License {f.downstream_license} ↛ {sorted(f.verdict.incompatible_with)}",
                detail=f"Incompatible via {f.predicate.rsplit('#', 1)[-1]}",
                data={
                    "predicate": f.predicate,
                    "upstream": f.upstream_iri,
                    "incompatible_with": sorted(f.verdict.incompatible_with),
                },
            ))
        return records
