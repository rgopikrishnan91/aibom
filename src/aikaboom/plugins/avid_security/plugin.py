"""AvidSecurityPlugin — wires snapshot+matcher+walker+emitter into the plugin contract."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from aikaboom.plugins import ConflictRecord, GraphOverlay, Scope, TabSpec
from aikaboom.plugins.avid_security.engine import AvidFinding, AvidFindings

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aikaboom" / "avid"


class AvidSecurityPlugin:
    name = "avid-security"

    def __init__(self, cache_dir: Optional[Path] = None, ttl_days: int = 10):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(
            os.environ.get("AIKABOOM_AVID_CACHE", DEFAULT_CACHE_DIR)
        )
        self.ttl_days = ttl_days

    def enabled(self) -> bool:
        return os.environ.get("AIKABOOM_AVID_DISABLED", "").lower() not in ("1", "true", "yes")

    def analyze(self, store, scope: Scope) -> AvidFindings:
        import json
        from aikaboom.plugins.avid_security.snapshot import AvidSnapshot, AvidIndex
        from aikaboom.plugins.avid_security.matcher import ComponentMatcher
        from aikaboom.plugins.avid_security.walker import walk_components

        snapshot = AvidSnapshot(cache_dir=self.cache_dir, ttl_days=self.ttl_days)
        snapshot.ensure_fresh()
        marker = json.loads(snapshot.marker_path.read_text())
        snapshot_sha = marker.get("sha", "unknown")

        index = AvidIndex(db_path=self.cache_dir / "avid.sqlite")
        if not index.db_path.exists():
            index.build(repo_dir=snapshot.repo_dir)
        matcher = ComponentMatcher(index)

        items: list[AvidFinding] = []
        for component in walk_components(store, scope):
            for m in matcher.match(component):
                items.append(AvidFinding(
                    component_iri=component.spdx_id,
                    component_label=component.hf_path,
                    avid_report_id=m.avid_report["report_id"],
                    tier=m.tier,
                    confidence=m.confidence,
                    matched_via=m.evidence.get("matched_via", ""),
                    match=m,
                ))
        return AvidFindings(items, snapshot_sha=snapshot_sha)

    def register_cli(self, parent_subparsers) -> None:
        from aikaboom.plugins.avid_security.cli import register_cli as _register
        _register(parent_subparsers, self)

    def web_blueprint(self):
        from aikaboom.plugins.avid_security.web import build_blueprint
        return build_blueprint(self)

    def bom_viewer_tab(self) -> Optional[TabSpec]:
        return TabSpec(
            label="AVID Security",
            url_template="/avid-security/{artifact_id}",
            sort_order=60,
        )

    def spdx_annotations(self, claim_iri: str, findings) -> list[dict]:
        # avid_security emits first-class Security-profile elements, not Annotations.
        return []

    def spdx_elements(self, claim_iri: str, findings) -> list[dict]:
        from aikaboom.plugins.avid_security.spdx import emit_security_elements
        matches = findings.matches() if hasattr(findings, "matches") else []
        if not matches:
            return []
        sha = getattr(findings, "snapshot_sha", "unknown")
        return emit_security_elements(matches, snapshot_sha=sha)

    def graph_overlay(self, findings) -> GraphOverlay:
        from aikaboom.plugins.avid_security.overlay import build_overlay
        return build_overlay(findings, plugin_name=self.name)

    def conflict_findings(self, findings) -> list[ConflictRecord]:
        records: list[ConflictRecord] = []
        for f in findings.violations():  # tier-1 / affected only
            records.append(ConflictRecord(
                category="avid-security",
                severity="high",
                subject_iri=f.component_iri,
                title=f"AVID {f.avid_report_id} affects {f.component_label}",
                detail=f"Exact-match AVID report (tier {f.tier}, {f.confidence} confidence)",
                data={"avid_report_id": f.avid_report_id, "matched_via": f.matched_via},
            ))
        return records
