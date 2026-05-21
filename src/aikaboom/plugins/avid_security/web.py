"""AVID-security Flask blueprint: /avid-security/<artifact_id>."""
from __future__ import annotations

import json
from urllib.parse import unquote

from flask import Blueprint, jsonify

from aikaboom.plugins import Scope


def build_blueprint(plugin) -> Blueprint:
    """Build the avid-security Flask blueprint bound to ``plugin``.

    Routes:
      * ``GET /avid-security/``                        — index: snapshot status (sha, fetched_at).
      * ``GET /avid-security/<artifact_id>``           — per-artifact JSON findings.
      * ``GET /avid-security/<artifact_id>/overlay.json`` — graph overlay payload.

    ``artifact_id`` is the URL-encoded IRI of the artifact to analyse.
    Mirrors license_compat/web.py: overlay route registered first, then .json, then HTML.
    """
    bp = Blueprint(
        "avid_security",
        __name__,
        url_prefix="/avid-security",
    )

    def _snapshot_info() -> dict:
        marker_path = plugin.cache_dir / "snapshot.json"
        if marker_path.exists():
            return json.loads(marker_path.read_text())
        return {"sha": "unknown", "fetched_at": None}

    def _analyse(artifact_id: str):
        from aikaboom.store import BomStore

        store = BomStore.open()
        iri = unquote(artifact_id)
        findings = plugin.analyze(store, Scope.single(iri))
        return findings

    @bp.get("/")
    def index():
        info = _snapshot_info()
        return jsonify({
            "plugin": plugin.name,
            "snapshot_sha": info.get("sha", "unknown"),
            "fetched_at": info.get("fetched_at"),
            "ttl_days": plugin.ttl_days,
        })

    # Overlay route registered FIRST so the more-specific ``/overlay.json``
    # suffix is matched before the generic ``<path:artifact_id>`` swallows it.
    @bp.get("/<path:artifact_id>/overlay.json")
    def overlay_json(artifact_id):
        findings = _analyse(artifact_id)
        overlay = plugin.graph_overlay(findings)
        return jsonify({
            "plugin": overlay.plugin_name,
            "edges": overlay.edge_attrs,
            "nodes": overlay.node_attrs,
        })

    @bp.get("/<path:artifact_id>")
    def view(artifact_id):
        findings = _analyse(artifact_id)
        info = _snapshot_info()
        return jsonify({
            **findings.to_dict(),
            "snapshot_sha": info.get("sha", "unknown"),
        })

    return bp
