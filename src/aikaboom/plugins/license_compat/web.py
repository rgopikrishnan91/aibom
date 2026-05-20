"""License-compat Flask blueprint: /license-compat/<artifact_id>."""
from __future__ import annotations

from urllib.parse import unquote

from flask import Blueprint, jsonify, render_template

from aikaboom.plugins import Scope
from aikaboom.plugins.license_compat.engine import (
    find_breaking_nodes,
    find_compatible_subchains,
)


def build_blueprint(plugin) -> Blueprint:
    """Build the license-compat Flask blueprint bound to ``plugin``.

    Routes:
      * ``GET /license-compat/<artifact_id>``       — HTML tab.
      * ``GET /license-compat/<artifact_id>.json``  — JSON payload for
        the same analysis, suitable for the BOM-viewer tab or scripts.

    ``artifact_id`` is the URL-encoded IRI of the artifact to analyse.
    """
    bp = Blueprint(
        "license_compat",
        __name__,
        url_prefix="/license-compat",
        template_folder="templates",
    )

    def _analyse(artifact_id: str):
        # Local import keeps module import cheap; the store/walker chain
        # only fires when a request actually lands.
        from aikaboom.store import BomStore
        from aikaboom.plugins.license_compat.walker import compute_license_frequencies

        store = BomStore.open()
        iri = unquote(artifact_id)
        findings = plugin.analyze(store, Scope.single(iri))
        matrix = plugin._matrix()
        freqs = compute_license_frequencies(store, matrix)
        subchains = find_compatible_subchains(findings)
        breaking = find_breaking_nodes(findings, matrix, freqs)
        return findings, subchains, breaking, matrix

    # JSON route is registered FIRST so Flask's path-converter doesn't
    # greedily swallow ``.json`` into the HTML route's ``artifact_id``.
    @bp.get("/<path:artifact_id>.json")
    def view_json(artifact_id):
        findings, subchains, breaking, _ = _analyse(artifact_id)
        return jsonify({
            **findings.to_dict(),
            "compatible_subchains": [
                {"size": c.size, "root": c.root, "artifacts": sorted(c.artifacts)}
                for c in subchains
            ],
            "breaking_nodes": [
                {
                    "artifact_iri": n.artifact_iri,
                    "label": n.label,
                    "license": n.license,
                    "blamed_in": n.blamed_in,
                    "affected_downstream": sorted(n.affected_downstream),
                    "fix_recommendations": {
                        "by_category": n.fix_recommendations.by_category,
                        "is_solvable": n.fix_recommendations.is_solvable,
                    },
                }
                for n in breaking
            ],
        })

    @bp.get("/<path:artifact_id>")
    def view(artifact_id):
        findings, subchains, breaking, matrix = _analyse(artifact_id)
        return render_template(
            "license_compat/tab.html",
            findings=list(findings),
            subchains=subchains,
            breaking=breaking,
            matrix_timestamp=matrix.timestamp,
            artifact_iri=unquote(artifact_id),
        )

    return bp
