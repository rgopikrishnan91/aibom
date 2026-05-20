"""License-compat CLI: license-check + license-audit subparsers and handlers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from aikaboom.plugins import Scope


def register_cli(parent_subparsers: "argparse._SubParsersAction", plugin) -> None:
    p_check = parent_subparsers.add_parser(
        "license-check",
        help="Check license compatibility for one artifact and its lineage",
    )
    p_check.add_argument("artifact", help="Artifact IRI, label, or platform id")
    p_check.add_argument("--depth", type=int, default=5)
    p_check.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_check.add_argument("--matrix", type=Path, default=None)
    p_check.add_argument("--violations-only", action="store_true")
    p_check.set_defaults(func=lambda args: _cmd_check(args, plugin))

    p_audit = parent_subparsers.add_parser(
        "license-audit",
        help="Sweep the entire stored graph for license-compat violations",
    )
    p_audit.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_audit.add_argument("--matrix", type=Path, default=None)
    p_audit.add_argument("--out", type=Path, default=None)
    p_audit.set_defaults(func=lambda args: _cmd_audit(args, plugin))


def _open_store():
    from aikaboom.store import BomStore
    return BomStore.open()


def _resolve_artifact_iri(store, candidate: str) -> Optional[str]:
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    # Try BomStore.resolve() if available; otherwise fall back to label match.
    if hasattr(store, "resolve"):
        try:
            r = store.resolve(candidate)
            if r and getattr(r, "artifact_iri", None):
                return r.artifact_iri
        except Exception:
            pass
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?a WHERE {{ ?a aibom:canonicalLabel "{candidate}" }} LIMIT 2
    """))
    if len(rows) == 1:
        return str(rows[0]["a"])
    return None


def _override_matrix(plugin, override: Optional[Path]) -> None:
    if override is not None:
        from aikaboom.plugins.license_compat.matrix import load_matrix
        plugin._matrix_cache = load_matrix(matrix_path=override)


def _cmd_check(args: argparse.Namespace, plugin) -> int:
    _override_matrix(plugin, args.matrix)
    store = _open_store()
    iri = _resolve_artifact_iri(store, args.artifact)
    if iri is None:
        print(f"Artifact not found: {args.artifact}", file=sys.stderr)
        return 3
    findings = plugin.analyze(store, Scope.single(iri, depth=args.depth))
    return _render_and_exit(findings, args, plugin)


def _cmd_audit(args: argparse.Namespace, plugin) -> int:
    _override_matrix(plugin, args.matrix)
    store = _open_store()
    findings = plugin.analyze(store, Scope.graph_wide())
    if args.out is not None:
        with args.out.open("w", encoding="utf-8") as fh:
            for item in findings.to_dict()["findings"]:
                fh.write(json.dumps(item) + "\n")
    return _render_and_exit(findings, args, plugin)


def _render_and_exit(findings, args, plugin) -> int:
    from aikaboom.plugins.license_compat.engine import (
        find_breaking_nodes,
        find_compatible_subchains,
    )

    matrix = plugin._matrix()
    from aikaboom.plugins.license_compat.walker import compute_license_frequencies
    store = _open_store()
    freqs = compute_license_frequencies(store, matrix)
    subchains = find_compatible_subchains(findings)
    breaking = find_breaking_nodes(findings, matrix, freqs)

    if args.format == "json":
        payload = {
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
        }
        print(json.dumps(payload, indent=2))
    elif args.format == "jsonl":
        for item in findings.to_dict()["findings"]:
            print(json.dumps(item))
    else:
        _render_text(findings, subchains, breaking, args)

    return 2 if findings.violations() else 0


def _render_text(findings, subchains, breaking, args) -> None:
    items = findings.violations() if args.violations_only else list(findings)
    if not items:
        print("No findings.")
    for f in items:
        marker = {"compatible": "OK ", "violation": "X  ", "unknown_upstream": "?  ",
                  "unknown_downstream": "?  ", "missing_data": "-  "}[f.verdict.status]
        pred = f.predicate.rsplit("#", 1)[-1]
        print(f"  {marker}{f.downstream_label} ({f.downstream_license}) "
              f"--{pred}--> {f.upstream_label} ({sorted(f.upstream_licenses)})   "
              f"{f.verdict.status.upper()}")
        if f.recommendation and f.recommendation.is_solvable:
            for cat, lics in f.recommendation.by_category.items():
                print(f"        {cat}: {', '.join(lics)}")
    if subchains:
        print(f"\nCompatible subchains ({len(subchains)}):")
        for i, c in enumerate(subchains, 1):
            print(f"  {i}. size={c.size} root={c.root}")
    if breaking:
        print(f"\nBreaking nodes ({len(breaking)}):")
        for n in breaking:
            print(f"  - {n.label} ({n.license})  blamed in {n.blamed_in}  "
                  f"affected {len(n.affected_downstream)} downstream")
    summary = {
        "edges": len(list(findings)),
        "compatible": sum(1 for f in findings if f.verdict.status == "compatible"),
        "violations": len(findings.violations()),
    }
    print(f"\nSummary: {summary['edges']} edges  |  "
          f"{summary['compatible']} compatible  |  "
          f"{summary['violations']} violations")
