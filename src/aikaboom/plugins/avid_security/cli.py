"""AVID-security CLI: avid-status + avid-refresh + avid-scan subparsers and handlers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register_cli(parent_subparsers: "argparse._SubParsersAction", plugin) -> None:
    """Register avid-status, avid-refresh, avid-scan under parent_subparsers.

    Mirrors license_compat/cli.py: flat subcommands added directly to parent,
    with the same ``register_cli(parent_subparsers, plugin)`` signature.
    """
    p_status = parent_subparsers.add_parser(
        "avid-status",
        help="Print the current AVID snapshot marker (sha, fetched_at, ttl_days)",
    )
    p_status.set_defaults(func=lambda args: _cmd_status(args, plugin))

    p_refresh = parent_subparsers.add_parser(
        "avid-refresh",
        help="Force a fresh AVID snapshot clone, ignoring TTL",
    )
    p_refresh.set_defaults(func=lambda args: _cmd_refresh(args, plugin))

    p_scan = parent_subparsers.add_parser(
        "avid-scan",
        help="Scan all BOM components against AVID (graph-wide); "
             "runs through plugin.analyze() against the live store",
    )
    p_scan.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_scan.add_argument("--out", type=Path, default=None)
    p_scan.set_defaults(func=lambda args: _cmd_scan(args, plugin))


def _cmd_status(args: argparse.Namespace, plugin) -> int:
    marker_path = Path(plugin.cache_dir) / "snapshot.json"
    if not marker_path.exists():
        print("No snapshot cached yet. Run 'avid-refresh' to fetch.", file=sys.stderr)
        return 1
    marker = json.loads(marker_path.read_text())
    print(json.dumps(marker, indent=2))
    return 0


def _cmd_refresh(args: argparse.Namespace, plugin) -> int:
    from aikaboom.plugins.avid_security.snapshot import AvidSnapshot
    snap = AvidSnapshot(cache_dir=Path(plugin.cache_dir), ttl_days=plugin.ttl_days)
    snap.force_refresh()
    marker = json.loads(snap.marker_path.read_text())
    print(f"Refreshed: sha={marker['sha']}  fetched_at={marker['fetched_at']}")
    return 0


def _cmd_scan(args: argparse.Namespace, plugin) -> int:
    """Graph-wide AVID scan using the live BomStore."""
    from aikaboom.store import BomStore
    from aikaboom.plugins import Scope

    store = BomStore.open()
    findings = plugin.analyze(store, Scope.graph_wide())

    if args.out is not None:
        with args.out.open("w", encoding="utf-8") as fh:
            for item in findings.to_dict()["findings"]:
                fh.write(json.dumps(item) + "\n")
        return 2 if findings.violations() else 0

    if args.format == "json":
        print(json.dumps(findings.to_dict(), indent=2))
    elif args.format == "jsonl":
        for item in findings.to_dict()["findings"]:
            print(json.dumps(item))
    else:
        _render_text(findings)

    return 2 if findings.violations() else 0


def _render_text(findings) -> None:
    items = list(findings)
    if not items:
        print("No AVID findings.")
        return
    for f in items:
        tier_label = {1: "AFFECTED", 2: "UNDER-INVESTIGATION", 3: "ADVISORY"}.get(f.tier, f"TIER-{f.tier}")
        print(f"  [{tier_label}] {f.component_label} → {f.avid_report_id} "
              f"(tier {f.tier}, {f.confidence}, via {f.matched_via})")
    violations = findings.violations()
    print(f"\nSummary: {len(items)} findings  |  {len(violations)} affected (tier-1)")
