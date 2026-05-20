"""aikaboom graph / aikaboom bom subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore, _validate_sparql_iri
from aikaboom.store.trust import VoteKind


def cmd_graph_stats(args: argparse.Namespace) -> int:
    store = BomStore.open()
    stats = store.stats()
    print(json.dumps(stats, indent=2))
    return 0


def cmd_graph_export(args: argparse.Namespace) -> int:
    store = BomStore.open()
    fmt = args.format
    store._backend.export(Path(args.path), fmt=fmt)
    print(f"Exported to {args.path} ({fmt}).")
    return 0


def cmd_graph_import(args: argparse.Namespace) -> int:
    store = BomStore.open()
    store._backend.import_(Path(args.path), fmt=args.format)
    print(f"Imported {args.path}.")
    return 0


def cmd_graph_query(args: argparse.Namespace) -> int:
    store = BomStore.open()
    rows = list(store._backend.select(args.sparql))
    for row in rows:
        print(json.dumps({k: str(v) for k, v in row.items()}))
    return 0


def cmd_bom_trust(args: argparse.Namespace) -> int:
    store = BomStore.open()
    store.record_trust_vote(args.claim_iri, VoteKind.TRUSTED)
    store.recompute_canonical_for_claim(args.claim_iri)
    print(f"Recorded TRUSTED vote on {args.claim_iri}.")
    return 0


def cmd_bom_flag(args: argparse.Namespace) -> int:
    store = BomStore.open()
    store.record_trust_vote(args.claim_iri, VoteKind.FLAGGED)
    store.recompute_canonical_for_claim(args.claim_iri)
    print(f"Recorded FLAGGED vote on {args.claim_iri}.")
    return 0


def cmd_bom_claims(args: argparse.Namespace) -> int:
    store = BomStore.open()
    platform, _, value = args.identifier.partition(":")
    claims = store.find_claims_for(
        [Identifier(platform or "huggingface", value or args.identifier)],
        use_case=args.use_case,
        mode=args.mode,
    )
    print(json.dumps(claims, indent=2))
    return 0


def cmd_bom_dispute(args: argparse.Namespace) -> int:
    store = BomStore.open()
    store.record_trust_vote(args.claim_iri, VoteKind.DISPUTED)
    store.recompute_canonical_for_claim(args.claim_iri)
    print(f"Recorded DISPUTED vote on {args.claim_iri}.")
    return 0


def cmd_bom_votes(args: argparse.Namespace) -> int:
    """List votes on a claim."""
    from aikaboom.store import vocab

    store = BomStore.open()
    q = f"""
    SELECT ?vote ?kind ?at ?agent WHERE {{
        ?vote <{vocab.trustVoteFor}> <{args.claim_iri}> ;
              <{vocab.voteKind}> ?kind ;
              <{vocab.votedAt}> ?at ;
              <{vocab.votedBy}> ?agent .
    }}
    ORDER BY DESC(?at)
    """
    rows = [{k: str(v) for k, v in row.items()} for row in store._backend.select(q)]
    print(json.dumps(rows, indent=2))
    return 0


def cmd_bom_show(args: argparse.Namespace) -> int:
    """Reconstruct and pretty-print a claim's BOM JSON."""
    store = BomStore.open()
    bom = store.reconstruct_bom(args.claim_iri)
    print(json.dumps(bom, indent=2, ensure_ascii=False))
    return 0


def cmd_bom_diff(args: argparse.Namespace) -> int:
    """Field-level diff between two claims."""
    store = BomStore.open()
    a = store.reconstruct_bom(args.claim_a)
    b = store.reconstruct_bom(args.claim_b)
    fields_a = a.get("direct_fields", {})
    fields_b = b.get("direct_fields", {})
    all_fields = sorted(set(fields_a) | set(fields_b))
    diff = []
    for f in all_fields:
        va = fields_a.get(f, {}).get("value")
        vb = fields_b.get(f, {}).get("value")
        if va != vb:
            diff.append({"field": f, "a": va, "b": vb})
    print(json.dumps(diff, indent=2, ensure_ascii=False))
    return 0


def cmd_graph_list(args: argparse.Namespace) -> int:
    """List artifacts with their primary identifier."""
    from aikaboom.store import vocab

    store = BomStore.open()
    q = f"""
    SELECT ?artifact ?label ?primary WHERE {{
        ?artifact a <{vocab.Artifact}> ;
                  <{vocab.primaryIdentifier}> ?primary .
        OPTIONAL {{ ?artifact <{vocab.canonicalLabel}> ?label . }}
    }}
    """
    rows = [{k: str(v) for k, v in row.items()} for row in store._backend.select(q)]
    print(json.dumps(rows, indent=2))
    return 0


def cmd_graph_show(args: argparse.Namespace) -> int:
    """Show all triples with the given IRI as subject."""
    store = BomStore.open()
    iri = _validate_sparql_iri(args.iri)
    q = f"SELECT ?p ?o WHERE {{ <{iri}> ?p ?o }}"
    rows = [{k: str(v) for k, v in row.items()} for row in store._backend.select(q)]
    print(json.dumps(rows, indent=2))
    return 0


def cmd_graph_merge(args: argparse.Namespace) -> int:
    """Merge artifact-b into artifact-a."""
    store = BomStore.open()
    store.merge_artifacts(into=args.artifact_a, from_=args.artifact_b)
    print(f"Merged {args.artifact_b} into {args.artifact_a}.")
    return 0


def cmd_graph_rebuild(args: argparse.Namespace) -> int:
    """Rebuild the graph from results/*.json + replay votes from votes.log."""
    from pathlib import Path
    from aikaboom.store.naming import Identifier

    store = BomStore.open()
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found; nothing to rebuild from.")
        return 1
    count = 0
    for results_file in results_dir.glob("*.json"):
        if ".cyclonedx" in results_file.stem or ".spdx" in results_file.stem:
            continue
        if results_file.stem.endswith(".recursive") or results_file.stem.endswith(".linked"):
            continue
        try:
            data = json.loads(results_file.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "direct_fields" not in data:
            continue
        repo = data.get("repo_id") or data.get("model_id") or results_file.stem
        idents = [Identifier("huggingface", repo)]
        run_meta = {
            "provider": "rebuild",
            "llm_model": "unknown",
            "prompt_version": "rebuild",
            "code_version": "rebuild",
            "mode": data.get("mode", "rag"),
            "use_case": data.get("use_case", "complete"),
        }
        store.save_claim(data, run_meta, identifiers=idents)
        count += 1
    print(f"Rebuilt graph from {count} BOM files.")
    return 0


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Attach the `graph` and `bom` subcommand trees to the main parser."""
    g = subparsers.add_parser("graph", help="Knowledge graph operations")
    g_sub = g.add_subparsers(dest="graph_cmd", required=True)

    g_sub.add_parser("stats", help="Counts of nodes and edges").set_defaults(func=cmd_graph_stats)
    g_sub.add_parser("list", help="List artifacts with their primary identifier").set_defaults(
        func=cmd_graph_list
    )

    p_show = g_sub.add_parser("show", help="Show all triples for a given IRI")
    p_show.add_argument("iri")
    p_show.set_defaults(func=cmd_graph_show)

    p_export = g_sub.add_parser("export", help="Dump the graph to a file")
    p_export.add_argument("path")
    p_export.add_argument("--format", choices=["nquads", "jsonld"], default="nquads")
    p_export.set_defaults(func=cmd_graph_export)

    p_import = g_sub.add_parser("import", help="Merge a dump into the graph")
    p_import.add_argument("path")
    p_import.add_argument("--format", choices=["nquads", "jsonld"], default="nquads")
    p_import.set_defaults(func=cmd_graph_import)

    p_query = g_sub.add_parser("query", help="Run a SPARQL query")
    p_query.add_argument("sparql")
    p_query.set_defaults(func=cmd_graph_query)

    p_merge = g_sub.add_parser("merge", help="Merge artifact-b into artifact-a")
    p_merge.add_argument("artifact_a")
    p_merge.add_argument("artifact_b")
    p_merge.set_defaults(func=cmd_graph_merge)

    g_sub.add_parser("rebuild", help="Rebuild the graph from results/*.json").set_defaults(
        func=cmd_graph_rebuild
    )

    b = subparsers.add_parser("bom", help="BOM-claim operations")
    b_sub = b.add_subparsers(dest="bom_cmd", required=True)

    p_trust = b_sub.add_parser("trust", help="Vote: this claim looks correct")
    p_trust.add_argument("claim_iri")
    p_trust.set_defaults(func=cmd_bom_trust)

    p_flag = b_sub.add_parser("flag", help="Vote: this claim looks wrong")
    p_flag.add_argument("claim_iri")
    p_flag.set_defaults(func=cmd_bom_flag)

    p_dispute = b_sub.add_parser("dispute", help="Vote: this claim is contested")
    p_dispute.add_argument("claim_iri")
    p_dispute.set_defaults(func=cmd_bom_dispute)

    p_votes = b_sub.add_parser("votes", help="List votes on a claim")
    p_votes.add_argument("claim_iri")
    p_votes.set_defaults(func=cmd_bom_votes)

    p_show_bom = b_sub.add_parser("show", help="Reconstruct and print a claim's BOM JSON")
    p_show_bom.add_argument("claim_iri")
    p_show_bom.set_defaults(func=cmd_bom_show)

    p_diff = b_sub.add_parser("diff", help="Field-level diff between two claims")
    p_diff.add_argument("claim_a")
    p_diff.add_argument("claim_b")
    p_diff.set_defaults(func=cmd_bom_diff)

    p_claims = b_sub.add_parser("claims", help="List claims for an artifact")
    p_claims.add_argument("identifier", help="e.g. huggingface:mistralai/Mistral-7B-v0.1")
    p_claims.add_argument("--use-case", default=None)
    p_claims.add_argument("--mode", default=None)
    p_claims.set_defaults(func=cmd_bom_claims)
