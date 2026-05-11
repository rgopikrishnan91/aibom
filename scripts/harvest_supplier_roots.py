#!/usr/bin/env python3
"""Harvest HuggingFace org handles and cluster them to canonical roots.

Walks ``HfApi.list_models()`` + ``HfApi.list_datasets()`` once, takes the
org-prefix segment of every repo ID, clusters the resulting set within
HF using ``normalize_org`` (Jaro-Winkler optional), and writes a JSON
file shaped::

    {
      "<canonical_root>": ["<variant>", "<variant>", ...],
      ...
    }

The output drops into ``src/aikaboom/config/supplier_alias_harvest.json``
where ``SupplierAliasIndex`` picks it up on next start. No model-card
fetches, no GitHub-side anything — that bridging happens at BOM-resolution
time via substring matching against these roots (see
``SupplierAliasIndex.is_same_supplier``).

Default ``--limit 10000`` per stream is enough to cover effectively
every active HF org. Set ``--limit 0`` to iterate the entire Hub
(slow, hundreds of thousands of repos).

Example
-------

    python scripts/harvest_supplier_roots.py \\
        --out src/aikaboom/config/supplier_alias_harvest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from aikaboom.utils.supplier_alias import normalize_org  # noqa: E402


def _hf_api():
    from huggingface_hub import HfApi  # imported lazily
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    return HfApi(token=token) if token else HfApi()


def _extract_orgs(stream: Iterable, limit: int) -> Set[str]:
    """Take the org-prefix of each repo ID; bail at `limit` rows. ``limit=0`` = unlimited."""
    orgs: Set[str] = set()
    seen = 0
    for entry in stream:
        seen += 1
        if limit and seen > limit:
            break
        repo_id = getattr(entry, "id", None) or getattr(entry, "modelId", None)
        if not repo_id or "/" not in repo_id:
            continue
        org = repo_id.split("/", 1)[0].strip()
        if org:
            orgs.add(org)
        if seen % 5000 == 0:
            print(f"  ... scanned {seen} entries, {len(orgs)} unique orgs", file=sys.stderr)
    return orgs


def _cluster(orgs: Iterable[str], jw_threshold: float) -> dict:
    """Cluster org handles by normalised form (Tier 1) then Jaro-Winkler (Tier 2).

    Returns ``{canonical_root: sorted_variants_list}`` where ``canonical_root``
    is the cluster's normalised form (lowercased, separator-stripped, suffix-stripped).
    """
    # Tier 1: bucket by normalize_org. Each non-empty bucket is a candidate cluster.
    by_norm: dict[str, Set[str]] = defaultdict(set)
    by_norm_no_suffix: dict[str, str] = {}
    for org in orgs:
        norm = normalize_org(org)
        if not norm:
            # All-suffix handles (e.g. "ai") would normalise to "" — keep them
            # under their lowercased original so they don't get merged with junk.
            norm = org.lower()
        by_norm[norm].add(org)
        by_norm_no_suffix.setdefault(norm, norm)

    # Substring-based clustering on the bucket keys (no Jaro-Winkler — its
    # transitive chains over-merged in practice). ``jw_threshold`` retained
    # in the signature for backwards compat but unused.
    _ = jw_threshold
    canonicals = list(by_norm.keys())
    parents = {c: c for c in canonicals}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Prefer the shorter canonical as the surviving root
        if len(rb) < len(ra):
            ra, rb = rb, ra
        parents[rb] = ra

    # Prefix-substring merge only — arbitrary `a in b` substring caused
    # transitive chains (e.g. "mental" ⊂ "experimental" pulled the mistral
    # cluster into the mental cluster). Requiring the stem to appear at
    # the START of the longer normalised form preserves the company-prefix
    # semantic ("google" prefix of "googleresearch") without falling for
    # mid-/suffix-embedded coincidences.
    _MIN_STEM = 6
    sorted_keys = sorted(canonicals, key=len)
    for i, a in enumerate(sorted_keys):
        if len(a) < _MIN_STEM:
            continue
        for b in sorted_keys[i + 1:]:
            if b.startswith(a):
                union(a, b)

    clusters: dict[str, Set[str]] = defaultdict(set)
    for norm, members in by_norm.items():
        root = find(norm)
        clusters[root].update(members)
    return {root: sorted(members) for root, members in clusters.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "src" / "aikaboom" / "config" / "supplier_alias_harvest.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max repos to scan per stream (models + datasets). 0 = unlimited.",
    )
    parser.add_argument(
        "--jw-threshold",
        type=float,
        default=0.92,
        help="Jaro-Winkler threshold for tier-2 cluster merge (set 0 to disable).",
    )
    parser.add_argument("--no-datasets", action="store_true", help="Skip the datasets stream.")
    args = parser.parse_args()

    api = _hf_api()

    print("Listing HF models…", file=sys.stderr)
    model_orgs = _extract_orgs(api.list_models(sort="downloads"), args.limit)
    print(f"  -> {len(model_orgs)} unique model orgs", file=sys.stderr)

    dataset_orgs: Set[str] = set()
    if not args.no_datasets:
        print("Listing HF datasets…", file=sys.stderr)
        dataset_orgs = _extract_orgs(api.list_datasets(sort="downloads"), args.limit)
        print(f"  -> {len(dataset_orgs)} unique dataset orgs", file=sys.stderr)

    all_orgs = model_orgs | dataset_orgs
    print(f"Total unique orgs: {len(all_orgs)}", file=sys.stderr)

    print("Clustering…", file=sys.stderr)
    clusters = _cluster(all_orgs, jw_threshold=args.jw_threshold)
    # Only emit clusters with ≥1 variant (i.e., everything); but drop the
    # ones whose root is empty/whitespace from the rare degenerate case.
    clusters = {k: v for k, v in clusters.items() if k.strip()}
    print(f"  -> {len(clusters)} canonical roots", file=sys.stderr)

    # Optional: stash some summary stats so the user can sanity-check
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"  -> {len(multi)} roots cover >1 variant", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clusters, indent=2, sort_keys=True))
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
