# worldofBOMs — Federation (Local-First Sharing)

There is no registry server. Two instances share knowledge by exchanging
dump files.

## Export
```
$ aikaboom graph export ~/bom-graph.nq
Exported to /home/gopi/bom-graph.nq (nquads).
```

`--format jsonld` produces JSON-LD instead — useful for ingestion by other
RDF tools.

## Transfer
Move the file via any means: `scp`, `rsync`, a USB drive, a shared
filesystem.

## Import on the receiving instance
```
$ aikaboom graph import ~/bom-graph.nq
Imported /home/gopi/bom-graph.nq.
```

Import is a *graph union*. Artifacts, versions, claims, and votes from the
incoming dump are merged with the local store. Trust score and canonical
pointers recompute automatically.

## Vote conflict resolution
If the same agent IRI has voted differently on the same claim in two
instances, the latest `votedAt` wins. Different agents' votes always
accumulate.

## Canonical drift
If two instances independently created different IRIs for what turns out
to be the same artifact (e.g., they had different versions of the
canonicalization rules), the import does not auto-merge. Instead, the
`potentialDuplicateOf` edges flag the collision. Use `aikaboom graph merge
<a> <b>` after review.

## Anonymization (proposed; see Open Questions in the spec)
`aikaboom graph export --redact` strips agent IRIs to opaque hashes for
public sharing. Not implemented in v1.

## Recovery from a bad import
The Oxigraph backend supports transactional rollback; the RDFLib backend
flushes to N-Quads after every operation but does not transaction-log.
If an import leaves the graph in a bad state, restore the previous
`~/.aikaboom/graph/store.nq` from a snapshot, or run `aikaboom graph
rebuild` to reconstruct from `results/*.json` and replay votes.
