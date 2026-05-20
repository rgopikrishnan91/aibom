# aikaboom graph / bom CLI Reference

## aikaboom graph

### stats
Print counts of artifacts, versions, claims, votes.

```
$ aikaboom graph stats
{
  "artifacts": 3,
  "versions": 5,
  "claims": 7,
  "votes": 12
}
```

### list
List all artifacts with their primary identifier and canonical label.

```
$ aikaboom graph list
[
  {"artifact": "bom:artifact/a3f8...", "label": "mistralai/Mistral-7B-v0.1",
   "primary": "huggingface:mistralai/mistral-7b-v0.1"}
]
```

### show IRI
Print every triple with the given IRI as subject.

```
$ aikaboom graph show bom:claim/9c1d2a8f
[
  {"p": "https://aikaboom.dev/aibom#useCase", "o": "license"},
  {"p": "https://aikaboom.dev/aibom#mode",    "o": "rag"}
]
```

### export FILE [--format nquads|jsonld]
Dump the entire graph.

```
$ aikaboom graph export ~/bom-graph.nq
Exported to /home/gopi/bom-graph.nq (nquads).
```

### import FILE [--format nquads|jsonld]
Merge a dump into the local graph. Vote attribution and canonical pointers
are recomputed automatically.

### query SPARQL
Run an arbitrary SPARQL query.

```
$ aikaboom graph query 'SELECT ?a WHERE { ?a a <https://aikaboom.dev/aibom#Model> }'
{"a": "bom:artifact/a3f8..."}
```

### merge ARTIFACT_A ARTIFACT_B
Merge `artifact_b` into `artifact_a`. All `hasVersion` and `identifier` edges
from `b` are added to `a`, then `b` is deleted. Use this to resolve
`potentialDuplicateOf` collisions surfaced by cross-identifier dedup.

### rebuild
Rebuild the graph from `results/*.json`. Used to recover from a corrupted
store or to seed the graph with previously-generated BOMs.

```
$ aikaboom graph rebuild
Rebuilt graph from 12 BOM files.
```

## aikaboom bom

### trust CLAIM_IRI
Record a trusted vote. Recomputes the canonical-claim pointer.

```
$ aikaboom bom trust bom:claim/9c1d2a8f...
Recorded TRUSTED vote on bom:claim/9c1d2a8f...
```

### flag CLAIM_IRI
Record a flagged vote.

### dispute CLAIM_IRI
Record a disputed vote (weighted -0.5).

### votes CLAIM_IRI
List every vote recorded on a claim, newest first.

### show CLAIM_IRI
Reconstruct and pretty-print the BOM JSON for a claim.

### diff CLAIM_A CLAIM_B
Field-level diff between two claims (only fields whose values differ).

```
$ aikaboom bom diff bom:claim/aaa bom:claim/bbb
[
  {"field": "license", "a": "Apache-2.0", "b": "MIT"}
]
```

### claims IDENTIFIER [--use-case X] [--mode Y]
List claims for an artifact, newest first.

```
$ aikaboom bom claims huggingface:mistralai/Mistral-7B-v0.1
[
  {
    "iri": "bom:claim/9c1d2a8f...",
    "created_at": "2026-05-14T10:00:00+00:00",
    "llm_model": "anthropic/claude-3-haiku"
  }
]
```
