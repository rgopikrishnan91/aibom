# worldofBOMs — Design Rationale

One paragraph per major design decision, explaining why we picked the
choice we did over the alternatives we considered.

## Why a graph store at all

BOMs are already graph-shaped — a model points to datasets it was trained
on, papers that describe it, code repos that host it, licenses, suppliers.
Storing them as flat JSON files throws away the edges. Persisting them as
a graph lets us dedupe across name variants, find an artifact via any of
its platform handles, and avoid recomputing what we already know. The
existing "recursive child BOM" feature already traverses this graph
implicitly; we're just materializing what's there.

## Why Oxigraph specifically

We compared Oxigraph, RDFLib, Neo4j, Kùzu, and Apache Jena Fuseki against
four constraints: SPDX 3.0.1 alignment (which is RDF), HF Spaces
deployability (no separate server), federation between instances (cheap
graph union), and provenance support (the `{value, source, conflict}`
triplet model). Oxigraph hits all four: embedded Rust core with Python
bindings, native RDF + SPARQL 1.1, supports RDF-star for per-triple
provenance, and `N-Quads` dump is one command. Neo4j requires a server
(killing HF Spaces); Kùzu is embedded but uses a property-graph model that
would need a translation layer to/from SPDX; Fuseki is RDF but server-based.
RDFLib is the fallback for the one platform where Oxigraph wheels don't
land.

## Why RDF-star instead of reified statements

The triplet field model `{value, source, conflict}` needs to attach metadata
(source, conflict kind) to individual statements. Classic RDF reification
would invent a `FieldClaim` node type per field, with `subject`/`predicate`/
`object` properties — verbose, hard to query naturally. RDF-star lets us
quote the original triple and annotate it directly. SPARQL over quoted
triples lets queries like "find all fields sourced from GitHub" stay
one-liners.

## Why three tiers (Artifact / ArtifactVersion / BOMClaim)

A two-tier model collapses "the thing" and "this generation's claims about
the thing" into one node, making it impossible to distinguish "we know
about this artifact" from "we have a fresh BOM for this version of it".
Three tiers separate identity, snapshot, and claim cleanly. The cost is
one extra layer of nodes; the benefit is being able to maintain many
alternative claims per version (different LLMs, different prompt versions)
without losing any.

## Why multi-identifier artifacts

The spec started with `bom:<platform>/<owner>/<name>@<version>` IRIs that
forced a single primary platform. But the project handles HF, GitHub, and
arXiv inputs — sometimes all three for one artifact, sometimes just one.
Forcing a primary platform meant the same artifact could end up under two
different IRIs depending on which input you provided first. The
multi-identifier model fixes this: an Artifact holds a set of platform
handles, primary chosen by priority order, IRI hashed from the primary.
Cross-identifier dedup runs on every request. The trade-off is a small
loss of "the IRI tells you the platform" for the much bigger gain of
stable identity across the artifact's lifetime.

## Why trust is silent in v1

Surfacing trust scores in the UI before trust data exists trains users on
a meaningless signal. The system needs a bootstrap period where votes
accumulate (primarily from implicit-use signals when users pick "use
cached" from the resolve prompt) before any score is informative. v1
builds the vote model and the aggregator; v2 will add UI surfaces once
real data is in.

## Why two options on the resolve prompt

Earlier iterations of the design had four options (use / regen-replace /
regen-keep-both / show-diff) and visible trust stars. User feedback
collapsed this to two options and no trust display. The simpler prompt
covers the 95% case; power-user features (diff between claims, choose a
specific older claim) are available via CLI but not promoted in the UI.

## Why implicit votes are weighted 0.25× explicit

Implicit-use votes are cheap to produce — every cache hit creates one.
Without weighting, a single popular artifact would accumulate a flood of
implicit votes that would drown out future explicit feedback. 0.25 is a
defensible starting point that lets implicit signal contribute without
overwhelming. The weight is configurable in `trust.py` and can be tuned
once we have real data.

## Why we don't auto-merge cross-identifier collisions

When cross-identifier lookup returns matches to multiple Artifacts — i.e.,
two previously-independent records turn out to refer to the same upstream
thing — we record `aibom:potentialDuplicateOf` edges instead of auto-
merging. Auto-merging is destructive; if our match is wrong, recovery is
painful. Manual `aikaboom graph merge <a> <b>` keeps the human in the loop
for the cases where ambiguity matters.

## Why the RDFLib fallback flushes to N-Quads on every write instead of using a SQLite store

`rdflib-sqlalchemy` and `rdflib-berkeleydb` both exist but have uneven wheel
coverage and add a dependency that breaks on the same platforms we're
falling back to in the first place. In-memory + atomic N-Quads flush is
fast enough for 10K–100K triples (the realistic v1 scale) and has zero
extra dependencies. If the graph grows past that, switching the fallback
to a real persistent store is a backend module swap, not a redesign.
