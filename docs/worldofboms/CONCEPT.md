# worldofBOMs — Concept

AIkaBoOM generates per-artifact Bills of Materials. The worldofBOMs knowledge
graph is what happens when you stop throwing each generated BOM into a JSON
file and forgetting about it: instead you remember every BOM you've ever
produced, dedupe across name variants and platform handles, and let the
collected knowledge prevent recomputing what's already known.

## Three tiers

Every BOM in the graph lives at one of three levels of identity:

1. **Artifact** — *the thing itself*. `mistralai/Mistral-7B-v0.1` is one
   artifact, whether you find it on HuggingFace, on GitHub, or in an arXiv
   paper. Different platform handles for the same upstream object collapse
   to one Artifact node.

2. **ArtifactVersion** — *a specific snapshot*. Commit `27d67f1b` of the
   Mistral-7B-v0.1 repo is one version. Each Artifact has many versions
   over time.

3. **BOMClaim** — *what a specific generation run said about a version*.
   Run aikaboom today with Claude, run it again next month with GPT-4o,
   and you get two BOMClaims about the same ArtifactVersion. They're not
   duplicates — they're alternative claims, each carrying provenance for
   the LLM model, prompt version, code version that produced it.

This separation is the whole point. It lets the graph answer
"have we seen this artifact before" independently of "do we trust the BOM
we have for it" and "do we have a fresh enough BOM for this version".

## Why RDF

Because SPDX 3.0.1 is RDF. The graph store and the SPDX export are the same
artifact at different scales: an SPDX JSON-LD file is a small RDF graph; the
knowledge graph is the union of all of them plus the edges between them.
Switching to a non-RDF backend would force a translation layer between two
things that are already the same. We chose Oxigraph because it's embedded
(no server to run, works on HF Spaces) and supports RDF-star, which is what
lets us attach the `{value, source, conflict}` triplet model to each field
without inventing new node types.

## Multi-identifier artifacts

An Artifact carries a *set* of platform identifiers, not just one. When you
generate a BOM with only an arXiv id today, and tomorrow with only a HF repo
that turns out to refer to the same paper, the graph finds the connection.
The first identifier you provide becomes the primary (used for the stable IRI
hash); the rest accumulate as aliases. Cross-identifier dedup runs on every
request — providing any one platform handle is enough to find what's there.

When the recursive walker hits an unresolvable reference like "trainedOn:
some internal dataset", it creates a *placeholder artifact* that's flagged
and excluded from primary-key matching until a real identifier appears.

## Trust (silent in v1)

Each BOMClaim carries a `trustScore` that's recomputed whenever a vote
arrives. Three vote sources:

- **Explicit** — `aikaboom bom trust <claim-iri>` records a positive vote.
  CLI only in v1; the web UI doesn't expose this yet.
- **Implicit-use** — every time you pick "use cached" from the resolve
  prompt, that's a quiet positive vote on the chosen claim. This is how
  the system bootstraps without any UI for explicit feedback.
- **Implicit-validate** — when a claim's exported BOM passes SPDX
  validation, that records another quiet positive vote.

Explicit votes weigh 1.0; implicit votes weigh 0.25. The aggregate score
is `(weighted_positives - weighted_negatives) / weighted_total` in range
[-1, +1]. None of this is shown in the v1 UI — that's deliberate. We need
data before surfaces.

## How the graph grows

Each generation enriches the graph: a new BOMClaim under an existing
ArtifactVersion, a new ArtifactVersion under an existing Artifact, or a
brand-new Artifact subgraph. Recursive walks compound this: each child
BOM (dataset, paper) is itself a candidate cache hit for any future model
that references the same thing.

Federation across instances is local-first: `aikaboom graph export | scp |
aikaboom graph import` merges two laptops' knowledge into one. Vote
attribution survives the round-trip, so trust accumulates across instances
without requiring a registry server.

## The resolve prompt

When you ask aikaboom to generate a BOM for something the graph already
has, you see:

    BOMs for mistralai/Mistral-7B-v0.1 @27d67f1b already exist:
      - claude-3-haiku    (2025-11-04)
      - gpt-4o-mini       (2025-12-19)

    You're about to generate with claude-opus-4-7.

      [u] use the most recent existing BOM
      [r] regenerate

Two options, no trust scores, no claim rankings. Picking `use` records an
implicit-use vote on the chosen claim. Picking `regenerate` runs the LLM
pipeline and adds a new BOMClaim alongside the existing ones — nothing is
deleted. In non-interactive contexts (CI, headless web POSTs) the default
is `use the most recent`, suppressing the prompt entirely.

## What the system is and isn't

The worldofBOMs graph **is**: a persistent, dedupe-aware, provenance-bearing
store of every BOM you've ever generated, designed to be exchanged with
other instances by file.

The worldofBOMs graph **is not** (in v1): a registry server, a SPARQL HTTP
endpoint, a graph visualizer, a multi-user identity system with auth, or
a Sybil-resistant reputation network. All of these are addressable later
without changing the storage layer.

Start with `docs/worldofboms/PIPELINE.md` once it exists for the
code-level walkthrough, or `docs/worldofboms/SCHEMA.md` for the full
vocabulary reference.
