# worldofBOMs — Knowledge Graph + Graph Store Backend

**Status:** Draft v3 (post-feedback round 2) for review
**Date:** 2026-05-14
**Author:** Gopi Krishnan Rajbahadur (with Claude)

## Problem

AIkaBoOM generates per-artifact BOMs (HuggingFace models, datasets, GitHub repos, arXiv papers) and writes them to disk as JSON, SPDX 3.0.1 JSON-LD, and CycloneDX 1.6. Each generation re-runs the full pipeline — metadata fetch, RAG/Direct LLM, conflict detection — even when the upstream artifact has not changed since the last run. There is no cross-session memory: two users (or the same user two days apart) pay the LLM cost twice for the same model at the same commit.

BOMs are already graph-shaped. A model BOM points to dataset BOMs (`trainedOn`), papers (`describedIn`), code repos (`hostedAt`), licenses, suppliers. The existing "recursive child BOM" feature traverses this graph implicitly. SPDX 3.0.1 is RDF. The triplet field model `{value, source, conflict}` is an attempt to represent provenance-bearing claims in JSON.

We can collapse all of this into a single persistent knowledge graph: store generated BOMs as RDF, dedupe by canonical artifact identity, accumulate user trust on what was generated, and let "world of BOMs" emerge as the union of every BOM anyone in the community has ever produced.

## Goals

1. **Persistent retrieval, with light consent.** When a BOM for `(artifact, version)` already exists, the user sees a short list of who generated it and picks `use` or `regenerate`. No trust scores or claim rankings shown in v1 UI — those run silently in the background.
2. **One node per artifact, multi-identifier.** Many BOMs may exist for the same upstream artifact — different LLM models, different prompt versions, different users. They are *alternative claims about the same node*, not duplicate nodes. An artifact can be identified by any of its known platform handles (HF, GitHub, arXiv, doi, etc.) — providing any one is enough to find the others.
3. **Graph-native.** Relationships between artifacts (model → dataset, paper → model, supplier → model) are first-class queryable edges.
4. **Provenance-preserving.** The triplet `{value, source, conflict}` round-trips losslessly into RDF and back. Every claim retains the data source that asserted it and the generation run that produced it.
5. **Trust accumulates silently in v1.** Trust scores update from implicit signals (each "use cached" choice is a quiet positive vote) and from a hidden vote API for power users / future UI. Scores are not displayed in v1; they're a bootstrap mechanism that's ready for surfacing in v2.
6. **Local-first, shareable.** Each install has a local graph. Two instances sync by exchanging a dump file. No registry server required.
7. **Pluggable backend.** Default Oxigraph; RDFLib + N-Quads fallback for environments where Oxigraph wheels are unavailable (HF Spaces, restricted CI).
8. **Documented end-to-end.** A conceptual primer and a concrete reference ship with v1; the system is not "done" until both exist.

## Non-Goals (v1)

- Public registry or federation protocol.
- SPARQL HTTP endpoint.
- Browser graph visualizer.
- Schema migration tooling beyond `rebuild`.
- Multi-tenant auth / ACLs / Sybil-resistant trust.
- Replacing the existing JSON / SPDX / CycloneDX exports — those stay.

## Backend Choice

The graph store has to fit four constraints: SPDX 3.0.1 alignment (which is RDF), HF Spaces deployability (no separate server process), federation between instances (graph union must be cheap and standards-based), and good provenance support (the triplet model is non-negotiable). Here's how the candidates compare:

| Backend | Model | Embedded? | Standards | Federation story | Verdict |
|---|---|---|---|---|---|
| **Oxigraph** | RDF + SPARQL 1.1 | Yes (Rust, Python bindings) | SPDX/PROV-O/DCAT all RDF — direct fit | `N-Quads` dump is one command; instances merge by union | ✅ **chosen** |
| RDFLib + N-Quads | RDF + SPARQL 1.1 | Yes (pure Python) | Same RDF alignment as Oxigraph | Same | ✅ **fallback** (HF Spaces) |
| Neo4j | Property graph + Cypher | No — server process (JVM) | Has neosemantics plugin for RDF, but second-class | Requires Neo4j-specific export | ❌ Server ops kills HF Spaces; SPDX mapping is a translation layer rather than the native model. |
| Kùzu | Property graph + Cypher | Yes | No native RDF; would need a side mapping | Custom dump format | ⚠️ Strong contender if we cared more about Cypher ergonomics than SPDX fit; rejected because the SPDX alignment is the strongest design pull. |
| Apache Jena Fuseki | RDF + SPARQL 1.1 | No — server process (JVM) | Excellent | Excellent | ❌ Same server-ops problem as Neo4j, without the property-graph upside. |

**Why Oxigraph, concretely:**

1. The graph store and the SPDX export are the *same artifact viewed at different scopes*. An SPDX 3.0.1 JSON-LD file is a small RDF graph. The knowledge graph is the union of all such graphs plus connecting edges. Switching to a non-RDF backend forces a permanent translation layer between the two.
2. `pip install pyoxigraph` works on every Tier-1 PyPI platform (linux x86_64/aarch64, macOS, Windows). HF Spaces is the one edge case where we may hit a wheel issue — hence the RDFLib fallback.
3. RDF-star is standardized and supported in both Oxigraph and modern RDFLib. It maps the `{value, source, conflict}` triplet model directly without inventing custom reified-statement node types.
4. SPARQL 1.1 is mature, well-documented, and lets power users run ad-hoc analytical queries without aikaboom needing to anticipate them.

**What we'd give up vs. Neo4j:** Cypher (which many devs find more ergonomic than SPARQL) and Bloom-style native visualization. Both are addressable later — Cypher via a translation library, viz via an external SPARQL-aware tool — without changing the storage layer.

**Reversibility:** the `GraphBackend` Protocol means a future switch to Kùzu or Neo4j is a backend module, not a rewrite. Identity, mapping, and the public `BomStore` API stay the same. If you want me to prototype on Kùzu as a parallel option for an embedded property-graph experience, that's a small follow-up — but I'd push back against making Neo4j the default for this project specifically because of the server-ops constraint.

## Architecture

```
                          ┌────────────────────────┐
   CLI / Web request ──►  │  cmd_generate /        │
                          │  /api/generate route   │
                          └─────────┬──────────────┘
                                    │
                                    ▼
                          ┌────────────────────────┐
                          │ BomStore.resolve(...)  │ new
                          │   - canonicalize id    │
                          │   - find candidates    │
                          │   - cache offer or     │
                          │     auto-decide        │
                          └─────────┬──────────────┘
                                    │ user picks: use / regen
                                    │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
            use cached         regenerate          keep both
                  │                 │                  │
                  ▼                 ▼                  ▼
            return BOM     ┌────────────────────────────────┐
            JSON from      │ AIBOMProcessor /               │
            graph          │ DATABOMProcessor (unchanged)   │
                           └─────────┬──────────────────────┘
                                     │ BOM JSON
                                     ▼
                           ┌────────────────────────┐
                           │ BomMapper.to_rdf(bom)  │ new
                           └─────────┬──────────────┘
                                     │ RDF quads
                                     ▼
                           ┌────────────────────────┐
                           │ GraphBackend           │ new
                           │   Oxigraph (default)   │
                           │   RDFLib   (fallback)  │
                           └────────────────────────┘
                                     ▲
                                     │
                           ┌─────────┴──────────────┐
                           │ Trust / Curation layer │ new
                           │   - vote endpoints     │
                           │   - score aggregation  │
                           │   - canonical-claim    │
                           │     pointer update     │
                           └────────────────────────┘
                                     ▲
                                     │
                           ┌─────────┴──────────────┐
                           │ aikaboom graph / bom   │
                           │ CLI subcommands        │
                           └────────────────────────┘
```

## Components

New module under `src/aikaboom/store/`:

| File | Responsibility |
|---|---|
| `naming.py` | Canonicalize artifact identifiers (case-fold, slug, alias-resolve via existing supplier index) and hash to a stable IRI. Pure functions, fully unit-tested. |
| `iris.py` | Mint IRIs for Artifact / ArtifactVersion / BOMClaim / GenerationRun / TrustVote nodes from canonicalized inputs. |
| `vocab.py` | Local namespaces (`bom:`, `aibom:`). Reuses SPDX, PROV-O, DCAT vocab where they already cover a concept. |
| `mapper.py` | `bom_to_rdf(bom_json, run_meta) -> rdflib.Dataset` and `rdf_to_bom(claim_iri) -> bom_json`. Round-trip lossless. |
| `backend.py` | `GraphBackend` Protocol — `add_quads`, `ask`, `select`, `construct`, `update`, `export`, `import_`. |
| `oxigraph_backend.py` | Default impl. On-disk store under `~/.aikaboom/graph/`. |
| `rdflib_backend.py` | Fallback. In-memory `Dataset` flushed to disk as N-Quads on every write (atomic rename). Fast enough for 10K–100K triples and avoids the SQLite-store wheel-coverage issue. |
| `store.py` | `BomStore` facade — `resolve`, `save_claim`, `record_trust_vote`, `recompute_canonical`, `find_*`, `stats`. |
| `cache_resolver.py` | Cache-resolution UX (prompt or auto-decide). |
| `trust.py` | Trust score aggregation and canonical-claim pointer maintenance. |
| `cli_graph.py` | `aikaboom graph` and `aikaboom bom` subcommands. |

## Data Model

The biggest change from v1: there are **three tiers**, not two. An artifact, its versions, and the (possibly many) BOM claims about each version are distinct nodes.

```
┌────────────────────────────────────┐
│ Artifact                           │   one node per canonical upstream identity
│ bom:artifact/<sha256-of-canon-id>  │   (e.g., mistralai/Mistral-7B-v0.1)
└──────────┬─────────────────────────┘
           │ aibom:hasVersion
           ▼
┌────────────────────────────────────┐
│ ArtifactVersion                    │   one node per upstream commit/version
│ bom:version/<artifact-hash>/<sha>  │
│   aibom:canonicalClaim ──┐         │   pointer to highest-trust claim
└──────────┬───────────────┼─────────┘
           │ aibom:hasClaim│
           ▼               ▼
┌────────────────────────────────────┐
│ BOMClaim                           │   one node per generation event
│ bom:claim/<uuid4>                  │   (model+provider+prompt+code → claim)
│   aibom:generatedBy <run>          │
│   aibom:trustScore  N              │
│   aibom:supersedes <prior-claim>?  │
│   <field claims via RDF-star>      │
└────────────────────────────────────┘
```

### Classes (RDF types)

| Class | Purpose |
|---|---|
| `aibom:Artifact` | Generic supertype for any subject of a BOM. |
| `aibom:Model` | Subclass: a HuggingFace model. |
| `aibom:Dataset` | Subclass: a dataset artifact. |
| `aibom:Paper` | Subclass: an arXiv paper. |
| `aibom:CodeRepo` | Subclass: a GitHub repository. |
| `aibom:ArtifactVersion` | A specific commit / version of an Artifact. |
| `aibom:BOMClaim` | One generation event's claim about an ArtifactVersion. Replaces the v1 `BOMSnapshot` term. |
| `aibom:GenerationRun` | The LLM provider/model/prompt/code combination that produced a claim. |
| `aibom:TrustVote` | A user's vote on a claim. |
| `aibom:Agent` | A user or generator (subclasses: `aibom:User`, `aibom:AIBomGenerator`). |
| `aibom:License`, `aibom:Supplier`, `aibom:Person`, `aibom:Source` | As in v1; reuse SPDX/PROV-O where possible. |

### Edges

| Predicate | Domain → Range |
|---|---|
| `aibom:hasVersion` | Artifact → ArtifactVersion |
| `aibom:hasClaim` | ArtifactVersion → BOMClaim |
| `aibom:canonicalClaim` | ArtifactVersion → BOMClaim |
| `aibom:generatedBy` | BOMClaim → GenerationRun |
| `aibom:supersedes` | BOMClaim → BOMClaim |
| `aibom:trustScore` | BOMClaim → xsd:decimal |
| `aibom:useCase` | BOMClaim → xsd:string (`license`, `complete`, etc.) |
| `aibom:mode` | BOMClaim → xsd:string (`rag`, `direct`) |
| `aibom:createdAt` | BOMClaim → xsd:dateTime |
| `aibom:schemaVersion` | BOMClaim → xsd:string |
| `aibom:trustVoteFor` | TrustVote → BOMClaim |
| `aibom:votedBy` | TrustVote → Agent |
| `aibom:voteKind` | TrustVote → `trusted`/`flagged`/`disputed` |
| `aibom:trainedOn` | Model → Dataset |
| `aibom:describedIn` | Model → Paper |
| `aibom:hostedAt` | Model → CodeRepo |
| `aibom:hasLicense` | Artifact → License (also `spdx:license`) |
| `aibom:suppliedBy` | Artifact → Supplier |
| `aibom:authoredBy` | Paper → Person |

### Field claims via RDF-star

A BOM field:

```json
"license": {"value": "Apache-2.0", "source": "huggingface", "conflict": null}
```

becomes:

```
bom:claim/<uuid> spdx:license <https://spdx.org/licenses/Apache-2.0> .

<< bom:claim/<uuid> spdx:license <https://spdx.org/licenses/Apache-2.0> >>
    aibom:assertedBy aibom:source/huggingface ;
    aibom:conflictKind aibom:noConflict .
```

The asserting *data source* (`huggingface` / `github` / `arxiv`) is captured per-triple via RDF-star, independent of which `GenerationRun` produced the claim node. SPARQL over quoted triples lets queries ask "which fields were sourced from GitHub" or "which fields had inter-source conflicts" naturally.

### Identity (IRIs)

| Node | IRI pattern | Example |
|---|---|---|
| Artifact | `bom:artifact/<sha256-of-canonical-id>` | `bom:artifact/a3f8…` |
| ArtifactVersion | `bom:version/<artifact-hash>/<upstream-version>` | `bom:version/a3f8…/27d67f1b` |
| BOMClaim | `bom:claim/<uuid4>` | `bom:claim/9c1d…` |
| GenerationRun | `bom:run/<hash-of-run-params>` | `bom:run/b4e2…` |
| TrustVote | `bom:vote/<uuid4>` | `bom:vote/771a…` |
| Source | `aibom:source/<huggingface\|github\|arxiv>` | `aibom:source/huggingface` |
| Agent (user) | `bom:agent/<sha256-of-user-id>` | `bom:agent/4f9c…` |

The `Artifact` IRI uses **`sha256(canonicalized primary identifier)`** rather than embedding the human-readable id, for two reasons: (1) it forces canonicalization to happen at IRI mint time, so two different inputs that canonicalize to the same identity *cannot* produce different IRIs by accident; (2) it makes the IRI URL-safe and bounded length regardless of upstream id quirks (slashes, dots, unicode).

The "primary identifier" is the highest-priority platform handle provided at creation time (priority order defined in the "Canonical Naming & Dedup" section: HF → GH → arXiv → DOI → URL). The canonicalized form is stored as `aibom:primaryIdentifier`; the full set of platform handles (including the primary, plus any others discovered later) is stored as `aibom:identifier` blank-node entries. A human-readable display string (e.g., `"Mistral 7B v0.1"`) is stored as `aibom:canonicalLabel`. Original pre-canonical input strings are kept as `aibom:alias` properties — multiple aliases per artifact are normal and queryable.

The IRI never changes once minted. If a later request reveals that an artifact has a *higher-priority* identifier than its current primary (e.g., the artifact was originally created from an arXiv id, and now an HF repo is being supplied for the same paper), the new identifier is added to the identifier set and `aibom:primaryIdentifier` updates — but the IRI hash stays. This trades a small loss of "the IRI tells you the primary platform" for the much larger gain of stable IRIs across the artifact's lifetime.

The `GenerationRun` IRI is `sha256(provider + llm_model + prompt_version + code_version + mode + use_case)`. Two generations with identical parameters share a single run node — useful for "how many distinct LLM × prompt combos have produced a claim for this version?" queries.

## Canonical Naming & Dedup

This is the system that prevents "Mistral-7B" and "mistral-7b" and "MistralAI/Mistral-7B-v0.1" from creating different artifact nodes — *and* that lets the same artifact be reached from any of its platform handles (HF, GitHub, arXiv, DOI, …) when only one is provided.

### Multi-identifier artifact model

An `Artifact` is not keyed on a single platform-prefixed string. It carries a **set of identifiers**, each typed by platform:

```
bom:artifact/<hash>
    a aibom:Model ;
    aibom:identifier [ aibom:platform "huggingface" ; aibom:value "mistralai/mistral-7b-v0.1" ] ;
    aibom:identifier [ aibom:platform "github"      ; aibom:value "mistralai/mistral-src" ] ;
    aibom:identifier [ aibom:platform "arxiv"       ; aibom:value "2310.06825" ] ;
    aibom:primaryIdentifier "huggingface:mistralai/mistral-7b-v0.1" ;
    aibom:canonicalLabel "Mistral 7B v0.1" ;
    aibom:canonRuleVersion "1" .
```

Providing *any one* of these identifiers is enough for lookup. The artifact is the union of what's known about it across platforms.

### Identifier priority (for primary key + display)

When multiple platform identifiers are available, **priority for the primary identifier** is:

1. `huggingface` (richest metadata, has commit-SHA versioning)
2. `github` (has commit-SHA versioning)
3. `arxiv` (paper-level; treated as artifact root if it's all we have)
4. `doi`
5. `url` (catch-all opaque identifier — last resort)

The first available in this order becomes `aibom:primaryIdentifier` and forms the IRI hash. The user can override with `--primary-platform <name>` if desired.

### Canonicalization pipeline (per identifier)

For each identifier value separately, apply:

1. **Lowercase** the entire value.
2. **Trim and normalize whitespace** (no leading/trailing space, no internal repeats).
3. **Strip URL noise** — for any value that came from a URL, parse and reduce to the canonical path component (`https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main` → `mistralai/mistral-7b-v0.1`).
4. **Resolve owner via the existing `default_alias_index()`** (`src/aikaboom/utils/supplier_alias.py`). Handles `mistralai ↔ Mistral AI`, `Qwen ↔ QwenLM`, etc. Reused — the project already trusts this index for supplier resolution.
5. **Normalize separators**: collapse `_` / `-` runs, strip trailing platform-suffix noise.

The canonical form of the *primary* identifier is then `sha256`'d, and that hash becomes the Artifact IRI.

Example with multiple inputs:

```
inputs:
  hf:    "MistralAI/Mistral-7B-v0.1"
  github: "https://github.com/mistralai/mistral-src"
  arxiv: "arxiv.org/abs/2310.06825v1"

canonicalized identifiers:
  huggingface:mistralai/mistral-7b-v0.1
  github:mistralai/mistral-src
  arxiv:2310.06825

primary (priority order):  huggingface:mistralai/mistral-7b-v0.1
Artifact IRI:              bom:artifact/sha256("huggingface:mistralai/mistral-7b-v0.1")
```

### Cross-identifier dedup (the redundancy step)

**Before minting a new Artifact**, `BomStore.resolve` runs a cross-identifier lookup:

```sparql
SELECT ?artifact WHERE {
  ?artifact aibom:identifier ?id .
  ?id aibom:platform ?p ; aibom:value ?v .
  VALUES (?p ?v) { ("huggingface" "mistralai/mistral-7b-v0.1")
                   ("github"      "mistralai/mistral-src")
                   ("arxiv"       "2310.06825") }
}
```

Three cases:

- **No matches** → mint a new Artifact using the highest-priority provided identifier.
- **All matches point to the same Artifact** → use it; add any new identifiers in the request as aliases on that Artifact.
- **Matches point to multiple Artifacts** (a soft collision — two previously-independent records turned out to refer to the same thing) → still mint/use the highest-priority match, but emit a warning and record `aibom:potentialDuplicateOf` edges between the candidates. The user can resolve with `aikaboom graph merge <a> <b>` when they're confident.

This means a request providing **only an arXiv ID** can find an Artifact that was originally created from an HF model, *if* a prior BOM captured the arXiv link as a field. The cross-identifier table grows as the graph grows: each generation enriches the identifier set of the artifacts it touches.

### What if no identifier is available at all?

Generation requires at least one identifier — that's a hard precondition (the metadata fetchers need something to fetch). If the user provides only an arXiv id and the arXiv fetcher succeeds, that's enough to mint an artifact. The recursive walker, on the other hand, sometimes encounters unresolvable references (`trainedOn: "some internal dataset"`); those become *placeholder artifacts* with `aibom:identifier [ aibom:platform "name-only" ; aibom:value "some internal dataset" ]`. Placeholder artifacts are flagged (`aibom:isPlaceholder true`) and never become primary key targets — they're stubs waiting for promotion when a real identifier appears.

### What this fixes

- Generating a BOM for `mistralai/Mistral-7B-v0.1` and later for `MistralAI/Mistral-7B-v0.1` produces **one Artifact node**.
- Generating with HF only, then later with arXiv only, finds the same Artifact via cross-identifier lookup.
- Two different LLMs (Claude vs GPT-4o) producing BOMs for the same artifact-version produce **one ArtifactVersion node** with **two BOMClaim children**.
- Recursive walks that encounter `squad` and `rajpurkar/squad` resolve to the same Dataset node.

## Data Flow

### Resolve step (replaces v1's silent cache hit)

Every generation request goes through `BomStore.resolve(input_ids, use_case, mode, run_meta)`:

1. **Canonicalize** each input identifier; cross-identifier lookup yields an existing Artifact or signals "new".
2. **Resolve version** (single platform API call to get the current commit SHA, as we do today).
3. **Find existing claims** for `(artifact_iri, version, use_case, mode)`:
   ```sparql
   SELECT ?claim ?createdAt ?provider ?llmModel WHERE {
     bom:version/<artifact>/<ver> aibom:hasClaim ?claim .
     ?claim aibom:useCase "<use_case>" ;
            aibom:mode "<mode>" ;
            aibom:createdAt ?createdAt ;
            aibom:generatedBy ?run .
     ?run aibom:provider ?provider ;
          aibom:llmModel ?llmModel .
   }
   ORDER BY DESC(?createdAt)
   ```
4. **Decision** (driven by `cache_resolver`):
   - **No existing claims** → fall through to generation.
   - **Existing claims, non-interactive context** (CLI `--cache=auto`, headless web POST) → return the most recent claim. No prompt.
   - **Existing claims, interactive context** (TTY or browser UI) → show the user a minimal prompt:

```
BOMs for mistralai/Mistral-7B-v0.1 @27d67f1b already exist:
  - claude-3-haiku    (2025-11-04)
  - gpt-4o-mini       (2025-12-19)

You're about to generate with claude-opus-4-7.

  [u] use the most recent existing BOM
  [r] regenerate
```

5. **Honor the choice**:
   - `use`: reconstruct BOM JSON from the most recent claim, return. **Quietly increments that claim's trust signal** (one implicit vote).
   - `regenerate`: run generator, save the result as a new claim alongside existing claims. No supersede edge, no deletion — the graph keeps everything; the user can prune later if needed.

Trust score, ranking, "canonical claim" pointer — all still computed in the background; just not shown in v1 UI. The two-option prompt covers the 95% case. Power-user features (diff between claims, explicit vote, choose-which-claim) are available via CLI but not promoted in the UI for v1.

Flag/parameter surface for non-interactive control: `--cache use|regen|prompt`, web `cache_policy` body field. `auto` aliases to `use`.

### Trust-gated recursive walks

`recursive_bom.generate_recursive_boms` gets two new params:

| Param | Effect |
|---|---|
| `--min-trust 0.7` | Only follow edges into BOMClaims with trustScore ≥ 0.7. Lower-trust BOMs trigger regeneration if `--regen-on-low-trust` is set, otherwise skip. |
| `--max-breadth 5` | Already exists logically as `recursive_safety_cap`; rename for clarity. |
| `--max-depth N` | Already exists. |
| `--regen-on-low-trust` | If a candidate child claim is below `min-trust`, generate a new claim rather than reusing it. |

This addresses the "depth/breadth limitation" with trust as a gating predicate, not just a budget cap.

### Miss / generation path

Same as v1: run the existing processor, mint a new BOMClaim, write quads transactionally, then update the `canonicalClaim` pointer if this new claim's trust score is highest for its version (initial trust = 0 for new claims, so no auto-promotion on creation).

## Trust & Curation (silent in v1 UI)

The trust system exists in v1 so that votes can accumulate from day one — but the UI does not display scores, vote counts, or "canonical claim" badges. Bootstrapping a meaningful trust signal requires data first; UI surfacing comes in v2 once enough signal exists to be useful.

### Vote model

A `TrustVote` is a small node:

```
bom:vote/<uuid>
    a aibom:TrustVote ;
    aibom:trustVoteFor bom:claim/<claim-uuid> ;
    aibom:votedBy bom:agent/<agent-hash> ;
    aibom:voteKind aibom:trusted ;     # trusted | flagged | disputed | implicit-use
    aibom:votedAt "2026-05-14T..."^^xsd:dateTime ;
    aibom:comment "license field looks wrong in HF source" .   # optional
```

### Implicit signals (the bootstrap mechanism)

Without any UI for explicit voting, v1 collects trust signal from user behavior:

- **`use cached` from the resolve prompt** → records an `aibom:voteKind aibom:implicit-use` vote on the chosen claim. This is the primary bootstrap signal.
- **Successful SPDX validation of a claim's exported BOM** → records an `aibom:implicit-validate` vote (claim made it through schema validation without error).
- **`regenerate`** → no vote. Choosing to regenerate is *not* a flag against the existing claim — the user may just want fresh data.

Implicit votes are weighted lower than explicit votes (multiplier 0.25 by default) so they bootstrap without overwhelming any future explicit signal.

### Score aggregation (v1: simple)

`aibom:trustScore` on a claim is recomputed whenever a vote is added/removed:

```
trustScore =   (w_trusted   * trusted_votes
             +  w_implicit  * implicit_votes
             -  w_flagged   * flagged_votes)
             / max(1, total_weighted_votes)
```

Default weights: explicit trusted/flagged = 1.0, implicit-use/validate = 0.25, disputed = -0.5. Range `[-1.0, +1.0]`. Initial score for a freshly generated claim is `0.0`.

This is intentionally crude. Sybil resistance, decay, reputation-weighted voting — all out of scope for v1 because they only matter at scale, and the *vote data* survives any future scoring change. The scoring function is a pure helper that can be swapped without re-collecting votes.

### Canonical claim pointer

Each ArtifactVersion has at most one `aibom:canonicalClaim` edge, pointing to the highest-trust claim for that version. Ties broken by recency. Recomputed on every vote or new claim. **Not surfaced in v1 UI**, but used internally to pick which claim to default-show on cache-hit (currently "most recent"; will become "canonical" once trust signal is meaningful).

### v1 surface (CLI-only, no Web UI exposure)

| Action | CLI | Web UI |
|---|---|---|
| Mark trusted | `aikaboom bom trust <claim-iri>` | not exposed in v1 |
| Flag | `aikaboom bom flag <claim-iri> [--comment "..."]` | not exposed in v1 |
| Dispute | `aikaboom bom dispute <claim-iri>` | not exposed in v1 |
| List votes | `aikaboom bom votes <claim-iri>` | not exposed in v1 |
| List claims | `aikaboom bom claims <artifact-iri> [@version]` | not exposed in v1 |

The Web UI in v1 shows BOMs by latest-claim-per-version and the resolve prompt list. No trust badges, no "alternatives" tab. v2 adds UI surfacing once the trust data has had time to accumulate.

For multi-user deployments, the agent identity is set via `AIKABOOM_AGENT_ID`. Single-user local install derives a default agent IRI from `getpass.getuser() + machine_id`.

### Trust persistence across instances

Votes are quads like everything else and are included in `aikaboom graph export`. Two instances merging their dumps accumulate each other's votes — the canonical pointers recompute on import. Conflicting votes from the same agent (same `bom:agent/...` IRI but different vote kinds for the same claim) are resolved by latest `votedAt`.

## Integration Points

| Existing code | Change |
|---|---|
| `cli.cmd_generate` (`src/aikaboom/cli.py:164`) | Wrap with `BomStore.resolve`; add `--cache use|regen|prompt` and `--min-trust` / `--regen-on-low-trust` flags. |
| Web `/api/generate` (`src/aikaboom/web/app.py`) | Same wrap; new `cache_policy` body field (default `prompt` for browser UI, `auto` for headless POSTs). Existing `force_refresh` becomes a shorthand for `cache_policy=regen`. |
| `recursive_bom.generate_recursive_boms` (`src/aikaboom/utils/recursive_bom.py`) | Add `min_trust`, `regen_on_low_trust`, `cache_policy` kwargs. |
| `utils/supplier_alias.py` | Unchanged. Reused by `store/naming.py` for canonicalization. |
| `utils/spdx_validator.py`, `schemas/` | Unchanged. SPDX JSON-LD output is still produced from the in-memory BOM JSON. |
| CycloneDX export | Unchanged. |
| Web UI BOM view | New "Trust" panel and "Alternatives" tab. |

The store is a **new optional layer**. With `AIKABOOM_GRAPH_DISABLE=1`, the system behaves exactly as today.

## CLI Surface

```
# Graph operations
aikaboom graph stats              # counts of artifacts/versions/claims/votes, disk size
aikaboom graph list               # list artifacts (with canonical-claim summary)
aikaboom graph show <iri>         # pretty-print one node and its neighborhood
aikaboom graph export <file>      # dump as JSON-LD or N-Quads
aikaboom graph import <file>      # merge a dump (recomputes canonical claims)
aikaboom graph query <sparql>     # run SPARQL
aikaboom graph rebuild            # rebuild from results/*.json + replay votes

# BOM-claim operations
aikaboom bom claims <id> [@version]    # list claims for an artifact[/version]
aikaboom bom show <claim-iri>          # show a specific claim
aikaboom bom trust <claim-iri>         # vote trusted
aikaboom bom flag <claim-iri> [--comment "..."]
aikaboom bom dispute <claim-iri>
aikaboom bom votes <claim-iri>         # list votes on a claim
aikaboom bom diff <claim-a> <claim-b>  # field-level diff between two claims
```

`aikaboom generate` gains:
- `--cache use|regen|prompt` (default `prompt` in TTY, `use` non-TTY).
- `--min-trust 0.0` (recursive walks).
- `--regen-on-low-trust` (recursive walks).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `AIKABOOM_GRAPH_DIR` | `~/.aikaboom/graph` | On-disk store location. |
| `AIKABOOM_GRAPH_BACKEND` | `auto` | `oxigraph` / `rdflib` / `auto`. |
| `AIKABOOM_GRAPH_TTL_DAYS` | `30` | Claim freshness window. After TTL, prompt defaults shift toward "regenerate". |
| `AIKABOOM_GRAPH_DISABLE` | `0` | If `1`, store is bypassed entirely. |
| `AIKABOOM_AGENT_ID` | `<user>@<machine>` | Identity used for trust votes. |
| `AIKABOOM_CACHE_POLICY_DEFAULT` | `prompt` (TTY), `use` (non-TTY) | Default cache policy when not specified. |

## Error Handling

- **Backend init failure** (Oxigraph wheel missing): fall back to RDFLib, log once at startup. Generation never blocks on store availability.
- **Mapping failure on save**: log + still write the BOM JSON to disk as today. The graph is best-effort.
- **Mapping failure on load**: treat as cache miss; never return a partially-reconstructed BOM. Log the failing claim IRI.
- **Backend disk corruption**: `aikaboom graph rebuild` regenerates from the JSON files in `results/` and replays votes from a side journal (votes are also appended to `~/.aikaboom/graph/votes.log` for this reason).
- **Canonicalization disagreement between versions of the alias index**: the old artifact node is kept; new aliases are added; an `aibom:supersededByCanonical` edge points from old → new artifact only if the rule explicitly migrates an identity. Default is no automatic migration — alias-index updates don't silently merge nodes.
- **Vote on a non-existent claim** (e.g., after dump import that missed claims): vote is held in a quarantine bucket and applied if/when the claim appears.

## Testing

| Test | Asserts |
|---|---|
| `test_naming.py` | Canonicalization is idempotent, alias-aware, and stable across runs. Hash determinism. |
| `test_iris.py` | All IRI minting is deterministic and URL-safe. |
| `test_mapper_roundtrip.py` | `rdf_to_bom(bom_to_rdf(b)) == b` for every BOM in `Golden_Set/` and `results/`. Property test with hypothesis for synthetic BOMs. |
| `test_resolve_prompt.py` | Resolve step offers correct choices given various cache states (none/one-claim/multi-claim/stale). |
| `test_dedup.py` | Generating with name variants (`Mistral-7B-v0.1`, `MistralAI/Mistral-7B-v0.1`) produces one Artifact node, multiple aliases. |
| `test_three_tier.py` | Generating same version with two different LLMs produces one ArtifactVersion, two BOMClaims, with `canonicalClaim` correctly resolved by trust. |
| `test_trust.py` | Voting updates trust scores, recomputes canonical claim, persists across export/import. |
| `test_recursive_trust.py` | `--min-trust` filters child reuse; `--regen-on-low-trust` triggers fresh generation. |
| `test_conflict_preservation.py` | RDF-star round-trips `inter` and `intra` conflicts. |
| `test_export_import.py` | Export → fresh store → import → graph identity including votes. |
| `test_backend_fallback.py` | Force Oxigraph unavailable, RDFLib backend takes over and tests still pass. |
| `test_cache_policies.py` | Each `--cache` value behaves correctly (use/regen/prompt). `use` records an implicit-use vote on the chosen claim. |
| `test_multi_identifier_dedup.py` | Generating with HF only, then with arXiv only that points to the same paper, finds the same Artifact via cross-identifier lookup; both identifiers stored as aliases. |
| `test_placeholder_artifact.py` | Recursive walker encountering an unresolvable reference creates a placeholder Artifact that's flagged and excluded from primary-key matching. |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Oxigraph wheels missing on some platforms (HF Spaces ARM, restricted CI) | RDFLib fallback. |
| RDF-star support uneven across backends | Both Oxigraph and modern RDFLib support it. Pin minimum versions. |
| Schema evolution breaks old claims | Claims tagged with `aibom:schemaVersion`; mapper reads old versions; new versions only on write. `aikaboom graph rebuild` is the escape hatch. |
| Canonicalization rule changes split or merge artifact nodes silently | Rule changes are versioned (`aibom:canonRuleVersion` on Artifact); migration is explicit (`aikaboom graph migrate-canon`). No silent merges. |
| Trust scoring is gameable in a multi-user setting | Out of scope for v1. The vote *data* is preserved (agent IRI, timestamp), so a smarter scorer can replace the simple aggregator without re-collecting votes. |
| Prompt fatigue from the resolve step | `--cache auto` for power users; the prompt only appears in interactive TTY / browser sessions; sensible defaults baked in (`use` if canonical claim trust ≥ 0.8 *and* same generation params, `prompt` otherwise). |
| Cache returns stale BOM after prompt/model change | `GenerationRun` IRI includes `prompt_version` and `code_version`; a bumped prompt produces a different run, which produces a different claim — visible in the prompt as "your run differs from the canonical claim's run." |

## Open Questions

Noted but non-blocking:

- **Auto-use threshold**: at what trust score does the resolve step quietly auto-use the canonical claim without prompting? Proposed default: 0.8. Configurable.
- **TTL semantics**: should TTL gate the prompt's defaults only, or also evict claims? Proposed: never evict; older claims demote toward "regenerate" default in the prompt.
- **Anonymization for export**: `aikaboom graph export --redact` strips agent IRIs to opaque hashes for public sharing.
- **Pre-resolve near-duplicate hint**: how aggressive should the "did you mean…?" surface be in the web UI? Lev distance ≤ 2 by default.

## Documentation Plan

Documentation is a first-class deliverable for v1, not an afterthought. The system has to be understandable both at the *concept* level (a new contributor asking "what is this thing and why") and the *concrete* level (a user asking "how do I write a SPARQL query that finds models trained on CC-BY datasets"). Both layers ship together.

### Conceptual layer

| Doc | Audience | Length | Content |
|---|---|---|---|
| `docs/worldofboms/CONCEPT.md` | New contributors, casual users, blog-post readers | ~1500 words | Mental model: what the knowledge graph is, why RDF, what an Artifact / ArtifactVersion / BOMClaim is, why three tiers, what trust does, how worldofBOMs grows over time. Single readable arc, no API noise. Diagrams: the three-tier model, the resolve flow, the federation flow. Lifted to the README's "How it works" pointer list. |
| `docs/worldofboms/RATIONALE.md` | Reviewers, future maintainers, anyone questioning a design call | ~1000 words | The "why these choices" doc. Backend comparison table from this spec, RDF-star vs reified statements, multi-identifier dedup vs platform-prefix, trust simplicity vs sophistication. Each major design call gets one paragraph. |

### Concrete layer

| Doc | Audience | Length | Content |
|---|---|---|---|
| `docs/worldofboms/PIPELINE.md` | Devs tracing the code | ~2000 words | End-to-end walkthrough: a user invocation flows through canonicalization → cross-identifier lookup → resolve prompt → generation → mapper → graph save → SPDX export. Mirrors the structure of the existing `docs/PIPELINE_WALKTHROUGH.md` with file:line refs at every step. |
| `docs/worldofboms/SCHEMA.md` | Devs writing SPARQL, integrators | ~3000 words | Full vocabulary reference. One section per class, one row per predicate, with example triples and example SPARQL snippet. Mirrors the style of the existing `docs/SPDX_3.0.1_FIELD_REFERENCE.md`. Canonical source for any other doc that references the schema. |
| `docs/worldofboms/CLI.md` | End users | ~1500 words | Every `aikaboom graph` and `aikaboom bom` subcommand with example invocations and example outputs. Tied to the help text so `aikaboom graph --help` and this doc stay in sync (test asserts both render the same option list). |
| `docs/worldofboms/API.md` | Python embedders | ~1500 words | `BomStore`, `GraphBackend`, `BomMapper` reference. Type signatures + minimum-viable example for each public function. |
| `docs/worldofboms/QUERIES.md` | Power users, analytics | ~1500 words | SPARQL cookbook. ~15 recipes: "find all models with Apache-2.0 license trained on a CC-BY dataset", "find claims with inter-source conflicts", "find unsourced field values", "find artifacts with the most BOMClaims", "find datasets that appear in the most models", etc. Each recipe: prose intent, query, sample output. |
| `docs/worldofboms/FEDERATION.md` | Multi-instance operators | ~1000 words | Export/import workflow, what gets merged, how vote conflicts resolve, how to handle canonical drift across instances, recovery from a bad import. |
| `docs/worldofboms/TROUBLESHOOTING.md` | All users | ~800 words | Common error states and recovery: backend init failure, mapping failure, disk corruption, canonicalization disagreements, duplicate-artifact warnings. |

### Inline documentation

- **Module docstrings**: every file in `src/aikaboom/store/` starts with a 5–10 line docstring naming responsibility, dependencies, and the doc that covers it.
- **Public function docstrings**: every public function in `BomStore`, `GraphBackend`, `BomMapper` has a docstring with a one-line summary, args/returns, and one minimum-viable example.
- **SPARQL constants**: queries defined as named constants with comments explaining intent, not inlined into call sites.

### README integration

The top-level `README.md` "How it works" section gets a new pointer: *"Want the worldofBOMs knowledge graph story? Start with `docs/worldofboms/CONCEPT.md` for the mental model, then `docs/worldofboms/PIPELINE.md` for the code-level walkthrough."* No content duplication — `README` stays the elevator pitch.

### Testing the docs

- **Link check**: a CI step verifies every internal markdown link resolves and every cited `file:line` ref exists.
- **CLI parity test**: parse `docs/worldofboms/CLI.md` and assert each documented command + flag set matches the actual argparse output.
- **Schema parity test**: parse `docs/worldofboms/SCHEMA.md` and assert each documented predicate appears in `vocab.py`.
- **Query test**: each SPARQL recipe in `QUERIES.md` runs against a fixture graph in CI and produces non-empty output.

### Authoring order during implementation

Docs are written **interleaved** with code, not at the end:

1. `CONCEPT.md` and `RATIONALE.md` are written *first*, before any module is implemented, locking the mental model. They serve as the implementer's brief.
2. `SCHEMA.md` is written *alongside* `vocab.py` — the two evolve together.
3. `API.md` is written *alongside* `store.py` and `backend.py`.
4. `CLI.md` is written *alongside* `cli_graph.py`.
5. `PIPELINE.md`, `QUERIES.md`, `FEDERATION.md`, `TROUBLESHOOTING.md` are written *after* the end-to-end flow works, so they reflect what actually happens, not what was planned.

The implementation plan (writing-plans skill) will sequence the doc deliverables alongside the code phases.

## Out of Scope (v1)

- SPARQL HTTP endpoint (users run SPARQL via `aikaboom graph query`).
- Public registry / federation protocol beyond dump exchange.
- Browser graph visualizer.
- Schema migration tooling beyond `rebuild` and `migrate-canon`.
- Multi-user auth / ACLs.
- Sybil-resistant trust (reputation-weighted voting, vote decay).
- Replacing JSON / SPDX / CycloneDX exports.

## Success Criteria

1. Generating the same BOM twice (same artifact, same version, same use case, within TTL) shows the user a minimal two-option prompt and makes zero LLM calls when the user picks `use`. The chosen claim's implicit-use trust signal increments silently.
2. Name variants of the same artifact (`Mistral-7B-v0.1`, `MistralAI/Mistral-7B-v0.1`) collapse to one Artifact node, with both originals retained as aliases.
3. Generating with HF only, then later with arXiv only that refers to the same artifact, finds the same Artifact via cross-identifier lookup. Both identifiers stored.
4. Two BOMs for the same artifact-version generated with different LLMs produce one ArtifactVersion with two BOMClaims. Canonical pointer is maintained internally (not shown in UI).
5. `aikaboom bom trust <iri>` records a vote that survives export → import.
6. Recursive walks honor `--min-trust` and `--max-depth` / `--max-breadth`.
7. Round-trip JSON ↔ RDF is lossless for every BOM in `Golden_Set/`.
8. RDFLib fallback is exercised in CI and produces identical results to Oxigraph.
9. No regression in existing CLI/web tests with `AIKABOOM_GRAPH_DISABLE=1`.
10. All seven planned documentation files exist, internal links resolve in CI, CLI doc matches argparse output, and the SPARQL cookbook recipes run green against a fixture graph.
