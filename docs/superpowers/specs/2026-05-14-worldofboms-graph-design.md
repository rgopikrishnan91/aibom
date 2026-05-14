# worldofBOMs — Knowledge Graph + Graph Store Backend

**Status:** Draft v2 (post-feedback) for review
**Date:** 2026-05-14
**Author:** Gopi Krishnan Rajbahadur (with Claude)

## Problem

AIkaBoOM generates per-artifact BOMs (HuggingFace models, datasets, GitHub repos, arXiv papers) and writes them to disk as JSON, SPDX 3.0.1 JSON-LD, and CycloneDX 1.6. Each generation re-runs the full pipeline — metadata fetch, RAG/Direct LLM, conflict detection — even when the upstream artifact has not changed since the last run. There is no cross-session memory: two users (or the same user two days apart) pay the LLM cost twice for the same model at the same commit.

BOMs are already graph-shaped. A model BOM points to dataset BOMs (`trainedOn`), papers (`describedIn`), code repos (`hostedAt`), licenses, suppliers. The existing "recursive child BOM" feature traverses this graph implicitly. SPDX 3.0.1 is RDF. The triplet field model `{value, source, conflict}` is an attempt to represent provenance-bearing claims in JSON.

We can collapse all of this into a single persistent knowledge graph: store generated BOMs as RDF, dedupe by canonical artifact identity, accumulate user trust on what was generated, and let "world of BOMs" emerge as the union of every BOM anyone in the community has ever produced.

## Goals

1. **Persistent retrieval, with consent.** A BOM for `(artifact, version)` that already exists is offered to the user (with its provenance and trust score) rather than silently reused. The user picks `use`, `regenerate`, or `regenerate-and-keep-both`.
2. **One node per artifact.** Many BOMs may exist for the same upstream artifact — different LLM models, different prompt versions, different users. They are *alternative claims about the same node*, not duplicate nodes. A canonical claim (the highest-trust one) acts as the head.
3. **Graph-native.** Relationships between artifacts (model → dataset, paper → model, supplier → model) are first-class queryable edges.
4. **Provenance-preserving.** The triplet `{value, source, conflict}` round-trips losslessly into RDF and back. Every claim retains the data source that asserted it and the generation run that produced it.
5. **Trust accumulates.** Users can mark a BOM "trusted" / "looks good"; trust scores rank competing claims, and the ranking is exportable so trust survives instance sharing.
6. **Local-first, shareable.** Each install has a local graph. Two instances sync by exchanging a dump file. No registry server required.
7. **Pluggable backend.** Default Oxigraph; RDFLib + N-Quads fallback for environments where Oxigraph wheels are unavailable (HF Spaces, restricted CI).

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
                                    │ user picks: use / regen / keep-both
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

The `Artifact` IRI uses **`sha256(canonical_id)`** rather than embedding the human-readable id, for two reasons: (1) it forces canonicalization to happen at IRI mint time, so two different inputs that canonicalize to the same identity *cannot* produce different IRIs by accident; (2) it makes the IRI URL-safe and bounded length regardless of upstream id quirks (slashes, dots, unicode).

The human-readable `canonical_id` is stored as a string property (`aibom:canonicalId`) on the Artifact node and indexed for search. The original (pre-canonical) input strings are stored as `aibom:alias` properties — multiple aliases per artifact are normal and queryable.

The `GenerationRun` IRI is `sha256(provider + llm_model + prompt_version + code_version + mode + use_case)`. Two generations with identical parameters share a single run node — useful for "how many distinct LLM × prompt combos have produced a claim for this version?" queries.

## Canonical Naming & Dedup

This is the system that prevents "Mistral-7B" and "mistral-7b" and "MistralAI/Mistral-7B-v0.1" from creating three different artifact nodes.

### Canonicalization pipeline

1. **Platform tag**: prefix with `huggingface:` / `github:` / `arxiv:` based on the source URL or explicit flag.
2. **Lowercase** the entire identifier.
3. **Trim and normalize whitespace** (no leading/trailing space, no internal repeats).
4. **Resolve owner via the existing `default_alias_index()`** (`src/aikaboom/utils/supplier_alias.py`). This already handles `mistralai ↔ Mistral AI`, `Qwen ↔ QwenLM`, etc. Reusing it is critical — the project already trusts this index for supplier resolution.
5. **Normalize separators**: collapse `_` and `-` runs, strip trailing version suffixes that duplicate the version field (`-v0.1` only if it matches the resolved version).
6. **Hash** the result with SHA-256. That hash becomes the artifact IRI.

Example:

```
input:           "MistralAI/Mistral-7B-v0.1"
platform-tagged: "huggingface:MistralAI/Mistral-7B-v0.1"
lowercased:      "huggingface:mistralai/mistral-7b-v0.1"
alias-resolved:  "huggingface:mistralai/mistral-7b-v0.1"   (mistralai is already canonical)
canonical_id:    "huggingface:mistralai/mistral-7b-v0.1"
sha256:          a3f8e9c2b1...
Artifact IRI:    bom:artifact/a3f8e9c2b1...
```

All of `MistralAI/Mistral-7B-v0.1`, `mistralai/Mistral-7B-v0.1`, `huggingface.co/mistralai/Mistral-7B-v0.1` (after URL parse) collapse to the same artifact node.

### Near-duplicate detection (out of canonicalization)

Canonicalization handles trivial variants. Genuine near-duplicates — different artifacts with confusingly similar names, e.g., `Mistral-7B` vs `Mistral-7B-Instruct` — are *not* collapsed. They're different artifacts. The Web UI surfaces near-matches on the resolve screen ("did you mean…?") using a simple Levenshtein/token-overlap heuristic, so the user can notice before generating a duplicate by mistake. This is a UX safety net, not a dedup mechanism.

### What this fixes

- Generating a BOM for `mistralai/Mistral-7B-v0.1` and later for `MistralAI/Mistral-7B-v0.1` produces **one Artifact node**, not two.
- Two different LLMs (Claude vs GPT-4o) producing BOMs for the same artifact-version produce **one ArtifactVersion node** with **two BOMClaim children**.
- Recursive walks that encounter `squad` and `rajpurkar/squad` resolve to the same Dataset node.

## Data Flow

### Resolve step (replaces v1's silent cache hit)

Every generation request goes through `BomStore.resolve(input_id, platform, use_case, mode, run_meta)`:

1. **Canonicalize** `input_id` → `canonical_id` → `artifact_iri`.
2. **Resolve version** (single platform API call to get the current commit SHA, as we do today).
3. **Find existing claims** for `(artifact_iri, version, use_case, mode)`:
   ```sparql
   SELECT ?claim ?run ?score ?createdAt WHERE {
     bom:version/<artifact>/<ver> aibom:hasClaim ?claim .
     ?claim aibom:useCase "<use_case>" ;
            aibom:mode "<mode>" ;
            aibom:generatedBy ?run ;
            aibom:trustScore ?score ;
            aibom:createdAt ?createdAt .
   }
   ORDER BY DESC(?score) DESC(?createdAt)
   ```
4. **Decision** (driven by `cache_resolver`):
   - **No existing claims** → fall through to generation.
   - **Existing claims, auto-decide mode** (CLI `--cache=auto`, or web `force_refresh=false`) → return the highest-trust claim within TTL.
   - **Existing claims, prompt mode** (default for interactive use) → show the user:

```
A BOM for mistralai/Mistral-7B-v0.1 @27d67f1b already exists.

  Claim A (canonical, trust ★★★★☆ from 12 votes)
    generated 2025-11-04 with claude-3-haiku, prompt v11
  Claim B (trust ★★☆☆☆ from 2 votes)
    generated 2025-12-19 with gpt-4o-mini, prompt v12

You are about to generate with claude-opus-4-7, prompt v12.

  [u] use canonical (claim A) — zero LLM cost
  [r] regenerate (replace lower-trust claims with same params)
  [k] regenerate and keep all (add a new claim alongside existing)
  [s] show full diff before deciding
```

5. **Honor the choice**:
   - `use`: reconstruct BOM JSON from the chosen claim, return.
   - `regenerate`: run generator, save new claim, mark older same-params claim as `aibom:supersedes`'d.
   - `keep-both`: run generator, save new claim, leave older claims untouched. The trust pointer (`canonicalClaim`) updates after the user later votes.

Non-interactive contexts (CI, web POSTs without UI) get a flag-controlled default: `--cache use|regen|keep-both|prompt`, web `cache_policy` body field. `auto` is alias for `use`.

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

## Trust & Curation

### Vote model

A `TrustVote` is a small node with three properties:

```
bom:vote/<uuid>
    a aibom:TrustVote ;
    aibom:trustVoteFor bom:claim/<claim-uuid> ;
    aibom:votedBy bom:agent/<agent-hash> ;
    aibom:voteKind aibom:trusted ;    # trusted | flagged | disputed
    aibom:votedAt "2026-05-14T..."^^xsd:dateTime ;
    aibom:comment "license field looks wrong in HF source" .   # optional
```

### Score aggregation (v1: simple)

`aibom:trustScore` on a claim is recomputed whenever a vote is added/removed:

```
trustScore = (trusted_votes - flagged_votes) / max(1, total_votes)
```

Range `[-1.0, +1.0]`. Initial score for a freshly generated claim is `0.0`. Generator-self-votes count once but are tagged so they're separable in queries.

This is intentionally crude. Sybil resistance, decay, weighted voting by reviewer reputation — all noted as out of scope for v1 because they only matter at scale, and the trust *data* survives any future scoring change.

### Canonical claim pointer

Each ArtifactVersion has at most one `aibom:canonicalClaim` edge, pointing to the highest-trust claim for that version. Ties broken by recency.

`trust.recompute_canonical(version_iri)` is called:
- After every vote that touches a claim for that version.
- After a new claim is created.
- During `aikaboom graph rebuild` for every version.

The Web UI's "view BOM" page renders the canonical claim by default, with a "see alternatives" toggle.

### UI / CLI surface

| Action | Web UI | CLI |
|---|---|---|
| Mark trusted | "Looks good 👍" button on BOM view | `aikaboom bom trust <claim-iri>` |
| Flag | "Flag for review 🚩" button | `aikaboom bom flag <claim-iri> [--comment "..."]` |
| Dispute | "Dispute" button (between trust and flag) | `aikaboom bom dispute <claim-iri>` |
| List votes | Vote panel on BOM view | `aikaboom bom votes <claim-iri>` |
| List claims for version | "Alternatives" tab | `aikaboom bom claims <artifact-iri> [@version]` |

For multi-user deployments, the agent identity is whichever the deployment sets (env var `AIKABOOM_AGENT_ID` or per-session token). Single-user local install gets a default agent IRI derived from `getpass.getuser() + machine_id`.

### Trust persistence across instances

Votes are quads like everything else and are included in `aikaboom graph export`. Two instances merging their dumps accumulate each other's votes — the canonical pointers recompute on import. Conflicting votes from the same agent (same `bom:agent/...` IRI but different vote kinds for the same claim) are resolved by latest `votedAt`.

## Integration Points

| Existing code | Change |
|---|---|
| `cli.cmd_generate` (`src/aikaboom/cli.py:164`) | Wrap with `BomStore.resolve`; add `--cache use|regen|keep-both|prompt` and `--min-trust` / `--regen-on-low-trust` flags. |
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
- `--cache use|regen|keep-both|prompt` (default `prompt` in TTY, `use` non-TTY).
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
| `test_cache_policies.py` | Each `--cache` value behaves correctly (use/regen/keep-both/prompt). |

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

## Out of Scope (v1)

- SPARQL HTTP endpoint (users run SPARQL via `aikaboom graph query`).
- Public registry / federation protocol beyond dump exchange.
- Browser graph visualizer.
- Schema migration tooling beyond `rebuild` and `migrate-canon`.
- Multi-user auth / ACLs.
- Sybil-resistant trust (reputation-weighted voting, vote decay).
- Replacing JSON / SPDX / CycloneDX exports.

## Success Criteria

1. Generating the same BOM twice (same artifact, same version, same use case, within TTL) prompts the user with the cached claim's provenance and makes zero LLM calls when the user picks `use`.
2. Name variants of the same artifact (`Mistral-7B-v0.1`, `MistralAI/Mistral-7B-v0.1`) collapse to one Artifact node, with both originals retained as aliases.
3. Two BOMs for the same artifact-version generated with different LLMs produce one ArtifactVersion with two BOMClaims; the canonical pointer reflects trust ranking.
4. Marking a claim trusted in the UI updates the canonical pointer if applicable; the vote survives export → import on a fresh install.
5. Recursive walks honor `--min-trust` and `--max-depth` / `--max-breadth`.
6. Round-trip JSON ↔ RDF is lossless for every BOM in `Golden_Set/`.
7. RDFLib fallback is exercised in CI and produces identical results to Oxigraph.
8. No regression in existing CLI/web tests with `AIKABOOM_GRAPH_DISABLE=1`.
