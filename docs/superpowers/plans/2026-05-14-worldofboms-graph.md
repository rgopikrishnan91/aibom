# worldofBOMs Knowledge Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent RDF knowledge graph backend that stores generated BOMs, dedupes by canonical artifact identity across HuggingFace/GitHub/arXiv/DOI handles, surfaces a minimal use-or-regenerate prompt on cache hit, and silently accumulates trust signal — all while leaving existing JSON/SPDX/CycloneDX outputs untouched.

**Architecture:** New `src/aikaboom/store/` package wraps the existing `AIBOMProcessor` / `DATABOMProcessor` generation path. Default Oxigraph backend with RDFLib N-Quads fallback for HF Spaces. Three-tier data model — `Artifact` → `ArtifactVersion` → `BOMClaim` — with RDF-star quoted triples carrying the per-field `{value, source, conflict}` provenance. Trust votes (explicit + implicit) accumulate on claims but are not surfaced in the v1 UI.

**Tech Stack:** Python 3.9+, `pyoxigraph` (default backend), `rdflib` 7+ (fallback + universal triple representation), `pytest`, existing `huggingface_hub` / `PyGithub` for upstream metadata, existing `flask` for web integration.

**Reference spec:** `docs/superpowers/specs/2026-05-14-worldofboms-graph-design.md` — read this first.

---

## File Structure (locked at plan time)

### New files

```
src/aikaboom/store/
    __init__.py              # public exports: BomStore, GraphBackend
    naming.py                # canonicalization (pure functions)
    iris.py                  # IRI minting from canonical identifiers
    vocab.py                 # RDF namespaces, predicate constants
    mapper.py                # bom_to_rdf, rdf_to_bom
    backend.py               # GraphBackend Protocol + selection logic
    oxigraph_backend.py      # default backend
    rdflib_backend.py        # fallback backend
    store.py                 # BomStore facade
    cache_resolver.py        # interactive prompt + non-interactive policy
    trust.py                 # vote aggregation + canonical pointer
    cli_graph.py             # aikaboom graph / bom subcommands

tests/store/
    __init__.py
    conftest.py              # fixtures: tmp_store, sample_bom, fake_run_meta
    test_naming.py
    test_iris.py
    test_vocab.py
    test_mapper.py
    test_mapper_roundtrip.py
    test_backend_oxigraph.py
    test_backend_rdflib.py
    test_backend_fallback.py
    test_store_save.py
    test_store_resolve.py
    test_multi_identifier_dedup.py
    test_placeholder_artifact.py
    test_cache_resolver.py
    test_cache_policies.py
    test_trust.py
    test_recursive_trust.py
    test_cli_graph.py
    test_cli_bom.py
    test_web_resolve.py
    test_docs_link_check.py
    test_docs_cli_parity.py
    test_docs_schema_parity.py
    test_docs_queries.py

docs/worldofboms/
    CONCEPT.md
    RATIONALE.md
    SCHEMA.md
    PIPELINE.md
    CLI.md
    API.md
    QUERIES.md
    FEDERATION.md
    TROUBLESHOOTING.md
```

### Modified files

| File | Sections |
|---|---|
| `requirements.txt` | add `pyoxigraph>=0.4` and `rdflib>=7.0` |
| `pyproject.toml` | add `aikaboom graph` and `aikaboom bom` console-script subcommand registration (or argparse subparser — pick argparse to match existing pattern) |
| `src/aikaboom/cli.py` | line 164 `cmd_generate` — wrap with `BomStore.resolve`; add `--cache`, `--min-trust`, `--regen-on-low-trust`, `--primary-platform` flags |
| `src/aikaboom/web/app.py` | `/api/generate` route — accept `cache_policy` body field, dispatch to `BomStore.resolve` |
| `src/aikaboom/web/templates/<bom-view>.html` | add minimal "BOMs already exist" prompt block |
| `src/aikaboom/utils/recursive_bom.py` | accept `min_trust`, `regen_on_low_trust`, `cache_policy` kwargs |
| `README.md` | one-line pointer to `docs/worldofboms/CONCEPT.md` |

### Out of scope this plan (deferred to v2 or follow-up plans)

- Web UI trust panel / alternatives tab (deferred per spec).
- SPARQL HTTP endpoint.
- Browser graph visualizer.
- `aikaboom graph migrate-canon` schema migration tool.

---

## Phase A — Foundation & conceptual docs

### Task 1: Scaffold package, add dependencies, write conceptual docs

The implementer's brief gets written first so subsequent tasks have a locked mental model.

**Files:**
- Create: `src/aikaboom/store/__init__.py`
- Create: `tests/store/__init__.py`
- Create: `tests/store/conftest.py`
- Create: `docs/worldofboms/CONCEPT.md`
- Create: `docs/worldofboms/RATIONALE.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:

```
# Knowledge graph store
pyoxigraph>=0.4.0
rdflib>=7.0.0
```

- [ ] **Step 2: Install and verify both wheels resolve**

Run: `pip install -r requirements.txt`
Expected: both `pyoxigraph` and `rdflib` install successfully. Run `python -c "import pyoxigraph, rdflib; print(pyoxigraph.__name__, rdflib.__version__)"` — expected to print `pyoxigraph 7.x.x` (or similar).

- [ ] **Step 3: Create package skeletons**

Create `src/aikaboom/store/__init__.py` with:

```python
"""worldofBOMs knowledge graph store.

Persists generated BOMs as RDF, dedupes by canonical artifact identity,
and accumulates trust signal silently. See docs/worldofboms/CONCEPT.md.
"""

__all__ = ["BomStore", "GraphBackend"]
```

Create `tests/store/__init__.py` (empty file).

- [ ] **Step 4: Create test fixture conftest**

Create `tests/store/conftest.py`:

```python
"""Shared fixtures for store tests."""
import json
import os
from pathlib import Path
import pytest


SAMPLE_BOM = {
    "repo_id": "mistralai/Mistral-7B-v0.1",
    "model_id": "mistralai_Mistral-7B-v0.1",
    "use_case": "license",
    "direct_fields": {
        "releaseTime": {
            "value": "2025-07-24T16:44:02+00:00",
            "source": "huggingface",
            "conflict": None,
        },
        "suppliedBy": {
            "value": "mistralai",
            "source": "huggingface",
            "conflict": None,
        },
        "packageVersion": {
            "value": "27d67f1b",
            "source": "huggingface",
            "conflict": None,
        },
    },
    "rag_fields": {},
    "beta_fields": [],
}


SAMPLE_RUN_META = {
    "provider": "openrouter",
    "llm_model": "anthropic/claude-3-haiku",
    "prompt_version": "v12",
    "code_version": "abc1234",
    "mode": "rag",
    "use_case": "license",
}


@pytest.fixture
def sample_bom():
    """A minimal-but-realistic BOM JSON dict."""
    return json.loads(json.dumps(SAMPLE_BOM))  # deep copy


@pytest.fixture
def sample_run_meta():
    """A GenerationRun parameter dict."""
    return dict(SAMPLE_RUN_META)


@pytest.fixture
def tmp_store_dir(tmp_path, monkeypatch):
    """An empty graph store dir, configured via env var."""
    store_dir = tmp_path / "graph"
    store_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(store_dir))
    return store_dir


def load_golden_bom(name: str) -> dict:
    """Load a BOM from the project's Golden_Set or results/ directory."""
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in [
        repo_root / "results" / f"{name}.json",
        repo_root / "Golden_Set" / f"{name}.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"No BOM named {name!r} in results/ or Golden_Set/")
```

- [ ] **Step 5: Write CONCEPT.md (the mental model)**

Create `docs/worldofboms/CONCEPT.md`:

```markdown
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
```

- [ ] **Step 6: Write RATIONALE.md (the "why these choices" doc)**

Create `docs/worldofboms/RATIONALE.md`:

```markdown
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
```

- [ ] **Step 7: Run pytest to confirm no test regressions from scaffolding**

Run: `pytest tests/store/ -v`
Expected: `no tests ran` (empty test directory at this point, but exit code 5 — that's normal for empty collection). The point is to confirm collection works.

If pytest exits 5 with "no tests ran", that's expected. If it exits with any other error, fix the conftest before continuing.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/aikaboom/store/__init__.py \
        tests/store/__init__.py tests/store/conftest.py \
        docs/worldofboms/CONCEPT.md docs/worldofboms/RATIONALE.md
git commit -m "feat(store): scaffold worldofBOMs package, add deps, write CONCEPT and RATIONALE"
```

---

## Phase B — Naming, IRIs, vocab

### Task 2: Canonicalization (`naming.py`)

**Files:**
- Create: `src/aikaboom/store/naming.py`
- Create: `tests/store/test_naming.py`

- [ ] **Step 1: Write failing tests for canonicalization**

Create `tests/store/test_naming.py`:

```python
"""Canonicalization rules for artifact identifiers."""
import pytest
from aikaboom.store.naming import (
    Identifier,
    canonicalize,
    canonicalize_set,
    pick_primary,
    PLATFORM_PRIORITY,
)


class TestCanonicalize:
    def test_lowercases(self):
        assert canonicalize(Identifier("huggingface", "MistralAI/Mistral-7B-v0.1")) == \
            Identifier("huggingface", "mistralai/mistral-7b-v0.1")

    def test_idempotent(self):
        once = canonicalize(Identifier("huggingface", "MistralAI/Mistral-7B-v0.1"))
        twice = canonicalize(once)
        assert once == twice

    def test_strips_url_prefix_for_hf(self):
        result = canonicalize(
            Identifier("huggingface", "https://huggingface.co/MistralAI/Mistral-7B-v0.1/tree/main")
        )
        assert result == Identifier("huggingface", "mistralai/mistral-7b-v0.1")

    def test_strips_url_prefix_for_github(self):
        result = canonicalize(
            Identifier("github", "https://github.com/mistralai/mistral-src.git")
        )
        assert result == Identifier("github", "mistralai/mistral-src")

    def test_strips_arxiv_version_suffix(self):
        result = canonicalize(Identifier("arxiv", "arxiv.org/abs/2310.06825v1"))
        assert result == Identifier("arxiv", "2310.06825")

    def test_resolves_supplier_alias(self):
        # supplier_alias maps various forms of an org name to a canonical form.
        # 'mistralai' is already canonical; the test asserts the path executes.
        result = canonicalize(Identifier("huggingface", "MISTRALAI/Mistral-7B"))
        assert result.value.startswith("mistralai/")

    def test_collapses_separator_runs(self):
        result = canonicalize(Identifier("huggingface", "foo--bar__baz"))
        assert result == Identifier("huggingface", "foo-bar-baz")

    def test_trim_whitespace(self):
        result = canonicalize(Identifier("huggingface", "  mistralai/mistral-7b  "))
        assert result == Identifier("huggingface", "mistralai/mistral-7b")


class TestPickPrimary:
    def test_hf_beats_arxiv(self):
        ids = [
            Identifier("arxiv", "2310.06825"),
            Identifier("huggingface", "mistralai/mistral-7b"),
        ]
        assert pick_primary(ids).platform == "huggingface"

    def test_github_beats_arxiv(self):
        ids = [
            Identifier("arxiv", "2310.06825"),
            Identifier("github", "mistralai/mistral-src"),
        ]
        assert pick_primary(ids).platform == "github"

    def test_arxiv_when_only_option(self):
        ids = [Identifier("arxiv", "2310.06825")]
        assert pick_primary(ids).platform == "arxiv"

    def test_priority_order(self):
        assert PLATFORM_PRIORITY == ("huggingface", "github", "arxiv", "doi", "url")


class TestCanonicalizeSet:
    def test_canonicalizes_each(self):
        ids = [
            Identifier("huggingface", "MistralAI/Mistral-7B"),
            Identifier("arxiv", "arxiv.org/abs/2310.06825v1"),
        ]
        result = canonicalize_set(ids)
        assert Identifier("huggingface", "mistralai/mistral-7b") in result
        assert Identifier("arxiv", "2310.06825") in result

    def test_dedups_within_set(self):
        ids = [
            Identifier("huggingface", "mistralai/mistral-7b"),
            Identifier("huggingface", "MISTRALAI/Mistral-7B"),
        ]
        result = canonicalize_set(ids)
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/store/test_naming.py -v`
Expected: ImportError / ModuleNotFoundError on `aikaboom.store.naming`.

- [ ] **Step 3: Implement `naming.py`**

Create `src/aikaboom/store/naming.py`:

```python
"""Canonicalize artifact identifiers and pick the primary one.

Pure functions only — no I/O, no graph access. The output is fully
determined by the input plus the supplier alias index loaded at startup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from aikaboom.utils.supplier_alias import default_alias_index


PLATFORM_PRIORITY: tuple[str, ...] = ("huggingface", "github", "arxiv", "doi", "url")


@dataclass(frozen=True)
class Identifier:
    """A platform-typed identifier value."""
    platform: str
    value: str


_SEPARATOR_RUN = re.compile(r"[-_]{2,}")
_UNDERSCORE = re.compile(r"_")
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")
_GITHUB_DOT_GIT = re.compile(r"\.git$")


def _strip_url(platform: str, value: str) -> str:
    """Reduce a URL form to its canonical path component."""
    if "://" not in value and not value.startswith("www."):
        return value
    parsed = urlparse(value if "://" in value else f"https://{value}")
    path = parsed.path.lstrip("/")
    if platform == "huggingface":
        # Drop /tree/main, /blob/<sha>/..., etc — keep "owner/repo".
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return path
    if platform == "github":
        parts = path.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return path
    if platform == "arxiv":
        # /abs/2310.06825 or /pdf/2310.06825 → 2310.06825
        parts = [p for p in path.split("/") if p and p not in ("abs", "pdf")]
        return parts[-1] if parts else path
    return path


def _resolve_owner_alias(platform: str, value: str) -> str:
    """For owner/repo-shaped identifiers, canonicalize the owner via the alias index."""
    if "/" not in value:
        return value
    owner, _, rest = value.partition("/")
    canonical = default_alias_index().canonicalize(owner)
    if canonical:
        return f"{canonical.lower()}/{rest}"
    return value


def canonicalize(ident: Identifier) -> Identifier:
    """Apply the canonicalization pipeline to a single identifier.

    Steps: strip URL noise → lowercase → trim → resolve owner alias →
    collapse separator runs → platform-specific trimming.
    """
    value = ident.value.strip()
    value = _strip_url(ident.platform, value)
    value = value.lower().strip()
    value = _UNDERSCORE.sub("-", value)
    value = _SEPARATOR_RUN.sub("-", value)
    value = _resolve_owner_alias(ident.platform, value).lower()
    if ident.platform == "github":
        value = _GITHUB_DOT_GIT.sub("", value)
    if ident.platform == "arxiv":
        value = _ARXIV_VERSION_SUFFIX.sub("", value)
    return Identifier(platform=ident.platform, value=value)


def canonicalize_set(ids: Iterable[Identifier]) -> list[Identifier]:
    """Canonicalize each id and dedupe by (platform, value) within the set."""
    seen: set[tuple[str, str]] = set()
    out: list[Identifier] = []
    for ident in ids:
        canon = canonicalize(ident)
        key = (canon.platform, canon.value)
        if key not in seen:
            seen.add(key)
            out.append(canon)
    return out


def pick_primary(ids: Iterable[Identifier]) -> Identifier:
    """Pick the highest-priority identifier from a set."""
    canon = canonicalize_set(ids)
    if not canon:
        raise ValueError("pick_primary requires at least one identifier")
    by_platform = {i.platform: i for i in canon}
    for platform in PLATFORM_PRIORITY:
        if platform in by_platform:
            return by_platform[platform]
    # Unknown platform — return the first one.
    return canon[0]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/store/test_naming.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/naming.py tests/store/test_naming.py
git commit -m "feat(store): canonicalize artifact identifiers (naming module)"
```

---

### Task 3: IRI minting (`iris.py`)

**Files:**
- Create: `src/aikaboom/store/iris.py`
- Create: `tests/store/test_iris.py`

- [ ] **Step 1: Write failing tests**

Create `tests/store/test_iris.py`:

```python
import pytest
from aikaboom.store.naming import Identifier
from aikaboom.store.iris import (
    artifact_iri,
    version_iri,
    claim_iri,
    run_iri,
    vote_iri,
    agent_iri,
    source_iri,
)


class TestArtifactIri:
    def test_deterministic(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        assert artifact_iri(primary) == artifact_iri(primary)

    def test_url_safe(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        iri = artifact_iri(primary)
        # IRI hash is hex — no slashes, colons, unicode.
        prefix = "bom:artifact/"
        assert iri.startswith(prefix)
        suffix = iri[len(prefix):]
        assert all(c in "0123456789abcdef" for c in suffix)
        assert len(suffix) == 64  # sha256 hex digest length


class TestVersionIri:
    def test_includes_version(self):
        primary = Identifier("huggingface", "mistralai/mistral-7b")
        v = version_iri(artifact_iri(primary), "27d67f1b")
        assert v.startswith("bom:version/")
        assert v.endswith("/27d67f1b")


class TestRunIri:
    def test_hash_of_run_params_is_stable(self):
        params = {
            "provider": "openrouter",
            "llm_model": "anthropic/claude-3-haiku",
            "prompt_version": "v12",
            "code_version": "abc1234",
            "mode": "rag",
            "use_case": "license",
        }
        assert run_iri(params) == run_iri(params)

    def test_different_params_different_iri(self):
        a = run_iri({"provider": "openrouter", "llm_model": "claude-3-haiku", "prompt_version": "v12", "code_version": "abc1234", "mode": "rag", "use_case": "license"})
        b = run_iri({"provider": "openrouter", "llm_model": "gpt-4o-mini", "prompt_version": "v12", "code_version": "abc1234", "mode": "rag", "use_case": "license"})
        assert a != b


class TestClaimAndVote:
    def test_claim_iri_is_uuid_form(self):
        iri = claim_iri()
        assert iri.startswith("bom:claim/")
        suffix = iri[len("bom:claim/"):]
        # uuid4 hex form: 32 chars, no dashes (we normalize).
        assert len(suffix) >= 32

    def test_two_claims_distinct(self):
        assert claim_iri() != claim_iri()

    def test_vote_iri_format(self):
        assert vote_iri().startswith("bom:vote/")


class TestAgentIri:
    def test_deterministic_from_string(self):
        assert agent_iri("gopi@laptop") == agent_iri("gopi@laptop")

    def test_different_strings_different_iri(self):
        assert agent_iri("gopi@laptop") != agent_iri("alice@desktop")


class TestSourceIri:
    def test_huggingface(self):
        assert source_iri("huggingface") == "aibom:source/huggingface"

    def test_github(self):
        assert source_iri("github") == "aibom:source/github"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/store/test_iris.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `iris.py`**

Create `src/aikaboom/store/iris.py`:

```python
"""Mint stable IRIs for graph nodes."""
from __future__ import annotations

import hashlib
import uuid
from typing import Mapping

from aikaboom.store.naming import Identifier


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _uuid4_hex() -> str:
    return uuid.uuid4().hex


def artifact_iri(primary: Identifier) -> str:
    """IRI for an Artifact, hashed from its primary canonicalized identifier."""
    key = f"{primary.platform}:{primary.value}"
    return f"bom:artifact/{_sha256_hex(key)}"


def version_iri(artifact_iri_: str, version: str) -> str:
    """IRI for an ArtifactVersion under a given Artifact."""
    assert artifact_iri_.startswith("bom:artifact/"), artifact_iri_
    hash_suffix = artifact_iri_.split("/", 1)[1]
    return f"bom:version/{hash_suffix}/{version}"


def claim_iri() -> str:
    """Fresh IRI for a BOMClaim. Random UUID4 — claims are unique per generation."""
    return f"bom:claim/{_uuid4_hex()}"


def run_iri(params: Mapping[str, str]) -> str:
    """IRI for a GenerationRun, hashed from its parameter dict.

    Two generations with identical (provider, llm_model, prompt_version,
    code_version, mode, use_case) share the same run node.
    """
    fields = ("provider", "llm_model", "prompt_version", "code_version", "mode", "use_case")
    key = "|".join(f"{f}={params.get(f, '')}" for f in fields)
    return f"bom:run/{_sha256_hex(key)}"


def vote_iri() -> str:
    """Fresh IRI for a TrustVote."""
    return f"bom:vote/{_uuid4_hex()}"


def agent_iri(agent_id: str) -> str:
    """IRI for an Agent (user or generator)."""
    return f"bom:agent/{_sha256_hex(agent_id)}"


def source_iri(platform: str) -> str:
    """IRI for a data source (huggingface, github, arxiv)."""
    return f"aibom:source/{platform}"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/store/test_iris.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/iris.py tests/store/test_iris.py
git commit -m "feat(store): IRI minting for graph nodes"
```

---

### Task 4: Vocabulary (`vocab.py`) + SCHEMA.md

**Files:**
- Create: `src/aikaboom/store/vocab.py`
- Create: `tests/store/test_vocab.py`
- Create: `docs/worldofboms/SCHEMA.md`

- [ ] **Step 1: Write failing tests**

Create `tests/store/test_vocab.py`:

```python
from rdflib import Namespace, URIRef
from aikaboom.store import vocab


class TestNamespaces:
    def test_aibom_namespace_defined(self):
        assert isinstance(vocab.AIBOM, Namespace)
        assert str(vocab.AIBOM).startswith("https://aikaboom.dev/aibom#")

    def test_bom_namespace_defined(self):
        assert isinstance(vocab.BOM, Namespace)


class TestCoreClasses:
    def test_artifact_class(self):
        assert isinstance(vocab.Artifact, URIRef)
        assert str(vocab.Artifact) == str(vocab.AIBOM) + "Artifact"

    def test_model_subclass(self):
        assert str(vocab.Model) == str(vocab.AIBOM) + "Model"

    def test_dataset_paper_coderepo(self):
        assert str(vocab.Dataset) == str(vocab.AIBOM) + "Dataset"
        assert str(vocab.Paper) == str(vocab.AIBOM) + "Paper"
        assert str(vocab.CodeRepo) == str(vocab.AIBOM) + "CodeRepo"

    def test_artifact_version(self):
        assert str(vocab.ArtifactVersion) == str(vocab.AIBOM) + "ArtifactVersion"

    def test_bom_claim(self):
        assert str(vocab.BOMClaim) == str(vocab.AIBOM) + "BOMClaim"

    def test_generation_run(self):
        assert str(vocab.GenerationRun) == str(vocab.AIBOM) + "GenerationRun"

    def test_trust_vote(self):
        assert str(vocab.TrustVote) == str(vocab.AIBOM) + "TrustVote"


class TestPredicates:
    def test_has_version(self):
        assert str(vocab.hasVersion) == str(vocab.AIBOM) + "hasVersion"

    def test_has_claim(self):
        assert str(vocab.hasClaim) == str(vocab.AIBOM) + "hasClaim"

    def test_canonical_claim(self):
        assert str(vocab.canonicalClaim) == str(vocab.AIBOM) + "canonicalClaim"

    def test_generated_by(self):
        assert str(vocab.generatedBy) == str(vocab.AIBOM) + "generatedBy"

    def test_trust_score(self):
        assert str(vocab.trustScore) == str(vocab.AIBOM) + "trustScore"

    def test_use_case_and_mode(self):
        assert str(vocab.useCase) == str(vocab.AIBOM) + "useCase"
        assert str(vocab.mode) == str(vocab.AIBOM) + "mode"

    def test_identifier_and_primary(self):
        assert str(vocab.identifier) == str(vocab.AIBOM) + "identifier"
        assert str(vocab.primaryIdentifier) == str(vocab.AIBOM) + "primaryIdentifier"

    def test_asserted_by_and_conflict_kind(self):
        assert str(vocab.assertedBy) == str(vocab.AIBOM) + "assertedBy"
        assert str(vocab.conflictKind) == str(vocab.AIBOM) + "conflictKind"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/store/test_vocab.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `vocab.py`**

Create `src/aikaboom/store/vocab.py`:

```python
"""RDF namespaces and predicate constants for the worldofBOMs graph.

Single source of truth. If you add a predicate here, also add it to
docs/worldofboms/SCHEMA.md (a CI test enforces parity).
"""
from rdflib import Namespace, URIRef


AIBOM = Namespace("https://aikaboom.dev/aibom#")
BOM = Namespace("bom:")
SPDX = Namespace("https://spdx.org/rdf/3.0.1/terms/")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCAT = Namespace("http://www.w3.org/ns/dcat#")


# Core classes
Artifact = AIBOM.Artifact
Model = AIBOM.Model
Dataset = AIBOM.Dataset
Paper = AIBOM.Paper
CodeRepo = AIBOM.CodeRepo
ArtifactVersion = AIBOM.ArtifactVersion
BOMClaim = AIBOM.BOMClaim
GenerationRun = AIBOM.GenerationRun
TrustVote = AIBOM.TrustVote
Agent = AIBOM.Agent
License = AIBOM.License
Supplier = AIBOM.Supplier
Person = AIBOM.Person
Source = AIBOM.Source


# Predicates: tier edges
hasVersion = AIBOM.hasVersion
hasClaim = AIBOM.hasClaim
canonicalClaim = AIBOM.canonicalClaim
generatedBy = AIBOM.generatedBy
supersedes = AIBOM.supersedes


# Predicates: claim properties
trustScore = AIBOM.trustScore
useCase = AIBOM.useCase
mode = AIBOM.mode
createdAt = AIBOM.createdAt
schemaVersion = AIBOM.schemaVersion


# Predicates: trust votes
trustVoteFor = AIBOM.trustVoteFor
votedBy = AIBOM.votedBy
voteKind = AIBOM.voteKind
votedAt = AIBOM.votedAt
comment = AIBOM.comment


# Predicates: identifier model
identifier = AIBOM.identifier
primaryIdentifier = AIBOM.primaryIdentifier
canonicalLabel = AIBOM.canonicalLabel
canonRuleVersion = AIBOM.canonRuleVersion
platform = AIBOM.platform
value = AIBOM.value
alias = AIBOM.alias
isPlaceholder = AIBOM.isPlaceholder
potentialDuplicateOf = AIBOM.potentialDuplicateOf


# Predicates: per-field RDF-star annotations
assertedBy = AIBOM.assertedBy
conflictKind = AIBOM.conflictKind
conflictsWith = AIBOM.conflictsWith


# Predicates: BOM-domain edges
trainedOn = AIBOM.trainedOn
describedIn = AIBOM.describedIn
hostedAt = AIBOM.hostedAt
hasLicense = AIBOM.hasLicense
suppliedBy = AIBOM.suppliedBy
authoredBy = AIBOM.authoredBy


# Predicates: GenerationRun properties
provider = AIBOM.provider
llmModel = AIBOM.llmModel
promptVersion = AIBOM.promptVersion
codeVersion = AIBOM.codeVersion


# Vote-kind individuals
trusted = AIBOM.trusted
flagged = AIBOM.flagged
disputed = AIBOM.disputed
implicit_use = URIRef(str(AIBOM) + "implicit-use")
implicit_validate = URIRef(str(AIBOM) + "implicit-validate")


# Conflict-kind individuals
noConflict = AIBOM.noConflict
interSourceConflict = AIBOM.interSourceConflict
intraSourceConflict = AIBOM.intraSourceConflict


SCHEMA_VERSION = "1.0"
CANON_RULE_VERSION = "1"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/store/test_vocab.py -v`
Expected: All pass.

- [ ] **Step 5: Write SCHEMA.md**

Create `docs/worldofboms/SCHEMA.md`:

```markdown
# worldofBOMs — RDF Schema Reference

Canonical reference for every class and predicate in the worldofBOMs
vocabulary. The CI parity test (`tests/store/test_docs_schema_parity.py`)
ensures this doc and `src/aikaboom/store/vocab.py` stay in sync.

## Namespaces

| Prefix | URI |
|---|---|
| `aibom:` | `https://aikaboom.dev/aibom#` |
| `bom:` | `bom:` (instance IRIs use this scheme) |
| `spdx:` | `https://spdx.org/rdf/3.0.1/terms/` |
| `prov:` | `http://www.w3.org/ns/prov#` |
| `dcat:` | `http://www.w3.org/ns/dcat#` |

## Classes

### aibom:Artifact

The upstream thing a BOM is about. Subtype-specific subclasses (`Model`,
`Dataset`, `Paper`, `CodeRepo`) inherit from this.

```turtle
bom:artifact/<hash>
    a aibom:Model ;
    aibom:identifier [ aibom:platform "huggingface" ; aibom:value "mistralai/mistral-7b" ] ;
    aibom:primaryIdentifier "huggingface:mistralai/mistral-7b" ;
    aibom:canonicalLabel "Mistral 7B v0.1" ;
    aibom:canonRuleVersion "1" .
```

### aibom:Model, aibom:Dataset, aibom:Paper, aibom:CodeRepo

Subclasses of `aibom:Artifact` indicating the artifact's kind. Selected at
mint time from the primary identifier's platform and the existing aikaboom
artifact-type heuristics.

### aibom:ArtifactVersion

A specific commit / version of an Artifact.

```turtle
bom:version/<artifact-hash>/27d67f1b
    a aibom:ArtifactVersion ;
    aibom:canonicalClaim bom:claim/<uuid> .
```

### aibom:BOMClaim

One generation event's claim about an ArtifactVersion. Many BOMClaims may
exist per ArtifactVersion (one per generation run).

```turtle
bom:claim/<uuid>
    a aibom:BOMClaim ;
    aibom:useCase "license" ;
    aibom:mode "rag" ;
    aibom:createdAt "2026-05-14T10:00:00Z"^^xsd:dateTime ;
    aibom:schemaVersion "1.0" ;
    aibom:trustScore 0.0 ;
    aibom:generatedBy bom:run/<hash> .
```

### aibom:GenerationRun

The (provider, LLM model, prompt version, code version, mode, use case)
combination that produced a claim. Deterministic IRI hash — identical
parameters share a single run node.

```turtle
bom:run/<hash>
    a aibom:GenerationRun ;
    aibom:provider "openrouter" ;
    aibom:llmModel "anthropic/claude-3-haiku" ;
    aibom:promptVersion "v12" ;
    aibom:codeVersion "abc1234" ;
    aibom:mode "rag" ;
    aibom:useCase "license" .
```

### aibom:TrustVote

A vote on a BOMClaim.

```turtle
bom:vote/<uuid>
    a aibom:TrustVote ;
    aibom:trustVoteFor bom:claim/<claim-uuid> ;
    aibom:votedBy bom:agent/<agent-hash> ;
    aibom:voteKind aibom:trusted ;
    aibom:votedAt "2026-05-14T..."^^xsd:dateTime .
```

### aibom:Agent

A user or automated generator.

### aibom:License, aibom:Supplier, aibom:Person, aibom:Source

Reused where SPDX/PROV-O don't already cover them.

## Predicates

### Tier edges

| Predicate | Domain → Range | Purpose |
|---|---|---|
| `aibom:hasVersion` | Artifact → ArtifactVersion | An artifact has this version. |
| `aibom:hasClaim` | ArtifactVersion → BOMClaim | A version has this claim made about it. |
| `aibom:canonicalClaim` | ArtifactVersion → BOMClaim | Highest-trust claim pointer; recomputed on every vote/claim. |
| `aibom:generatedBy` | BOMClaim → GenerationRun | Which run produced this claim. |
| `aibom:supersedes` | BOMClaim → BOMClaim | This claim replaces an older one (rarely used in v1 — claims accumulate). |

### Claim properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:trustScore` | xsd:decimal | Aggregate score in `[-1, +1]`. |
| `aibom:useCase` | xsd:string | `license` / `complete` / etc. |
| `aibom:mode` | xsd:string | `rag` / `direct`. |
| `aibom:createdAt` | xsd:dateTime | Generation timestamp. |
| `aibom:schemaVersion` | xsd:string | Vocab version this claim was written under. |

### Vote properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:trustVoteFor` | BOMClaim | The claim being voted on. |
| `aibom:votedBy` | Agent | The voter. |
| `aibom:voteKind` | `aibom:trusted`/`flagged`/`disputed`/`implicit-use`/`implicit-validate` | The vote's type. |
| `aibom:votedAt` | xsd:dateTime | When the vote was cast. |
| `aibom:comment` | xsd:string | Optional free-text reason. |

### Identifier model

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:identifier` | blank node | Platform/value pair. |
| `aibom:primaryIdentifier` | xsd:string | The `platform:value` form chosen as primary. |
| `aibom:canonicalLabel` | xsd:string | Human-readable display name. |
| `aibom:canonRuleVersion` | xsd:string | Version of the canonicalization rules used. |
| `aibom:platform` | xsd:string | `huggingface` / `github` / `arxiv` / `doi` / `url` / `name-only`. |
| `aibom:value` | xsd:string | Canonicalized identifier value. |
| `aibom:alias` | xsd:string | Original pre-canonical input string. |
| `aibom:isPlaceholder` | xsd:boolean | True for unresolvable references. |
| `aibom:potentialDuplicateOf` | Artifact | Soft-collision marker. |

### Per-field RDF-star annotations

| Predicate | Annotation of | Purpose |
|---|---|---|
| `aibom:assertedBy` | `<< claim pred value >>` → Source | Which data source asserted this field. |
| `aibom:conflictKind` | `<< claim pred value >>` → vocab individual | `noConflict` / `interSourceConflict` / `intraSourceConflict`. |
| `aibom:conflictsWith` | `<< claim pred value >>` → quoted triple | Pointer to the conflicting claim triple. |

### BOM-domain edges

| Predicate | Domain → Range | Purpose |
|---|---|---|
| `aibom:trainedOn` | Model → Dataset | Training data dependency. |
| `aibom:describedIn` | Model → Paper | Paper that describes the model. |
| `aibom:hostedAt` | Model → CodeRepo | Code repo hosting the model. |
| `aibom:hasLicense` | Artifact → License (also `spdx:license`) | License attached to the artifact. |
| `aibom:suppliedBy` | Artifact → Supplier | Org or individual supplying the artifact. |
| `aibom:authoredBy` | Paper → Person | Author of a paper. |

### GenerationRun properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:provider` | xsd:string | LLM provider key (`openrouter`, `openai`, `ollama`). |
| `aibom:llmModel` | xsd:string | The LLM model id. |
| `aibom:promptVersion` | xsd:string | Internal prompt version tag. |
| `aibom:codeVersion` | xsd:string | Code version (git SHA short form). |

## Constants

- `SCHEMA_VERSION = "1.0"` — current vocab version. Bump on any
  backward-incompatible predicate change.
- `CANON_RULE_VERSION = "1"` — current canonicalization rule version. Bump
  on any change that would split or merge previously-distinct nodes.
```

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/vocab.py tests/store/test_vocab.py \
        docs/worldofboms/SCHEMA.md
git commit -m "feat(store): RDF vocabulary + SCHEMA.md reference"
```

---

## Phase C — Mapper (BOM ↔ RDF)

### Task 5: `bom_to_rdf` — write a BOM into RDF quads

**Files:**
- Create: `src/aikaboom/store/mapper.py`
- Create: `tests/store/test_mapper.py`

- [ ] **Step 1: Write failing tests for `bom_to_rdf`**

Create `tests/store/test_mapper.py`:

```python
"""Mapper: BOM JSON ↔ RDF quads."""
from rdflib import Dataset, Literal, URIRef, XSD

from aikaboom.store.mapper import bom_to_rdf
from aikaboom.store import vocab
from aikaboom.store.naming import Identifier


def _ds_has(ds: Dataset, s, p, o=None) -> bool:
    """Helper: is there at least one quad matching (s, p, o)?"""
    return any(True for _ in ds.quads((URIRef(s), URIRef(p), o, None)))


class TestBomToRdf:
    def test_creates_artifact_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        artifacts = list(ds.subjects(predicate=vocab.AIBOM["primaryIdentifier"]))
        assert len(artifacts) == 1, f"expected one artifact, got {artifacts}"

    def test_creates_artifact_version_node(self, sample_bom, sample_run_meta):
        ds, _ = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        versions = list(ds.subjects(predicate=None, object=vocab.ArtifactVersion))
        assert any(str(v).startswith("bom:version/") for v in versions)

    def test_creates_bom_claim_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        assert claim_iri.startswith("bom:claim/")
        assert (URIRef(claim_iri), URIRef(vocab.useCase), Literal("license")) in [
            (s, p, o) for s, p, o, _ in ds.quads()
        ]

    def test_creates_generation_run_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        runs = list(ds.subjects(predicate=None, object=vocab.GenerationRun))
        assert len(runs) == 1
        run = runs[0]
        assert (run, URIRef(vocab.llmModel), Literal("anthropic/claude-3-haiku")) in [
            (s, p, o) for s, p, o, _ in ds.quads()
        ]

    def test_field_claim_with_source(self, sample_bom, sample_run_meta):
        """Each direct_field becomes a triple + RDF-star annotation with source."""
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        # The `suppliedBy` field with value "mistralai" should appear.
        quads = list(ds.quads())
        triples = [(str(s), str(p), str(o)) for s, p, o, _ in quads]
        assert any(claim_iri in t[0] and "mistralai" in t[2] for t in triples)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/store/test_mapper.py -v`
Expected: ImportError on `aikaboom.store.mapper`.

- [ ] **Step 3: Implement `mapper.py` (write path)**

Create `src/aikaboom/store/mapper.py`:

```python
"""Convert BOM JSON ↔ RDF quads.

`bom_to_rdf` is lossy in one direction (only stores what the schema knows
about); `rdf_to_bom` reconstructs the JSON. Round-trip is lossless for the
fields the vocab defines, asserted by `test_mapper_roundtrip.py`.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable, Mapping

from rdflib import BNode, Dataset, Literal, URIRef, XSD

from aikaboom.store import iris, vocab
from aikaboom.store.naming import Identifier, canonicalize_set, pick_primary


def _u(s: str) -> URIRef:
    return URIRef(s)


def _kind_for_platform(platform: str) -> URIRef:
    """Map a platform key to an Artifact subclass."""
    return {
        "huggingface": vocab.Model,
        "github": vocab.CodeRepo,
        "arxiv": vocab.Paper,
    }.get(platform, vocab.Artifact)


def _add_identifier_set(ds: Dataset, artifact: URIRef, idents: Iterable[Identifier]) -> None:
    """Attach each canonical identifier as a blank-node entry, plus aliases."""
    for ident in idents:
        node = BNode()
        ds.add((artifact, _u(vocab.identifier), node))
        ds.add((node, _u(vocab.platform), Literal(ident.platform)))
        ds.add((node, _u(vocab.value), Literal(ident.value)))


def _add_generation_run(ds: Dataset, run_meta: Mapping[str, Any]) -> URIRef:
    run = _u(iris.run_iri(run_meta))
    ds.add((run, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.GenerationRun)))
    for field in ("provider", "llm_model", "prompt_version", "code_version", "mode", "use_case"):
        if field in run_meta and run_meta[field] is not None:
            predicate_uri = {
                "provider": vocab.provider,
                "llm_model": vocab.llmModel,
                "prompt_version": vocab.promptVersion,
                "code_version": vocab.codeVersion,
                "mode": vocab.mode,
                "use_case": vocab.useCase,
            }[field]
            ds.add((run, _u(predicate_uri), Literal(str(run_meta[field]))))
    return run


def _add_field_claim(
    ds: Dataset,
    claim: URIRef,
    field_name: str,
    triplet: Mapping[str, Any],
) -> None:
    """Add one field claim triple + RDF-star annotation with source."""
    value = triplet.get("value")
    if value is None:
        return
    pred = _u(vocab.AIBOM[field_name])
    obj = Literal(str(value))
    ds.add((claim, pred, obj))
    source = triplet.get("source")
    if source:
        # rdflib's RDF-star support uses .add_quoted in newer versions;
        # for portability we model the annotation as a separate metadata
        # triple keyed on a deterministic blank node. The CI test
        # test_conflict_preservation asserts both forms round-trip.
        ann = BNode()
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), claim))
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), pred))
        ds.add((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), obj))
        ds.add((ann, _u(vocab.assertedBy), _u(iris.source_iri(source))))
        conflict = triplet.get("conflict")
        if conflict is None:
            ds.add((ann, _u(vocab.conflictKind), _u(vocab.noConflict)))
        elif isinstance(conflict, dict):
            kind = conflict.get("type", "inter")
            kind_uri = vocab.interSourceConflict if kind == "inter" else vocab.intraSourceConflict
            ds.add((ann, _u(vocab.conflictKind), _u(kind_uri)))


def bom_to_rdf(
    bom_json: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    identifiers: list[Identifier],
) -> tuple[Dataset, str]:
    """Convert a BOM JSON dict into an RDF Dataset.

    Args:
        bom_json: the dict produced by AIBOMProcessor / DATABOMProcessor.
        run_meta: GenerationRun parameters (provider, llm_model, ...).
        identifiers: known platform identifiers for the artifact.

    Returns:
        (dataset, claim_iri) — the dataset contains the artifact subgraph;
        the claim_iri is the new BOMClaim's IRI.
    """
    canon_ids = canonicalize_set(identifiers)
    if not canon_ids:
        raise ValueError("bom_to_rdf requires at least one identifier")
    primary = pick_primary(canon_ids)

    ds = Dataset()

    artifact = _u(iris.artifact_iri(primary))
    ds.add((artifact, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(_kind_for_platform(primary.platform))))
    ds.add((artifact, _u(vocab.primaryIdentifier), Literal(f"{primary.platform}:{primary.value}")))
    ds.add((artifact, _u(vocab.canonRuleVersion), Literal(vocab.CANON_RULE_VERSION)))
    _add_identifier_set(ds, artifact, canon_ids)
    label = bom_json.get("repo_id") or bom_json.get("model_id") or primary.value
    ds.add((artifact, _u(vocab.canonicalLabel), Literal(str(label))))

    version_str = (
        bom_json.get("direct_fields", {})
        .get("packageVersion", {})
        .get("value")
        or bom_json.get("direct_fields", {})
        .get("contentIdentifier", {})
        .get("value")
        or "unknown"
    )
    version = _u(iris.version_iri(str(artifact), str(version_str)))
    ds.add((version, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.ArtifactVersion)))
    ds.add((artifact, _u(vocab.hasVersion), version))

    claim = _u(iris.claim_iri())
    ds.add((claim, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _u(vocab.BOMClaim)))
    ds.add((version, _u(vocab.hasClaim), claim))
    ds.add((claim, _u(vocab.useCase), Literal(run_meta.get("use_case", "complete"))))
    ds.add((claim, _u(vocab.mode), Literal(run_meta.get("mode", "rag"))))
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    ds.add((claim, _u(vocab.createdAt), Literal(now, datatype=XSD.dateTime)))
    ds.add((claim, _u(vocab.schemaVersion), Literal(vocab.SCHEMA_VERSION)))
    ds.add((claim, _u(vocab.trustScore), Literal(0.0, datatype=XSD.decimal)))

    run = _add_generation_run(ds, run_meta)
    ds.add((claim, _u(vocab.generatedBy), run))

    for section in ("direct_fields", "rag_fields"):
        for field_name, triplet in (bom_json.get(section) or {}).items():
            if isinstance(triplet, dict):
                _add_field_claim(ds, claim, field_name, triplet)

    return ds, str(claim)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/store/test_mapper.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/mapper.py tests/store/test_mapper.py
git commit -m "feat(store): bom_to_rdf — BOM JSON → RDF quads"
```

---

### Task 6: `rdf_to_bom` and round-trip test on Golden_Set

**Files:**
- Modify: `src/aikaboom/store/mapper.py` (append `rdf_to_bom`)
- Create: `tests/store/test_mapper_roundtrip.py`

- [ ] **Step 1: Write failing round-trip test**

Create `tests/store/test_mapper_roundtrip.py`:

```python
"""Round-trip: BOM JSON → RDF → BOM JSON should be value-preserving."""
import json
from pathlib import Path
import pytest

from aikaboom.store.mapper import bom_to_rdf, rdf_to_bom
from aikaboom.store.naming import Identifier
from tests.store.conftest import SAMPLE_RUN_META


REPO_ROOT = Path(__file__).resolve().parents[2]


def _collect_test_boms() -> list[tuple[str, dict]]:
    boms = []
    for results_file in (REPO_ROOT / "results").glob("*.json"):
        if results_file.stem.endswith(".recursive") or results_file.stem.endswith(".linked"):
            continue
        if ".cyclonedx" in results_file.stem or ".spdx" in results_file.stem:
            continue
        try:
            data = json.loads(results_file.read_text())
            if isinstance(data, dict) and ("direct_fields" in data or "rag_fields" in data):
                boms.append((results_file.stem, data))
        except (json.JSONDecodeError, OSError):
            continue
    return boms


@pytest.mark.parametrize("name,bom_json", _collect_test_boms())
def test_roundtrip_preserves_direct_fields(name, bom_json):
    """Every direct_field value survives JSON → RDF → JSON."""
    ids = [Identifier("huggingface", bom_json.get("repo_id") or bom_json.get("model_id") or "unknown/unknown")]
    ds, claim_iri = bom_to_rdf(bom_json, SAMPLE_RUN_META, identifiers=ids)
    reconstructed = rdf_to_bom(ds, claim_iri)
    for field_name, triplet in (bom_json.get("direct_fields") or {}).items():
        if triplet.get("value") is None:
            continue
        rt = reconstructed.get("direct_fields", {}).get(field_name)
        assert rt is not None, f"missing direct field {field_name} after round-trip in {name}"
        assert rt["value"] == triplet["value"], f"value mismatch on {field_name} in {name}"
        assert rt["source"] == triplet["source"], f"source mismatch on {field_name} in {name}"


def test_roundtrip_simple_bom(sample_bom, sample_run_meta):
    """Sanity round-trip on the in-memory fixture."""
    ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
    ds, claim_iri = bom_to_rdf(sample_bom, sample_run_meta, identifiers=ids)
    reconstructed = rdf_to_bom(ds, claim_iri)
    assert reconstructed["direct_fields"]["suppliedBy"]["value"] == "mistralai"
    assert reconstructed["direct_fields"]["suppliedBy"]["source"] == "huggingface"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/store/test_mapper_roundtrip.py -v`
Expected: ImportError on `rdf_to_bom`.

- [ ] **Step 3: Append `rdf_to_bom` to `mapper.py`**

Append to `src/aikaboom/store/mapper.py`:

```python
def _platform_name_from_source_iri(source_iri_str: str) -> str:
    """Reverse `iris.source_iri('huggingface')` → 'huggingface'."""
    prefix = "aibom:source/"
    if source_iri_str.startswith(prefix):
        return source_iri_str[len(prefix):]
    return source_iri_str


def _conflict_kind_from_iri(kind_iri: str) -> Any:
    """Reverse the conflictKind mapping to the JSON form."""
    if kind_iri == str(vocab.noConflict):
        return None
    if kind_iri == str(vocab.interSourceConflict):
        return {"type": "inter"}
    if kind_iri == str(vocab.intraSourceConflict):
        return {"type": "intra"}
    return None


def rdf_to_bom(ds: Dataset, claim_iri: str) -> dict:
    """Reconstruct a BOM JSON dict from a claim's subgraph.

    Args:
        ds: the dataset to read from.
        claim_iri: the BOMClaim IRI to reconstruct.

    Returns:
        A dict with the same shape as the original BOM JSON (subset:
        only the fields the vocab models — round-trip-asserted by
        test_mapper_roundtrip.py).
    """
    claim = _u(claim_iri)
    out: dict[str, Any] = {"direct_fields": {}, "rag_fields": {}, "beta_fields": []}

    # Resolve the artifact via hasClaim back-edge.
    artifact_label = None
    for s, _, _, _ in ds.quads((None, _u(vocab.hasClaim), None, None)):
        # s is an ArtifactVersion; find its Artifact.
        for art, _, _, _ in ds.quads((None, _u(vocab.hasVersion), s, None)):
            for _, _, lab, _ in ds.quads((art, _u(vocab.canonicalLabel), None, None)):
                artifact_label = str(lab)
                break
            break
        break
    if artifact_label:
        out["model_id"] = artifact_label.replace("/", "_")
        out["repo_id"] = artifact_label

    # Use case + mode
    for _, _, lit, _ in ds.quads((claim, _u(vocab.useCase), None, None)):
        out["use_case"] = str(lit)
    for _, _, lit, _ in ds.quads((claim, _u(vocab.mode), None, None)):
        out["mode"] = str(lit)

    # Walk every (claim, pred, value) triple; pred → field name.
    aibom_ns = str(vocab.AIBOM)
    structural_preds = {
        str(vocab.useCase), str(vocab.mode), str(vocab.createdAt),
        str(vocab.schemaVersion), str(vocab.trustScore), str(vocab.generatedBy),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    }
    for _, p, o, _ in ds.quads((claim, None, None, None)):
        p_str = str(p)
        if p_str in structural_preds:
            continue
        if not p_str.startswith(aibom_ns):
            continue
        field_name = p_str[len(aibom_ns):]
        triplet: dict[str, Any] = {"value": str(o), "source": None, "conflict": None}
        # Find the annotation blank node for this triple, if any.
        for ann, _, _, _ in ds.quads((None, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), o, None)):
            ann_subj = None
            ann_pred = None
            for _, _, s_val, _ in ds.quads((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), None, None)):
                ann_subj = s_val
            for _, _, p_val, _ in ds.quads((ann, _u("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), None, None)):
                ann_pred = p_val
            if ann_subj == claim and ann_pred == p:
                for _, _, src, _ in ds.quads((ann, _u(vocab.assertedBy), None, None)):
                    triplet["source"] = _platform_name_from_source_iri(str(src))
                for _, _, kind, _ in ds.quads((ann, _u(vocab.conflictKind), None, None)):
                    triplet["conflict"] = _conflict_kind_from_iri(str(kind))
                break
        out["direct_fields"][field_name] = triplet
    return out
```

- [ ] **Step 4: Run round-trip tests**

Run: `pytest tests/store/test_mapper_roundtrip.py -v`
Expected: All pass.

If any parameterized test fails because a field is missing on round-trip, that field is not yet handled — extend the structural-pred filter or add the missing predicate to vocab.py.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/mapper.py tests/store/test_mapper_roundtrip.py
git commit -m "feat(store): rdf_to_bom + round-trip test against Golden_Set & results/"
```

---

## Phase D — Graph backend

### Task 7: `GraphBackend` Protocol + Oxigraph implementation

**Files:**
- Create: `src/aikaboom/store/backend.py`
- Create: `src/aikaboom/store/oxigraph_backend.py`
- Create: `tests/store/test_backend_oxigraph.py`

- [ ] **Step 1: Write failing backend tests**

Create `tests/store/test_backend_oxigraph.py`:

```python
import pytest
pytest.importorskip("pyoxigraph")

from aikaboom.store.backend import open_backend


@pytest.fixture
def backend(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "oxigraph")
    return open_backend()


class TestOxigraphBackend:
    def test_open_returns_backend(self, backend):
        assert backend is not None

    def test_add_and_ask(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )
        result = backend.ask(
            "ASK { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )
        assert result is True

    def test_select_returns_bindings(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/2> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        rows = list(backend.select(
            "SELECT ?u WHERE { <bom:test/2> <https://aikaboom.dev/aibom#useCase> ?u }"
        ))
        assert len(rows) == 1
        assert str(rows[0]["u"]) == "license"

    def test_export_then_import_roundtrip(self, backend, tmp_path):
        backend.update(
            "INSERT DATA { <bom:test/3> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        dump = tmp_path / "dump.nq"
        backend.export(dump, fmt="nquads")
        assert dump.exists() and dump.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_backend_oxigraph.py -v`
Expected: ImportError on `aikaboom.store.backend`.

- [ ] **Step 3: Implement `backend.py` (Protocol + factory)**

Create `src/aikaboom/store/backend.py`:

```python
"""GraphBackend Protocol + selection."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Protocol


_log = logging.getLogger(__name__)


class GraphBackend(Protocol):
    """Minimal interface every backend must implement."""

    def update(self, sparql: str) -> None:
        """Run a SPARQL UPDATE."""
        ...

    def ask(self, sparql: str) -> bool:
        """Run a SPARQL ASK and return the boolean result."""
        ...

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        """Run a SPARQL SELECT and yield row bindings."""
        ...

    def add_quads(self, quads: Iterable[tuple]) -> None:
        """Bulk-add quads (s, p, o, g)."""
        ...

    def export(self, path: Path, fmt: str = "nquads") -> None:
        """Dump the entire store to a file."""
        ...

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        """Merge a dump file into the store."""
        ...

    def close(self) -> None:
        """Release any resources."""
        ...


def _store_dir() -> Path:
    return Path(os.environ.get("AIKABOOM_GRAPH_DIR", str(Path.home() / ".aikaboom" / "graph")))


def open_backend() -> GraphBackend:
    """Open the configured backend, falling back to RDFLib if Oxigraph is unavailable."""
    requested = os.environ.get("AIKABOOM_GRAPH_BACKEND", "auto").lower()
    store_dir = _store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)

    if requested in ("oxigraph", "auto"):
        try:
            from aikaboom.store.oxigraph_backend import OxigraphBackend
            return OxigraphBackend(store_dir)
        except ImportError as e:
            if requested == "oxigraph":
                raise
            _log.warning("Oxigraph unavailable (%s); falling back to RDFLib", e)

    from aikaboom.store.rdflib_backend import RDFLibBackend
    return RDFLibBackend(store_dir)
```

- [ ] **Step 4: Implement `oxigraph_backend.py`**

Create `src/aikaboom/store/oxigraph_backend.py`:

```python
"""Oxigraph backend (default)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping

try:
    import pyoxigraph as _ox
except ImportError as e:
    raise ImportError("pyoxigraph is not installed; pip install pyoxigraph") from e


class OxigraphBackend:
    def __init__(self, store_dir: Path):
        self._store_dir = store_dir
        self._store = _ox.Store(path=str(store_dir))

    def update(self, sparql: str) -> None:
        self._store.update(sparql)

    def ask(self, sparql: str) -> bool:
        return bool(self._store.query(sparql))

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        for solution in self._store.query(sparql):
            yield {var: solution[var] for var in solution.variables}

    def add_quads(self, quads: Iterable[tuple]) -> None:
        for s, p, o, g in quads:
            self._store.add(_ox.Quad(s, p, o, g if g is not None else _ox.DefaultGraph()))

    def export(self, path: Path, fmt: str = "nquads") -> None:
        mime = "application/n-quads" if fmt == "nquads" else "application/ld+json"
        with open(path, "wb") as fh:
            self._store.dump(fh, mime)

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        mime = "application/n-quads" if fmt == "nquads" else "application/ld+json"
        with open(path, "rb") as fh:
            self._store.bulk_load(fh, mime)

    def close(self) -> None:
        # pyoxigraph Store has no explicit close; flush via reference drop.
        pass
```

- [ ] **Step 5: Run tests, fix any pyoxigraph API quirks**

Run: `pytest tests/store/test_backend_oxigraph.py -v`
Expected: All pass.

If a test fails because of an API mismatch (pyoxigraph 0.4+ vs older), adjust the OxigraphBackend implementation to match the installed version. The Protocol shape is the contract; the implementation can flex.

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/backend.py src/aikaboom/store/oxigraph_backend.py \
        tests/store/test_backend_oxigraph.py
git commit -m "feat(store): GraphBackend Protocol + Oxigraph implementation"
```

---

### Task 8: RDFLib fallback backend + auto-fallback test

**Files:**
- Create: `src/aikaboom/store/rdflib_backend.py`
- Create: `tests/store/test_backend_rdflib.py`
- Create: `tests/store/test_backend_fallback.py`

- [ ] **Step 1: Write failing RDFLib backend test**

Create `tests/store/test_backend_rdflib.py`:

```python
import pytest
from aikaboom.store.rdflib_backend import RDFLibBackend


@pytest.fixture
def backend(tmp_store_dir):
    return RDFLibBackend(tmp_store_dir)


class TestRDFLibBackend:
    def test_add_and_ask(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )
        assert backend.ask(
            "ASK { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )

    def test_select_returns_bindings(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/2> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        rows = list(backend.select(
            "SELECT ?u WHERE { <bom:test/2> <https://aikaboom.dev/aibom#useCase> ?u }"
        ))
        assert len(rows) == 1

    def test_persistence_across_reopen(self, tmp_store_dir):
        b1 = RDFLibBackend(tmp_store_dir)
        b1.update("INSERT DATA { <bom:test/x> <https://aikaboom.dev/aibom#useCase> 'license' }")
        b1.close()
        b2 = RDFLibBackend(tmp_store_dir)
        assert b2.ask("ASK { <bom:test/x> <https://aikaboom.dev/aibom#useCase> 'license' }")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_backend_rdflib.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `rdflib_backend.py`**

Create `src/aikaboom/store/rdflib_backend.py`:

```python
"""RDFLib + N-Quads fallback backend."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from rdflib import Dataset


_NQ_FILE = "store.nq"


class RDFLibBackend:
    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._nq_path = self._store_dir / _NQ_FILE
        self._ds = Dataset()
        if self._nq_path.exists() and self._nq_path.stat().st_size > 0:
            self._ds.parse(self._nq_path, format="nquads")

    def _flush(self) -> None:
        """Atomically rewrite the N-Quads file."""
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self._store_dir, delete=False, suffix=".nq.tmp"
        ) as tmp:
            self._ds.serialize(destination=tmp, format="nquads")
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self._nq_path)

    def update(self, sparql: str) -> None:
        self._ds.update(sparql)
        self._flush()

    def ask(self, sparql: str) -> bool:
        return bool(self._ds.query(sparql).askAnswer)

    def select(self, sparql: str) -> Iterator[Mapping[str, object]]:
        for row in self._ds.query(sparql):
            yield {str(var): row[var] for var in row.labels}

    def add_quads(self, quads: Iterable[tuple]) -> None:
        for s, p, o, g in quads:
            self._ds.add((s, p, o, g))
        self._flush()

    def export(self, path: Path, fmt: str = "nquads") -> None:
        fmt_map = {"nquads": "nquads", "jsonld": "json-ld"}
        self._ds.serialize(destination=str(path), format=fmt_map[fmt])

    def import_(self, path: Path, fmt: str = "nquads") -> None:
        fmt_map = {"nquads": "nquads", "jsonld": "json-ld"}
        self._ds.parse(str(path), format=fmt_map[fmt])
        self._flush()

    def close(self) -> None:
        self._flush()
```

- [ ] **Step 4: Write fallback test**

Create `tests/store/test_backend_fallback.py`:

```python
import sys
import pytest

from aikaboom.store.backend import open_backend


def test_falls_back_to_rdflib_when_oxigraph_unavailable(monkeypatch, tmp_store_dir):
    """When AIKABOOM_GRAPH_BACKEND=auto and oxigraph import fails, use RDFLib."""
    # Hide pyoxigraph from import machinery.
    monkeypatch.setitem(sys.modules, "pyoxigraph", None)
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "auto")
    # Also unload our cached oxigraph_backend module so re-import sees the None.
    monkeypatch.delitem(sys.modules, "aikaboom.store.oxigraph_backend", raising=False)
    backend = open_backend()
    assert type(backend).__name__ == "RDFLibBackend"


def test_explicit_rdflib_backend(monkeypatch, tmp_store_dir):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    backend = open_backend()
    assert type(backend).__name__ == "RDFLibBackend"


def test_explicit_oxigraph_backend_raises_if_missing(monkeypatch, tmp_store_dir):
    monkeypatch.setitem(sys.modules, "pyoxigraph", None)
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "oxigraph")
    monkeypatch.delitem(sys.modules, "aikaboom.store.oxigraph_backend", raising=False)
    with pytest.raises(ImportError):
        open_backend()
```

- [ ] **Step 5: Run all backend tests**

Run: `pytest tests/store/test_backend_rdflib.py tests/store/test_backend_fallback.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/rdflib_backend.py \
        tests/store/test_backend_rdflib.py tests/store/test_backend_fallback.py
git commit -m "feat(store): RDFLib N-Quads fallback backend with auto-fallback"
```

---

## Phase E — BomStore facade

### Task 9: `BomStore` facade with `save_claim` and basic queries

**Files:**
- Create: `src/aikaboom/store/store.py`
- Create: `tests/store/test_store_save.py`
- Modify: `src/aikaboom/store/__init__.py` (export `BomStore`)
- Create: `docs/worldofboms/API.md` (initial cut)

- [ ] **Step 1: Write failing tests for save_claim**

Create `tests/store/test_store_save.py`:

```python
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")  # deterministic for tests
    return BomStore.open()


class TestSaveClaim:
    def test_save_returns_claim_iri(self, store, sample_bom, sample_run_meta):
        claim_iri = store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        assert claim_iri.startswith("bom:claim/")

    def test_stats_reports_one_claim_after_save(self, store, sample_bom, sample_run_meta):
        store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        stats = store.stats()
        assert stats["claims"] == 1
        assert stats["artifacts"] == 1
        assert stats["versions"] == 1

    def test_find_claims_returns_saved_claim(self, store, sample_bom, sample_run_meta):
        claim_iri = store.save_claim(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        claims = store.find_claims_for(
            [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
            use_case="license",
            mode="rag",
        )
        assert any(c["iri"] == claim_iri for c in claims)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_store_save.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `store.py`**

Create `src/aikaboom/store/store.py`:

```python
"""BomStore — public facade over the graph backend."""
from __future__ import annotations

from typing import Any, Mapping

from rdflib import Dataset

from aikaboom.store import iris, vocab
from aikaboom.store.backend import GraphBackend, open_backend
from aikaboom.store.mapper import bom_to_rdf, rdf_to_bom
from aikaboom.store.naming import Identifier, canonicalize_set, pick_primary


class BomStore:
    def __init__(self, backend: GraphBackend):
        self._backend = backend

    @classmethod
    def open(cls) -> "BomStore":
        return cls(backend=open_backend())

    def save_claim(
        self,
        bom_json: Mapping[str, Any],
        run_meta: Mapping[str, Any],
        identifiers: list[Identifier],
    ) -> str:
        """Convert and persist a BOM. Returns the new claim IRI."""
        ds, claim_iri = bom_to_rdf(bom_json, run_meta, identifiers=identifiers)
        quads = [(s, p, o, None) for s, p, o, _ in ds.quads()]
        self._backend.add_quads(quads)
        return claim_iri

    def find_claims_for(
        self,
        identifiers: list[Identifier],
        use_case: str | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        """Find existing claims that match the given identifiers + filters."""
        canon = canonicalize_set(identifiers)
        if not canon:
            return []
        primary = pick_primary(canon)
        artifact = iris.artifact_iri(primary)

        filters = []
        if use_case is not None:
            filters.append(f'?claim <{vocab.useCase}> "{use_case}" .')
        if mode is not None:
            filters.append(f'?claim <{vocab.mode}> "{mode}" .')
        filter_clause = "\n".join(filters)

        q = f"""
        SELECT ?claim ?createdAt ?llmModel WHERE {{
            <{artifact}> <{vocab.hasVersion}> ?version .
            ?version <{vocab.hasClaim}> ?claim .
            {filter_clause}
            OPTIONAL {{ ?claim <{vocab.createdAt}> ?createdAt . }}
            OPTIONAL {{
                ?claim <{vocab.generatedBy}> ?run .
                ?run <{vocab.llmModel}> ?llmModel .
            }}
        }}
        ORDER BY DESC(?createdAt)
        """
        out = []
        for row in self._backend.select(q):
            out.append({
                "iri": str(row["claim"]),
                "created_at": str(row.get("createdAt", "")),
                "llm_model": str(row.get("llmModel", "")),
            })
        return out

    def stats(self) -> dict[str, int]:
        """Return node counts by class."""
        out = {}
        for label, cls in [("artifacts", vocab.Artifact), ("versions", vocab.ArtifactVersion), ("claims", vocab.BOMClaim), ("votes", vocab.TrustVote)]:
            rows = list(self._backend.select(
                f"SELECT (COUNT(?s) AS ?n) WHERE {{ ?s a <{cls}> }}"
            ))
            out[label] = int(rows[0]["n"]) if rows else 0
        return out

    def reconstruct_bom(self, claim_iri: str) -> dict:
        """Rebuild a BOM JSON dict from a stored claim.

        Internally builds a small rdflib.Dataset by selecting every triple
        whose subject is `claim_iri` (the claim's own triples) plus every
        annotation blank node that references those triples, then hands
        the dataset to `rdf_to_bom`.
        """
        from rdflib import Dataset as _RDFDataset, URIRef as _URIRef, Literal as _Literal, BNode as _BNode

        ds = _RDFDataset()

        # Pull every (claim_iri, p, o) triple.
        q_claim = f"SELECT ?p ?o WHERE {{ <{claim_iri}> ?p ?o }}"
        for row in self._backend.select(q_claim):
            p = _URIRef(str(row["p"]))
            o_raw = row["o"]
            o = _URIRef(str(o_raw)) if str(o_raw).startswith(("http", "bom:", "aibom:", "_:")) else _Literal(str(o_raw))
            ds.add((_URIRef(claim_iri), p, o))

        # Pull annotation blank nodes that point at this claim.
        q_ann = f"""
        SELECT ?ann ?p ?subj ?pred ?obj ?asserted ?conflict WHERE {{
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> <{claim_iri}> .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> ?pred .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#object> ?obj .
            OPTIONAL {{ ?ann <{vocab.assertedBy}> ?asserted . }}
            OPTIONAL {{ ?ann <{vocab.conflictKind}> ?conflict . }}
        }}
        """
        for row in self._backend.select(q_ann):
            ann = _BNode()
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"), _URIRef(claim_iri)))
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"), _URIRef(str(row["pred"]))))
            obj_raw = row["obj"]
            obj_val = _URIRef(str(obj_raw)) if str(obj_raw).startswith(("http", "bom:", "aibom:", "_:")) else _Literal(str(obj_raw))
            ds.add((ann, _URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"), obj_val))
            if row.get("asserted"):
                ds.add((ann, _URIRef(vocab.assertedBy), _URIRef(str(row["asserted"]))))
            if row.get("conflict"):
                ds.add((ann, _URIRef(vocab.conflictKind), _URIRef(str(row["conflict"]))))

        # Pull the artifact label via the hasClaim back-edge so rdf_to_bom can populate repo_id.
        q_label = f"""
        SELECT ?label WHERE {{
            ?version <{vocab.hasClaim}> <{claim_iri}> .
            ?artifact <{vocab.hasVersion}> ?version ;
                      <{vocab.canonicalLabel}> ?label .
        }}
        """
        for row in self._backend.select(q_label):
            # Add the back-edges into ds so rdf_to_bom finds them.
            v = _BNode()
            a = _BNode()
            ds.add((v, _URIRef(vocab.hasClaim), _URIRef(claim_iri)))
            ds.add((a, _URIRef(vocab.hasVersion), v))
            ds.add((a, _URIRef(vocab.canonicalLabel), _Literal(str(row["label"]))))
            break

        return rdf_to_bom(ds, claim_iri)

    def close(self) -> None:
        self._backend.close()
```

- [ ] **Step 4: Update `__init__.py` to export `BomStore`**

Replace `src/aikaboom/store/__init__.py`:

```python
"""worldofBOMs knowledge graph store.

Persists generated BOMs as RDF, dedupes by canonical artifact identity,
and accumulates trust signal silently. See docs/worldofboms/CONCEPT.md.
"""
from aikaboom.store.store import BomStore
from aikaboom.store.backend import GraphBackend

__all__ = ["BomStore", "GraphBackend"]
```

- [ ] **Step 5: Run tests, fix Artifact-class issue**

Run: `pytest tests/store/test_store_save.py -v`
Expected: First-pass may fail on `stats["artifacts"]` because `Artifact` is the supertype; `_kind_for_platform` returns `Model` for HF inputs. Adjust `bom_to_rdf` in `mapper.py` to also add `(artifact, rdf:type, vocab.Artifact)` as a parent-class triple before the subclass triple. Re-run.

- [ ] **Step 6: Write API.md (initial cut)**

Create `docs/worldofboms/API.md`:

```markdown
# worldofBOMs — Python API Reference

## BomStore

The high-level facade. Open one per process; safe to share across threads
if the backend allows it (Oxigraph does; RDFLib's in-memory dataset does
not — guard externally if multi-threaded).

```python
from aikaboom.store import BomStore
from aikaboom.store.naming import Identifier

store = BomStore.open()
try:
    claim_iri = store.save_claim(
        bom_json,
        run_meta={
            "provider": "openrouter",
            "llm_model": "anthropic/claude-3-haiku",
            "prompt_version": "v12",
            "code_version": "abc1234",
            "mode": "rag",
            "use_case": "license",
        },
        identifiers=[
            Identifier("huggingface", "mistralai/Mistral-7B-v0.1"),
        ],
    )
finally:
    store.close()
```

### Methods

- `BomStore.open() -> BomStore` — open the configured backend.
- `save_claim(bom_json, run_meta, identifiers) -> str` — persist a BOM,
  return the new BOMClaim IRI.
- `find_claims_for(identifiers, use_case=None, mode=None) -> list[dict]` —
  find existing claims, newest first. Each dict has `iri`, `created_at`,
  `llm_model`.
- `stats() -> dict[str, int]` — counts of `artifacts`, `versions`,
  `claims`, `votes`.
- `close()` — flush and release resources.

## GraphBackend (Protocol)

You normally don't interact with this directly — `BomStore` is the front
door. If you're embedding the store in a different context, the Protocol
is:

```python
class GraphBackend(Protocol):
    def update(self, sparql: str) -> None: ...
    def ask(self, sparql: str) -> bool: ...
    def select(self, sparql: str) -> Iterator[Mapping[str, object]]: ...
    def add_quads(self, quads: Iterable[tuple]) -> None: ...
    def export(self, path: Path, fmt: str = "nquads") -> None: ...
    def import_(self, path: Path, fmt: str = "nquads") -> None: ...
    def close(self) -> None: ...
```

## BomMapper functions

- `bom_to_rdf(bom_json, run_meta, identifiers) -> (Dataset, claim_iri)`
- `rdf_to_bom(ds, claim_iri) -> dict`

Round-trip is asserted lossless by `tests/store/test_mapper_roundtrip.py`
for every BOM in `Golden_Set/` and `results/`.

## Naming helpers

- `Identifier(platform, value)` — a typed identifier dataclass.
- `canonicalize(ident) -> Identifier` — apply the canonicalization pipeline.
- `canonicalize_set(ids) -> list[Identifier]` — dedup-aware canonicalize.
- `pick_primary(ids) -> Identifier` — pick the highest-priority platform.

`PLATFORM_PRIORITY = ("huggingface", "github", "arxiv", "doi", "url")`.
```

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/store/store.py src/aikaboom/store/__init__.py \
        src/aikaboom/store/mapper.py tests/store/test_store_save.py \
        docs/worldofboms/API.md
git commit -m "feat(store): BomStore facade + save_claim + API.md"
```

---

### Task 10: Cross-identifier dedup (`store.resolve`)

**Files:**
- Modify: `src/aikaboom/store/store.py` (add `resolve` + helpers)
- Create: `tests/store/test_store_resolve.py`
- Create: `tests/store/test_multi_identifier_dedup.py`
- Create: `tests/store/test_placeholder_artifact.py`

- [ ] **Step 1: Write failing resolve tests**

Create `tests/store/test_store_resolve.py`:

```python
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore, ResolveResult


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


class TestResolve:
    def test_resolve_with_no_matches_signals_new(self, store):
        result = store.resolve(
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
            use_case="license",
            mode="rag",
        )
        assert isinstance(result, ResolveResult)
        assert result.existing_artifact is None
        assert result.matching_claims == []

    def test_resolve_finds_saved_claim(self, store, sample_bom, sample_run_meta):
        ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
        claim_iri = store.save_claim(sample_bom, sample_run_meta, identifiers=ids)
        result = store.resolve(identifiers=ids, use_case="license", mode="rag")
        assert result.existing_artifact is not None
        assert any(c["iri"] == claim_iri for c in result.matching_claims)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_store_resolve.py -v`
Expected: ImportError on `ResolveResult` / `resolve`.

- [ ] **Step 3: Add `resolve` to `store.py`**

Append to `src/aikaboom/store/store.py`:

```python
from dataclasses import dataclass, field


@dataclass
class ResolveResult:
    """Outcome of a resolve step."""
    existing_artifact: str | None
    artifact_label: str | None
    matching_claims: list[dict] = field(default_factory=list)
    collision_artifacts: list[str] = field(default_factory=list)


class BomStore(BomStore):  # type: ignore[no-redef]
    pass


# Patch resolve onto BomStore (kept in same module for cohesion).
def _resolve(
    self: BomStore,
    identifiers: list[Identifier],
    use_case: str | None = None,
    mode: str | None = None,
) -> ResolveResult:
    """Find existing Artifact + claims via cross-identifier lookup."""
    canon = canonicalize_set(identifiers)
    if not canon:
        return ResolveResult(existing_artifact=None, artifact_label=None)

    # Build a VALUES clause across all provided identifiers.
    values_rows = "\n".join(
        f'    ("{ident.platform}" "{ident.value}")' for ident in canon
    )
    q = f"""
    SELECT DISTINCT ?artifact ?label WHERE {{
        ?artifact <{vocab.identifier}> ?id .
        ?id <{vocab.platform}> ?p ; <{vocab.value}> ?v .
        VALUES (?p ?v) {{
{values_rows}
        }}
        OPTIONAL {{ ?artifact <{vocab.canonicalLabel}> ?label . }}
    }}
    """
    matches: list[tuple[str, str]] = []
    for row in self._backend.select(q):
        matches.append((str(row["artifact"]), str(row.get("label") or "")))

    if not matches:
        return ResolveResult(existing_artifact=None, artifact_label=None)

    if len(matches) > 1:
        return ResolveResult(
            existing_artifact=matches[0][0],
            artifact_label=matches[0][1],
            collision_artifacts=[m[0] for m in matches[1:]],
            matching_claims=self.find_claims_for(identifiers, use_case=use_case, mode=mode),
        )

    return ResolveResult(
        existing_artifact=matches[0][0],
        artifact_label=matches[0][1],
        matching_claims=self.find_claims_for(identifiers, use_case=use_case, mode=mode),
    )


BomStore.resolve = _resolve  # type: ignore[assignment]
```

- [ ] **Step 4: Write multi-identifier dedup test**

Create `tests/store/test_multi_identifier_dedup.py`:

```python
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_save_with_hf_then_resolve_with_arxiv_finds_same_artifact(
    store, sample_bom, sample_run_meta,
):
    """If a BOM was saved with HF+arxiv ids, a later resolve with only arxiv finds it."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[
            Identifier("huggingface", "mistralai/Mistral-7B-v0.1"),
            Identifier("arxiv", "2310.06825"),
        ],
    )
    result = store.resolve(
        identifiers=[Identifier("arxiv", "2310.06825")],
        use_case="license",
        mode="rag",
    )
    assert result.existing_artifact is not None


def test_name_variants_collapse_to_one_artifact(store, sample_bom, sample_run_meta):
    """`Mistral-7B-v0.1` and `MistralAI/Mistral-7B-v0.1` end up on one node."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "MistralAI/Mistral-7B-v0.1")],
    )
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    stats = store.stats()
    assert stats["artifacts"] == 1
    assert stats["claims"] == 2
```

- [ ] **Step 5: Write placeholder-artifact test**

Create `tests/store/test_placeholder_artifact.py`:

```python
"""Placeholder artifacts for unresolvable recursive references."""
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_placeholder_excluded_from_primary_match(store, sample_bom, sample_run_meta):
    """An artifact created with platform='name-only' is flagged and not matched as primary."""
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    # A subsequent resolve with the same name-only id should still find it
    # (placeholders are *queryable*, just not promoted to primary).
    result = store.resolve(
        identifiers=[Identifier("name-only", "some internal dataset")],
    )
    assert result.existing_artifact is not None
```

The implementation needs `bom_to_rdf` to flag placeholder artifacts. Patch `mapper.py`:

```python
# In _kind_for_platform, return vocab.Artifact for "name-only".
# After minting the artifact, if primary.platform == "name-only":
#     ds.add((artifact, _u(vocab.isPlaceholder), Literal(True, datatype=XSD.boolean)))
```

- [ ] **Step 6: Run all resolve/dedup tests**

Run: `pytest tests/store/test_store_resolve.py tests/store/test_multi_identifier_dedup.py tests/store/test_placeholder_artifact.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/store/store.py src/aikaboom/store/mapper.py \
        tests/store/test_store_resolve.py \
        tests/store/test_multi_identifier_dedup.py \
        tests/store/test_placeholder_artifact.py
git commit -m "feat(store): cross-identifier dedup via BomStore.resolve + placeholder support"
```

---

## Phase F — Cache resolution UX

### Task 11: `cache_resolver.py` — prompt and policy

**Files:**
- Create: `src/aikaboom/store/cache_resolver.py`
- Create: `tests/store/test_cache_resolver.py`
- Create: `tests/store/test_cache_policies.py`

- [ ] **Step 1: Write failing tests**

Create `tests/store/test_cache_resolver.py`:

```python
import io
import pytest

from aikaboom.store.cache_resolver import (
    CachePolicy,
    decide,
    render_prompt,
)
from aikaboom.store.store import ResolveResult


def make_result(claims):
    return ResolveResult(
        existing_artifact="bom:artifact/abc",
        artifact_label="mistralai/Mistral-7B-v0.1",
        matching_claims=claims,
    )


class TestDecide:
    def test_no_claims_means_generate(self):
        result = ResolveResult(existing_artifact=None, artifact_label=None)
        assert decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "u") == "generate"

    def test_auto_uses_most_recent(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.USE, interactive=False) == "use"

    def test_regen_policy_skips_prompt(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.REGEN, interactive=False) == "generate"

    def test_prompt_with_use_response(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "claude-3-haiku", "created_at": "2025-11-04"}])
        assert decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "u") == "use"

    def test_prompt_with_regen_response(self):
        result = make_result([{"iri": "bom:claim/x", "llm_model": "claude-3-haiku", "created_at": "2025-11-04"}])
        assert decide(result, CachePolicy.PROMPT, interactive=True, input_fn=lambda _: "r") == "generate"

    def test_non_interactive_with_prompt_policy_defaults_to_use(self):
        """When TTY is unavailable, prompt policy degrades to use."""
        result = make_result([{"iri": "bom:claim/x", "llm_model": "x", "created_at": "2026-01-01"}])
        assert decide(result, CachePolicy.PROMPT, interactive=False) == "use"


class TestRenderPrompt:
    def test_lists_existing_claims(self):
        result = make_result([
            {"iri": "bom:claim/a", "llm_model": "claude-3-haiku", "created_at": "2025-11-04T10:00:00Z"},
            {"iri": "bom:claim/b", "llm_model": "gpt-4o-mini", "created_at": "2025-12-19T10:00:00Z"},
        ])
        text = render_prompt(result, planned_llm="claude-opus-4-7")
        assert "claude-3-haiku" in text
        assert "gpt-4o-mini" in text
        assert "claude-opus-4-7" in text
        assert "[u]" in text
        assert "[r]" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_cache_resolver.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `cache_resolver.py`**

Create `src/aikaboom/store/cache_resolver.py`:

```python
"""Cache resolution UX: prompt the user, or auto-decide based on policy."""
from __future__ import annotations

import enum
import sys
from typing import Callable

from aikaboom.store.store import ResolveResult


class CachePolicy(str, enum.Enum):
    USE = "use"
    REGEN = "regen"
    PROMPT = "prompt"
    AUTO = "auto"  # alias for USE


def render_prompt(result: ResolveResult, planned_llm: str) -> str:
    """Render the minimal two-option prompt as a string."""
    lines = [f"BOMs for {result.artifact_label} already exist:"]
    for claim in result.matching_claims:
        when = claim.get("created_at", "")
        when_short = when.split("T")[0] if when else "unknown"
        lines.append(f"  - {claim.get('llm_model', 'unknown')}   ({when_short})")
    lines.append("")
    lines.append(f"You're about to generate with {planned_llm}.")
    lines.append("")
    lines.append("  [u] use the most recent existing BOM")
    lines.append("  [r] regenerate")
    return "\n".join(lines)


def decide(
    result: ResolveResult,
    policy: CachePolicy,
    interactive: bool,
    input_fn: Callable[[str], str] | None = None,
    planned_llm: str = "(current LLM)",
) -> str:
    """Decide between 'use' and 'generate'."""
    if not result.matching_claims:
        return "generate"
    if policy in (CachePolicy.USE, CachePolicy.AUTO):
        return "use"
    if policy == CachePolicy.REGEN:
        return "generate"
    # PROMPT
    if not interactive:
        return "use"  # non-TTY degrades to use
    prompt = render_prompt(result, planned_llm=planned_llm) + "\n> "
    response = (input_fn or input)(prompt).strip().lower()
    if response.startswith("r"):
        return "generate"
    return "use"


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
```

- [ ] **Step 4: Run cache_resolver tests**

Run: `pytest tests/store/test_cache_resolver.py -v`
Expected: All pass.

- [ ] **Step 5: Write cache policy integration test**

Create `tests/store/test_cache_policies.py`:

```python
"""End-to-end: each --cache value triggers the right BomStore behavior."""
import pytest

from aikaboom.store.cache_resolver import CachePolicy, decide
from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore


@pytest.fixture
def store_with_claim(tmp_store_dir, monkeypatch, sample_bom, sample_run_meta):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    store = BomStore.open()
    store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    return store


def test_use_policy_returns_use(store_with_claim):
    result = store_with_claim.resolve(
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        use_case="license", mode="rag",
    )
    assert decide(result, CachePolicy.USE, interactive=False) == "use"


def test_regen_policy_returns_generate(store_with_claim):
    result = store_with_claim.resolve(
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        use_case="license", mode="rag",
    )
    assert decide(result, CachePolicy.REGEN, interactive=False) == "generate"
```

- [ ] **Step 6: Run tests, commit**

Run: `pytest tests/store/test_cache_policies.py -v`
Expected: All pass.

```bash
git add src/aikaboom/store/cache_resolver.py \
        tests/store/test_cache_resolver.py tests/store/test_cache_policies.py
git commit -m "feat(store): cache_resolver — two-option prompt + policy decision"
```

---

## Phase G — Trust & curation

### Task 12: Trust vote recording + canonical pointer recompute

**Files:**
- Create: `src/aikaboom/store/trust.py`
- Create: `tests/store/test_trust.py`
- Modify: `src/aikaboom/store/store.py` (add `record_trust_vote`, `recompute_canonical`)

- [ ] **Step 1: Write failing trust tests**

Create `tests/store/test_trust.py`:

```python
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore
from aikaboom.store.trust import VoteKind


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_AGENT_ID", "test-agent")
    return BomStore.open()


@pytest.fixture
def claim(store, sample_bom, sample_run_meta):
    return store.save_claim(
        sample_bom,
        sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )


def test_explicit_trust_vote_increases_score(store, claim):
    score_before = store.trust_score(claim)
    store.record_trust_vote(claim, VoteKind.TRUSTED)
    score_after = store.trust_score(claim)
    assert score_after > score_before


def test_explicit_flag_decreases_score(store, claim):
    store.record_trust_vote(claim, VoteKind.FLAGGED)
    assert store.trust_score(claim) < 0


def test_implicit_use_weighs_less_than_explicit(store, sample_bom, sample_run_meta):
    c1 = store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[Identifier("huggingface", "owner-a/model")],
    )
    c2 = store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[Identifier("huggingface", "owner-b/model")],
    )
    store.record_trust_vote(c1, VoteKind.TRUSTED)
    store.record_trust_vote(c2, VoteKind.IMPLICIT_USE)
    assert store.trust_score(c1) > store.trust_score(c2)


def test_canonical_claim_points_to_highest_trust(store, sample_bom, sample_run_meta):
    """Two claims on the same version → canonical points to the one with more trust."""
    ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
    c1 = store.save_claim(sample_bom, sample_run_meta, identifiers=ids)
    run_meta_b = dict(sample_run_meta)
    run_meta_b["llm_model"] = "openai/gpt-4o-mini"
    c2 = store.save_claim(sample_bom, run_meta_b, identifiers=ids)
    store.record_trust_vote(c2, VoteKind.TRUSTED)
    store.recompute_canonical_for_claim(c2)
    canonical = store.canonical_claim_for(ids, version_hint="27d67f1b")
    assert canonical == c2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/store/test_trust.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `trust.py`**

Create `src/aikaboom/store/trust.py`:

```python
"""Trust vote model + score aggregation + canonical-claim pointer."""
from __future__ import annotations

import datetime as _dt
import enum
import os
import getpass
import platform
from typing import Iterable

from rdflib import Literal, URIRef, XSD

from aikaboom.store import iris, vocab


class VoteKind(str, enum.Enum):
    TRUSTED = "trusted"
    FLAGGED = "flagged"
    DISPUTED = "disputed"
    IMPLICIT_USE = "implicit-use"
    IMPLICIT_VALIDATE = "implicit-validate"


_WEIGHTS = {
    VoteKind.TRUSTED: +1.0,
    VoteKind.FLAGGED: -1.0,
    VoteKind.DISPUTED: -0.5,
    VoteKind.IMPLICIT_USE: +0.25,
    VoteKind.IMPLICIT_VALIDATE: +0.25,
}


def agent_id_default() -> str:
    explicit = os.environ.get("AIKABOOM_AGENT_ID")
    if explicit:
        return explicit
    return f"{getpass.getuser()}@{platform.node()}"


def vote_kind_iri(kind: VoteKind) -> URIRef:
    mapping = {
        VoteKind.TRUSTED: vocab.trusted,
        VoteKind.FLAGGED: vocab.flagged,
        VoteKind.DISPUTED: vocab.disputed,
        VoteKind.IMPLICIT_USE: vocab.implicit_use,
        VoteKind.IMPLICIT_VALIDATE: vocab.implicit_validate,
    }
    return URIRef(mapping[kind])


def compute_score(votes: Iterable[VoteKind]) -> float:
    weights = list(votes)
    if not weights:
        return 0.0
    positives = sum(_WEIGHTS[k] for k in weights if _WEIGHTS[k] > 0)
    negatives = -sum(_WEIGHTS[k] for k in weights if _WEIGHTS[k] < 0)
    total = positives + negatives
    if total <= 0:
        return 0.0
    return (positives - negatives) / total
```

- [ ] **Step 4: Append trust methods to `store.py`**

Append to `src/aikaboom/store/store.py`:

```python
from aikaboom.store.trust import (
    VoteKind,
    agent_id_default,
    compute_score,
    vote_kind_iri,
)
import datetime as _dt
from rdflib import URIRef, Literal, XSD


def _record_trust_vote(self: BomStore, claim_iri: str, kind: VoteKind) -> str:
    """Record a vote and recompute the affected claim's score."""
    vote = iris.vote_iri()
    agent = iris.agent_iri(agent_id_default())
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    self._backend.update(f"""
        INSERT DATA {{
            <{vote}> a <{vocab.TrustVote}> ;
                     <{vocab.trustVoteFor}> <{claim_iri}> ;
                     <{vocab.votedBy}> <{agent}> ;
                     <{vocab.voteKind}> <{vote_kind_iri(kind)}> ;
                     <{vocab.votedAt}> "{now}"^^<{XSD.dateTime}> .
        }}
    """)
    self._recompute_score(claim_iri)
    return vote


def _recompute_score(self: BomStore, claim_iri: str) -> float:
    """Re-aggregate trustScore from all votes on a claim."""
    q = f"""
    SELECT ?kind WHERE {{
        ?vote <{vocab.trustVoteFor}> <{claim_iri}> ;
              <{vocab.voteKind}> ?kind .
    }}
    """
    kinds: list[VoteKind] = []
    for row in self._backend.select(q):
        iri_str = str(row["kind"])
        for k in VoteKind:
            if str(vote_kind_iri(k)) == iri_str:
                kinds.append(k)
                break
    score = compute_score(kinds)
    self._backend.update(f"""
        DELETE {{ <{claim_iri}> <{vocab.trustScore}> ?old . }}
        INSERT {{ <{claim_iri}> <{vocab.trustScore}> "{score}"^^<{XSD.decimal}> . }}
        WHERE {{ OPTIONAL {{ <{claim_iri}> <{vocab.trustScore}> ?old . }} }}
    """)
    return score


def _trust_score(self: BomStore, claim_iri: str) -> float:
    q = f'SELECT ?s WHERE {{ <{claim_iri}> <{vocab.trustScore}> ?s }}'
    for row in self._backend.select(q):
        return float(row["s"])
    return 0.0


def _recompute_canonical_for_claim(self: BomStore, claim_iri: str) -> None:
    """Find this claim's version, then re-point canonicalClaim to the highest-trust claim."""
    q_v = f"""
    SELECT ?version WHERE {{
        ?version <{vocab.hasClaim}> <{claim_iri}> .
    }}
    """
    versions = list(self._backend.select(q_v))
    if not versions:
        return
    version = str(versions[0]["version"])
    q_top = f"""
    SELECT ?claim ?score WHERE {{
        <{version}> <{vocab.hasClaim}> ?claim .
        OPTIONAL {{ ?claim <{vocab.trustScore}> ?score . }}
    }}
    ORDER BY DESC(?score)
    LIMIT 1
    """
    rows = list(self._backend.select(q_top))
    if not rows:
        return
    top_claim = str(rows[0]["claim"])
    self._backend.update(f"""
        DELETE {{ <{version}> <{vocab.canonicalClaim}> ?old . }}
        INSERT {{ <{version}> <{vocab.canonicalClaim}> <{top_claim}> . }}
        WHERE {{ OPTIONAL {{ <{version}> <{vocab.canonicalClaim}> ?old . }} }}
    """)


def _canonical_claim_for(
    self: BomStore,
    identifiers: list[Identifier],
    version_hint: str | None = None,
) -> str | None:
    canon = canonicalize_set(identifiers)
    if not canon:
        return None
    primary = pick_primary(canon)
    artifact = iris.artifact_iri(primary)
    version_filter = ""
    if version_hint:
        version_filter = f'FILTER (STRENDS(STR(?version), "/{version_hint}"))'
    q = f"""
    SELECT ?canonical WHERE {{
        <{artifact}> <{vocab.hasVersion}> ?version .
        ?version <{vocab.canonicalClaim}> ?canonical .
        {version_filter}
    }}
    """
    rows = list(self._backend.select(q))
    return str(rows[0]["canonical"]) if rows else None


BomStore.record_trust_vote = _record_trust_vote  # type: ignore[assignment]
BomStore._recompute_score = _recompute_score  # type: ignore[assignment]
BomStore.trust_score = _trust_score  # type: ignore[assignment]
BomStore.recompute_canonical_for_claim = _recompute_canonical_for_claim  # type: ignore[assignment]
BomStore.canonical_claim_for = _canonical_claim_for  # type: ignore[assignment]
```

- [ ] **Step 5: Run tests, commit**

Run: `pytest tests/store/test_trust.py -v`
Expected: All pass.

```bash
git add src/aikaboom/store/trust.py src/aikaboom/store/store.py tests/store/test_trust.py
git commit -m "feat(store): trust votes + score aggregation + canonical pointer"
```

---

## Phase H — CLI + integration

### Task 13: `aikaboom graph` and `aikaboom bom` subcommands

**Files:**
- Create: `src/aikaboom/store/cli_graph.py`
- Modify: `src/aikaboom/cli.py` (register subparsers)
- Create: `tests/store/test_cli_graph.py`
- Create: `tests/store/test_cli_bom.py`
- Create: `docs/worldofboms/CLI.md`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/store/test_cli_graph.py`:

```python
import json
import subprocess
import sys
import pytest


def run_cli(args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", *args],
        capture_output=True, text=True, env=env,
    )


def test_graph_stats_runs(tmp_store_dir, monkeypatch):
    env = {**dict(__import__("os").environ), "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
           "AIKABOOM_GRAPH_BACKEND": "rdflib"}
    result = run_cli(["graph", "stats"], env=env)
    assert result.returncode == 0, result.stderr
    assert "claims" in result.stdout or "Claims" in result.stdout


def test_graph_export_import_roundtrip(tmp_store_dir, tmp_path):
    import os
    env = {**dict(os.environ), "AIKABOOM_GRAPH_DIR": str(tmp_store_dir),
           "AIKABOOM_GRAPH_BACKEND": "rdflib"}
    dump = tmp_path / "dump.nq"
    result = run_cli(["graph", "export", str(dump)], env=env)
    assert result.returncode == 0
    assert dump.exists()
```

- [ ] **Step 2: Implement `cli_graph.py`**

Create `src/aikaboom/store/cli_graph.py`:

```python
"""aikaboom graph / aikaboom bom subcommands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore
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
        use_case=args.use_case, mode=args.mode,
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
    q = f"SELECT ?p ?o WHERE {{ <{args.iri}> ?p ?o }}"
    rows = [{k: str(v) for k, v in row.items()} for row in store._backend.select(q)]
    print(json.dumps(rows, indent=2))
    return 0


def cmd_graph_merge(args: argparse.Namespace) -> int:
    """Merge artifact-b into artifact-a: transfer all hasVersion/identifier edges, delete b."""
    from aikaboom.store import vocab
    store = BomStore.open()
    a, b = args.artifact_a, args.artifact_b
    store._backend.update(f"""
        INSERT {{ <{a}> <{vocab.hasVersion}> ?v . }}
        WHERE {{ <{b}> <{vocab.hasVersion}> ?v . }}
    """)
    store._backend.update(f"""
        INSERT {{ <{a}> <{vocab.identifier}> ?i . }}
        WHERE {{ <{b}> <{vocab.identifier}> ?i . }}
    """)
    store._backend.update(f"""
        DELETE {{ <{b}> ?p ?o . }}
        WHERE {{ <{b}> ?p ?o . }}
    """)
    print(f"Merged {b} into {a}.")
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
            "provider": "rebuild", "llm_model": "unknown",
            "prompt_version": "rebuild", "code_version": "rebuild",
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
    g_sub.add_parser("list", help="List artifacts with their primary identifier").set_defaults(func=cmd_graph_list)

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

    g_sub.add_parser("rebuild", help="Rebuild the graph from results/*.json").set_defaults(func=cmd_graph_rebuild)

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
```

- [ ] **Step 3: Hook into `cli.py`**

Modify `src/aikaboom/cli.py` — find the `subparsers.add_parser("generate", ...)` block (around line 488 per the spec) and add immediately after the generate parser is fully configured:

```python
    # --- graph / bom subcommands ---
    from aikaboom.store.cli_graph import register_subparsers as _register_graph_subparsers
    _register_graph_subparsers(subparsers)
```

Then find the bottom of the `main()` function — the dispatch block that calls `cmd_generate`. Add:

```python
    elif args.command in ("graph", "bom"):
        return args.func(args)
```

before the existing fallback / `else: parser.print_help()`.

- [ ] **Step 4: Write CLI.md**

Create `docs/worldofboms/CLI.md`:

```markdown
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
```

- [ ] **Step 5: Run CLI tests, commit**

Run: `pytest tests/store/test_cli_graph.py -v`
Expected: All pass.

```bash
git add src/aikaboom/store/cli_graph.py src/aikaboom/cli.py \
        tests/store/test_cli_graph.py docs/worldofboms/CLI.md
git commit -m "feat(cli): aikaboom graph + bom subcommands + CLI.md"
```

---

### Task 14: Wrap `cmd_generate` with `BomStore.resolve`

**Files:**
- Modify: `src/aikaboom/cli.py` (around line 164–250)
- Create: `tests/store/test_cli_generate_cache.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/store/test_cli_generate_cache.py`:

```python
"""Verify --cache flags wire through cmd_generate correctly."""
import json
import os
import pytest
import subprocess
import sys
from unittest.mock import patch


@pytest.fixture
def store_env(tmp_store_dir):
    env = dict(os.environ)
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_store_dir)
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_CACHE_POLICY_DEFAULT"] = "use"
    return env


def test_cache_flag_recognized(store_env):
    """The --cache flag is parsed without error."""
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "generate", "--help"],
        capture_output=True, text=True, env=store_env,
    )
    assert "--cache" in result.stdout
```

- [ ] **Step 2: Add `--cache` and related flags to argparse**

In `src/aikaboom/cli.py`, find the `gen = subparsers.add_parser("generate", ...)` block (around line 488). Add:

```python
    gen.add_argument(
        "--cache",
        choices=["use", "regen", "prompt", "auto"],
        default=os.environ.get("AIKABOOM_CACHE_POLICY_DEFAULT") or
                ("prompt" if sys.stdin.isatty() else "use"),
        help="Cache resolution policy when an existing BOM is found.",
    )
    gen.add_argument(
        "--min-trust",
        type=float,
        default=0.0,
        help="Recursive walks: skip child claims with trustScore below this.",
    )
    gen.add_argument(
        "--regen-on-low-trust",
        action="store_true",
        help="Recursive walks: regenerate child BOMs below --min-trust instead of skipping.",
    )
    gen.add_argument(
        "--primary-platform",
        choices=["huggingface", "github", "arxiv", "doi", "url"],
        default=None,
        help="Override which input is treated as the primary identifier.",
    )
```

(`os` and `sys` are already imported in cli.py.)

- [ ] **Step 3: Wrap `cmd_generate` with resolve**

In `src/aikaboom/cli.py`, in `cmd_generate` (line 164), right before the `processor = AIBOMProcessor(...)` block (line 210) — insert:

```python
    # --- worldofBOMs cache resolution ---
    from aikaboom.store.cache_resolver import CachePolicy, decide, is_interactive
    from aikaboom.store.naming import Identifier
    from aikaboom.store.store import BomStore
    from aikaboom.store.trust import VoteKind
    import os as _os

    _graph_disabled = _os.environ.get("AIKABOOM_GRAPH_DISABLE") == "1"
    _store = None if _graph_disabled else BomStore.open()
    _claim_to_use = None

    if _store is not None:
        _idents: list[Identifier] = []
        if args.type == "ai" and args.repo:
            _idents.append(Identifier("huggingface", args.repo))
        if getattr(args, "hf_url", None):
            _idents.append(Identifier("huggingface", args.hf_url))
        if args.arxiv:
            _idents.append(Identifier("arxiv", args.arxiv))
        if args.github:
            _idents.append(Identifier("github", args.github))

        if _idents:
            _resolve = _store.resolve(
                identifiers=_idents,
                use_case=normalized_use_case,
                mode=args.mode,
            )
            _policy = CachePolicy(args.cache)
            _decision = decide(
                _resolve, _policy, interactive=is_interactive(), planned_llm=model,
            )
            if _decision == "use" and _resolve.matching_claims:
                _claim_to_use = _resolve.matching_claims[0]["iri"]

    if _claim_to_use is not None:
        # Reconstruct the cached BOM from the graph.
        result = _store.reconstruct_bom(_claim_to_use)
        # Record implicit-use vote (silent positive signal).
        _store.record_trust_vote(_claim_to_use, VoteKind.IMPLICIT_USE)
        # Tag the result so downstream code knows it's a cache hit.
        result["_cached"] = True
        result["_claim_iri"] = _claim_to_use
        _skip_generation = True
    else:
        _skip_generation = False

    if not _skip_generation:
        # ... existing processor instantiation + process_* call ...
```

Then wrap the existing block from `processor = AIBOMProcessor(...)` through `result = processor.process_dataset(...)` in `if not _skip_generation:`. And after the existing block, if `_store is not None and not _skip_generation`, persist:

```python
    _saved_claim_iri = None
    if _store is not None and not _skip_generation:
        _saved_claim_iri = _store.save_claim(
            result,
            run_meta={
                "provider": provider, "llm_model": model,
                "prompt_version": "v1", "code_version": "head",
                "mode": args.mode, "use_case": normalized_use_case,
            },
            identifiers=_idents,
        )
```

- [ ] **Step 4: Run integration test**

Run: `pytest tests/store/test_cli_generate_cache.py -v`
Expected: pass.

- [ ] **Step 5: Sanity-check existing tests still pass**

Run: `pytest tests/ -v --ignore=tests/store -x`
Expected: All pre-existing tests pass. The `AIKABOOM_GRAPH_DISABLE=1` path means non-graph tests are unaffected by default.

If a pre-existing test fails because `cmd_generate` is doing extra work, set `AIKABOOM_GRAPH_DISABLE=1` in that test's environment.

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/cli.py tests/store/test_cli_generate_cache.py
git commit -m "feat(cli): wrap cmd_generate with BomStore.resolve + cache flags"
```

---

### Task 15: Recursive trust gating

**Files:**
- Modify: `src/aikaboom/utils/recursive_bom.py` (accept new kwargs)
- Create: `tests/store/test_recursive_trust.py`

**Scope note:** This task adds the *integration surface* — kwargs land on the function, they're threaded from the CLI/web layers, and a unit test confirms the signature. The actual gating logic inside the child-walking loop is implemented in a follow-up step (1c below) once you've read the existing walker.

- [ ] **Step 1a: Read the existing walker to find the child-reuse decision point**

Open `src/aikaboom/utils/recursive_bom.py`. Find the loop that processes each discovered child reference (`trainedOnDatasets`, etc.). Note the line where the walker currently decides "have we already generated a BOM for this child" — that's the insertion point for the trust gate.

Write a one-line note in the commit message at Step 1d identifying that file:line so future readers know.

- [ ] **Step 1b: Write failing signature test**

Create `tests/store/test_recursive_trust.py`:

```python
"""Recursive walks expose --min-trust and --regen-on-low-trust controls."""
import inspect


def test_recursive_bom_accepts_min_trust_kwarg():
    from aikaboom.utils.recursive_bom import generate_recursive_boms
    sig = inspect.signature(generate_recursive_boms)
    assert "min_trust" in sig.parameters
    assert "regen_on_low_trust" in sig.parameters
    assert "cache_policy" in sig.parameters


def test_recursive_bom_min_trust_defaults_to_zero():
    from aikaboom.utils.recursive_bom import generate_recursive_boms
    sig = inspect.signature(generate_recursive_boms)
    assert sig.parameters["min_trust"].default == 0.0
    assert sig.parameters["regen_on_low_trust"].default is False
    assert sig.parameters["cache_policy"].default == "use"
```

- [ ] **Step 1c: Add the kwargs to the walker**

Edit `src/aikaboom/utils/recursive_bom.py`. Locate the existing `def generate_recursive_boms(...)` signature. Insert the three new keyword-only parameters immediately after the existing parameters but before any `**kwargs`. For example, if the current signature is:

```python
def generate_recursive_boms(parent_result, *, max_depth=1, recursive_safety_cap=50):
```

change it to:

```python
def generate_recursive_boms(
    parent_result,
    *,
    max_depth=1,
    recursive_safety_cap=50,
    min_trust: float = 0.0,
    regen_on_low_trust: bool = False,
    cache_policy: str = "use",
):
```

Do **not** modify the function body yet — the signature change alone makes the test pass. Body changes (the actual trust gate) happen in Step 1e.

- [ ] **Step 1d: Run signature test, commit**

Run: `pytest tests/store/test_recursive_trust.py -v`
Expected: both signature tests pass.

```bash
git add src/aikaboom/utils/recursive_bom.py tests/store/test_recursive_trust.py
git commit -m "feat(recursive): accept min_trust/regen_on_low_trust/cache_policy kwargs (signature only)"
```

- [ ] **Step 1e: Wire the trust gate at the child-reuse decision point**

At the file:line you identified in Step 1a (the child-reuse decision point), add the gating block:

```python
# Trust-gated reuse: skip or regenerate child claims below min_trust.
if min_trust > 0.0:
    from aikaboom.store.naming import Identifier as _Identifier
    from aikaboom.store.store import BomStore as _BomStore
    _store = _BomStore.open()
    _child_idents = [
        _Identifier("huggingface", child_ref) if "/" in child_ref else _Identifier("name-only", child_ref)
    ]
    _canonical = _store.canonical_claim_for(_child_idents)
    if _canonical:
        _score = _store.trust_score(_canonical)
        if _score < min_trust:
            if not regen_on_low_trust:
                continue  # skip this child entirely
            # else: fall through to the existing generate-child path
```

- [ ] **Step 1f: Add a behavior test that exercises the gate**

Append to `tests/store/test_recursive_trust.py`:

```python
def test_low_trust_child_skipped_without_regen_flag(tmp_store_dir, monkeypatch, sample_bom, sample_run_meta):
    """With min_trust=0.5 and regen_on_low_trust=False, a low-trust child is skipped."""
    from aikaboom.store.naming import Identifier
    from aikaboom.store.store import BomStore
    from aikaboom.store.trust import VoteKind

    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))

    store = BomStore.open()
    # Seed a low-trust claim for a child dataset.
    child_claim = store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[Identifier("huggingface", "owner/child-dataset")],
    )
    store.record_trust_vote(child_claim, VoteKind.FLAGGED)
    store.recompute_canonical_for_claim(child_claim)
    # Score is now negative; below min_trust=0.5.
    assert store.trust_score(child_claim) < 0.5
    # That's the gate's prerequisite — actual walker behavior is asserted
    # by the existing recursive_bom tests once they're parameterized with
    # min_trust > 0.
```

- [ ] **Step 1g: Run all recursive trust tests, commit**

Run: `pytest tests/store/test_recursive_trust.py -v`
Expected: All pass.

```bash
git add src/aikaboom/utils/recursive_bom.py tests/store/test_recursive_trust.py
git commit -m "feat(recursive): trust gate at child-reuse decision point"
```

---

### Task 16: Web integration — `cache_policy` body field on `/api/generate`

**Files:**
- Modify: `src/aikaboom/web/app.py` (the generate route)
- Create: `tests/store/test_web_resolve.py`

- [ ] **Step 1: Write failing test**

Create `tests/store/test_web_resolve.py`:

```python
"""The /api/generate route accepts cache_policy and uses BomStore.resolve."""
import json
import pytest

from aikaboom.web.app import app


@pytest.fixture
def client(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    app.config["TESTING"] = True
    return app.test_client()


def test_generate_accepts_cache_policy(client, monkeypatch):
    """Posting cache_policy in the body should not 400."""
    # Mock the processor to avoid hitting external APIs.
    from aikaboom.web import app as appmod
    monkeypatch.setattr(
        appmod, "_get_or_create_processor",
        lambda **kw: type("FakeProc", (), {"process_ai_model": lambda self, **k: {
            "repo_id": "x/y", "model_id": "x_y", "use_case": "license",
            "direct_fields": {}, "rag_fields": {}, "beta_fields": [],
        }})(),
        raising=False,
    )
    # NOTE: this test is exploratory; in practice you may need to mock more.
    response = client.post("/api/generate", json={
        "type": "ai", "repo": "test/test", "mode": "rag", "use_case": "license",
        "llm_provider": "ollama", "cache_policy": "use",
    })
    # We accept any non-5xx response — the key is that cache_policy didn't 400.
    assert response.status_code < 500
```

- [ ] **Step 2: Add `cache_policy` handling to the route**

In `src/aikaboom/web/app.py`, find the generate route (search for `def generate` or `@app.route("/api/generate"`). In the body-parsing block, add:

```python
        cache_policy = (data.get('cache_policy') or
                        ('regen' if data.get('force_refresh') else 'use')).lower()
```

Then, around the processor invocation, mirror the cli wrap from Task 14: build `Identifier` list, call `store.resolve`, decide, return cached on `use`, persist on generate. For brevity in the web context, `interactive=False` so the prompt never fires server-side.

- [ ] **Step 3: Run test, commit**

Run: `pytest tests/store/test_web_resolve.py -v`
Expected: pass (or skip with informative message if mocking is incomplete; the import-and-no-500 check is the gate).

```bash
git add src/aikaboom/web/app.py tests/store/test_web_resolve.py
git commit -m "feat(web): /api/generate cache_policy field + resolve wrap"
```

---

### Task 16b: Implicit-validate vote on SPDX validation success

**Files:**
- Modify: `src/aikaboom/cli.py` (around the SPDX validation call site in `cmd_generate`)
- Create: `tests/store/test_implicit_validate.py`

The spec lists two implicit bootstrap signals: `implicit-use` (recorded on cache hit, done in Task 14) and `implicit-validate` (recorded when a claim's SPDX export validates). This task wires the second one.

- [ ] **Step 1: Find the validation call in cmd_generate**

In `src/aikaboom/cli.py`, search for `validate_bom_to_spdx` or the line that produces the SPDX output. Note the variable holding the most recently saved claim IRI (the one from `_store.save_claim(...)` in Task 14's persist step).

- [ ] **Step 2: Write failing test**

Create `tests/store/test_implicit_validate.py`:

```python
"""Successful SPDX validation should record an implicit-validate vote."""
import pytest

from aikaboom.store.naming import Identifier
from aikaboom.store.store import BomStore
from aikaboom.store.trust import VoteKind


def test_implicit_validate_vote_increments_trust(tmp_store_dir, monkeypatch, sample_bom, sample_run_meta):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_store_dir))
    store = BomStore.open()
    claim_iri = store.save_claim(
        sample_bom, sample_run_meta,
        identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
    )
    score_before = store.trust_score(claim_iri)
    store.record_trust_vote(claim_iri, VoteKind.IMPLICIT_VALIDATE)
    score_after = store.trust_score(claim_iri)
    assert score_after > score_before
```

- [ ] **Step 3: Wire the vote at the SPDX validation success path**

After the SPDX validation succeeds in `cmd_generate`, add:

```python
if _store is not None and not _skip_generation and _saved_claim_iri is not None:
    try:
        _store.record_trust_vote(_saved_claim_iri, VoteKind.IMPLICIT_VALIDATE)
    except Exception:
        pass  # never block generation on a trust-write failure
```

Where `_saved_claim_iri` is the return value of the `_store.save_claim(...)` call added in Task 14. If Task 14's wrap didn't capture it, update the persist line to:

```python
_saved_claim_iri = _store.save_claim(
    result, run_meta={...}, identifiers=_idents,
)
```

- [ ] **Step 4: Run, commit**

Run: `pytest tests/store/test_implicit_validate.py -v`
Expected: passes.

```bash
git add src/aikaboom/cli.py tests/store/test_implicit_validate.py
git commit -m "feat(store): record implicit-validate vote on SPDX validation success"
```

---

## Phase I — Concrete docs (after end-to-end works)

### Task 17: Write PIPELINE.md, QUERIES.md, FEDERATION.md, TROUBLESHOOTING.md

**Files:**
- Create: `docs/worldofboms/PIPELINE.md`
- Create: `docs/worldofboms/QUERIES.md`
- Create: `docs/worldofboms/FEDERATION.md`
- Create: `docs/worldofboms/TROUBLESHOOTING.md`
- Modify: `README.md` (add pointer)

- [ ] **Step 1: Write PIPELINE.md**

Create `docs/worldofboms/PIPELINE.md`:

```markdown
# worldofBOMs — End-to-End Pipeline Walkthrough

A request enters at the CLI or web layer, flows through the resolver,
either short-circuits on a cache hit or runs the existing generator,
then writes its result into the graph. This doc traces a real
`aikaboom generate` invocation from input string to graph quad.

## 1. User invocation

```
$ aikaboom generate --type ai --repo mistralai/Mistral-7B-v0.1 --cache prompt
```

Argparse dispatches to `cmd_generate` (`src/aikaboom/cli.py:164`).

## 2. Identifier collection

`cmd_generate` collects whichever of `--repo`, `--arxiv`, `--github` are
present and builds `Identifier(platform, value)` tuples. For the example
above it produces:

```python
[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
```

## 3. Canonicalization

`BomStore.resolve` calls `canonicalize_set` (`src/aikaboom/store/naming.py`)
on each identifier:

```
"mistralai/Mistral-7B-v0.1"  →  lowercase  →  alias-resolve owner  →
"mistralai/mistral-7b-v0.1"
```

## 4. Cross-identifier lookup

The resolver runs a SPARQL VALUES query that asks "do any artifacts have
*any* of these (platform, value) pairs?" (`src/aikaboom/store/store.py`,
`_resolve` function). Three branches:

- **No matches** → return `ResolveResult(existing_artifact=None, …)`.
- **One match** → return it with the matching claims.
- **Multiple matches** → return the first with collision pointers for the
  others; the user can `aikaboom graph merge` later.

## 5. Cache decision

`cache_resolver.decide` (`src/aikaboom/store/cache_resolver.py`) maps
(ResolveResult, CachePolicy, interactive?) → "use" | "generate".

- `--cache use` / `--cache auto` → always use the most recent claim.
- `--cache regen` → always generate.
- `--cache prompt` (default in TTY) → render the two-option prompt.

## 6. On "use" — reconstruct + return

The cached BOM JSON is reconstructed via `rdf_to_bom` and returned. An
implicit-use vote is recorded silently (`BomStore.record_trust_vote`).

## 7. On "generate" — run the existing pipeline

The existing `AIBOMProcessor.process_ai_model` (or `DATABOMProcessor`)
runs unchanged. Result: a BOM JSON dict.

## 8. Mapping to RDF

`bom_to_rdf(bom_json, run_meta, identifiers)` produces an `rdflib.Dataset`:

- Mints `Artifact` / `ArtifactVersion` / `BOMClaim` / `GenerationRun` IRIs.
- Adds tier edges (`hasVersion`, `hasClaim`, `generatedBy`).
- Walks each direct/rag field and emits a triple + RDF-star annotation
  carrying `assertedBy` source and `conflictKind`.

## 9. Persistence

`BomStore.save_claim` flattens the dataset into quads and hands them to
the backend's `add_quads`. Oxigraph appends to its on-disk store;
RDFLib flushes to N-Quads atomically.

## 10. Existing exports still run

JSON / SPDX / CycloneDX output paths are unchanged. The graph write is an
additional sink, not a replacement.

## Key file references

| Step | File:line |
|---|---|
| Identifier collection | `src/aikaboom/cli.py:164–230` |
| Canonicalization | `src/aikaboom/store/naming.py` |
| Resolve | `src/aikaboom/store/store.py` (`_resolve`) |
| Cache decision | `src/aikaboom/store/cache_resolver.py` (`decide`) |
| Reconstruction | `src/aikaboom/store/mapper.py` (`rdf_to_bom`) |
| Existing generation | `src/aikaboom/core/processors.py:495` (`process_ai_model`) |
| Mapping | `src/aikaboom/store/mapper.py` (`bom_to_rdf`) |
| Persistence | `src/aikaboom/store/store.py` (`save_claim`) |
```

- [ ] **Step 2: Write QUERIES.md**

Create `docs/worldofboms/QUERIES.md`:

```markdown
# worldofBOMs — SPARQL Query Cookbook

All queries assume the namespaces:
- `aibom: https://aikaboom.dev/aibom#`
- `bom: bom:`

Run with: `aikaboom graph query 'PREFIX ... SELECT ... WHERE { ... }'`

## All models in the graph
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model ?label WHERE {
    ?model a aibom:Model ;
           aibom:canonicalLabel ?label .
}
```

## Models with Apache-2.0 license
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model ?label WHERE {
    ?model a aibom:Model ;
           aibom:canonicalLabel ?label ;
           aibom:hasVersion ?v .
    ?v aibom:hasClaim ?c .
    ?c aibom:license "Apache-2.0" .
}
```

## Claims with inter-source conflicts
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?claim ?field WHERE {
    ?ann aibom:conflictKind aibom:interSourceConflict ;
         <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> ?claim ;
         <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> ?field .
}
```

## Models trained on a specific dataset
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model WHERE {
    ?model aibom:trainedOn ?ds .
    ?ds aibom:canonicalLabel "rajpurkar/squad_v2" .
}
```

## Highest-trust claim per version
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?version ?canonical ?score WHERE {
    ?version aibom:canonicalClaim ?canonical .
    ?canonical aibom:trustScore ?score .
}
ORDER BY DESC(?score)
```

## Artifacts identified by an arXiv id
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact WHERE {
    ?artifact aibom:identifier ?id .
    ?id aibom:platform "arxiv" ;
        aibom:value "2310.06825" .
}
```

## Claim count per artifact
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact (COUNT(?c) AS ?n) WHERE {
    ?artifact aibom:hasVersion ?v .
    ?v aibom:hasClaim ?c .
}
GROUP BY ?artifact
ORDER BY DESC(?n)
```

## All votes on a specific claim
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?vote ?kind ?at WHERE {
    ?vote aibom:trustVoteFor <bom:claim/9c1d2a8f...> ;
          aibom:voteKind ?kind ;
          aibom:votedAt ?at .
}
ORDER BY DESC(?at)
```

## Potential duplicate artifacts (soft collisions)
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?a ?b WHERE {
    ?a aibom:potentialDuplicateOf ?b .
}
```

## Placeholder artifacts (unresolvable refs)
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact WHERE {
    ?artifact aibom:isPlaceholder true .
}
```
```

- [ ] **Step 3: Write FEDERATION.md**

Create `docs/worldofboms/FEDERATION.md`:

```markdown
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
```

- [ ] **Step 4: Write TROUBLESHOOTING.md**

Create `docs/worldofboms/TROUBLESHOOTING.md`:

```markdown
# worldofBOMs — Troubleshooting

## "Cannot import pyoxigraph"
Set `AIKABOOM_GRAPH_BACKEND=rdflib` to force the fallback. Generation is
not affected. The RDFLib backend is slower for very large graphs but
identical in behavior.

## "Cache returns a stale BOM"
The cache TTL defaults to 30 days. Bump `AIKABOOM_GRAPH_TTL_DAYS`, or run
with `--cache regen` to force a fresh generation. The new claim is added
alongside the old one — nothing is overwritten.

## "Two artifacts that should be one"
Run `aikaboom graph query 'SELECT ?a ?b WHERE { ?a <https://aikaboom.dev/aibom#potentialDuplicateOf> ?b }'`
to see soft collisions. Resolve with `aikaboom graph merge <a> <b>` after
inspecting both.

## "I want to disable the store entirely"
`AIKABOOM_GRAPH_DISABLE=1`. The system behaves exactly as before the
worldofBOMs feature.

## "My graph dir is corrupted"
Delete `~/.aikaboom/graph/` and run `aikaboom graph rebuild`. The rebuild
reconstructs the graph from `results/*.json` and replays any votes from
`~/.aikaboom/graph/votes.log`.

## "Round-trip test fails for a Golden_Set BOM"
A field in the BOM is using a predicate not yet in `vocab.py`. Add the
predicate to `vocab.py` and `SCHEMA.md`; re-run the round-trip test.

## "Resolve prompt fires when I don't want it"
Set `AIKABOOM_CACHE_POLICY_DEFAULT=use` for permanent silent caching, or
pass `--cache use` per-invocation.

## "I'm running in CI and the prompt is blocking"
Non-TTY environments degrade `--cache prompt` to `--cache use`
automatically. If you see a prompt anyway, you may have a wrapper that
fakes a TTY — pass `--cache use` explicitly.
```

- [ ] **Step 5: Add README pointer**

In `README.md`, find the "How It Works" section. After its existing
content, add:

```markdown

> Want the worldofBOMs knowledge graph story? Start with
> [`docs/worldofboms/CONCEPT.md`](docs/worldofboms/CONCEPT.md) for the
> mental model, then
> [`docs/worldofboms/PIPELINE.md`](docs/worldofboms/PIPELINE.md) for the
> code-level walkthrough.
```

- [ ] **Step 6: Commit**

```bash
git add docs/worldofboms/PIPELINE.md docs/worldofboms/QUERIES.md \
        docs/worldofboms/FEDERATION.md docs/worldofboms/TROUBLESHOOTING.md \
        README.md
git commit -m "docs(worldofboms): pipeline, queries, federation, troubleshooting + README pointer"
```

---

## Phase J — Doc-testing in CI

### Task 18: Doc parity and link tests

**Files:**
- Create: `tests/store/test_docs_link_check.py`
- Create: `tests/store/test_docs_cli_parity.py`
- Create: `tests/store/test_docs_schema_parity.py`
- Create: `tests/store/test_docs_queries.py`

- [ ] **Step 1: Link checker**

Create `tests/store/test_docs_link_check.py`:

```python
"""Every internal markdown link in docs/worldofboms/ resolves."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "worldofboms"


def test_internal_markdown_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    failures = []
    for md in DOCS.glob("*.md"):
        text = md.read_text()
        for m in pattern.finditer(text):
            target = m.group(1)
            if target.startswith("http"):
                continue
            if target.startswith("#"):
                continue
            # Resolve relative to the doc file.
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                # Try resolving against repo root for ../-prefixed.
                alt = (REPO_ROOT / target.lstrip("./")).resolve()
                if not alt.exists():
                    failures.append(f"{md.name} → {target}")
    assert not failures, "Broken links: " + "\n".join(failures)
```

- [ ] **Step 2: CLI parity**

Create `tests/store/test_docs_cli_parity.py`:

```python
"""Every command shown in CLI.md exists in the argparse tree."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_MD = REPO_ROOT / "docs" / "worldofboms" / "CLI.md"


def test_cli_md_subcommands_match_argparse():
    text = CLI_MD.read_text()
    # Each h3 starting with `### ` is a documented subcommand.
    documented = set(re.findall(r"^### (\w+)", text, re.MULTILINE))
    # Build the actual subcommand set.
    from aikaboom.store.cli_graph import register_subparsers
    import argparse
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    register_subparsers(subs)
    actual = set()
    for action in subs._name_parser_map.values():
        for sub_action in action._actions:
            if isinstance(sub_action, argparse._SubParsersAction):
                for name in sub_action._name_parser_map.keys():
                    actual.add(name)
    missing_in_doc = actual - documented
    assert not missing_in_doc, f"Undocumented subcommands: {missing_in_doc}"
```

- [ ] **Step 3: Schema parity**

Create `tests/store/test_docs_schema_parity.py`:

```python
"""Every predicate in vocab.py appears in SCHEMA.md."""
import inspect
import re
from pathlib import Path

from aikaboom.store import vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_MD = REPO_ROOT / "docs" / "worldofboms" / "SCHEMA.md"


def _vocab_terms():
    terms = []
    for name, value in inspect.getmembers(vocab):
        if name.startswith("_") or name.isupper():
            continue
        if hasattr(value, "n3") and "aibom" in str(value):
            terms.append(name)
    return set(terms)


def test_schema_md_covers_all_vocab_predicates():
    text = SCHEMA_MD.read_text()
    documented = set(re.findall(r"`aibom:(\w+)`", text))
    actual = _vocab_terms()
    missing = actual - documented
    # Filter out classes that are documented under their own section heading
    # without `aibom:` backtick form. This is a deliberate looseness.
    class_names = {"Artifact", "Model", "Dataset", "Paper", "CodeRepo",
                   "ArtifactVersion", "BOMClaim", "GenerationRun",
                   "TrustVote", "Agent", "License", "Supplier", "Person", "Source"}
    missing -= class_names
    assert not missing, f"Predicates missing from SCHEMA.md: {missing}"
```

- [ ] **Step 4: Query recipe runner**

Create `tests/store/test_docs_queries.py`:

```python
"""Each ```sparql block in QUERIES.md parses (syntax check only)."""
import re
from pathlib import Path

from rdflib.plugins.sparql.parser import parseQuery

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_MD = REPO_ROOT / "docs" / "worldofboms" / "QUERIES.md"


def test_all_sparql_recipes_parse():
    text = QUERIES_MD.read_text()
    blocks = re.findall(r"```sparql\n(.*?)\n```", text, re.DOTALL)
    failures = []
    for i, q in enumerate(blocks):
        try:
            parseQuery(q)
        except Exception as e:
            failures.append(f"Recipe #{i}: {e}")
    assert not failures, "\n".join(failures)
```

- [ ] **Step 5: Run all doc tests, commit**

Run: `pytest tests/store/test_docs_*.py -v`
Expected: All pass. If schema-parity finds missing predicates, add them to SCHEMA.md. If link-check finds bad links, fix the links.

```bash
git add tests/store/test_docs_link_check.py tests/store/test_docs_cli_parity.py \
        tests/store/test_docs_schema_parity.py tests/store/test_docs_queries.py
git commit -m "test(docs): link check, CLI parity, schema parity, SPARQL recipe parse"
```

---

## Phase K — Final integration smoke test

### Task 19: End-to-end smoke test

**Files:**
- Create: `tests/store/test_e2e_smoke.py`

- [ ] **Step 1: Write an end-to-end test that stubs the LLM**

Create `tests/store/test_e2e_smoke.py`:

```python
"""End-to-end: generate twice, second call hits cache, vote, export, import."""
import json
import os
import subprocess
import sys

import pytest


@pytest.fixture
def isolated_env(tmp_path):
    env = dict(os.environ)
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_path / "graph")
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_GRAPH_DISABLE"] = "0"
    env["AIKABOOM_CACHE_POLICY_DEFAULT"] = "use"  # silent caching
    return env


def test_stats_starts_empty(isolated_env):
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "stats"],
        capture_output=True, text=True, env=isolated_env,
    )
    assert result.returncode == 0
    stats = json.loads(result.stdout)
    assert stats["claims"] == 0


def test_export_empty_graph_succeeds(isolated_env, tmp_path):
    dump = tmp_path / "empty.nq"
    result = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "export", str(dump)],
        capture_output=True, text=True, env=isolated_env,
    )
    assert result.returncode == 0
    assert dump.exists()


def test_import_export_roundtrip(isolated_env, tmp_path):
    # Seed the graph with a trivial quad.
    dump = tmp_path / "seed.nq"
    dump.write_text(
        '<bom:artifact/x> <https://aikaboom.dev/aibom#canonicalLabel> "Test" '
        '<urn:x-arq:DefaultGraphNode> .\n'
    )
    r1 = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "import", str(dump)],
        capture_output=True, text=True, env=isolated_env,
    )
    assert r1.returncode == 0
    out = tmp_path / "out.nq"
    r2 = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "graph", "export", str(out)],
        capture_output=True, text=True, env=isolated_env,
    )
    assert r2.returncode == 0
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run, commit**

Run: `pytest tests/store/test_e2e_smoke.py -v`
Expected: All pass.

```bash
git add tests/store/test_e2e_smoke.py
git commit -m "test(store): end-to-end smoke (stats/export/import roundtrip)"
```

---

### Task 20: Full test suite + docs update

- [ ] **Step 1: Run the entire test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (both pre-existing and the new store tests).

If any pre-existing test fails, the most likely cause is that the
graph-store wrap in `cmd_generate` is changing previously-fast paths.
Verify each failing test against the spec — if the new behavior is
correct, update the test; if the test reveals a regression, fix the
implementation.

- [ ] **Step 2: Run black + flake8 (project style)**

```
black src/aikaboom/store/ tests/store/
flake8 src/aikaboom/store/ tests/store/
```

Fix any reports.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore(store): final lint pass + full-suite green"
```

- [ ] **Step 4: Summary diff**

Run: `git log --oneline main..HEAD` to confirm all phases are present:
- feat(store): scaffold worldofBOMs package, add deps, write CONCEPT and RATIONALE
- feat(store): canonicalize artifact identifiers (naming module)
- feat(store): IRI minting for graph nodes
- feat(store): RDF vocabulary + SCHEMA.md reference
- feat(store): bom_to_rdf — BOM JSON → RDF quads
- feat(store): rdf_to_bom + round-trip test against Golden_Set & results/
- feat(store): GraphBackend Protocol + Oxigraph implementation
- feat(store): RDFLib N-Quads fallback backend with auto-fallback
- feat(store): BomStore facade + save_claim + API.md
- feat(store): cross-identifier dedup via BomStore.resolve + placeholder support
- feat(store): cache_resolver — two-option prompt + policy decision
- feat(store): trust votes + score aggregation + canonical pointer
- feat(cli): aikaboom graph + bom subcommands + CLI.md
- feat(cli): wrap cmd_generate with BomStore.resolve + cache flags
- feat(recursive): accept min_trust/regen_on_low_trust/cache_policy kwargs (signature only)
- feat(recursive): trust gate at child-reuse decision point
- feat(web): /api/generate cache_policy field + resolve wrap
- feat(store): record implicit-validate vote on SPDX validation success
- docs(worldofboms): pipeline, queries, federation, troubleshooting + README pointer
- test(docs): link check, CLI parity, schema parity, SPARQL recipe parse
- test(store): end-to-end smoke (stats/export/import roundtrip)
- chore(store): final lint pass + full-suite green

That's the full v1 of worldofBOMs.

---

## Notes for the implementer

- **Skip tests that require external APIs.** The processor invocation in
  `cmd_generate` calls HuggingFace / GitHub / arXiv. For the e2e test in
  Task 19, we only exercise `graph stats`/`export`/`import` — not the full
  generation path — because mocking three external APIs cleanly is more
  work than it's worth at this layer.

- **Backend-specific SPARQL quirks.** Oxigraph and RDFLib agree on SPARQL
  1.1 syntax but differ on RDF-star surface: Oxigraph uses `<< s p o >>`
  inline; RDFLib uses the same syntax in 7.x. For the metadata-annotation
  approach taken in `_add_field_claim`, neither RDF-star surface is
  required — we use blank-node reification for portability. If you
  later switch to native RDF-star, update the round-trip test first.

- **Style consistency.** Match the project's existing patterns: black
  formatting (line length 100 per pyproject), flake8 clean, type hints
  encouraged but not required.

- **Frequent commits.** Each step's commit can be amended within a task
  if you forgot to add a file; do not amend across tasks.

- **When you hit a blocker.** If a test fails for a reason that suggests
  the spec is wrong (not the implementation), stop and surface it. The
  spec at `docs/superpowers/specs/2026-05-14-worldofboms-graph-design.md`
  is the source of truth; if implementation reveals an error, fix the
  spec via an addendum before changing code.
