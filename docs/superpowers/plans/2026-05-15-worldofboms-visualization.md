# worldofBOMs Knowledge-Graph Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the worldofBOMs RDF graph genuinely connected (persist `trainedOn`/`testedOn`/`dependsOn` edges between artifacts at every BOM save) and add a browser `worldofBOMs` tab with global + ego-centric views, lineage queries, and SPDX export.

**Architecture:** Three layers. **Part A** (write side) adds two vocab predicates and an `store/edges.py` module that resolves a BOM's relationship-field values into artifact-to-artifact edges, called from `BomStore.save_claim`. **Part B** (read side) adds `store/graph_view.py` — pure query functions over a `BomStore` — exposed through six `/worldofboms/*` Flask routes. **Part C** (frontend) adds a `worldofBOMs` tab to `index.html` reusing the existing Cytoscape + side-panel machinery from the recursive tab.

**Tech Stack:** Python 3.12, rdflib / Oxigraph (`BomStore`), Flask, Cytoscape.js + cytoscape-dagre, pytest.

**Spec:** `docs/superpowers/specs/2026-05-15-worldofboms-visualization-design.md`

**Branch:** `worldofboms-graph` (worktree `.claude/worktrees/worldofboms-spec`). All commits land here, on top of draft PR #48.

**Conventions for every task:**
- Tests for store logic use the existing `tests/store/` fixtures: `store` (rdflib backend), `sample_bom`, `sample_run_meta`, `tmp_store_dir` (see `tests/store/conftest.py`).
- Run a single test with `pytest tests/store/test_xxx.py::test_name -v`.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- CLAUDE.md asks for `gitnexus_impact` before editing a symbol and `gitnexus_detect_changes` before committing — run these where the tools are available; they do not block progress if unavailable.

---

## Part A — Graph Connectivity (write side)

### Task A1: Add `testedOn` and `dependsOn` to the vocab

**Files:**
- Modify: `src/aikaboom/store/vocab.py`
- Modify: `docs/worldofboms/SCHEMA.md`
- Test: `tests/store/test_vocab.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_vocab.py`:

```python
def test_testedon_and_dependson_predicates_exist():
    from aikaboom.store import vocab
    assert str(vocab.testedOn) == "https://aikaboom.dev/aibom#testedOn"
    assert str(vocab.dependsOn) == "https://aikaboom.dev/aibom#dependsOn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_vocab.py::test_testedon_and_dependson_predicates_exist -v`
Expected: FAIL with `AttributeError: ... has no attribute 'testedOn'`

- [ ] **Step 3: Add the predicates**

In `src/aikaboom/store/vocab.py`, in the `# Predicates: BOM-domain edges` block, immediately after `trainedOn = AIBOM.trainedOn`:

```python
testedOn = AIBOM.testedOn
dependsOn = AIBOM.dependsOn
```

- [ ] **Step 4: Add the SCHEMA.md rows**

In `docs/worldofboms/SCHEMA.md`, find the table row `| aibom:trainedOn | Model → Dataset |` and add two rows directly below it, matching the existing column layout:

```
| `aibom:testedOn` | Model → Dataset |
| `aibom:dependsOn` | Model → Model / Dataset → Dataset |
```

(If `SCHEMA.md`'s table has more columns, fill them following the `trainedOn` row's pattern.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/store/test_vocab.py::test_testedon_and_dependson_predicates_exist tests/store/test_docs_schema_parity.py -v`
Expected: PASS (the parity test confirms vocab ↔ SCHEMA.md agree).

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/vocab.py docs/worldofboms/SCHEMA.md tests/store/test_vocab.py
git commit -m "feat(store): add testedOn + dependsOn vocab predicates"
```

---

### Task A2: Extract `BomStore.merge_artifacts` from the CLI

The placeholder-promotion path (A7) and `aikaboom graph merge` must share one merge implementation.

**Files:**
- Modify: `src/aikaboom/store/store.py`
- Modify: `src/aikaboom/store/cli_graph.py:138-167` (the `cmd_graph_merge` function)
- Test: `tests/store/test_store_save.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_store_save.py`:

```python
def test_merge_artifacts_transfers_incoming_edges(store):
    """merge_artifacts(into, from_) redirects edges that pointed at from_."""
    from rdflib import URIRef
    a = "bom:artifact/real"
    b = "bom:artifact/placeholder"
    model = "bom:artifact/model"
    # model --dependsOn--> placeholder b
    store._backend.add_quads([
        (URIRef(model), URIRef("https://aikaboom.dev/aibom#dependsOn"), URIRef(b), None),
        (URIRef(b), URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
         URIRef("https://aikaboom.dev/aibom#Artifact"), None),
    ])
    store.merge_artifacts(into=a, from_=b)
    # The edge now points at a; nothing points at b; b has no outgoing triples.
    rows = list(store._backend.select(
        f"SELECT ?s WHERE {{ ?s <https://aikaboom.dev/aibom#dependsOn> <{a}> }}"))
    assert len(rows) == 1
    leftover = list(store._backend.select(f"SELECT ?p WHERE {{ <{b}> ?p ?o }}"))
    assert leftover == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_store_save.py::test_merge_artifacts_transfers_incoming_edges -v`
Expected: FAIL with `AttributeError: 'BomStore' object has no attribute 'merge_artifacts'`

- [ ] **Step 3: Add `merge_artifacts` to `BomStore`**

In `src/aikaboom/store/store.py`, add this method to the `BomStore` class (place it just before `def close`):

```python
    def merge_artifacts(self, into: str, from_: str) -> None:
        """Merge artifact `from_` into `into`.

        Transfers `from_`'s versions, identifiers, and every incoming edge
        onto `into`, then deletes `from_`. Used by `aikaboom graph merge`
        and by placeholder promotion at save time.
        """
        a = _validate_sparql_iri(into)
        b = _validate_sparql_iri(from_)
        self._backend.update(
            f"INSERT {{ <{a}> <{vocab.hasVersion}> ?v . }} "
            f"WHERE {{ <{b}> <{vocab.hasVersion}> ?v . }}"
        )
        self._backend.update(
            f"INSERT {{ <{a}> <{vocab.identifier}> ?i . }} "
            f"WHERE {{ <{b}> <{vocab.identifier}> ?i . }}"
        )
        # Redirect every incoming reference to b → a so none dangle.
        self._backend.update(
            f"INSERT {{ ?s ?p <{a}> . }} "
            f"WHERE {{ ?s ?p <{b}> . FILTER(?s != <{a}>) }}"
        )
        self._backend.update(f"DELETE {{ ?s ?p <{b}> . }} WHERE {{ ?s ?p <{b}> . }}")
        self._backend.update(f"DELETE {{ <{b}> ?p ?o . }} WHERE {{ <{b}> ?p ?o . }}")
```

- [ ] **Step 4: Make the CLI delegate to it**

In `src/aikaboom/store/cli_graph.py`, replace the entire body of `cmd_graph_merge` with:

```python
def cmd_graph_merge(args: argparse.Namespace) -> int:
    """Merge artifact-b into artifact-a."""
    store = BomStore.open()
    store.merge_artifacts(into=args.artifact_a, from_=args.artifact_b)
    print(f"Merged {args.artifact_b} into {args.artifact_a}.")
    return 0
```

The `from aikaboom.store import vocab` import inside the old function body can be removed if it is now unused there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/store/test_store_save.py::test_merge_artifacts_transfers_incoming_edges tests/store/test_cli_graph.py -v`
Expected: PASS (both the new test and the existing CLI graph tests).

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/store.py src/aikaboom/store/cli_graph.py tests/store/test_store_save.py
git commit -m "refactor(store): extract merge_artifacts; CLI merge delegates to it"
```

---

### Task A3: `extract_relationship_targets` — pull edge targets from a BOM

**Files:**
- Create: `src/aikaboom/store/edges.py`
- Test: `tests/store/test_edges.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/store/test_edges.py`:

```python
"""Edge extraction + persistence: artifact-to-artifact relationships."""

from aikaboom.store.edges import extract_relationship_targets


def _bom_with(field, value):
    return {"direct_fields": {field: {"value": value, "source": "huggingface",
                                      "conflict": None}}, "rag_fields": {}}


def test_extracts_trainedon_target():
    bom = _bom_with("trainedOnDatasets", "squad")
    assert ("trainedOn", "squad") in extract_relationship_targets(bom)


def test_extracts_testedon_and_dependson():
    assert ("testedOn", "glue") in extract_relationship_targets(
        _bom_with("testedOnDatasets", "glue"))
    assert ("dependsOn", "bert-base") in extract_relationship_targets(
        _bom_with("modelLineage", "bert-base"))


def test_splits_multi_value_strings():
    targets = extract_relationship_targets(_bom_with("trainedOnDatasets", "squad, glue; mnli"))
    names = {t for _, t in targets}
    assert names == {"squad", "glue", "mnli"}


def test_drops_non_walkable_targets():
    # arXiv refs are filtered by _is_walkable_target
    targets = extract_relationship_targets(_bom_with("sourceInfo", "arXiv:2108.07732"))
    assert targets == []


def test_ignores_unknown_and_empty_fields():
    assert extract_relationship_targets({"direct_fields": {}, "rag_fields": {}}) == []
    assert extract_relationship_targets(_bom_with("trainedOnDatasets", None)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_edges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.store.edges'`

- [ ] **Step 3: Create `src/aikaboom/store/edges.py`**

```python
"""Artifact-to-artifact relationship edges for the worldofBOMs graph.

A BOM's relationship fields (trainedOnDatasets, testedOnDatasets,
modelLineage, sourceInfo) name other artifacts. This module turns those
names into real `trainedOn` / `testedOn` / `dependsOn` edges between
Artifact nodes, so the stored graph is connected rather than a set of
disconnected stars.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from aikaboom.utils.recursive_bom import (
    AI_RELATIONSHIP_FIELDS,
    DATA_RELATIONSHIP_FIELDS,
    _is_walkable_target,
)

# {bom_field_name: edge_predicate_name}. Reuses the single source of truth
# in recursive_bom.py — the second tuple element is the predicate.
_FIELD_TO_PREDICATE: dict[str, str] = {
    field: spec[1]
    for field, spec in {**AI_RELATIONSHIP_FIELDS, **DATA_RELATIONSHIP_FIELDS}.items()
}

_TARGET_SPLIT = re.compile(r"[;,\n]")


def _split_targets(value: Any) -> list[str]:
    """Normalize a relationship-field value into a list of target names."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = [str(v) for v in value]
    else:
        raw = _TARGET_SPLIT.split(str(value))
    return [t.strip() for t in raw if t and t.strip()]


def extract_relationship_targets(bom_json: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return `(predicate, target_name)` pairs for every walkable edge target.

    `predicate` is one of "trainedOn" / "testedOn" / "dependsOn".
    """
    out: list[tuple[str, str]] = []
    for section in ("direct_fields", "rag_fields"):
        fields = bom_json.get(section) or {}
        if not isinstance(fields, Mapping):
            continue
        for field_name, predicate in _FIELD_TO_PREDICATE.items():
            triplet = fields.get(field_name)
            if not isinstance(triplet, Mapping):
                continue
            for target in _split_targets(triplet.get("value")):
                if _is_walkable_target(target):
                    out.append((predicate, target))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/store/test_edges.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/edges.py tests/store/test_edges.py
git commit -m "feat(store): extract relationship-edge targets from a BOM"
```

---

### Task A4: Resolve an edge target to an Artifact IRI (identity → name → placeholder)

**Files:**
- Modify: `src/aikaboom/store/edges.py`
- Test: `tests/store/test_edges.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_edges.py`:

```python
import pytest
from aikaboom.store.store import BomStore
from aikaboom.store.naming import Identifier
from aikaboom.store.edges import resolve_edge_target, canon_name


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def test_canon_name_lowercases_and_collapses():
    assert canon_name("SQuAD") == "squad"
    assert canon_name("Common_Crawl") == canon_name("common-crawl")


def test_resolve_edge_target_mints_placeholder_when_unknown(store):
    iri, minted = resolve_edge_target(store, "totally-unknown-dataset")
    assert iri.startswith("bom:artifact/")
    assert minted is True
    # The placeholder is flagged.
    rows = list(store._backend.select(
        f"SELECT ?o WHERE {{ <{iri}> <https://aikaboom.dev/aibom#isPlaceholder> ?o }}"))
    assert len(rows) == 1


def test_resolve_edge_target_finds_existing_artifact_by_label(store, sample_bom,
                                                              sample_run_meta):
    # sample_bom's repo_id is "mistralai/Mistral-7B-v0.1" → canonicalLabel.
    store.save_claim(sample_bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")])
    iri, minted = resolve_edge_target(store, "mistralai/Mistral-7B-v0.1")
    assert minted is False  # matched the real artifact, no placeholder
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_edges.py::test_resolve_edge_target_mints_placeholder_when_unknown -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_edge_target'`

- [ ] **Step 3: Add `canon_name`, the name-label match, and `resolve_edge_target`**

Append to `src/aikaboom/store/edges.py`:

```python
from rdflib import Literal, URIRef, XSD
from rdflib.namespace import RDF

from aikaboom.store import iris, vocab
from aikaboom.store.naming import Identifier, canonicalize


def canon_name(name: str) -> str:
    """Canonicalize a free-text artifact name for identity comparison.

    Reuses the identifier canonicalization pipeline with the `name-only`
    platform (lowercase, separator collapse) — the conservative identity
    layer, never the fuzzy supplier triage.
    """
    return canonicalize(Identifier("name-only", name)).value


def _find_artifact_by_label(store, name: str) -> str | None:
    """Return a non-placeholder Artifact IRI whose canonical label matches `name`."""
    target = canon_name(name)
    rows = store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact a <{vocab.Artifact}> ;
                      <{vocab.canonicalLabel}> ?label .
            FILTER NOT EXISTS {{ ?artifact <{vocab.isPlaceholder}> true }}
        }}
        """
    )
    for row in rows:
        if canon_name(str(row["label"])) == target:
            return str(row["artifact"])
    return None


def _mint_placeholder(store, name: str) -> str:
    """Create a flagged placeholder Artifact for an unresolved name; return its IRI."""
    ident = canonicalize(Identifier("name-only", name))
    art = iris.artifact_iri(ident)
    if store._backend.ask(f"ASK {{ <{art}> a <{vocab.Artifact}> }}"):
        return art  # already minted by an earlier edge
    quads = [
        (URIRef(art), RDF.type, URIRef(vocab.Artifact), None),
        (URIRef(art), URIRef(vocab.isPlaceholder),
         Literal(True, datatype=XSD.boolean), None),
        (URIRef(art), URIRef(vocab.canonicalLabel), Literal(name), None),
        (URIRef(art), URIRef(vocab.primaryIdentifier),
         Literal(f"name-only:{ident.value}"), None),
    ]
    ident_node = URIRef(f"{art}/id")
    quads += [
        (URIRef(art), URIRef(vocab.identifier), ident_node, None),
        (ident_node, URIRef(vocab.platform), Literal("name-only"), None),
        (ident_node, URIRef(vocab.value), Literal(ident.value), None),
    ]
    store._backend.add_quads(quads)
    return art


def resolve_edge_target(store, name: str) -> tuple[str, bool]:
    """Resolve a relationship target name to an Artifact IRI.

    Resolution order (identity layer only — no fuzzy matching here):
      1. identifier match via `store.resolve` (the cache-hit path);
      2. exact name-label match against an existing non-placeholder artifact;
      3. mint a flagged placeholder.

    Returns `(artifact_iri, minted_placeholder)`.
    """
    resolved = store.resolve([Identifier("name-only", name)])
    if resolved.existing_artifact:
        return resolved.existing_artifact, False
    by_label = _find_artifact_by_label(store, name)
    if by_label:
        return by_label, False
    return _mint_placeholder(store, name), True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_edges.py -v`
Expected: PASS (all tests including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/edges.py tests/store/test_edges.py
git commit -m "feat(store): resolve edge targets — identifier, name-label, placeholder"
```

---

### Task A5: `add_relationship_edges` — write the edges (idempotent)

**Files:**
- Modify: `src/aikaboom/store/edges.py`
- Test: `tests/store/test_edges.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_edges.py`:

```python
from aikaboom.store.edges import add_relationship_edges


def test_add_relationship_edges_writes_edge(store):
    src = "bom:artifact/source-model"
    bom = _bom_with("trainedOnDatasets", "squad")
    add_relationship_edges(store, src, bom)
    rows = list(store._backend.select(
        f"SELECT ?t WHERE {{ <{src}> <{'https://aikaboom.dev/aibom#trainedOn'}> ?t }}"))
    assert len(rows) == 1


def test_add_relationship_edges_is_idempotent(store):
    src = "bom:artifact/source-model"
    bom = _bom_with("trainedOnDatasets", "squad")
    add_relationship_edges(store, src, bom)
    add_relationship_edges(store, src, bom)  # second call must not duplicate
    rows = list(store._backend.select(
        f"SELECT ?t WHERE {{ <{src}> <{'https://aikaboom.dev/aibom#trainedOn'}> ?t }}"))
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_edges.py::test_add_relationship_edges_writes_edge -v`
Expected: FAIL with `ImportError: cannot import name 'add_relationship_edges'`

- [ ] **Step 3: Add `add_relationship_edges`**

Append to `src/aikaboom/store/edges.py`:

```python
from aikaboom.store.store import _validate_sparql_iri


def add_relationship_edges(store, source_artifact_iri: str,
                           bom_json: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Persist `trainedOn`/`testedOn`/`dependsOn` edges from a saved BOM.

    For each relationship target in `bom_json`, resolves it to an Artifact
    IRI (Task A4) and writes one edge triple. Edge writes are idempotent —
    an `ASK` guard skips a triple that already exists.

    Returns the list of `(source, predicate, target)` edges added.
    """
    src = _validate_sparql_iri(source_artifact_iri)
    added: list[tuple[str, str, str]] = []
    for predicate, target_name in extract_relationship_targets(bom_json):
        pred_uri = str(getattr(vocab, predicate))
        target_iri, _minted = resolve_edge_target(store, target_name)
        tgt = _validate_sparql_iri(target_iri)
        if tgt == src:
            continue  # never self-loop
        if store._backend.ask(f"ASK {{ <{src}> <{pred_uri}> <{tgt}> }}"):
            continue
        store._backend.add_quads(
            [(URIRef(src), URIRef(pred_uri), URIRef(tgt), None)]
        )
        added.append((src, predicate, tgt))
    return added
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_edges.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/edges.py tests/store/test_edges.py
git commit -m "feat(store): add_relationship_edges — idempotent edge persistence"
```

---

### Task A6: Wire edge creation into `BomStore.save_claim`

**Files:**
- Modify: `src/aikaboom/store/store.py` (`save_claim`, lines 73-83)
- Test: `tests/store/test_store_save.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_store_save.py`:

```python
def test_save_claim_creates_relationship_edge(store, sample_run_meta):
    """Saving a model BOM with trainedOnDatasets creates a trainedOn edge."""
    from aikaboom.store.naming import Identifier
    bom = {
        "repo_id": "acme/model-x",
        "use_case": "complete",
        "direct_fields": {
            "trainedOnDatasets": {"value": "squad", "source": "huggingface",
                                  "conflict": None},
        },
        "rag_fields": {},
    }
    store.save_claim(bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "acme/model-x")])
    rows = list(store._backend.select(
        "SELECT ?s ?t WHERE { ?s <https://aikaboom.dev/aibom#trainedOn> ?t }"))
    assert len(rows) == 1


def test_save_claim_connects_to_existing_dataset_node(store, sample_run_meta):
    """A model BOM naming an already-stored dataset links to it — no new node."""
    from aikaboom.store.naming import Identifier
    dataset_bom = {"repo_id": "rajpurkar/squad", "use_case": "complete",
                   "direct_fields": {}, "rag_fields": {}}
    store.save_claim(dataset_bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "rajpurkar/squad")])
    before = store.stats()["artifacts"]
    model_bom = {"repo_id": "acme/model-y", "use_case": "complete",
                 "direct_fields": {"trainedOnDatasets": {
                     "value": "rajpurkar/squad", "source": "huggingface",
                     "conflict": None}}, "rag_fields": {}}
    store.save_claim(model_bom, sample_run_meta,
                     identifiers=[Identifier("huggingface", "acme/model-y")])
    after = store.stats()["artifacts"]
    # +1 for the model only — the dataset edge reused the existing node.
    assert after == before + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_store_save.py::test_save_claim_creates_relationship_edge -v`
Expected: FAIL — no `trainedOn` triple exists.

- [ ] **Step 3: Wire edge creation into `save_claim`**

In `src/aikaboom/store/store.py`, replace the `save_claim` method body with:

```python
    def save_claim(
        self,
        bom_json: Mapping[str, Any],
        run_meta: Mapping[str, Any],
        identifiers: list[Identifier],
    ) -> str:
        """Convert and persist a BOM. Returns the new claim IRI.

        After the artifact subgraph is written, relationship fields are
        resolved into `trainedOn`/`testedOn`/`dependsOn` edges so the graph
        stays connected. Edge creation is best-effort — a failure there
        never loses the saved claim.
        """
        ds, claim_iri = bom_to_rdf(bom_json, run_meta, identifiers=identifiers)
        quads = [(s, p, o, None) for s, p, o, _ in ds.quads()]
        self._backend.add_quads(quads)

        try:
            from aikaboom.store.edges import add_relationship_edges

            artifact = iris.artifact_iri(pick_primary(canonicalize_set(identifiers)))
            add_relationship_edges(self, artifact, bom_json)
        except Exception as e:  # noqa: BLE001 — never let edges break a save
            import logging

            logging.getLogger(__name__).warning("edge creation failed: %s", e)
        return claim_iri
```

(The `iris`, `pick_primary`, `canonicalize_set` names are already imported at the top of `store.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_store_save.py -v`
Expected: PASS (new tests + all existing save tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/store.py tests/store/test_store_save.py
git commit -m "feat(store): save_claim writes relationship edges (connected graph)"
```

---

### Task A7: Reverse placeholder promotion on save

When a real BOM arrives for a name a placeholder was minted under, merge the placeholder into the real artifact.

**Files:**
- Modify: `src/aikaboom/store/edges.py`
- Modify: `src/aikaboom/store/store.py` (`save_claim`)
- Test: `tests/store/test_edges.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_edges.py`:

```python
from aikaboom.store.edges import promote_placeholders_for
from aikaboom.store.naming import Identifier as _Id


def test_placeholder_is_promoted_into_a_later_real_bom(store, sample_run_meta):
    # 1. A model BOM names dataset "squad" → placeholder minted + edge.
    model_bom = _bom_with("trainedOnDatasets", "squad")
    store.save_claim(model_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "acme/model-z")])
    # 2. A real BOM for "squad" arrives.
    real_bom = {"repo_id": "squad", "use_case": "complete",
                "direct_fields": {}, "rag_fields": {}}
    real_iri_artifacts_before = store.stats()["artifacts"]
    store.save_claim(real_bom, sample_run_meta,
                     identifiers=[_Id("huggingface", "squad")])
    # The placeholder was merged away: the model's trainedOn edge now points
    # at the real artifact, and no placeholder artifact remains.
    placeholders = list(store._backend.select(
        "SELECT ?a WHERE { ?a <https://aikaboom.dev/aibom#isPlaceholder> true }"))
    assert placeholders == []
    edges = list(store._backend.select(
        "SELECT ?s ?t WHERE { ?s <https://aikaboom.dev/aibom#trainedOn> ?t }"))
    assert len(edges) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_edges.py::test_placeholder_is_promoted_into_a_later_real_bom -v`
Expected: FAIL — a placeholder artifact still exists.

- [ ] **Step 3: Add `promote_placeholders_for`**

Append to `src/aikaboom/store/edges.py`:

```python
def promote_placeholders_for(store, real_artifact_iri: str, label: str) -> list[str]:
    """Merge any name-only placeholders that match `label` into the real artifact.

    Exact canonical-name equality only — never fuzzy. Returns the list of
    placeholder IRIs that were merged away.
    """
    target = canon_name(label)
    real = _validate_sparql_iri(real_artifact_iri)
    merged: list[str] = []
    rows = list(store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact <{vocab.isPlaceholder}> true ;
                      <{vocab.canonicalLabel}> ?label .
        }}
        """
    ))
    for row in rows:
        placeholder = str(row["artifact"])
        if placeholder == real:
            continue
        if canon_name(str(row["label"])) == target:
            store.merge_artifacts(into=real, from_=placeholder)
            merged.append(placeholder)
    return merged
```

- [ ] **Step 4: Call it from `save_claim`**

In `src/aikaboom/store/store.py`, inside `save_claim`'s `try:` block, after the `add_relationship_edges(...)` line, add:

```python
            from aikaboom.store.edges import promote_placeholders_for

            label = bom_json.get("repo_id") or bom_json.get("model_id")
            if label:
                promote_placeholders_for(self, artifact, str(label))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/store/test_edges.py tests/store/test_store_save.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/store/edges.py src/aikaboom/store/store.py tests/store/test_edges.py
git commit -m "feat(store): promote name-only placeholders into later real BOMs"
```

---

### Task A8: Confidence-triage → `potentialDuplicateOf` soft edges

When a placeholder is minted but the supplier-alias triage finds a confident-but-inexact match, record a soft `potentialDuplicateOf` edge — never auto-merge.

**Files:**
- Modify: `src/aikaboom/store/edges.py` (`resolve_edge_target`)
- Test: `tests/store/test_edges.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_edges.py`:

```python
def test_confident_inexact_match_records_potential_duplicate(store, sample_run_meta):
    """A placeholder minted near an existing artifact gets a potentialDuplicateOf edge."""
    from aikaboom.store.naming import Identifier
    # Existing real artifact under owner "qwen".
    store.save_claim({"repo_id": "qwen/chat", "use_case": "complete",
                      "direct_fields": {}, "rag_fields": {}},
                     sample_run_meta, identifiers=[Identifier("huggingface", "qwen/chat")])
    # Edge target "QwenLM/chat" — same supplier, inexact name → placeholder + soft edge.
    iri, minted = resolve_edge_target(store, "QwenLM/chat")
    assert minted is True
    dup = list(store._backend.select(
        f"SELECT ?o WHERE {{ <{iri}> <https://aikaboom.dev/aibom#potentialDuplicateOf> ?o }}"))
    assert len(dup) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_edges.py::test_confident_inexact_match_records_potential_duplicate -v`
Expected: FAIL — no `potentialDuplicateOf` triple.

- [ ] **Step 3: Add the triage check after placeholder minting**

In `src/aikaboom/store/edges.py`, replace the final `return _mint_placeholder(store, name), True` line of `resolve_edge_target` with:

```python
    placeholder = _mint_placeholder(store, name)
    _record_potential_duplicates(store, placeholder, name)
    return placeholder, True
```

Then append this helper to the module:

```python
def _record_potential_duplicates(store, placeholder_iri: str, name: str) -> None:
    """Soft-link a placeholder to confident-but-inexact existing artifacts.

    Uses the supplier-alias confidence triage (Jaro-Winkler tier included).
    Records `potentialDuplicateOf` — a hint for the UI — and never merges.
    """
    from aikaboom.utils.supplier_alias import default_alias_index

    index = default_alias_index()
    owner = name.partition("/")[0] if "/" in name else name
    rows = list(store._backend.select(
        f"""
        SELECT ?artifact ?label WHERE {{
            ?artifact a <{vocab.Artifact}> ;
                      <{vocab.canonicalLabel}> ?label .
            FILTER NOT EXISTS {{ ?artifact <{vocab.isPlaceholder}> true }}
        }}
        """
    ))
    src = _validate_sparql_iri(placeholder_iri)
    for row in rows:
        cand_label = str(row["label"])
        cand_owner = cand_label.partition("/")[0] if "/" in cand_label else cand_label
        if canon_name(cand_label) == canon_name(name):
            continue  # exact — handled by name-label match, not a "duplicate"
        if index.is_same_supplier(owner, cand_owner):
            tgt = _validate_sparql_iri(str(row["artifact"]))
            if not store._backend.ask(
                f"ASK {{ <{src}> <{vocab.potentialDuplicateOf}> <{tgt}> }}"
            ):
                store._backend.add_quads(
                    [(URIRef(src), URIRef(vocab.potentialDuplicateOf),
                      URIRef(tgt), None)]
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_edges.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/edges.py tests/store/test_edges.py
git commit -m "feat(store): potentialDuplicateOf soft edges from confidence triage"
```

---

### Task A9: Verify `graph rebuild` reconstructs edges retroactively

Edge creation lives in `save_claim`, which `cmd_graph_rebuild` already calls — this task is a regression test, no production change.

**Files:**
- Test: `tests/store/test_cli_graph.py`

- [ ] **Step 1: Write the test**

Add to `tests/store/test_cli_graph.py` (match the file's existing import + fixture style):

```python
def test_rebuild_reconstructs_relationship_edges(tmp_path, monkeypatch):
    """graph rebuild replays results/*.json through save_claim, so edges form."""
    import json
    from types import SimpleNamespace
    from aikaboom.store.cli_graph import cmd_graph_rebuild
    from aikaboom.store.store import BomStore

    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(tmp_path / "graph"))
    results = tmp_path / "results"
    results.mkdir()
    (results / "acme_m.json").write_text(json.dumps({
        "repo_id": "acme/m", "use_case": "complete",
        "direct_fields": {"trainedOnDatasets": {"value": "squad",
                          "source": "huggingface", "conflict": None}},
        "rag_fields": {},
    }))
    monkeypatch.chdir(tmp_path)
    cmd_graph_rebuild(SimpleNamespace())
    store = BomStore.open()
    edges = list(store._backend.select(
        "SELECT ?s ?t WHERE { ?s <https://aikaboom.dev/aibom#trainedOn> ?t }"))
    assert len(edges) == 1
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/store/test_cli_graph.py::test_rebuild_reconstructs_relationship_edges -v`
Expected: PASS (no production change needed — if it fails, edge wiring from A6 is broken).

- [ ] **Step 3: Commit**

```bash
git add tests/store/test_cli_graph.py
git commit -m "test(store): graph rebuild reconstructs relationship edges"
```

---

## Part B — Backend Read Side

### Task B1: `graph_view.full_graph`

**Files:**
- Create: `src/aikaboom/store/graph_view.py`
- Test: `tests/store/test_graph_view.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/store/test_graph_view.py`:

```python
"""Read-side graph queries for the worldofBOMs visualization."""

import pytest

from aikaboom.store.store import BomStore
from aikaboom.store.naming import Identifier
from aikaboom.store import graph_view


@pytest.fixture
def store(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    return BomStore.open()


def _save_model_with_dataset(store, run_meta, model="acme/m", dataset="squad"):
    bom = {"repo_id": model, "use_case": "complete",
           "direct_fields": {"trainedOnDatasets": {"value": dataset,
                             "source": "huggingface", "conflict": None}},
           "rag_fields": {}}
    store.save_claim(bom, run_meta, identifiers=[Identifier("huggingface", model)])


def test_full_graph_returns_nodes_and_edges(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    assert len(g["nodes"]) == 2          # model + (placeholder) dataset
    assert len(g["edges"]) == 1
    edge = g["edges"][0]
    assert edge["predicate"] == "trainedOn"
    labels = {n["label"] for n in g["nodes"]}
    assert "acme/m" in labels


def test_full_graph_marks_placeholder_nodes(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    placeholders = [n for n in g["nodes"] if n["is_placeholder"]]
    assert len(placeholders) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_graph_view.py::test_full_graph_returns_nodes_and_edges -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.store.graph_view'`

- [ ] **Step 3: Create `src/aikaboom/store/graph_view.py`**

```python
"""Read-side graph queries for the worldofBOMs browser visualization.

Pure functions over a `BomStore`. `web/app.py` calls these; all graph
SPARQL lives here so the Flask layer stays thin.
"""

from __future__ import annotations

from aikaboom.store import vocab
from aikaboom.store.store import _validate_sparql_iri

# Artifact-to-artifact edge predicates we render.
_EDGE_PREDICATES = {
    str(vocab.trainedOn): "trainedOn",
    str(vocab.testedOn): "testedOn",
    str(vocab.dependsOn): "dependsOn",
}

_KIND_BY_CLASS = {
    str(vocab.Model): "Model",
    str(vocab.Dataset): "Dataset",
    str(vocab.Paper): "Paper",
    str(vocab.CodeRepo): "CodeRepo",
}


def _node_rows(store) -> list[dict]:
    """Every Artifact as a node dict."""
    rows = store._backend.select(
        f"""
        SELECT ?artifact ?label ?placeholder WHERE {{
            ?artifact a <{vocab.Artifact}> .
            OPTIONAL {{ ?artifact <{vocab.canonicalLabel}> ?label . }}
            OPTIONAL {{ ?artifact <{vocab.isPlaceholder}> ?placeholder . }}
        }}
        """
    )
    out: dict[str, dict] = {}
    for row in rows:
        iri = str(row["artifact"])
        if iri in out:
            continue
        out[iri] = {
            "iri": iri,
            "label": str(row.get("label") or iri.rsplit("/", 1)[-1]),
            "is_placeholder": bool(row.get("placeholder")),
            "kind": _kind_for(store, iri),
            "claim_count": _claim_count(store, iri),
        }
    return list(out.values())


def _kind_for(store, artifact_iri: str) -> str:
    iri = _validate_sparql_iri(artifact_iri)
    for row in store._backend.select(f"SELECT ?c WHERE {{ <{iri}> a ?c }}"):
        kind = _KIND_BY_CLASS.get(str(row["c"]))
        if kind:
            return kind
    return "Artifact"


def _claim_count(store, artifact_iri: str) -> int:
    iri = _validate_sparql_iri(artifact_iri)
    rows = list(store._backend.select(
        f"""
        SELECT (COUNT(?claim) AS ?n) WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
        }}
        """
    ))
    return int(rows[0]["n"]) if rows and rows[0].get("n") is not None else 0


def _edge_rows(store) -> list[dict]:
    """Every artifact-to-artifact relationship edge."""
    out: list[dict] = []
    for pred_uri, pred_name in _EDGE_PREDICATES.items():
        for row in store._backend.select(
            f"SELECT ?s ?t WHERE {{ ?s <{pred_uri}> ?t }}"
        ):
            out.append({"source": str(row["s"]), "target": str(row["t"]),
                        "predicate": pred_name})
    return out


def full_graph(store) -> dict:
    """The whole graph: every artifact node and every relationship edge."""
    return {"nodes": _node_rows(store), "edges": _edge_rows(store)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_graph_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/graph_view.py tests/store/test_graph_view.py
git commit -m "feat(store): graph_view.full_graph — nodes + edges for the viz"
```

---

### Task B2: `graph_view.ego_graph` — directional traversal

**Semantics (clarifies spec B.1):** edges point dependent → dependency (`Model trainedOn Dataset` = the model depends on the dataset). `upstream` = a node's dependencies, reached by following edges *forward* from the focus. `downstream` = a node's dependents, reached by following edges *backward*. `both` = the union.

**Files:**
- Modify: `src/aikaboom/store/graph_view.py`
- Test: `tests/store/test_graph_view.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_graph_view.py`:

```python
def test_ego_graph_upstream_returns_dependencies(store, sample_run_meta):
    # m --trainedOn--> squad   (squad is m's dependency = upstream of m)
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = store_full = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    ego = graph_view.ego_graph(store, m_iri, direction="up", depth=None)
    labels = {n["label"] for n in ego["nodes"]}
    assert "acme/m" in labels and "squad" in labels
    assert ego["focus"] == m_iri


def test_ego_graph_downstream_excludes_dependencies(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    ego = graph_view.ego_graph(store, m_iri, direction="down", depth=None)
    # m has no dependents → downstream ego is just m itself.
    assert {n["label"] for n in ego["nodes"]} == {"acme/m"}


def test_ego_graph_both_is_the_union(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    squad_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "squad")
    ego = graph_view.ego_graph(store, squad_iri, direction="both", depth=None)
    assert {n["label"] for n in ego["nodes"]} == {"acme/m", "squad"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_graph_view.py::test_ego_graph_upstream_returns_dependencies -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'ego_graph'`

- [ ] **Step 3: Add `ego_graph`**

Append to `src/aikaboom/store/graph_view.py`:

```python
def ego_graph(store, artifact_iri: str, direction: str = "both",
              depth: int | None = None) -> dict:
    """Breadth-first ego subgraph around `artifact_iri`.

    direction: "up" follows edges forward (dependencies), "down" follows
    them backward (dependents), "both" does the union. `depth` caps the
    hop count; None means unlimited (full lineage).
    """
    focus = _validate_sparql_iri(artifact_iri)
    all_edges = _edge_rows(store)
    forward: dict[str, list[dict]] = {}
    backward: dict[str, list[dict]] = {}
    for e in all_edges:
        forward.setdefault(e["source"], []).append(e)
        backward.setdefault(e["target"], []).append(e)

    keep_nodes: set[str] = {focus}
    keep_edges: list[dict] = []
    seen_edges: set[tuple] = set()
    frontier = [focus]
    hops = 0
    while frontier and (depth is None or hops < depth):
        nxt: list[str] = []
        for node in frontier:
            steps: list[tuple[dict, str]] = []
            if direction in ("up", "both"):
                steps += [(e, e["target"]) for e in forward.get(node, [])]
            if direction in ("down", "both"):
                steps += [(e, e["source"]) for e in backward.get(node, [])]
            for edge, other in steps:
                key = (edge["source"], edge["predicate"], edge["target"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    keep_edges.append(edge)
                if other not in keep_nodes:
                    keep_nodes.add(other)
                    nxt.append(other)
        frontier = nxt
        hops += 1

    nodes = [n for n in _node_rows(store) if n["iri"] in keep_nodes]
    return {"nodes": nodes, "edges": keep_edges, "focus": focus}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_graph_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/graph_view.py tests/store/test_graph_view.py
git commit -m "feat(store): graph_view.ego_graph — directional lineage traversal"
```

---

### Task B3: `graph_view.lineage_query` — the four presets

**Files:**
- Modify: `src/aikaboom/store/graph_view.py`
- Test: `tests/store/test_graph_view.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_graph_view.py`:

```python
def test_lineage_query_lists_datasets_in_lineage(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    rows = graph_view.lineage_query(store, m_iri, preset="datasets", direction="up")
    assert any(r["label"] == "squad" for r in rows)


def test_lineage_query_lists_models_in_lineage(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    rows = graph_view.lineage_query(store, m_iri, preset="models", direction="both")
    assert any(r["label"] == "acme/m" for r in rows)


def test_lineage_query_unknown_preset_raises(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    g = graph_view.full_graph(store)
    iri = g["nodes"][0]["iri"]
    with pytest.raises(ValueError):
        graph_view.lineage_query(store, iri, preset="nonsense", direction="up")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_graph_view.py::test_lineage_query_lists_datasets_in_lineage -v`
Expected: FAIL with `AttributeError: ... 'lineage_query'`

- [ ] **Step 3: Add `lineage_query`**

Append to `src/aikaboom/store/graph_view.py`:

```python
_PRESETS = ("licenses", "datasets", "models", "conflicts")


def lineage_query(store, artifact_iri: str, preset: str,
                  direction: str = "both") -> list[dict]:
    """Run one preset query over the ego node set of `artifact_iri`.

    presets: "licenses", "datasets", "models", "conflicts".
    """
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {_PRESETS}")
    ego = ego_graph(store, artifact_iri, direction=direction, depth=None)
    nodes = ego["nodes"]

    if preset == "datasets":
        return [{"iri": n["iri"], "label": n["label"]}
                for n in nodes if n["kind"] == "Dataset"]
    if preset == "models":
        return [{"iri": n["iri"], "label": n["label"]}
                for n in nodes if n["kind"] == "Model"]
    if preset == "licenses":
        out: list[dict] = []
        for n in nodes:
            for lic in _licenses_for(store, n["iri"]):
                out.append({"artifact": n["label"], "license": lic})
        return out
    # conflicts
    out = []
    for n in nodes:
        for kind in _conflicts_for(store, n["iri"]):
            out.append({"artifact": n["label"], "conflict": kind})
    return out


def _licenses_for(store, artifact_iri: str) -> list[str]:
    """License values from the artifact's canonical-claim field literals."""
    iri = _validate_sparql_iri(artifact_iri)
    rows = store._backend.select(
        f"""
        SELECT DISTINCT ?lic WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            ?claim <{vocab.AIBOM}licenseName> ?lic .
        }}
        """
    )
    return [str(r["lic"]) for r in rows]


def _conflicts_for(store, artifact_iri: str) -> list[str]:
    """Conflict kinds annotated on the artifact's claims."""
    iri = _validate_sparql_iri(artifact_iri)
    rows = store._backend.select(
        f"""
        SELECT DISTINCT ?kind WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            ?ann <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> ?claim ;
                 <{vocab.conflictKind}> ?kind .
            FILTER(?kind != <{vocab.noConflict}>)
        }}
        """
    )
    return [str(r["kind"]).rsplit("#", 1)[-1] for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_graph_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/graph_view.py tests/store/test_graph_view.py
git commit -m "feat(store): graph_view.lineage_query — four lineage presets"
```

---

### Task B4: `graph_view.raw_query` — SELECT-only SPARQL

**Files:**
- Modify: `src/aikaboom/store/graph_view.py`
- Test: `tests/store/test_graph_view.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_graph_view.py`:

```python
def test_raw_query_runs_select(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    rows = graph_view.raw_query(store, "SELECT ?s WHERE { ?s a ?c } LIMIT 5")
    assert isinstance(rows, list)


@pytest.mark.parametrize("bad", [
    "INSERT DATA { <a:x> <a:y> <a:z> }",
    "DELETE WHERE { ?s ?p ?o }",
    "  delete  { ?s ?p ?o } where { ?s ?p ?o }",
    "DROP ALL",
])
def test_raw_query_rejects_mutations(store, bad):
    with pytest.raises(ValueError):
        graph_view.raw_query(store, bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_graph_view.py::test_raw_query_runs_select -v`
Expected: FAIL with `AttributeError: ... 'raw_query'`

- [ ] **Step 3: Add `raw_query`**

Append to `src/aikaboom/store/graph_view.py`:

```python
import re as _re

_MUTATION_KEYWORDS = _re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY)\b", _re.IGNORECASE
)


def raw_query(store, sparql: str) -> list[dict]:
    """Run a read-only SPARQL query. Rejects anything that can mutate the store."""
    if _MUTATION_KEYWORDS.search(sparql or ""):
        raise ValueError("only read-only SELECT/ASK queries are allowed")
    # Strip PREFIX/comment lines, then require SELECT or ASK as the first keyword.
    stripped = "\n".join(
        ln for ln in (sparql or "").splitlines()
        if ln.strip() and not ln.strip().upper().startswith(("PREFIX", "#", "BASE"))
    ).strip()
    if not _re.match(r"(SELECT|ASK)\b", stripped, _re.IGNORECASE):
        raise ValueError("query must start with SELECT or ASK")
    return [{k: str(v) for k, v in row.items()} for row in store._backend.select(sparql)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_graph_view.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/graph_view.py tests/store/test_graph_view.py
git commit -m "feat(store): graph_view.raw_query — guarded read-only SPARQL"
```

---

### Task B5: `graph_view.ego_spdx_bundle` — SPDX linked-bundle export

**Files:**
- Modify: `src/aikaboom/store/graph_view.py`
- Test: `tests/store/test_graph_view.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/store/test_graph_view.py`:

```python
def test_ego_spdx_bundle_has_context_and_graph(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta, model="acme/m", dataset="squad")
    g = graph_view.full_graph(store)
    m_iri = next(n["iri"] for n in g["nodes"] if n["label"] == "acme/m")
    bundle = graph_view.ego_spdx_bundle(store, m_iri, direction="both")
    assert "@context" in bundle and "@graph" in bundle
    assert isinstance(bundle["@graph"], list) and len(bundle["@graph"]) > 0


def test_ego_spdx_bundle_full_scope(store, sample_run_meta):
    _save_model_with_dataset(store, sample_run_meta)
    bundle = graph_view.ego_spdx_bundle(store, artifact_iri=None, direction="both")
    assert "@graph" in bundle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_graph_view.py::test_ego_spdx_bundle_has_context_and_graph -v`
Expected: FAIL with `AttributeError: ... 'ego_spdx_bundle'`

- [ ] **Step 3: Add `ego_spdx_bundle`**

This reuses the recursive feature's `build_linked_spdx_bundle` (`utils/recursive_bom.py`). For each artifact in the ego set we reconstruct its canonical claim's BOM (`store.reconstruct_bom`) and feed the focus as the parent and the rest as `recursive_result["generated"]` children.

Append to `src/aikaboom/store/graph_view.py`:

```python
def _canonical_claim_iri(store, artifact_iri: str) -> str | None:
    iri = _validate_sparql_iri(artifact_iri)
    rows = list(store._backend.select(
        f"""
        SELECT ?claim ?createdAt WHERE {{
            <{iri}> <{vocab.hasVersion}> ?v . ?v <{vocab.hasClaim}> ?claim .
            OPTIONAL {{ ?claim <{vocab.createdAt}> ?createdAt . }}
        }}
        ORDER BY DESC(?createdAt)
        """
    ))
    return str(rows[0]["claim"]) if rows else None


def ego_spdx_bundle(store, artifact_iri: str | None,
                    direction: str = "both") -> dict:
    """Assemble an SPDX 3.0.1 linked bundle for an ego view (or the whole graph).

    `artifact_iri=None` exports every artifact. Otherwise exports the ego
    set for the given direction. Reconstructs each member's canonical BOM
    and links them via the recursive feature's linked-bundle builder.
    """
    from aikaboom.utils.recursive_bom import build_linked_spdx_bundle

    if artifact_iri is None:
        members = [n["iri"] for n in _node_rows(store)]
        focus_iri = members[0] if members else None
    else:
        ego = ego_graph(store, artifact_iri, direction=direction, depth=None)
        members = [n["iri"] for n in ego["nodes"]]
        focus_iri = artifact_iri

    if not focus_iri:
        return {"@context": None, "@graph": []}

    focus_claim = _canonical_claim_iri(store, focus_iri)
    parent_meta = store.reconstruct_bom(focus_claim) if focus_claim else {"repo_id": focus_iri}

    generated = []
    for iri in members:
        if iri == focus_iri:
            continue
        claim = _canonical_claim_iri(store, iri)
        child_bom = store.reconstruct_bom(claim) if claim else {"repo_id": iri}
        generated.append({
            "bom_type": "data",
            "target": child_bom.get("repo_id", iri),
            "parent": parent_meta.get("repo_id", focus_iri),
            "depth": 1,
            "relationship_type": "dependsOn",
            "metadata": child_bom,
        })
    recursive_result = {"generated": generated, "deepest_level_reached": 1}
    return build_linked_spdx_bundle(parent_meta, recursive_result, bom_type="ai")
```

> **Note for the implementer:** `build_linked_spdx_bundle` reads `node["spdx_data"]` for each child (see its body in `recursive_bom.py`). If reconstruction does not yield SPDX directly, run each `child_bom` through `SPDXValidator(bom_type=...).validate_and_convert(...)` and put the result under `"spdx_data"` in the `generated` entry. Verify against the test in Step 2 and adjust the `generated` dict shape until `@graph` is non-empty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_graph_view.py -v`
Expected: PASS. If `@graph` is empty, apply the implementer note above.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/store/graph_view.py tests/store/test_graph_view.py
git commit -m "feat(store): graph_view.ego_spdx_bundle — SPDX linked-bundle export"
```

---

### Task B6: Flask read routes — `/worldofboms/graph`, `/ego`, `/bom`, `/stats`

**Files:**
- Modify: `src/aikaboom/web/app.py`
- Test: `tests/test_web_ui_features.py`

- [ ] **Step 1: Write the failing test**

Add a class to `tests/test_web_ui_features.py` (the `client` fixture already exists at the top of the file):

```python
class TestWorldOfBomsRoutes:
    def test_graph_route_returns_nodes_and_edges(self, client):
        resp = client.get("/worldofboms/graph")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "nodes" in data and "edges" in data

    def test_stats_route_returns_counts(self, client):
        resp = client.get("/worldofboms/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "artifacts" in data and "edges" in data

    def test_graph_route_survives_unavailable_store(self, client, monkeypatch):
        monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "1")
        resp = client.get("/worldofboms/graph")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["nodes"] == [] and data.get("store_unavailable") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsRoutes -v`
Expected: FAIL with 404 — routes do not exist.

- [ ] **Step 3: Add the routes + a store helper**

In `src/aikaboom/web/app.py`, add this helper near `_try_resolve_cache` (around line 345):

```python
def _open_graph_store():
    """Open the graph store for read routes, or None if unavailable/disabled."""
    if os.environ.get('AIKABOOM_GRAPH_DISABLE') == '1':
        return None
    try:
        from aikaboom.store.store import BomStore
        return BomStore.open()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ graph store unavailable for worldofBOMs: {e}")
        return None
```

Then add the routes (place them after the `download` route, near line 1340):

```python
@app.route('/worldofboms/graph', methods=['GET'])
def worldofboms_graph():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'nodes': [], 'edges': [], 'store_unavailable': True})
    try:
        return jsonify(graph_view.full_graph(store))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms graph failed: {e}")
        return jsonify({'nodes': [], 'edges': [], 'store_unavailable': True})


@app.route('/worldofboms/stats', methods=['GET'])
def worldofboms_stats():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'artifacts': 0, 'versions': 0, 'claims': 0,
                        'edges': 0, 'store_unavailable': True})
    try:
        stats = dict(store.stats())
        stats['edges'] = len(graph_view._edge_rows(store))
        return jsonify(stats)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms stats failed: {e}")
        return jsonify({'artifacts': 0, 'edges': 0, 'store_unavailable': True})


@app.route('/worldofboms/ego/<path:artifact>', methods=['GET'])
def worldofboms_ego(artifact):
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'nodes': [], 'edges': [], 'focus': artifact,
                        'store_unavailable': True})
    direction = request.args.get('direction', 'both')
    depth_arg = request.args.get('depth')
    depth = int(depth_arg) if depth_arg and depth_arg.isdigit() else None
    try:
        return jsonify(graph_view.ego_graph(store, artifact,
                                            direction=direction, depth=depth))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms ego failed: {e}")
        return jsonify({'nodes': [], 'edges': [], 'focus': artifact,
                        'store_unavailable': True})


@app.route('/worldofboms/bom/<path:artifact>', methods=['GET'])
def worldofboms_bom(artifact):
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'store_unavailable': True})
    try:
        claim = graph_view._canonical_claim_iri(store, artifact)
        if not claim:
            return jsonify({'error': 'no claim for artifact'}), 404
        return jsonify(store.reconstruct_bom(claim))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms bom failed: {e}")
        return jsonify({'error': str(e)}), 500
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsRoutes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/web/app.py tests/test_web_ui_features.py
git commit -m "feat(web): /worldofboms graph/ego/bom/stats read routes"
```

---

### Task B7: Flask routes — `/worldofboms/query` and `/worldofboms/export`

**Files:**
- Modify: `src/aikaboom/web/app.py`
- Test: `tests/test_web_ui_features.py`

- [ ] **Step 1: Write the failing test**

Add to `TestWorldOfBomsRoutes` in `tests/test_web_ui_features.py`:

```python
    def test_query_route_runs_preset(self, client):
        resp = client.post("/worldofboms/query",
                            json={"preset": "datasets", "artifact": "bom:artifact/none",
                                  "direction": "up"},
                            content_type="application/json")
        assert resp.status_code == 200
        assert "rows" in resp.get_json()

    def test_query_route_rejects_mutating_sparql(self, client):
        resp = client.post("/worldofboms/query",
                            json={"sparql": "DELETE WHERE { ?s ?p ?o }"},
                            content_type="application/json")
        assert resp.status_code == 400

    def test_export_route_returns_jsonld(self, client):
        resp = client.get("/worldofboms/export?scope=full")
        assert resp.status_code == 200
        assert "@graph" in resp.get_json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsRoutes::test_query_route_runs_preset -v`
Expected: FAIL with 404.

- [ ] **Step 3: Add the routes**

In `src/aikaboom/web/app.py`, after the routes from B6:

```python
@app.route('/worldofboms/query', methods=['POST'])
def worldofboms_query():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'rows': [], 'store_unavailable': True})
    data = request.get_json(silent=True) or {}
    try:
        if data.get('sparql'):
            rows = graph_view.raw_query(store, data['sparql'])
        else:
            rows = graph_view.lineage_query(
                store, data.get('artifact', ''),
                preset=data.get('preset', 'datasets'),
                direction=data.get('direction', 'both'),
            )
        return jsonify({'rows': rows})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms query failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/worldofboms/export', methods=['GET'])
def worldofboms_export():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'@context': None, '@graph': [], 'store_unavailable': True})
    scope = request.args.get('scope', 'full')
    artifact = request.args.get('artifact') if scope == 'ego' else None
    direction = request.args.get('direction', 'both')
    try:
        bundle = graph_view.ego_spdx_bundle(store, artifact, direction=direction)
        resp = jsonify(bundle)
        fname = 'worldofboms-graph.spdx.json' if scope == 'full' \
            else 'worldofboms-ego.spdx.json'
        resp.headers['Content-Disposition'] = f'attachment; filename={fname}'
        return resp
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms export failed: {e}")
        return jsonify({'error': str(e)}), 500
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsRoutes -v`
Expected: PASS (all routes).

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/web/app.py tests/test_web_ui_features.py
git commit -m "feat(web): /worldofboms query + SPDX export routes"
```

---

## Part C — Frontend (`worldofBOMs` tab)

> Frontend tasks add HTML/CSS/JS to `src/aikaboom/web/templates/index.html`. Each is verified by a server-side smoke test asserting the rendered template contains the expected markup, plus a manual browser check. The new tab gets its **own** Cytoscape instance (`worldCy`) and helpers — it does not share the recursive tab's `recursiveCy`.

### Task C1: worldofBOMs tab scaffold (HTML + CSS)

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`
- Test: `tests/test_web_ui_features.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web_ui_features.py`:

```python
class TestWorldOfBomsTab:
    def test_index_has_worldofboms_tab(self, client):
        html = client.get("/").get_data(as_text=True)
        assert "switchTab(event, 'worldofboms')" in html
        assert 'id="worldofbomsTab"' in html
        assert 'id="worldGraphCanvas"' in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsTab -v`
Expected: FAIL — markup absent.

- [ ] **Step 3: Add the tab button**

In `src/aikaboom/web/templates/index.html`, after the recursive tab button (line ~3196 `<button class="tab" onclick="switchTab(event, 'recursive')" ...>`), add:

```html
                <button class="tab" onclick="switchTab(event, 'worldofboms')" title="worldofBOMs — the connected knowledge graph of every BOM generated">worldofBOMs <span class="beta-tag">Beta</span></button>
```

- [ ] **Step 4: Add the tab-content block**

After the closing `</div>` of `<div id="recursiveTab" class="tab-content">` (the recursive tab block ends ~line 3435), add a new tab-content block. It reuses the recursive tab's `.graph-wrap` / `.graph-controls` / `.side-panel` structure with `world`-prefixed ids:

```html
            <div id="worldofbomsTab" class="tab-content">
                <div class="bom-shell" data-bom-shell="worldofboms">
                    <div class="bom-shell-toolbar" id="worldToolbar">
                        <div class="view-toggle" role="tablist" id="worldViewToggle">
                            <button type="button" class="is-active" data-world-view="global">Global</button>
                            <button type="button" data-world-view="ego" disabled id="worldEgoToggle">Ego</button>
                        </div>
                        <div class="view-toggle" id="worldDirection" style="margin-left:8px;">
                            <button type="button" class="is-active" data-world-dir="both">Both</button>
                            <button type="button" data-world-dir="up">Upstream</button>
                            <button type="button" data-world-dir="down">Downstream</button>
                        </div>
                        <input type="search" id="worldSearch" placeholder="Find a node…"
                               style="margin-left:8px;font-size:12px;padding:3px 8px;">
                        <div style="flex:1"></div>
                        <span id="worldStats" style="font-size:11.5px;color:var(--paper-mute);"></span>
                        <a id="worldDownload" class="mock-button"
                           style="margin-left:10px;font-size:11.5px;" href="#">⬇ Download ▾</a>
                    </div>
                    <div class="recursive-body">
                        <div class="graph-wrap" id="worldGraphWrap">
                            <div id="worldGraphCanvas" class="graph-canvas"></div>
                            <div class="graph-legend">
                                <span><i></i> trainedOn</span>
                                <span><i class="is-tested"></i> testedOn</span>
                                <span><i class="is-depends"></i> dependsOn</span>
                            </div>
                            <div class="graph-controls" id="worldGraphControls" aria-label="Graph controls">
                                <button type="button" id="worldZoomIn"  title="Zoom in" aria-label="Zoom in">+</button>
                                <button type="button" id="worldZoomOut" title="Zoom out" aria-label="Zoom out">−</button>
                                <button type="button" id="worldFit"     title="Fit" aria-label="Fit">⤢</button>
                                <button type="button" id="worldReset"   title="Reset" aria-label="Reset">⟲</button>
                                <button type="button" id="worldFull"    title="Fullscreen" aria-label="Fullscreen">⛶</button>
                            </div>
                            <div id="worldGraphEmpty" class="graph-empty">No BOMs in the graph yet — generate one to start your worldofBOMs.</div>
                        </div>
                        <aside id="worldSidePanel" class="side-panel" aria-hidden="true">
                            <div class="side-panel-header">
                                <div style="min-width:0;flex:1;">
                                    <div id="worldSidePanelTitle" class="side-panel-title">—</div>
                                    <div id="worldSidePanelMeta" class="side-panel-meta"></div>
                                </div>
                                <button type="button" class="side-panel-close" id="worldSidePanelClose" aria-label="Close">×</button>
                            </div>
                            <div class="side-panel-toolbar">
                                <div class="view-toggle">
                                    <button type="button" class="is-active" data-world-pane="bom">BOM</button>
                                    <button type="button" data-world-pane="lineage">Lineage &amp; queries</button>
                                </div>
                            </div>
                            <div class="side-panel-body" id="worldPaneBom" data-world-pane-body="bom">
                                <div id="worldSidePanelFlat" class="flat-bom"></div>
                            </div>
                            <div class="side-panel-body" id="worldPaneLineage" data-world-pane-body="lineage" style="display:none;">
                                <div id="worldLineageResults"></div>
                            </div>
                        </aside>
                    </div>
                </div>
            </div>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsTab -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/web/templates/index.html tests/test_web_ui_features.py
git commit -m "feat(web): worldofBOMs tab scaffold (HTML)"
```

---

### Task C2: Global graph render

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`
- Test: manual browser check

- [ ] **Step 1: Add the worldofBOMs JS module**

In `index.html`, inside the main `<script>` block (after `renderRecursiveGraph` and its helpers, ~line 6475), add the worldofBOMs controller. Start with state, layout config, node-kind colours, and the global renderer:

```javascript
        // ===== worldofBOMs tab =====================================
        let worldCy = null;
        let worldView = 'global';        // 'global' | 'ego'
        let worldDirection = 'both';     // 'both' | 'up' | 'down'
        let worldFocusIri = null;
        let worldFullGraph = null;       // cached {nodes, edges}

        const WORLD_KIND_COLOR = {
            Model:   { bg: '#e6ecff', border: '#3451b2', fg: '#0d1117' },
            Dataset: { bg: '#fff1e5', border: '#f0b079', fg: '#5a2900' },
            Paper:   { bg: '#eafaef', border: '#1a7f37', fg: '#04260f' },
            CodeRepo:{ bg: '#f3eefc', border: '#8250df', fg: '#1f0a3d' },
            Artifact:{ bg: '#ffffff', border: '#d0d7de', fg: '#0d1117' },
        };

        function _worldGraphStyle() {
            return [
                { selector: 'node', style: {
                    'shape': 'round-rectangle', 'label': 'data(label)',
                    'background-color': 'data(bg)', 'border-color': 'data(border)',
                    'color': 'data(fg)', 'border-width': 1,
                    'font-family': "'IBM Plex Sans', system-ui, sans-serif",
                    'font-size': 12, 'text-valign': 'center', 'text-halign': 'center',
                    'text-wrap': 'wrap', 'text-max-width': '150px', 'padding': '10px',
                    'width': 'label', 'height': 'label',
                    'min-width': '80px', 'min-height': '28px' } },
                { selector: 'node[isPlaceholder = "true"]', style: {
                    'background-opacity': 0, 'border-style': 'dashed',
                    'border-color': '#9aa4b2', 'color': '#6e7781' } },
                { selector: 'node[isFocus = "true"]', style: {
                    'border-width': 4, 'border-color': '#d99a00' } },
                { selector: 'node.dimmed', style: { 'opacity': 0.25 } },
                { selector: 'edge', style: {
                    'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
                    'line-color': '#3451b2', 'target-arrow-color': '#3451b2',
                    'width': 1.5, 'label': 'data(predicate)',
                    'font-family': "'IBM Plex Mono', monospace", 'font-size': 9,
                    'color': '#57606a', 'text-background-color': '#fff',
                    'text-background-opacity': 1, 'text-background-padding': '2px',
                    'text-rotation': 'autorotate' } },
                { selector: 'edge[predicate = "testedOn"]', style: {
                    'line-color': '#1a7f37', 'target-arrow-color': '#1a7f37' } },
                { selector: 'edge[predicate = "dependsOn"]', style: {
                    'line-color': '#6e7781', 'target-arrow-color': '#6e7781',
                    'line-style': 'dashed' } },
                { selector: 'edge[predicate = "potentialDuplicateOf"]', style: {
                    'line-color': '#d99a00', 'target-arrow-color': '#d99a00',
                    'line-style': 'dotted' } },
                { selector: 'edge.dimmed', style: { 'opacity': 0.12 } },
            ];
        }

        function _worldElements(graph) {
            // Disambiguate label clashes with a platform-ish badge.
            const labelCounts = {};
            graph.nodes.forEach(n => { labelCounts[n.label] = (labelCounts[n.label] || 0) + 1; });
            const nodes = graph.nodes.map(n => {
                const c = WORLD_KIND_COLOR[n.kind] || WORLD_KIND_COLOR.Artifact;
                let label = n.label;
                if (labelCounts[n.label] > 1) label = `${n.label}\n(${n.kind})`;
                return { data: {
                    id: n.iri, label, kind: n.kind,
                    isPlaceholder: String(!!n.is_placeholder),
                    isFocus: 'false',
                    bg: c.bg, border: c.border, fg: c.fg } };
            });
            const edges = graph.edges.map((e, i) => ({ data: {
                id: `we${i}`, source: e.source, target: e.target,
                predicate: e.predicate } }));
            return { nodes, edges };
        }

        async function loadWorldGraph() {
            const empty = document.getElementById('worldGraphEmpty');
            try {
                const resp = await fetch('/worldofboms/graph');
                worldFullGraph = await resp.json();
            } catch (e) {
                worldFullGraph = { nodes: [], edges: [], store_unavailable: true };
            }
            if (!worldFullGraph.nodes || worldFullGraph.nodes.length === 0) {
                empty.style.display = '';
                empty.textContent = worldFullGraph.store_unavailable
                    ? 'Graph store unavailable.'
                    : 'No BOMs in the graph yet — generate one to start your worldofBOMs.';
                if (worldCy) { try { worldCy.destroy(); } catch (_) {} worldCy = null; }
                return;
            }
            empty.style.display = 'none';
            worldView = 'global';
            worldFocusIri = null;
            renderWorldGraph(_worldElements(worldFullGraph));
            refreshWorldStats();
        }

        function renderWorldGraph(elements) {
            const canvas = document.getElementById('worldGraphCanvas');
            if (worldCy) { try { worldCy.destroy(); } catch (_) {} }
            worldCy = cytoscape({
                container: canvas, elements, wheelSensitivity: 0.2,
                minZoom: 0.05, maxZoom: 3.0, style: _worldGraphStyle(),
                layout: { name: 'cose', animate: false, padding: 28,
                          nodeOverlap: 24, idealEdgeLength: 90, fit: true },
            });
            worldCy.on('tap', 'node', (evt) => openWorldEgo(evt.target.id()));
            worldCy.on('tap', (evt) => { if (evt.target === worldCy) closeWorldSidePanel(); });
            setTimeout(() => { try { worldCy.resize(); worldCy.fit(undefined, 28); }
                               catch (_) {} }, 80);
        }

        async function refreshWorldStats() {
            try {
                const s = await (await fetch('/worldofboms/stats')).json();
                document.getElementById('worldStats').textContent =
                    `${s.artifacts || 0} artifacts · ${s.edges || 0} edges · ${s.claims || 0} claims`;
            } catch (_) { /* ignore */ }
        }
```

(`openWorldEgo`, `closeWorldSidePanel` are defined in C3/C4. Defining functions used before declaration is fine — they are hoisted / called only on user interaction.)

- [ ] **Step 2: Load the graph when the tab is opened**

In the `switchTab` function (line ~4448), add a tab-specific hook. Change `switchTab` to:

```javascript
        function switchTab(ev, tabName) {
            currentTab = tabName;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            (ev.currentTarget || ev.target).classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabName + 'Tab').classList.add('active');
            if (tabName === 'worldofboms') { _wireWorldControls(); loadWorldGraph(); }
        }
```

Apply the same `if (tabName === 'worldofboms')` line to `switchTabByName` (line ~4471) right before its closing brace.

- [ ] **Step 3: Manual verification**

Run the app: `python -m aikaboom.web.app` (or `aikaboom serve`). Generate at least one BOM so the graph is non-empty, open the worldofBOMs tab. Expected: nodes render, edges between a model and its datasets are visible, the stats line shows counts.

- [ ] **Step 4: Run the smoke test + commit**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsTab -v`
Expected: PASS.

```bash
git add src/aikaboom/web/templates/index.html
git commit -m "feat(web): worldofBOMs global graph render"
```

---

### Task C3: Ego view + direction control

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`

- [ ] **Step 1: Add ego-view functions**

Append to the worldofBOMs JS module in `index.html`:

```javascript
        async function openWorldEgo(artifactIri) {
            worldFocusIri = artifactIri;
            worldView = 'ego';
            document.getElementById('worldEgoToggle').disabled = false;
            document.querySelectorAll('#worldViewToggle button').forEach(b =>
                b.classList.toggle('is-active', b.dataset.worldView === 'ego'));
            let ego;
            try {
                ego = await (await fetch(
                    `/worldofboms/ego/${encodeURIComponent(artifactIri)}?direction=${worldDirection}`
                )).json();
            } catch (e) { ego = { nodes: [], edges: [], focus: artifactIri }; }
            const els = _worldElements(ego);
            els.nodes.forEach(n => { if (n.data.id === artifactIri) n.data.isFocus = 'true'; });
            renderWorldGraph(els);
            // dagre layout reads better for a directional ego view.
            try {
                worldCy.layout({ name: 'dagre', rankDir: 'LR', nodeSep: 40,
                                 rankSep: 80, fit: true, padding: 28 }).run();
            } catch (_) { /* dagre optional */ }
            openWorldSidePanel(artifactIri);
        }

        function showWorldGlobal() {
            worldView = 'global';
            worldFocusIri = null;
            document.querySelectorAll('#worldViewToggle button').forEach(b =>
                b.classList.toggle('is-active', b.dataset.worldView === 'global'));
            closeWorldSidePanel();
            if (worldFullGraph) renderWorldGraph(_worldElements(worldFullGraph));
        }

        function setWorldDirection(dir) {
            worldDirection = dir;
            document.querySelectorAll('#worldDirection button').forEach(b =>
                b.classList.toggle('is-active', b.dataset.worldDir === dir));
            if (worldView === 'ego' && worldFocusIri) {
                openWorldEgo(worldFocusIri);   // re-fetch with the new direction
                loadWorldLineage();            // refresh the lineage tab (C5)
            }
        }
```

- [ ] **Step 2: Wire the control buttons**

Append the `_wireWorldControls` function (called from `switchTab`):

```javascript
        function _wireWorldControls() {
            const once = (id, fn) => {
                const el = document.getElementById(id);
                if (el && !el.dataset.wired) { el.addEventListener('click', fn);
                                               el.dataset.wired = '1'; }
            };
            document.querySelectorAll('#worldViewToggle button').forEach(b => {
                if (b.dataset.wired) return;
                b.addEventListener('click', () => {
                    if (b.dataset.worldView === 'global') showWorldGlobal();
                    else if (worldFocusIri) openWorldEgo(worldFocusIri);
                });
                b.dataset.wired = '1';
            });
            document.querySelectorAll('#worldDirection button').forEach(b => {
                if (b.dataset.wired) return;
                b.addEventListener('click', () => setWorldDirection(b.dataset.worldDir));
                b.dataset.wired = '1';
            });
            once('worldZoomIn',  () => worldCy && worldCy.zoom(
                { level: Math.min(worldCy.zoom() * 1.25, worldCy.maxZoom()),
                  renderedPosition: { x: worldCy.width()/2, y: worldCy.height()/2 } }));
            once('worldZoomOut', () => worldCy && worldCy.zoom(
                { level: Math.max(worldCy.zoom() / 1.25, worldCy.minZoom()),
                  renderedPosition: { x: worldCy.width()/2, y: worldCy.height()/2 } }));
            once('worldFit', () => { if (worldCy) { worldCy.resize();
                                                    worldCy.fit(undefined, 28); } });
            once('worldReset', showWorldGlobal);
            once('worldFull', () => {
                const wrap = document.getElementById('worldGraphWrap');
                const full = !wrap.classList.contains('is-fullscreen');
                wrap.classList.toggle('is-fullscreen', full);
                document.body.classList.toggle('has-graph-fullscreen', full);
                setTimeout(() => { if (worldCy) { worldCy.resize();
                                                  worldCy.fit(undefined, 28); } }, 40);
            });
            const search = document.getElementById('worldSearch');
            if (search && !search.dataset.wired) {
                search.addEventListener('input', () => {
                    if (!worldCy) return;
                    const q = search.value.trim().toLowerCase();
                    if (!q) { worldCy.nodes().removeClass('dimmed'); return; }
                    worldCy.nodes().forEach(n => {
                        const hit = n.data('label').toLowerCase().includes(q);
                        n.toggleClass('dimmed', !hit);
                        if (hit) worldCy.animate({ center: { eles: n } },
                                                 { duration: 250 });
                    });
                });
                search.dataset.wired = '1';
            }
        }
```

- [ ] **Step 3: Manual verification**

Open the worldofBOMs tab, click a node. Expected: the view switches to the ego subgraph, the focus node has an amber ring, the Direction buttons re-fetch upstream/downstream/both, and the side panel opens. The search box dims non-matching nodes.

- [ ] **Step 4: Commit**

```bash
git add src/aikaboom/web/templates/index.html
git commit -m "feat(web): worldofBOMs ego view + direction + controls"
```

---

### Task C4: Side panel — BOM tab

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`

- [ ] **Step 1: Add side-panel functions**

Append to the worldofBOMs JS module:

```javascript
        async function openWorldSidePanel(artifactIri) {
            const panel = document.getElementById('worldSidePanel');
            const node = (worldFullGraph?.nodes || []).find(n => n.iri === artifactIri);
            document.getElementById('worldSidePanelTitle').textContent =
                node ? node.label : artifactIri;
            document.getElementById('worldSidePanelMeta').textContent =
                node ? `${node.kind} · ${node.claim_count} claim(s)` : '';
            // BOM pane
            const flat = document.getElementById('worldSidePanelFlat');
            flat.innerHTML = '<div class="json-placeholder">Loading BOM…</div>';
            try {
                const bom = await (await fetch(
                    `/worldofboms/bom/${encodeURIComponent(artifactIri)}`)).json();
                if (bom.error || bom.store_unavailable) {
                    flat.innerHTML = '<div class="json-placeholder">No BOM stored for this node (placeholder).</div>';
                } else {
                    renderBOM(bom, flat);
                }
            } catch (e) {
                flat.innerHTML = '<div class="json-placeholder">Failed to load BOM.</div>';
            }
            // default to the BOM pane
            _showWorldPane('bom');
            panel.classList.add('is-open');
            panel.setAttribute('aria-hidden', 'false');
            setTimeout(() => { if (worldCy) { worldCy.resize();
                                              worldCy.fit(undefined, 28); } }, 220);
            loadWorldLineage();   // pre-load the lineage pane (C5)
        }

        function closeWorldSidePanel() {
            const panel = document.getElementById('worldSidePanel');
            if (!panel) return;
            panel.classList.remove('is-open');
            panel.setAttribute('aria-hidden', 'true');
            setTimeout(() => { if (worldCy) { worldCy.resize();
                                              worldCy.fit(undefined, 28); } }, 220);
        }

        function _showWorldPane(name) {
            document.querySelectorAll('#worldSidePanel [data-world-pane]').forEach(b =>
                b.classList.toggle('is-active', b.dataset.worldPane === name));
            document.getElementById('worldPaneBom').style.display =
                name === 'bom' ? '' : 'none';
            document.getElementById('worldPaneLineage').style.display =
                name === 'lineage' ? '' : 'none';
        }
```

- [ ] **Step 2: Wire the close button and pane toggle**

Add to the end of `_wireWorldControls` (before its closing brace):

```javascript
            once('worldSidePanelClose', closeWorldSidePanel);
            document.querySelectorAll('#worldSidePanel [data-world-pane]').forEach(b => {
                if (b.dataset.wired) return;
                b.addEventListener('click', () => _showWorldPane(b.dataset.worldPane));
                b.dataset.wired = '1';
            });
```

- [ ] **Step 3: Manual verification**

Click a node → the side panel shows that node's reconstructed BOM under the BOM tab. Placeholder nodes show "No BOM stored". The BOM / Lineage toggle switches panes.

- [ ] **Step 4: Commit**

```bash
git add src/aikaboom/web/templates/index.html
git commit -m "feat(web): worldofBOMs side panel — BOM pane"
```

---

### Task C5: Lineage pane — preset queries + raw SPARQL

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`

- [ ] **Step 1: Add the lineage-pane renderer**

Append to the worldofBOMs JS module:

```javascript
        const WORLD_PRESETS = [
            { id: 'licenses', label: 'Licenses across the lineage' },
            { id: 'datasets', label: 'All datasets in the lineage' },
            { id: 'models',   label: 'All models in the lineage' },
            { id: 'conflicts',label: 'Conflicts anywhere in the lineage' },
        ];

        function loadWorldLineage() {
            const root = document.getElementById('worldLineageResults');
            if (!worldFocusIri) { root.innerHTML =
                '<div class="json-placeholder">Click a node to see its lineage.</div>';
                return; }
            const buttons = WORLD_PRESETS.map(p =>
                `<button class="mock-button world-preset" data-preset="${p.id}"
                  style="display:block;width:100%;text-align:left;margin:4px 0;font-size:12px;">
                  ▸ ${p.label}</button>`).join('');
            root.innerHTML = `
                <div style="font-size:11.5px;color:var(--paper-mute);margin-bottom:6px;">
                  Scope: <strong>${worldDirection}</strong> — change with the Direction control.</div>
                ${buttons}
                <div id="worldQueryResult" style="margin-top:8px;"></div>
                <details style="margin-top:12px;">
                  <summary style="font-size:12px;cursor:pointer;">Advanced: SPARQL</summary>
                  <textarea id="worldSparql" rows="4"
                    style="width:100%;font-family:'IBM Plex Mono',monospace;font-size:11px;margin-top:6px;"
                    placeholder="SELECT ?s WHERE { ?s a ?c } LIMIT 20"></textarea>
                  <button class="mock-button" id="worldSparqlRun"
                    style="font-size:11.5px;margin-top:4px;">Run query</button>
                </details>`;
            root.querySelectorAll('.world-preset').forEach(b =>
                b.addEventListener('click', () => runWorldPreset(b.dataset.preset)));
            const runBtn = document.getElementById('worldSparqlRun');
            runBtn.addEventListener('click', runWorldSparql);
        }

        function _renderWorldRows(rows) {
            const out = document.getElementById('worldQueryResult');
            if (!rows || rows.length === 0) {
                out.innerHTML = '<div class="json-placeholder">No results.</div>';
                return;
            }
            const cols = Object.keys(rows[0]);
            const head = cols.map(c => `<th style="text-align:left;padding:3px 6px;">${c}</th>`).join('');
            const body = rows.map(r =>
                '<tr>' + cols.map(c =>
                    `<td style="padding:3px 6px;border-top:1px solid #eee;">${
                        escapeHtml(String(r[c] ?? ''))}</td>`).join('') + '</tr>').join('');
            out.innerHTML = `<table style="width:100%;border-collapse:collapse;font-size:11.5px;">
                <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
        }

        async function runWorldPreset(preset) {
            const out = document.getElementById('worldQueryResult');
            out.innerHTML = '<div class="json-placeholder">Running…</div>';
            try {
                const resp = await fetch('/worldofboms/query', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ preset, artifact: worldFocusIri,
                                           direction: worldDirection }) });
                const data = await resp.json();
                if (data.error) { out.innerHTML =
                    `<div class="json-placeholder">${escapeHtml(data.error)}</div>`; return; }
                _renderWorldRows(data.rows);
            } catch (e) {
                out.innerHTML = '<div class="json-placeholder">Query failed.</div>';
            }
        }

        async function runWorldSparql() {
            const out = document.getElementById('worldQueryResult');
            const sparql = document.getElementById('worldSparql').value.trim();
            if (!sparql) return;
            out.innerHTML = '<div class="json-placeholder">Running…</div>';
            try {
                const resp = await fetch('/worldofboms/query', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sparql }) });
                const data = await resp.json();
                if (data.error) { out.innerHTML =
                    `<div class="json-placeholder">${escapeHtml(data.error)}</div>`; return; }
                _renderWorldRows(data.rows);
            } catch (e) {
                out.innerHTML = '<div class="json-placeholder">Query failed.</div>';
            }
        }
```

- [ ] **Step 2: Manual verification**

Open a node's side panel → Lineage & queries tab → click each preset; results render as a table. Expand "Advanced: SPARQL", run `SELECT ?s WHERE { ?s a ?c } LIMIT 5` → table appears. A `DELETE` query shows the rejection error.

- [ ] **Step 3: Commit**

```bash
git add src/aikaboom/web/templates/index.html
git commit -m "feat(web): worldofBOMs lineage pane — presets + raw SPARQL"
```

---

### Task C6: Download dropdown

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`

- [ ] **Step 1: Add download wiring**

Append to the worldofBOMs JS module:

```javascript
        function _worldExportUrl(scope) {
            if (scope === 'ego' && worldFocusIri) {
                return `/worldofboms/export?scope=ego&artifact=${
                    encodeURIComponent(worldFocusIri)}&direction=${worldDirection}`;
            }
            return '/worldofboms/export?scope=full';
        }

        function _wireWorldDownload() {
            const link = document.getElementById('worldDownload');
            if (!link || link.dataset.wired) return;
            link.addEventListener('click', (ev) => {
                ev.preventDefault();
                const inEgo = worldView === 'ego' && worldFocusIri;
                const choice = inEgo
                    ? window.confirm(
                        'OK = download this ego view (' + worldDirection +
                        ') as SPDX.\nCancel = download the whole graph.')
                    : false;
                window.location.href = _worldExportUrl(
                    (inEgo && choice) ? 'ego' : 'full');
            });
            link.dataset.wired = '1';
        }
```

Add `_wireWorldDownload();` to the end of `_wireWorldControls`.

> **Note:** This uses a confirm() to choose ego-vs-full. If the codebase's download-dropdown component (PR #45, search `download` dropdown markup in `index.html`) is easy to reuse, prefer wiring `#worldDownload` as a real dropdown with explicit "Whole graph" / "This ego view ▾ (upstream / downstream / both)" items pointing at `_worldExportUrl(...)`. The confirm() is the minimal acceptable fallback.

- [ ] **Step 2: Manual verification**

In global view, click Download → the whole-graph SPDX file downloads. In ego view, click Download → choose ego → the ego SPDX file downloads. Open both files; each has `@context` and `@graph`.

- [ ] **Step 3: Commit**

```bash
git add src/aikaboom/web/templates/index.html
git commit -m "feat(web): worldofBOMs SPDX download (whole graph / ego view)"
```

---

### Task C7: Post-generation refresh + final polish

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`
- Test: `tests/test_web_ui_features.py`

- [ ] **Step 1: Write the failing test**

Add to `TestWorldOfBomsTab` in `tests/test_web_ui_features.py`:

```python
    def test_worldofboms_tab_has_lineage_and_direction_controls(self, client):
        html = client.get("/").get_data(as_text=True)
        assert 'id="worldDirection"' in html
        assert 'data-world-pane="lineage"' in html
        assert 'id="worldDownload"' in html
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_web_ui_features.py::TestWorldOfBomsTab::test_worldofboms_tab_has_lineage_and_direction_controls -v`
Expected: PASS already (markup added in C1) — this locks the contract. If it fails, the C1 markup is incomplete; fix it.

- [ ] **Step 3: Refresh the graph after a generation completes**

In `index.html`, find `refocusAfterGeneration()` being called after generation (line ~3971, end of the result-handling block). Immediately after that call, add:

```javascript
                    // Keep the worldofBOMs graph fresh — a new BOM just grew it.
                    if (typeof worldFullGraph !== 'undefined') {
                        worldFullGraph = null;
                        if (document.getElementById('worldofbomsTab')
                              ?.classList.contains('active')) {
                            loadWorldGraph();
                        }
                    }
```

- [ ] **Step 4: Manual verification**

With the worldofBOMs tab open, generate a new BOM in the Generate tab. Return to worldofBOMs — the new artifact and its edges appear. Generate a model whose `trainedOnDatasets` names an artifact already in the graph — confirm it connects to the existing node rather than adding a duplicate.

- [ ] **Step 5: Run the full web test suite + commit**

Run: `pytest tests/test_web_ui_features.py -v`
Expected: PASS.

```bash
git add src/aikaboom/web/templates/index.html tests/test_web_ui_features.py
git commit -m "feat(web): refresh worldofBOMs graph after each generation"
```

---

## Part D — Documentation

### Task D1: worldofBOMs visualization docs

**Files:**
- Create: `docs/worldofboms/VISUALIZATION.md`
- Modify: `docs/worldofboms/CONCEPT.md`
- Modify: `docs/superpowers/specs/2026-05-14-worldofboms-graph-design.md`
- Test: `tests/store/test_docs_link_check.py`

- [ ] **Step 1: Create `docs/worldofboms/VISUALIZATION.md`**

Write a reference covering: the `worldofBOMs` tab; global vs ego view; the Direction control (upstream = dependencies, downstream = dependents, both); the four preset lineage queries and the raw SPARQL box; SPDX linked-bundle download (whole graph / ego); how relationship edges (`trainedOn`/`testedOn`/`dependsOn`) are created at save time; placeholder nodes and `potentialDuplicateOf` hints (referencing spec sections A.3–A.6). Keep it consistent in tone with the other `docs/worldofboms/*.md` files.

- [ ] **Step 2: Remove "graph visualizer" from the non-goals**

In `docs/worldofboms/CONCEPT.md`, in the "What the system is and isn't" section, remove `a graph visualizer,` from the `is not (in v1)` sentence. Add a short line pointing to `VISUALIZATION.md`.

In `docs/superpowers/specs/2026-05-14-worldofboms-graph-design.md`, remove the `- Browser graph visualizer.` line from **both** Non-Goals lists (there are two — near line 30 and near line 620), or replace with `- Browser graph visualizer — delivered in the 2026-05-15 follow-up spec.`

- [ ] **Step 3: Run the docs tests**

Run: `pytest tests/store/test_docs_link_check.py tests/store/test_docs_schema_parity.py -v`
Expected: PASS (no broken links introduced; schema parity still holds).

- [ ] **Step 4: Commit**

```bash
git add docs/worldofboms/VISUALIZATION.md docs/worldofboms/CONCEPT.md docs/superpowers/specs/2026-05-14-worldofboms-graph-design.md
git commit -m "docs(worldofboms): visualization reference; retire viz non-goal"
```

---

## Final Verification

- [ ] **Run the full store + web suites**

Run: `pytest tests/store/ tests/test_web_ui_features.py -v`
Expected: all PASS.

- [ ] **Run the whole test suite for regressions**

Run: `pytest -q`
Expected: no new failures versus the pre-change baseline.

- [ ] **GitNexus change check**

Run `gitnexus_detect_changes()` (if available) and confirm the affected symbols match this plan's scope.

- [ ] **Manual end-to-end**

Start the app, generate a model BOM with recursive mode off whose `trainedOnDatasets` names a known dataset, then a fresh model that names the same dataset. Open worldofBOMs: confirm one dataset node with two incoming edges, ego view + direction work, preset queries return rows, and the SPDX download opens.

---

## Self-Review Notes (plan author)

- **Spec coverage:** A.1→T-A1, A.2→T-A3, A.3→T-A3/A4/A5, A.4→T-A4/A7, A.5→T-A6/A9, A.6→T-A4/A8, B.1→T-B1..B5, B.2→T-B6/B7, B.3→T-B3, C.1→T-C1, C.2→T-C2, C.3→T-C3, C.4→T-C4, C.5→T-C5/C6, C.6→T-C2/C3/C7, Testing→tests in every task, Docs→T-D1. All spec sections map to a task.
- **Type consistency:** `resolve_edge_target` returns `(iri, minted)` — used consistently in A4/A5/A8. `full_graph`/`ego_graph` return `{nodes, edges[, focus]}` with node keys `iri/label/kind/is_placeholder/claim_count` — consumed unchanged by the routes (B6/B7) and the frontend (`_worldElements`). `merge_artifacts(into, from_)` signature is identical in A2 (definition), A7 (caller), and the CLI delegation.
- **Known soft spot:** Task B5 (`ego_spdx_bundle`) depends on the exact child-dict shape `build_linked_spdx_bundle` expects (`spdx_data` vs `metadata`); the implementer note in B5 Step 3 tells the engineer to adjust against the Step-2 test until `@graph` is non-empty.
