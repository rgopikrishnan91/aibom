# worldofBOMs PR #48 — three followups — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land three usability/verification followups on the `worldofboms-graph` branch so PR #48 ships clean: (1) an "open a past BOM" affordance on the splash, (2) an end-to-end test proving the worldofBOMs graph reuses existing nodes on insert, (3) replace the raw SPARQL textarea with four scoped action buttons + one new rebuild endpoint.

**Architecture:**
- All work commits onto `worldofboms-graph` (PR #48 stays the single landing PR for this work — confirmed by the user).
- Order is B → A → C: the e2e test runs first because it tells us whether the store-layer reuse claims hold under the public web path before we spend cycles on UI changes.
- Spec: `docs/superpowers/specs/2026-05-20-worldofboms-followups-design.md`.

**Tech Stack:**
- Backend: Flask (Python 3.12), rdflib-backed BomStore.
- Frontend: vanilla JS in a single Jinja template (`src/aikaboom/web/templates/index.html`).
- Tests: pytest for store + web routes; pytest + Playwright for UI smoke (skipped if Chromium unavailable).

**Worktree:** `/home/gopi/aikaboom/aibom/.claude/worktrees/worldofboms-spec` (already on `worldofboms-graph`).

---

## File structure (what changes where)

| File | Section | Purpose |
|---|---|---|
| `tests/store/test_e2e_reuse_via_process.py` | B | NEW — 3 scenarios: identical-model reuse, cross-identifier reuse, edge-target dedup. Drives `/process` via Flask test client. |
| `src/aikaboom/web/templates/index.html` | A | EDIT — add splash hint anchor + `_refreshSplashHistoryHint()` JS hook. |
| `tests/test_splash_history_hint.py` | A | NEW — Playwright smoke (skipped without Chromium) for the splash hint. |
| `src/aikaboom/web/app.py` | C | EDIT — drop sparql branch in `/worldofboms/query`; add `POST /worldofboms/rebuild`. |
| `src/aikaboom/web/templates/index.html` | C | EDIT — rewrite `loadWorldLineage()`: remove SPARQL textarea + handler, add four action buttons with scope-chooser for "Generate BOM". |
| `tests/store/test_worldofboms_rebuild.py` | C | NEW — exercises `POST /worldofboms/rebuild` against a temp `bom-history/` with two fixture rows. |
| `tests/store/test_worldofboms_query_no_sparql.py` | C | NEW — asserts `POST /worldofboms/query` with a `sparql:` body now ignores it (or 400s) and only the preset branch is reachable. |
| `tests/test_world_actions_smoke.py` | C | NEW — Playwright smoke that the four buttons exist, the SPARQL `<details>` is gone, and clicking "Open this BOM" rehydrates the main viewer. |

No store-layer code changes are expected. If Section B scenarios fail, the resulting fix may add code in `src/aikaboom/store/`; that fix is out-of-plan and gets its own task added inline.

---

## Section B — End-to-end node-reuse verification

### Task B1: Test fixture scaffold for e2e reuse via `/process`

**Files:**
- Create: `tests/store/test_e2e_reuse_via_process.py`

This test exercises the public `/process` Flask route twice and asserts the second insert reuses the first artifact. To avoid hitting the LLM/network, we monkeypatch `get_processor` with a fake that returns a deterministic BOM dict — same pattern used in `tests/store/test_web_resolve.py`. The store is real (`AIKABOOM_GRAPH_DISABLE=0`), backed by an rdflib in-process graph rooted in `tmp_store_dir`.

- [ ] **Step 1: Write the failing scaffold + scenario 1 (identical-model reuse)**

Create `tests/store/test_e2e_reuse_via_process.py`:

```python
"""End-to-end: a second /process for the same model reuses the first artifact.

The store's de-dup story is exercised piecewise in test_multi_identifier_dedup,
test_store_resolve, test_edges, etc. This test closes the loop by driving the
public Flask /process route twice and asserting BomStore.stats() shows the
expected node counts (no duplicates).

Per the worldofBOMs followups spec
(docs/superpowers/specs/2026-05-20-worldofboms-followups-design.md), section B.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A Flask test client with a real (per-test) graph store on disk."""
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    # Skip the link-fallback agent — no Gemini key in tests, and we don't
    # want the route's exception path muddying the assertions.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Re-route bom-history so the test doesn't pollute the repo copy.
    history_dir = tmp_path / "bom-history"
    history_dir.mkdir()

    from aikaboom.web.app import app

    app.config["TESTING"] = True
    app.config["HISTORY_FOLDER"] = str(history_dir)
    app.config["HISTORY_INDEX"] = str(history_dir / "index.json")
    return app.test_client()


@pytest.fixture
def fake_ai_processor(monkeypatch):
    """Patch get_processor so /process never calls an LLM.

    Returns a callable that takes the repo_id to embed in the fake BOM, so
    each scenario can pin its own model name.
    """
    from aikaboom.web import app as appmod

    class _FakeAIProc:
        use_case = "license"

        def __init__(self, repo_id: str):
            self._repo_id = repo_id

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": self._repo_id,
                "model_id": self._repo_id.replace("/", "_"),
                "use_case": "license",
                "direct_fields": {
                    "license": {
                        "value": "Apache-2.0",
                        "source": "huggingface",
                        "conflict": None,
                    },
                },
                "rag_fields": {},
                "beta_fields": [],
            }

    def install(repo_id: str):
        monkeypatch.setattr(
            appmod, "get_processor",
            lambda **_kw: _FakeAIProc(repo_id),
        )

    return install


def _post_process(client, repo_id: str):
    """POST /process for an AI BOM with a single repo_id identifier."""
    resp = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "repo_id": repo_id,
            "use_case": "license",
            "mode": "rag",
            "skip_fallback": True,  # avoid GEMINI_API_KEY path
            "validate_spdx": False,
        },
        content_type="application/json",
    )
    return resp


def _open_store():
    """Open the live BomStore against the env-configured backend."""
    from aikaboom.store.store import BomStore
    return BomStore.open()


def test_identical_model_reuses_artifact(client, fake_ai_processor):
    """Two /process calls for the same repo_id → 1 artifact, 2 claims."""
    fake_ai_processor("mistralai/Mistral-7B-v0.1")

    r1 = _post_process(client, "mistralai/Mistral-7B-v0.1")
    assert r1.status_code == 200, r1.get_data(as_text=True)

    r2 = _post_process(client, "mistralai/Mistral-7B-v0.1")
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    assert stats["artifacts"] == 1, f"expected 1 artifact, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"
```

- [ ] **Step 2: Run the scenario-1 test and confirm it fails for the right reason if it does**

Run:

```bash
pytest tests/store/test_e2e_reuse_via_process.py::test_identical_model_reuses_artifact -xvs
```

Expected outcomes:
- **PASS** → store-level reuse holds end-to-end. Proceed to step 3.
- **FAIL with `artifacts == 2`** → real bug: the second insert is making a fresh artifact. STOP and inspect `_try_resolve_cache` + the `_store.save_claim` call site in `src/aikaboom/web/app.py` around lines 860–920. Open an out-of-plan investigation before writing scenarios 2/3.
- **FAIL with import errors / 500** → fixture wiring is wrong (likely the monkeypatch order or the fake processor signature). Fix before continuing.

- [ ] **Step 3: Add scenario 2 (cross-identifier reuse) — write it red first**

Append to `tests/store/test_e2e_reuse_via_process.py`:

```python
def test_cross_identifier_reuses_artifact(client, fake_ai_processor):
    """BOM 1 saved with (hf=X, arxiv=Y); BOM 2 with only arxiv=Y → 1 artifact, 2 claims."""
    repo_id = "mistralai/Mistral-7B-v0.1"
    arxiv_url = "https://arxiv.org/abs/2310.06825"

    fake_ai_processor(repo_id)
    r1 = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "repo_id": repo_id,
            "arxiv_url": arxiv_url,
            "use_case": "license", "mode": "rag",
            "skip_fallback": True, "validate_spdx": False,
        },
        content_type="application/json",
    )
    assert r1.status_code == 200, r1.get_data(as_text=True)

    # Second run — same fake processor, but the request body omits repo_id
    # so only the arxiv identifier is supplied. The store's resolve() must
    # still find the artifact saved above.
    r2 = client.post(
        "/process",
        json={
            "bom_type": "ai",
            "arxiv_url": arxiv_url,
            "use_case": "license", "mode": "rag",
            "skip_fallback": True, "validate_spdx": False,
        },
        content_type="application/json",
    )
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    assert stats["artifacts"] == 1, f"expected 1 artifact, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"
```

Run:

```bash
pytest tests/store/test_e2e_reuse_via_process.py::test_cross_identifier_reuses_artifact -xvs
```

- **PASS** → cross-identifier reuse confirmed end-to-end. Proceed.
- **FAIL** → likely the fake processor's BOM doesn't carry the arxiv reference back through `save_claim`'s identifier path. Add an out-of-plan task to inspect — but first re-read `_try_resolve_cache` to confirm identifiers from the request body (not the processor output) are what gets passed to `save_claim`.

- [ ] **Step 4: Add scenario 3 (edge-target dedup) — write it red first**

Append:

```python
def test_dependency_edge_reuses_existing_artifact(client, fake_ai_processor, monkeypatch):
    """BOM A saved for model M; BOM B for M' lists M as trainedOn → edge points to M's IRI, no duplicate."""
    from aikaboom.web import app as appmod

    # First run: stash artifact for the dependency target.
    fake_ai_processor("upstream/teacher-model")
    r1 = _post_process(client, "upstream/teacher-model")
    assert r1.status_code == 200, r1.get_data(as_text=True)

    # Second run: BOM for the student model that lists the teacher in its
    # trainedOnDatasets field. Reinstall the fake processor to return a BOM
    # whose direct_fields carry the relationship target.
    class _StudentProc:
        use_case = "license"

        def process_ai_model(self, **_kwargs):
            return {
                "repo_id": "downstream/student-model",
                "model_id": "downstream_student-model",
                "use_case": "license",
                "direct_fields": {
                    "license": {"value": "Apache-2.0", "source": "huggingface", "conflict": None},
                    "trainedOnDatasets": {
                        "value": "upstream/teacher-model",
                        "source": "huggingface",
                        "conflict": None,
                    },
                },
                "rag_fields": {},
                "beta_fields": [],
            }

    monkeypatch.setattr(appmod, "get_processor", lambda **_kw: _StudentProc())
    r2 = _post_process(client, "downstream/student-model")
    assert r2.status_code == 200, r2.get_data(as_text=True)

    stats = _open_store().stats()
    # 2 artifacts (teacher + student), 2 claims. No third "ghost" artifact
    # for the teacher reference inside the student BOM.
    assert stats["artifacts"] == 2, f"expected 2 artifacts, got {stats}"
    assert stats["claims"] == 2, f"expected 2 claims, got {stats}"

    # And the trainedOn edge points to the teacher's IRI, not a fresh one.
    from aikaboom.store import vocab
    store = _open_store()
    rows = list(store._backend.select(f"""
        SELECT ?src ?tgt WHERE {{
            ?src <{vocab.trainedOn}> ?tgt .
        }}
    """))
    assert len(rows) == 1, f"expected 1 trainedOn edge, got {rows}"
    teacher_iri = rows[0]["tgt"]
    # The teacher IRI must appear as the subject of a hasVersion/hasClaim
    # chain — i.e., it's the teacher artifact we saved in r1, not a
    # placeholder created on the fly.
    has_claims = list(store._backend.select(f"""
        SELECT ?v WHERE {{
            <{teacher_iri}> <{vocab.hasVersion}> ?v .
        }}
    """))
    assert has_claims, f"trainedOn target {teacher_iri} has no version/claim — duplicate artifact"
```

Run:

```bash
pytest tests/store/test_e2e_reuse_via_process.py::test_dependency_edge_reuses_existing_artifact -xvs
```

- **PASS** → edge-target dedup confirmed.
- **FAIL with extra artifact** → the relationship field on the student BOM created a fresh artifact for the teacher instead of resolving to the existing one. Inspect `edges.add_relationship_edges` and `_find_artifact_by_label` — likely the teacher's canonical label didn't match the relationship target string (canon_name normalisation gap). Add an out-of-plan store-layer fix task before continuing.

- [ ] **Step 5: Run all three scenarios together**

Run:

```bash
pytest tests/store/test_e2e_reuse_via_process.py -xvs
```

Expected: 3 passed.

- [ ] **Step 6: Commit Section B**

```bash
git add tests/store/test_e2e_reuse_via_process.py
git commit -m "test(store): e2e — second /process for same model reuses artifact

Three scenarios driving the public /process Flask route to close the
loop on the store's de-dup story:

  1. Identical model_id twice → 1 artifact, 2 claims.
  2. (hf, arxiv) then arxiv-only → 1 artifact, 2 claims (cross-id resolve).
  3. Student BOM with trainedOnDatasets pointing at a prior teacher BOM →
     2 artifacts, 1 trainedOn edge, target is the teacher's real IRI
     (no duplicate ghost artifact).

Per worldofBOMs followups spec section B
(docs/superpowers/specs/2026-05-20-worldofboms-followups-design.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section A — Splash "Open a past BOM" affordance

### Task A1: Add the splash hint anchor + JS hook

**Files:**
- Modify: `src/aikaboom/web/templates/index.html` (splash card around line 3253; History controller `History.upsert` / `History.load`; CSS area around line 920)

- [ ] **Step 1: Write the failing Playwright smoke for the hint**

Create `tests/test_splash_history_hint.py`:

```python
"""Playwright smoke: the splash card shows a link to open past BOMs when
bom-history/ has rows, and hides it when empty.

Skipped if Chromium isn't installed locally — same pattern as
tests/test_pipeline_ui_smoke.py.
"""

import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


def _chromium_available() -> bool:
    cache = Path.home() / ".cache" / "ms-playwright"
    return cache.exists() and any(d.name.startswith("chromium") for d in cache.iterdir())


if not _chromium_available():
    pytest.skip("chromium not installed", allow_module_level=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, timeout_s: float = 30.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="module")
def flask_with_history(tmp_path_factory):
    """Boot Flask with a seeded bom-history dir containing one row."""
    work = tmp_path_factory.mktemp("aibom_splash_hint")
    history = work / "bom-history"
    history.mkdir()
    # One stub artifact + matching index row.
    bom = {"repo_id": "stub/model", "model_id": "stub_model",
           "use_case": "license",
           "direct_fields": {}, "rag_fields": {}, "beta_fields": []}
    (history / "abc12345_model_aibom.json").write_text(json.dumps(bom))
    (history / "index.json").write_text(json.dumps([{
        "hash": "abc12345", "subject": "stub/model", "bom_type": "ai",
        "created_at": "2026-05-20T00:00:00+00:00",
        "artifacts": {"bom": "abc12345_model_aibom.json"},
    }]))

    port = _free_port()
    env = {
        **__import__("os").environ,
        "BOM_HOST": "127.0.0.1", "BOM_PORT": str(port),
        "AIKABOOM_GRAPH_DISABLE": "1",  # don't touch graph for this smoke
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    # Patch the app's HISTORY_FOLDER via an env-var override the harness
    # exposes. If no such env-var exists yet, run the test against the
    # repo's real bom-history (still works as long as it has ≥1 row).
    env["AIKABOOM_HISTORY_DIR"] = str(history)

    proc = subprocess.Popen(
        [sys.executable, "-m", "aikaboom.web.app"],
        env=env, cwd=PROJECT_ROOT,
    )
    if not _wait_for_http(f"http://127.0.0.1:{port}/"):
        proc.kill()
        pytest.fail("flask server didn't start")
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=5)


def test_splash_hint_visible_with_one_history_row(flask_with_history):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(flask_with_history)
        # Wait for History.load() to populate.
        page.wait_for_function("window.History && window.History.rows && window.History.rows.length >= 1")
        # The hint anchor should be visible inside the splash card.
        hint = page.locator("#resultEmpty #splashHistoryHint")
        assert hint.is_visible(), "splash history hint should appear when N>=1"
        assert "past BOM" in hint.inner_text().lower()
        # Clicking it should swap the splash for the History panel.
        hint.click()
        page.wait_for_selector("#historyTab.tab-content.active, .result-pane.is-history-open")
        browser.close()
```

This test needs an `AIKABOOM_HISTORY_DIR` env-var override in `app.py`; if it doesn't exist yet, **add it now** as a one-line change. Locate the `HISTORY_FOLDER` assignment in `src/aikaboom/web/app.py` (line 57):

```python
app.config['HISTORY_FOLDER'] = os.environ.get(
    'AIKABOOM_HISTORY_DIR',
    os.path.join(_PROJECT_ROOT, 'bom-history'),
)
app.config['HISTORY_INDEX'] = os.path.join(app.config['HISTORY_FOLDER'], 'index.json')
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest tests/test_splash_history_hint.py::test_splash_hint_visible_with_one_history_row -xvs
```

Expected: FAIL (`splash history hint should appear when N>=1` — the anchor doesn't exist yet). If Playwright/Chromium isn't installed, the module skips — install with `playwright install chromium` if you want browser coverage; otherwise rely on the manual check in Step 6.

- [ ] **Step 3: Add the splash hint HTML**

In `src/aikaboom/web/templates/index.html`, find the `<div class="splash-hint">` block inside `#resultEmpty` (around line 3253). Add a new line **immediately after** the closing `</div>` of `.splash-hint` and **before** the closing `</div>` of `#resultEmpty`:

```html
                <div class="splash-history-hint" id="splashHistoryHint" hidden>
                    …or open one of your <strong><span id="splashHistoryCount">0</span> past BOMs</strong>
                    <a href="#" class="splash-history-link"
                       onclick="event.preventDefault(); switchTabByName('history');"
                       aria-label="Open History">→</a>
                </div>
```

- [ ] **Step 4: Add the matching CSS**

In the same template, find the `.splash-hint` rule block (around line 920) and **append** below it:

```css
        .splash-history-hint {
            font-family: var(--sans);
            font-size: 12.5px;
            color: var(--paper-mute);
            margin-top: 10px;
            line-height: 1.5;
        }
        .splash-history-hint[hidden] { display: none; }
        .splash-history-hint strong { color: var(--ink); font-weight: 600; }
        .splash-history-link {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            margin-left: 4px;
            padding: 0 4px;
            border-radius: 3px;
        }
        .splash-history-link:hover { background: var(--primary-soft); }
```

- [ ] **Step 5: Wire `_refreshSplashHistoryHint()` into the History controller**

In the template's `<script>` block, find the History controller (search for `History.load` — around line 5256 on the branch). Add a helper near the top of that section, then call it from `History.load` (after the rows are populated) and `History.upsert` (every time a row is added):

```javascript
        function _refreshSplashHistoryHint() {
            const hint = document.getElementById('splashHistoryHint');
            const count = document.getElementById('splashHistoryCount');
            if (!hint || !count) return;
            const n = (window.History && History.rows) ? History.rows.length : 0;
            if (n > 0) {
                count.textContent = String(n);
                hint.hidden = false;
            } else {
                hint.hidden = true;
            }
        }
```

Then locate the body of `History.load = async function ...` and add `_refreshSplashHistoryHint();` as the last statement of its success path (after `this.rows = data.rows || [];`). Locate `History.upsert = function (row) ...` and add `_refreshSplashHistoryHint();` as its last statement.

- [ ] **Step 6: Run the Playwright test until it passes (or do a manual check)**

Run:

```bash
pytest tests/test_splash_history_hint.py::test_splash_hint_visible_with_one_history_row -xvs
```

Expected: PASS.

If Playwright isn't available, do a manual check:

```bash
AIKABOOM_HISTORY_DIR=$(pwd)/bom-history python -m aikaboom.web.app
# In a browser: open http://127.0.0.1:5000. The splash should show the
# "open one of your N past BOMs →" line. Click it; the History panel
# should open in place of the splash.
```

- [ ] **Step 7: Commit Section A**

```bash
git add src/aikaboom/web/templates/index.html src/aikaboom/web/app.py tests/test_splash_history_hint.py
git commit -m "ui(splash): show 'open one of your N past BOMs' affordance

The splash card on a fresh page load and after '+ New BOM' had no
visible bridge to past work — the History tab button was visible
(commit fa532fe) but reads as chrome, not navigation. First-time-after-
return visitors missed it.

Add one quiet line at the bottom of #resultEmpty:
  …or open one of your **N past BOMs** →
where N comes from the already-loaded History.rows. Hidden when N=0.
Clicking it calls switchTabByName('history'), which already toggles
is-history-open on the empty pane so the History panel renders in
place of the splash.

Also adds AIKABOOM_HISTORY_DIR env-var override on the Flask app so
tests can point HISTORY_FOLDER at a temp dir. TDD: Playwright smoke
verifies the hint shows with N>=1 and clicking it opens History.

Per worldofBOMs followups spec section A
(docs/superpowers/specs/2026-05-20-worldofboms-followups-design.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section C — SPARQL textarea → action buttons (+ rebuild endpoint)

### Task C1: Backend — add `POST /worldofboms/rebuild`, gate sparql in `/worldofboms/query`

**Files:**
- Modify: `src/aikaboom/web/app.py` (handler at line 1538; add new route below)
- Create: `tests/store/test_worldofboms_rebuild.py`
- Create: `tests/store/test_worldofboms_query_no_sparql.py`

- [ ] **Step 1: Write the failing test for `POST /worldofboms/rebuild`**

Create `tests/store/test_worldofboms_rebuild.py`:

```python
"""POST /worldofboms/rebuild re-ingests every row in bom-history/index.json.

Idempotent: relying on the same store dedup verified in
tests/store/test_e2e_reuse_via_process.py, calling rebuild twice yields
the same stats as calling it once.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    history = tmp_path / "bom-history"; history.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    monkeypatch.setenv("AIKABOOM_HISTORY_DIR", str(history))

    # Two fixture BOMs with distinct identifiers → 2 artifacts.
    bom_a = {
        "repo_id": "owner-a/model", "model_id": "owner-a_model",
        "use_case": "license",
        "direct_fields": {"license": {"value": "Apache-2.0", "source": "huggingface", "conflict": None}},
        "rag_fields": {}, "beta_fields": [],
    }
    bom_b = {
        "repo_id": "owner-b/model", "model_id": "owner-b_model",
        "use_case": "license",
        "direct_fields": {"license": {"value": "MIT", "source": "huggingface", "conflict": None}},
        "rag_fields": {}, "beta_fields": [],
    }
    (history / "aaa11111_model_aibom.json").write_text(json.dumps(bom_a))
    (history / "bbb22222_model_aibom.json").write_text(json.dumps(bom_b))
    (history / "index.json").write_text(json.dumps([
        {"hash": "aaa11111", "subject": "owner-a/model", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:00+00:00",
         "artifacts": {"bom": "aaa11111_model_aibom.json"}},
        {"hash": "bbb22222", "subject": "owner-b/model", "bom_type": "ai",
         "created_at": "2026-05-20T00:00:01+00:00",
         "artifacts": {"bom": "bbb22222_model_aibom.json"}},
    ]))

    from aikaboom.web.app import app
    app.config["TESTING"] = True
    app.config["HISTORY_FOLDER"] = str(history)
    app.config["HISTORY_INDEX"] = str(history / "index.json")
    return app.test_client()


def test_rebuild_ingests_every_history_row(client):
    """First rebuild produces 2 artifacts + 2 claims from 2 history rows."""
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["processed"] == 2
    assert body["artifacts"] == 2
    assert body["claims"] == 2


def test_rebuild_is_idempotent(client):
    """A second rebuild does not duplicate nodes."""
    client.post("/worldofboms/rebuild")
    r = client.post("/worldofboms/rebuild")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # processed counts what the endpoint walked; artifacts/claims are
    # store totals afterward.
    assert body["processed"] == 2
    assert body["artifacts"] == 2
    # Two claims per row from two rebuilds = 4 — or 2 if save_claim
    # treats identical (model, run_meta) signatures as a single claim.
    # Today save_claim always writes a fresh claim_iri per call, so 4 is
    # correct. Update if save_claim's contract changes.
    assert body["claims"] == 4
```

- [ ] **Step 2: Run the failing tests**

```bash
pytest tests/store/test_worldofboms_rebuild.py -xvs
```

Expected: FAIL (404 on `/worldofboms/rebuild` — route doesn't exist yet).

- [ ] **Step 3: Add `POST /worldofboms/rebuild` to `src/aikaboom/web/app.py`**

Add this route **after** the existing `/worldofboms/export` handler (around line 1582 — find the closing `return jsonify({'error': str(e)}), 500` of `worldofboms_export`):

```python
@app.route('/worldofboms/rebuild', methods=['POST'])
def worldofboms_rebuild():
    """Re-ingest every BOM under bom-history/ into the worldofBOMs graph.

    Idempotent — relies on store dedup (artifact IRI from canonical
    identifier set + BomStore.resolve cross-identifier lookup), so calling
    this twice does not create duplicate artifact nodes. New claim_iris
    are produced on each call (each rebuild is a fresh save_claim), so
    the ``claims`` count grows by N rows per rebuild.

    Returns ``{processed, artifacts, claims}`` so the UI can show a toast.
    """
    store = _open_graph_store()
    if store is None:
        return jsonify({'processed': 0, 'artifacts': 0, 'claims': 0,
                        'store_unavailable': True})

    from aikaboom.store.naming import Identifier

    rows = _history_load()
    processed = 0
    for row in rows:
        bom_filename = (row.get('artifacts') or {}).get('bom')
        if not bom_filename:
            continue
        bom_path = os.path.join(app.config['HISTORY_FOLDER'], bom_filename)
        if not os.path.exists(bom_path):
            continue
        try:
            with open(bom_path, 'r', encoding='utf-8') as f:
                bom_json = json.load(f)
        except Exception as e:
            print(f"⚠️ rebuild: failed to read {bom_filename}: {e}")
            continue

        # Reconstruct identifiers from the BOM body. Mirrors
        # _try_resolve_cache's identifier assembly (web/app.py:395-404),
        # but reads from the BOM dict itself since the row metadata only
        # carries the subject string, not the typed identifiers.
        idents = []
        repo_id = bom_json.get('repo_id') or bom_json.get('model_id')
        if repo_id:
            idents.append(Identifier('huggingface', str(repo_id)))
        arxiv = bom_json.get('arxiv_paper') or bom_json.get('arxiv_url')
        if arxiv:
            idents.append(Identifier('arxiv', str(arxiv)))
        gh = bom_json.get('github_link') or bom_json.get('github_url')
        if gh:
            idents.append(Identifier('github', str(gh)))
        if not idents:
            continue

        try:
            store.save_claim(
                bom_json,
                run_meta={
                    'provider': 'rebuild', 'llm_model': 'rebuild',
                    'prompt_version': 'rebuild', 'code_version': 'head',
                    'mode': 'rebuild',
                    'use_case': bom_json.get('use_case', 'complete'),
                },
                identifiers=idents,
            )
            processed += 1
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ rebuild: save_claim failed for {bom_filename}: {e}")

    stats = store.stats()
    return jsonify({
        'processed': processed,
        'artifacts': stats.get('artifacts', 0),
        'claims': stats.get('claims', 0),
    })
```

- [ ] **Step 4: Run the rebuild tests and confirm they pass**

```bash
pytest tests/store/test_worldofboms_rebuild.py -xvs
```

Expected: 2 passed.

- [ ] **Step 5: Write the failing test for the sparql branch removal**

Create `tests/store/test_worldofboms_query_no_sparql.py`:

```python
"""POST /worldofboms/query no longer honours an `sparql` field.

The web UI used to expose a raw SPARQL textarea inside the lineage
side-panel. Per the worldofBOMs followups spec section C, that surface
is removed: the query handler accepts only the preset branch now.

graph_view.raw_query is *not* removed — CLI + tests still use it.
"""

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    graph_dir = tmp_path / "graph"; graph_dir.mkdir()
    monkeypatch.setenv("AIKABOOM_GRAPH_DIR", str(graph_dir))
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    monkeypatch.setenv("AIKABOOM_GRAPH_DISABLE", "0")
    from aikaboom.web.app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_query_with_sparql_field_is_rejected(client):
    """A request body with `sparql: ...` returns 400 with a clear message."""
    r = client.post("/worldofboms/query", json={
        "sparql": "SELECT ?s WHERE { ?s ?p ?o } LIMIT 5",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert "sparql" in (body.get("error") or "").lower()


def test_query_with_preset_still_works(client):
    """The preset branch is unchanged."""
    r = client.post("/worldofboms/query", json={
        "preset": "datasets", "direction": "both",
    })
    # Empty graph → empty rows, not 4xx/5xx.
    assert r.status_code == 200
    body = r.get_json()
    assert "rows" in body
```

- [ ] **Step 6: Run it red**

```bash
pytest tests/store/test_worldofboms_query_no_sparql.py -xvs
```

Expected: `test_query_with_sparql_field_is_rejected` FAILS (currently 200 OK with rows from `raw_query`).

- [ ] **Step 7: Remove the sparql branch from `/worldofboms/query`**

In `src/aikaboom/web/app.py`, locate `worldofboms_query` (line 1538). Replace the body:

```python
@app.route('/worldofboms/query', methods=['POST'])
def worldofboms_query():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'rows': [], 'store_unavailable': True})
    data = request.get_json(silent=True) or {}
    if 'sparql' in data:
        # Raw SPARQL was a UI surface that invited graph-literacy the
        # target user does not have. Removed per the worldofBOMs
        # followups spec section C. graph_view.raw_query is still
        # available for CLI and tests.
        return jsonify({
            'error': "raw 'sparql' field is no longer accepted by /worldofboms/query; "
                     "use 'preset' (one of: licenses, datasets, models, conflicts)"
        }), 400
    try:
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
```

- [ ] **Step 8: Re-run both test files**

```bash
pytest tests/store/test_worldofboms_query_no_sparql.py tests/store/test_worldofboms_rebuild.py -xvs
```

Expected: all passed.

- [ ] **Step 9: Commit Task C1**

```bash
git add src/aikaboom/web/app.py tests/store/test_worldofboms_rebuild.py tests/store/test_worldofboms_query_no_sparql.py
git commit -m "feat(web): /worldofboms/rebuild + reject raw sparql on /query

Two backend changes for the worldofBOMs followups:

  * POST /worldofboms/rebuild re-ingests every BOM in bom-history/
    into the graph. Idempotent (relies on store dedup verified by
    tests/store/test_e2e_reuse_via_process.py). Returns
    {processed, artifacts, claims} for a UI toast.

  * POST /worldofboms/query no longer honours a 'sparql' field;
    callers get a 400 explaining the surface was removed. The
    underlying graph_view.raw_query stays available for CLI + tests
    (no change to the store layer).

Per spec section C.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task C2: Frontend — rewrite the lineage pane with action buttons

**Files:**
- Modify: `src/aikaboom/web/templates/index.html`:
  - `loadWorldLineage()` function (around line 7170)
  - `runWorldSparql()` function (remove)
  - The `WORLD_PRESETS` constant (extend with action button definitions or add a sibling)

- [ ] **Step 1: Write the failing Playwright smoke for the action buttons**

Create `tests/test_world_actions_smoke.py`:

```python
"""Playwright smoke: the lineage pane has four action buttons and no
SPARQL textarea, and the actions wire to the right endpoints.

Skipped without Chromium.
"""

import json, shutil, socket, subprocess, sys, time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


def _chromium_available() -> bool:
    cache = Path.home() / ".cache" / "ms-playwright"
    return cache.exists() and any(d.name.startswith("chromium") for d in cache.iterdir())


if not _chromium_available():
    pytest.skip("chromium not installed", allow_module_level=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url, timeout_s=30.0):
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="module")
def flask_server(tmp_path_factory):
    work = tmp_path_factory.mktemp("aibom_world_actions")
    port = _free_port()
    env = {
        **__import__("os").environ,
        "BOM_HOST": "127.0.0.1", "BOM_PORT": str(port),
        "AIKABOOM_GRAPH_DISABLE": "1",  # not needed for DOM-level smoke
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "AIKABOOM_HISTORY_DIR": str(work / "history"),
    }
    (work / "history").mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "aikaboom.web.app"], env=env, cwd=PROJECT_ROOT,
    )
    if not _wait_for_http(f"http://127.0.0.1:{port}/"):
        proc.kill(); pytest.fail("flask didn't start")
    yield f"http://127.0.0.1:{port}"
    proc.terminate(); proc.wait(timeout=5)


def test_lineage_pane_has_no_sparql_textarea(flask_server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(flask_server)
        # Open the worldofBOMs tab; force-call loadWorldLineage with a
        # synthetic focus so the pane renders.
        page.evaluate("""
            switchTabByName('world');
            window.worldFocusIri = 'urn:test:artifact:demo';
            loadWorldLineage();
        """)
        # Action buttons present.
        for action in ('download-spdx-bundle', 'generate-bom',
                       'open-bom', 'rebuild-graph'):
            sel = f'[data-world-action="{action}"]'
            assert page.locator(sel).count() >= 1, f"missing action button {action}"
        # SPARQL textarea is gone.
        assert page.locator("#worldSparql").count() == 0
        assert page.locator("#worldSparqlRun").count() == 0
        browser.close()
```

- [ ] **Step 2: Run it red**

```bash
pytest tests/test_world_actions_smoke.py -xvs
```

Expected: FAIL (textarea still present, action buttons not present).

- [ ] **Step 3: Rewrite `loadWorldLineage()`**

In `src/aikaboom/web/templates/index.html`, locate `function loadWorldLineage()` (around line 7170). Replace the whole function with:

```javascript
        function loadWorldLineage() {
            const root = document.getElementById('worldLineageResults');
            if (!worldFocusIri) {
                root.innerHTML =
                    '<div class="json-placeholder">Click a node to see its lineage.</div>';
                return;
            }
            // Resolve the focused node so we can gate the placeholder/
            // open-bom buttons by claim_count + kind.
            const node = (worldFullGraph?.nodes || []).find(n => n.iri === worldFocusIri);
            const hasClaims = !!(node && (node.claim_count || 0) > 0);
            const isPlaceholder = !!(node && (node.kind === 'placeholder' || !hasClaims));

            // ----- Query presets (unchanged) -----
            const presetButtons = WORLD_PRESETS.map(p =>
                `<button class="mock-button world-preset" data-preset="${p.id}"
                  style="display:block;width:100%;text-align:left;margin:4px 0;font-size:12px;">
                  ▸ ${p.label}</button>`).join('');

            // ----- Action buttons -----
            // Each carries a data-world-action so the Playwright smoke
            // can assert them deterministically.
            const downloadBtn = `
                <button class="mock-button world-action"
                        data-world-action="download-spdx-bundle"
                        style="display:block;width:100%;text-align:left;margin:4px 0;font-size:12px;">
                  ⬇ Download SPDX bundle for this lineage</button>`;

            // Scope chooser for the generate-BOM action — Upstream / Both
            // are gated until the recursive walker grows that direction.
            const generateBtn = `
                <div class="world-action-group"
                     style="border:1px solid var(--paper-edge);border-radius:4px;
                            padding:6px 8px;margin:6px 0;${isPlaceholder ? '' : 'opacity:0.5;'}">
                  <div style="font-size:11.5px;margin-bottom:4px;color:var(--paper-mute);">
                    Generate BOM for this node${isPlaceholder ? '' : ' (already has a BOM)'}
                  </div>
                  <label style="display:block;font-size:12px;margin:2px 0;">
                    <input type="radio" name="genScope" value="node" checked
                           ${isPlaceholder ? '' : 'disabled'}> Just this node
                  </label>
                  <label style="display:block;font-size:12px;margin:2px 0;">
                    <input type="radio" name="genScope" value="downstream"
                           ${isPlaceholder ? '' : 'disabled'}> Downstream walk
                  </label>
                  <label style="display:block;font-size:12px;margin:2px 0;color:var(--paper-faint);">
                    <input type="radio" name="genScope" value="upstream" disabled>
                    Upstream walk
                    <em style="font-style:normal;font-size:10.5px;">
                      (downstream-only today — upstream lands in a followup)</em>
                  </label>
                  <label style="display:block;font-size:12px;margin:2px 0;color:var(--paper-faint);">
                    <input type="radio" name="genScope" value="both" disabled>
                    Both
                    <em style="font-style:normal;font-size:10.5px;">
                      (gated on upstream support)</em>
                  </label>
                  <button class="mock-button world-action"
                          data-world-action="generate-bom"
                          ${isPlaceholder ? '' : 'disabled'}
                          style="margin-top:6px;font-size:12px;">
                    ▶ Generate BOM
                  </button>
                </div>`;

            const openBtn = `
                <button class="mock-button world-action"
                        data-world-action="open-bom"
                        ${hasClaims ? '' : 'disabled'}
                        style="display:block;width:100%;text-align:left;margin:4px 0;font-size:12px;
                               ${hasClaims ? '' : 'opacity:0.5;'}">
                  ↗ Open this BOM in the main viewer</button>`;

            const rebuildBtn = `
                <button class="mock-button world-action"
                        data-world-action="rebuild-graph"
                        style="display:block;width:100%;text-align:left;margin:4px 0;font-size:12px;">
                  ⟳ Refresh / rebuild graph from history</button>`;

            root.innerHTML = `
                <div style="font-size:11.5px;color:var(--paper-mute);margin-bottom:6px;">
                  Scope: <strong>${worldDirection}</strong> — change with the Direction control.
                </div>

                <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;
                            color:var(--paper-faint);margin:6px 0 2px;">Queries</div>
                ${presetButtons}

                <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;
                            color:var(--paper-faint);margin:10px 0 2px;">Actions</div>
                ${downloadBtn}
                ${generateBtn}
                ${openBtn}
                ${rebuildBtn}

                <div id="worldQueryResult" style="margin-top:8px;"></div>`;

            // Wire presets.
            root.querySelectorAll('.world-preset').forEach(b =>
                b.addEventListener('click', () => runWorldPreset(b.dataset.preset)));

            // Wire actions.
            root.querySelector('[data-world-action="download-spdx-bundle"]')
                ?.addEventListener('click', runWorldDownloadBundle);
            root.querySelector('[data-world-action="generate-bom"]')
                ?.addEventListener('click', () => runWorldGenerateBom(node));
            root.querySelector('[data-world-action="open-bom"]')
                ?.addEventListener('click', () => runWorldOpenBom(node));
            root.querySelector('[data-world-action="rebuild-graph"]')
                ?.addEventListener('click', runWorldRebuild);
        }
```

- [ ] **Step 4: Remove `runWorldSparql()` and its wiring**

In the same template, locate `async function runWorldSparql()` (around line 7240). **Delete the entire function.** Then search for any remaining `worldSparql` references and remove them (there should be none after the `loadWorldLineage` rewrite above — the old function defined the textarea inline).

- [ ] **Step 5: Add the four new action functions**

Add these **immediately after** `runWorldPreset` in the same `<script>` block:

```javascript
        // -- worldofBOMs action button handlers ----------------------------

        function runWorldDownloadBundle() {
            if (!worldFocusIri) return;
            const url = `/worldofboms/export?scope=ego`
                + `&artifact=${encodeURIComponent(worldFocusIri)}`
                + `&direction=${encodeURIComponent(worldDirection)}`;
            // Navigate so the browser saves the attachment.
            window.open(url, '_blank');
        }

        function runWorldGenerateBom(node) {
            if (!node) return;
            // Read scope from the radio group; default to 'node'.
            const scopeEl = document.querySelector(
                'input[name="genScope"]:checked');
            const scope = scopeEl ? scopeEl.value : 'node';

            // Map node kind → bom_type. Anything other than dataset goes
            // to 'ai' (a console warning so the user knows we guessed).
            let bomType = 'ai';
            if (node.kind === 'Dataset') bomType = 'data';
            else if (node.kind && node.kind !== 'Model') {
                console.warn(`worldofBOMs generate: unknown kind ${node.kind}, defaulting bom_type=ai`);
            }

            // Pre-fill the form.
            const radio = document.getElementById('bom_type_' + bomType);
            if (radio) {
                radio.checked = true;
                radio.dispatchEvent(new Event('change', { bubbles: true }));
            }
            const subjectField = bomType === 'data'
                ? document.getElementById('hf_repo_id')
                : document.getElementById('repo_id');
            if (subjectField && node.label) subjectField.value = node.label;

            // Recursive toggle: only enabled for 'downstream' scope.
            // (Upstream/both are disabled in the chooser today.)
            const recursiveCb = document.getElementById('recursive_bom');
            if (recursiveCb) recursiveCb.checked = (scope === 'downstream');

            // Persist + switch to the form pane.
            try { _saveFormStorage(); } catch (_) { /* helper may not be in scope */ }
            const formPane = document.querySelector('.form-pane') || form;
            try { formPane.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (_) {}
            showStatus('success',
                `Form pre-filled from <strong>${escapeHtml(node.label || worldFocusIri)}</strong>.
                 Tweak inputs and click Generate.`);
        }

        async function runWorldOpenBom(node) {
            if (!node || !worldFocusIri) return;
            try {
                const resp = await fetch(
                    `/worldofboms/bom/${encodeURIComponent(worldFocusIri)}`);
                const bom = await resp.json();
                if (bom.error || bom.store_unavailable) {
                    showStatus('error',
                        'No BOM stored for this node (placeholder).');
                    return;
                }
                // Rehydrate the main BOM viewer via the existing helper
                // used by History row clicks.
                rehydrateFreshGen({ metadata: bom });
                switchTabByName('complete');
                showStatus('success',
                    `Opened <strong>${escapeHtml(node.label)}</strong> from worldofBOMs.`);
            } catch (e) {
                showStatus('error', `Failed to open BOM: ${escapeHtml(String(e))}`);
            }
        }

        async function runWorldRebuild() {
            const out = document.getElementById('worldQueryResult');
            if (out) out.innerHTML =
                '<div class="json-placeholder">Rebuilding graph from history…</div>';
            try {
                const resp = await fetch('/worldofboms/rebuild', { method: 'POST' });
                const body = await resp.json();
                if (body.store_unavailable) {
                    if (out) out.innerHTML =
                        '<div class="json-placeholder">Graph store unavailable.</div>';
                    return;
                }
                if (out) out.innerHTML =
                    `<div class="json-placeholder">
                       Rebuilt — processed ${body.processed} rows;
                       graph now has ${body.artifacts} artifacts,
                       ${body.claims} claims.
                     </div>`;
                // Refresh the rendered graph so new nodes appear.
                if (typeof loadWorldGraph === 'function') loadWorldGraph();
            } catch (e) {
                if (out) out.innerHTML =
                    `<div class="json-placeholder">Rebuild failed: ${escapeHtml(String(e))}</div>`;
            }
        }
```

- [ ] **Step 6: Run the Playwright smoke and confirm it passes**

```bash
pytest tests/test_world_actions_smoke.py -xvs
```

Expected: PASS.

If Playwright isn't installed, do a manual smoke:

```bash
python -m aikaboom.web.app
# Browser: open /, switch to worldofBOMs tab, click any node.
#   - The lineage pane should show the four query presets followed by
#     four action buttons under "Actions"; no SPARQL textarea.
#   - "Download SPDX bundle" opens a new tab with the JSON download.
#   - "Generate BOM" pre-fills the form and switches to the form pane.
#   - "Open this BOM" loads the BOM into the BOM-with-Provenance viewer
#     (only enabled for nodes with claim_count > 0).
#   - "Refresh / rebuild graph" shows a count toast and re-renders the graph.
```

- [ ] **Step 7: Commit Task C2**

```bash
git add src/aikaboom/web/templates/index.html tests/test_world_actions_smoke.py
git commit -m "ui(worldofboms): replace SPARQL textarea with four action buttons

The side-panel lineage pane offered four query presets plus an
'Advanced: SPARQL' textarea. The textarea invited graph literacy the
target user does not have; per the worldofBOMs followups spec section C,
it is replaced with four scoped action buttons grouped under 'Actions':

  * Download SPDX bundle for this lineage (wires /worldofboms/export
    ?scope=ego — route already existed)
  * Generate BOM for this node — pre-fills the main form with bom_type
    + subject, with a scope chooser: 'Just this node' / 'Downstream
    walk' / 'Upstream walk' / 'Both'. Upstream and Both are disabled
    until the recursive walker grows upstream support (followup).
    Enabled only for placeholder / claim_count==0 nodes.
  * Open this BOM in the main viewer — rehydrates the BOM-with-
    Provenance viewer via the existing History helper. Enabled only
    for claim_count>0 nodes.
  * Refresh / rebuild graph from history — calls the new
    POST /worldofboms/rebuild (added in the previous commit).

runWorldSparql + the <details>Advanced: SPARQL</details> block are
removed. graph_view.raw_query stays in the store layer for CLI + tests
(no store changes).

TDD: Playwright smoke asserts the four [data-world-action] buttons
exist and the textarea is gone.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the full store test suite to catch regressions**

```bash
pytest tests/store/ -x
```

Expected: all passed.

- [ ] **Run the UI smoke suite (if Playwright installed)**

```bash
pytest tests/test_pipeline_ui_smoke.py tests/test_splash_history_hint.py tests/test_world_actions_smoke.py -x
```

Expected: all passed (or all skipped if Chromium unavailable, in which case the manual smokes from each section need to be done by hand before pushing).

- [ ] **Confirm clean commit history**

```bash
git log --oneline -6
```

Expected order (top to bottom):
- `ui(worldofboms): replace SPARQL textarea with four action buttons`
- `feat(web): /worldofboms/rebuild + reject raw sparql on /query`
- `ui(splash): show 'open one of your N past BOMs' affordance`
- `test(store): e2e — second /process for same model reuses artifact`
- `docs(worldofboms): three-followup design — splash hint, e2e reuse test, action buttons`
- (earlier commits on worldofboms-graph)

- [ ] **Push to update PR #48**

```bash
git push origin worldofboms-graph
```

The four new commits land on PR #48; reviewers see the followups inline with the original work.

---

## Self-review (notes left for the executing agent)

1. **Spec coverage:** Section A → Task A1. Section B → Tasks B1 (all three scenarios in one file). Section C → Tasks C1 (backend) + C2 (frontend). All spec requirements have at least one task.

2. **Placeholder scan:** No "TBD" / "implement later" / "similar to Task N" / "add appropriate validation" — every code block is concrete and pasteable.

3. **Type consistency:**
   - `_open_graph_store` / `_history_load` are existing helpers in `src/aikaboom/web/app.py`; verified present at the line numbers cited.
   - `BomStore.stats()` keys (`artifacts`, `claims`, `versions`, `votes`) confirmed against `store.py`'s `stats` method.
   - `vocab.trainedOn` confirmed exists (`AI_RELATIONSHIP_FIELDS` maps `trainedOnDatasets → trainedOn`).
   - `_FakeAIProc.process_ai_model(**_kwargs)` mirrors the signature in `tests/store/test_web_resolve.py`.
   - The new `AIKABOOM_HISTORY_DIR` env-var is introduced once (Task A1 Step 1) and reused in Task C1's fixtures and Task C2's Playwright fixture.

4. **Risks worth flagging at execution time:**
   - The Playwright tests skip cleanly without Chromium. If you skip them, do the manual browser smokes called out at the end of each section before pushing — there's no other coverage of the JS rewrites.
   - The rebuild test asserts `claims == 4` after two rebuilds. If a future change makes `save_claim` deduplicate on `(artifact, run_meta)` signature, that assertion needs to drop to `2`. A comment in the test calls this out.
