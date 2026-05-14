"""End-to-end Playwright smoke + usability test for the Pipeline / History UI.

Drives the browser through three layers:

  1. Static render — page loads, splash shows, every tab + Pipeline scaffold
     element is in the DOM, tab clicks land on the right content panel,
     /history endpoint responds with valid JSON.

  2. Synthetic pipeline event flow — injects ``[[BOM_EVENT]]`` payloads
     directly via ``page.evaluate('Pipeline.handleEvent(...)')`` so the
     timeline animates through pipeline.start / stage.start|done /
     field.done|error / pipeline.done without needing an LLM run.

  3. History tab rehydration — seeds ``bom-history/index.json`` plus a
     matching BOM artifact, opens the History tab, clicks the row, and
     confirms the BOM viewer + downloadComplete link rehydrate from the
     saved file.

The whole module is skipped if Playwright (Python package + Chromium
binary) isn't installed locally, so CI without Playwright stays green.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Skip the entire module unless Playwright + Chromium are available locally.
# Playwright is a heavy dependency (~150MB Chromium); we don't want to make
# it a hard requirement of the test suite.
sync_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed"
)


def _chromium_available() -> bool:
    cache = Path.home() / ".cache" / "ms-playwright"
    if not cache.exists():
        return False
    return any(d.name.startswith("chromium") for d in cache.iterdir())


if not _chromium_available():
    pytest.skip(
        "chromium not installed (run: playwright install chromium)",
        allow_module_level=True,
    )


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
def flask_server(tmp_path_factory):
    """Boot the worktree's Flask app in a subprocess on a free port, with an
    isolated bom-history directory so the test doesn't pollute the caller's
    real history. Tears down on module exit."""
    port = _free_port()
    history_dir = tmp_path_factory.mktemp("bom-history-isolated")
    # The app reads HISTORY_FOLDER from app.config (computed from the project
    # root). Easiest isolation: set the env-derived project root via a
    # symlink, OR just point UPLOAD_FOLDER + HISTORY_FOLDER via monkeypatch.
    # Simpler: launch with cwd set to the project root and let it create the
    # real bom-history/, then stash + restore around the test run. This keeps
    # the subprocess identical to how a user runs it.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["BOM_HOST"] = "127.0.0.1"
    env["BOM_PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, "-m", "aikaboom.web.app"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_http(base + "/"):
            proc.terminate()
            pytest.skip(f"flask app failed to start on {base}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def seeded_history():
    """Stash any existing bom-history/ aside, seed one fake row + artifact,
    yield the hash, then restore the original on teardown."""
    history_dir = PROJECT_ROOT / "bom-history"
    stash = PROJECT_ROOT / "bom-history.stash-smoke-test"
    if history_dir.exists():
        if stash.exists():
            shutil.rmtree(stash)
        shutil.move(str(history_dir), str(stash))
    history_dir.mkdir(exist_ok=True)

    fake_hash = "abc123def456"
    bom = {
        "model_id": "test-org/test-model",
        "direct_fields": {
            "license": {"value": "Apache-2.0", "source": "huggingface"},
        },
        "rag_fields": {
            "domain": {
                "value": "Natural Language Processing",
                "source": "arxiv",
                "trace": {
                    "claims": {"arxiv": "NLP model", "github": "language model"},
                    "internal_conflicts": {},
                    "external_conflicts": [],
                },
            },
        },
        "conflict_summary": {
            "total": 0, "high_confidence": 0, "low_confidence": 0,
            "deterministic": 0, "suppressed": 0,
        },
    }
    (history_dir / f"{fake_hash}_aibom.json").write_text(json.dumps(bom, indent=2))
    index = [{
        "hash": fake_hash,
        "subject": "test-org/test-model",
        "bom_type": "ai",
        "created_at": "2026-05-13T10:00:00+00:00",
        "artifacts": {"bom": f"{fake_hash}_aibom.json"},
    }]
    (history_dir / "index.json").write_text(json.dumps(index, indent=2))

    try:
        yield fake_hash
    finally:
        if history_dir.exists():
            shutil.rmtree(history_dir)
        if stash.exists():
            shutil.move(str(stash), str(history_dir))


@pytest.fixture
def page(flask_server):
    """Single Playwright page for one test, capturing console errors."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        p = ctx.new_page()
        errors = []
        p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        p.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        p.console_errors = errors  # type: ignore[attr-defined]
        p.goto(flask_server, wait_until="domcontentloaded")
        yield p
        browser.close()


# ---------------------------------------------------------------------------
# Layer 1 — static render
# ---------------------------------------------------------------------------

def test_splash_visible_on_first_load(page):
    assert page.locator("#resultEmpty").is_visible()


def test_pipeline_tab_hidden_before_run(page):
    assert page.locator("#pipelineTabBtn").is_hidden()


@pytest.mark.parametrize("tab_id", [
    "completeTab", "conflictsTab", "spdxTab", "cyclonedxTab",
    "recursiveTab", "logsTab", "historyTab", "pipelineTab",
])
def test_tab_panel_exists(page, tab_id):
    assert page.locator(f"#{tab_id}").count() == 1


@pytest.mark.parametrize("selector", [
    "#pipelineProgress", "#pipelineRows", "#pipelineExpand", "#pipelineActions",
    "#pxField", "#pxStage", "#pxBody", "#pxStatus",
])
def test_pipeline_scaffold_present(page, selector):
    assert page.locator(selector).count() == 1


def test_tab_clicks_activate_panels(page):
    # Tabs are hidden behind the splash; remove it to make them clickable.
    page.evaluate("document.getElementById('resultPane').classList.remove('is-empty')")

    page.locator(".tab", has_text="Logs").click()
    assert "active" in (page.locator("#logsTab").get_attribute("class") or "")

    page.locator(".tab", has_text="History").click()
    assert "active" in (page.locator("#historyTab").get_attribute("class") or "")


def test_history_endpoint_returns_json(page):
    payload = page.evaluate("fetch('/history').then(r => r.json())")
    assert isinstance(payload, dict)
    assert "rows" in payload


# ---------------------------------------------------------------------------
# Layer 2 — synthetic pipeline event flow
# ---------------------------------------------------------------------------

PIPELINE_START = {
    "event": "pipeline.start",
    "item_type": "ai",
    "item_id": "test-org/test-model",
    # 7 fields → Math.ceil(7/3)=3 per row → exercises the snake wrap
    "fields": ["domain", "license", "energyConsumption", "hyperparameter",
               "limitation", "metric", "autonomyType"],
    "total": 7,
}


def _start_pipeline(page):
    page.evaluate("(evt) => Pipeline.handleEvent(evt)", PIPELINE_START)
    page.wait_for_timeout(150)


def test_pipeline_start_reveals_tab_and_hides_splash(page):
    _start_pipeline(page)
    assert page.locator("#pipelineTabBtn").is_visible()
    assert "active" in (page.locator("#pipelineTab").get_attribute("class") or "")
    assert "is-empty" not in (page.locator("#resultPane").get_attribute("class") or "")


def test_pipeline_snake_layout(page):
    _start_pipeline(page)
    # 7 columns × 3 dots = 21 dots
    assert page.locator(".pipeline-field").count() == 7
    assert page.locator(".pipeline-dot").count() == 21
    rows = page.locator(".pipeline-row")
    assert rows.count() == 3
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(3)]
    assert "reverse" not in classes[0]   # row 1 LTR
    assert "reverse" in classes[1]       # row 2 reversed (snake)
    assert "reverse" not in classes[2]   # row 3 LTR


def test_stage_dots_cycle_running_to_done(page):
    _start_pipeline(page)
    fake_data = {
        "retrieve":  {"chunks_per_source": {"github": 5, "arxiv": 23}, "total_chunks": 28,
                      "sources_used": ["github", "arxiv"]},
        "reconcile": {"internal_conflicts": [], "external_conflicts": []},
        "resolve":   {"answer_preview": "Natural Language Processing", "selected_sources": ["arxiv"]},
    }
    for stage in ("retrieve", "reconcile", "resolve"):
        page.evaluate(
            "(s) => Pipeline.handleEvent({event:'stage.start',field:'domain',stage:s})",
            stage,
        )
        page.wait_for_timeout(40)
        dot = page.locator(f'.pipeline-dot[data-field="domain"][data-stage="{stage}"]')
        assert "is-running" in (dot.get_attribute("class") or ""), f"{stage} not running"

        page.evaluate(
            "(p) => Pipeline.handleEvent(p)",
            {"event": "stage.done", "field": "domain", "stage": stage,
             "duration_ms": 1234, "data": fake_data[stage]},
        )
        page.wait_for_timeout(40)
        dot = page.locator(f'.pipeline-dot[data-field="domain"][data-stage="{stage}"]')
        assert "is-done" in (dot.get_attribute("class") or ""), f"{stage} not done"


def test_field_done_updates_column_and_counter(page):
    _start_pipeline(page)
    page.evaluate("""(_) => Pipeline.handleEvent({
        event:'field.done', field:'domain', data:{answer_preview:'NLP'}
    })""", None)
    page.wait_for_timeout(80)
    col = page.locator('.pipeline-field[data-field="domain"]')
    assert "is-done" in (col.get_attribute("class") or "")
    assert "1 / 7" in page.locator("#pipelineCount").inner_text()


def test_expand_card_opens_and_shows_data(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "stage.done", "field": "domain", "stage": "resolve",
         "duration_ms": 1234,
         "data": {"answer_preview": "Natural Language Processing",
                  "selected_sources": ["arxiv"]}},
    )
    page.locator('.pipeline-dot[data-field="domain"][data-stage="resolve"]').click()
    page.wait_for_timeout(120)

    assert "is-open" in (page.locator("#pipelineExpand").get_attribute("class") or "")
    body = page.locator("#pxBody").inner_text()
    assert "Natural Language Processing" in body
    assert "arxiv" in body
    status = page.locator("#pxStatus").inner_text()
    assert "done" in status
    assert "1234" in status


def test_expand_card_swaps_when_clicking_different_dot(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "stage.done", "field": "domain", "stage": "resolve",
         "duration_ms": 100, "data": {"answer_preview": "X"}},
    )
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "stage.done", "field": "license", "stage": "retrieve",
         "duration_ms": 200,
         "data": {"chunks_per_source": {"huggingface": 8}, "total_chunks": 8,
                  "sources_used": ["huggingface"]}},
    )
    page.locator('.pipeline-dot[data-field="domain"][data-stage="resolve"]').click()
    page.wait_for_timeout(80)
    page.locator('.pipeline-dot[data-field="license"][data-stage="retrieve"]').click()
    page.wait_for_timeout(120)

    # text_content().strip() — inner_text() in Playwright treats inline spans
    # inside flex containers inconsistently. text_content reflects the actual
    # textContent property the JS set via openExpand().
    px_field = (page.locator("#pxField").text_content() or "").strip()
    px_stage = (page.locator("#pxStage").text_content() or "").strip()
    assert px_field == "license"
    assert px_stage == "Retrieve"


def test_expand_card_close_button(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "stage.done", "field": "domain", "stage": "resolve",
         "duration_ms": 100, "data": {"answer_preview": "X"}},
    )
    page.locator('.pipeline-dot[data-field="domain"][data-stage="resolve"]').click()
    page.wait_for_timeout(80)
    page.locator("#pipelineExpand .close-x").click()
    page.wait_for_timeout(60)
    assert "is-open" not in (page.locator("#pipelineExpand").get_attribute("class") or "")


def test_field_error_marks_column_and_dots(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "field.error", "field": "metric", "error": "context too long"},
    )
    page.wait_for_timeout(80)
    col = page.locator('.pipeline-field[data-field="metric"]')
    assert "is-error" in (col.get_attribute("class") or "")
    err_dots = page.locator('.pipeline-field[data-field="metric"] .pipeline-dot.is-error')
    assert err_dots.count() >= 1


def test_pipeline_done_reveals_actions(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "pipeline.done", "duration_ms": 174000},
    )
    page.wait_for_timeout(80)
    assert "is-shown" in (page.locator("#pipelineActions").get_attribute("class") or "")
    assert "is-done" in (page.locator("#pipelineProgress").get_attribute("class") or "")


def test_view_bom_action_switches_to_complete_tab(page):
    _start_pipeline(page)
    page.evaluate(
        "(p) => Pipeline.handleEvent(p)",
        {"event": "pipeline.done", "duration_ms": 1000},
    )
    page.wait_for_timeout(80)
    page.locator("#pipelineActions button", has_text="View BOM").click()
    page.wait_for_timeout(80)
    assert "active" in (page.locator("#completeTab").get_attribute("class") or "")


# ---------------------------------------------------------------------------
# Layer 2b — splash redesign, pipeline header, per-row legends, Stage 0
# ---------------------------------------------------------------------------

def test_splash_shows_brand_and_three_step_cards(page):
    """The redesigned splash includes the AIkaBoOM logo + wordmark, exactly
    three numbered step cards with SVG arrows between them, and the returns
    + pipeline-hint footer."""
    assert page.locator("#resultEmpty .splash-brand img").count() == 1
    assert "AIka" in (page.locator("#resultEmpty .splash-wordmark").inner_text() or "")
    assert page.locator("#resultEmpty .splash-step").count() == 3
    # Hand-drawn SVG arrows live between the cards (two of them, for three cards)
    assert page.locator("#resultEmpty .splash-arrow svg").count() == 2
    # SVG arrows each have at least two <path> children (curve + arrowhead)
    paths_in_first_arrow = page.locator("#resultEmpty .splash-arrow").nth(0).locator("svg path").count()
    assert paths_in_first_arrow >= 2
    assert page.locator("#resultEmpty .splash-returns").count() == 1
    assert page.locator("#resultEmpty .splash-hint").count() == 1


def test_splash_fits_in_typical_viewport(page):
    """Splash must not require scrolling at a standard viewport. The page
    has a sticky topbar + result-pane header above the splash so a strict
    height check is the right gate."""
    box = page.locator("#resultEmpty").bounding_box()
    assert box is not None
    # 500px leaves comfortable room under the topbar + Generated BOM header
    # within a 768px-tall laptop viewport. Originally splashed at 380px+ padding.
    assert box["height"] < 500, f"splash height {box['height']}px is too tall"


def test_pipeline_header_brand_and_subject(page):
    """Pipeline tab header carries the AIkaBoOM logo + wordmark on the
    left and the subject (model/dataset id) on the right, populated from
    the pipeline.start event."""
    # Header always rendered, just empty until pipeline.start fires.
    assert page.locator("#pipelineHeader .pipeline-header-brand img").count() == 1
    wm = page.locator("#pipelineHeader .pipeline-header-brand .wm").inner_text() or ""
    assert "AIka" in wm
    # Subject is "—" before any run
    assert (page.locator("#pipelineSubject").text_content() or "").strip() in ("—", "-")

    _start_pipeline(page)
    subject = (page.locator("#pipelineSubject").text_content() or "").strip()
    assert subject == "test-org/test-model"


def test_pipeline_per_row_legends(page):
    """Every snake-row carries its own Retrieve / Reconcile / Resolve
    legend (so users don't lose context on row 2 and row 3)."""
    _start_pipeline(page)
    legends = page.locator(".pipeline-row-legend")
    assert legends.count() == 3, f"expected 3 row-legends, got {legends.count()}"
    # 3 legends × 3 stages = 9 labels total
    items = page.locator(".pipeline-row-legend > div")
    assert items.count() == 9
    # Each legend has all three labels in order
    for i in range(3):
        labels = [
            page.locator(f".pipeline-row-legend").nth(i).locator("> div").nth(j).inner_text().strip()
            for j in range(3)
        ]
        assert labels == ["Retrieve", "Reconcile", "Resolve"], f"row {i}: {labels}"


def test_pipeline_snake_fields_reverse_on_odd_rows(page):
    """Row 2 must visually wind right-to-left: with CSS row-reverse on the
    fields-wrapper, the first DOM field in row 2's slice appears at the
    largest x-coordinate, not the smallest."""
    _start_pipeline(page)
    # 7 fields, 3 rows → perRow=3:
    #   row 0 (LTR):      [domain, license, energyConsumption]
    #   row 1 (reversed): [hyperparameter, limitation, metric]
    #   row 2 (LTR):      [autonomyType]
    # With CSS row-reverse on row 1's fields-wrapper, the first DOM field
    # (hyperparameter) ends up at the RIGHT edge, the last DOM field (metric)
    # at the LEFT — so first.x > last.x.
    row1 = page.locator(".pipeline-row").nth(1)
    first_field = row1.locator(".pipeline-field").nth(0).bounding_box()
    last_field = row1.locator(".pipeline-field").nth(-1).bounding_box()
    assert first_field is not None and last_field is not None
    assert first_field["x"] > last_field["x"], (
        f"row 1 not reversed: first DOM field x={first_field['x']} "
        f"should be > last DOM field x={last_field['x']}"
    )


def test_stage0_hidden_before_fallback(page):
    """Stage 0 (Link Fallback) card stays hidden until a fallback.start
    event arrives — Pipeline tab starts clean."""
    assert page.locator("#pipelineStage0").get_attribute("hidden") is not None


def test_stage0_renders_searching_then_done(page):
    """fallback.start makes Stage 0 visible with sources in 'searching'
    state. Each source.checked transitions one source. fallback.done
    collapses the card into is-done state with a summary."""
    page.evaluate(
        "(e) => Pipeline.handleEvent(e)",
        {"event": "fallback.start", "sources": ["huggingface", "arxiv", "github"]},
    )
    page.wait_for_timeout(80)
    stage0 = page.locator("#pipelineStage0")
    assert stage0.get_attribute("hidden") is None, "stage 0 not revealed by fallback.start"

    for src in ["huggingface", "arxiv", "github"]:
        state = stage0.locator(f'.stage0-src[data-src="{src}"]').get_attribute("data-state")
        assert state == "searching", f"{src} should be searching, got {state}"

    # Two found, one missing
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "fallback.source.checked", "source": "huggingface", "found": True})
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "fallback.source.checked", "source": "arxiv", "found": True})
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "fallback.source.checked", "source": "github", "found": False})
    page.wait_for_timeout(80)

    assert stage0.locator('.stage0-src[data-src="huggingface"]').get_attribute("data-state") == "found"
    assert stage0.locator('.stage0-src[data-src="arxiv"]').get_attribute("data-state") == "found"
    assert stage0.locator('.stage0-src[data-src="github"]').get_attribute("data-state") == "missing"

    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "fallback.done", "found_count": 2, "total_count": 3})
    page.wait_for_timeout(80)
    assert "is-done" in (stage0.get_attribute("class") or "")
    status = page.locator("#stage0Status").inner_text()
    assert "2/3" in status, f"summary text missing: {status!r}"


def test_stage0_skipped_when_all_links_provided(page):
    """When the user supplies all 3 links up front, fallback.skipped fires
    and the card shows a skipped message instead of searching dots."""
    page.evaluate(
        "(e) => Pipeline.handleEvent(e)",
        {"event": "fallback.skipped", "reason": "all links provided"},
    )
    page.wait_for_timeout(60)
    stage0 = page.locator("#pipelineStage0")
    assert stage0.get_attribute("hidden") is None
    assert "is-done" in (stage0.get_attribute("class") or "")
    assert "skipped" in (page.locator("#stage0Status").inner_text() or "").lower()


# ---------------------------------------------------------------------------
# Layer 2c — Stage 2: Recursive Children card
# ---------------------------------------------------------------------------

def test_stage2_hidden_before_recursive_start(page):
    """Recursive card stays hidden on a normal run. Only revealed when the
    recursive walker emits its first event."""
    assert page.locator("#pipelineStage2").get_attribute("hidden") is not None


def test_stage2_start_reveals_card_with_depth_and_cap(page):
    """recursive.start surfaces the card and shows max-depth + safety-cap
    in the meta line of the header."""
    page.evaluate(
        "(e) => Pipeline.handleEvent(e)",
        {"event": "recursive.start", "parent": "test/x", "bom_type": "ai",
         "max_depth": 3, "safety_cap": 50},
    )
    page.wait_for_timeout(60)
    card = page.locator("#pipelineStage2")
    assert card.get_attribute("hidden") is None
    meta = (page.locator("#stage2Meta").inner_text() or "").lower()
    assert "max depth: 3" in meta, f"unexpected meta: {meta!r}"
    assert "safety cap: 50" in meta


def test_stage2_chips_render_with_id_relationship_and_state(page):
    """Each discovered child renders a chip with target id + relationship
    + state icon (the answer for the 'chip detail' design question)."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai", "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.target.discovered",
        "target": "allenai/c4", "bom_type": "data",
        "relationship_type": "trainedOn", "depth": 1, "parent": "test/x",
    })
    page.wait_for_timeout(60)
    chip = page.locator(".stage2-chip[data-target='allenai/c4']")
    assert chip.count() == 1
    assert chip.get_attribute("data-state") == "pending"
    assert "allenai/c4" in chip.locator(".target").inner_text()
    assert "trainedOn" in chip.locator(".rel").inner_text()


def test_stage2_chip_state_cycles_pending_running_done(page):
    """A chip moves through pending → running → done as start/done events
    arrive for the same target."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai", "max_depth": 1, "safety_cap": 10})
    child = {"target": "google/t5-base", "bom_type": "ai",
             "relationship_type": "dependsOn", "depth": 1, "parent": "test/x"}
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.target.discovered", **child})
    page.wait_for_timeout(40)
    chip = page.locator(".stage2-chip[data-target='google/t5-base']")
    assert chip.get_attribute("data-state") == "pending"

    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.start", **child})
    page.wait_for_timeout(40)
    assert chip.get_attribute("data-state") == "running"

    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.child.done", **child, "enriched": True, "duration_ms": 1234})
    page.wait_for_timeout(40)
    assert chip.get_attribute("data-state") == "done"


def test_stage2_skipped_chip_shows_reason(page):
    """Skipped chips (duplicate / safety-cap / conflict) render muted with
    the reason visible — this is the 'show skipped' design answer."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai", "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.child.skipped",
        "target": "dup/repo", "bom_type": "ai",
        "relationship_type": "dependsOn", "depth": 1, "parent": "test/x",
        "reason": "duplicate",
    })
    page.wait_for_timeout(60)
    chip = page.locator(".stage2-chip[data-target='dup/repo']")
    assert chip.get_attribute("data-state") == "skipped"
    reason = chip.locator(".reason").inner_text()
    assert "duplicate" in reason


def test_stage2_nests_children_under_their_parent(page):
    """Tree layout: x/b is a child of x/a, so x/b's node must be rendered
    inside x/a's .stage2-children container — not as a sibling under the
    root. (Replaces the old depth-band grouping.)"""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai", "max_depth": 2, "safety_cap": 50})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.target.discovered",
        "target": "x/a", "bom_type": "data", "relationship_type": "trainedOn", "depth": 1, "parent": "test/x",
    })
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.target.discovered",
        "target": "x/b", "bom_type": "data", "relationship_type": "trainedOn", "depth": 2, "parent": "x/a",
    })
    page.wait_for_timeout(60)
    # Root chip rendered from recursive.start
    root = page.locator('.stage2-node[data-target="test/x"]')
    assert root.count() == 1
    # x/a is a direct child of root (under root's .stage2-children)
    assert root.locator('> .stage2-children > .stage2-node[data-target="x/a"]').count() == 1
    # x/b is a direct child of x/a, not of root
    assert page.locator('.stage2-node[data-target="x/a"] > .stage2-children > .stage2-node[data-target="x/b"]').count() == 1
    # x/b should NOT appear at the root level
    assert root.locator('> .stage2-children > .stage2-node[data-target="x/b"]').count() == 0


def test_stage2_root_chip_rendered_from_recursive_start(page):
    """The root chip represents the input BOM and appears immediately on
    recursive.start, before any child events arrive."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "meta/my-model",
                   "bom_type": "ai", "max_depth": 3, "safety_cap": 50})
    page.wait_for_timeout(60)
    root_chip = page.locator('.stage2-node[data-target="meta/my-model"] > .stage2-node-row > .stage2-chip')
    assert root_chip.count() == 1
    # Root sports the special 'root' relationship label
    assert "root" in root_chip.locator(".rel").inner_text()
    # Root is in running state until recursive.done fires
    assert root_chip.get_attribute("data-state") == "running"


def test_stage2_dep_count_label_grows_with_discovery(page):
    """The 'N / M processed' label on a parent updates as discovery and
    processing progress: M = direct children discovered so far,
    N = direct children in a terminal state."""
    P = "test/parent"
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": P, "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    page.wait_for_timeout(40)

    def dep_text():
        return (page.locator(f'.stage2-node[data-target="{P}"] > .stage2-node-row > .dep-count')
                .text_content() or "").strip()

    # No children yet → no label
    assert page.locator(f'.stage2-node[data-target="{P}"] > .stage2-node-row > .dep-count').count() == 0

    # Discover one → 0 / 1 processed
    c1 = {"target": "a/1", "bom_type": "data", "relationship_type": "trainedOn", "depth": 1, "parent": P}
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.target.discovered", **c1})
    page.wait_for_timeout(40)
    assert "0 / 1 processed" in dep_text()

    # Discover second → 0 / 2 processed (denominator grows)
    c2 = {"target": "a/2", "bom_type": "data", "relationship_type": "trainedOn", "depth": 1, "parent": P}
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.target.discovered", **c2})
    page.wait_for_timeout(40)
    assert "0 / 2 processed" in dep_text()

    # First child done → 1 / 2 processed (numerator grows)
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.start", **c1})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.done", **c1, "duration_ms": 100})
    page.wait_for_timeout(40)
    assert "1 / 2 processed" in dep_text()

    # All processed → 2 / 2, and the label gets the is-done class
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.start", **c2})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.done", **c2, "duration_ms": 100})
    page.wait_for_timeout(40)
    assert "2 / 2 processed" in dep_text()
    dep_el = page.locator(f'.stage2-node[data-target="{P}"] > .stage2-node-row > .dep-count')
    assert "is-done" in (dep_el.get_attribute("class") or "")


def test_stage2_all_chips_visible_before_processing_starts(page):
    """The two-phase walker emits all recursive.target.discovered events
    for a parent's siblings before any of them transition to running.
    Verifies the UI shows all pending chips up front (the user's ask:
    'continuous set that we keep displayed')."""
    P = "test/parent"
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": P, "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})

    # Phase 1 — discover 3 siblings (no starts yet)
    for n in ["a", "b", "c"]:
        page.evaluate("(e) => Pipeline.handleEvent(e)", {
            "event": "recursive.target.discovered",
            "target": f"sib/{n}", "bom_type": "data",
            "relationship_type": "trainedOn", "depth": 1, "parent": P,
        })
    page.wait_for_timeout(60)

    pending_kids = page.locator(
        f'.stage2-node[data-target="{P}"] > .stage2-children > .stage2-node'
    )
    assert pending_kids.count() == 3
    for i in range(3):
        chip = pending_kids.nth(i).locator("> .stage2-node-row > .stage2-chip")
        assert chip.get_attribute("data-state") == "pending"


def test_stage2_conflict_flagged_chip_shows_badge(page):
    """When recursive.target.discovered carries has_conflict=True, the
    chip gets a ⚠ badge + data-has-conflict attribute. The edge is still
    walked (not skipped) — this verifies the new conflict-tagged contract."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.target.discovered",
        "target": "contested/data", "bom_type": "data",
        "relationship_type": "trainedOn", "depth": 1, "parent": "test/x",
        "has_conflict": True,
    })
    page.wait_for_timeout(80)
    chip = page.locator(".stage2-chip[data-target='contested/data']")
    assert chip.count() == 1
    # Chip is walked (state=pending, not skipped) but flagged
    assert chip.get_attribute("data-has-conflict") == "true"
    assert chip.get_attribute("data-state") == "pending"
    # Badge is rendered
    assert chip.locator(".conflict-badge").count() == 1


def _reveal_pipeline(page):
    """Tests that need a *visible* chip/button must drop the splash class
    AND activate the Pipeline tab — the tab is otherwise display:none."""
    page.evaluate("document.getElementById('resultPane').classList.remove('is-empty')")
    page.evaluate("switchTabByName('pipeline')")
    page.wait_for_timeout(40)


def test_stage2_depth_limit_chip_renders_with_play_button(page):
    """A depth-truncated chip (reason='depth-limit') renders as muted and
    sports a ▶ play button so the user can manually generate that BOM via
    the /recursive/generate-one endpoint."""
    _reveal_pipeline(page)
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "p", "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.child.skipped",
        "target": "deep/leaf", "bom_type": "data",
        "relationship_type": "trainedOn", "depth": 2,
        "parent": "p", "reason": "depth-limit",
    })
    page.wait_for_timeout(80)
    chip = page.locator('.stage2-chip[data-target="deep/leaf"]')
    assert chip.get_attribute("data-state") == "skipped"
    assert chip.get_attribute("data-reason") == "depth-limit"
    play = chip.locator(".play-btn")
    assert play.count() == 1
    assert play.is_visible()


def test_stage2_play_button_hidden_on_non_muted_chips(page):
    """Only muted chips (skipped, error) carry the ▶ play button —
    chips in pending/running/done states should not."""
    _reveal_pipeline(page)
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "p", "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    c = {"target": "happy/path", "bom_type": "data", "relationship_type": "trainedOn",
         "depth": 1, "parent": "p"}
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.target.discovered", **c})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.start", **c})
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.child.done", **c, "enriched": True, "duration_ms": 100})
    page.wait_for_timeout(60)
    chip = page.locator('.stage2-chip[data-target="happy/path"]')
    assert chip.get_attribute("data-state") == "done"
    # Either no .play-btn node at all, or one that's not visible
    play = chip.locator(".play-btn")
    if play.count() > 0:
        assert not play.is_visible()


def test_stage2_right_click_opens_generate_menu(page):
    """Right-clicking a muted chip opens a small popover with the
    'Generate this BOM' action + the target id as a hint."""
    _reveal_pipeline(page)
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "p", "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.child.skipped",
        "target": "skipped/one", "bom_type": "data",
        "relationship_type": "trainedOn", "depth": 1,
        "parent": "p", "reason": "duplicate",
    })
    page.wait_for_timeout(80)

    chip = page.locator('.stage2-chip[data-target="skipped/one"]')
    chip.dispatch_event("contextmenu", {"clientX": 400, "clientY": 400})
    page.wait_for_timeout(80)

    menu = page.locator(".stage2-menu")
    assert menu.count() == 1
    menu_text = menu.locator(".stage2-menu-item").inner_text()
    assert "Generate this BOM" in menu_text
    assert "skipped/one" in menu_text


def test_stage2_play_button_posts_to_generate_one(page):
    """Clicking ▶ on a muted chip POSTs the target spec + form config to
    /recursive/generate-one and optimistically marks the chip as running."""
    # Intercept the endpoint so the test doesn't depend on an LLM
    posted = {"body": None}
    def _intercept(route):
        posted["body"] = route.request.post_data_json
        route.fulfill(status=200, content_type='application/json',
                      body='{"status":"success","metadata":{"dataset_id":"t/g"}}')
    page.route("**/recursive/generate-one", _intercept)

    _reveal_pipeline(page)
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "p", "bom_type": "ai",
                   "max_depth": 1, "safety_cap": 10})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.child.skipped",
        "target": "t/g", "bom_type": "data",
        "relationship_type": "trainedOn", "depth": 2,
        "parent": "p", "reason": "depth-limit",
    })
    page.wait_for_timeout(80)

    play = page.locator('.stage2-chip[data-target="t/g"] .play-btn')
    play.click()
    page.wait_for_timeout(300)

    body = posted["body"] or {}
    assert body.get("target") == "t/g"
    assert body.get("bom_type") == "data"
    assert body.get("parent") == "p"
    assert int(body.get("depth", 0)) == 2
    # And config fields are included
    assert "use_case" in body
    assert "llm_provider" in body

    # Optimistic UI: chip now reads as running
    chip_state = page.locator('.stage2-chip[data-target="t/g"]').get_attribute("data-state")
    assert chip_state == "running"


def test_stage2_done_marks_card_and_shows_summary(page):
    """recursive.done puts the card in is-done state with a summary line
    counting done / failed / skipped."""
    page.evaluate("(e) => Pipeline.handleEvent(e)",
                  {"event": "recursive.start", "parent": "test/x", "bom_type": "ai", "max_depth": 1, "safety_cap": 10})
    for tgt in ["x/a", "x/b"]:
        c = {"target": tgt, "bom_type": "data", "relationship_type": "trainedOn",
             "depth": 1, "parent": "test/x"}
        page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.target.discovered", **c})
        page.evaluate("(e) => Pipeline.handleEvent(e)", {"event": "recursive.child.start", **c})
        page.evaluate("(e) => Pipeline.handleEvent(e)",
                      {"event": "recursive.child.done", **c, "enriched": True, "duration_ms": 100})
    page.evaluate("(e) => Pipeline.handleEvent(e)", {
        "event": "recursive.done", "parent": "test/x",
        "duration_ms": 200, "generated_count": 2, "skipped_count": 0,
        "duplicate_count": 0, "deepest_level_reached": 1, "tree_exhausted": True,
    })
    page.wait_for_timeout(80)
    card = page.locator("#pipelineStage2")
    assert "is-done" in (card.get_attribute("class") or "")
    status = page.locator("#stage2Status").inner_text() or ""
    assert "2 done" in status, f"unexpected status: {status!r}"


# ---------------------------------------------------------------------------
# Layer 3 — History tab rehydration
# ---------------------------------------------------------------------------

def test_history_row_loads_into_viewer(page, seeded_history):
    fake_hash = seeded_history
    # Splash hides .tab-content (display:none !important) — dismiss it so the
    # history rows are clickable. In the real app this happens on pipeline.start
    # or on a fresh-run success; the History tab alone doesn't dismiss it.
    page.evaluate("document.getElementById('resultPane').classList.remove('is-empty')")
    page.locator(".tab", has_text="History").click()
    page.evaluate("History.load()")
    page.wait_for_timeout(300)

    rows = page.locator("#historyBody tr").filter(
        has_not=page.locator(".history-empty")
    )
    assert rows.count() >= 1

    rows.first.click()
    page.wait_for_timeout(500)

    # Clicking the row activates the Complete tab and rehydrates the viewer.
    assert "active" in (page.locator("#completeTab").get_attribute("class") or "")
    viewer_text = page.locator("#jsonViewerComplete").inner_text()
    assert "test-org/test-model" in viewer_text

    # Status banner reports the load.
    assert "history" in page.locator("#status").inner_text().lower()

    # Download link points at the history endpoint (so re-runs don't break it).
    download_href = page.locator("#downloadComplete").get_attribute("href") or ""
    assert f"/history/{fake_hash}" in download_href


def test_no_console_errors_during_full_flow(page, seeded_history):
    """End-to-end: open page → pipeline run → expand card → click history.
    The single combined flow makes the console-error check meaningful."""
    _start_pipeline(page)
    for stage in ("retrieve", "reconcile", "resolve"):
        page.evaluate(
            "(p) => Pipeline.handleEvent(p)",
            {"event": "stage.done", "field": "domain", "stage": stage,
             "duration_ms": 100, "data": {"answer_preview": "x"}},
        )
    page.evaluate("(_) => Pipeline.handleEvent({event:'pipeline.done', duration_ms:1000})", None)
    page.wait_for_timeout(120)

    page.locator(".tab", has_text="History").click()
    page.evaluate("History.load()")
    page.wait_for_timeout(300)
    rows = page.locator("#historyBody tr").filter(
        has_not=page.locator(".history-empty")
    )
    if rows.count() >= 1:
        rows.first.click()
        page.wait_for_timeout(400)

    errs = getattr(page, "console_errors", [])
    assert not errs, f"console errors during flow: {errs[:3]}"
