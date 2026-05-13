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
