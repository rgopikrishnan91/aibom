"""Playwright smoke for the post-generation continuation flow.

Covers three small UX additions:

1. **`+ New BOM` button** in the result-pane header. Hidden when the
   splash is up, visible once a result lands, and on click clears the
   form (with a confirm only when the form was edited from defaults)
   then surfaces an Undo snackbar.

2. **History banner**. When a history row is loaded, a sticky banner
   appears above the tabs with two actions:
   - *Use as template* — pre-fills the form (bom_type + repo/dataset id
     + use_case) but does not run.
   - *Back to current work* — restores the last fresh-gen, or drops the
     result pane back to the splash.

3. **Reset confirm + undo snackbar**. Form's existing `Reset` button
   now goes through the same gate as `+ New BOM`: confirm only when
   the form was edited; snackbar offers an Undo within 5 s that
   restores the pre-reset snapshot.

Skipped if Playwright + Chromium aren't installed locally.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
def flask_server():
    port = _free_port()
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


AI_BOM_SAMPLE = {
    "repo_id": "mistralai/Mistral-7B-v0.1",
    "model_id": "mistralai_Mistral-7B-v0.1",
    "use_case": "complete",
    "direct_fields": {"license": "Apache-2.0"},
    "rag_fields":    {"model_name": "Mistral 7B"},
}


@pytest.fixture
def page(flask_server):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        # Auto-accept native confirm dialogs so the test can assert the
        # reset path; we'll override per-test for the "skip when unedited"
        # case below.
        p = ctx.new_page()
        errors = []
        p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        p.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        p.console_errors = errors  # type: ignore[attr-defined]
        p.goto(flask_server, wait_until="domcontentloaded")
        yield p
        browser.close()


def _simulate_fresh_gen(page):
    """Pretend a fresh BOM came back from /process — flip the splash off,
    populate the viewers, capture lastFreshGenState. Mirrors what the real
    result-handler does, without needing an LLM."""
    page.evaluate(
        """
        (d) => {
            lastFreshGenState = {
                metadata: d, spdx: null, cyclonedx: null
            };
            viewMode = 'fresh';
            formIsEdited = false;
            renderBOM(d, document.getElementById('flatViewerComplete'));
            displayJSON(d, document.getElementById('jsonViewerComplete'));
            document.getElementById('resultPane').classList.remove('is-empty');
        }
        """,
        AI_BOM_SAMPLE,
    )


# ---- #1: + New BOM button -----------------------------------------------

def test_new_bom_button_hidden_before_result(page):
    assert page.locator("#newBomBtn").is_hidden(), "should be hidden when splash is up"


def test_new_bom_button_visible_after_result(page):
    _simulate_fresh_gen(page)
    assert page.locator("#newBomBtn").is_visible()


def test_new_bom_no_confirm_when_form_clean(page):
    _simulate_fresh_gen(page)
    # No confirm should fire because formIsEdited was reset by the gen.
    confirm_fired = {"count": 0}
    page.on("dialog", lambda d: (confirm_fired.__setitem__("count", confirm_fired["count"] + 1), d.accept()))
    page.locator("#newBomBtn").click()
    page.wait_for_timeout(120)
    assert confirm_fired["count"] == 0, "should not confirm when form was clean"
    # Splash is back
    assert "is-empty" in (page.locator("#resultPane").get_attribute("class") or "")


def test_new_bom_confirms_when_form_edited(page):
    _simulate_fresh_gen(page)
    # Type something into a form input — dirties the form.
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(50)
    seen = {"count": 0}
    page.on("dialog", lambda d: (seen.__setitem__("count", seen["count"] + 1), d.accept()))
    page.locator("#newBomBtn").click()
    page.wait_for_timeout(120)
    assert seen["count"] == 1, "should confirm when form was edited"


def test_new_bom_shows_undo_snackbar(page):
    _simulate_fresh_gen(page)
    page.on("dialog", lambda d: d.accept())  # accept any confirm
    page.locator("#newBomBtn").click()
    page.wait_for_timeout(120)
    assert page.locator("#undoSnackbar").is_visible()
    txt = page.locator("#undoSnackbarText").inner_text()
    assert "cleared" in txt.lower()


def test_new_bom_undo_restores_form_inputs(page):
    _simulate_fresh_gen(page)
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.on("dialog", lambda d: d.accept())
    page.locator("#newBomBtn").click()
    page.wait_for_timeout(120)
    assert page.input_value("#repo_id") == "", "form should be cleared"
    page.locator("#undoSnackbarBtn").click()
    page.wait_for_timeout(80)
    assert page.input_value("#repo_id") == "anthropic/claude-haiku", "Undo should restore"
    assert page.locator("#undoSnackbar").is_hidden()


# ---- #2: History banner --------------------------------------------------

def _seed_history_row(page, hash_value="abc123", subject="mistralai/Mistral-7B-v0.1"):
    """Inject a fake history-rehydration directly via the History namespace.
    Avoids needing a real bom-history/ on disk for the banner-state tests."""
    page.evaluate(
        """
        (args) => {
            // Touch the private state via the public loadInto path is
            // overkill for a banner test — set state directly.
            History.upsert({
                hash: args.hash, subject: args.subject,
                bom_type: 'ai', created_at: '2026-05-14T10:00:00Z', artifacts: {}
            });
            // Mirror what loadInto would do, minus the network round-trip.
            renderBOM(args.bom, document.getElementById('flatViewerComplete'));
            displayJSON(args.bom, document.getElementById('jsonViewerComplete'));
            document.getElementById('resultPane').classList.remove('is-empty');
            // Drive the banner via the same path loadInto uses.
            viewMode = 'history';
            showHistoryBanner(args.subject);
            // Stuff the loaded-row caches so Use-as-template can read them.
            History.__test_setLoaded(
                { hash: args.hash, subject: args.subject, bom_type: 'ai' },
                args.bom
            );
        }
        """,
        {"hash": hash_value, "subject": subject, "bom": AI_BOM_SAMPLE},
    )


def test_history_banner_appears_on_load(page):
    # Add a test-only setter that the smoke tests can call without poking
    # privates. This is the cleanest way to bridge the IIFE-private state.
    page.evaluate("""
      History.__test_setLoaded = (row, bom) => {
          // Mutate the closure's `_lastLoadedRow` / `_lastLoadedBom` by
          // calling the same internal hook loadInto uses. We do that by
          // dispatching a fake loadInto if available, otherwise stashing
          // on the History object directly (smoke-test bridge only).
          Object.defineProperty(History, '_lastLoadedRow', { value: row, configurable: true });
          Object.defineProperty(History, '_lastLoadedBom', { value: bom, configurable: true });
      };
    """)
    _seed_history_row(page)
    assert page.locator("#historyBanner").is_visible()
    subj = page.locator("#historyBannerSubject").inner_text()
    assert "Mistral" in subj


def test_use_as_template_seeds_form(page):
    page.evaluate("""
      History.__test_setLoaded = (row, bom) => {
          Object.defineProperty(History, '_lastLoadedRow', { value: row, configurable: true });
          Object.defineProperty(History, '_lastLoadedBom', { value: bom, configurable: true });
      };
    """)
    _seed_history_row(page)
    page.locator("#useAsTemplateBtn").click()
    page.wait_for_timeout(80)
    # bom_type radio flipped to ai (already default, but check explicitly)
    assert page.is_checked("#bom_type_ai")
    # repo_id field pre-filled from the row's subject
    assert page.input_value("#repo_id") == "mistralai/Mistral-7B-v0.1"
    # use_case hydrated from the loaded BOM's metadata
    assert page.input_value("#use_case") == "complete"
    # Banner gone
    assert page.locator("#historyBanner").is_hidden()


def test_back_to_current_work_restores_fresh_gen(page):
    # 1. Generate a fresh BOM
    _simulate_fresh_gen(page)
    # 2. Now load from history (banner appears, viewMode = history)
    page.evaluate("""
      History.__test_setLoaded = (row, bom) => {
          Object.defineProperty(History, '_lastLoadedRow', { value: row, configurable: true });
          Object.defineProperty(History, '_lastLoadedBom', { value: bom, configurable: true });
      };
    """)
    page.evaluate(
        """
        (args) => {
            renderBOM(args.bom, document.getElementById('flatViewerComplete'));
            viewMode = 'history';
            showHistoryBanner(args.subject);
        }
        """,
        {"subject": "different/model", "bom": {"model_id": "different-one"}},
    )
    assert page.locator("#historyBanner").is_visible()
    # 3. Click Back — banner hides, viewMode flips to fresh
    page.locator("#backToWorkBtn").click()
    page.wait_for_timeout(120)
    assert page.locator("#historyBanner").is_hidden()
    view = page.evaluate("viewMode")
    assert view == "fresh"


def test_back_to_current_work_falls_back_to_splash_when_no_fresh(page):
    page.evaluate("""
      History.__test_setLoaded = (row, bom) => {
          Object.defineProperty(History, '_lastLoadedRow', { value: row, configurable: true });
          Object.defineProperty(History, '_lastLoadedBom', { value: bom, configurable: true });
      };
    """)
    _seed_history_row(page)
    # lastFreshGenState is still null at this point (no _simulate_fresh_gen yet)
    page.locator("#backToWorkBtn").click()
    page.wait_for_timeout(80)
    # Result pane should be back in splash mode
    assert "is-empty" in (page.locator("#resultPane").get_attribute("class") or "")
    view = page.evaluate("viewMode")
    assert view == "splash"


# ---- #3: Reset confirm + undo (covered above via same gate) ------------

def test_reset_button_uses_same_gate(page):
    """The form's existing Reset button should route through requestReset
    so it shares the confirm + undo behaviour with `+ New BOM`."""
    _simulate_fresh_gen(page)
    page.fill("#repo_id", "anthropic/claude-haiku")
    seen = {"count": 0}
    page.on("dialog", lambda d: (seen.__setitem__("count", seen["count"] + 1), d.accept()))
    page.locator('button.btn.btn-ghost', has_text="Reset").click()
    page.wait_for_timeout(120)
    assert seen["count"] == 1
    assert page.locator("#undoSnackbar").is_visible()


# ---- 4. Cross-cutting ----------------------------------------------------

def test_no_console_errors_during_ux_flow(page):
    _simulate_fresh_gen(page)
    page.on("dialog", lambda d: d.accept())
    page.locator("#newBomBtn").click()
    page.wait_for_timeout(100)
    page.locator("#undoSnackbarBtn").click()
    page.wait_for_timeout(100)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"
