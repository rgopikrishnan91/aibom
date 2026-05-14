"""Playwright smoke for post-generation tab auto-refocus.

After a `/process` response lands, `refocusAfterGeneration()` swings the
user from the Pipeline tab back to the BOM (`complete`) tab — but only
if they're still on Pipeline. If they navigated elsewhere during the
run (Logs, Conflicts, History), we respect that.

This test calls `refocusAfterGeneration()` directly with simulated
`currentTab` values; the real wiring (`/process` success handler →
refocusAfterGeneration()) is verified by reading the source.
"""

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


@pytest.fixture
def page(flask_server):
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
        # Splash hides the tabs — clear it so switchTabByName has visible
        # targets to land on.
        p.evaluate("document.getElementById('resultPane').classList.remove('is-empty')")
        yield p
        browser.close()


# ---- Wiring (source-level) ----------------------------------------------

def test_refocus_helper_is_defined(page):
    assert page.evaluate("typeof refocusAfterGeneration") == "function"


def test_response_handler_calls_refocus():
    """Sanity: the success branch of the /process response handler must
    invoke `refocusAfterGeneration()`, otherwise the user-visible behaviour
    is dead code."""
    template = (PROJECT_ROOT / "src/aikaboom/web/templates/index.html").read_text()
    # Look for the call inside the success branch, *not* the helper
    # definition itself.
    # The helper is named `refocusAfterGeneration`; one definition + one
    # call inside the response handler ⇒ at least 2 occurrences.
    assert template.count("refocusAfterGeneration") >= 2, (
        "expected refocusAfterGeneration to be defined AND called from the "
        "/process response handler"
    )


# ---- Behaviour ----------------------------------------------------------

def test_refocus_switches_when_user_on_pipeline(page):
    # Stage: user is on the Pipeline tab (pipeline.start put them there).
    page.evaluate("currentTab = 'pipeline'; switchTabByName('pipeline')")
    assert page.evaluate("currentTab") == "pipeline"
    page.evaluate("refocusAfterGeneration()")
    # 700 ms scheduled inside; give it 900 ms to fire.
    page.wait_for_timeout(900)
    assert page.evaluate("currentTab") == "complete"


def test_refocus_no_op_when_user_on_other_tab(page):
    # User wandered to Logs during the run.
    page.evaluate("switchTabByName('logs')")
    assert page.evaluate("currentTab") == "logs"
    page.evaluate("refocusAfterGeneration()")
    page.wait_for_timeout(900)
    # Should stay on Logs — refocus respects manual navigation.
    assert page.evaluate("currentTab") == "logs"


def test_refocus_no_op_when_user_navigated_away_mid_delay(page):
    """Edge case: user is on Pipeline when refocus is invoked, but
    switches to Logs during the 700 ms delay. The inner re-check should
    detect this and skip the switch."""
    page.evaluate("switchTabByName('pipeline')")
    assert page.evaluate("currentTab") == "pipeline"
    # Kick off the refocus, then immediately yank the user to Logs.
    page.evaluate(
        """
        refocusAfterGeneration();
        setTimeout(() => switchTabByName('logs'), 100);
        """
    )
    # Wait past the 700 ms refocus window.
    page.wait_for_timeout(900)
    assert page.evaluate("currentTab") == "logs", (
        "refocus should not have switched to complete after user moved to logs"
    )


def test_refocus_no_op_when_user_on_complete_already(page):
    # If they're already on Complete, refocus has nothing to do.
    page.evaluate("switchTabByName('complete')")
    page.evaluate("refocusAfterGeneration()")
    page.wait_for_timeout(900)
    assert page.evaluate("currentTab") == "complete"


# ---- Cross-cutting ------------------------------------------------------

def test_no_console_errors_during_refocus(page):
    page.evaluate("switchTabByName('pipeline')")
    page.evaluate("refocusAfterGeneration()")
    page.wait_for_timeout(900)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"
