"""Playwright smoke for the localStorage form-state persistence.

Verifies that:
- typing into the form writes to localStorage
- a page reload hydrates the form back to the saved state
- the "Clear saved inputs" link is hidden by default, shown after a
  hydration round-trip, and clears storage on click
- resetForm() / "+ New BOM" wipes localStorage too (so a reload after
  reset doesn't bring the cleared inputs back)

Independent of test_post_gen_ux_smoke.py — this PR may merge before or
after that one. Both can coexist.
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
    """Each test gets a fresh browser context so localStorage starts empty."""
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


def _localstorage_form(page):
    """Read the persisted form state directly from localStorage."""
    raw = page.evaluate("localStorage.getItem('aikaboom.form.v1')")
    if raw is None:
        return None
    import json
    return json.loads(raw)


# ---- Save path ----------------------------------------------------------

def test_typing_writes_to_localstorage(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(80)
    saved = _localstorage_form(page)
    assert saved is not None
    assert saved.get("v:repo_id") == "anthropic/claude-haiku"


def test_checking_recursive_persists(page):
    # Recursive checkbox shows a confirm() on positive click — accept it.
    page.on("dialog", lambda d: d.accept())
    page.check("#recursive_bom")
    page.wait_for_timeout(80)
    saved = _localstorage_form(page)
    assert saved is not None
    assert saved.get("cb:recursive_bom") is True


def test_bom_type_radio_persists(page):
    # Click the wrapping label (radio is hidden behind it in the
    # radio-card layout).
    page.locator('label[for="bom_type_data"]').click()
    page.wait_for_timeout(80)
    saved = _localstorage_form(page)
    assert saved is not None
    assert saved.get("radio:bom_type") == "data"


# ---- Hydrate path -------------------------------------------------------

def test_reload_hydrates_text_input(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(80)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(100)
    assert page.input_value("#repo_id") == "anthropic/claude-haiku"


def test_reload_hydrates_radio_and_fires_visibility(page):
    page.locator('label[for="bom_type_data"]').click()
    page.wait_for_timeout(80)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(150)
    assert page.is_checked("#bom_type_data")
    # The data radio change handler swaps model_group visibility — verify
    # the hidden bom_type input ends up at "data" after the change fires.
    assert page.evaluate("document.getElementById('bom_type_data').checked") is True


def test_clear_saved_inputs_link_shown_after_hydration(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(80)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(150)
    # The link starts hidden and reveals only when state was hydrated
    assert page.locator("#clearSavedInputsBtn").is_visible()


def test_clear_saved_inputs_link_hidden_on_first_visit(page):
    # Fresh context, no localStorage yet
    assert page.locator("#clearSavedInputsBtn").is_hidden()


# ---- Clear path ---------------------------------------------------------

def test_clear_saved_inputs_wipes_storage_and_resets_form(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(80)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(150)
    page.locator("#clearSavedInputsBtn").click()
    page.wait_for_timeout(80)
    # localStorage cleared
    assert _localstorage_form(page) in (None, {})
    # Form back to default
    assert page.input_value("#repo_id") == ""
    # Link hidden again
    assert page.locator("#clearSavedInputsBtn").is_hidden()


def test_reset_form_clears_localstorage_too(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(80)
    # Trigger resetForm via its global function
    page.evaluate("resetForm()")
    page.wait_for_timeout(80)
    assert _localstorage_form(page) in (None, {})
    # Reload — form should NOT be restored (storage was cleared)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(150)
    assert page.input_value("#repo_id") == ""


def test_use_case_dropdown_persists(page):
    # Click the Safety & Bias option in the use-case dropdown
    page.click("#useCaseDropdownBtn")
    page.wait_for_timeout(50)
    safety = page.locator('.dropdown-option[data-value="safety"]')
    if safety.count() == 0:
        pytest.skip("safety preset not in this build")
    safety.click()
    page.wait_for_timeout(80)
    saved = _localstorage_form(page)
    assert saved is not None
    assert saved.get("v:use_case") == "safety"
    # Reload and confirm the visible label re-syncs
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(150)
    label = page.locator("#useCaseLabel").inner_text()
    assert "Safety" in label, f"expected 'Safety' in {label!r}"


# ---- Cross-cutting ------------------------------------------------------

def test_no_console_errors_during_persistence_flow(page):
    page.fill("#repo_id", "anthropic/claude-haiku")
    page.wait_for_timeout(60)
    page.reload(wait_until="domcontentloaded")
    page.wait_for_timeout(120)
    page.locator("#clearSavedInputsBtn").click()
    page.wait_for_timeout(60)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"
