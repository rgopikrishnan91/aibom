"""Playwright smoke for the consolidated Download dropdown.

The 5-button strip (Provenance / SPDX / CycloneDX / Recursive / Linked SPDX)
collapsed into a single `Download ▾` menu. Each menu item retains its
per-format href and the existing per-item display toggling (set by the
/process success handler + History rehydration), so a format that's
absent from this run stays hidden in the menu.

Smoke covers: trigger visibility (gated by .is-empty), menu open/close
on toggle click, outside-click closes, Escape closes, item click closes,
per-item href + visibility plumbing, divider separates standard from
beta items.
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
        yield p
        browser.close()


def _simulate_post_gen(page, *, with_spdx=True, with_cdx=True,
                       with_recursive=False, with_linked=False):
    """Drive the post-generation state machine without an actual LLM call:
    flip the splash off, expose the strip, and set per-item href + display
    on the menu items the same way the real /process handler does."""
    page.evaluate(
        """
        (opts) => {
            document.getElementById('resultPane').classList.remove('is-empty');
            const strip = document.getElementById('downloadButtons');
            if (strip) strip.style.display = 'flex';
            const show = (id, href) => {
                const el = document.getElementById(id);
                if (!el) return;
                el.href = href;
                el.style.display = 'flex';
            };
            show('downloadComplete', '/download/test-bom.json');
            if (opts.spdx)      show('downloadSPDX',      '/download/test-spdx.json');
            if (opts.cdx)       show('downloadCDX',       '/download/test-cdx.json');
            if (opts.recursive) show('downloadRecursive', '/download/test-recursive.json');
            if (opts.linked)    show('downloadLinked',    '/download/test-linked.json');
        }
        """,
        {"spdx": with_spdx, "cdx": with_cdx,
         "recursive": with_recursive, "linked": with_linked},
    )


# ---- Trigger visibility -------------------------------------------------

def test_trigger_hidden_before_first_run(page):
    # is-empty rule on .result-pane hides the strip until a BOM lands.
    assert page.locator("#downloadToggle").is_hidden()


def test_trigger_visible_after_first_run(page):
    _simulate_post_gen(page)
    assert page.locator("#downloadToggle").is_visible()


# ---- Open / close -------------------------------------------------------

def test_menu_starts_closed(page):
    _simulate_post_gen(page)
    assert page.locator("#downloadMenu").is_hidden()


def test_toggle_opens_menu(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(50)
    assert page.locator("#downloadMenu").is_visible()
    assert page.locator("#downloadToggle").get_attribute("aria-expanded") == "true"


def test_toggle_closes_open_menu(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    assert page.locator("#downloadMenu").is_hidden()
    assert page.locator("#downloadToggle").get_attribute("aria-expanded") == "false"


def test_outside_click_closes_menu(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    # Click somewhere outside the dropdown
    page.locator(".pane-title").first.click()
    page.wait_for_timeout(40)
    assert page.locator("#downloadMenu").is_hidden()


def test_escape_closes_menu(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    page.keyboard.press("Escape")
    page.wait_for_timeout(40)
    assert page.locator("#downloadMenu").is_hidden()


def test_item_click_closes_menu(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    # Suppress the navigation that the real anchor click would trigger
    # — we only care that the menu closes.
    page.evaluate("""
        document.querySelectorAll('.download-item').forEach(a => {
            a.addEventListener('click', e => e.preventDefault());
        });
    """)
    page.locator("#downloadComplete").click()
    page.wait_for_timeout(120)
    assert page.locator("#downloadMenu").is_hidden()


# ---- Per-item plumbing --------------------------------------------------

def test_present_items_show_href_and_title(page):
    _simulate_post_gen(page, with_spdx=True, with_cdx=True)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    # Provenance always shown
    assert page.locator("#downloadComplete").is_visible()
    assert page.locator("#downloadComplete").get_attribute("href") == "/download/test-bom.json"
    title = page.locator("#downloadComplete .download-item-title").inner_text()
    # The simulated post-gen doesn't run the textContent update, so the
    # default markup-time title text survives.
    assert "Provenance BOM" in title
    # SPDX present
    assert page.locator("#downloadSPDX").is_visible()
    # CDX present
    assert page.locator("#downloadCDX").is_visible()


def test_absent_items_stay_hidden(page):
    _simulate_post_gen(page, with_spdx=True, with_cdx=False,
                       with_recursive=False, with_linked=False)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    assert page.locator("#downloadCDX").is_hidden()
    assert page.locator("#downloadRecursive").is_hidden()
    assert page.locator("#downloadLinked").is_hidden()


def test_divider_separates_standard_from_beta(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    # The divider sits between SPDX (standard) and CycloneDX (beta) in DOM order
    ordering = page.evaluate("""
      [...document.querySelectorAll('#downloadMenu > *')]
        .map(el => el.id || el.className)
    """)
    # Find indices of #downloadSPDX, the divider, and #downloadCDX
    spdx_idx = next(i for i, x in enumerate(ordering) if x == "downloadSPDX")
    cdx_idx  = next(i for i, x in enumerate(ordering) if x == "downloadCDX")
    div_idx  = next(i for i, x in enumerate(ordering) if "download-divider" in x)
    assert spdx_idx < div_idx < cdx_idx, (
        f"expected SPDX → divider → CDX, got order: {ordering}"
    )


# ---- Cross-cutting ------------------------------------------------------

def test_no_console_errors_during_dropdown_flow(page):
    _simulate_post_gen(page)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    page.keyboard.press("Escape")
    page.wait_for_timeout(40)
    page.locator("#downloadToggle").click()
    page.wait_for_timeout(40)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"
