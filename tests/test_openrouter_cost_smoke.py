"""Playwright smoke test for the OpenRouter catalog dropdown.

Verifies that when the user clicks "Load model catalog", each model in
the resulting <select> shows its inference cost ($/Mtok) alongside the
name and context length, and that free / partially-priced models render
without crashing.

The test intercepts /models?provider=openrouter via Playwright's network
route mocking so it doesn't hit the real OpenRouter API.

Skipped when Playwright (Python package + Chromium) isn't installed —
matches the conditional skip in test_pipeline_ui_smoke.py.
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
    """Boot the worktree's Flask app in a subprocess on a free port."""
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


# Fixture catalog covering the four interesting cases:
#   - paid model with low cost (sub-$1/Mtok → 2 decimals)
#   - paid model with mid cost (≥$10/Mtok → 1 decimal)
#   - paid model with high cost (≥$100/Mtok → integer)
#   - free model (both prices zero)
FIXTURE_MODELS = [
    {
        "id": "test/cheap",
        "name": "Cheap Model",
        "context_length": 8192,
        "pricing": {"prompt": "0.0000005", "completion": "0.0000007"},
    },
    {
        "id": "test/mid",
        "name": "Mid Model",
        "context_length": 128000,
        "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
    },
    {
        "id": "test/premium",
        "name": "Premium Model",
        "context_length": 200000,
        "pricing": {"prompt": "0.00015", "completion": "0.0006"},
    },
    {
        "id": "test/free",
        "name": "Free Model",
        "context_length": 4096,
        "pricing": {"prompt": "0", "completion": "0"},
    },
]


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
        # Intercept the OpenRouter catalog endpoint with our fixture
        p.route(
            "**/models?provider=openrouter",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"models": FIXTURE_MODELS}),
            ),
        )
        p.console_errors = errors  # type: ignore[attr-defined]
        p.goto(flask_server, wait_until="domcontentloaded")
        yield p
        browser.close()


def _load_catalog(page):
    page.locator("#btn_load_all_models").click()
    # Wait until the select has populated past its single default option
    page.wait_for_function(
        "() => document.getElementById('openrouter_model_select').options.length > 2",
        timeout=5000,
    )


def _option_text_for(page, value: str) -> str:
    return page.eval_on_selector(
        f"#openrouter_model_select option[value='{value}']",
        "el => el.textContent",
    )


def test_option_text_shows_cheap_model_price(page):
    _load_catalog(page)
    text = _option_text_for(page, "test/cheap")
    # 0.0000005 $/tok → $0.50/Mtok input; 0.0000007 → $0.70/Mtok output
    assert "$0.50 in" in text, f"missing input price in option text: {text!r}"
    assert "$0.70 out" in text, f"missing output price in option text: {text!r}"
    assert "per Mtok" in text


def test_option_text_shows_mid_model_one_decimal(page):
    _load_catalog(page)
    text = _option_text_for(page, "test/mid")
    # 0.0000025 → $2.50/Mtok (2 decimals); 0.00001 → $10.0/Mtok (1 decimal)
    assert "$2.50 in" in text
    assert "$10.0 out" in text


def test_option_text_shows_premium_model_integer(page):
    _load_catalog(page)
    text = _option_text_for(page, "test/premium")
    # 0.00015 → $150/Mtok (integer ≥100); 0.0006 → $600/Mtok
    assert "$150 in" in text
    assert "$600 out" in text


def test_free_model_shows_free_label(page):
    _load_catalog(page)
    text = _option_text_for(page, "test/free")
    assert "free" in text.lower()
    # Free models should NOT show a $ price
    assert "$" not in text


def test_context_length_still_displayed(page):
    _load_catalog(page)
    # Existing display: ctx label should still be present alongside the price
    text = _option_text_for(page, "test/cheap")
    assert "ctx: 8K" in text
    text2 = _option_text_for(page, "test/mid")
    assert "ctx: 128K" in text2


def test_no_console_errors_during_catalog_load(page):
    _load_catalog(page)
    # Spend a tick letting any async errors surface
    page.wait_for_timeout(100)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"


def test_selecting_priced_model_sets_hidden_input(page):
    _load_catalog(page)
    page.select_option("#openrouter_model_select", "test/premium")
    value = page.eval_on_selector("#openrouter_model", "el => el.value")
    assert value == "test/premium"
