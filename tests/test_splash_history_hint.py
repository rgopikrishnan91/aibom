"""Playwright smoke: the splash card shows a link to open past BOMs when
bom-history/ has rows, and hides it when empty.

Skipped if Chromium isn't installed locally — same pattern as
tests/test_pipeline_ui_smoke.py.
"""

import json
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
    import os
    work = tmp_path_factory.mktemp("aibom_splash_hint")
    history = work / "bom-history"
    history.mkdir()
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
        **os.environ,
        "BOM_HOST": "127.0.0.1", "BOM_PORT": str(port),
        "AIKABOOM_GRAPH_DISABLE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "AIKABOOM_HISTORY_DIR": str(history),
    }
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


@pytest.fixture(scope="module")
def flask_with_empty_history(tmp_path_factory):
    """Boot Flask with an empty bom-history (just an empty index.json)."""
    import os
    work = tmp_path_factory.mktemp("aibom_splash_hint_empty")
    history = work / "bom-history"
    history.mkdir()
    (history / "index.json").write_text("[]")

    port = _free_port()
    env = {
        **os.environ,
        "BOM_HOST": "127.0.0.1", "BOM_PORT": str(port),
        "AIKABOOM_GRAPH_DISABLE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "AIKABOOM_HISTORY_DIR": str(history),
    }
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


def test_splash_hint_hidden_when_history_is_empty(flask_with_empty_history):
    """No past BOMs -> the hint must NOT appear on the splash."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(flask_with_empty_history)
        # Wait for History.load() to settle (it sets rows = [] in this case).
        page.wait_for_function(
            "window.History && Array.isArray(window.History.rows)")
        hint = page.locator("#resultEmpty #splashHistoryHint")
        # Element exists in the DOM (hidden attr set) but should not be visible.
        assert hint.count() == 1
        assert hint.is_hidden(), "splash history hint should stay hidden when N=0"
        browser.close()


def test_splash_hint_visible_with_one_history_row(flask_with_history):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(flask_with_history)
        # Wait for History.load() to populate.
        page.wait_for_function(
            "window.History && window.History.rows && window.History.rows.length >= 1")
        hint = page.locator("#resultEmpty #splashHistoryHint")
        assert hint.is_visible(), "splash history hint should appear when N>=1"
        assert "past bom" in hint.inner_text().lower()
        # Clicking the arrow link should swap the splash for the History panel.
        hint.locator("a.splash-history-link").click()
        page.wait_for_selector("#historyTab.tab-content.active")
        browser.close()
