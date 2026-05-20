"""Playwright smoke: the worldofBOMs lineage pane has four action buttons
and no SPARQL textarea. Skipped without Chromium.
"""

import json, socket, subprocess, sys, time
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


def _wait_for_http(url, timeout_s=60.0):
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
    """Boot Flask for the DOM-level smoke."""
    import os
    work = tmp_path_factory.mktemp("aibom_world_actions")
    (work / "history").mkdir()
    port = _free_port()
    env = {
        **os.environ,
        "BOM_HOST": "127.0.0.1", "BOM_PORT": str(port),
        "AIKABOOM_GRAPH_DISABLE": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "AIKABOOM_HISTORY_DIR": str(work / "history"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "aikaboom.web.app"], env=env, cwd=PROJECT_ROOT,
    )
    if not _wait_for_http(f"http://127.0.0.1:{port}/"):
        proc.kill(); pytest.fail("flask didn't start")
    yield f"http://127.0.0.1:{port}"
    proc.terminate(); proc.wait(timeout=5)


def test_lineage_pane_has_four_actions_and_no_sparql(flask_server):
    """Force-render the lineage pane with a synthetic focus and inspect the DOM."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(flask_server)
        # Open the worldofBOMs tab and force a synthetic focus + render.
        page.evaluate("""
            switchTabByName('world');
            worldFocusIri = 'urn:test:artifact:demo';
            // Minimal placeholder node so the gating logic can compute kind/claim_count.
            worldFullGraph = {
                nodes: [{iri: 'urn:test:artifact:demo', label: 'demo/node',
                         kind: 'Model', claim_count: 0}],
                edges: [],
            };
            loadWorldLineage();
        """)
        # All four action buttons present, keyed by data-world-action.
        for action in ('download-spdx-bundle', 'generate-bom',
                       'open-bom', 'rebuild-graph'):
            sel = f'[data-world-action="{action}"]'
            assert page.locator(sel).count() >= 1, f"missing action button: {action}"
        # SPARQL surface is gone.
        assert page.locator("#worldSparql").count() == 0
        assert page.locator("#worldSparqlRun").count() == 0
        # The scope chooser radios are present (4 options, 2 disabled).
        scope_radios = page.locator('input[name="genScope"]')
        assert scope_radios.count() == 4
        # The upstream + both options are disabled.
        assert page.locator('input[name="genScope"][value="upstream"]').is_disabled()
        assert page.locator('input[name="genScope"][value="both"]').is_disabled()
        # "Just this node" is checked by default for a placeholder node.
        assert page.locator('input[name="genScope"][value="node"]').is_checked()
        browser.close()
