"""Playwright smoke test for BOM display grouping.

Covers three new renderers:

1. Provenance BOM → Required / Optional / Diagnostics groups, with required
   set derived from LF SPDX AIBOM spec tables 8 (AI profile) and 10
   (Dataset profile). Identifier keys (model_id, repo_id, dataset_id) are
   shown above the groups as meta rows. Required-but-missing fields surface
   as italic warnings inside the Required group.

2. SPDX 3.0.1 → @graph items grouped by their `type` field, each item
   labeled by `name` (falling back to spdxId tail), instead of by array
   index (the previous rendering).

3. CycloneDX 1.6 → `components[]` grouped by `type`, each labeled by
   `name@version` (with bom-ref fallback). Dependencies labeled by `ref`.

Each renderer dispatches off `renderBOM(data, target)`, picked by data shape.
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
        # First-run splash hides the result pane — clear it so injected
        # data renders into a visible target.
        p.evaluate("document.getElementById('resultPane').classList.remove('is-empty')")
        yield p
        browser.close()


# ---- Fixture data --------------------------------------------------------

AI_BOM_SAMPLE = {
    "repo_id": "mistralai/Mistral-7B-v0.1",
    "model_id": "mistralai_Mistral-7B-v0.1",
    "use_case": "complete",
    "direct_fields": {
        "license": "Apache-2.0",
        "suppliedBy": "Mistral AI",
        "packageVersion": "v0.1",
        # Note: releaseTime, downloadLocation, buildTime, spdxId intentionally
        # missing — test asserts these surface as required-missing rows.
    },
    "rag_fields": {
        "model_name": "Mistral 7B",
        # Optional fields per spec table 9
        "hyperparameter": "transformer; 7B params",
        "domain": "natural language processing",
        "typeOfModel": "decoder-only LLM",
    },
    "beta_fields": ["cyclonedx", "recursive_bom"],
    "conflict_summary": {"total": 0},
}

DATA_BOM_SAMPLE = {
    "dataset_id": "rajpurkar_squad_v2",
    "use_case": "complete",
    "direct_fields": {
        "license": "CC-BY-SA-4.0",
        "originatedBy": "Rajpurkar et al.",
        "datasetType": "text",
    },
    "rag_fields": {
        "name": "SQuAD v2",
        "intendedUse": "extractive QA benchmarking",
        "datasetSize": "150k QA pairs",
    },
}

SPDX_SAMPLE = {
    "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
    "@graph": [
        {"type": "CreationInfo", "spdxId": "urn:spdx:CI-1", "created": "2026-05-13T12:00:00Z"},
        {"type": "Person",       "spdxId": "urn:spdx:Person-Claude", "name": "Claude AI"},
        {"type": "Organization", "spdxId": "urn:spdx:Org-Anthropic",  "name": "Anthropic"},
        {"type": "SpdxDocument", "spdxId": "urn:spdx:Doc-1", "name": "test-doc"},
        {"type": "Bom",          "spdxId": "urn:spdx:Bom-1"},
        {"type": "ai_AIPackage", "spdxId": "urn:spdx:AI-llama3", "name": "llama-3-8b"},
        {"type": "dataset_DatasetPackage", "spdxId": "urn:spdx:DS-squad", "name": "squad"},
        {"type": "Relationship", "spdxId": "urn:spdx:R-1",
         "relationshipType": "trainedOn",
         "from": "urn:spdx:AI-llama3",
         "to":   ["urn:spdx:DS-squad"]},
        {"type": "Relationship", "spdxId": "urn:spdx:R-2",
         "relationshipType": "hasDeclaredLicense",
         "from": "urn:spdx:AI-llama3",
         "to":   ["urn:spdx:Lic-1"]},
        {"type": "simplelicensing_LicenseExpression",
         "spdxId": "urn:spdx:Lic-1",
         "simplelicensing_licenseExpression": "Apache-2.0"},
    ],
}

CDX_SAMPLE = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "version": 1,
    "serialNumber": "urn:uuid:abc-123",
    "metadata": {
        "timestamp": "2026-05-13T12:00:00Z",
        "tools": [{"vendor": "Anthropic", "name": "AIkaBoOM"}],
    },
    "components": [
        {"type": "machine-learning-model", "bom-ref": "m1", "name": "llama-3-8b", "version": "1.0"},
        {"type": "data", "bom-ref": "d1", "name": "squad", "version": "v2"},
        {"type": "library", "bom-ref": "l1", "name": "transformers", "version": "4.40.0"},
    ],
    "dependencies": [
        {"ref": "m1", "dependsOn": ["d1", "l1"]},
    ],
}


# ---- 1. Provenance BOM grouping -----------------------------------------

def test_ai_bom_renders_required_and_optional_groups(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    titles = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group .bom-group-title",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "Required" in titles
    assert "Optional" in titles


def test_ai_bom_required_group_lists_missing_fields(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    # Required group is open by default — its body should contain entries
    # for every spec-mandatory field, including missing ones.
    page.locator(
        "#flatViewerComplete .bom-group:has(.bom-group-title:text('Required'))"
    ).first.wait_for(state="visible")
    rows = page.locator(
        "#flatViewerComplete .bom-group:has(.bom-group-title:text('Required')) .flat-row"
    )
    keys = [rows.nth(i).locator(".flat-key").inner_text() for i in range(rows.count())]
    # Sanity: at least one present (license) and one missing (releaseTime).
    assert "license" in keys
    assert "releaseTime" in keys
    # Missing fields should carry the is-missing class
    missing_keys = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group .flat-row.is-missing .flat-key",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "releaseTime" in missing_keys
    assert "downloadLocation" in missing_keys


def test_ai_bom_required_meta_count_reflects_presence(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    meta = page.locator(
        "#flatViewerComplete .bom-group:has(.bom-group-title:text('Required')) .bom-group-meta"
    ).first.inner_text()
    # Direct (3: license, suppliedBy, packageVersion) + RAG (1: model_name) = 4 of 9 present
    assert "4 of 9 present" in meta


def test_ai_bom_optional_group_contains_rag_fields(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    # Open the Optional group programmatically — clicking through Playwright
    # would require the tab pane to be the visible one (it is here, but the
    # group-body display:none triggers a 'not visible' click failure).
    page.evaluate("""
      Array.from(document.querySelectorAll('#flatViewerComplete .bom-group'))
        .filter(g => g.querySelector('.bom-group-title')?.textContent.trim() === 'Optional')
        .forEach(g => g.classList.add('is-open'));
    """)
    keys = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group .flat-row .flat-key",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "hyperparameter" in keys
    assert "domain" in keys
    assert "typeOfModel" in keys


def test_ai_bom_identifier_strip_shows_model_id(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    # identifier rows are .bom-group-header.is-meta — assert model_id is one
    metas = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group-header.is-meta .flat-key",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "model_id" in metas
    assert "repo_id" in metas


def test_ai_bom_diagnostics_group_present(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    titles = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group .bom-group-title",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "Diagnostics" in titles


def test_data_bom_uses_dataset_required_set(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        DATA_BOM_SAMPLE,
    )
    # Dataset profile requires originatedBy + datasetType (which AI doesn't)
    keys = page.eval_on_selector_all(
        "#flatViewerComplete .bom-group:has(.bom-group-title:text('Required')) .flat-row .flat-key",
        "els => els.map(e => e.textContent.trim())",
    )
    assert "originatedBy" in keys
    assert "datasetType" in keys
    # AI-only key should NOT be in required for a data BOM
    assert "suppliedBy" not in keys


# ---- 2. SPDX grouping ----------------------------------------------------

def test_spdx_groups_graph_by_type(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerSPDX'))",
        SPDX_SAMPLE,
    )
    titles = page.eval_on_selector_all(
        "#flatViewerSPDX .bom-group .bom-group-title",
        "els => els.map(e => e.textContent.trim())",
    )
    # Should have one group per type in the @graph
    for expected in ("CreationInfo", "Person", "Organization", "SpdxDocument",
                     "Bom", "ai_AIPackage", "dataset_DatasetPackage",
                     "Relationship", "simplelicensing_LicenseExpression"):
        assert expected in titles, f"missing SPDX type group: {expected!r} (got {titles!r})"


def _open_all_groups(page, container_id):
    page.evaluate(f"""
      document.querySelectorAll('#{container_id} .bom-group')
        .forEach(g => g.classList.add('is-open'));
    """)


def test_spdx_items_labeled_by_name_not_index(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerSPDX'))",
        SPDX_SAMPLE,
    )
    _open_all_groups(page, "flatViewerSPDX")
    # Find the ai_AIPackage group's rows specifically
    keys = page.evaluate("""
      (() => {
        const g = [...document.querySelectorAll('#flatViewerSPDX .bom-group')]
            .find(g => g.querySelector('.bom-group-title')?.textContent.trim() === 'ai_AIPackage');
        if (!g) return [];
        return [...g.querySelectorAll('.flat-row .flat-key')].map(e => e.textContent.trim());
      })()
    """)
    # The single AI package should be keyed by its name, not "0"
    assert "llama-3-8b" in keys, f"expected llama-3-8b in {keys!r}"
    assert "0" not in keys


def test_spdx_relationship_label_shows_type_and_endpoints(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerSPDX'))",
        SPDX_SAMPLE,
    )
    _open_all_groups(page, "flatViewerSPDX")
    keys = page.evaluate("""
      (() => {
        const g = [...document.querySelectorAll('#flatViewerSPDX .bom-group')]
            .find(g => g.querySelector('.bom-group-title')?.textContent.trim() === 'Relationship');
        if (!g) return [];
        return [...g.querySelectorAll('.flat-row .flat-key')].map(e => e.textContent.trim());
      })()
    """)
    labels = " | ".join(keys)
    assert "trainedOn" in labels
    assert "→" in labels
    assert "AI-llama3" in labels or "llama3" in labels


def test_spdx_context_rendered_as_link(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerSPDX'))",
        SPDX_SAMPLE,
    )
    href = page.eval_on_selector(
        "#flatViewerSPDX .bom-group-header.is-meta a",
        "a => a.getAttribute('href')",
    )
    assert href == SPDX_SAMPLE["@context"]


# ---- 3. CycloneDX grouping -----------------------------------------------

def test_cdx_components_grouped_by_type(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerCDX'))",
        CDX_SAMPLE,
    )
    titles = page.eval_on_selector_all(
        "#flatViewerCDX .bom-group .bom-group-title",
        "els => els.map(e => e.textContent.trim())",
    )
    # Each component type becomes its own group titled `components · <type>`
    assert "components · machine-learning-model" in titles
    assert "components · data" in titles
    assert "components · library" in titles


def test_cdx_components_labeled_name_at_version(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerCDX'))",
        CDX_SAMPLE,
    )
    _open_all_groups(page, "flatViewerCDX")
    keys = page.evaluate("""
      (() => {
        const g = [...document.querySelectorAll('#flatViewerCDX .bom-group')]
            .find(g => g.querySelector('.bom-group-title')?.textContent.trim()
                       === 'components · machine-learning-model');
        if (!g) return [];
        return [...g.querySelectorAll('.flat-row .flat-key')].map(e => e.textContent.trim());
      })()
    """)
    assert "llama-3-8b@1.0" in keys, f"got {keys!r}"


def test_cdx_dependencies_labeled_by_ref(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerCDX'))",
        CDX_SAMPLE,
    )
    _open_all_groups(page, "flatViewerCDX")
    keys = page.evaluate("""
      (() => {
        const g = [...document.querySelectorAll('#flatViewerCDX .bom-group')]
            .find(g => g.querySelector('.bom-group-title')?.textContent.trim() === 'dependencies');
        if (!g) return [];
        return [...g.querySelectorAll('.flat-row .flat-key')].map(e => e.textContent.trim());
      })()
    """)
    assert "m1" in keys


def test_cdx_document_meta_row_shows_format_and_version(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerCDX'))",
        CDX_SAMPLE,
    )
    text = page.locator(
        "#flatViewerCDX .bom-group-header.is-meta .flat-val"
    ).first.inner_text()
    assert "CycloneDX" in text
    assert "spec v1.6" in text


# ---- 4. Cross-cutting ----------------------------------------------------

def test_no_console_errors_during_renders(page):
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerComplete'))",
        AI_BOM_SAMPLE,
    )
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerSPDX'))",
        SPDX_SAMPLE,
    )
    page.evaluate(
        "(d) => renderBOM(d, document.getElementById('flatViewerCDX'))",
        CDX_SAMPLE,
    )
    page.wait_for_timeout(100)
    errs = getattr(page, "console_errors", [])
    assert not errs, f"unexpected console errors: {errs}"


def test_dispatcher_falls_back_to_flat_for_unknown_shape(page):
    # Bare dict with no spec markers — should render via the legacy flat path.
    page.evaluate(
        "renderBOM({a: 1, b: 'hello'}, document.getElementById('flatViewerComplete'))"
    )
    # No group sections; just flat rows directly under the container
    direct_rows = page.locator("#flatViewerComplete > .flat-row").count()
    assert direct_rows >= 2
    group_count = page.locator("#flatViewerComplete > .bom-group").count()
    assert group_count == 0
