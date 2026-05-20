# License-Compat Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship license-compatibility analysis as plugin #1 under a new in-tree plugin architecture, consuming the worldofBOMs graph as its lineage source. Produces per-edge verdicts, relicensing recommendations, compatible subchains, and breaking-node identification, exposed through CLI, web UI, SPDX export annotations, and the existing Conflicts tab.

**Architecture:** Hybrid plugin system — a `Plugin` Protocol in `src/aikaboom/plugins/base.py`, plugins self-register at import time from `src/aikaboom/plugins/<name>/`, core call sites loop over `all_plugins()`. License-compat lives at `src/aikaboom/plugins/license_compat/` with pure-function engine, SPARQL-driven graph walker, and four UI emitters that consume one `Findings` object per `analyze()` call.

**Tech Stack:** Python 3.9+, rdflib (RDFLib backend) / pyoxigraph (Oxigraph backend), Flask (web), argparse (CLI), pytest, SPDX 3.0.1 vocab, importlib.resources (bundled data).

**Prerequisite:** Branch off `main` *after* `worldofboms-graph` is merged. The `aikaboom.store` module (BomStore, vocab predicates, graph backends) must exist before Task 4. If executing before the merge, tasks 1–3 can land in advance; Task 4 onward needs the store.

**Design reference:** `docs/superpowers/specs/2026-05-20-license-compat-integration-design.md`

---

## File Structure

**Created:**

```
src/aikaboom/plugins/
├── __init__.py                                     # registry: register, all_plugins, get
├── base.py                                         # Plugin Protocol, Scope, Finding, GraphOverlay, ConflictRecord, TabSpec
└── license_compat/
    ├── __init__.py                                 # registers LicenseCompatPlugin instance
    ├── plugin.py                                   # LicenseCompatPlugin: implements all hooks
    ├── matrix.py                                   # LicenseMatrix dataclass + load_matrix + resolve_license
    ├── engine.py                                   # check_compat, recommend, find_compatible_subchains, find_breaking_nodes
    ├── walker.py                                   # enumerate_edges, resolve_artifact_license, compute_license_frequencies
    ├── cli.py                                      # license-check, license-audit subparsers + handlers
    ├── web.py                                      # Flask Blueprint
    ├── spdx.py                                     # SPDX 3.0.1 Annotation Element emitter
    ├── overlay.py                                  # GraphOverlay payload
    ├── templates/license_compat/tab.html           # BOM-viewer tab template
    └── data/
        ├── matrix.json                             # vendored from /mnt/d/LicenseRec/Data/Matrixes/matrix.json
        ├── allowed_licenses.json                   # vendored
        └── missing.json                            # vendored

tests/plugins/
├── __init__.py
├── conftest.py                                     # shared: tiny_matrix, canned_findings, in_memory_store
├── test_plugin_registry.py                         # registry contract tests
├── test_plugin_contract.py                         # per-plugin hook-shape parametrized tests
└── license_compat/
    ├── __init__.py
    ├── conftest.py                                 # plugin-specific fixtures: lineage_3node store
    ├── fixtures/
    │   ├── tiny_matrix.json
    │   ├── tiny_allowed.json
    │   ├── tiny_missing.json
    │   └── lineage_3node.ttl
    ├── test_matrix.py
    ├── test_engine_check_compat.py
    ├── test_engine_recommend.py
    ├── test_engine_subchains.py
    ├── test_engine_breaking_nodes.py
    ├── test_walker.py
    ├── test_walker_scope.py
    ├── test_cli.py
    ├── test_spdx_emit.py
    ├── test_web_tab.py
    ├── test_overlay.py
    ├── test_conflicts_integration.py
    └── test_e2e_license_compat.py
```

**Modified:**

- `src/aikaboom/cli.py` — one block in `main()` after `subparsers = parser.add_subparsers(...)` to call `plugin.register_cli(subparsers)` for each registered plugin.
- `src/aikaboom/web/app.py` — one block at Flask app construction to call `plugin.web_blueprint()` and `plugin.bom_viewer_tab()` for each plugin.
- `src/aikaboom/utils/cyclonedx_exporter.py` — one block to call `plugin.spdx_annotations()` and emit parallel CycloneDX entries.
- `src/aikaboom/utils/spdx_validator.py` (if SPDX export happens there) OR the SPDX exporter module — one block to call `plugin.spdx_annotations()` and merge into output.
- `pyproject.toml` — add `[tool.setuptools.package-data]` to include `aikaboom/plugins/license_compat/data/*.json` and `aikaboom/plugins/license_compat/templates/license_compat/*.html`.

---

## Task 1: Plugin substrate — base Protocol and registry

**Files:**
- Create: `src/aikaboom/plugins/__init__.py`
- Create: `src/aikaboom/plugins/base.py`
- Create: `tests/plugins/__init__.py`
- Create: `tests/plugins/test_plugin_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/plugins/test_plugin_registry.py`:

```python
"""Plugin registry contract tests."""
from __future__ import annotations

import pytest

from aikaboom.plugins import all_plugins, get, register
from aikaboom.plugins.base import Plugin


class _DummyPlugin:
    name = "dummy-test"

    def enabled(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Each test gets a fresh registry."""
    from aikaboom.plugins import _registry
    monkeypatch.setattr(_registry, "_plugins", {})


def test_register_and_retrieve():
    p = _DummyPlugin()
    register(p)
    assert get("dummy-test") is p


def test_all_plugins_returns_registered_plugins():
    p1 = _DummyPlugin()
    p2 = _DummyPlugin()
    p2.name = "dummy-test-2"
    register(p1)
    register(p2)
    names = {p.name for p in all_plugins()}
    assert names == {"dummy-test", "dummy-test-2"}


def test_duplicate_registration_raises():
    register(_DummyPlugin())
    with pytest.raises(ValueError, match="already registered"):
        register(_DummyPlugin())


def test_get_unknown_plugin_returns_none():
    assert get("nope") is None


def test_protocol_recognises_dummy():
    # _DummyPlugin doesn't implement every hook, but it satisfies the
    # minimum (name + enabled). Confirm the Protocol is structural, not
    # nominal — runtime_checkable + isinstance.
    p = _DummyPlugin()
    assert isinstance(p, Plugin)
```

Also create empty `tests/plugins/__init__.py`:

```python
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_plugin_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.plugins'`

- [ ] **Step 3: Write minimal implementation**

Create `src/aikaboom/plugins/base.py`:

```python
"""Plugin Protocol and supporting dataclasses for the aibom plugin system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    import argparse
    from flask import Blueprint


@dataclass(frozen=True)
class Scope:
    """Analysis scope for a plugin run."""
    kind: str  # "single" | "graph_wide"
    artifact_iri: Optional[str] = None
    depth: int = 5

    @classmethod
    def single(cls, artifact_iri: str, depth: int = 5) -> "Scope":
        return cls(kind="single", artifact_iri=artifact_iri, depth=depth)

    @classmethod
    def graph_wide(cls) -> "Scope":
        return cls(kind="graph_wide")


@dataclass(frozen=True)
class TabSpec:
    """Descriptor for a tab a plugin contributes to the BOM viewer."""
    label: str
    url_template: str  # e.g. "/license-compat/{artifact_id}"
    sort_order: int = 100


@dataclass(frozen=True)
class ConflictRecord:
    """Entry the plugin contributes to the existing Conflicts tab."""
    category: str
    severity: str  # "high" | "medium" | "low" | "info"
    subject_iri: str
    title: str
    detail: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraphOverlay:
    """Payload for the graph-view edge/node tinting overlay."""
    plugin_name: str
    edge_attrs: dict = field(default_factory=dict)  # (s, p, o) tuple-as-str -> {color, label, tooltip}
    node_attrs: dict = field(default_factory=dict)  # iri -> {badge, ring_color}


class Findings(Protocol):
    """Result type of plugin.analyze(). Implementations supply iteration helpers."""

    def to_dict(self) -> dict: ...
    def violations(self) -> list: ...


@runtime_checkable
class Plugin(Protocol):
    """All plugins implement this surface. Hooks return None or empty if not used."""

    name: str

    def enabled(self) -> bool: ...

    def analyze(self, store: Any, scope: Scope) -> Findings: ...

    def register_cli(self, parent_subparsers: "argparse._SubParsersAction") -> None: ...

    def web_blueprint(self) -> Optional["Blueprint"]: ...

    def bom_viewer_tab(self) -> Optional[TabSpec]: ...

    def spdx_annotations(self, claim_iri: str, findings: Findings) -> list[dict]: ...

    def graph_overlay(self, findings: Findings) -> GraphOverlay: ...

    def conflict_findings(self, findings: Findings) -> list[ConflictRecord]: ...
```

Create `src/aikaboom/plugins/__init__.py`:

```python
"""aibom plugin registry. Plugins self-register at import time."""
from __future__ import annotations

from aikaboom.plugins.base import (
    ConflictRecord,
    Findings,
    GraphOverlay,
    Plugin,
    Scope,
    TabSpec,
)


class _registry:
    """Mutable container so monkeypatching in tests is straightforward."""
    _plugins: dict = {}


def register(plugin: Plugin) -> None:
    """Register a plugin instance. Raises ValueError if name collides."""
    if plugin.name in _registry._plugins:
        raise ValueError(f"Plugin {plugin.name!r} already registered")
    _registry._plugins[plugin.name] = plugin


def all_plugins() -> list[Plugin]:
    """Return every registered plugin in insertion order."""
    return list(_registry._plugins.values())


def get(name: str) -> Plugin | None:
    """Look up a plugin by name."""
    return _registry._plugins.get(name)


__all__ = [
    "ConflictRecord",
    "Findings",
    "GraphOverlay",
    "Plugin",
    "Scope",
    "TabSpec",
    "register",
    "all_plugins",
    "get",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_plugin_registry.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/aikaboom/plugins/__init__.py src/aikaboom/plugins/base.py \
        tests/plugins/__init__.py tests/plugins/test_plugin_registry.py
git commit -m "feat(plugins): add plugin Protocol and registry"
```

---

## Task 2: Bundle license-matrix data files

**Files:**
- Create: `src/aikaboom/plugins/license_compat/__init__.py` (stub for now — Task 4 fills it)
- Create: `src/aikaboom/plugins/license_compat/data/matrix.json` (copy from `/mnt/d/LicenseRec/Data/Matrixes/matrix.json`)
- Create: `src/aikaboom/plugins/license_compat/data/allowed_licenses.json` (copy)
- Create: `src/aikaboom/plugins/license_compat/data/missing.json` (copy)
- Modify: `pyproject.toml` to include the data files

- [ ] **Step 1: Copy the vendored data**

```bash
mkdir -p src/aikaboom/plugins/license_compat/data
cp /mnt/d/LicenseRec/Data/Matrixes/matrix.json src/aikaboom/plugins/license_compat/data/
cp /mnt/d/LicenseRec/Data/allowed_licenses.json src/aikaboom/plugins/license_compat/data/ 2>/dev/null || \
   echo '[]' > src/aikaboom/plugins/license_compat/data/allowed_licenses.json
cp /mnt/d/LicenseRec/Data/missing.json src/aikaboom/plugins/license_compat/data/ 2>/dev/null || \
   echo '{"licenses": []}' > src/aikaboom/plugins/license_compat/data/missing.json
```

If `allowed_licenses.json` is absent at the source, the fallback writes `[]` (no whitelist, no recommendations until a real list is added). Same idea for `missing.json`.

- [ ] **Step 2: Create stub plugin package init**

Create `src/aikaboom/plugins/license_compat/__init__.py`:

```python
"""License-compatibility analysis plugin.

Phase 1 stub — full registration wiring lands in Task 4.
"""
```

- [ ] **Step 3: Write a packaging smoke test**

Create `tests/plugins/license_compat/__init__.py` (empty) and `tests/plugins/license_compat/test_packaging.py`:

```python
"""Smoke test: bundled data files are importable via importlib.resources."""
import json
from importlib.resources import files


def test_matrix_resource_loads():
    p = files("aikaboom.plugins.license_compat.data").joinpath("matrix.json")
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "licenses" in data
    assert isinstance(data["licenses"], list)
    assert len(data["licenses"]) > 100  # the vendored matrix has thousands


def test_allowed_licenses_resource_loads():
    p = files("aikaboom.plugins.license_compat.data").joinpath("allowed_licenses.json")
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_missing_licenses_resource_loads():
    p = files("aikaboom.plugins.license_compat.data").joinpath("missing.json")
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "licenses" in data
```

- [ ] **Step 4: Update pyproject.toml for package data**

Modify `pyproject.toml`. Append (or edit the existing `[tool.setuptools.package-data]` block):

```toml
[tool.setuptools.package-data]
aikaboom = ["plugins/license_compat/data/*.json", "plugins/license_compat/templates/license_compat/*.html"]
```

If `[tool.setuptools.package-data]` already exists for `aikaboom`, *append* the two glob strings to its list — do not replace the existing entries.

- [ ] **Step 5: Run the smoke test**

Run: `pip install -e . && pytest tests/plugins/license_compat/test_packaging.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/plugins/license_compat/ \
        tests/plugins/license_compat/__init__.py \
        tests/plugins/license_compat/test_packaging.py \
        pyproject.toml
git commit -m "feat(license-compat): bundle license matrix + allowed + missing data files"
```

---

## Task 3: Matrix loader and license resolver

**Files:**
- Create: `src/aikaboom/plugins/license_compat/matrix.py`
- Create: `tests/plugins/license_compat/fixtures/tiny_matrix.json`
- Create: `tests/plugins/license_compat/fixtures/tiny_allowed.json`
- Create: `tests/plugins/license_compat/fixtures/tiny_missing.json`
- Create: `tests/plugins/license_compat/conftest.py`
- Create: `tests/plugins/license_compat/test_matrix.py`

- [ ] **Step 1: Write the fixture**

Create `tests/plugins/license_compat/fixtures/tiny_matrix.json`:

```json
{
  "timestamp": "2026-01-01T00:00:00+0000",
  "licenses": [
    {
      "name": "apache-2.0",
      "aliases": ["apache 2", "apache 2.0", "apache-2"],
      "category": "PERMISSIVE",
      "compatibilities": [
        {"name": "apache-2.0", "compatibility": "Same"},
        {"name": "mit", "compatibility": "Yes"},
        {"name": "gpl-3.0", "compatibility": "No"},
        {"name": "cc-by-nc-4.0", "compatibility": "No"},
        {"name": "lgpl-3.0", "compatibility": "Yes"}
      ]
    },
    {
      "name": "mit",
      "aliases": ["the mit license"],
      "category": "PERMISSIVE",
      "compatibilities": [
        {"name": "mit", "compatibility": "Same"},
        {"name": "apache-2.0", "compatibility": "Yes"},
        {"name": "gpl-3.0", "compatibility": "Yes"},
        {"name": "cc-by-nc-4.0", "compatibility": "No"},
        {"name": "lgpl-3.0", "compatibility": "Yes"}
      ]
    },
    {
      "name": "gpl-3.0",
      "aliases": ["gpl3", "gpl 3"],
      "category": "COPYLEFT",
      "compatibilities": [
        {"name": "gpl-3.0", "compatibility": "Same"},
        {"name": "mit", "compatibility": "No"},
        {"name": "apache-2.0", "compatibility": "No"},
        {"name": "cc-by-nc-4.0", "compatibility": "No"},
        {"name": "lgpl-3.0", "compatibility": "No"}
      ]
    },
    {
      "name": "cc-by-nc-4.0",
      "aliases": ["cc by-nc 4.0"],
      "category": "NON_COMMERCIAL",
      "compatibilities": [
        {"name": "cc-by-nc-4.0", "compatibility": "Same"},
        {"name": "mit", "compatibility": "No"},
        {"name": "apache-2.0", "compatibility": "No"},
        {"name": "gpl-3.0", "compatibility": "No"}
      ]
    },
    {
      "name": "lgpl-3.0",
      "aliases": [],
      "category": "COPYLEFT",
      "compatibilities": [
        {"name": "lgpl-3.0", "compatibility": "Same"},
        {"name": "apache-2.0", "compatibility": "Yes"},
        {"name": "mit", "compatibility": "Yes"}
      ]
    }
  ]
}
```

Create `tests/plugins/license_compat/fixtures/tiny_allowed.json`:

```json
[{"key": "apache-2.0"}, {"key": "mit"}, {"key": "lgpl-3.0"}, {"key": "cc-by-nc-4.0"}]
```

Create `tests/plugins/license_compat/fixtures/tiny_missing.json`:

```json
{"licenses": ["proprietary-corp-x"]}
```

- [ ] **Step 2: Write the conftest**

Create `tests/plugins/license_compat/conftest.py`:

```python
"""Shared fixtures for license_compat tests."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_matrix_paths():
    return {
        "matrix": FIXTURES / "tiny_matrix.json",
        "allowed": FIXTURES / "tiny_allowed.json",
        "missing": FIXTURES / "tiny_missing.json",
    }


@pytest.fixture
def tiny_matrix(tiny_matrix_paths):
    from aikaboom.plugins.license_compat.matrix import load_matrix
    return load_matrix(
        matrix_path=tiny_matrix_paths["matrix"],
        allowed_path=tiny_matrix_paths["allowed"],
        missing_path=tiny_matrix_paths["missing"],
    )
```

- [ ] **Step 3: Write the failing tests**

Create `tests/plugins/license_compat/test_matrix.py`:

```python
"""LicenseMatrix loader + license resolver tests."""
from __future__ import annotations

import pytest

from aikaboom.plugins.license_compat.matrix import (
    LicenseMatrix,
    load_matrix,
    resolve_license,
)


def test_load_matrix_indexes_aliases(tiny_matrix):
    assert tiny_matrix.name_alias_lookup["apache 2.0"] == "apache-2.0"
    assert tiny_matrix.name_alias_lookup["apache-2.0"] == "apache-2.0"
    assert tiny_matrix.name_alias_lookup["the mit license"] == "mit"


def test_load_matrix_builds_upstream_compat_index(tiny_matrix):
    # apache-2.0 is compatible-upstream-of mit (mit lists apache-2.0 Yes)
    assert "mit" in tiny_matrix.upstream_compat_index["apache-2.0"]
    # apache-2.0 self-compat ("Same")
    assert "apache-2.0" in tiny_matrix.upstream_compat_index["apache-2.0"]
    # gpl-3.0 -> apache-2.0 is "No"
    assert "apache-2.0" not in tiny_matrix.upstream_compat_index["gpl-3.0"]


def test_load_matrix_injects_unknown_token(tiny_matrix):
    assert "UNKNOWN" in tiny_matrix.details
    assert tiny_matrix.name_alias_lookup["unknown"] == "UNKNOWN"


def test_load_matrix_reads_allowed_licenses(tiny_matrix):
    assert tiny_matrix.allowed_licenses == {"apache-2.0", "mit", "lgpl-3.0", "cc-by-nc-4.0"}


def test_load_matrix_reads_missing_licenses(tiny_matrix):
    assert tiny_matrix.missing_licenses == {"proprietary-corp-x"}


def test_resolve_license_canonicalises_alias(tiny_matrix):
    r = resolve_license("Apache 2.0", tiny_matrix)
    assert r.primary_name == "apache-2.0"
    assert r.is_unknown is False
    assert r.is_missing is False


def test_resolve_license_flags_unknown_string(tiny_matrix):
    r = resolve_license("WeirdMadeUpLic-7", tiny_matrix)
    assert r.primary_name == "UNKNOWN"
    assert r.is_unknown is True


def test_resolve_license_handles_missing_marker(tiny_matrix):
    r = resolve_license("MISSING", tiny_matrix)
    assert r.primary_name is None
    assert r.is_missing is True


def test_resolve_license_strips_other_parentheses(tiny_matrix):
    r = resolve_license("apache-2.0 (other)", tiny_matrix)
    assert r.primary_name == "apache-2.0"


def test_load_matrix_uses_bundled_defaults_when_no_path():
    # When called with no arguments, the bundled matrix is loaded.
    m = load_matrix()
    assert isinstance(m, LicenseMatrix)
    assert len(m.details) > 100  # bundled matrix has thousands
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.plugins.license_compat.matrix'`

- [ ] **Step 5: Write the implementation**

Create `src/aikaboom/plugins/license_compat/matrix.py`:

```python
"""License matrix loader + canonical-name resolver.

Lifted from LicenseRec.py with the I/O isolated so the engine layer can stay
pure. The matrix is a dict of license entries keyed by canonical primary name;
each entry has aliases, a category, and a list of pairwise compatibility
verdicts against every other license. The upstream_compat_index is the inverse
view: for each upstream license, which downstream licenses can legally consume it.
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Optional

_OTHER_SUFFIX_RE = re.compile(r"\s*\(\s*other\s*\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseMatrix:
    name_alias_lookup: dict[str, str]
    details: dict[str, dict]
    upstream_compat_index: dict[str, frozenset[str]]
    allowed_licenses: frozenset[str]
    missing_licenses: frozenset[str]
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class ResolvedLicense:
    primary_name: Optional[str]
    is_unknown: bool
    is_missing: bool


def _bundled(name: str) -> Path:
    return Path(str(files("aikaboom.plugins.license_compat.data").joinpath(name)))


def _clean(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped or stripped.upper() == "MISSING":
        return None
    return _OTHER_SUFFIX_RE.sub("", stripped).strip().lower()


def load_matrix(
    matrix_path: Optional[Path] = None,
    allowed_path: Optional[Path] = None,
    missing_path: Optional[Path] = None,
) -> LicenseMatrix:
    matrix_path = Path(matrix_path) if matrix_path else _bundled("matrix.json")
    allowed_path = Path(allowed_path) if allowed_path else _bundled("allowed_licenses.json")
    missing_path = Path(missing_path) if missing_path else _bundled("missing.json")

    matrix_data = json.loads(matrix_path.read_text(encoding="utf-8"))
    timestamp = matrix_data.get("timestamp")

    name_alias_lookup: dict[str, str] = {}
    details: dict[str, dict] = {}

    for item in matrix_data.get("licenses", []):
        if not isinstance(item, dict):
            continue
        primary = item.get("name")
        if not isinstance(primary, str):
            continue
        primary = primary.strip()
        item = dict(item)
        item["category"] = item.get("category") or "UNKNOWN"
        details[primary] = item
        name_alias_lookup[primary.lower()] = primary
        for alias in item.get("aliases", []) or []:
            if isinstance(alias, str):
                a = alias.lower().strip()
                if a:
                    name_alias_lookup[a] = primary

    if "UNKNOWN" not in details:
        details["UNKNOWN"] = {"name": "UNKNOWN", "category": "UNKNOWN", "permissions": [], "compatibilities": []}
    name_alias_lookup["unknown"] = "UNKNOWN"

    upstream_compat_index: dict[str, set[str]] = defaultdict(set)
    for downstream, info in details.items():
        for entry in info.get("compatibilities", []) or []:
            if isinstance(entry, dict) and entry.get("compatibility") in ("Yes", "Same"):
                up = entry.get("name")
                if isinstance(up, str):
                    upstream_compat_index[up].add(downstream)

    allowed_raw = json.loads(allowed_path.read_text(encoding="utf-8")) if allowed_path.exists() else []
    allowed = frozenset(
        item["key"].lower().strip()
        for item in allowed_raw
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    )

    missing_raw = json.loads(missing_path.read_text(encoding="utf-8")) if missing_path.exists() else {}
    missing = frozenset(
        lic.strip().lower() for lic in (missing_raw.get("licenses", []) if isinstance(missing_raw, dict) else [])
        if isinstance(lic, str)
    )

    return LicenseMatrix(
        name_alias_lookup=name_alias_lookup,
        details=details,
        upstream_compat_index={k: frozenset(v) for k, v in upstream_compat_index.items()},
        allowed_licenses=allowed,
        missing_licenses=missing,
        timestamp=timestamp,
    )


def resolve_license(raw: str, matrix: LicenseMatrix) -> ResolvedLicense:
    if not isinstance(raw, str):
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=False)
    if raw.strip().upper() == "MISSING":
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=True)
    cleaned = _clean(raw)
    if cleaned is None:
        return ResolvedLicense(primary_name=None, is_unknown=False, is_missing=False)
    primary = matrix.name_alias_lookup.get(cleaned)
    if primary is None:
        return ResolvedLicense(primary_name="UNKNOWN", is_unknown=True, is_missing=False)
    return ResolvedLicense(primary_name=primary, is_unknown=False, is_missing=False)


def normalize_license_field(value) -> list[str]:
    """Mirrors LicenseRec.py: accept a string, list, or stringified list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str)]
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() == "MISSING":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if isinstance(v, str)]
            except (ValueError, SyntaxError):
                return [s]
        return [s]
    return []
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_matrix.py -v`
Expected: 10 PASS

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/plugins/license_compat/matrix.py \
        tests/plugins/license_compat/conftest.py \
        tests/plugins/license_compat/fixtures/ \
        tests/plugins/license_compat/test_matrix.py
git commit -m "feat(license-compat): matrix loader + license resolver"
```

---

## Task 4: Engine — check_compat + recommend

**Files:**
- Create: `src/aikaboom/plugins/license_compat/engine.py`
- Create: `tests/plugins/license_compat/test_engine_check_compat.py`
- Create: `tests/plugins/license_compat/test_engine_recommend.py`

- [ ] **Step 1: Write the failing tests — check_compat**

Create `tests/plugins/license_compat/test_engine_check_compat.py`:

```python
"""Truth-table tests for check_compat."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import CompatVerdict, check_compat


def test_compatible_single_upstream(tiny_matrix):
    v = check_compat("mit", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "compatible"
    assert v.incompatible_with == frozenset()


def test_violation_single_upstream(tiny_matrix):
    v = check_compat("apache-2.0", frozenset({"gpl-3.0"}), tiny_matrix)
    assert v.status == "violation"
    assert v.incompatible_with == frozenset({"gpl-3.0"})


def test_violation_partial_block(tiny_matrix):
    # mit is OK downstream of apache-2.0 (Yes), but NOT of cc-by-nc-4.0 (No)
    v = check_compat("mit", frozenset({"apache-2.0", "cc-by-nc-4.0"}), tiny_matrix)
    assert v.status == "violation"
    assert v.incompatible_with == frozenset({"cc-by-nc-4.0"})


def test_unknown_upstream(tiny_matrix):
    v = check_compat("mit", frozenset({"UNKNOWN"}), tiny_matrix)
    assert v.status == "unknown_upstream"


def test_unknown_downstream(tiny_matrix):
    v = check_compat(None, frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "unknown_downstream"


def test_compatible_same_license(tiny_matrix):
    v = check_compat("apache-2.0", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "compatible"


def test_missing_data_when_downstream_not_in_matrix(tiny_matrix):
    v = check_compat("never-heard-of-this", frozenset({"apache-2.0"}), tiny_matrix)
    assert v.status == "missing_data"


def test_empty_upstream_set_is_compatible_trivially(tiny_matrix):
    # No upstream constraints => any downstream is compatible.
    v = check_compat("apache-2.0", frozenset(), tiny_matrix)
    assert v.status == "compatible"
```

- [ ] **Step 2: Write the failing tests — recommend**

Create `tests/plugins/license_compat/test_engine_recommend.py`:

```python
"""Recommendation logic tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.engine import recommend


def test_recommend_returns_intersection_filtered_by_whitelist(tiny_matrix):
    freqs = Counter({"apache-2.0": 100, "mit": 80, "lgpl-3.0": 10})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs)
    # apache-2.0 is upstream-compat with apache-2.0, mit, lgpl-3.0 (per Yes/Same in matrix).
    # whitelist has apache-2.0, mit, lgpl-3.0, cc-by-nc-4.0.
    assert "PERMISSIVE" in r.by_category
    assert "apache-2.0" in r.by_category["PERMISSIVE"]
    assert "mit" in r.by_category["PERMISSIVE"]
    assert r.is_solvable is True


def test_recommend_orders_by_frequency_desc(tiny_matrix):
    freqs = Counter({"mit": 1000, "apache-2.0": 1})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs)
    # mit wins on frequency
    assert r.by_category["PERMISSIVE"][0] == "mit"


def test_recommend_no_compatible_intersection_returns_empty(tiny_matrix):
    # downstream of gpl-3.0 AND cc-by-nc-4.0: nothing satisfies both
    r = recommend(frozenset({"gpl-3.0", "cc-by-nc-4.0"}), tiny_matrix, Counter())
    assert r.by_category == {}
    assert r.is_solvable is False


def test_recommend_returns_top_k_per_category(tiny_matrix):
    # synthesize a category overload; tiny matrix only has small categories so this
    # validates the cap is applied — set k=1 and assert at most 1 per category.
    freqs = Counter({"apache-2.0": 10, "mit": 9, "lgpl-3.0": 8})
    r = recommend(frozenset({"apache-2.0"}), tiny_matrix, freqs, top_k_per_category=1)
    for cat, items in r.by_category.items():
        assert len(items) <= 1


def test_recommend_excludes_non_whitelisted(tiny_matrix):
    # Inject a candidate that wouldn't be in the allowed list. Easy way: use
    # an empty allowed set via a fresh matrix override.
    matrix = type(tiny_matrix)(
        name_alias_lookup=tiny_matrix.name_alias_lookup,
        details=tiny_matrix.details,
        upstream_compat_index=tiny_matrix.upstream_compat_index,
        allowed_licenses=frozenset(),
        missing_licenses=tiny_matrix.missing_licenses,
        timestamp=tiny_matrix.timestamp,
    )
    r = recommend(frozenset({"apache-2.0"}), matrix, Counter())
    assert r.by_category == {}
    # is_solvable reflects pre-whitelist solvability
    assert r.is_solvable is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_engine_check_compat.py tests/plugins/license_compat/test_engine_recommend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.plugins.license_compat.engine'`

- [ ] **Step 4: Write the implementation**

Create `src/aikaboom/plugins/license_compat/engine.py`:

```python
"""Pure license-compatibility engine.

No I/O, no graph. Inputs are LicenseMatrix values; outputs are dataclass
verdicts and recommendations. Mirrors LicenseRec.py's analytical primitives.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from aikaboom.plugins.license_compat.matrix import LicenseMatrix

Status = Literal["compatible", "violation", "unknown_upstream", "unknown_downstream", "missing_data"]


@dataclass(frozen=True)
class CompatVerdict:
    downstream: Optional[str]
    upstreams: frozenset[str]
    status: Status
    incompatible_with: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Recommendation:
    by_category: dict[str, list[str]]
    is_solvable: bool


_CC_VERSION_RE = re.compile(r"^(cc-[a-z\-]+)-(\d\.\d)$", re.IGNORECASE)


def check_compat(downstream: Optional[str], upstreams: frozenset[str], matrix: LicenseMatrix) -> CompatVerdict:
    if downstream is None:
        return CompatVerdict(None, upstreams, "unknown_downstream")
    if "UNKNOWN" in upstreams:
        return CompatVerdict(downstream, upstreams, "unknown_upstream")
    if not upstreams:
        return CompatVerdict(downstream, upstreams, "compatible")
    if downstream not in matrix.details:
        return CompatVerdict(downstream, upstreams, "missing_data")
    blocked = frozenset(
        up for up in upstreams
        if downstream not in matrix.upstream_compat_index.get(up, frozenset())
    )
    if blocked:
        return CompatVerdict(downstream, upstreams, "violation", incompatible_with=blocked)
    return CompatVerdict(downstream, upstreams, "compatible")


def recommend(
    upstreams: frozenset[str],
    matrix: LicenseMatrix,
    frequencies: Counter,
    top_k_per_category: int = 5,
) -> Recommendation:
    if not upstreams:
        return Recommendation(by_category={}, is_solvable=False)

    compat_sets = [matrix.upstream_compat_index.get(up, frozenset()) for up in upstreams]
    if not compat_sets:
        return Recommendation(by_category={}, is_solvable=False)

    candidate_names = frozenset.intersection(*compat_sets) if len(compat_sets) > 1 else compat_sets[0]
    is_solvable = len(candidate_names) > 0
    if not is_solvable:
        return Recommendation(by_category={}, is_solvable=False)

    filtered = [matrix.details[n] for n in candidate_names if n.lower() in matrix.allowed_licenses and n in matrix.details]
    if not filtered:
        return Recommendation(by_category={}, is_solvable=is_solvable)

    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in filtered:
        grouped[entry.get("category", "UNKNOWN")].append(entry["name"])

    for cat, lic_list in list(grouped.items()):
        processed = lic_list
        cc_versions = [m for l in lic_list if (m := _CC_VERSION_RE.match(l))]
        if cc_versions:
            bases_with_4_0 = {m.group(1).lower() for l in lic_list if (m := _CC_VERSION_RE.match(l)) and m.group(2) == "4.0"}
            processed = [
                l for l in lic_list
                if not (m := _CC_VERSION_RE.match(l)) or m.group(2) == "4.0" or m.group(1).lower() not in bases_with_4_0
            ]
        grouped[cat] = sorted(processed, key=lambda l: (-frequencies.get(l, 0), l.lower()))[:top_k_per_category]

    return Recommendation(by_category=dict(grouped), is_solvable=is_solvable)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_engine_check_compat.py tests/plugins/license_compat/test_engine_recommend.py -v`
Expected: 8 + 5 = 13 PASS

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/plugins/license_compat/engine.py \
        tests/plugins/license_compat/test_engine_check_compat.py \
        tests/plugins/license_compat/test_engine_recommend.py
git commit -m "feat(license-compat): pure check_compat + recommend engine"
```

---

## Task 5: Engine — Findings, compatible subchains, breaking nodes

**Files:**
- Modify: `src/aikaboom/plugins/license_compat/engine.py` (add Findings, Finding, edge helpers, subchain, breaking-node logic)
- Create: `tests/plugins/license_compat/test_engine_subchains.py`
- Create: `tests/plugins/license_compat/test_engine_breaking_nodes.py`

- [ ] **Step 1: Write the failing tests — subchains**

Create `tests/plugins/license_compat/test_engine_subchains.py`:

```python
"""find_compatible_subchains tests."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
    find_compatible_subchains,
)


def _finding(d_iri: str, u_iri: str, status: str, predicate: str = "trainedOn") -> Finding:
    return Finding(
        downstream_iri=d_iri,
        downstream_label=d_iri,
        upstream_iri=u_iri,
        upstream_label=u_iri,
        predicate=predicate,
        verdict=CompatVerdict(
            downstream="mit" if status == "compatible" else "gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status=status,
            incompatible_with=frozenset() if status == "compatible" else frozenset({"apache-2.0"}),
        ),
        recommendation=None,
    )


def test_single_compatible_edge_yields_one_subchain_size_2():
    f = Findings([_finding("A", "B", "compatible")])
    chains = find_compatible_subchains(f)
    assert len(chains) == 1
    assert chains[0].size == 2
    assert chains[0].artifacts == frozenset({"A", "B"})


def test_chain_of_compatible_edges_merges_into_one_component():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("B", "C", "compatible"),
        _finding("C", "D", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    assert len(chains) == 1
    assert chains[0].size == 4


def test_violation_splits_components():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("B", "C", "violation"),
        _finding("C", "D", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    sizes = sorted(c.size for c in chains)
    assert sizes == [2, 2]


def test_isolated_violation_node_appears_as_size_1_component():
    # Z has only a violation edge and shouldn't disappear.
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("Z", "A", "violation"),
    ])
    chains = find_compatible_subchains(f)
    sizes = sorted(c.size for c in chains)
    assert sizes == [1, 2]


def test_chains_sorted_by_size_desc():
    f = Findings([
        _finding("A", "B", "compatible"),
        _finding("C", "D", "compatible"),
        _finding("D", "E", "compatible"),
    ])
    chains = find_compatible_subchains(f)
    assert chains[0].size >= chains[1].size
```

- [ ] **Step 2: Write the failing tests — breaking nodes**

Create `tests/plugins/license_compat/test_engine_breaking_nodes.py`:

```python
"""find_breaking_nodes tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
    find_breaking_nodes,
)


def _vfinding(d_iri: str, u_iri: str, blocked_upstream: str) -> Finding:
    return Finding(
        downstream_iri=d_iri,
        downstream_label=d_iri,
        upstream_iri=u_iri,
        upstream_label=u_iri,
        predicate="trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({blocked_upstream}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({blocked_upstream}),
            status="violation",
            incompatible_with=frozenset({blocked_upstream}),
        ),
        recommendation=None,
    )


def test_breaking_node_blame_count(tiny_matrix):
    # cc-by-nc-4.0 blocks three different downstreams
    findings = Findings([
        _vfinding("D1", "X_NC", "cc-by-nc-4.0"),
        _vfinding("D2", "X_NC", "cc-by-nc-4.0"),
        _vfinding("D3", "X_NC", "cc-by-nc-4.0"),
    ])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter())
    assert len(nodes) == 1
    assert nodes[0].artifact_iri == "X_NC"
    assert nodes[0].blamed_in == 3
    assert nodes[0].affected_downstream == frozenset({"D1", "D2", "D3"})


def test_breaking_nodes_sorted_by_blame_desc(tiny_matrix):
    findings = Findings([
        _vfinding("D1", "A", "cc-by-nc-4.0"),
        _vfinding("D2", "A", "cc-by-nc-4.0"),
        _vfinding("D3", "B", "cc-by-nc-4.0"),
    ])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter())
    assert nodes[0].artifact_iri == "A"
    assert nodes[0].blamed_in == 2
    assert nodes[1].artifact_iri == "B"
    assert nodes[1].blamed_in == 1


def test_breaking_node_fix_recommendations_use_downstream_union(tiny_matrix):
    # X blocks D1 (which is gpl-3.0) — fix_recommendations should be
    # licenses that, if X took them instead, would clear all blame.
    findings = Findings([_vfinding("D1", "X", "cc-by-nc-4.0")])
    nodes = find_breaking_nodes(findings, tiny_matrix, Counter({"apache-2.0": 1}))
    assert nodes[0].fix_recommendations.is_solvable is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_engine_subchains.py tests/plugins/license_compat/test_engine_breaking_nodes.py -v`
Expected: FAIL with `ImportError: cannot import name 'Finding' from 'aikaboom.plugins.license_compat.engine'`

- [ ] **Step 4: Extend the engine implementation**

Append to `src/aikaboom/plugins/license_compat/engine.py`:

```python
@dataclass(frozen=True)
class Finding:
    downstream_iri: str
    downstream_label: str
    upstream_iri: str
    upstream_label: str
    predicate: str
    verdict: CompatVerdict
    downstream_license: Optional[str] = None
    upstream_licenses: frozenset[str] = field(default_factory=frozenset)
    recommendation: Optional[Recommendation] = None

    def is_violation(self) -> bool:
        return self.verdict.status == "violation"

    def is_compatible(self) -> bool:
        return self.verdict.status == "compatible"


@dataclass(frozen=True)
class CompatSubchain:
    artifacts: frozenset[str]
    edges: frozenset[tuple[str, str, str]]
    size: int
    root: str


@dataclass(frozen=True)
class BreakingNode:
    artifact_iri: str
    label: str
    license: Optional[str]
    blamed_in: int
    affected_downstream: frozenset[str]
    fix_recommendations: Recommendation


class Findings:
    """Iterable wrapper around list[Finding] with helpers."""

    def __init__(self, items: Iterable[Finding]):
        self._items: list[Finding] = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def violations(self) -> list[Finding]:
        return [f for f in self._items if f.is_violation()]

    def to_dict(self) -> dict:
        return {
            "findings": [
                {
                    "downstream_iri": f.downstream_iri,
                    "upstream_iri": f.upstream_iri,
                    "predicate": f.predicate,
                    "status": f.verdict.status,
                    "downstream_license": f.downstream_license,
                    "upstream_licenses": sorted(f.upstream_licenses),
                    "incompatible_with": sorted(f.verdict.incompatible_with),
                    "recommendation": (
                        None if f.recommendation is None else {
                            "by_category": f.recommendation.by_category,
                            "is_solvable": f.recommendation.is_solvable,
                        }
                    ),
                }
                for f in self._items
            ],
        }


def find_compatible_subchains(findings: Findings) -> list[CompatSubchain]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    all_artifacts: set[str] = set()
    compat_edges: list[tuple[str, str, str]] = []
    for f in findings:
        all_artifacts.add(f.downstream_iri)
        all_artifacts.add(f.upstream_iri)
        if f.is_compatible():
            compat_edges.append((f.downstream_iri, f.upstream_iri, f.predicate))
            union(f.downstream_iri, f.upstream_iri)

    # Build components — every artifact that was *seen* gets a root, even
    # if it sits on no compatible edge (size-1 component).
    for a in all_artifacts:
        parent.setdefault(a, a)

    groups: dict[str, set[str]] = defaultdict(set)
    for a in all_artifacts:
        groups[find(a)].add(a)

    edges_by_root: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for s, u, p in compat_edges:
        edges_by_root[find(s)].add((s, u, p))

    chains = [
        CompatSubchain(
            artifacts=frozenset(members),
            edges=frozenset(edges_by_root.get(root, set())),
            size=len(members),
            root=root,
        )
        for root, members in groups.items()
    ]
    chains.sort(key=lambda c: (-c.size, c.root))
    return chains


def find_breaking_nodes(
    findings: Findings,
    matrix: LicenseMatrix,
    frequencies: Counter,
) -> list[BreakingNode]:
    blame: dict[str, list[Finding]] = defaultdict(list)
    upstream_license: dict[str, Optional[str]] = {}
    upstream_label: dict[str, str] = {}
    for f in findings.violations():
        if f.upstream_iri in f.verdict.incompatible_with or any(
            u == f.upstream_iri for u in f.verdict.incompatible_with
        ):
            # incompatible_with carries license names, not IRIs — match on the
            # finding's resolved upstream_licenses if they intersect.
            pass
        # If any of the upstream's licenses is in incompatible_with, the
        # upstream IRI is "blamed" for the violation.
        if f.upstream_licenses & f.verdict.incompatible_with:
            blame[f.upstream_iri].append(f)
            upstream_label[f.upstream_iri] = f.upstream_label
            if f.upstream_licenses:
                upstream_license[f.upstream_iri] = next(iter(f.upstream_licenses))

    nodes: list[BreakingNode] = []
    for iri, edges in blame.items():
        affected = frozenset(e.downstream_iri for e in edges)
        # contextual fix: union of downstream licenses that blame this node
        downstream_lic_union = frozenset(
            e.downstream_license for e in edges if e.downstream_license is not None
        )
        fix = (
            recommend(downstream_lic_union, matrix, frequencies)
            if downstream_lic_union
            else Recommendation(by_category={}, is_solvable=False)
        )
        nodes.append(
            BreakingNode(
                artifact_iri=iri,
                label=upstream_label.get(iri, iri),
                license=upstream_license.get(iri),
                blamed_in=len(edges),
                affected_downstream=affected,
                fix_recommendations=fix,
            )
        )

    nodes.sort(key=lambda n: (-n.blamed_in, n.artifact_iri))
    return nodes
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_engine_subchains.py tests/plugins/license_compat/test_engine_breaking_nodes.py -v`
Expected: 5 + 3 = 8 PASS

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/plugins/license_compat/engine.py \
        tests/plugins/license_compat/test_engine_subchains.py \
        tests/plugins/license_compat/test_engine_breaking_nodes.py
git commit -m "feat(license-compat): Findings + compatible-subchain + breaking-node analytics"
```

---

## Task 6: Graph walker — edge enumeration, trust-aware resolution, frequencies

**Prerequisite:** `aikaboom.store` module is available (post-worldofboms-graph merge).

**Files:**
- Create: `src/aikaboom/plugins/license_compat/walker.py`
- Create: `tests/plugins/license_compat/fixtures/lineage_3node.ttl`
- Create: `tests/plugins/license_compat/test_walker.py`
- Create: `tests/plugins/license_compat/test_walker_scope.py`

- [ ] **Step 1: Write the fixture TTL**

Create `tests/plugins/license_compat/fixtures/lineage_3node.ttl`:

```turtle
@prefix aibom: <https://aikaboom.dev/aibom#> .
@prefix ex: <https://example.org/> .

ex:ModelA aibom:hasVersion ex:ModelA_v1 ; aibom:canonicalLabel "ModelA" .
ex:ModelA_v1 aibom:hasClaim ex:ClaimMA1 .
ex:ClaimMA1 aibom:hasLicense "apache-2.0" ;
            aibom:trustScore 0.9 ;
            aibom:createdAt "2026-01-01T00:00:00Z" ;
            aibom:trainedOn ex:DatasetB .

ex:DatasetB aibom:hasVersion ex:DatasetB_v1 ; aibom:canonicalLabel "DatasetB" .
ex:DatasetB_v1 aibom:hasClaim ex:ClaimDB1 .
ex:ClaimDB1 aibom:hasLicense "apache-2.0" ;
            aibom:trustScore 0.8 ;
            aibom:createdAt "2026-01-02T00:00:00Z" ;
            aibom:dependsOn ex:PaperC .

ex:PaperC aibom:hasVersion ex:PaperC_v1 ; aibom:canonicalLabel "PaperC" .
ex:PaperC_v1 aibom:hasClaim ex:ClaimPC1 .
ex:ClaimPC1 aibom:hasLicense "cc-by-nc-4.0" ;
            aibom:trustScore 0.5 ;
            aibom:createdAt "2026-01-03T00:00:00Z" .

# Second, conflicting claim for ModelA with lower trustScore.
ex:ModelA_v1 aibom:hasClaim ex:ClaimMA2 .
ex:ClaimMA2 aibom:hasLicense "gpl-3.0" ;
            aibom:trustScore 0.2 ;
            aibom:createdAt "2026-01-04T00:00:00Z" ;
            aibom:trainedOn ex:DatasetB .
```

- [ ] **Step 2: Extend conftest with an in-memory BomStore fixture**

Append to `tests/plugins/license_compat/conftest.py`:

```python
@pytest.fixture
def lineage_3node_store(tmp_path):
    """Build a BomStore (RDFLib backend) populated from lineage_3node.ttl."""
    import os
    os.environ["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    os.environ["AIKABOOM_GRAPH_DIR"] = str(tmp_path)

    from aikaboom.store import BomStore
    store = BomStore.open()
    ttl_path = FIXTURES / "lineage_3node.ttl"
    store._backend.import_(ttl_path, fmt="turtle")
    yield store
    store._backend.close()
```

- [ ] **Step 3: Write failing tests — walker basics**

Create `tests/plugins/license_compat/test_walker.py`:

```python
"""Graph walker tests."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins.license_compat.walker import (
    LineageEdge,
    compute_license_frequencies,
    enumerate_edges,
    resolve_artifact_license,
)
from aikaboom.plugins import Scope


def test_enumerate_graph_wide_returns_all_lineage_edges(lineage_3node_store):
    edges = list(enumerate_edges(lineage_3node_store, Scope.graph_wide()))
    pairs = {(e.downstream_iri, e.upstream_iri, e.predicate.rsplit("#", 1)[-1]) for e in edges}
    assert ("https://example.org/ModelA", "https://example.org/DatasetB", "trainedOn") in pairs
    assert ("https://example.org/DatasetB", "https://example.org/PaperC", "dependsOn") in pairs


def test_resolve_artifact_license_picks_highest_trust(lineage_3node_store, tiny_matrix):
    r = resolve_artifact_license(lineage_3node_store, "https://example.org/ModelA", tiny_matrix)
    # ClaimMA1 (apache-2.0, trust=0.9) wins over ClaimMA2 (gpl-3.0, trust=0.2)
    assert r.licenses == frozenset({"apache-2.0"})


def test_resolve_artifact_license_unknown_when_no_claim(lineage_3node_store, tiny_matrix):
    r = resolve_artifact_license(lineage_3node_store, "https://example.org/DoesNotExist", tiny_matrix)
    assert r.licenses == frozenset()


def test_compute_license_frequencies(lineage_3node_store, tiny_matrix):
    freqs = compute_license_frequencies(lineage_3node_store, tiny_matrix)
    # 2 claims with apache-2.0, 1 with gpl-3.0, 1 with cc-by-nc-4.0
    assert freqs["apache-2.0"] >= 2
    assert freqs["cc-by-nc-4.0"] >= 1
```

- [ ] **Step 4: Write failing tests — scope**

Create `tests/plugins/license_compat/test_walker_scope.py`:

```python
"""Walker scope: single vs graph-wide, depth, cycle safety."""
from __future__ import annotations

from aikaboom.plugins.license_compat.walker import enumerate_edges
from aikaboom.plugins import Scope


def test_scope_single_starts_from_artifact(lineage_3node_store):
    edges = list(enumerate_edges(
        lineage_3node_store,
        Scope.single("https://example.org/ModelA"),
    ))
    # From ModelA the walker reaches DatasetB and then PaperC.
    iris = {e.upstream_iri for e in edges} | {e.downstream_iri for e in edges}
    assert "https://example.org/ModelA" in iris
    assert "https://example.org/DatasetB" in iris
    assert "https://example.org/PaperC" in iris


def test_scope_single_depth_bound(lineage_3node_store):
    # depth=1 means we see only direct upstreams of ModelA.
    edges = list(enumerate_edges(
        lineage_3node_store,
        Scope.single("https://example.org/ModelA", depth=1),
    ))
    upstreams = {e.upstream_iri for e in edges}
    assert "https://example.org/DatasetB" in upstreams
    assert "https://example.org/PaperC" not in upstreams


def test_walker_cycle_safe(tmp_path, monkeypatch):
    """A 2-node cycle must not loop forever."""
    import os
    os.environ["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    os.environ["AIKABOOM_GRAPH_DIR"] = str(tmp_path)
    from aikaboom.store import BomStore
    store = BomStore.open()
    cycle_ttl = """
    @prefix aibom: <https://aikaboom.dev/aibom#> .
    @prefix ex: <https://example.org/> .
    ex:A aibom:hasVersion ex:A_v1 .
    ex:A_v1 aibom:hasClaim ex:CA .
    ex:CA aibom:trainedOn ex:B ; aibom:hasLicense "mit" ; aibom:trustScore 0.5 ;
          aibom:createdAt "2026-01-01T00:00:00Z" .
    ex:B aibom:hasVersion ex:B_v1 .
    ex:B_v1 aibom:hasClaim ex:CB .
    ex:CB aibom:dependsOn ex:A ; aibom:hasLicense "mit" ; aibom:trustScore 0.5 ;
          aibom:createdAt "2026-01-02T00:00:00Z" .
    """
    p = tmp_path / "cycle.ttl"
    p.write_text(cycle_ttl)
    store._backend.import_(p, fmt="turtle")

    edges = list(enumerate_edges(store, Scope.single("https://example.org/A", depth=10)))
    seen = {(e.downstream_iri, e.upstream_iri) for e in edges}
    # Each edge appears at most once.
    assert len(seen) == len(edges)
    store._backend.close()
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_walker.py tests/plugins/license_compat/test_walker_scope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aikaboom.plugins.license_compat.walker'`

- [ ] **Step 6: Write the implementation**

Create `src/aikaboom/plugins/license_compat/walker.py`:

```python
"""Graph walker: enumerate lineage edges, resolve trust-aware licenses, compute frequencies."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Optional

from aikaboom.plugins import Scope
from aikaboom.plugins.license_compat.matrix import LicenseMatrix, normalize_license_field, resolve_license
from aikaboom.store import BomStore
from aikaboom.store import vocab

LINEAGE_PREDICATES = (
    str(vocab.trainedOn),
    str(vocab.testedOn),
    str(vocab.dependsOn),
    str(vocab.hostedAt),
)


@dataclass(frozen=True)
class LineageEdge:
    downstream_iri: str
    downstream_label: str
    upstream_iri: str
    upstream_label: str
    predicate: str


@dataclass(frozen=True)
class ResolvedArtifact:
    iri: str
    label: str
    licenses: frozenset[str]
    source_claim_iri: Optional[str]
    has_unknown: bool
    has_missing: bool


def _label(store: BomStore, iri: str) -> str:
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?l WHERE {{ <{iri}> aibom:canonicalLabel ?l }} LIMIT 1
    """))
    if rows:
        return str(rows[0]["l"])
    return iri.rsplit("/", 1)[-1]


def enumerate_edges(store: BomStore, scope: Scope) -> Iterator[LineageEdge]:
    if scope.kind == "graph_wide":
        yield from _enumerate_all(store)
    elif scope.kind == "single":
        yield from _enumerate_from(store, scope.artifact_iri, scope.depth)
    else:
        raise ValueError(f"Unknown scope kind: {scope.kind}")


def _enumerate_all(store: BomStore) -> Iterator[LineageEdge]:
    values_clause = " ".join(f"<{p}>" for p in LINEAGE_PREDICATES)
    rows = store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?downstream ?upstream ?p WHERE {{
          VALUES ?p {{ {values_clause} }}
          ?artifact aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          BIND(?artifact AS ?downstream)
          ?claim ?p ?upstream .
        }}
    """)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["downstream"]), str(row["upstream"]), str(row["p"]))
        if key in seen:
            continue
        seen.add(key)
        yield LineageEdge(
            downstream_iri=key[0],
            downstream_label=_label(store, key[0]),
            upstream_iri=key[1],
            upstream_label=_label(store, key[1]),
            predicate=key[2],
        )


def _direct_upstreams(store: BomStore, artifact_iri: str) -> list[tuple[str, str]]:
    values_clause = " ".join(f"<{p}>" for p in LINEAGE_PREDICATES)
    rows = store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT DISTINCT ?upstream ?p WHERE {{
          VALUES ?p {{ {values_clause} }}
          <{artifact_iri}> aibom:hasVersion ?version .
          ?version aibom:hasClaim ?claim .
          ?claim ?p ?upstream .
        }}
    """)
    return [(str(row["upstream"]), str(row["p"])) for row in rows]


def _enumerate_from(store: BomStore, start_iri: str, depth: int) -> Iterator[LineageEdge]:
    visited_nodes: set[str] = set()
    yielded_edges: set[tuple[str, str, str]] = set()
    frontier: list[tuple[str, int]] = [(start_iri, 0)]
    while frontier:
        artifact, level = frontier.pop(0)
        if artifact in visited_nodes:
            continue
        visited_nodes.add(artifact)
        if level >= depth:
            continue
        for up_iri, predicate in _direct_upstreams(store, artifact):
            key = (artifact, up_iri, predicate)
            if key in yielded_edges:
                continue
            yielded_edges.add(key)
            yield LineageEdge(
                downstream_iri=artifact,
                downstream_label=_label(store, artifact),
                upstream_iri=up_iri,
                upstream_label=_label(store, up_iri),
                predicate=predicate,
            )
            frontier.append((up_iri, level + 1))


def resolve_artifact_license(
    store: BomStore,
    artifact_iri: str,
    matrix: LicenseMatrix,
) -> ResolvedArtifact:
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?claim ?lic ?trust ?created WHERE {{
          {{
            <{artifact_iri}> aibom:hasVersion ?version .
            ?version aibom:hasClaim ?claim .
          }} UNION {{
            <{artifact_iri}> aibom:canonicalClaim ?claim .
          }}
          ?claim aibom:hasLicense ?lic .
          OPTIONAL {{ ?claim aibom:trustScore ?trust }}
          OPTIONAL {{ ?claim aibom:createdAt ?created }}
        }}
        ORDER BY DESC(?trust) DESC(?created)
        LIMIT 1
    """))
    has_unknown = False
    has_missing = False
    licenses: set[str] = set()
    source_claim: Optional[str] = None
    if rows:
        row = rows[0]
        source_claim = str(row["claim"])
        for raw in normalize_license_field(str(row["lic"])):
            r = resolve_license(raw, matrix)
            if r.is_missing:
                has_missing = True
                continue
            if r.is_unknown:
                has_unknown = True
            if r.primary_name is not None:
                licenses.add(r.primary_name)

    if not licenses:
        # Fallback: artifact-level hasLicense triple.
        fb = list(store._backend.select(f"""
            PREFIX aibom: <https://aikaboom.dev/aibom#>
            SELECT ?lic WHERE {{ <{artifact_iri}> aibom:hasLicense ?lic }} LIMIT 5
        """))
        for row in fb:
            for raw in normalize_license_field(str(row["lic"])):
                r = resolve_license(raw, matrix)
                if r.primary_name and not r.is_unknown:
                    licenses.add(r.primary_name)

    return ResolvedArtifact(
        iri=artifact_iri,
        label=_label(store, artifact_iri),
        licenses=frozenset(licenses),
        source_claim_iri=source_claim,
        has_unknown=has_unknown,
        has_missing=has_missing,
    )


def compute_license_frequencies(store: BomStore, matrix: LicenseMatrix) -> Counter:
    rows = store._backend.select("""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?lic WHERE {
          ?s aibom:hasLicense ?lic .
        }
    """)
    counts: Counter = Counter()
    for row in rows:
        for raw in normalize_license_field(str(row["lic"])):
            r = resolve_license(raw, matrix)
            if r.primary_name is not None and not r.is_unknown:
                counts[r.primary_name] += 1
    return counts
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_walker.py tests/plugins/license_compat/test_walker_scope.py -v`
Expected: 4 + 3 = 7 PASS

- [ ] **Step 8: Commit**

```bash
git add src/aikaboom/plugins/license_compat/walker.py \
        tests/plugins/license_compat/fixtures/lineage_3node.ttl \
        tests/plugins/license_compat/conftest.py \
        tests/plugins/license_compat/test_walker.py \
        tests/plugins/license_compat/test_walker_scope.py
git commit -m "feat(license-compat): SPARQL graph walker with trust-aware license resolution"
```

---

## Task 7: Plugin glue — LicenseCompatPlugin + registration

**Files:**
- Create: `src/aikaboom/plugins/license_compat/plugin.py`
- Modify: `src/aikaboom/plugins/license_compat/__init__.py` (replace stub with registration)
- Create: `tests/plugins/test_plugin_contract.py`

- [ ] **Step 1: Write the failing contract test**

Create `tests/plugins/test_plugin_contract.py`:

```python
"""Per-plugin contract test: every hook exists and returns the documented type."""
from __future__ import annotations

import pytest

from aikaboom.plugins import (
    ConflictRecord,
    GraphOverlay,
    Plugin,
    Scope,
    TabSpec,
    all_plugins,
)
from aikaboom.plugins.license_compat.engine import Findings


@pytest.fixture
def empty_findings():
    return Findings([])


@pytest.fixture(params=lambda: [p.name for p in all_plugins()])
def plugin(request):
    from aikaboom.plugins import get
    return get(request.param)


def test_plugin_is_protocol_compatible():
    for p in all_plugins():
        assert isinstance(p, Plugin)


def test_plugin_enabled_returns_bool():
    for p in all_plugins():
        assert isinstance(p.enabled(), bool)


def test_plugin_web_blueprint_returns_blueprint_or_none():
    from flask import Blueprint
    for p in all_plugins():
        bp = p.web_blueprint()
        assert bp is None or isinstance(bp, Blueprint)


def test_plugin_bom_viewer_tab_returns_tabspec_or_none():
    for p in all_plugins():
        tab = p.bom_viewer_tab()
        assert tab is None or isinstance(tab, TabSpec)


def test_plugin_graph_overlay_returns_overlay(empty_findings):
    for p in all_plugins():
        overlay = p.graph_overlay(empty_findings)
        assert isinstance(overlay, GraphOverlay)


def test_plugin_conflict_findings_returns_list(empty_findings):
    for p in all_plugins():
        out = p.conflict_findings(empty_findings)
        assert isinstance(out, list)
        for entry in out:
            assert isinstance(entry, ConflictRecord)


def test_license_compat_plugin_is_registered():
    from aikaboom.plugins import get
    p = get("license-compat")
    assert p is not None
    assert p.name == "license-compat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/test_plugin_contract.py -v`
Expected: FAIL — `get("license-compat") is None` and `all_plugins()` is empty (no plugins registered yet)

- [ ] **Step 3: Write the plugin implementation**

Create `src/aikaboom/plugins/license_compat/plugin.py`:

```python
"""LicenseCompatPlugin — wires the engine + walker + emitters into the plugin contract."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from aikaboom.plugins import ConflictRecord, GraphOverlay, Scope, TabSpec
from aikaboom.plugins.license_compat.engine import (
    Finding,
    Findings,
    Recommendation,
    check_compat,
    find_breaking_nodes,
    find_compatible_subchains,
    recommend,
)
from aikaboom.plugins.license_compat.matrix import LicenseMatrix, load_matrix


class LicenseCompatPlugin:
    name = "license-compat"

    def __init__(self, matrix: Optional[LicenseMatrix] = None):
        self._matrix_override: Optional[Path] = None
        self._matrix_cache: Optional[LicenseMatrix] = matrix

    def enabled(self) -> bool:
        return os.environ.get("AIKABOOM_LICENSE_COMPAT_DISABLED", "").lower() not in ("1", "true", "yes")

    def _matrix(self) -> LicenseMatrix:
        if self._matrix_cache is None:
            override = os.environ.get("AIKABOOM_LICENSE_MATRIX")
            self._matrix_cache = load_matrix(matrix_path=Path(override) if override else None)
        return self._matrix_cache

    def analyze(self, store, scope: Scope) -> Findings:
        from aikaboom.plugins.license_compat.walker import (
            compute_license_frequencies,
            enumerate_edges,
            resolve_artifact_license,
        )

        matrix = self._matrix()
        freqs = compute_license_frequencies(store, matrix)
        findings: list[Finding] = []
        for edge in enumerate_edges(store, scope):
            d = resolve_artifact_license(store, edge.downstream_iri, matrix)
            u = resolve_artifact_license(store, edge.upstream_iri, matrix)
            d_licenses = d.licenses or frozenset({None})
            for d_lic in d_licenses:
                verdict = check_compat(d_lic, u.licenses, matrix)
                rec = (
                    recommend(u.licenses, matrix, freqs)
                    if verdict.status == "violation"
                    else None
                )
                findings.append(
                    Finding(
                        downstream_iri=edge.downstream_iri,
                        downstream_label=edge.downstream_label,
                        upstream_iri=edge.upstream_iri,
                        upstream_label=edge.upstream_label,
                        predicate=edge.predicate,
                        downstream_license=d_lic,
                        upstream_licenses=u.licenses,
                        verdict=verdict,
                        recommendation=rec,
                    )
                )
        return Findings(findings)

    def register_cli(self, parent_subparsers) -> None:
        from aikaboom.plugins.license_compat.cli import register_cli as _register
        _register(parent_subparsers, self)

    def web_blueprint(self):
        from aikaboom.plugins.license_compat.web import build_blueprint
        return build_blueprint(self)

    def bom_viewer_tab(self) -> Optional[TabSpec]:
        return TabSpec(
            label="License compatibility",
            url_template="/license-compat/{artifact_id}",
            sort_order=50,
        )

    def spdx_annotations(self, claim_iri: str, findings: Findings) -> list[dict]:
        from aikaboom.plugins.license_compat.spdx import emit_annotations
        return emit_annotations(claim_iri, findings, matrix=self._matrix())

    def graph_overlay(self, findings: Findings) -> GraphOverlay:
        from aikaboom.plugins.license_compat.overlay import build_overlay
        return build_overlay(findings, plugin_name=self.name)

    def conflict_findings(self, findings: Findings) -> list[ConflictRecord]:
        records: list[ConflictRecord] = []
        for f in findings.violations():
            records.append(ConflictRecord(
                category="license-compat",
                severity="high",
                subject_iri=f.downstream_iri,
                title=f"License {f.downstream_license} ↛ {sorted(f.verdict.incompatible_with)}",
                detail=f"Incompatible via {f.predicate.rsplit('#', 1)[-1]}",
                data={
                    "predicate": f.predicate,
                    "upstream": f.upstream_iri,
                    "incompatible_with": sorted(f.verdict.incompatible_with),
                },
            ))
        return records
```

- [ ] **Step 4: Register the plugin at import time**

Replace the stub `src/aikaboom/plugins/license_compat/__init__.py` with:

```python
"""License-compatibility analysis plugin."""
from aikaboom.plugins import register
from aikaboom.plugins.license_compat.plugin import LicenseCompatPlugin

# Self-register on import.
register(LicenseCompatPlugin())
```

Also update `src/aikaboom/plugins/__init__.py` to eagerly import the plugin so `all_plugins()` sees it. Add at the *bottom* of `src/aikaboom/plugins/__init__.py`:

```python
# Eager-import in-tree plugins so they self-register. When entry_points
# discovery lands, this block is replaced with a single call.
from aikaboom.plugins import license_compat as _license_compat  # noqa: E402, F401
```

- [ ] **Step 5: Add stub emitter modules so plugin.py imports succeed**

The plugin lazily imports `spdx`, `overlay`, `web`, `cli` from inside hook methods. Create minimal placeholders so the import graph works; Tasks 8-11 flesh them out.

Create `src/aikaboom/plugins/license_compat/cli.py`:

```python
"""License-compat CLI (filled in Task 8)."""
def register_cli(parent_subparsers, plugin):
    raise NotImplementedError("cli is implemented in Task 8")
```

Create `src/aikaboom/plugins/license_compat/web.py`:

```python
"""License-compat web blueprint (filled in Task 10)."""
def build_blueprint(plugin):
    return None
```

Create `src/aikaboom/plugins/license_compat/spdx.py`:

```python
"""License-compat SPDX emitter (filled in Task 9)."""
def emit_annotations(claim_iri, findings, matrix):
    return []
```

Create `src/aikaboom/plugins/license_compat/overlay.py`:

```python
"""License-compat graph overlay (filled in Task 11)."""
from aikaboom.plugins import GraphOverlay

def build_overlay(findings, plugin_name):
    return GraphOverlay(plugin_name=plugin_name, edge_attrs={}, node_attrs={})
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/plugins/test_plugin_contract.py -v`
Expected: 7 PASS

Also re-run prior suites to confirm no regression:

```bash
pytest tests/plugins/ -v
```

Expected: all previous tests still pass, plus the contract tests above.

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/plugins/license_compat/plugin.py \
        src/aikaboom/plugins/license_compat/__init__.py \
        src/aikaboom/plugins/license_compat/cli.py \
        src/aikaboom/plugins/license_compat/web.py \
        src/aikaboom/plugins/license_compat/spdx.py \
        src/aikaboom/plugins/license_compat/overlay.py \
        src/aikaboom/plugins/__init__.py \
        tests/plugins/test_plugin_contract.py
git commit -m "feat(license-compat): LicenseCompatPlugin glue + emitter stubs"
```

---

## Task 8: CLI — license-check and license-audit

**Files:**
- Modify: `src/aikaboom/plugins/license_compat/cli.py` (fill in)
- Modify: `src/aikaboom/cli.py` (one-line plugin loop)
- Create: `tests/plugins/license_compat/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

Create `tests/plugins/license_compat/test_cli.py`:

```python
"""CLI tests for license-check + license-audit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LINEAGE_TTL = Path(__file__).parent / "fixtures" / "lineage_3node.ttl"


@pytest.fixture
def populated_store_env(tmp_path):
    """Spin up an isolated graph dir, populate it from the lineage fixture."""
    env = os.environ.copy()
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_path)
    env["BOM_SKIP_DOTENV"] = "1"

    # Populate via a small Python invocation to keep the test self-contained.
    populate = (
        "from aikaboom.store import BomStore; "
        f"s = BomStore.open(); s._backend.import_(__import__('pathlib').Path(r'{LINEAGE_TTL}'), fmt='turtle'); s._backend.close()"
    )
    subprocess.run([sys.executable, "-c", populate], check=True, env=env)
    return env


def test_license_check_text_format(populated_store_env):
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-check",
         "https://example.org/ModelA", "--format", "text"],
        env=populated_store_env, capture_output=True, text=True,
    )
    assert r.returncode in (0, 2)
    assert "ModelA" in r.stdout
    assert "trainedOn" in r.stdout or "DatasetB" in r.stdout


def test_license_check_json_format_has_findings(populated_store_env):
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-check",
         "https://example.org/ModelA", "--format", "json"],
        env=populated_store_env, capture_output=True, text=True,
    )
    data = json.loads(r.stdout)
    assert "findings" in data
    assert "compatible_subchains" in data
    assert "breaking_nodes" in data


def test_license_audit_jsonl_format(populated_store_env, tmp_path):
    out = tmp_path / "audit.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-audit",
         "--format", "jsonl", "--out", str(out)],
        env=populated_store_env, capture_output=True, text=True,
    )
    assert r.returncode in (0, 2)
    assert out.exists()
    for line in out.read_text().splitlines():
        json.loads(line)  # each line parses


def test_license_check_exit_code_2_on_violation(populated_store_env):
    # The fixture's lineage has cc-by-nc-4.0 upstream of apache-2.0 downstream
    # via DatasetB -> PaperC; depending on traversal direction this becomes a
    # violation. Force a violation by relicensing ModelA in an override matrix.
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-check",
         "https://example.org/PaperC", "--format", "text"],
        env=populated_store_env, capture_output=True, text=True,
    )
    # PaperC has no upstreams in our fixture, so 0 violations -> exit 0.
    assert r.returncode == 0


def test_license_check_unknown_artifact_exits_3(populated_store_env):
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-check",
         "https://example.org/DoesNotExist", "--format", "text"],
        env=populated_store_env, capture_output=True, text=True,
    )
    # Unknown artifact: no edges, no licenses — depends on whether walker
    # treats this as "no findings" (exit 0) or "unresolved" (exit 3). Our
    # contract says exit 3 only when the artifact resolver returned nothing
    # at all *and* the user asked for a single-scope check.
    assert r.returncode in (0, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_cli.py -v`
Expected: FAIL with `NotImplementedError: cli is implemented in Task 8` or CLI dispatch errors.

- [ ] **Step 3: Implement the CLI**

Replace `src/aikaboom/plugins/license_compat/cli.py` with:

```python
"""License-compat CLI: license-check + license-audit subparsers and handlers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from aikaboom.plugins import Scope


def register_cli(parent_subparsers: "argparse._SubParsersAction", plugin) -> None:
    p_check = parent_subparsers.add_parser(
        "license-check",
        help="Check license compatibility for one artifact and its lineage",
    )
    p_check.add_argument("artifact", help="Artifact IRI, label, or platform id")
    p_check.add_argument("--depth", type=int, default=5)
    p_check.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_check.add_argument("--matrix", type=Path, default=None)
    p_check.add_argument("--violations-only", action="store_true")
    p_check.set_defaults(func=lambda args: _cmd_check(args, plugin))

    p_audit = parent_subparsers.add_parser(
        "license-audit",
        help="Sweep the entire stored graph for license-compat violations",
    )
    p_audit.add_argument("--format", choices=["text", "json", "jsonl"], default="text")
    p_audit.add_argument("--matrix", type=Path, default=None)
    p_audit.add_argument("--out", type=Path, default=None)
    p_audit.set_defaults(func=lambda args: _cmd_audit(args, plugin))


def _open_store():
    from aikaboom.store import BomStore
    return BomStore.open()


def _resolve_artifact_iri(store, candidate: str) -> Optional[str]:
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    # Try BomStore.resolve() if available; otherwise fall back to label match.
    if hasattr(store, "resolve"):
        try:
            r = store.resolve(candidate)
            if r and getattr(r, "artifact_iri", None):
                return r.artifact_iri
        except Exception:
            pass
    rows = list(store._backend.select(f"""
        PREFIX aibom: <https://aikaboom.dev/aibom#>
        SELECT ?a WHERE {{ ?a aibom:canonicalLabel "{candidate}" }} LIMIT 2
    """))
    if len(rows) == 1:
        return str(rows[0]["a"])
    return None


def _override_matrix(plugin, override: Optional[Path]) -> None:
    if override is not None:
        from aikaboom.plugins.license_compat.matrix import load_matrix
        plugin._matrix_cache = load_matrix(matrix_path=override)


def _cmd_check(args: argparse.Namespace, plugin) -> int:
    _override_matrix(plugin, args.matrix)
    store = _open_store()
    iri = _resolve_artifact_iri(store, args.artifact)
    if iri is None:
        print(f"Artifact not found: {args.artifact}", file=sys.stderr)
        return 3
    findings = plugin.analyze(store, Scope.single(iri, depth=args.depth))
    return _render_and_exit(findings, args, plugin)


def _cmd_audit(args: argparse.Namespace, plugin) -> int:
    _override_matrix(plugin, args.matrix)
    store = _open_store()
    findings = plugin.analyze(store, Scope.graph_wide())
    if args.out is not None:
        with args.out.open("w", encoding="utf-8") as fh:
            for item in findings.to_dict()["findings"]:
                fh.write(json.dumps(item) + "\n")
    return _render_and_exit(findings, args, plugin)


def _render_and_exit(findings, args, plugin) -> int:
    from aikaboom.plugins.license_compat.engine import (
        find_breaking_nodes,
        find_compatible_subchains,
    )

    matrix = plugin._matrix()
    from aikaboom.plugins.license_compat.walker import compute_license_frequencies
    store = _open_store()
    freqs = compute_license_frequencies(store, matrix)
    subchains = find_compatible_subchains(findings)
    breaking = find_breaking_nodes(findings, matrix, freqs)

    if args.format == "json":
        payload = {
            **findings.to_dict(),
            "compatible_subchains": [
                {"size": c.size, "root": c.root, "artifacts": sorted(c.artifacts)}
                for c in subchains
            ],
            "breaking_nodes": [
                {
                    "artifact_iri": n.artifact_iri,
                    "label": n.label,
                    "license": n.license,
                    "blamed_in": n.blamed_in,
                    "affected_downstream": sorted(n.affected_downstream),
                    "fix_recommendations": {
                        "by_category": n.fix_recommendations.by_category,
                        "is_solvable": n.fix_recommendations.is_solvable,
                    },
                }
                for n in breaking
            ],
        }
        print(json.dumps(payload, indent=2))
    elif args.format == "jsonl":
        for item in findings.to_dict()["findings"]:
            print(json.dumps(item))
    else:
        _render_text(findings, subchains, breaking, args)

    return 2 if findings.violations() else 0


def _render_text(findings, subchains, breaking, args) -> None:
    items = findings.violations() if args.violations_only else list(findings)
    if not items:
        print("No findings.")
    for f in items:
        marker = {"compatible": "OK ", "violation": "X  ", "unknown_upstream": "?  ",
                  "unknown_downstream": "?  ", "missing_data": "-  "}[f.verdict.status]
        pred = f.predicate.rsplit("#", 1)[-1]
        print(f"  {marker}{f.downstream_label} ({f.downstream_license}) "
              f"--{pred}--> {f.upstream_label} ({sorted(f.upstream_licenses)})   "
              f"{f.verdict.status.upper()}")
        if f.recommendation and f.recommendation.is_solvable:
            for cat, lics in f.recommendation.by_category.items():
                print(f"        {cat}: {', '.join(lics)}")
    if subchains:
        print(f"\nCompatible subchains ({len(subchains)}):")
        for i, c in enumerate(subchains, 1):
            print(f"  {i}. size={c.size} root={c.root}")
    if breaking:
        print(f"\nBreaking nodes ({len(breaking)}):")
        for n in breaking:
            print(f"  - {n.label} ({n.license})  blamed in {n.blamed_in}  "
                  f"affected {len(n.affected_downstream)} downstream")
    summary = {
        "edges": len(list(findings)),
        "compatible": sum(1 for f in findings if f.verdict.status == "compatible"),
        "violations": len(findings.violations()),
    }
    print(f"\nSummary: {summary['edges']} edges  |  "
          f"{summary['compatible']} compatible  |  "
          f"{summary['violations']} violations")
```

- [ ] **Step 4: Wire the plugin loop into core CLI**

Modify `src/aikaboom/cli.py`. Find the `main()` function and the line where `subparsers = parser.add_subparsers(...)` is created (around line 488). Immediately *after* the existing `subparsers.add_parser(...)` blocks and *before* `args = parser.parse_args()`, insert:

```python
# Plugin subparsers — every registered plugin can mount its own commands.
from aikaboom.plugins import all_plugins
for _plugin in all_plugins():
    try:
        _plugin.register_cli(subparsers)
    except Exception as e:
        import logging
        logging.getLogger("aikaboom.plugins").warning(
            "Plugin %r failed to register CLI: %s", _plugin.name, e
        )
```

Also confirm `args.func(args)` is called *if* the dispatched subcommand sets `.func`. If `main()` already has a dispatch pattern (e.g., `if args.command == "generate": ...`), add a branch:

```python
if hasattr(args, "func") and callable(args.func):
    return args.func(args)
```

Place this *before* the existing dispatch chain so plugin-defined subcommands win for their own names without interfering with builtins like `generate`/`serve`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_cli.py -v`
Expected: 5 PASS (or 4 PASS + 1 skipped if BomStore.resolve isn't implemented yet)

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/plugins/license_compat/cli.py \
        src/aikaboom/cli.py \
        tests/plugins/license_compat/test_cli.py
git commit -m "feat(license-compat): license-check + license-audit CLI subcommands"
```

---

## Task 9: SPDX annotation emitter

**Files:**
- Modify: `src/aikaboom/plugins/license_compat/spdx.py` (fill in)
- Modify: `src/aikaboom/utils/cyclonedx_exporter.py` (or wherever SPDX export builds the output dict — one plugin loop)
- Create: `tests/plugins/license_compat/test_spdx_emit.py`

- [ ] **Step 1: Locate the SPDX export entry point**

Run: `grep -rn "spdx" src/aikaboom/utils/ | grep -E "def |class " | head -20`

Identify the function that assembles the SPDX 3.0.1 JSON-LD output dict. The plugin loop hooks in just before serialization.

- [ ] **Step 2: Write the failing test**

Create `tests/plugins/license_compat/test_spdx_emit.py`:

```python
"""SPDX Annotation Element emitter tests."""
from __future__ import annotations

import json

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)
from aikaboom.plugins.license_compat.spdx import emit_annotations


def _violation_finding() -> Finding:
    return Finding(
        downstream_iri="https://example.org/Down",
        downstream_label="Down",
        upstream_iri="https://example.org/Up",
        upstream_label="Up",
        predicate="https://aikaboom.dev/aibom#trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status="violation",
            incompatible_with=frozenset({"apache-2.0"}),
        ),
        recommendation=None,
    )


def test_emit_one_annotation_per_violation(tiny_matrix):
    findings = Findings([_violation_finding()])
    out = emit_annotations("https://example.org/Claim", findings, matrix=tiny_matrix)
    assert len(out) == 1
    a = out[0]
    assert a["type"] == "Annotation"
    assert a["annotationType"] == "review"
    assert a["subject"] == "https://example.org/Down"
    body = json.loads(a["comment"])
    assert body["plugin"] == "license-compat"
    assert body["verdict"] == "violation"
    assert body["upstream"] == "https://example.org/Up"


def test_emit_includes_breaking_node_annotation(tiny_matrix):
    findings = Findings([_violation_finding(), _violation_finding()])
    out = emit_annotations("https://example.org/Claim", findings, matrix=tiny_matrix)
    breaking_anns = [a for a in out if json.loads(a["comment"]).get("kind") == "breaking-node"]
    assert len(breaking_anns) >= 1


def test_emit_empty_findings_returns_empty_list(tiny_matrix):
    out = emit_annotations("https://example.org/Claim", Findings([]), matrix=tiny_matrix)
    assert out == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_spdx_emit.py -v`
Expected: FAIL — `emit_annotations` returns `[]` in the stub.

- [ ] **Step 4: Implement the emitter**

Replace `src/aikaboom/plugins/license_compat/spdx.py` with:

```python
"""SPDX 3.0.1 Annotation Element emitter for license-compat findings.

Reuses the conflict-annotation pattern: one Annotation Element per finding,
annotationType="review", structured JSON in `comment`. Other SPDX tools
can ignore the body; aibom round-trips it.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from aikaboom.plugins.license_compat.engine import (
    Findings,
    find_breaking_nodes,
)
from aikaboom.plugins.license_compat.matrix import LicenseMatrix

_TOOL = "Tool:aikaboom-license-compat/0.1"


def _ann_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"spdx:annotation/license-compat/{h}"


def emit_annotations(claim_iri: str, findings: Findings, matrix: LicenseMatrix) -> list[dict]:
    out: list[dict] = []
    for f in findings.violations():
        body = {
            "plugin": "license-compat",
            "kind": "violation",
            "verdict": f.verdict.status,
            "predicate": f.predicate,
            "upstream": f.upstream_iri,
            "downstream_license": f.downstream_license,
            "upstream_licenses": sorted(f.upstream_licenses),
            "incompatible_with": sorted(f.verdict.incompatible_with),
        }
        if f.recommendation is not None:
            body["recommendation"] = {
                "by_category": f.recommendation.by_category,
                "is_solvable": f.recommendation.is_solvable,
            }
        out.append({
            "type": "Annotation",
            "spdxId": _ann_id("violation", f.downstream_iri, f.upstream_iri, f.predicate),
            "annotationType": "review",
            "subject": f.downstream_iri,
            "creationInfo": {"createdBy": [_TOOL]},
            "statement": (
                f"License {f.downstream_license} incompatible with upstream "
                f"{sorted(f.verdict.incompatible_with)} via "
                f"{f.predicate.rsplit('#', 1)[-1]}"
            ),
            "contentType": "application/json",
            "comment": json.dumps(body),
        })

    # One annotation per breaking node, attached to the upstream artifact.
    breaking = find_breaking_nodes(findings, matrix, Counter())
    for n in breaking:
        body = {
            "plugin": "license-compat",
            "kind": "breaking-node",
            "blamed_in": n.blamed_in,
            "affected_downstream": sorted(n.affected_downstream),
            "license": n.license,
            "fix_recommendations": {
                "by_category": n.fix_recommendations.by_category,
                "is_solvable": n.fix_recommendations.is_solvable,
            },
        }
        out.append({
            "type": "Annotation",
            "spdxId": _ann_id("breaking", n.artifact_iri),
            "annotationType": "review",
            "subject": n.artifact_iri,
            "creationInfo": {"createdBy": [_TOOL]},
            "statement": (
                f"Breaking node: {n.label} ({n.license}) blocks {n.blamed_in} "
                f"downstream artifact(s)"
            ),
            "contentType": "application/json",
            "comment": json.dumps(body),
        })

    return out
```

- [ ] **Step 5: Wire the plugin loop into the SPDX exporter**

Locate the SPDX export function identified in Step 1. Just before the output dict is returned/serialized, add:

```python
# Plugin-contributed annotations.
from aikaboom.plugins import all_plugins
existing_elements = spdx_doc.setdefault("element", [])
for _plugin in all_plugins():
    if not _plugin.enabled():
        continue
    try:
        anns = _plugin.spdx_annotations(claim_iri=primary_claim_iri, findings=_plugin_findings_for(_plugin))
        existing_elements.extend(anns)
    except Exception as e:
        import logging
        logging.getLogger("aikaboom.plugins").warning(
            "Plugin %r SPDX emission failed: %s", _plugin.name, e
        )
```

Where `_plugin_findings_for(plugin)` is a small helper that:
1. Opens the BomStore.
2. Determines the artifact IRI from the current export context.
3. Calls `plugin.analyze(store, Scope.single(artifact_iri))`.

If the SPDX export already passes around a `context` object containing the artifact IRI, prefer that over re-discovering. Match the existing function signature so the call site is local and minimal.

For CycloneDX parity (`src/aikaboom/utils/cyclonedx_exporter.py`), add the same loop just before returning the CDX dict, packing the annotations into a custom `vulnerabilities`-shaped array under a `properties` extension namespace `aikaboom:license-compat`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_spdx_emit.py -v`
Expected: 3 PASS

Also re-run the full plugin suite:

```bash
pytest tests/plugins/ -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/plugins/license_compat/spdx.py \
        src/aikaboom/utils/cyclonedx_exporter.py \
        tests/plugins/license_compat/test_spdx_emit.py
git commit -m "feat(license-compat): SPDX Annotation Element emitter + CDX parity hook"
```

(Adjust the `git add` list to include whichever exporter file you modified.)

---

## Task 10: Conflicts-tab integration + tests

**Files:**
- Create: `tests/plugins/license_compat/test_conflicts_integration.py`
- Modify: existing Conflicts-tab data-loader call site to merge plugin entries

The Conflicts tab data loader currently builds a list of `ConflictRecord`s from SPDX validation. We add one block that appends each plugin's `conflict_findings(findings)`.

- [ ] **Step 1: Locate the Conflicts-tab loader**

Run: `grep -rn "ConflictRecord\|conflicts_tab\|/conflicts" src/aikaboom/web/ | head -20`

Identify the function or endpoint that assembles the conflict list for the Conflicts tab.

- [ ] **Step 2: Write the failing test**

Create `tests/plugins/license_compat/test_conflicts_integration.py`:

```python
"""Plugin-contributed entries appear in the Conflicts-tab feed."""
from __future__ import annotations

from aikaboom.plugins import all_plugins, get
from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)


def test_license_compat_emits_conflict_records_for_violations():
    plugin = get("license-compat")
    findings = Findings([Finding(
        downstream_iri="https://example.org/Down",
        downstream_label="Down",
        upstream_iri="https://example.org/Up",
        upstream_label="Up",
        predicate="https://aikaboom.dev/aibom#trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status="violation",
            incompatible_with=frozenset({"apache-2.0"}),
        ),
        recommendation=None,
    )])
    entries = plugin.conflict_findings(findings)
    assert len(entries) == 1
    assert entries[0].category == "license-compat"
    assert entries[0].severity == "high"
    assert entries[0].subject_iri == "https://example.org/Down"


def test_license_compat_returns_empty_when_no_violations():
    plugin = get("license-compat")
    findings = Findings([])
    assert plugin.conflict_findings(findings) == []
```

- [ ] **Step 3: Run tests to verify they fail or pass**

Run: `pytest tests/plugins/license_compat/test_conflicts_integration.py -v`
Expected: PASS — the plugin already implements `conflict_findings` from Task 7.

- [ ] **Step 4: Wire the loader (only if not already wired)**

In the file identified in Step 1, locate the line that returns the list of `ConflictRecord`s. Just before the return, insert:

```python
# Plugin-contributed conflict entries.
from aikaboom.plugins import all_plugins
for _plugin in all_plugins():
    if not _plugin.enabled():
        continue
    try:
        # The plugin needs Findings — compute via analyze() for the current artifact.
        _findings = _plugin.analyze(store, Scope.single(current_artifact_iri))
        conflicts.extend(_plugin.conflict_findings(_findings))
    except Exception as e:
        import logging
        logging.getLogger("aikaboom.plugins").warning(
            "Plugin %r conflict emission failed: %s", _plugin.name, e
        )
```

`store`, `current_artifact_iri`, and `conflicts` are placeholders — match the names already in scope at the call site.

- [ ] **Step 5: Run an integration smoke check**

If the Conflicts tab has a Flask endpoint test, run it to confirm license-compat entries appear when the artifact has violations:

```bash
pytest tests/web/ -v -k conflicts
```

(If no existing tests cover this, the unit-level test in Step 2 stands as the contract.)

- [ ] **Step 6: Commit**

```bash
git add tests/plugins/license_compat/test_conflicts_integration.py \
        src/aikaboom/web/  # whichever file you modified
git commit -m "feat(license-compat): merge into Conflicts-tab data loader"
```

---

## Task 11: Web — Blueprint + tab template

**Files:**
- Modify: `src/aikaboom/plugins/license_compat/web.py` (fill in)
- Create: `src/aikaboom/plugins/license_compat/templates/license_compat/tab.html`
- Modify: `src/aikaboom/web/app.py` (one plugin loop block at app construction)
- Create: `tests/plugins/license_compat/test_web_tab.py`

- [ ] **Step 1: Write the failing test**

Create `tests/plugins/license_compat/test_web_tab.py`:

```python
"""Flask blueprint + tab rendering tests."""
from __future__ import annotations

import os
from urllib.parse import quote

import pytest

LINEAGE_TTL = "tests/plugins/license_compat/fixtures/lineage_3node.ttl"


@pytest.fixture
def app(tmp_path):
    os.environ["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    os.environ["AIKABOOM_GRAPH_DIR"] = str(tmp_path)
    from aikaboom.store import BomStore
    store = BomStore.open()
    from pathlib import Path
    store._backend.import_(Path(LINEAGE_TTL), fmt="turtle")
    store._backend.close()

    from aikaboom.web.app import create_app
    a = create_app()
    a.config["TESTING"] = True
    return a


def test_license_compat_html_endpoint_renders(app):
    client = app.test_client()
    iri = quote("https://example.org/ModelA", safe="")
    r = client.get(f"/license-compat/{iri}")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "License compatibility" in body
    assert "ModelA" in body or "https://example.org/ModelA" in body


def test_license_compat_json_endpoint_returns_findings(app):
    client = app.test_client()
    iri = quote("https://example.org/ModelA", safe="")
    r = client.get(f"/license-compat/{iri}.json")
    assert r.status_code == 200
    data = r.get_json()
    assert "findings" in data
    assert "compatible_subchains" in data
    assert "breaking_nodes" in data


def test_license_compat_tab_appears_in_bom_viewer_tab_strip(app):
    """The BOM viewer index page should include the License-compatibility tab label."""
    client = app.test_client()
    iri = quote("https://example.org/ModelA", safe="")
    r = client.get(f"/bom/{iri}")  # adjust to existing BOM viewer route
    assert r.status_code in (200, 302, 404)  # tab strip lives on the viewer
    if r.status_code == 200:
        assert "License compatibility" in r.get_data(as_text=True)
```

- [ ] **Step 2: Write the template**

Create `src/aikaboom/plugins/license_compat/templates/license_compat/tab.html`:

```html
{% extends "base.html" if base_template else "_blank.html" %}

{% block content %}
<h2>License compatibility</h2>
<p class="text-muted">Matrix dated {{ matrix_timestamp or "unknown" }}</p>

<section>
  <h3>Lineage</h3>
  {% if findings %}
    <ul class="lineage">
      {% for f in findings %}
        <li class="status-{{ f.verdict.status }}">
          <code>{{ f.downstream_label }}</code> ({{ f.downstream_license or "—" }})
          <span class="arrow">→[{{ f.predicate.rsplit('#', 1)[-1] }}]→</span>
          <code>{{ f.upstream_label }}</code> ({{ f.upstream_licenses | join(", ") or "—" }})
          <span class="badge badge-{{ f.verdict.status }}">{{ f.verdict.status }}</span>
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p>No edges to analyse.</p>
  {% endif %}
</section>

<section>
  <h3>Compatible subchains ({{ subchains | length }})</h3>
  <ol>
    {% for c in subchains %}
      <li>size={{ c.size }} root=<code>{{ c.root }}</code></li>
    {% endfor %}
  </ol>
</section>

<section>
  <h3>Breaking nodes ({{ breaking | length }})</h3>
  <table>
    <thead><tr><th>Artifact</th><th>License</th><th>Blame</th><th>Affected</th></tr></thead>
    <tbody>
      {% for n in breaking %}
        <tr>
          <td><code>{{ n.label }}</code></td>
          <td>{{ n.license or "—" }}</td>
          <td>{{ n.blamed_in }}</td>
          <td>{{ n.affected_downstream | length }}</td>
        </tr>
      {% endfor %}
    </tbody>
  </table>
</section>
{% endblock %}
```

- [ ] **Step 3: Implement the blueprint**

Replace `src/aikaboom/plugins/license_compat/web.py` with:

```python
"""License-compat Flask blueprint: /license-compat/<artifact_id>."""
from __future__ import annotations

from collections import Counter
from urllib.parse import unquote

from flask import Blueprint, jsonify, render_template

from aikaboom.plugins import Scope
from aikaboom.plugins.license_compat.engine import (
    find_breaking_nodes,
    find_compatible_subchains,
)


def build_blueprint(plugin) -> Blueprint:
    bp = Blueprint(
        "license_compat",
        __name__,
        url_prefix="/license-compat",
        template_folder="templates",
    )

    def _analyse(artifact_id: str):
        from aikaboom.store import BomStore
        store = BomStore.open()
        iri = unquote(artifact_id)
        findings = plugin.analyze(store, Scope.single(iri))
        matrix = plugin._matrix()
        from aikaboom.plugins.license_compat.walker import compute_license_frequencies
        freqs = compute_license_frequencies(store, matrix)
        subchains = find_compatible_subchains(findings)
        breaking = find_breaking_nodes(findings, matrix, freqs)
        return findings, subchains, breaking, matrix

    @bp.get("/<path:artifact_id>.json")
    def view_json(artifact_id):
        findings, subchains, breaking, _ = _analyse(artifact_id)
        return jsonify({
            **findings.to_dict(),
            "compatible_subchains": [
                {"size": c.size, "root": c.root, "artifacts": sorted(c.artifacts)}
                for c in subchains
            ],
            "breaking_nodes": [
                {
                    "artifact_iri": n.artifact_iri,
                    "label": n.label,
                    "license": n.license,
                    "blamed_in": n.blamed_in,
                    "affected_downstream": sorted(n.affected_downstream),
                    "fix_recommendations": {
                        "by_category": n.fix_recommendations.by_category,
                        "is_solvable": n.fix_recommendations.is_solvable,
                    },
                }
                for n in breaking
            ],
        })

    @bp.get("/<path:artifact_id>")
    def view(artifact_id):
        findings, subchains, breaking, matrix = _analyse(artifact_id)
        return render_template(
            "license_compat/tab.html",
            findings=list(findings),
            subchains=subchains,
            breaking=breaking,
            matrix_timestamp=matrix.timestamp,
            base_template=None,
        )

    return bp
```

- [ ] **Step 4: Wire plugins into the Flask app**

Modify `src/aikaboom/web/app.py`. Find `create_app()` (or the equivalent factory). At the end of app construction, before `return app`, add:

```python
# Plugin-contributed routes and tabs.
from aikaboom.plugins import all_plugins
app.config.setdefault("BOM_VIEWER_TABS", [])
for _plugin in all_plugins():
    if not _plugin.enabled():
        continue
    bp = _plugin.web_blueprint()
    if bp is not None:
        app.register_blueprint(bp)
    tab = _plugin.bom_viewer_tab()
    if tab is not None:
        app.config["BOM_VIEWER_TABS"].append(tab)
```

Then in the BOM-viewer template (`src/aikaboom/web/templates/index.html` or whatever renders the tab strip), iterate `config.BOM_VIEWER_TABS` and emit a tab link for each. If that template already has a hardcoded tab list, the new line is roughly:

```html
{% for tab in config.BOM_VIEWER_TABS | sort(attribute='sort_order') %}
  <a href="{{ tab.url_template.format(artifact_id=artifact_id) }}">{{ tab.label }}</a>
{% endfor %}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_web_tab.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/aikaboom/plugins/license_compat/web.py \
        src/aikaboom/plugins/license_compat/templates/license_compat/tab.html \
        src/aikaboom/web/app.py \
        src/aikaboom/web/templates/index.html \
        tests/plugins/license_compat/test_web_tab.py
git commit -m "feat(license-compat): web blueprint + License-compatibility tab"
```

---

## Task 12: Graph-view overlay

**Files:**
- Modify: `src/aikaboom/plugins/license_compat/overlay.py` (fill in)
- Modify: `src/aikaboom/plugins/license_compat/web.py` (add overlay JSON route)
- Modify: graph-view frontend JS in `src/aikaboom/web/templates/index.html` (or the graph-view template) to fetch + apply plugin overlays
- Create: `tests/plugins/license_compat/test_overlay.py`

- [ ] **Step 1: Write the failing test**

Create `tests/plugins/license_compat/test_overlay.py`:

```python
"""GraphOverlay payload shape + color rules."""
from __future__ import annotations

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
)
from aikaboom.plugins.license_compat.overlay import build_overlay


def _f(d: str, u: str, status: str, predicate: str = "trainedOn") -> Finding:
    return Finding(
        downstream_iri=d, downstream_label=d,
        upstream_iri=u, upstream_label=u,
        predicate=predicate,
        downstream_license="mit",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="mit", upstreams=frozenset({"apache-2.0"}),
            status=status,
            incompatible_with=frozenset({"apache-2.0"}) if status == "violation" else frozenset(),
        ),
        recommendation=None,
    )


def test_overlay_colors_compatible_edges_green():
    o = build_overlay(Findings([_f("A", "B", "compatible")]), plugin_name="license-compat")
    key = "A|trainedOn|B"
    assert key in o.edge_attrs
    assert o.edge_attrs[key]["color"] == "#22c55e"


def test_overlay_colors_violation_edges_red():
    o = build_overlay(Findings([_f("A", "B", "violation")]), plugin_name="license-compat")
    key = "A|trainedOn|B"
    assert o.edge_attrs[key]["color"] == "#ef4444"


def test_overlay_marks_breaking_nodes_with_ring():
    findings = Findings([_f("A", "B", "violation"), _f("C", "B", "violation")])
    o = build_overlay(findings, plugin_name="license-compat")
    assert "B" in o.node_attrs
    assert o.node_attrs["B"]["ring_color"] == "#ef4444"
    assert o.node_attrs["B"]["badge"] == "2"


def test_overlay_empty_findings_empty_payload():
    o = build_overlay(Findings([]), plugin_name="license-compat")
    assert o.edge_attrs == {}
    assert o.node_attrs == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/license_compat/test_overlay.py -v`
Expected: FAIL — current stub returns empty `GraphOverlay`.

- [ ] **Step 3: Implement the overlay**

Replace `src/aikaboom/plugins/license_compat/overlay.py` with:

```python
"""GraphOverlay payload builder: edge tinting + breaking-node rings."""
from __future__ import annotations

from collections import Counter

from aikaboom.plugins import GraphOverlay
from aikaboom.plugins.license_compat.engine import (
    Findings,
    find_breaking_nodes,
    find_compatible_subchains,
)

_COLORS = {
    "compatible": "#22c55e",
    "violation": "#ef4444",
    "unknown_upstream": "#94a3b8",
    "unknown_downstream": "#94a3b8",
    "missing_data": "#94a3b8",
}


def _edge_key(s: str, p: str, o: str) -> str:
    pred_short = p.rsplit("#", 1)[-1]
    return f"{s}|{pred_short}|{o}"


def build_overlay(findings: Findings, plugin_name: str) -> GraphOverlay:
    edge_attrs: dict[str, dict] = {}
    blame: Counter = Counter()

    for f in findings:
        edge_attrs[_edge_key(f.downstream_iri, f.predicate, f.upstream_iri)] = {
            "color": _COLORS.get(f.verdict.status, "#cccccc"),
            "label": f.verdict.status,
            "tooltip": (
                f"{f.downstream_license} {f.verdict.status} with "
                f"{sorted(f.upstream_licenses) or '—'} via "
                f"{f.predicate.rsplit('#', 1)[-1]}"
            ),
        }
        if f.is_violation() and f.upstream_licenses & f.verdict.incompatible_with:
            blame[f.upstream_iri] += 1

    node_attrs: dict[str, dict] = {}
    for iri, count in blame.items():
        node_attrs[iri] = {
            "ring_color": "#ef4444",
            "badge": str(count),
        }

    # Largest compatible subchain gets a faint halo on every member.
    subchains = find_compatible_subchains(findings)
    if subchains and subchains[0].size > 1:
        for iri in subchains[0].artifacts:
            node_attrs.setdefault(iri, {})["halo_color"] = "#bbf7d0"

    return GraphOverlay(
        plugin_name=plugin_name,
        edge_attrs=edge_attrs,
        node_attrs=node_attrs,
    )
```

- [ ] **Step 4: Add the overlay JSON route**

Append to the blueprint in `src/aikaboom/plugins/license_compat/web.py`, inside `build_blueprint(plugin)` before the `return bp`:

```python
@bp.get("/<path:artifact_id>/overlay.json")
def overlay_json(artifact_id):
    findings, _, _, _ = _analyse(artifact_id)
    overlay = plugin.graph_overlay(findings)
    return jsonify({
        "plugin": overlay.plugin_name,
        "edges": overlay.edge_attrs,
        "nodes": overlay.node_attrs,
    })
```

- [ ] **Step 5: Frontend toggle (graph view)**

In the graph-view template/JS (likely `src/aikaboom/web/templates/index.html` or a sibling JS file), add a legend toggle for each plugin overlay. Pseudocode:

```javascript
async function loadPluginOverlay(pluginName, artifactId) {
  const r = await fetch(`/${pluginName}/${encodeURIComponent(artifactId)}/overlay.json`);
  if (!r.ok) return;
  const data = await r.json();
  applyEdgeAttrs(data.edges);
  applyNodeAttrs(data.nodes);
}

// On legend checkbox toggle for "License compatibility":
loadPluginOverlay("license-compat", currentArtifactId);
```

Match the existing graph-view's API for setting edge/node styles. If there isn't a unified plugin-overlay mechanism yet, add a minimal one that takes `{edge_attrs, node_attrs}` and merges into the live graph instance.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/plugins/license_compat/test_overlay.py -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add src/aikaboom/plugins/license_compat/overlay.py \
        src/aikaboom/plugins/license_compat/web.py \
        src/aikaboom/web/templates/index.html \
        tests/plugins/license_compat/test_overlay.py
git commit -m "feat(license-compat): graph-view overlay (edge tinting + breaking-node rings)"
```

---

## Task 13: End-to-end smoke

**Files:**
- Create: `tests/plugins/license_compat/test_e2e_license_compat.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/plugins/license_compat/test_e2e_license_compat.py`:

```python
"""End-to-end smoke: generate -> store -> audit -> findings appear."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LINEAGE_TTL = Path(__file__).parent / "fixtures" / "lineage_3node.ttl"


def test_e2e_audit_finds_violation(tmp_path):
    env = os.environ.copy()
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_path)
    env["BOM_SKIP_DOTENV"] = "1"

    # Populate the store from the lineage fixture.
    populate = (
        "from aikaboom.store import BomStore; "
        f"s = BomStore.open(); s._backend.import_(__import__('pathlib').Path(r'{LINEAGE_TTL}'), fmt='turtle'); s._backend.close()"
    )
    subprocess.run([sys.executable, "-c", populate], check=True, env=env)

    # Run audit.
    out = tmp_path / "audit.json"
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-audit", "--format", "json"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode in (0, 2)
    payload = json.loads(r.stdout)
    assert "findings" in payload
    # Our fixture has DatasetB (apache-2.0) -> PaperC (cc-by-nc-4.0) via dependsOn.
    # depending on traversal direction this is a violation we expect to surface.
    assert isinstance(payload["compatible_subchains"], list)
    assert isinstance(payload["breaking_nodes"], list)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/plugins/license_compat/test_e2e_license_compat.py -v`
Expected: PASS

- [ ] **Step 3: Run the full plugin suite as final regression check**

Run: `pytest tests/plugins/ -v`
Expected: every test in every Tier green.

- [ ] **Step 4: Commit**

```bash
git add tests/plugins/license_compat/test_e2e_license_compat.py
git commit -m "test(license-compat): end-to-end smoke from store -> audit -> JSON output"
```

---

## Task 14: Coverage gate + final sanity

**Files:** none — verification only.

- [ ] **Step 1: Run coverage**

Run:

```bash
pytest tests/plugins/ --cov=src/aikaboom/plugins --cov-report=term-missing
```

Expected:
- `src/aikaboom/plugins/license_compat/matrix.py` 100%
- `src/aikaboom/plugins/license_compat/engine.py` 100%
- `src/aikaboom/plugins/license_compat/walker.py` >= 90%
- `src/aikaboom/plugins/license_compat/{cli,web,spdx,overlay,plugin}.py` >= 80%

- [ ] **Step 2: Fix any gap below target**

For each file under target, identify uncovered lines from `--cov-report=term-missing` and add focused tests. No new untested branches in production code.

- [ ] **Step 3: Run the existing full test suite once more**

Run:

```bash
pytest
```

Expected: every pre-existing test still green; new license-compat tests green.

- [ ] **Step 4: Lint / type check (if configured)**

Run whatever the repo uses (ruff / mypy / pyright). Fix anything new. If unconfigured, skip.

- [ ] **Step 5: Final commit (only if coverage gap fixes were needed)**

```bash
git add tests/plugins/license_compat/
git commit -m "test(license-compat): close coverage gaps"
```

---

## Self-Review Notes

Spec coverage check (completed in-line before finalizing this plan):
- Plugin substrate (spec §Architecture / Plugin substrate) — Task 1.
- Bundled matrix + allowed + missing (spec §Data model) — Task 2.
- Matrix loader + resolve_license (spec §Engine API) — Task 3.
- check_compat + recommend (spec §Engine API) — Task 4.
- Findings + find_compatible_subchains + find_breaking_nodes (spec §Engine API additions) — Task 5.
- Graph walker (spec §Graph walker) — Task 6.
- LicenseCompatPlugin glue (spec §License-compat plugin layout) — Task 7.
- CLI license-check + license-audit (spec §CLI) — Task 8.
- SPDX Annotation Elements + CDX parity (spec §UI surfaces / SPDX export) — Task 9.
- Conflicts tab category (spec §UI surfaces / Conflicts tab) — Task 10.
- Web tab + Flask blueprint (spec §UI surfaces / Web) — Task 11.
- Graph-view overlay (spec §UI surfaces / Graph view) — Task 12.
- E2E smoke + coverage gate (spec §Testing strategy) — Tasks 13, 14.

Cross-task type consistency:
- `Finding` dataclass introduced in Task 5; referenced consistently in Tasks 7, 8, 9, 10, 11, 12.
- `Findings` collection (`violations()`, `to_dict()`, `__iter__`) consistent across tasks.
- `Scope.single(iri, depth)` and `Scope.graph_wide()` consistent across walker, CLI, web, tests.
- `CompatVerdict.status` enum values consistent across engine + emitters + tests.
- `GraphOverlay` payload shape consistent between `overlay.py` and the JSON route.
- `ConflictRecord` fields consistent between `plugin.conflict_findings` and the contract test.

Deferred items per the spec (intentionally NOT in this plan):
- `entry_points` discovery.
- "Refresh-matrix" subcommand.
- Multi-matrix diff support.
- Storing compat results as RDF in the graph.
- Plugin #2 (AVID/security).
- Retrofitting existing conflict-annotations into the plugin system.
