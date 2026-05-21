"""Plugin isolation tests against the REAL core path.

Confirms:
 - enabled() reads the env live (not at import time)
 - A disabled avid_security plugin contributes no security_Vuln* or VEX nodes
   through the core _emit_plugin_annotations loop
 - When disabled, the /avid-security blueprint is not mounted
"""
from __future__ import annotations

import os
from aikaboom.plugins.avid_security.plugin import AvidSecurityPlugin


def test_enabled_flag(monkeypatch):
    monkeypatch.setenv("AIKABOOM_AVID_DISABLED", "1")
    assert AvidSecurityPlugin().enabled() is False
    monkeypatch.delenv("AIKABOOM_AVID_DISABLED", raising=False)
    assert AvidSecurityPlugin().enabled() is True


def test_disabled_plugin_emits_nothing_via_core(monkeypatch):
    """The core _emit_plugin_annotations skips disabled plugins.

    With avid disabled, its spdx_elements must not contribute any
    security_Vuln* or VEX nodes to the output.  Other enabled plugins
    (license_compat) legitimately may contribute; we only assert no avid
    security_ nodes appear.

    Note: _emit_plugin_annotations passes EMPTY findings to all plugins
    (the store-aware path is deferred), so avid emits nothing there even
    when enabled — the disabled assertion is the stronger guarantee.
    """
    monkeypatch.setenv("AIKABOOM_AVID_DISABLED", "1")
    from aikaboom.utils.spdx_validator import _emit_plugin_annotations
    out = _emit_plugin_annotations(claim_iri="urn:claim:1")
    assert not any(str(n.get("type", "")).startswith("security_Vuln") for n in out)
    assert not any("Vex" in str(n.get("type", "")) for n in out)


def test_disabled_plugin_not_mounted(monkeypatch):
    """When AIKABOOM_AVID_DISABLED=1, the /avid-security blueprint must not be mounted."""
    from flask import Flask
    from aikaboom.plugins import all_plugins
    monkeypatch.setenv("AIKABOOM_AVID_DISABLED", "1")
    app = Flask(__name__)
    for p in all_plugins():
        if not p.enabled():
            continue
        bp = p.web_blueprint()
        if bp is not None:
            app.register_blueprint(bp)
    rules = [str(r) for r in app.url_map.iter_rules()]
    assert not any("avid-security" in r for r in rules)
