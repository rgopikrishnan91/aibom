from aikaboom.plugins.base import Plugin
from aikaboom.utils.spdx_validator import _emit_plugin_annotations


def test_plugin_protocol_has_spdx_elements():
    # The Protocol must declare spdx_elements so plugins can emit arbitrary SPDX nodes.
    assert hasattr(Plugin, "spdx_elements")


def test_emit_plugin_collects_spdx_elements(monkeypatch):
    class _ElemPlugin:
        name = "elem-test"
        def enabled(self): return True
        def spdx_annotations(self, claim_iri, findings):
            return [{"type": "Annotation", "spdxId": "urn:ann"}]
        def spdx_elements(self, claim_iri, findings):
            return [{"type": "security_Vulnerability", "spdxId": "urn:vuln"}]

    monkeypatch.setattr(
        "aikaboom.utils.spdx_validator.all_plugins",
        lambda: [_ElemPlugin()],
        raising=False,
    )
    # _emit_plugin_annotations imports all_plugins lazily from aikaboom.plugins;
    # patch there too:
    import aikaboom.plugins
    monkeypatch.setattr(aikaboom.plugins, "all_plugins", lambda: [_ElemPlugin()])

    out = _emit_plugin_annotations(claim_iri="urn:claim:1")
    ids = {n["spdxId"] for n in out}
    assert "urn:ann" in ids
    assert "urn:vuln" in ids


def test_emit_plugin_tolerates_plugin_without_spdx_elements(monkeypatch):
    # A plugin that only implements spdx_annotations (like license_compat
    # historically) must not break the loop.
    class _AnnOnly:
        name = "ann-only"
        def enabled(self): return True
        def spdx_annotations(self, claim_iri, findings):
            return [{"type": "Annotation", "spdxId": "urn:ann2"}]
        # no spdx_elements

    import aikaboom.plugins
    monkeypatch.setattr(aikaboom.plugins, "all_plugins", lambda: [_AnnOnly()])
    out = _emit_plugin_annotations(claim_iri="urn:c")
    assert {"urn:ann2"} == {n["spdxId"] for n in out}
