def test_avid_security_is_registered_on_import():
    import aikaboom.plugins  # eager-imports in-tree plugins
    from aikaboom.plugins import get
    plugin = get("avid-security")
    assert plugin is not None
    assert plugin.name == "avid-security"


def test_both_plugins_registered():
    import aikaboom.plugins
    from aikaboom.plugins import all_plugins
    names = {p.name for p in all_plugins()}
    assert "license-compat" in names
    assert "avid-security" in names
