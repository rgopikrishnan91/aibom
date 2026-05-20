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
