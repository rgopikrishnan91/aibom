import sys
import pytest

from aikaboom.store.backend import open_backend


def test_falls_back_to_rdflib_when_oxigraph_unavailable(monkeypatch, tmp_store_dir):
    """When AIKABOOM_GRAPH_BACKEND=auto and oxigraph import fails, use RDFLib."""
    # Hide pyoxigraph from import machinery.
    monkeypatch.setitem(sys.modules, "pyoxigraph", None)
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "auto")
    # Also unload our cached oxigraph_backend module so re-import sees the None.
    monkeypatch.delitem(sys.modules, "aikaboom.store.oxigraph_backend", raising=False)
    backend = open_backend()
    assert type(backend).__name__ == "RDFLibBackend"


def test_explicit_rdflib_backend(monkeypatch, tmp_store_dir):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "rdflib")
    backend = open_backend()
    assert type(backend).__name__ == "RDFLibBackend"


def test_explicit_oxigraph_backend_raises_if_missing(monkeypatch, tmp_store_dir):
    monkeypatch.setitem(sys.modules, "pyoxigraph", None)
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "oxigraph")
    monkeypatch.delitem(sys.modules, "aikaboom.store.oxigraph_backend", raising=False)
    with pytest.raises(ImportError):
        open_backend()
