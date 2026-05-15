import pytest
pytest.importorskip("pyoxigraph")

from aikaboom.store.backend import open_backend


@pytest.fixture
def backend(tmp_store_dir, monkeypatch):
    monkeypatch.setenv("AIKABOOM_GRAPH_BACKEND", "oxigraph")
    return open_backend()


class TestOxigraphBackend:
    def test_open_returns_backend(self, backend):
        assert backend is not None

    def test_add_and_ask(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )
        result = backend.ask(
            "ASK { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }"
        )
        assert result is True

    def test_select_returns_bindings(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/2> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        rows = list(backend.select(
            "SELECT ?u WHERE { <bom:test/2> <https://aikaboom.dev/aibom#useCase> ?u }"
        ))
        assert len(rows) == 1
        assert str(rows[0]["u"]) == "license"

    def test_export_then_import_roundtrip(self, backend, tmp_path):
        backend.update(
            "INSERT DATA { <bom:test/3> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        dump = tmp_path / "dump.nq"
        backend.export(dump, fmt="nquads")
        assert dump.exists() and dump.stat().st_size > 0

        # Import the dump into a fresh store and verify the triple survived.
        import os
        fresh_dir = tmp_path / "fresh_store"
        fresh_dir.mkdir()
        prev_dir = os.environ.get("AIKABOOM_GRAPH_DIR")
        os.environ["AIKABOOM_GRAPH_DIR"] = str(fresh_dir)
        try:
            from aikaboom.store.backend import open_backend
            fresh = open_backend()
            fresh.import_(dump, fmt="nquads")
            assert fresh.ask(
                "ASK { <bom:test/3> <https://aikaboom.dev/aibom#useCase> 'license' }"
            )
        finally:
            if prev_dir is not None:
                os.environ["AIKABOOM_GRAPH_DIR"] = prev_dir
            else:
                os.environ.pop("AIKABOOM_GRAPH_DIR", None)
