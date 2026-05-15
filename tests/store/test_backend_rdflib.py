import pytest
from aikaboom.store.rdflib_backend import RDFLibBackend


@pytest.fixture
def backend(tmp_store_dir):
    return RDFLibBackend(tmp_store_dir)


class TestRDFLibBackend:
    def test_add_and_ask(self, backend):
        backend.update("INSERT DATA { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }")
        assert backend.ask("ASK { <bom:test/1> <https://aikaboom.dev/aibom#trustScore> 0.5 }")

    def test_select_returns_bindings(self, backend):
        backend.update(
            "INSERT DATA { <bom:test/2> <https://aikaboom.dev/aibom#useCase> 'license' }"
        )
        rows = list(
            backend.select(
                "SELECT ?u WHERE { <bom:test/2> <https://aikaboom.dev/aibom#useCase> ?u }"
            )
        )
        assert len(rows) == 1
        assert str(rows[0]["u"]) == "license"

    def test_persistence_across_reopen(self, tmp_store_dir):
        b1 = RDFLibBackend(tmp_store_dir)
        b1.update("INSERT DATA { <bom:test/x> <https://aikaboom.dev/aibom#useCase> 'license' }")
        b1.close()
        b2 = RDFLibBackend(tmp_store_dir)
        assert b2.ask("ASK { <bom:test/x> <https://aikaboom.dev/aibom#useCase> 'license' }")
