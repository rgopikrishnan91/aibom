"""Mapper: BOM JSON ↔ RDF quads."""
from rdflib import Literal, URIRef

from aikaboom.store.mapper import bom_to_rdf
from aikaboom.store import vocab
from aikaboom.store.naming import Identifier


class TestBomToRdf:
    def test_creates_artifact_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        artifacts = list(ds.subjects(predicate=vocab.AIBOM["primaryIdentifier"]))
        assert len(artifacts) == 1, f"expected one artifact, got {artifacts}"

    def test_creates_artifact_version_node(self, sample_bom, sample_run_meta):
        ds, _ = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        versions = list(ds.subjects(predicate=None, object=vocab.ArtifactVersion))
        assert any(str(v).startswith("bom:version/") for v in versions)

    def test_creates_bom_claim_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        assert claim_iri.startswith("bom:claim/")
        assert (URIRef(claim_iri), URIRef(vocab.useCase), Literal("license")) in [
            (s, p, o) for s, p, o, _ in ds.quads()
        ]

    def test_creates_generation_run_node(self, sample_bom, sample_run_meta):
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        runs = list(ds.subjects(predicate=None, object=vocab.GenerationRun))
        assert len(runs) == 1
        run = runs[0]
        assert (run, URIRef(vocab.llmModel), Literal("anthropic/claude-3-haiku")) in [
            (s, p, o) for s, p, o, _ in ds.quads()
        ]

    def test_field_claim_with_source(self, sample_bom, sample_run_meta):
        """Each direct_field becomes a triple + reified annotation with source."""
        ds, claim_iri = bom_to_rdf(
            sample_bom,
            sample_run_meta,
            identifiers=[Identifier("huggingface", "mistralai/Mistral-7B-v0.1")],
        )
        # The `suppliedBy` field with value "mistralai" should appear.
        quads = list(ds.quads())
        triples = [(str(s), str(p), str(o)) for s, p, o, _ in quads]
        assert any(claim_iri in t[0] and "mistralai" in t[2] for t in triples)

    def test_handles_explicit_null_packageVersion(self, sample_bom, sample_run_meta):
        """A BOM with packageVersion: None should not crash."""
        from aikaboom.store.naming import Identifier
        sample_bom["direct_fields"]["packageVersion"] = None
        ds, claim_iri = bom_to_rdf(
            sample_bom, sample_run_meta,
            identifiers=[Identifier("huggingface", "x/y")],
        )
        assert claim_iri.startswith("bom:claim/")
