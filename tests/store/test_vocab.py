from rdflib import Namespace, URIRef
from aikaboom.store import vocab


class TestNamespaces:
    def test_aibom_namespace_defined(self):
        assert isinstance(vocab.AIBOM, Namespace)
        assert str(vocab.AIBOM).startswith("https://aikaboom.dev/aibom#")

    def test_bom_namespace_defined(self):
        assert isinstance(vocab.BOM, Namespace)


class TestCoreClasses:
    def test_artifact_class(self):
        assert isinstance(vocab.Artifact, URIRef)
        assert str(vocab.Artifact) == str(vocab.AIBOM) + "Artifact"

    def test_model_subclass(self):
        assert str(vocab.Model) == str(vocab.AIBOM) + "Model"

    def test_dataset_paper_coderepo(self):
        assert str(vocab.Dataset) == str(vocab.AIBOM) + "Dataset"
        assert str(vocab.Paper) == str(vocab.AIBOM) + "Paper"
        assert str(vocab.CodeRepo) == str(vocab.AIBOM) + "CodeRepo"

    def test_artifact_version(self):
        assert str(vocab.ArtifactVersion) == str(vocab.AIBOM) + "ArtifactVersion"

    def test_bom_claim(self):
        assert str(vocab.BOMClaim) == str(vocab.AIBOM) + "BOMClaim"

    def test_generation_run(self):
        assert str(vocab.GenerationRun) == str(vocab.AIBOM) + "GenerationRun"

    def test_trust_vote(self):
        assert str(vocab.TrustVote) == str(vocab.AIBOM) + "TrustVote"


class TestPredicates:
    def test_has_version(self):
        assert str(vocab.hasVersion) == str(vocab.AIBOM) + "hasVersion"

    def test_has_claim(self):
        assert str(vocab.hasClaim) == str(vocab.AIBOM) + "hasClaim"

    def test_canonical_claim(self):
        assert str(vocab.canonicalClaim) == str(vocab.AIBOM) + "canonicalClaim"

    def test_generated_by(self):
        assert str(vocab.generatedBy) == str(vocab.AIBOM) + "generatedBy"

    def test_trust_score(self):
        assert str(vocab.trustScore) == str(vocab.AIBOM) + "trustScore"

    def test_use_case_and_mode(self):
        assert str(vocab.useCase) == str(vocab.AIBOM) + "useCase"
        assert str(vocab.mode) == str(vocab.AIBOM) + "mode"

    def test_identifier_and_primary(self):
        assert str(vocab.identifier) == str(vocab.AIBOM) + "identifier"
        assert str(vocab.primaryIdentifier) == str(vocab.AIBOM) + "primaryIdentifier"

    def test_asserted_by_and_conflict_kind(self):
        assert str(vocab.assertedBy) == str(vocab.AIBOM) + "assertedBy"
        assert str(vocab.conflictKind) == str(vocab.AIBOM) + "conflictKind"
