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


def test_testedon_and_dependson_predicates_exist():
    from aikaboom.store import vocab
    assert str(vocab.testedOn) == "https://aikaboom.dev/aibom#testedOn"
    assert str(vocab.dependsOn) == "https://aikaboom.dev/aibom#dependsOn"


import pytest  # noqa: E402
from aikaboom.store import vocab as vocab_module  # noqa: E402


def _public_urirefs():
    """Collect every public URIRef constant exported by vocab.py."""
    out = {}
    for name in dir(vocab_module):
        if name.startswith("_"):
            continue
        value = getattr(vocab_module, name)
        if isinstance(value, URIRef):
            out[name] = value
    return out


@pytest.mark.parametrize("name,uriref", sorted(_public_urirefs().items()))
def test_every_public_uriref_resolves_under_aibom_namespace(name, uriref):
    """Every URIRef constant must point to <AIBOM>{local-name}.

    Local name is the Python identifier, except `implicit_use` /
    `implicit_validate` use the hyphenated forms `implicit-use` /
    `implicit-validate`.
    """
    snake_to_hyphen = {
        "implicit_use": "implicit-use",
        "implicit_validate": "implicit-validate",
    }
    expected_local = snake_to_hyphen.get(name, name)
    assert (
        str(uriref) == str(vocab_module.AIBOM) + expected_local
    ), f"vocab.{name} points to {uriref!r}, expected {str(vocab_module.AIBOM) + expected_local!r}"
