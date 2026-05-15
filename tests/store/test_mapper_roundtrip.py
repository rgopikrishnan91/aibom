"""Round-trip: BOM JSON → RDF → BOM JSON should be value-preserving."""
import json
from pathlib import Path
import pytest

from aikaboom.store.mapper import bom_to_rdf, rdf_to_bom
from aikaboom.store.naming import Identifier
from tests.store.conftest import SAMPLE_RUN_META


REPO_ROOT = Path(__file__).resolve().parents[2]
BOM_CORPUS = Path(__file__).resolve().parent / "fixtures" / "bom_corpus"


def _collect_test_boms() -> list[tuple[str, dict]]:
    boms = []
    if not BOM_CORPUS.exists():
        return boms
    for f in BOM_CORPUS.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and ("direct_fields" in data or "rag_fields" in data):
                boms.append((f.stem, data))
        except (json.JSONDecodeError, OSError):
            continue
    return boms


_DISCOVERED_BOMS = _collect_test_boms()
assert len(_DISCOVERED_BOMS) >= 1, (
    f"No BOM corpus found in {BOM_CORPUS}. The round-trip test is meaningless "
    f"without real-BOM coverage; check that the fixtures directory is tracked."
)


@pytest.mark.parametrize("name,bom_json", _DISCOVERED_BOMS)
def test_roundtrip_preserves_direct_fields(name, bom_json):
    """Every direct_field value survives JSON → RDF → JSON."""
    ids = [Identifier("huggingface", bom_json.get("repo_id") or bom_json.get("model_id") or "unknown/unknown")]
    ds, claim_iri = bom_to_rdf(bom_json, SAMPLE_RUN_META, identifiers=ids)
    reconstructed = rdf_to_bom(ds, claim_iri)
    for field_name, triplet in (bom_json.get("direct_fields") or {}).items():
        if not isinstance(triplet, dict):
            # Flat-string direct_fields aren't modeled by the vocab; bom_to_rdf
            # skips them, so the round-trip cannot recover them. Mirror that.
            continue
        if triplet.get("value") is None:
            continue
        rt = reconstructed.get("direct_fields", {}).get(field_name)
        assert rt is not None, f"missing direct field {field_name} after round-trip in {name}"
        assert rt["value"] == triplet["value"], f"value mismatch on {field_name} in {name}"
        assert rt["source"] == triplet["source"], f"source mismatch on {field_name} in {name}"


def test_roundtrip_simple_bom(sample_bom, sample_run_meta):
    """Sanity round-trip on the in-memory fixture."""
    ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
    ds, claim_iri = bom_to_rdf(sample_bom, sample_run_meta, identifiers=ids)
    reconstructed = rdf_to_bom(ds, claim_iri)
    assert reconstructed["direct_fields"]["suppliedBy"]["value"] == "mistralai"
    assert reconstructed["direct_fields"]["suppliedBy"]["source"] == "huggingface"


def test_roundtrip_preserves_inter_source_conflict(sample_bom, sample_run_meta):
    """A direct field with an inter-source conflict round-trips losslessly."""
    sample_bom["direct_fields"]["license"] = {
        "value": "Apache-2.0",
        "source": "huggingface",
        "conflict": {"type": "inter"},
    }
    ids = [Identifier("huggingface", "mistralai/Mistral-7B-v0.1")]
    ds, claim_iri = bom_to_rdf(sample_bom, sample_run_meta, identifiers=ids)
    reconstructed = rdf_to_bom(ds, claim_iri)
    license_field = reconstructed["direct_fields"].get("license")
    assert license_field is not None
    assert license_field["value"] == "Apache-2.0"
    assert license_field["source"] == "huggingface"
    assert license_field["conflict"] == {"type": "inter"}
