"""
Hardened SPDX 3.0.1 validator tests.

These tests ARE the spec compliance proof: each one removes or corrupts
a required element and asserts the validator catches it. If all pass,
any document accepted by our validator is structurally sound per
SPDX 3.0.1.
"""
import copy
import pytest
from aikaboom.utils.spdx_validator import SPDXValidator


def _make_valid_ai_spdx():
    """Generate a known-good SPDX 3.0.1 AI BOM for mutation testing."""
    v = SPDXValidator(bom_type='ai')
    bom = {
        "repo_id": "test/model",
        "direct_fields": {
            "suppliedBy": {"value": "TestOrg", "source": "hf", "conflict": None},
            "license": {"value": "MIT", "source": "hf", "conflict": None},
            "downloadLocation": {"value": "https://example.com", "source": "hf", "conflict": None},
        },
        "rag_fields": {
            "model_name": {"value": "TestModel", "source": "hf", "conflict": None},
            "domain": {"value": "NLP", "source": "hf", "conflict": None},
        },
    }
    return v.validate_and_convert(bom)


def _make_valid_dataset_spdx():
    v = SPDXValidator(bom_type='data')
    bom = {
        "dataset_id": "test/ds",
        "direct_metadata": {"name": "TestDS", "license": "Apache-2.0"},
        "rag_metadata": {"intendedUse": "Research"},
        "urls": {"huggingface": "https://hf.co/datasets/test/ds"},
    }
    return v.validate_and_convert(bom)


def _find_elem(graph, elem_type):
    return next((e for e in graph if e.get('type') == elem_type), None)


def _remove_elem(spdx, elem_type):
    """Return a copy with all elements of the given type removed."""
    out = copy.deepcopy(spdx)
    out['@graph'] = [e for e in out['@graph'] if e.get('type') != elem_type]
    return out


class TestOurEmitterOutputPasses:
    """Our own emitter's output must pass our own validator."""

    def test_ai_bom_valid(self):
        spdx = _make_valid_ai_spdx()
        v = SPDXValidator(bom_type='ai')
        ok, errs = v.validate_spdx_bom(spdx)
        assert ok, f"AI BOM emitter output failed validation: {errs}"

    def test_dataset_bom_valid(self):
        spdx = _make_valid_dataset_spdx()
        v = SPDXValidator(bom_type='data')
        ok, errs = v.validate_spdx_bom(spdx)
        assert ok, f"Dataset BOM emitter output failed validation: {errs}"

    def test_strict_mode_passes(self):
        spdx = _make_valid_ai_spdx()
        v = SPDXValidator(bom_type='ai')
        ok, errs = v.validate_spdx_bom(spdx, strict=True)
        assert ok, f"AI BOM failed strict validation: {errs}"


class TestMissingRequiredElements:

    def test_missing_creation_info(self):
        spdx = _remove_elem(_make_valid_ai_spdx(), 'CreationInfo')
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('CreationInfo' in e for e in errs)

    def test_missing_spdx_document(self):
        spdx = _remove_elem(_make_valid_ai_spdx(), 'SpdxDocument')
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('SpdxDocument' in e for e in errs)

    def test_missing_bom(self):
        spdx = _remove_elem(_make_valid_ai_spdx(), 'Bom')
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('Bom' in e for e in errs)

    def test_missing_ai_package(self):
        spdx = _remove_elem(_make_valid_ai_spdx(), 'AI_AIPackage')
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('AI_AIPackage' in e for e in errs)

    def test_missing_dataset_package(self):
        spdx = _remove_elem(_make_valid_dataset_spdx(), 'dataset_DatasetPackage')
        ok, errs = SPDXValidator(bom_type='data').validate_spdx_bom(spdx)
        assert not ok
        assert any('dataset_DatasetPackage' in e for e in errs)


class TestCreationInfoProperties:

    def test_wrong_spec_version(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        ci = _find_elem(spdx['@graph'], 'CreationInfo')
        ci['specVersion'] = '2.3'
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any("specVersion must be '3.0.1'" in e for e in errs)

    def test_missing_created_timestamp(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        ci = _find_elem(spdx['@graph'], 'CreationInfo')
        del ci['created']
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any("missing 'created'" in e for e in errs)

    def test_missing_created_by(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        ci = _find_elem(spdx['@graph'], 'CreationInfo')
        del ci['createdBy']
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any("missing 'createdBy'" in e for e in errs)

    def test_bad_timestamp_format_strict(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        ci = _find_elem(spdx['@graph'], 'CreationInfo')
        ci['created'] = '2024-01-01 12:00:00'
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx, strict=True)
        assert not ok
        assert any('ISO 8601' in e for e in errs)


class TestCrossReferenceIntegrity:

    def test_broken_creation_info_ref(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        pkg = _find_elem(spdx['@graph'], 'AI_AIPackage')
        pkg['creationInfo'] = 'urn:spdx:NONEXISTENT'
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('unknown ID' in e and 'NONEXISTENT' in e for e in errs)

    def test_broken_relationship_from(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        rel = _find_elem(spdx['@graph'], 'Relationship')
        rel['from'] = 'urn:spdx:GHOST'
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('GHOST' in e for e in errs)

    def test_broken_relationship_to(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        rel = _find_elem(spdx['@graph'], 'Relationship')
        rel['to'] = ['urn:spdx:PHANTOM']
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('PHANTOM' in e for e in errs)

    def test_duplicate_spdx_ids(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        # Give two different elements the same spdxId
        spdx['@graph'][0]['spdxId'] = 'urn:spdx:DUPE'
        spdx['@graph'][1]['spdxId'] = 'urn:spdx:DUPE'
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any('Duplicate ID' in e for e in errs)


class TestDocumentAndBomProperties:

    def test_spdx_document_missing_profile_conformance(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        doc = _find_elem(spdx['@graph'], 'SpdxDocument')
        doc['profileConformance'] = []
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any("profileConformance must include 'core'" in e for e in errs)

    def test_spdx_document_missing_root_element(self):
        spdx = copy.deepcopy(_make_valid_ai_spdx())
        doc = _find_elem(spdx['@graph'], 'SpdxDocument')
        del doc['rootElement']
        ok, errs = SPDXValidator(bom_type='ai').validate_spdx_bom(spdx)
        assert not ok
        assert any("missing 'rootElement'" in e for e in errs)
