import json
import pytest
from aikaboom.plugins.avid_security.matcher import ComponentMatcher, Match
from aikaboom.plugins.avid_security.walker import Component
from aikaboom.plugins.avid_security.snapshot import AvidIndex


@pytest.fixture
def index_with_bert(tmp_path):
    repo = tmp_path / "avid-db"
    (repo / "reports" / "2022").mkdir(parents=True)
    (repo / "reports" / "2022" / "AVID-2022-R0001.json").write_text(json.dumps({
        "metadata": {"report_id": "AVID-2022-R0001"},
        "affects": {"developer": [], "deployer": ["HuggingFace"],
                    "artifacts": [{"type": "Model", "name": "bert-base-uncased"}]},
        "impact": {"avid": {"risk_domain": ["Ethics"],
                            "sep_view": ["E0101: Group fairness"],
                            "lifecycle_view": ["L05: Evaluation"]}},
        "reported_date": "2022-11-09",
    }))
    (repo / "reports" / "2026").mkdir(parents=True)
    (repo / "reports" / "2026" / "AVID-2026-R0478.json").write_text(json.dumps({
        "metadata": {"report_id": "AVID-2026-R0478"},
        "affects": {"developer": ["Google"], "deployer": ["Together AI"],
                    "artifacts": [{"type": "Model", "name": "gemma-3n-E4B-it"}]},
        "impact": {"avid": {"risk_domain": ["Security"],
                            "sep_view": ["S0403: Adversarial Example"],
                            "lifecycle_view": ["L05: Evaluation"]}},
        "reported_date": "2026-03-16",
    }))
    idx = AvidIndex(db_path=tmp_path / "avid.sqlite")
    idx.build(repo_dir=repo)
    return idx


def _component(hf_path, developer=None, kind="Model", bases=(), scope="principal"):
    return Component(
        kind=kind, hf_path=hf_path, developer=developer,
        base_models=bases, scope_in_bom=scope,
        spdx_id=f"urn:aibom:pkg:{hf_path.replace('/', '__').lower()}",
    )


def test_tier1_exact_bare_name(index_with_bert):
    m = ComponentMatcher(index_with_bert)
    component = _component("bert-base-uncased", developer="Hugging Face")
    matches = m.match(component)
    assert len(matches) == 1
    assert matches[0].tier == 1
    assert matches[0].confidence == "high"
    assert matches[0].avid_report["report_id"] == "AVID-2022-R0001"
    assert matches[0].evidence["matched_via"] == "exact_bare_name"


def test_tier2_base_model_lineage(index_with_bert):
    m = ComponentMatcher(index_with_bert)
    # dslim/bert-base-NER is a real fine-tune of bert-base-uncased
    component = _component(
        "dslim/bert-base-NER", developer="dslim",
        bases=("bert-base-uncased",), scope="principal",
    )
    matches = m.match(component)
    assert any(m_.tier == 2 and m_.confidence == "medium" for m_ in matches)
    t2 = next(m_ for m_ in matches if m_.tier == 2)
    assert t2.avid_report["report_id"] == "AVID-2022-R0001"
    assert t2.evidence["matched_via"] == "base_model_lineage"
    assert t2.evidence["base_model"] == "bert-base-uncased"


def test_tier3_family_plus_developer(index_with_bert):
    m = ComponentMatcher(index_with_bert)
    # Same family as AVID-2026-R0478 (gemma-3n-E4B-it) but a different variant
    component = _component("google/gemma-3n-E2B-it", developer="Google")
    matches = m.match(component)
    t3 = [x for x in matches if x.tier == 3]
    assert len(t3) == 1
    assert t3[0].confidence == "low"
    assert t3[0].evidence["matched_via"] == "family_prefix_developer"
    assert t3[0].evidence["family_prefix"] == "gemma-3n"


def test_tier3_negative_developer_mismatch(index_with_bert):
    m = ComponentMatcher(index_with_bert)
    component = _component("someone-else/gemma-3n-E2B-it", developer="IndependentResearcher")
    matches = m.match(component)
    assert matches == []  # no Tier 3 because developer doesn't match
