import json
from pathlib import Path
import pytest

from aikaboom.plugins.avid_security.snapshot import AvidIndex, family_prefixes

@pytest.fixture
def sample_repo(tmp_path):
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
    return repo

def test_family_prefixes_basic():
    assert family_prefixes("gemma-3n-e4b-it") == ["gemma-3n-e4b", "gemma-3n"]
    assert family_prefixes("bert-base-uncased") == ["bert-base"]
    assert family_prefixes("gemma-3n") == []
    assert family_prefixes("singleword") == []

def test_index_build_inserts_rows(tmp_path, sample_repo):
    idx = AvidIndex(db_path=tmp_path / "avid.sqlite")
    idx.build(repo_dir=sample_repo)
    rows = idx.find_exact("bert-base-uncased", artifact_kind="Model")
    assert len(rows) == 1
    assert rows[0]["report_id"] == "AVID-2022-R0001"

def test_index_find_by_family_prefix_and_developer(tmp_path, sample_repo):
    idx = AvidIndex(db_path=tmp_path / "avid.sqlite")
    idx.build(repo_dir=sample_repo)
    rows = idx.find_by_family_prefix("gemma-3n", developer="Google", artifact_kind="Model")
    assert len(rows) == 1
    assert rows[0]["report_id"] == "AVID-2026-R0478"
