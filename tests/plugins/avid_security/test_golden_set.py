# tests/plugins/avid_security/test_golden_set.py
import csv
import shutil
from pathlib import Path
import pytest

from aikaboom.plugins.avid_security.snapshot import AvidIndex
from aikaboom.plugins.avid_security.matcher import ComponentMatcher
from aikaboom.plugins.avid_security.walker import Component

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "Golden_Set" / "AVID_Security"


def _load_cases():
    rows = list(csv.DictReader((GOLDEN_DIR / "cases.csv").open()))
    return [pytest.param(r, id=r["case_id"]) for r in rows]


@pytest.fixture(scope="module")
def golden_index(tmp_path_factory):
    work = tmp_path_factory.mktemp("avid")
    repo = work / "avid-db"
    (repo / "reports" / "fixtures").mkdir(parents=True)
    for src in (GOLDEN_DIR / "avid_fixtures").glob("AVID-*.json"):
        shutil.copy(src, repo / "reports" / "fixtures" / src.name)
    idx = AvidIndex(db_path=work / "avid.sqlite")
    idx.build(repo_dir=repo)
    return idx


@pytest.mark.parametrize("case", _load_cases())
def test_avid_golden_case(case, golden_index):
    bases = tuple(b for b in case["component_base_models"].split(";") if b)
    component = Component(
        kind=case["component_kind"],
        hf_path=case["component_hf_path"],
        developer=case["component_developer"] or None,
        base_models=bases,
        scope_in_bom=case["component_scope"],
        spdx_id=f"urn:aibom:pkg:{case['component_hf_path'].replace('/', '__').lower()}",
    )
    matches = ComponentMatcher(golden_index).match(component)
    matches = [m for m in matches if m.avid_report["report_id"] == case["avid_report_id"]]

    if case["expected_tier"] == "no_match":
        assert matches == [], f"{case['case_id']}: expected no match, got {matches}"
        return

    assert len(matches) == 1, f"{case['case_id']}: expected exactly one match, got {len(matches)}"
    m = matches[0]
    assert m.tier == int(case["expected_tier"]), f"{case['case_id']}: tier mismatch"
    assert m.confidence == case["expected_confidence"], f"{case['case_id']}: confidence mismatch"

    from aikaboom.plugins.avid_security.spdx import emit_security_elements
    nodes = emit_security_elements([m], snapshot_sha="golden-fixture")
    vex = next(n for n in nodes if n["type"].startswith("security_Vex"))
    assert vex["relationshipType"] == case["expected_vex_status"], (
        f"{case['case_id']}: vex status mismatch"
    )
    statement = vex.get("security_actionStatement") or vex.get("security_statusNotes", "")
    assert case["expected_statement_contains"] in statement, (
        f"{case['case_id']}: statement missing '{case['expected_statement_contains']}'"
    )
