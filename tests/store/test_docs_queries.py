"""Each ```sparql block in QUERIES.md parses (syntax check only)."""
import re
from pathlib import Path

from rdflib.plugins.sparql.parser import parseQuery

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_MD = REPO_ROOT / "docs" / "worldofboms" / "QUERIES.md"


def test_all_sparql_recipes_parse():
    text = QUERIES_MD.read_text()
    blocks = re.findall(r"```sparql\n(.*?)\n```", text, re.DOTALL)
    failures = []
    for i, q in enumerate(blocks):
        try:
            parseQuery(q)
        except Exception as e:
            failures.append(f"Recipe #{i}: {e}")
    assert not failures, "\n".join(failures)
