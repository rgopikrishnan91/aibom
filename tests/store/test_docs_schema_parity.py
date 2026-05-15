"""Every predicate in vocab.py appears in SCHEMA.md."""
import inspect
import re
from pathlib import Path

from aikaboom.store import vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_MD = REPO_ROOT / "docs" / "worldofboms" / "SCHEMA.md"


def _vocab_terms():
    terms = []
    for name, value in inspect.getmembers(vocab):
        if name.startswith("_") or name.isupper():
            continue
        if hasattr(value, "n3") and "aibom" in str(value):
            terms.append(name)
    return set(terms)


def test_schema_md_covers_all_vocab_predicates():
    text = SCHEMA_MD.read_text()
    # Capture both `aibom:foo` and `aibom:foo-bar` (hyphenated IRIs). The
    # Python attribute uses an underscore, so normalize hyphens to underscores
    # for comparison against vocab names.
    raw_documented = re.findall(r"`aibom:([A-Za-z][A-Za-z0-9_-]*)`", text)
    documented = {term.replace("-", "_") for term in raw_documented}
    actual = _vocab_terms()
    missing = actual - documented
    # Filter out classes that are documented under their own section heading
    # without `aibom:` backtick form. This is a deliberate looseness.
    class_names = {"Artifact", "Model", "Dataset", "Paper", "CodeRepo",
                   "ArtifactVersion", "BOMClaim", "GenerationRun",
                   "TrustVote", "Agent", "License", "Supplier", "Person", "Source"}
    missing -= class_names
    assert not missing, f"Predicates missing from SCHEMA.md: {missing}"
