"""Every internal markdown link in docs/worldofboms/ resolves."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "worldofboms"


def test_internal_markdown_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    failures = []
    for md in DOCS.glob("*.md"):
        text = md.read_text()
        for m in pattern.finditer(text):
            target = m.group(1)
            if target.startswith("http"):
                continue
            if target.startswith("#"):
                continue
            # Strip in-document anchor for resolution.
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            # Resolve relative to the doc file.
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                # Try resolving against repo root for ../-prefixed.
                alt = (REPO_ROOT / path_part.lstrip("./")).resolve()
                if not alt.exists():
                    failures.append(f"{md.name} -> {target}")
    assert not failures, "Broken links: " + "\n".join(failures)
