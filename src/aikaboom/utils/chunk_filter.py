"""Filter chunks that are structurally empty (whitespace, lone markdown
headers, table separators) before they enter the retrieval pool.

The retrieval pipeline used to drop chunks shorter than 100 chars,
which threw out genuinely valuable short content (a one-liner like
``License: MIT`` is gold for the license field). The intent of the
filter was always to remove garbage — empty chunks, whitespace-only
strings, chunks that are nothing but markdown structural lines —
not to gate on raw length. This helper does that intent properly.

A chunk is considered useful if, after stripping pure-structural
lines (markdown headers, horizontal rules, code-fence markers, table
separators, blank lines), at least one line of substantive content
remains.
"""
from __future__ import annotations

import re


# Lines that are pure markdown structure (no prose).
# Note: setext-style headers (Title\n=====) need multi-line lookahead and
# are rare enough in model cards / READMEs that we skip them — ATX
# headers (## Title) cover ~99% of real-world content.
_STRUCTURAL_LINE = re.compile(
    r"""^\s*(?:
        \#+\s+.*           # markdown header (# Title, ## Subtitle, ...)
      | -{3,}              # horizontal rule / yaml frontmatter divider
      | `{3,}.*            # code-fence opener/closer
      | \|[\s\-:|]+\|      # markdown table separator (| --- | --- |)
    )\s*$""",
    re.VERBOSE,
)

# Absolute floor — a chunk shorter than this after stripping is genuine
# noise (e.g. a single bullet glyph that survived chunking, a stray
# punctuation mark). Set deliberately low so short-but-useful content
# like "MIT" or "120 kWh" still passes.
_MIN_CONTENT_CHARS = 5


def is_useful_chunk(text: str) -> bool:
    """Return ``True`` if ``text`` carries any substantive content.

    A chunk is useful unless it's empty, whitespace-only, below the
    minimum content floor, or composed entirely of markdown structural
    lines (headers, dividers, code fences, table separators).
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < _MIN_CONTENT_CHARS:
        return False
    for line in stripped.splitlines():
        if not line.strip():
            continue
        if _STRUCTURAL_LINE.match(line):
            continue
        return True
    return False
