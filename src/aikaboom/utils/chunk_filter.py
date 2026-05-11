"""Filter chunks that are structurally empty (whitespace, lone markdown
headers, table separators, HuggingFace dataset/model-card placeholder
boilerplate) before they enter the retrieval pool.

The retrieval pipeline used to drop chunks shorter than 100 chars,
which threw out genuinely valuable short content (a one-liner like
``License: MIT`` is gold for the license field). The intent of the
filter was always to remove garbage — empty chunks, whitespace-only
strings, chunks that are nothing but markdown structural lines, or
chunks dominated by ``[More Information Needed]`` placeholder
boilerplate — not to gate on raw length.

The placeholder filter (Phase 12 / Finding #37) is critical for
dataset BOMs: HuggingFace dataset cards routinely ship with ~80 % of
their sections filled in as ``[More Information Needed]``. Those
chunks were outscoring real arxiv prose in retrieval because the
retriever was matching on section-header *structure* rather than
*content*. The diagnostic in ``scripts/diagnose_dataset_retrieval.py``
showed a chunk consisting of literally one ``[More Information
Needed]`` placeholder ranking above the paper's introduction.

A chunk is considered useful if, after stripping pure-structural
lines (markdown headers, horizontal rules, code-fence markers, table
separators, blank lines) AND placeholder lines, at least one line of
substantive content remains.
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


# Placeholder boilerplate that HuggingFace card templates leave behind
# when a section is unfilled. These lines carry no information but used
# to score well in retrieval because the surrounding section header
# was a structural lexical match for the HyDE passage. Common variants:
# ``[More Information Needed]``, ``[Needs More Information]``,
# ``[Optional]``, ``[TBD]``, ``[Coming soon]``, plus bullet-prefixed
# variants ``- [More Information Needed]`` and bare ``TBD`` / ``N/A``.
_PLACEHOLDER_LINE = re.compile(
    r"""^\s*(?:
          [-*+]\s*           # optional bullet glyph
        | \d+\.\s*           # or numbered list
        )?
        \[?\s*
        (?:
            more\s+information\s+needed
          | needs?\s+more\s+information
          | more\s+info\s+needed
          | needs?\s+manual\s+review
          | optional
          | tbd
          | t\.b\.d
          | coming\s+soon
          | placeholder
          | not\s+provided
          | not\s+specified
          | not\s+applicable
          | n/?a
          | none
        )
        \s*\]?
        [.\s]*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Absolute floor — a chunk shorter than this after stripping is genuine
# noise (e.g. a single bullet glyph that survived chunking, a stray
# punctuation mark). Set deliberately low so short-but-useful content
# like "MIT" or "120 kWh" still passes.
_MIN_CONTENT_CHARS = 5


def _is_substantive(line: str) -> bool:
    """A line is substantive if it isn't blank, structural, or a placeholder."""
    if not line.strip():
        return False
    if _STRUCTURAL_LINE.match(line):
        return False
    if _PLACEHOLDER_LINE.match(line):
        return False
    return True


def is_useful_chunk(text: str) -> bool:
    """Return ``True`` if ``text`` carries any substantive content.

    A chunk is useful unless it's empty, whitespace-only, below the
    minimum content floor, or composed entirely of structural /
    placeholder lines.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < _MIN_CONTENT_CHARS:
        return False
    return any(_is_substantive(line) for line in stripped.splitlines())
