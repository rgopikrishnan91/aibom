"""GitHub-link extraction from HuggingFace model card text.

Adapted from SAILResearch/replication-25-synchronization-Patterns
(``Codes/GH_link_extraction_from_model_card.py``). Demo I/O stripped;
exposed as a pure function that takes card text + model name and returns
the filtered list of GitHub URLs.

Citation:
    Ajibode, A. et al. (2025). "Synchronization Patterns between
    HuggingFace and GitHub for Pre-Trained Language Models."
    https://github.com/SAILResearch/replication-25-synchronization-Patterns

License:
    SAIL upstream has no LICENSE file at time of lift (2026-05-08).
    Reuse confirmed verbally by the maintainer (per AIkaBoOM Phase 12
    project notes); when SAIL adds a LICENSE, mirror it here.
"""
from __future__ import annotations

import re
from typing import List, Optional


_GITHUB_URL_RE = re.compile(r'https?://github\.com/[^\s/$.?#][^\s]*')


def extract_github_urls_from_card(
    card_text: Optional[str],
    model_name: str,
) -> List[str]:
    """Return GitHub URLs in `card_text` whose path overlaps `model_name`.

    Mirrors the SAIL heuristic: a URL counts as a likely
    code-repository-for-this-model link when its path contains at least
    one segment of the HF repo_id (split on ``/`` then on ``-``/``_``).
    Trailing markdown punctuation is stripped.

    Args:
        card_text: full text of the HF model card README (may be ``None``).
        model_name: HF repo_id, e.g. ``"mistralai/Mistral-7B-v0.1"``.

    Returns:
        Sorted list of unique cleaned URLs. Empty list when no card or
        no qualifying URL.
    """
    if not card_text:
        return []

    # Build search terms from the repo path.
    parts = (model_name or "").split('/')
    left = parts[0] if parts else ""
    right = parts[1] if len(parts) > 1 else ""
    left_segs = re.split(r'[-_]', left) if left else []
    right_segs = re.split(r'[-_]', right) if right else []
    terms = {t for t in (left_segs + ([right_segs[0]] if right_segs else [])) if t}
    if not terms:
        return []

    out: set[str] = set()
    for raw in _GITHUB_URL_RE.findall(card_text):
        # Strip trailing markdown noise (close-paren, comma, period).
        url = re.sub(r'[).,]+$', '', raw)
        if any(t in url for t in terms):
            out.add(url)
    return sorted(out)


def extract_org_from_github_url(url: str) -> Optional[str]:
    """Pull the org/owner segment from a github.com URL.

    >>> extract_org_from_github_url("https://github.com/QwenLM/Qwen2.5")
    'QwenLM'
    """
    if not url or 'github.com' not in url:
        return None
    parts = url.split('/')
    # https: // github.com / OWNER / REPO ...
    #   0     1      2         3       4
    if len(parts) >= 4:
        return parts[3] or None
    return None


__all__ = ["extract_github_urls_from_card", "extract_org_from_github_url"]
