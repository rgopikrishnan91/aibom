"""Token-count helper.

Used by ``tests/test_question_bank_optimized.py`` and
``tools/optimize_question_bank.py`` to bound the HyDE
hypothetical-passage length against the embedder's 512-token window.

Tokenizer preference, in order:

1. ``transformers.AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")``
   — the **exact** WordPiece tokenizer the runtime embedder uses. Pixel-perfect
   counts. Requires HF Hub access on first call (or a populated cache).
2. ``tiktoken.encoding_for_model("gpt-4")`` (cl100k_base) — OpenAI BPE.
   Within ~10–15% of BGE WordPiece for English text. Requires network
   on first call to fetch the BPE table.
3. Conservative ``len(text) / 3.5`` — overcounts by ~30% on English
   technical prose. Always works offline.

The chosen tokenizer is cached after the first successful resolve.
"""
from __future__ import annotations

from typing import Callable, Optional


_BGE_MODEL = "BAAI/bge-small-en-v1.5"
_TIKTOKEN_MODEL = "gpt-4"  # cl100k_base, the closest widely-available BPE
_FALLBACK_CHARS_PER_TOKEN = 3.5  # conservative WordPiece-ish estimate


_counter: Optional[Callable[[str], int]] = None
_chosen: Optional[str] = None


def _resolve() -> None:
    """Pick the best available tokenizer once and cache the result."""
    global _counter, _chosen

    # Try the real BGE tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        tok = AutoTokenizer.from_pretrained(_BGE_MODEL)
        _counter = lambda s: len(tok.encode(s, add_special_tokens=False))
        _chosen = f"transformers/{_BGE_MODEL}"
        return
    except Exception:
        pass

    # Try tiktoken (BPE — close enough for sanity-check)
    try:
        import tiktoken  # type: ignore[import-not-found]
        enc = tiktoken.encoding_for_model(_TIKTOKEN_MODEL)
        _counter = lambda s: len(enc.encode(s))
        _chosen = f"tiktoken/{_TIKTOKEN_MODEL}"
        return
    except Exception:
        pass

    # Fall back to conservative char-ratio estimate
    _counter = lambda s: int(len(s) / _FALLBACK_CHARS_PER_TOKEN)
    _chosen = "fallback/chars-per-token"


def count_tokens(text: str) -> int:
    """Return an approximate WordPiece-token count for ``text``.

    Uses the most-accurate tokenizer available (see module docstring);
    falls back to a conservative estimate offline.
    """
    if _counter is None:
        _resolve()
    return _counter(text or "")  # type: ignore[misc]


def chosen_tokenizer() -> str:
    """Return a label for the resolved tokenizer (for logging / diagnostics)."""
    if _chosen is None:
        _resolve()
    return _chosen or "unknown"
