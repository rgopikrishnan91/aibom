"""Worktree-level pytest config.

The shared venv at ../.venv/ has `aikaboom` installed in editable mode
from the parent repo's src/. When running tests from this worktree, we
need pytest to pick up THIS worktree's src/aikaboom/ (which contains the
new store/ package) rather than the parent repo's. Prepending the
worktree's src/ to sys.path achieves that without disturbing the parent
repo's install.

Remove this file when the worktree is merged back to main.
"""
import sys
from pathlib import Path

_WORKTREE_SRC = Path(__file__).parent / "src"
if str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))
