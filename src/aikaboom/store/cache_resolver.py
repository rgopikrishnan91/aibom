"""Cache resolution UX: prompt the user, or auto-decide based on policy."""

from __future__ import annotations

import enum
import sys
from typing import Callable

from aikaboom.store.store import ResolveResult


class CachePolicy(str, enum.Enum):
    USE = "use"
    REGEN = "regen"
    PROMPT = "prompt"
    AUTO = "auto"  # alias for USE


def render_prompt(result: ResolveResult, planned_llm: str) -> str:
    """Render the minimal two-option prompt as a string."""
    lines = [f"BOMs for {result.artifact_label} already exist:"]
    for claim in result.matching_claims:
        when = claim.get("created_at", "")
        when_short = when.split("T")[0] if when else "unknown"
        lines.append(f"  - {claim.get('llm_model', 'unknown')}   ({when_short})")
    lines.append("")
    lines.append(f"You're about to generate with {planned_llm}.")
    lines.append("")
    lines.append("  [u] use the most recent existing BOM")
    lines.append("  [r] regenerate")
    return "\n".join(lines)


def decide(
    result: ResolveResult,
    policy: CachePolicy,
    interactive: bool,
    input_fn: Callable[[str], str] | None = None,
    planned_llm: str = "(current LLM)",
) -> str:
    """Decide between 'use' and 'generate'."""
    if not result.matching_claims:
        return "generate"
    if policy in (CachePolicy.USE, CachePolicy.AUTO):
        return "use"
    if policy == CachePolicy.REGEN:
        return "generate"
    # PROMPT
    if not interactive:
        return "use"  # non-TTY degrades to use
    prompt = render_prompt(result, planned_llm=planned_llm) + "\n> "
    response = (input_fn or input)(prompt).strip().lower()
    if response.startswith("r"):
        return "generate"
    return "use"


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
