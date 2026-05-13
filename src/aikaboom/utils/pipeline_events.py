"""Shared helper for emitting structured pipeline events to the web UI.

The web frontend listens on the SSE /logs stream and parses lines that
start with ``[[BOM_EVENT]]`` as JSON pipeline events (pipeline.start /
stage.start / stage.done / field.done / recursive.* etc.). Modules that
want to participate in that stream call :func:`emit` here.

Emission is gated by :data:`EMIT_EVENTS`. The web app flips it on at
startup; the CLI keeps it off so terminal output stays clean.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

# Flipped to True by the web app during startup. Off by default so the
# CLI doesn't print [[BOM_EVENT]] noise on every run.
EMIT_EVENTS: bool = False

EVENT_PREFIX: str = "[[BOM_EVENT]]"


def emit(payload: Dict[str, Any]) -> None:
    """Print a structured pipeline event to stdout.

    Best-effort: never raises into the caller. Adds a ``ts`` (epoch seconds)
    field if the caller hasn't already set one.
    """
    if not EMIT_EVENTS:
        return
    try:
        payload.setdefault("ts", time.time())
        print(f"{EVENT_PREFIX}{json.dumps(payload, default=str)}", flush=True)
    except Exception:
        # Event emission is decorative; never let it break the BOM pipeline.
        pass
