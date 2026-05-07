"""CycloneDX validator backed by the ``sbom-utility`` Go CLI.

`sbom-utility <https://github.com/CycloneDX/sbom-utility>`_ validates
CycloneDX 1.2-1.7 BOMs against the official, embedded JSON schemas.
We adopt it for our CycloneDX 1.7 emission. SPDX support stops at
2.3, so the in-tree SPDX 3.0.1 validator stays in
``aikaboom.utils.spdx_validator``.

The wrapper is optional at runtime: if the binary isn't on PATH it
returns a "skipped" result rather than raising, so the rest of the
pipeline continues.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

BINARY = "sbom-utility"
TIMEOUT_SECONDS = 30
INSTALL_HINT = (
    "sbom-utility binary not found on PATH. Install via "
    "tools/install_sbom_utility.sh or see "
    "https://github.com/CycloneDX/sbom-utility/releases."
)


def is_available() -> bool:
    """True iff the ``sbom-utility`` binary is on PATH."""
    return shutil.which(BINARY) is not None


def validate_cyclonedx(bom: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Validate a CycloneDX BOM against the official JSON schema.

    Args:
        bom: either a CycloneDX dict, a path string, or a :class:`Path`
            pointing to a CycloneDX JSON file.

    Returns:
        ``{valid, errors, validator}``. When the binary is not installed,
        ``valid=None`` and ``validator="skipped"`` (with a ``reason``
        field) so callers can degrade gracefully.
    """
    if not is_available():
        return {
            "valid": None,
            "errors": [],
            "validator": "skipped",
            "reason": INSTALL_HINT,
        }

    cleanup_path = None
    if isinstance(bom, dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(bom, f)
            path = f.name
        cleanup_path = Path(path)
    else:
        path = str(bom)

    try:
        proc = subprocess.run(
            [BINARY, "validate", "-i", path, "--quiet"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "errors": [f"sbom-utility timed out after {TIMEOUT_SECONDS}s"],
            "validator": "sbom-utility",
        }
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)

    if proc.returncode == 0:
        return {"valid": True, "errors": [], "validator": "sbom-utility"}

    # sbom-utility writes errors to stderr (and sometimes stdout). Combine
    # both, keep non-empty lines.
    raw = "\n".join(filter(None, (proc.stderr, proc.stdout)))
    errors = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not errors:
        errors = [f"sbom-utility exited with code {proc.returncode}"]
    return {"valid": False, "errors": errors, "validator": "sbom-utility"}
