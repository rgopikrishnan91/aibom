"""End-to-end smoke: populate store -> license-audit -> JSON output has all sections."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

LINEAGE_TTL = Path(__file__).parent / "fixtures" / "lineage_3node.ttl"


def test_e2e_audit_returns_full_json_payload(tmp_path):
    env = os.environ.copy()
    env["AIKABOOM_GRAPH_BACKEND"] = "rdflib"
    env["AIKABOOM_GRAPH_DIR"] = str(tmp_path)
    env["BOM_SKIP_DOTENV"] = "1"
    # Worktree-only: ensure subprocesses pick up THIS worktree's src/aikaboom
    # ahead of the shared venv's editable install (which points at a sibling
    # worktree). Mirrors the pattern in test_cli.py's populated_store_env.
    # Remove this block when the worktree is merged back to main.
    worktree_src = str(Path(__file__).resolve().parents[3] / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        worktree_src + os.pathsep + existing_pp if existing_pp else worktree_src
    )

    # Populate the store from the lineage fixture.
    populate = (
        "from pathlib import Path; "
        "from aikaboom.store import BomStore; "
        "import rdflib; "
        "store = BomStore.open(); "
        f"g = rdflib.Graph().parse(r'{LINEAGE_TTL}', format='turtle'); "
        "store._backend.add_quads([(s, p, o, None) for s, p, o in g]); "
        "store._backend.close()"
    )
    subprocess.run([sys.executable, "-c", populate], check=True, env=env)

    # Run audit.
    r = subprocess.run(
        [sys.executable, "-m", "aikaboom.cli", "license-audit", "--format", "json"],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode in (0, 2), (
        f"audit exited {r.returncode}; stderr: {r.stderr[:500]}"
    )
    payload = json.loads(r.stdout)
    assert "findings" in payload
    assert "compatible_subchains" in payload
    assert "breaking_nodes" in payload
    assert isinstance(payload["compatible_subchains"], list)
    assert isinstance(payload["breaking_nodes"], list)
