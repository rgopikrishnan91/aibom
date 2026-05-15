"""Mint stable IRIs for graph nodes."""
from __future__ import annotations

import hashlib
import uuid
from typing import Mapping

from aikaboom.store.naming import Identifier


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _uuid4_hex() -> str:
    return uuid.uuid4().hex


def artifact_iri(primary: Identifier) -> str:
    """IRI for an Artifact, hashed from its primary canonicalized identifier."""
    key = f"{primary.platform}:{primary.value}"
    return f"bom:artifact/{_sha256_hex(key)}"


def version_iri(artifact_iri_: str, version: str) -> str:
    """IRI for an ArtifactVersion under a given Artifact."""
    assert artifact_iri_.startswith("bom:artifact/"), artifact_iri_
    hash_suffix = artifact_iri_.split("/", 1)[1]
    return f"bom:version/{hash_suffix}/{version}"


def claim_iri() -> str:
    """Fresh IRI for a BOMClaim. Random UUID4 — claims are unique per generation."""
    return f"bom:claim/{_uuid4_hex()}"


def run_iri(params: Mapping[str, str]) -> str:
    """IRI for a GenerationRun, hashed from its parameter dict.

    Two generations with identical (provider, llm_model, prompt_version,
    code_version, mode, use_case) share the same run node.
    """
    fields = ("provider", "llm_model", "prompt_version", "code_version", "mode", "use_case")
    key = "|".join(f"{f}={params.get(f, '')}" for f in fields)
    return f"bom:run/{_sha256_hex(key)}"


def vote_iri() -> str:
    """Fresh IRI for a TrustVote."""
    return f"bom:vote/{_uuid4_hex()}"


def agent_iri(agent_id: str) -> str:
    """IRI for an Agent (user or generator)."""
    return f"bom:agent/{_sha256_hex(agent_id)}"


def source_iri(platform: str) -> str:
    """IRI for a data source (huggingface, github, arxiv)."""
    return f"aibom:source/{platform}"
