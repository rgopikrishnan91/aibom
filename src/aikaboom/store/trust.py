"""Trust vote model + score aggregation + canonical-claim pointer."""
from __future__ import annotations

import enum
import getpass
import os
import platform
from typing import Iterable

from rdflib import URIRef

from aikaboom.store import vocab


class VoteKind(str, enum.Enum):
    TRUSTED = "trusted"
    FLAGGED = "flagged"
    DISPUTED = "disputed"
    IMPLICIT_USE = "implicit-use"
    IMPLICIT_VALIDATE = "implicit-validate"


_WEIGHTS: dict["VoteKind", float] = {
    VoteKind.TRUSTED: +1.0,
    VoteKind.FLAGGED: -1.0,
    VoteKind.DISPUTED: -0.5,
    VoteKind.IMPLICIT_USE: +0.25,
    VoteKind.IMPLICIT_VALIDATE: +0.25,
}


def agent_id_default() -> str:
    """Resolve an Agent identifier from env or fall back to `<user>@<host>`."""
    explicit = os.environ.get("AIKABOOM_AGENT_ID")
    if explicit:
        return explicit
    return f"{getpass.getuser()}@{platform.node()}"


def vote_kind_iri(kind: VoteKind) -> URIRef:
    """IRI for a VoteKind individual under the AIBOM namespace."""
    mapping = {
        VoteKind.TRUSTED: vocab.trusted,
        VoteKind.FLAGGED: vocab.flagged,
        VoteKind.DISPUTED: vocab.disputed,
        VoteKind.IMPLICIT_USE: vocab.implicit_use,
        VoteKind.IMPLICIT_VALIDATE: vocab.implicit_validate,
    }
    return URIRef(mapping[kind])


def compute_score(votes: Iterable[VoteKind]) -> float:
    """Aggregate votes into a trust score clipped to [-1.0, +1.0].

    Each vote contributes its weight to a running sum. Implicit votes
    contribute less than explicit ones so a single TRUSTED outranks a
    single IMPLICIT_USE. The sum is clipped to [-1.0, +1.0] so accumulated
    votes saturate rather than grow unbounded. An empty vote set yields
    0.0.
    """
    weights = [_WEIGHTS[k] for k in votes]
    if not weights:
        return 0.0
    total = sum(weights)
    if total > 1.0:
        return 1.0
    if total < -1.0:
        return -1.0
    return total
